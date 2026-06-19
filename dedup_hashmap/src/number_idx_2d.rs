//! `build_number_idx_2d_kernel` — the stage-3c NUMBER-band emission core.
//!
//! Single concern: given the FLAT per-segment carrier-context arrays the
//! Python front matter (`build_batched_carriers`) concatenates off the
//! shared call_target columns, identify + locate every surviving
//! NUMBER-band carrier and emit each `TokenType`'s `idx_2d` gather-offset
//! rows (ALG-7 fixed-width + ALG-2 F128 + ALG-8 VC2 multi-chunk packing)
//! as a deterministic GIL-released integer/byte state machine. Mirrors the
//! numpy logic of `_batched_carriers.py` (carrier recovery, lines 309-377)
//! + `_emit_vc2` / `_emit_f128` / `_emit_fixed_fp` exactly — it
//! re-implements NO decode rule beyond reproducing that vectorised
//! arithmetic as a scalar walk.
//!
//! The Python-object DFS walk (consuming the Step-1 extractor's
//! `iter_call_target_columns`) and the per-DFS-call_target slice
//! reconstruction stay in Python; this kernel owns only the flat-array
//! compute the numpy emitters did, so it runs under `py.detach`.
//!
//! ## Carrier order
//!
//! Carriers are walked in body order (ascending flat body index), which is
//! exactly the canonical DFS-then-stream stage-3 linearisation. Within a
//! `TokenType` the emitted rows are carrier order, LSB-first per carrier —
//! identical to the numpy `block_idx == b` boolean-mask gather, which
//! preserves the body scan order.
//!
//! ## NUMBER block indexing
//!
//! `block = shifted_id - band_lo`; block 0 = VC2, 1..5 = F16/BF16/F32/F64/
//! F80, 6 = F128 (the canonical `_NUMBER_BLOCK_TOKEN_TYPES` order). Fixed
//! widths per block come from the caller (`fixed_widths`).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// Per-segment CSR length from a base array + total flat length. Mirrors
/// `_seg_lengths.seg_lengths_from_base`: `len[s] = base[s+1] - base[s]`
/// with the implicit final boundary at `flat_len`.
fn seg_len_at(base: &[i64], flat_len: i64, s: usize) -> i64 {
    let hi = if s + 1 < base.len() {
        base[s + 1]
    } else {
        flat_len
    };
    hi - base[s]
}

/// One carrier's recovered location + per-block context.
struct Carrier {
    block: usize,          // shifted_id - band_lo
    byte_offset: i64,      // first inline-byte offset of the payload
    raw_position: i64,     // raw-stream position (ALG-8 reads L at +1)
    expanded_position: i64,// within-segment expanded position
    seg: usize,            // owning kept-segment id
}

/// Flat per-type row accumulator. `rows` is row-major `n_rows * width`.
struct TypeRows {
    width: usize,
    rows: Vec<u32>,
    rows_per_carrier: Vec<i64>,
    carrier_seg: Vec<i64>,
}

impl TypeRows {
    fn new(width: usize) -> Self {
        TypeRows {
            width,
            rows: Vec::new(),
            rows_per_carrier: Vec::new(),
            carrier_seg: Vec::new(),
        }
    }
}

/// The kernel's pure-Rust core (no PyO3 in the signature) so unit tests can
/// drive it directly. Returns `(per_block_rows, f128_is_nan_or_inf,
/// vc2_chunk_indices)`.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    expanded_body: &[i64],
    painted_body: &[bool],
    body_seg_len: &[i64],
    real_pos_flat: &[i64],
    real_seg_base: &[i64],
    digit_flat: &[i64],
    digit_base: &[i64],
    seg_slice_start: &[i64],
    seg_painted_vc2_flat: &[i64],
    seg_painted_offsets: &[i64],
    seg_surviving: &[i64],
    seg_runlen_base: &[i64],
    runlen_number_flat: &[i64],
    seg_f128_base: &[i64],
    f128_full_mask_flat: &[bool],
    band_lo: i64,
    band_hi: i64,
    fixed_widths: &[usize], // length 7, indexed by block
) -> Result<(Vec<TypeRows>, Vec<bool>, Vec<i64>), String> {
    let n_blocks = fixed_widths.len();
    let vc2_block = 0usize;
    let f128_block = n_blocks - 1;
    let total_body = expanded_body.len();

    if painted_body.len() != total_body {
        return Err(format!(
            "expanded_body ({}) and painted_body ({}) length mismatch",
            total_body,
            painted_body.len()
        ));
    }
    let n_kept = body_seg_len.len();

    // --- segmented expanded->raw recovery (per body slot) -------------
    // Mirrors `_batched_carriers` lines 316-358: body_seg_offsets (CSR),
    // body_seg_id via repeat, carrier mask = in_number_band & ~painted,
    // and the segmented `cumsum(is_real) - 1 - seg_carry_in` real-index
    // map. We fuse all of it into one scan, tracking a per-segment running
    // real-count carry exactly as the numpy `seg_carry_in` subtraction.
    let mut carriers: Vec<Carrier> = Vec::new();

    let mut body_seg_offsets: Vec<i64> = Vec::with_capacity(n_kept + 1);
    body_seg_offsets.push(0);
    for &l in body_seg_len {
        let last = *body_seg_offsets.last().unwrap();
        body_seg_offsets.push(last + l);
    }
    if *body_seg_offsets.last().unwrap() != total_body as i64 {
        return Err(format!(
            "body_seg_len sum {} != total_body {}",
            body_seg_offsets.last().unwrap(),
            total_body
        ));
    }

    // Walk segment by segment so the per-segment real-index reset is exact
    // (`real_idx_per_slot` is a within-segment cumsum(is_real) - 1). This
    // reproduces `global_cum - 1 - seg_carry_in[seg]` without the global
    // array.
    for seg in 0..n_kept {
        let lo = body_seg_offsets[seg] as usize;
        let hi = body_seg_offsets[seg + 1] as usize;
        let mut real_running = 0i64; // within-segment count of is_real seen
        for j in lo..hi {
            let is_real = !painted_body[j];
            if is_real {
                real_running += 1;
            }
            let shifted = expanded_body[j];
            let in_band = shifted >= band_lo && shifted < band_hi;
            if in_band && is_real {
                // real_idx_per_slot = within-seg cumsum(is_real) - 1.
                let real_idx_local = real_running - 1;
                let real_global = real_seg_base[seg] + real_idx_local;
                if real_global < 0 || (real_global as usize) >= real_pos_flat.len()
                {
                    return Err(format!(
                        "carrier real_global {real_global} out of \
                         real_pos_flat bounds {}",
                        real_pos_flat.len()
                    ));
                }
                let raw_position = real_pos_flat[real_global as usize];
                // expanded position within segment = (j - seg_start) + 1.
                let expanded_position = (j as i64 - lo as i64) + 1;
                let block = (shifted - band_lo) as usize;
                if block >= n_blocks {
                    return Err(format!(
                        "carrier block {block} >= n_blocks {n_blocks}"
                    ));
                }
                // byte offset = seg_slice_start[seg] +
                //               digit_flat[digit_base[seg] + raw + 1].
                let gather = digit_base[seg] + raw_position + 1;
                if gather < 0 || (gather as usize) >= digit_flat.len() {
                    return Err(format!(
                        "carrier digit gather {gather} out of bounds {}",
                        digit_flat.len()
                    ));
                }
                let byte_offset =
                    seg_slice_start[seg] + digit_flat[gather as usize];
                carriers.push(Carrier {
                    block,
                    byte_offset,
                    raw_position,
                    expanded_position,
                    seg,
                });
            }
        }
    }

    // --- per-block emission -------------------------------------------
    let mut per_block: Vec<TypeRows> =
        fixed_widths.iter().map(|&w| TypeRows::new(w)).collect();
    let mut f128_flags: Vec<bool> = Vec::new();
    let mut vc2_chunk_indices: Vec<i64> = Vec::new();

    let runlen_len = runlen_number_flat.len() as i64;
    let f128_len = f128_full_mask_flat.len() as i64;

    for c in &carriers {
        let tr = &mut per_block[c.block];
        if c.block == vc2_block {
            emit_vc2(
                c,
                tr,
                seg_runlen_base,
                runlen_number_flat,
                runlen_len,
                seg_painted_vc2_flat,
                seg_painted_offsets,
                seg_surviving,
                &mut vc2_chunk_indices,
            )?;
        } else if c.block == f128_block {
            emit_f128(
                c,
                tr,
                seg_f128_base,
                f128_full_mask_flat,
                f128_len,
                &mut f128_flags,
            );
        } else {
            emit_fixed(c, tr);
        }
    }

    Ok((per_block, f128_flags, vc2_chunk_indices))
}

/// ALG-7 fixed-width FP: one row of `width` contiguous byte offsets.
fn emit_fixed(c: &Carrier, tr: &mut TypeRows) {
    let p = c.byte_offset;
    for k in 0..tr.width {
        tr.rows.push((p + k as i64) as u32);
    }
    tr.rows_per_carrier.push(1);
    tr.carrier_seg.push(c.seg as i64);
}

/// ALG-2 F128: 2 rows (LSB limb bytes 8..15, then MSB limb 0..7) for a
/// finite source; 1 row (MSB limb 0..7) for NaN/Inf. Finite signal =
/// `extra_f128_mask[expanded_pos + 1]` against the FULL per-segment mask.
fn emit_f128(
    c: &Carrier,
    tr: &mut TypeRows,
    seg_f128_base: &[i64],
    f128_full_mask_flat: &[bool],
    f128_len: i64,
    f128_flags: &mut Vec<bool>,
) {
    let lookahead_local = c.expanded_position + 1;
    let seg_full_len = seg_len_at(seg_f128_base, f128_len, c.seg);
    let is_finite = if lookahead_local < seg_full_len {
        let gather = seg_f128_base[c.seg] + lookahead_local;
        // `lookahead_local < seg_full_len` guarantees in-bounds.
        f128_full_mask_flat[gather as usize]
    } else {
        false
    };
    f128_flags.push(!is_finite);
    let p = c.byte_offset;
    if is_finite {
        // chunk 0 = LSB limb (+8), chunk 1 = MSB limb (+0).
        for k in 0..8 {
            tr.rows.push((p + 8 + k) as u32);
        }
        for k in 0..8 {
            tr.rows.push((p + k) as u32);
        }
        tr.rows_per_carrier.push(2);
    } else {
        // single chunk = MSB limb (+0).
        for k in 0..8 {
            tr.rows.push((p + k) as u32);
        }
        tr.rows_per_carrier.push(1);
    }
    tr.carrier_seg.push(c.seg as i64);
}

/// ALG-8 VC2 variable-length multi-chunk packing. `K_full = max(1,
/// ceil(L/8))`; `K_visible = 1 + min(trailing_painted_run, K_full - 1)`.
/// Each chunk c spans `[p + L - 8*(c+1), p + L - 8*c)` intersected with
/// `[p, p + L)`; bytes below `p` reference inline_bytes[0] (pad => 0).
#[allow(clippy::too_many_arguments)]
fn emit_vc2(
    c: &Carrier,
    tr: &mut TypeRows,
    seg_runlen_base: &[i64],
    runlen_number_flat: &[i64],
    runlen_len: i64,
    seg_painted_vc2_flat: &[i64],
    seg_painted_offsets: &[i64],
    seg_surviving: &[i64],
    vc2_chunk_indices: &mut Vec<i64>,
) -> Result<(), String> {
    // L = runlen_number[p_carrier + 1] from the owning segment, guarded to
    // the carrier's OWN segment (out of range => L = 0 => K_full = 1).
    let seg = c.seg;
    let seg_runlen_len = seg_len_at(seg_runlen_base, runlen_len, seg);
    let lookahead_raw = c.raw_position + 1;
    let l = if lookahead_raw < seg_runlen_len {
        let gather = seg_runlen_base[seg] + lookahead_raw;
        runlen_number_flat[gather as usize]
    } else {
        0
    };
    let k_full = std::cmp::max(1i64, (l + 7) / 8);

    // Segmented trailing-painted-run at expanded_position + 1, capped by
    // the surviving prefix. `run` = count of consecutive painted-True
    // positions at lookahead within the segment's surviving prefix.
    let lookahead = c.expanded_position + 1;
    let surviving = seg_surviving[seg];
    let run_lengths = if lookahead < surviving {
        // flat index into the per-segment painted prefix.
        let seg_off = seg_painted_offsets[seg];
        let seg_end = seg_painted_offsets[seg + 1];
        let start = seg_off + lookahead;
        // count consecutive True from `start` until first False or seg_end.
        let mut run = 0i64;
        let mut idx = start;
        while idx < seg_end && seg_painted_vc2_flat[idx as usize] != 0 {
            run += 1;
            idx += 1;
        }
        run
    } else {
        0
    };
    let k_visible = 1 + std::cmp::min(run_lengths, k_full - 1);

    let p = c.byte_offset;
    for chunk in 0..k_visible {
        let unclipped_start = p + l - 8 * (chunk + 1);
        for b in 0..8 {
            let col = unclipped_start + b;
            if col < p {
                tr.rows.push(0u32);
            } else {
                tr.rows.push(col as u32);
            }
        }
        vc2_chunk_indices.push(chunk);
    }
    tr.rows_per_carrier.push(k_visible);
    tr.carrier_seg.push(seg as i64);
    Ok(())
}

/// PyO3 entry. Takes the flat per-segment arrays + band bounds + the
/// per-block fixed widths; returns a tuple
/// `(per_block_list, f128_is_nan_or_inf, vc2_chunk_indices)` where
/// `per_block_list[b] = (rows_2d_u32, rows_per_carrier_i64,
/// carrier_seg_i64)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_number_idx_2d_kernel<'py>(
    py: Python<'py>,
    expanded_body: numpy::PyReadonlyArray1<'py, i64>,
    painted_body: numpy::PyReadonlyArray1<'py, bool>,
    body_seg_len: numpy::PyReadonlyArray1<'py, i64>,
    real_pos_flat: numpy::PyReadonlyArray1<'py, i64>,
    real_seg_base: numpy::PyReadonlyArray1<'py, i64>,
    digit_flat: numpy::PyReadonlyArray1<'py, i64>,
    digit_base: numpy::PyReadonlyArray1<'py, i64>,
    seg_slice_start: numpy::PyReadonlyArray1<'py, i64>,
    seg_painted_vc2_flat: numpy::PyReadonlyArray1<'py, i64>,
    seg_painted_offsets: numpy::PyReadonlyArray1<'py, i64>,
    seg_surviving: numpy::PyReadonlyArray1<'py, i64>,
    seg_runlen_base: numpy::PyReadonlyArray1<'py, i64>,
    runlen_number_flat: numpy::PyReadonlyArray1<'py, i64>,
    seg_f128_base: numpy::PyReadonlyArray1<'py, i64>,
    f128_full_mask_flat: numpy::PyReadonlyArray1<'py, bool>,
    band_lo: i64,
    band_hi: i64,
    fixed_widths: Vec<usize>,
) -> PyResult<Bound<'py, PyTuple>> {
    let expanded_body = expanded_body.as_slice()?;
    let painted_body = painted_body.as_slice()?;
    let body_seg_len = body_seg_len.as_slice()?;
    let real_pos_flat = real_pos_flat.as_slice()?;
    let real_seg_base = real_seg_base.as_slice()?;
    let digit_flat = digit_flat.as_slice()?;
    let digit_base = digit_base.as_slice()?;
    let seg_slice_start = seg_slice_start.as_slice()?;
    let seg_painted_vc2_flat = seg_painted_vc2_flat.as_slice()?;
    let seg_painted_offsets = seg_painted_offsets.as_slice()?;
    let seg_surviving = seg_surviving.as_slice()?;
    let seg_runlen_base = seg_runlen_base.as_slice()?;
    let runlen_number_flat = runlen_number_flat.as_slice()?;
    let seg_f128_base = seg_f128_base.as_slice()?;
    let f128_full_mask_flat = f128_full_mask_flat.as_slice()?;

    let (per_block, f128_flags, vc2_chunk_indices) = py
        .detach(|| {
            run_kernel(
                expanded_body,
                painted_body,
                body_seg_len,
                real_pos_flat,
                real_seg_base,
                digit_flat,
                digit_base,
                seg_slice_start,
                seg_painted_vc2_flat,
                seg_painted_offsets,
                seg_surviving,
                seg_runlen_base,
                runlen_number_flat,
                seg_f128_base,
                f128_full_mask_flat,
                band_lo,
                band_hi,
                &fixed_widths,
            )
        })
        .map_err(PyValueError::new_err)?;

    let block_list = PyList::empty(py);
    for tr in per_block.into_iter() {
        let n_rows = tr.rows_per_carrier.iter().sum::<i64>() as usize;
        let rows_2d = PyArray2::<u32>::zeros(py, [n_rows, tr.width], false);
        // SAFETY: freshly allocated, exclusively owned here.
        unsafe {
            rows_2d.as_slice_mut()?.copy_from_slice(&tr.rows);
        }
        let rpc = PyArray1::from_vec(py, tr.rows_per_carrier);
        let cseg = PyArray1::from_vec(py, tr.carrier_seg);
        let triple = PyTuple::new(py, [rows_2d.into_any(), rpc.into_any(), cseg.into_any()])?;
        block_list.append(triple)?;
    }
    let f128_arr = PyArray1::from_vec(py, f128_flags);
    let vc2_arr = PyArray1::from_vec(py, vc2_chunk_indices);
    PyTuple::new(
        py,
        [
            block_list.into_any(),
            f128_arr.into_any(),
            vc2_arr.into_any(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default NUMBER-block widths: VC2=8, F16=2, BF16=2, F32=4, F64=8,
    // F80=10, F128=8.
    fn widths() -> Vec<usize> {
        vec![8, 2, 2, 4, 8, 10, 8]
    }

    /// Single-segment helper: build the flat inputs for one call_target
    /// whose body axis is `expanded[1:surviving]`. `real_positions` lists
    /// the raw positions of real tokens; `digit_cumsum` the N+1 prefix.
    #[allow(clippy::too_many_arguments)]
    fn one_seg(
        expanded_body: Vec<i64>,
        painted_body: Vec<bool>,
        real_positions: Vec<i64>,
        digit_cumsum: Vec<i64>,
        runlen_number: Vec<i64>,
        f128_full: Vec<bool>,
        painted_vc2_prefix: Vec<i64>,
        surviving: i64,
        slice_start: i64,
    ) -> (Vec<TypeRows>, Vec<bool>, Vec<i64>) {
        let body_len = expanded_body.len() as i64;
        run_kernel(
            &expanded_body,
            &painted_body,
            &[body_len],
            &real_positions,
            &[0],
            &digit_cumsum,
            &[0],
            &[slice_start],
            &painted_vc2_prefix,
            &[0, surviving],
            &[surviving],
            &[0],
            &runlen_number,
            &[0],
            &f128_full,
            1,
            8,
            &widths(),
        )
        .unwrap()
    }

    #[test]
    fn f32_single_row() {
        // raw: [F32_carrier, b0,b1,b2,b3]; real token at pos 0; digit_cumsum
        // counts inline-digit bytes: [0,0,1,2,3,4]. expanded body = [F32=4]
        // (the prepend is dropped from the body axis).
        let (per_block, f128, vc2) = one_seg(
            vec![4],          // F32 shifted id at body slot 0
            vec![false],
            vec![0],          // real position 0 (the carrier)
            vec![0, 0, 1, 2, 3, 4],
            vec![0, 1, 1, 1, 1],
            vec![false],      // f128 full mask (unused here)
            vec![0, 0],       // vc2 painted prefix [:surviving]
            2,                // surviving
            1,                // slice_start
        );
        // block 3 == F32, width 4, one row [1,2,3,4].
        assert_eq!(per_block[3].rows, vec![1, 2, 3, 4]);
        assert_eq!(per_block[3].rows_per_carrier, vec![1]);
        assert!(f128.is_empty());
        assert!(vc2.is_empty());
    }

    #[test]
    fn f128_finite_two_chunks() {
        // F128 carrier with painted continuation -> finite -> 2 chunks.
        // body = [F128=7, F128=7]; painted at body slot 1 (f128 mask).
        // full f128 mask over expanded (len 3 incl prepend): [F,F,T].
        let (per_block, f128, _vc2) = one_seg(
            vec![7, 7],
            vec![false, true],
            vec![0],
            vec![0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            vec![0; 17],
            vec![false, false, true], // full f128 mask: carrier+1 painted
            vec![0, 0, 0],
            3,
            1,
        );
        // block 6 == F128, width 8. LSB limb (p+8..p+16) then MSB (p..p+8),
        // p=1 -> [9..17) then [1..9).
        assert_eq!(
            per_block[6].rows,
            vec![9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(per_block[6].rows_per_carrier, vec![2]);
        assert_eq!(f128, vec![false]);
    }

    #[test]
    fn f128_nan_inf_one_chunk() {
        // No painted continuation -> NaN/Inf -> 1 chunk (MSB limb).
        let (per_block, f128, _vc2) = one_seg(
            vec![7],
            vec![false],
            vec![0],
            vec![0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            vec![0; 17],
            vec![false, false], // full f128 mask: carrier+1 NOT painted
            vec![0, 0],
            2,
            1,
        );
        assert_eq!(per_block[6].rows, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(per_block[6].rows_per_carrier, vec![1]);
        assert_eq!(f128, vec![true]);
    }

    #[test]
    fn vc2_l17_three_chunks_msb_pad() {
        // L=17 -> K=3; 2 painted continuations. p=1.
        // chunk0 LSB: [p+9..p+17), chunk1: [p+1..p+9), chunk2 MSB: 7 pad + p.
        let (per_block, _f128, vc2) = one_seg(
            vec![1, 1, 1],            // VC2 carrier + 2 painted (body)
            vec![false, true, true],
            vec![0],
            // digit_cumsum N+1 over raw [VC2, 17 bytes]: [0,0,1,...,17].
            {
                let mut dc = vec![0i64];
                for i in 0..=17 {
                    dc.push(i);
                }
                dc
            },
            // runlen_number over raw: [0, 17, ...] (carrier+1 => 17).
            {
                let mut rl = vec![0i64, 17];
                for _ in 0..16 {
                    rl.push(0);
                }
                rl
            },
            vec![false; 4],
            // painted vc2 prefix [:surviving=4]: [F,F,T,T].
            vec![0, 0, 1, 1],
            4,
            1,
        );
        let p = 1i64;
        let expect: Vec<u32> = vec![
            // chunk 0
            (p + 9) as u32, (p + 10) as u32, (p + 11) as u32, (p + 12) as u32,
            (p + 13) as u32, (p + 14) as u32, (p + 15) as u32, (p + 16) as u32,
            // chunk 1
            (p + 1) as u32, (p + 2) as u32, (p + 3) as u32, (p + 4) as u32,
            (p + 5) as u32, (p + 6) as u32, (p + 7) as u32, (p + 8) as u32,
            // chunk 2 (MSB): 7 pad + p
            0, 0, 0, 0, 0, 0, 0, p as u32,
        ];
        assert_eq!(per_block[0].rows, expect);
        assert_eq!(per_block[0].rows_per_carrier, vec![3]);
        assert_eq!(vc2, vec![0, 1, 2]);
    }

    #[test]
    fn vc2_l0_one_chunk_all_pad() {
        // L=0 -> K=1, single all-pad chunk. runlen_number[carrier+1] = 0.
        let (per_block, _f128, vc2) = one_seg(
            vec![1],
            vec![false],
            vec![0],
            vec![0, 0],
            vec![0, 0],
            vec![false],
            vec![0, 0],
            2,
            1,
        );
        assert_eq!(per_block[0].rows, vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(per_block[0].rows_per_carrier, vec![1]);
        assert_eq!(vc2, vec![0]);
    }

    #[test]
    fn vc2_mid_cut_drops_msb() {
        // L=17 (K=3) but surviving cuts after the first continuation ->
        // K_visible = 2 (chunks 0,1 only). painted prefix [:surviving=3].
        let (per_block, _f128, vc2) = one_seg(
            vec![1, 1], // body has only carrier + 1 surviving painted
            vec![false, true],
            vec![0],
            {
                let mut dc = vec![0i64];
                for i in 0..=17 {
                    dc.push(i);
                }
                dc
            },
            {
                let mut rl = vec![0i64, 17];
                for _ in 0..16 {
                    rl.push(0);
                }
                rl
            },
            vec![false; 3],
            vec![0, 0, 1], // painted prefix [:surviving=3] = [F,F,T]
            3,
            1,
        );
        let p = 1i64;
        let expect: Vec<u32> = vec![
            (p + 9) as u32, (p + 10) as u32, (p + 11) as u32, (p + 12) as u32,
            (p + 13) as u32, (p + 14) as u32, (p + 15) as u32, (p + 16) as u32,
            (p + 1) as u32, (p + 2) as u32, (p + 3) as u32, (p + 4) as u32,
            (p + 5) as u32, (p + 6) as u32, (p + 7) as u32, (p + 8) as u32,
        ];
        assert_eq!(per_block[0].rows, expect);
        assert_eq!(per_block[0].rows_per_carrier, vec![2]);
        assert_eq!(vc2, vec![0, 1]);
    }

    #[test]
    fn adversarial_f128_flag_flip_changes_chunk_count() {
        // Perturbing the F128 finite signal (full-mask continuation flag)
        // MUST change the emitted chunk count (2 finite vs 1 NaN/Inf).
        let finite = one_seg(
            vec![7, 7],
            vec![false, true],
            vec![0],
            vec![0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            vec![0; 17],
            vec![false, false, true],
            vec![0, 0, 0],
            3,
            1,
        );
        let nan = one_seg(
            vec![7],
            vec![false],
            vec![0],
            vec![0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            vec![0; 17],
            vec![false, false], // flipped: no continuation
            vec![0, 0],
            2,
            1,
        );
        assert_ne!(finite.0[6].rows_per_carrier, nan.0[6].rows_per_carrier);
        assert_eq!(finite.0[6].rows_per_carrier, vec![2]);
        assert_eq!(nan.0[6].rows_per_carrier, vec![1]);
    }

    #[test]
    fn vc2_terminal_carrier_no_oob() {
        // Two segments, each with a VC2 carrier at its last raw position.
        // runlen_number_flat = [5,0,0, 5,0,0]; the guard must read L=0 for
        // the terminal carriers, not the neighbour's 5 (or OOB on the last).
        let (per_block, _f128, _vc2) = run_kernel(
            &[1, 1],          // body: one VC2 carrier per segment
            &[false, false],
            &[1, 1],          // body_seg_len: 1 each
            &[2, 2],          // real_pos_flat: raw pos 2 in each seg
            &[0, 1],          // real_seg_base
            &[0, 0, 0, 1, 0, 0, 0, 1], // digit_cumsum: two segs (N+1=4 each)
            &[0, 4],          // digit_base
            &[1, 1],          // seg_slice_start
            &[0, 0, 0, 0, 0, 0], // painted vc2 prefix (none)
            &[0, 3, 6],       // seg_painted_offsets
            &[3, 3],          // seg_surviving
            &[0, 3],          // seg_runlen_base
            &[5, 0, 0, 5, 0, 0], // runlen_number_flat
            &[0, 1],          // seg_f128_base (unused for VC2)
            &[false, false],  // f128 full mask
            1,
            8,
            &widths(),
        )
        .unwrap();
        // Both VC2 carriers -> L=0 -> single all-pad chunk each.
        assert_eq!(per_block[0].rows_per_carrier, vec![1, 1]);
        assert_eq!(per_block[0].rows, vec![0; 16]); // 2 carriers * 8 zeros
    }
}
