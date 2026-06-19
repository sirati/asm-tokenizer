//! `build_inline_bytes_kernel` — stage-3a surviving inline-byte gather.
//!
//! Single concern: given the FLAT `DenseColumns` columns (full-DFS-axis
//! RAW arrays + EXPANDED extra masks + CSR offsets + per-node `is_cut` /
//! `surviving_token_count` scalars), lift every surviving inline-digit
//! byte payload from every level-4 call_target into ONE flat `u8` buffer
//! (leading-zero pad at index 0) and emit the per-node contributed byte
//! COUNT. Python derives `inline_byte_slices` from the counts via cumsum.
//!
//! Mirrors the numpy `_surviving_bytes` + `build_inline_bytes`
//! (`_inline_bytes.py`) EXACTLY — it re-implements NO decode rule beyond
//! reproducing that per-call_target masked gather as a deterministic
//! scalar walk:
//!
//! NOT cut (`is_cut[e] == false`):
//! * emit `raw_tokens[raw_slice][number_mask[raw_slice]]` truncated to
//!   `u8` (the inline-band ids are `< 256` so the truncation is lossless;
//!   numpy's `.astype(np.uint8)` keeps the low byte, `as u8` matches).
//!
//! CUT (`is_cut[e] == true`):
//! * `partial = surviving_token_count[e]`; emit nothing when `partial <= 1`
//!   (only the prepend or nothing survives).
//! * EXPANDED body `[1, partial)`: `painted = extra_value_v2_mask |
//!   extra_f128_mask`; `n_carriers_consumed = count(!painted)`. Emit
//!   nothing when `0`.
//! * RAW carriers = positions where `real_mask` is True; `p_last` = the
//!   `(n_carriers_consumed - 1)`-th such raw position.
//! * `L_last = runlen_number[p_last + 1]` when `p_last + 1 < N` else 0.
//! * Keep every `number_mask` True raw position with index
//!   `< p_last + 1 + L_last`, truncated to `u8`.
//!
//! ## Buffer order
//!
//! Index 0 is the leading-zero pad. Nodes are walked in full-DFS order
//! (`0..n_nodes`); within a node, raw-stream order (ascending). Identical
//! to the numpy concatenation, which preserves that scan order.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// CSR slice `[base[e], base[e + 1])` for node `e`. The base arrays carry
/// the explicit `n_nodes + 1` final boundary (`DenseColumns.raw_offsets` /
/// `node_offsets`), so no `flat_len` fallback is needed.
#[inline]
fn csr_slice(base: &[i64], e: usize) -> (usize, usize) {
    (base[e] as usize, base[e + 1] as usize)
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests can drive it
/// directly. Returns `(inline_bytes, counts)` — the flat `u8` buffer with
/// a leading-zero pad and the per-node contributed byte count.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    raw_tokens: &[u16],
    number_mask: &[bool],
    real_mask: &[bool],
    runlen_number: &[u16],
    extra_value_v2_mask: &[bool],
    extra_f128_mask: &[bool],
    is_cut: &[bool],
    surviving_token_count: &[i64],
    raw_offsets: &[i64],
    node_offsets: &[i64],
) -> Result<(Vec<u8>, Vec<i64>), String> {
    let n_nodes = surviving_token_count.len();
    if is_cut.len() != n_nodes
        || raw_offsets.len() != n_nodes + 1
        || node_offsets.len() != n_nodes + 1
    {
        return Err(format!(
            "per-node arrays disagree on n_nodes ({n_nodes}): is_cut {} \
             raw_offsets {} node_offsets {}",
            is_cut.len(),
            raw_offsets.len(),
            node_offsets.len()
        ));
    }

    // Leading-zero pad at index 0 (plan D9 + Stage 3 step 1).
    let mut inline_bytes: Vec<u8> = vec![0u8];
    let mut counts: Vec<i64> = Vec::with_capacity(n_nodes);

    for e in 0..n_nodes {
        let (raw_lo, raw_hi) = csr_slice(raw_offsets, e);
        let n = raw_hi - raw_lo;
        if raw_hi > raw_tokens.len()
            || raw_hi > number_mask.len()
            || raw_hi > real_mask.len()
        {
            return Err(format!(
                "node {e} raw slice [{raw_lo}, {raw_hi}) exceeds flat \
                 raw_tokens / number_mask / real_mask arrays"
            ));
        }

        let before = inline_bytes.len();

        if !is_cut[e] {
            // ---- fully-included path: raw_tokens[number_mask] -> u8 ----
            for local in 0..n {
                let p = raw_lo + local;
                if number_mask[p] {
                    inline_bytes.push(raw_tokens[p] as u8);
                }
            }
            counts.push((inline_bytes.len() - before) as i64);
            continue;
        }

        // ---- cut path ----
        let partial = surviving_token_count[e];
        if partial <= 1 {
            counts.push(0);
            continue;
        }

        // EXPANDED body axis = node_slice[1:partial]. `painted` slots
        // (extra_*_mask True) borrow their carrier's raw position; only
        // non-painted (real) body slots consume a fresh raw carrier.
        let (exp_lo, exp_hi) = csr_slice(node_offsets, e);
        let body_len = (partial - 1) as usize;
        let body_start = exp_lo + 1;
        let body_end = body_start + body_len;
        if body_end > exp_hi
            || body_end > extra_value_v2_mask.len()
            || body_end > extra_f128_mask.len()
        {
            return Err(format!(
                "node {e} cut body [{body_start}, {body_end}) exceeds \
                 expanded slice [{exp_lo}, {exp_hi}) or flat masks"
            ));
        }

        let mut n_carriers_consumed: i64 = 0;
        for j in 0..body_len {
            let painted =
                extra_value_v2_mask[body_start + j] || extra_f128_mask[body_start + j];
            if !painted {
                n_carriers_consumed += 1;
            }
        }
        if n_carriers_consumed == 0 {
            counts.push(0);
            continue;
        }

        // Raw positions of carriers (np.nonzero(real_mask[raw_slice])).
        // p_last = the (n_carriers_consumed - 1)-th raw carrier position.
        let target = (n_carriers_consumed - 1) as usize;
        let mut seen: usize = 0;
        let mut p_last: Option<usize> = None;
        for local in 0..n {
            if real_mask[raw_lo + local] {
                if seen == target {
                    p_last = Some(local);
                    break;
                }
                seen += 1;
            }
        }
        let p_last = p_last.ok_or_else(|| {
            format!(
                "node {e} carrier index {target} exceeds the node's \
                 real-position count"
            )
        })?;

        // L_last = runlen_number[p_last + 1] when p_last + 1 < N else 0.
        let l_last: i64 = if p_last + 1 < n {
            let idx = raw_lo + p_last + 1;
            if idx >= runlen_number.len() {
                return Err(format!(
                    "node {e} runlen idx {idx} out of bounds"
                ));
            }
            i64::from(runlen_number[idx])
        } else {
            0
        };

        // number_mask_keep: keep number_mask True positions with local
        // raw index < p_last + 1 + L_last. (number_mask[cutoff:] = False.)
        let cutoff = p_last as i64 + 1 + l_last;
        for local in 0..n {
            if (local as i64) >= cutoff {
                break;
            }
            let p = raw_lo + local;
            if number_mask[p] {
                inline_bytes.push(raw_tokens[p] as u8);
            }
        }
        counts.push((inline_bytes.len() - before) as i64);
    }

    Ok((inline_bytes, counts))
}

/// PyO3 wrapper: borrow the flat `DenseColumns` columns, run the gather
/// under `py.detach`, and hand back `(inline_bytes_u8, counts_i64)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_inline_bytes_kernel<'py>(
    py: Python<'py>,
    raw_tokens: numpy::PyReadonlyArray1<'py, u16>,
    number_mask: numpy::PyReadonlyArray1<'py, bool>,
    real_mask: numpy::PyReadonlyArray1<'py, bool>,
    runlen_number: numpy::PyReadonlyArray1<'py, u16>,
    extra_value_v2_mask: numpy::PyReadonlyArray1<'py, bool>,
    extra_f128_mask: numpy::PyReadonlyArray1<'py, bool>,
    is_cut: numpy::PyReadonlyArray1<'py, bool>,
    surviving_token_count: numpy::PyReadonlyArray1<'py, i64>,
    raw_offsets: numpy::PyReadonlyArray1<'py, i64>,
    node_offsets: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let raw_tokens = raw_tokens.as_slice()?;
    let number_mask = number_mask.as_slice()?;
    let real_mask = real_mask.as_slice()?;
    let runlen_number = runlen_number.as_slice()?;
    let extra_value_v2_mask = extra_value_v2_mask.as_slice()?;
    let extra_f128_mask = extra_f128_mask.as_slice()?;
    let is_cut = is_cut.as_slice()?;
    let surviving_token_count = surviving_token_count.as_slice()?;
    let raw_offsets = raw_offsets.as_slice()?;
    let node_offsets = node_offsets.as_slice()?;

    let (inline_bytes, counts) = py
        .detach(|| {
            run_kernel(
                raw_tokens,
                number_mask,
                real_mask,
                runlen_number,
                extra_value_v2_mask,
                extra_f128_mask,
                is_cut,
                surviving_token_count,
                raw_offsets,
                node_offsets,
            )
        })
        .map_err(PyValueError::new_err)?;

    let bytes_arr = PyArray1::from_vec(py, inline_bytes);
    let counts_arr = PyArray1::from_vec(py, counts);
    PyTuple::new(py, [bytes_arr.into_any(), counts_arr.into_any()])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `run_kernel` over a SINGLE node built from a raw stream.
    /// `number_mask` / `real_mask` are derived the way the production
    /// `InlineDecodeState` derives them: number = `tok < 257`,
    /// real = `tok > 256`. `runlen_number` is the run-start length of the
    /// `number_mask` runs (production `run_lengths(number_mask)`).
    fn one_node(
        raw: Vec<u16>,
        extra_vc2: Vec<bool>,
        extra_f128: Vec<bool>,
        is_cut: bool,
        surviving: i64,
    ) -> (Vec<u8>, Vec<i64>) {
        let n = raw.len();
        let number_mask: Vec<bool> = raw.iter().map(|&t| t < 257).collect();
        let real_mask: Vec<bool> = raw.iter().map(|&t| t > 256).collect();
        let runlen_number = run_lengths(&number_mask);
        // expanded length = 1 (prepend) + #real carriers + #painted; the
        // tests pass extra masks sized to the expanded stream directly.
        let exp_len = extra_vc2.len();
        run_kernel(
            &raw,
            &number_mask,
            &real_mask,
            &runlen_number,
            &extra_vc2,
            &extra_f128,
            &[is_cut],
            &[surviving],
            &[0, n as i64],
            &[0, exp_len as i64],
        )
        .unwrap()
    }

    /// Reference `run_lengths`: for a boolean mask, each run-start slot
    /// carries the run length, other slots carry 0 (production semantics).
    fn run_lengths(mask: &[bool]) -> Vec<u16> {
        let mut out = vec![0u16; mask.len()];
        let mut i = 0;
        while i < mask.len() {
            if mask[i] {
                let start = i;
                while i < mask.len() && mask[i] {
                    i += 1;
                }
                out[start] = (i - start) as u16;
            } else {
                i += 1;
            }
        }
        out
    }

    #[test]
    fn not_cut_gathers_number_mask_bytes() {
        // F16 carrier (258) + 2 inline bytes, then identity carrier (264)
        // + 1 inline byte. number_mask = [F,T,T,F,T] -> bytes [0x12,0x34,0xAB].
        let (bytes, counts) = one_node(
            vec![258, 0x12, 0x34, 264, 0xAB],
            // expanded body irrelevant when not cut; pass a 1-slot prepend.
            vec![false],
            vec![false],
            false,
            999,
        );
        assert_eq!(bytes, vec![0, 0x12, 0x34, 0xAB]);
        assert_eq!(counts, vec![3]);
    }

    #[test]
    fn not_cut_truncates_to_low_byte() {
        // a raw token in the number band stays < 256, but assert the
        // narrowing keeps the low byte exactly across the u8 range.
        let (bytes, counts) =
            one_node(vec![264, 0x00, 0xFF, 0x80], vec![false], vec![false], false, 999);
        assert_eq!(bytes, vec![0, 0x00, 0xFF, 0x80]);
        assert_eq!(counts, vec![3]);
    }

    #[test]
    fn cut_at_one_emits_nothing() {
        // partial_cut_length <= 1 -> only the prepend survives.
        let (bytes, counts) =
            one_node(vec![258, 0x01, 0x02], vec![false], vec![false], true, 1);
        assert_eq!(bytes, vec![0]);
        assert_eq!(counts, vec![0]);
    }

    #[test]
    fn cut_keeps_full_payload_of_last_consumed_carrier() {
        // VC2 (257? use a real carrier id) L=5 single chunk. body =
        // [prepend, VC2_carrier]; cut surviving=2 -> 1 carrier consumed,
        // p_last=0, L_last=5 -> keep all 5 bytes.
        // raw: [VC2=257, b0..b4]; number_mask=[F,T,T,T,T,T].
        let raw = vec![257, 0x10, 0x20, 0x30, 0x40, 0x50];
        let (bytes, counts) = one_node(
            raw,
            vec![false, false], // prepend + carrier (no painted)
            vec![false, false],
            true,
            2,
        );
        assert_eq!(bytes, vec![0, 0x10, 0x20, 0x30, 0x40, 0x50]);
        assert_eq!(counts, vec![5]);
    }

    #[test]
    fn cut_multichunk_keeps_full_payload_even_when_chunk_dropped() {
        // VC2 L=17 (K=3): body=[prepend,carrier,painted1,painted2]. Cut
        // surviving=3 -> body[1:3]=[carrier,painted1]; 1 non-painted ->
        // 1 carrier consumed; p_last=0, L_last=17 -> all 17 bytes kept.
        let payload: Vec<u16> = (1..=17).collect();
        let mut raw = vec![257u16];
        raw.extend(payload.iter());
        let (bytes, counts) = one_node(
            raw,
            vec![false, false, true, true], // prepend, carrier, 2 painted
            vec![false, false, false, false],
            true,
            3,
        );
        let expected: Vec<u8> = std::iter::once(0u8)
            .chain((1u8..=17u8).map(|b| b))
            .collect();
        assert_eq!(bytes, expected);
        assert_eq!(counts, vec![17]);
    }

    #[test]
    fn cut_drops_trailing_carrier_bytes() {
        // Two single-chunk carriers F16(258,L=2) then F64(261,L=2 here).
        // raw: [258, a0, a1, 261, b0, b1]. body=[prepend, c1, c2]. Cut
        // surviving=2 -> only first carrier consumed -> p_last=0, L_last=2
        // -> keep bytes up to local index 0+1+2=3 -> [a0,a1]; b0,b1 dropped.
        let raw = vec![258, 0xA0, 0xA1, 261, 0xB0, 0xB1];
        let (bytes, counts) = one_node(
            raw,
            vec![false, false, false], // prepend + 2 real carriers
            vec![false, false, false],
            true,
            2,
        );
        assert_eq!(bytes, vec![0, 0xA0, 0xA1]);
        assert_eq!(counts, vec![2]);
    }

    #[test]
    fn cut_second_carrier_consumed_keeps_both() {
        // Same two carriers, cut surviving=3 -> both consumed; p_last is
        // raw pos of 2nd carrier (local 3), L_last=2 -> keep up to local
        // 3+1+2=6 -> all 4 bytes.
        let raw = vec![258, 0xA0, 0xA1, 261, 0xB0, 0xB1];
        let (bytes, counts) =
            one_node(raw, vec![false, false, false], vec![false, false, false], true, 3);
        assert_eq!(bytes, vec![0, 0xA0, 0xA1, 0xB0, 0xB1]);
        assert_eq!(counts, vec![4]);
    }

    #[test]
    fn empty_node_zero_count() {
        let (bytes, counts) =
            one_node(vec![], vec![false], vec![false], false, 999);
        assert_eq!(bytes, vec![0]);
        assert_eq!(counts, vec![0]);
    }

    #[test]
    fn multi_node_pad_and_abutting() {
        // node0: F16 2 bytes; node1: identity 1 byte. Pad once, counts
        // [2, 1].
        let raw = vec![258, 0x01, 0x02, /* node1 */ 264, 0xFF];
        let number_mask: Vec<bool> = raw.iter().map(|&t| t < 257).collect();
        let real_mask: Vec<bool> = raw.iter().map(|&t| t > 256).collect();
        let runlen = run_lengths(&number_mask);
        let (bytes, counts) = run_kernel(
            &raw,
            &number_mask,
            &real_mask,
            &runlen,
            &[false, false], // expanded: prepend per node (not cut, unused)
            &[false, false],
            &[false, false],
            &[999, 999],
            &[0, 3, 5],
            &[0, 1, 2],
        )
        .unwrap();
        assert_eq!(bytes, vec![0, 0x01, 0x02, 0xFF]);
        assert_eq!(counts, vec![2, 1]);
    }

    #[test]
    fn adversarial_perturbed_cutoff_diverges() {
        // Same as cut_drops_trailing_carrier_bytes; assert that mutating
        // surviving from 2 to 3 (a different cut) changes the output.
        let raw = vec![258, 0xA0, 0xA1, 261, 0xB0, 0xB1];
        let (bytes2, _) = one_node(
            raw.clone(),
            vec![false, false, false],
            vec![false, false, false],
            true,
            2,
        );
        let (bytes3, _) = one_node(
            raw,
            vec![false, false, false],
            vec![false, false, false],
            true,
            3,
        );
        assert_ne!(
            bytes2, bytes3,
            "a different cut MUST yield different surviving bytes"
        );
    }
}
