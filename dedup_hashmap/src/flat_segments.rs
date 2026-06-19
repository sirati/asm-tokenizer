//! `build_flat_segments_kernel` — stage-3c NUMBER-band flat-segment build.
//!
//! Single concern: given the FLAT `DenseColumns` columns (full-DFS-axis
//! RAW + DIGIT + EXPANDED arrays + CSR offsets + per-node
//! `surviving_token_count` scalar), the kept-node DFS index
//! (`kept_node_index`), and each FULL-DFS node's `inline_byte_slices.start`
//! (`slice_start_per_node`), walk every KEPT call_target once and
//! concatenate the per-segment NUMBER-band context arrays the GIL-released
//! number emission kernel (`build_number_idx_2d_kernel`) consumes — exactly
//! the per-kept-node slice + concatenate the numpy `build_flat_segments`
//! (`_flat_segments.py`) performs.
//!
//! This re-implements NO decode rule. It only SLICES + CONCATENATES the
//! `DenseColumns` flats the same way the numpy loop slices the per-node
//! column views; the carrier recovery / byte-offset arithmetic / per-type
//! emission all stay in `build_number_idx_2d_kernel`, reading these arrays.
//!
//! ## Per-kept-node (`i`-th kept node `e = kept_node_index[i]`) emission
//!
//! `surviving = surviving_token_count[e]`; `raw = [raw_off[e], raw_off[e+1])`;
//! `dig = [dig_off[e], dig_off[e+1])`; `exp = [node_off[e], node_off[e+1])`.
//!
//! * `seg_surviving[i] = surviving`
//! * `seg_slice_start[i] = slice_start_per_node[e]`
//! * `body_seg_len[i] = max(surviving - 1, 0)`
//! * `expanded_body  += expanded[exp.0 + 1 .. exp.0 + surviving]` (i64)
//! * `painted_body   += (extra_vc2 | extra_f128)[exp.0 + 1 .. exp.0 + surviving]`
//! * `seg_painted_vc2 += extra_vc2[exp.0 .. exp.0 + surviving]` (i64);
//!   `painted_prefix_seg_len[i] = surviving`
//! * `real_pos_flat  += local raw indices where real_mask[raw] is True`;
//!   `real_seg_base[i]` = running base
//! * `digit_flat     += digit_cumsum[dig]`; `digit_base[i]` = running base
//! * `runlen_flat    += runlen_number[raw]`; `seg_runlen_base[i]` = base
//! * `f128_full_flat += extra_f128[exp]` (FULL, not surviving-clipped);
//!   `seg_f128_base[i]` = base
//!
//! `seg_painted_offsets` is the `cumsum` of `painted_prefix_seg_len`
//! (length `n_kept + 1`). The numpy twin's `painted_prefix_seg_len` IS
//! `surviving`, so `seg_painted_offsets[i+1] - seg_painted_offsets[i] ==
//! seg_surviving[i]`.
//!
//! ## Order
//!
//! Kept nodes are walked in `kept_node_index` order (== ascending DFS /
//! emission order), and each per-segment concatenation appends in that
//! order — byte-identical to the numpy `np.concatenate` of the per-node
//! chunks, which preserves the same scan order.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// All flat outputs of the kernel, in `FlatSegments` field order (the
/// `*_flat` concatenations + the per-segment CSR bases / scalars).
struct FlatSegmentsOut {
    expanded_body: Vec<i64>,
    painted_body: Vec<bool>,
    body_seg_len: Vec<i64>,
    real_pos_flat: Vec<i64>,
    real_seg_base: Vec<i64>,
    digit_flat: Vec<i64>,
    digit_base: Vec<i64>,
    seg_slice_start: Vec<i64>,
    seg_painted_vc2_flat: Vec<i64>,
    seg_painted_offsets: Vec<i64>,
    seg_surviving: Vec<i64>,
    seg_runlen_base: Vec<i64>,
    runlen_number_flat: Vec<i64>,
    seg_f128_base: Vec<i64>,
    f128_full_mask_flat: Vec<bool>,
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it
/// directly. Mirrors numpy `build_flat_segments` over `DenseColumns`.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    surviving_token_count: &[i64],
    expanded: &[u16],
    extra_value_v2_mask: &[bool],
    extra_f128_mask: &[bool],
    real_mask: &[bool],
    runlen_number: &[u16],
    digit_cumsum: &[u32],
    raw_offsets: &[i64],
    digit_offsets: &[i64],
    node_offsets: &[i64],
    kept_node_index: &[i64],
    slice_start_per_node: &[i64],
) -> Result<FlatSegmentsOut, String> {
    let n_nodes = surviving_token_count.len();
    if raw_offsets.len() != n_nodes + 1
        || digit_offsets.len() != n_nodes + 1
        || node_offsets.len() != n_nodes + 1
        || slice_start_per_node.len() != n_nodes
    {
        return Err(format!(
            "per-node arrays disagree on n_nodes ({n_nodes}): raw_offsets {} \
             digit_offsets {} node_offsets {} slice_start_per_node {}",
            raw_offsets.len(),
            digit_offsets.len(),
            node_offsets.len(),
            slice_start_per_node.len(),
        ));
    }

    let n_kept = kept_node_index.len();
    let mut out = FlatSegmentsOut {
        expanded_body: Vec::new(),
        painted_body: Vec::new(),
        body_seg_len: Vec::with_capacity(n_kept),
        real_pos_flat: Vec::new(),
        real_seg_base: Vec::with_capacity(n_kept),
        digit_flat: Vec::new(),
        digit_base: Vec::with_capacity(n_kept),
        seg_slice_start: Vec::with_capacity(n_kept),
        seg_painted_vc2_flat: Vec::new(),
        seg_painted_offsets: vec![0i64; n_kept + 1],
        seg_surviving: Vec::with_capacity(n_kept),
        seg_runlen_base: Vec::with_capacity(n_kept),
        runlen_number_flat: Vec::new(),
        seg_f128_base: Vec::with_capacity(n_kept),
        f128_full_mask_flat: Vec::new(),
    };

    let mut real_running: i64 = 0;
    let mut digit_running: i64 = 0;
    let mut runlen_running: i64 = 0;
    let mut f128_running: i64 = 0;

    for (i, &e_i64) in kept_node_index.iter().enumerate() {
        if e_i64 < 0 || (e_i64 as usize) >= n_nodes {
            return Err(format!(
                "kept_node_index[{i}] = {e_i64} out of range [0, {n_nodes})"
            ));
        }
        let e = e_i64 as usize;
        let surviving = surviving_token_count[e];
        if surviving < 0 {
            return Err(format!(
                "node {e} surviving_token_count {surviving} is negative"
            ));
        }
        let surviving_u = surviving as usize;

        // RAW-space slice.
        let raw_lo = raw_offsets[e] as usize;
        let raw_hi = raw_offsets[e + 1] as usize;
        if raw_hi > real_mask.len() || raw_hi > runlen_number.len() {
            return Err(format!(
                "node {e} raw slice [{raw_lo}, {raw_hi}) exceeds flat \
                 real_mask / runlen_number arrays"
            ));
        }
        // DIGIT-cumsum slice.
        let dig_lo = digit_offsets[e] as usize;
        let dig_hi = digit_offsets[e + 1] as usize;
        if dig_hi > digit_cumsum.len() {
            return Err(format!(
                "node {e} digit slice [{dig_lo}, {dig_hi}) exceeds flat \
                 digit_cumsum array"
            ));
        }
        // EXPANDED-space slice.
        let exp_lo = node_offsets[e] as usize;
        let exp_hi = node_offsets[e + 1] as usize;
        if exp_hi > expanded.len()
            || exp_hi > extra_value_v2_mask.len()
            || exp_hi > extra_f128_mask.len()
        {
            return Err(format!(
                "node {e} expanded slice [{exp_lo}, {exp_hi}) exceeds flat \
                 expanded / extra_value_v2_mask / extra_f128_mask arrays"
            ));
        }
        // The surviving prefix must fit the expanded slice (surviving <=
        // predicted_full_length); guard so the body / vc2-prefix windows
        // are in-bounds even on a malformed cut.
        if exp_lo + surviving_u > exp_hi {
            return Err(format!(
                "node {e} surviving {surviving} exceeds expanded length \
                 {} (slice [{exp_lo}, {exp_hi}))",
                exp_hi - exp_lo
            ));
        }

        out.seg_surviving.push(surviving);
        out.seg_slice_start.push(slice_start_per_node[e]);
        let body = if surviving > 1 { surviving - 1 } else { 0 };
        out.body_seg_len.push(body);

        // Body axis: expanded[1:surviving] + painted (vc2 | f128) over the
        // same body positions (slot j == expanded[j + 1]).
        let body_start = exp_lo + 1;
        let body_end = exp_lo + surviving_u; // == body_start + body
        for p in body_start..body_end {
            out.expanded_body.push(i64::from(expanded[p]));
            out.painted_body
                .push(extra_value_v2_mask[p] || extra_f128_mask[p]);
        }

        // Surviving-prefix VC2 painted mask (axis [:surviving]).
        for p in exp_lo..exp_lo + surviving_u {
            out.seg_painted_vc2_flat
                .push(extra_value_v2_mask[p] as i64);
        }
        // painted_prefix_seg_len[i] == surviving.
        out.seg_painted_offsets[i + 1] =
            out.seg_painted_offsets[i] + surviving;

        // real positions = local raw indices where real_mask is True.
        out.real_seg_base.push(real_running);
        let mut n_real: i64 = 0;
        for local in 0..(raw_hi - raw_lo) {
            if real_mask[raw_lo + local] {
                out.real_pos_flat.push(local as i64);
                n_real += 1;
            }
        }
        real_running += n_real;

        // digit_cumsum (N + 1 slots per segment).
        out.digit_base.push(digit_running);
        for p in dig_lo..dig_hi {
            out.digit_flat.push(i64::from(digit_cumsum[p]));
        }
        digit_running += (dig_hi - dig_lo) as i64;

        // runlen_number over the raw slice.
        out.seg_runlen_base.push(runlen_running);
        for p in raw_lo..raw_hi {
            out.runlen_number_flat.push(i64::from(runlen_number[p]));
        }
        runlen_running += (raw_hi - raw_lo) as i64;

        // FULL extra_f128_mask (NOT surviving-clipped) over the expanded
        // slice.
        out.seg_f128_base.push(f128_running);
        for p in exp_lo..exp_hi {
            out.f128_full_mask_flat.push(extra_f128_mask[p]);
        }
        f128_running += (exp_hi - exp_lo) as i64;
    }

    Ok(out)
}

/// PyO3 wrapper: borrow the flat `DenseColumns` columns + kept index +
/// per-node slice starts, run the concat under `py.detach`, and return the
/// 15 flat arrays in `FlatSegments` field order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_flat_segments_kernel<'py>(
    py: Python<'py>,
    surviving_token_count: numpy::PyReadonlyArray1<'py, i64>,
    expanded: numpy::PyReadonlyArray1<'py, u16>,
    extra_value_v2_mask: numpy::PyReadonlyArray1<'py, bool>,
    extra_f128_mask: numpy::PyReadonlyArray1<'py, bool>,
    real_mask: numpy::PyReadonlyArray1<'py, bool>,
    runlen_number: numpy::PyReadonlyArray1<'py, u16>,
    digit_cumsum: numpy::PyReadonlyArray1<'py, u32>,
    raw_offsets: numpy::PyReadonlyArray1<'py, i64>,
    digit_offsets: numpy::PyReadonlyArray1<'py, i64>,
    node_offsets: numpy::PyReadonlyArray1<'py, i64>,
    kept_node_index: numpy::PyReadonlyArray1<'py, i64>,
    slice_start_per_node: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let surviving_token_count = surviving_token_count.as_slice()?;
    let expanded = expanded.as_slice()?;
    let extra_value_v2_mask = extra_value_v2_mask.as_slice()?;
    let extra_f128_mask = extra_f128_mask.as_slice()?;
    let real_mask = real_mask.as_slice()?;
    let runlen_number = runlen_number.as_slice()?;
    let digit_cumsum = digit_cumsum.as_slice()?;
    let raw_offsets = raw_offsets.as_slice()?;
    let digit_offsets = digit_offsets.as_slice()?;
    let node_offsets = node_offsets.as_slice()?;
    let kept_node_index = kept_node_index.as_slice()?;
    let slice_start_per_node = slice_start_per_node.as_slice()?;

    let out = py
        .detach(|| {
            run_kernel(
                surviving_token_count,
                expanded,
                extra_value_v2_mask,
                extra_f128_mask,
                real_mask,
                runlen_number,
                digit_cumsum,
                raw_offsets,
                digit_offsets,
                node_offsets,
                kept_node_index,
                slice_start_per_node,
            )
        })
        .map_err(PyValueError::new_err)?;

    let arrays: [Bound<'py, PyAny>; 15] = [
        PyArray1::from_vec(py, out.expanded_body).into_any(),
        PyArray1::from_vec(py, out.painted_body).into_any(),
        PyArray1::from_vec(py, out.body_seg_len).into_any(),
        PyArray1::from_vec(py, out.real_pos_flat).into_any(),
        PyArray1::from_vec(py, out.real_seg_base).into_any(),
        PyArray1::from_vec(py, out.digit_flat).into_any(),
        PyArray1::from_vec(py, out.digit_base).into_any(),
        PyArray1::from_vec(py, out.seg_slice_start).into_any(),
        PyArray1::from_vec(py, out.seg_painted_vc2_flat).into_any(),
        PyArray1::from_vec(py, out.seg_painted_offsets).into_any(),
        PyArray1::from_vec(py, out.seg_surviving).into_any(),
        PyArray1::from_vec(py, out.seg_runlen_base).into_any(),
        PyArray1::from_vec(py, out.runlen_number_flat).into_any(),
        PyArray1::from_vec(py, out.seg_f128_base).into_any(),
        PyArray1::from_vec(py, out.f128_full_mask_flat).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `run_kernel` over a SINGLE kept node. The masks are sized to
    /// the expanded stream (`extra_*`) / raw stream (`real_mask`,
    /// `runlen_number`) / digit-cumsum stream the caller passes.
    #[allow(clippy::too_many_arguments)]
    fn one_node(
        surviving: i64,
        expanded: Vec<u16>,
        extra_vc2: Vec<bool>,
        extra_f128: Vec<bool>,
        real_mask: Vec<bool>,
        runlen_number: Vec<u16>,
        digit_cumsum: Vec<u32>,
        slice_start: i64,
    ) -> FlatSegmentsOut {
        let n_raw = real_mask.len() as i64;
        let n_dig = digit_cumsum.len() as i64;
        let n_exp = expanded.len() as i64;
        run_kernel(
            &[surviving],
            &expanded,
            &extra_vc2,
            &extra_f128,
            &real_mask,
            &runlen_number,
            &digit_cumsum,
            &[0, n_raw],
            &[0, n_dig],
            &[0, n_exp],
            &[0], // kept index = the single node
            &[slice_start],
        )
        .unwrap()
    }

    #[test]
    fn not_cut_full_segment() {
        // expanded = [prepend, c0, c1, c2] (surviving = 4 = full).
        // extra_vc2 = [F, T, F, F]; extra_f128 = [F, F, T, F].
        // real_mask raw = [T, T, F, T]; runlen = [2, 0, 0, 1].
        // digit_cumsum = [0, 1, 2, 3, 5] (N+1 = 5 for N=4 raw).
        let out = one_node(
            4,
            vec![10, 11, 12, 13],
            vec![false, true, false, false],
            vec![false, false, true, false],
            vec![true, true, false, true],
            vec![2, 0, 0, 1],
            vec![0, 1, 2, 3, 5],
            7,
        );
        // body = expanded[1:4] = [11, 12, 13]; painted = vc2|f128 over
        // [1:4] = [T, T, F].
        assert_eq!(out.expanded_body, vec![11, 12, 13]);
        assert_eq!(out.painted_body, vec![true, true, false]);
        assert_eq!(out.body_seg_len, vec![3]);
        // vc2 prefix [:4] = [F,T,F,F] -> [0,1,0,0]; offsets [0,4].
        assert_eq!(out.seg_painted_vc2_flat, vec![0, 1, 0, 0]);
        assert_eq!(out.seg_painted_offsets, vec![0, 4]);
        assert_eq!(out.seg_surviving, vec![4]);
        // real positions where real_mask True: locals 0,1,3.
        assert_eq!(out.real_pos_flat, vec![0, 1, 3]);
        assert_eq!(out.real_seg_base, vec![0]);
        // digit_cumsum copied in full; base 0.
        assert_eq!(out.digit_flat, vec![0, 1, 2, 3, 5]);
        assert_eq!(out.digit_base, vec![0]);
        // runlen over raw; base 0.
        assert_eq!(out.runlen_number_flat, vec![2, 0, 0, 1]);
        assert_eq!(out.seg_runlen_base, vec![0]);
        // f128 FULL over expanded = [F,F,T,F]; base 0.
        assert_eq!(out.f128_full_mask_flat, vec![false, false, true, false]);
        assert_eq!(out.seg_f128_base, vec![0]);
        assert_eq!(out.seg_slice_start, vec![7]);
    }

    #[test]
    fn cut_clips_body_and_vc2_prefix_but_keeps_full_f128() {
        // expanded length 4, surviving = 2 (cut). body = expanded[1:2] =
        // [11]; painted over [1:2]. vc2-prefix [:2]. f128 FULL = all 4.
        let out = one_node(
            2,
            vec![10, 11, 12, 13],
            vec![false, true, true, false],
            vec![true, false, false, true],
            vec![true, true, false, true],
            vec![1, 0, 0, 1],
            vec![0, 1, 2, 3, 5],
            3,
        );
        assert_eq!(out.expanded_body, vec![11]);
        assert_eq!(out.painted_body, vec![true]); // vc2[1]=T
        assert_eq!(out.body_seg_len, vec![1]);
        assert_eq!(out.seg_painted_vc2_flat, vec![0, 1]); // vc2[:2]
        assert_eq!(out.seg_painted_offsets, vec![0, 2]);
        assert_eq!(out.seg_surviving, vec![2]);
        // f128 FULL over the WHOLE expanded slice, not clipped to surviving.
        assert_eq!(
            out.f128_full_mask_flat,
            vec![true, false, false, true]
        );
    }

    #[test]
    fn surviving_one_emits_empty_body_but_one_vc2_prefix() {
        // surviving = 1 -> body empty; vc2-prefix [:1].
        let out = one_node(
            1,
            vec![10, 11, 12],
            vec![true, false, false],
            vec![false, false, false],
            vec![true, true, true],
            vec![3, 0, 0],
            vec![0, 1, 2, 3],
            0,
        );
        assert!(out.expanded_body.is_empty());
        assert!(out.painted_body.is_empty());
        assert_eq!(out.body_seg_len, vec![0]);
        assert_eq!(out.seg_painted_vc2_flat, vec![1]); // vc2[:1] = [T]
        assert_eq!(out.seg_painted_offsets, vec![0, 1]);
        // FULL f128 + runlen + digit still copied over the whole node.
        assert_eq!(out.f128_full_mask_flat.len(), 3);
        assert_eq!(out.runlen_number_flat, vec![3, 0, 0]);
        assert_eq!(out.digit_flat, vec![0, 1, 2, 3]);
    }

    #[test]
    fn multi_node_abutting_bases() {
        // two kept nodes; check the running CSR bases abut.
        let out = run_kernel(
            &[2, 3],
            // expanded: node0 len2, node1 len3.
            &[100, 101, /* node1 */ 200, 201, 202],
            &[false, true, /* node1 */ false, true, false],
            &[false, false, /* node1 */ true, false, false],
            // raw: node0 len2, node1 len2.
            &[true, false, /* node1 */ true, true],
            &[1, 0, /* node1 */ 2, 0],
            // digit: node0 N+1=3, node1 N+1=3.
            &[0, 1, 1, /* node1 */ 0, 2, 2],
            &[0, 2, 4],          // raw_offsets
            &[0, 3, 6],          // digit_offsets
            &[0, 2, 5],          // node_offsets
            &[0, 1],             // kept both
            &[5, 9],             // slice starts
        )
        .unwrap();
        assert_eq!(out.seg_surviving, vec![2, 3]);
        assert_eq!(out.seg_slice_start, vec![5, 9]);
        // node0 real positions: local 0 (real_mask=[T,F]); node1: locals 0,1.
        assert_eq!(out.real_pos_flat, vec![0, 0, 1]);
        assert_eq!(out.real_seg_base, vec![0, 1]); // node1 base = 1
        // digit bases: node0 0, node1 3.
        assert_eq!(out.digit_base, vec![0, 3]);
        // runlen bases: node0 0, node1 2.
        assert_eq!(out.seg_runlen_base, vec![0, 2]);
        // f128 bases: node0 0, node1 2 (expanded lens 2, 3).
        assert_eq!(out.seg_f128_base, vec![0, 2]);
        // painted offsets cumsum of surviving = [0, 2, 5].
        assert_eq!(out.seg_painted_offsets, vec![0, 2, 5]);
    }

    #[test]
    fn empty_kept_yields_only_offset_zero() {
        let out = run_kernel(
            &[0],
            &[10],
            &[false],
            &[false],
            &[true],
            &[1],
            &[0, 1],
            &[0, 1],
            &[0, 2],
            &[0, 1],
            &[],   // no kept nodes
            &[0],
        )
        .unwrap();
        assert!(out.expanded_body.is_empty());
        assert_eq!(out.seg_painted_offsets, vec![0]);
        assert!(out.seg_surviving.is_empty());
    }

    #[test]
    fn adversarial_full_f128_vs_clipped_diverges() {
        // A cut node: the FULL f128 mask MUST differ from a surviving-clip.
        let out = one_node(
            2,
            vec![10, 11, 12, 13],
            vec![false, false, false, false],
            vec![false, false, true, true], // True only PAST the cut
            vec![true, true, false, true],
            vec![1, 0, 0, 1],
            vec![0, 1, 2, 3, 5],
            0,
        );
        // If we had clipped to surviving=2, the trailing True f128 slots
        // would be dropped; the FULL contract keeps them.
        assert_eq!(
            out.f128_full_mask_flat,
            vec![false, false, true, true]
        );
        // sanity: clipped would have been [false, false] only.
        assert_ne!(out.f128_full_mask_flat, vec![false, false]);
    }
}
