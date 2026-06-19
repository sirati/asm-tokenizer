//! `build_family_band_reduction_kernel` — one GIL-free segmented pass over
//! the per-node surviving prefix shared by THREE vector-decode call sites.
//!
//! Single concern: given the flat post-promotion / post-strip / post-shift
//! `expanded` u16 stream, the per-node CSR `node_offsets`, and the per-node
//! `surviving` prefix length, walk every node's surviving prefix ONCE and
//! emit everything the three callers recompute from the SAME preamble
//! (`node_id`/`offset_in_node`/`within = offset_in_node < surviving[node]`):
//!
//! * `count_surviving_batched` (`_surviving_counts.py`): the per-node
//!   IDENTITY-band and NUMBER-band cardinalities over the surviving prefix
//!   (`surviving_identity_count`, `surviving_number_chunk_count`).
//! * `_build_instream_columns` (`_remap_inputs.py`): the per-node in-stream
//!   identity slot CSR — gather each node's surviving IDENTITY-band ids, DROP
//!   the first per node (the prepend slot), and map each remaining id to a
//!   FUNCTION slot (`func_slot_lut`) or a COUNTER slot (`counter_slot_lut`)
//!   indexed by `shifted_id - identity_lo`. Emits `instream_off` (CSR),
//!   `instream_func_slot`, `instream_counter_slot`.
//! * `build_number_chunk_columns` (`_number_chunk_columns.py`): the per-chunk
//!   `out_block = expanded - number_lo` and `ct_ordinal = node_id`, in
//!   ascending (DFS-then-stream) position order. The per-(block, node) slice
//!   gather + the per-row variant CSR stay in Python (a different concern).
//!
//! It re-implements NO band layout (the bands + slot LUTs are kernel params,
//! so a vocab-layout change reshapes both Python and Rust) and NO gather /
//! variant-CSR logic — it only RE-EXPRESSES the shared preamble once, so the
//! three thin Python adapters cannot diverge from each other.
//!
//! ## Per-node (`e`, window `[node_offsets[e], node_offsets[e + 1])`)
//!
//! For each position `p` in the window with `offset = p - node_offsets[e]`:
//!   * counts iff `offset < surviving[e]` (the surviving clip, identical to
//!     numpy's `offset_in_node < surviving[node_id]`).
//!   * IDENTITY-band (`identity_lo <= expanded[p] < identity_hi`) survivors
//!     bump `surviving_identity_count[e]`; all but the FIRST per node become
//!     in-stream slots (the prepend drop).
//!   * NUMBER-band (`number_lo <= expanded[p] < number_hi`) survivors bump
//!     `surviving_number_chunk_count[e]` and emit one `(out_block, e)` chunk.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Everything the three call sites slice out of the single segmented pass.
#[cfg_attr(test, derive(Debug, PartialEq))]
struct FamilyBandReductionOut {
    /// `count_surviving_batched` — per-node IDENTITY-band cardinality.
    surviving_identity_count: Vec<i64>,
    /// `count_surviving_batched` — per-node NUMBER-band cardinality.
    surviving_number_chunk_count: Vec<i64>,
    /// `_build_instream_columns` — per-node in-stream slot CSR (`n_nodes + 1`).
    instream_off: Vec<i64>,
    /// `_build_instream_columns` — in-stream FUNCTION slot (COUNTER `-1`).
    instream_func_slot: Vec<i64>,
    /// `_build_instream_columns` — in-stream COUNTER slot (FUNCTION `-1`).
    instream_counter_slot: Vec<i64>,
    /// `build_number_chunk_columns` — per-chunk `expanded - number_lo`.
    number_out_block: Vec<i64>,
    /// `build_number_chunk_columns` — per-chunk owning node id (`ct_ordinal`).
    number_ct_ordinal: Vec<i64>,
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it directly.
/// One pass over `expanded`, mirroring the three numpy preamble twins.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    expanded: &[u16],
    node_offsets: &[i64],
    surviving: &[i64],
    number_lo: u16,
    number_hi: u16,
    identity_lo: u16,
    identity_hi: u16,
    func_slot_lut: &[i64],
    counter_slot_lut: &[i64],
) -> Result<FamilyBandReductionOut, String> {
    if node_offsets.is_empty() {
        return Err("node_offsets must have length n_nodes + 1 (>= 1)".into());
    }
    let n_nodes = node_offsets.len() - 1;
    if surviving.len() != n_nodes {
        return Err(format!(
            "surviving disagrees on n_nodes ({n_nodes}): got {}",
            surviving.len()
        ));
    }
    let band = (identity_hi - identity_lo) as usize;
    if func_slot_lut.len() != band || counter_slot_lut.len() != band {
        return Err(format!(
            "slot LUTs must have length identity_hi - identity_lo ({band}): \
             func {} counter {}",
            func_slot_lut.len(),
            counter_slot_lut.len(),
        ));
    }
    let total = expanded.len();

    let mut out = FamilyBandReductionOut {
        surviving_identity_count: vec![0i64; n_nodes],
        surviving_number_chunk_count: vec![0i64; n_nodes],
        instream_off: vec![0i64; n_nodes + 1],
        instream_func_slot: Vec::new(),
        instream_counter_slot: Vec::new(),
        number_out_block: Vec::new(),
        number_ct_ordinal: Vec::new(),
    };

    let mut instream_running: i64 = 0;
    for e in 0..n_nodes {
        let start = node_offsets[e];
        let end = node_offsets[e + 1];
        if start < 0 || end < start {
            return Err(format!(
                "node {e} window [start={start}, end={end}) is malformed"
            ));
        }
        let end_u = end as usize;
        if end_u > total {
            return Err(format!(
                "node {e} window end {end_u} exceeds expanded length {total}"
            ));
        }
        // The surviving clip: only the first `surviving[e]` positions of the
        // node's window count (numpy: offset_in_node < surviving[node_id]).
        // Negative surviving lengths clip to zero (no position included).
        let surv = surviving[e].max(0) as usize;
        let start_u = start as usize;
        let window_len = end_u - start_u;
        let limit = start_u + surv.min(window_len);

        let mut seen_identity_in_node = false;
        for p in start_u..limit {
            let id = expanded[p];
            if id >= identity_lo && id < identity_hi {
                out.surviving_identity_count[e] += 1;
                if seen_identity_in_node {
                    // In-stream slot: drop the FIRST identity per node (the
                    // prepend), map the rest via the band-indexed LUTs.
                    let k = (id - identity_lo) as usize;
                    out.instream_func_slot.push(func_slot_lut[k]);
                    out.instream_counter_slot.push(counter_slot_lut[k]);
                    instream_running += 1;
                } else {
                    seen_identity_in_node = true;
                }
            } else if id >= number_lo && id < number_hi {
                out.surviving_number_chunk_count[e] += 1;
                out.number_out_block.push((id - number_lo) as i64);
                out.number_ct_ordinal.push(e as i64);
            }
        }
        out.instream_off[e + 1] = instream_running;
    }

    Ok(out)
}

/// PyO3 wrapper: borrow the expanded stream + per-node CSR + surviving prefix
/// + band constants + slot LUTs, run the single segmented pass under
/// `py.detach` (GIL-free), and return the seven family outputs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_family_band_reduction_kernel<'py>(
    py: Python<'py>,
    expanded: numpy::PyReadonlyArray1<'py, u16>,
    node_offsets: numpy::PyReadonlyArray1<'py, i64>,
    surviving: numpy::PyReadonlyArray1<'py, i64>,
    number_lo: u16,
    number_hi: u16,
    identity_lo: u16,
    identity_hi: u16,
    func_slot_lut: numpy::PyReadonlyArray1<'py, i64>,
    counter_slot_lut: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let expanded = expanded.as_slice()?;
    let node_offsets = node_offsets.as_slice()?;
    let surviving = surviving.as_slice()?;
    let func_slot_lut = func_slot_lut.as_slice()?;
    let counter_slot_lut = counter_slot_lut.as_slice()?;

    let out = py
        .detach(|| {
            run_kernel(
                expanded,
                node_offsets,
                surviving,
                number_lo,
                number_hi,
                identity_lo,
                identity_hi,
                func_slot_lut,
                counter_slot_lut,
            )
        })
        .map_err(PyValueError::new_err)?;

    let arrays: [Bound<'py, PyAny>; 7] = [
        PyArray1::from_vec(py, out.surviving_identity_count).into_any(),
        PyArray1::from_vec(py, out.surviving_number_chunk_count).into_any(),
        PyArray1::from_vec(py, out.instream_off).into_any(),
        PyArray1::from_vec(py, out.instream_func_slot).into_any(),
        PyArray1::from_vec(py, out.instream_counter_slot).into_any(),
        PyArray1::from_vec(py, out.number_out_block).into_any(),
        PyArray1::from_vec(py, out.number_ct_ordinal).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bands mirror the shifted vocab layout: NUMBER [1, 8), IDENTITY [8, 16).
    const NLO: u16 = 1;
    const NHI: u16 = 8;
    const ILO: u16 = 8;
    const IHI: u16 = 16;

    // Slot LUTs indexed by (shifted_id - ILO). For the tests: identity ids
    // 8 / 9 / 10 are FUNCTION slots 0 / 1 / 2; id 13 is a COUNTER slot 5.
    fn func_lut() -> Vec<i64> {
        vec![0, 1, 2, -1, -1, -1, -1, -1]
    }
    fn counter_lut() -> Vec<i64> {
        vec![-1, -1, -1, -1, -1, 5, -1, -1]
    }

    fn run(
        expanded: &[u16],
        node_offsets: &[i64],
        surviving: &[i64],
    ) -> FamilyBandReductionOut {
        run_kernel(
            expanded,
            node_offsets,
            surviving,
            NLO,
            NHI,
            ILO,
            IHI,
            &func_lut(),
            &counter_lut(),
        )
        .unwrap()
    }

    #[test]
    fn empty_input_no_nodes() {
        let out = run(&[], &[0], &[]);
        assert!(out.surviving_identity_count.is_empty());
        assert!(out.surviving_number_chunk_count.is_empty());
        assert_eq!(out.instream_off, vec![0]);
        assert!(out.instream_func_slot.is_empty());
        assert!(out.number_out_block.is_empty());
    }

    #[test]
    fn single_node_counts_drop_first_and_chunks() {
        // node0: [8 (id prepend), 2 (num), 9 (id), 13 (id)] all surviving.
        // identity count = 3; first (8) dropped; in-stream = [9, 13].
        // number count = 1; out_block = 2 - 1 = 1; ct_ordinal = 0.
        let out = run(&[8, 2, 9, 13], &[0, 4], &[4]);
        assert_eq!(out.surviving_identity_count, vec![3]);
        assert_eq!(out.surviving_number_chunk_count, vec![1]);
        assert_eq!(out.instream_off, vec![0, 2]);
        assert_eq!(out.instream_func_slot, vec![1, -1]); // 9->func1, 13->none
        assert_eq!(out.instream_counter_slot, vec![-1, 5]); // 9->none, 13->cnt5
        assert_eq!(out.number_out_block, vec![1]);
        assert_eq!(out.number_ct_ordinal, vec![0]);
    }

    #[test]
    fn surviving_clip_excludes_tail() {
        // node0 window has 4 slots but only 2 survive: [8, 9 | 10, 13].
        // Only [8, 9] count: identity = 2, drop first (8), in-stream = [9].
        let out = run(&[8, 9, 10, 13], &[0, 4], &[2]);
        assert_eq!(out.surviving_identity_count, vec![2]);
        assert_eq!(out.instream_off, vec![0, 1]);
        assert_eq!(out.instream_func_slot, vec![1]); // only 9
    }

    #[test]
    fn multi_node_independent_first_of_node_drop() {
        // node0: [8, 9] -> id=2, drop 8, in-stream [9].
        // node1: [10, 13] -> id=2, drop 10, in-stream [13].
        let out = run(&[8, 9, 10, 13], &[0, 2, 4], &[2, 2]);
        assert_eq!(out.surviving_identity_count, vec![2, 2]);
        assert_eq!(out.instream_off, vec![0, 1, 2]);
        assert_eq!(out.instream_func_slot, vec![1, -1]); // 9->func1, 13->none
        assert_eq!(out.instream_counter_slot, vec![-1, 5]); // 9->none, 13->cnt5
    }

    #[test]
    fn all_dropped_node_contributes_nothing() {
        // node0 surviving=0 -> no positions count; node1 normal.
        let out = run(&[8, 9, 8, 9], &[0, 2, 4], &[0, 2]);
        assert_eq!(out.surviving_identity_count, vec![0, 2]);
        assert_eq!(out.instream_off, vec![0, 0, 1]); // node0 empty, node1 1
        assert_eq!(out.instream_func_slot, vec![1]);
    }

    #[test]
    fn node_with_single_identity_has_zero_instream() {
        // node0: [8] surviving -> identity count 1, but it IS the prepend,
        // so in-stream is empty (drop-first edge).
        let out = run(&[8], &[0, 1], &[1]);
        assert_eq!(out.surviving_identity_count, vec![1]);
        assert_eq!(out.instream_off, vec![0, 0]);
        assert!(out.instream_func_slot.is_empty());
    }

    #[test]
    fn number_band_boundaries_left_closed_right_open() {
        // id 1 in number band, id 8 NOT (it's identity_lo), id 0 in neither.
        let out = run(&[0, 1, 7, 8], &[0, 4], &[4]);
        // number: ids 1 and 7 -> count 2; out_block 0 and 6.
        assert_eq!(out.surviving_number_chunk_count, vec![2]);
        assert_eq!(out.number_out_block, vec![0, 6]);
        // identity: id 8 only -> count 1 (it is the prepend, no in-stream).
        assert_eq!(out.surviving_identity_count, vec![1]);
        assert_eq!(out.instream_off, vec![0, 0]);
    }
}
