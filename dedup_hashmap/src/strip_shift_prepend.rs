//! `build_strip_shift_prepend_kernel` — batched strip + shift + prepend.
//!
//! Single concern: given the per-node raw working stream (`working`, with
//! the VC2/F128 continuation slots already PAINTED upstream), the per-
//! position VC2/F128 painted masks (`extra_vc2_raw`, `extra_f128_raw`), the
//! per-node CSR (`rec_starts` + `counts`), and each node's already-shifted
//! self-token (`self_token_ids`), produce the EXPANDED stream the vector
//! path's `batched_expand` emits: drop every `<= reserved_digit_count` slot
//! (the inline-digit band + sign marker), shift each surviving id DOWN by
//! `reserved_digit_count`, and prepend each node's self-token at its slot 0.
//! This is the GIL-released twin of numpy `_strip_shift_prepend`
//! (`_rewrite.py`), mirroring scalar `expand_tokens` steps 3 + 4 batched.
//!
//! Returns `(expanded, node_offsets, extra_value_v2_mask, extra_f128_mask)`
//! over the EXPANDED stream (slot 0 per node = the prepended self-token,
//! both masks False there). It re-implements NO decode rule; it only
//! KEEP-filters + shifts + scatters the painted working stream the same way
//! the numpy twin does, preserving the #92 per-node-length discipline (each
//! node's body length is the count of surviving slots in its raw window, so
//! consecutive empty bodies never merge).
//!
//! ## Per-node (`e`, raw window `[rec_starts[e], rec_starts[e] + counts[e])`)
//!
//! * `body_len[e]` = number of raw slots `> reserved_digit_count` in window.
//! * `own_length[e]` = `body_len[e] + 1`; `node_offsets` = exclusive cumsum.
//! * slot 0 of node `e` = `self_token_ids[e]` (masks False).
//! * each kept raw slot `p` (in window order, rank `r` among the node's
//!   survivors) lands at `node_offsets[e] + 1 + r`:
//!     - `expanded[dst] = working[p] - reserved_digit_count`
//!     - `extra_value_v2_mask[dst] = extra_vc2_raw[p]`
//!     - `extra_f128_mask[dst]    = extra_f128_raw[p]`

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The four expanded-stream outputs in `_strip_shift_prepend` return order.
#[cfg_attr(test, derive(Debug))]
struct StripShiftPrependOut {
    expanded: Vec<u16>,
    node_offsets: Vec<i64>,
    extra_value_v2_mask: Vec<bool>,
    extra_f128_mask: Vec<bool>,
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it
/// directly. Mirrors numpy `_strip_shift_prepend`.
fn run_kernel(
    working: &[u16],
    extra_vc2_raw: &[bool],
    extra_f128_raw: &[bool],
    rec_starts: &[i64],
    counts: &[i64],
    self_token_ids: &[u16],
    reserved_digit_count: u16,
) -> Result<StripShiftPrependOut, String> {
    let n_nodes = rec_starts.len();
    if counts.len() != n_nodes || self_token_ids.len() != n_nodes {
        return Err(format!(
            "per-node arrays disagree on n_nodes ({n_nodes}): rec_starts {} \
             counts {} self_token_ids {}",
            rec_starts.len(),
            counts.len(),
            self_token_ids.len(),
        ));
    }
    let total = working.len();
    if extra_vc2_raw.len() != total || extra_f128_raw.len() != total {
        return Err(format!(
            "painted masks disagree with working length {total}: \
             extra_vc2_raw {} extra_f128_raw {}",
            extra_vc2_raw.len(),
            extra_f128_raw.len(),
        ));
    }

    // First pass: per-node body length (count of kept slots) -> node_offsets.
    let mut node_offsets = vec![0i64; n_nodes + 1];
    let mut running: i64 = 0;
    for e in 0..n_nodes {
        let start = rec_starts[e];
        let cnt = counts[e];
        if start < 0 || cnt < 0 {
            return Err(format!(
                "node {e} window [start={start}, count={cnt}) has a negative \
                 bound"
            ));
        }
        let start_u = start as usize;
        let end_u = start_u + cnt as usize;
        if end_u > total {
            return Err(format!(
                "node {e} window end {end_u} exceeds working length {total}"
            ));
        }
        let mut body_len: i64 = 0;
        for p in start_u..end_u {
            if working[p] > reserved_digit_count {
                body_len += 1;
            }
        }
        running += body_len + 1; // + the prepended self-token
        node_offsets[e + 1] = running;
    }

    let total_expanded = running as usize;
    let mut out = StripShiftPrependOut {
        expanded: vec![0u16; total_expanded],
        node_offsets,
        extra_value_v2_mask: vec![false; total_expanded],
        extra_f128_mask: vec![false; total_expanded],
    };

    // Second pass: prepend self-token + scatter the shifted survivors.
    for e in 0..n_nodes {
        let base = out.node_offsets[e] as usize;
        out.expanded[base] = self_token_ids[e];
        let start_u = rec_starts[e] as usize;
        let end_u = start_u + counts[e] as usize;
        let mut rank: usize = 0;
        for p in start_u..end_u {
            let w = working[p];
            if w > reserved_digit_count {
                let dst = base + 1 + rank;
                out.expanded[dst] = w - reserved_digit_count;
                out.extra_value_v2_mask[dst] = extra_vc2_raw[p];
                out.extra_f128_mask[dst] = extra_f128_raw[p];
                rank += 1;
            }
        }
    }

    Ok(out)
}

/// PyO3 wrapper: borrow the painted working stream + masks + per-node CSR +
/// self-tokens, run the strip/shift/prepend under `py.detach`, and return
/// the four expanded-stream arrays in `_strip_shift_prepend` order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_strip_shift_prepend_kernel<'py>(
    py: Python<'py>,
    working: numpy::PyReadonlyArray1<'py, u16>,
    extra_vc2_raw: numpy::PyReadonlyArray1<'py, bool>,
    extra_f128_raw: numpy::PyReadonlyArray1<'py, bool>,
    rec_starts: numpy::PyReadonlyArray1<'py, i64>,
    counts: numpy::PyReadonlyArray1<'py, i64>,
    self_token_ids: numpy::PyReadonlyArray1<'py, u16>,
    reserved_digit_count: u16,
) -> PyResult<Bound<'py, PyTuple>> {
    let working = working.as_slice()?;
    let extra_vc2_raw = extra_vc2_raw.as_slice()?;
    let extra_f128_raw = extra_f128_raw.as_slice()?;
    let rec_starts = rec_starts.as_slice()?;
    let counts = counts.as_slice()?;
    let self_token_ids = self_token_ids.as_slice()?;

    let out = py
        .detach(|| {
            run_kernel(
                working,
                extra_vc2_raw,
                extra_f128_raw,
                rec_starts,
                counts,
                self_token_ids,
                reserved_digit_count,
            )
        })
        .map_err(PyValueError::new_err)?;

    let arrays: [Bound<'py, PyAny>; 4] = [
        PyArray1::from_vec(py, out.expanded).into_any(),
        PyArray1::from_vec(py, out.node_offsets).into_any(),
        PyArray1::from_vec(py, out.extra_value_v2_mask).into_any(),
        PyArray1::from_vec(py, out.extra_f128_mask).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESERVED: u16 = 256;

    #[test]
    fn drops_band_shifts_and_prepends() {
        // node0 raw: [300, 100, 400] (100 <= 256 dropped); self = 5.
        // body = [300-256, 400-256] = [44, 144]; expanded = [5, 44, 144].
        let out = run_kernel(
            &[300, 100, 400],
            &[false, false, true],
            &[false, false, false],
            &[0],
            &[3],
            &[5],
            RESERVED,
        )
        .unwrap();
        assert_eq!(out.expanded, vec![5, 44, 144]);
        assert_eq!(out.node_offsets, vec![0, 3]);
        // vc2 painted only at raw 400 (slot 2 of expanded body -> dst 2).
        assert_eq!(out.extra_value_v2_mask, vec![false, false, true]);
        assert_eq!(out.extra_f128_mask, vec![false, false, false]);
    }

    #[test]
    fn empty_body_node_still_gets_self_token() {
        // node with ALL slots <= reserved -> body empty, only self-token.
        let out = run_kernel(
            &[10, 20],
            &[false, false],
            &[false, false],
            &[0],
            &[2],
            &[7],
            RESERVED,
        )
        .unwrap();
        assert_eq!(out.expanded, vec![7]);
        assert_eq!(out.node_offsets, vec![0, 1]);
    }

    #[test]
    fn consecutive_empty_bodies_do_not_merge() {
        // #92: two adjacent empty-body nodes must NOT collapse their slot 0.
        // node0 empty, node1 has [257]->1, node2 empty.
        let out = run_kernel(
            &[10, 257, 20],
            &[false, false, false],
            &[false, false, false],
            &[0, 1, 2],
            &[1, 1, 1],
            &[100, 200, 300],
            RESERVED,
        )
        .unwrap();
        // node0: [100]; node1: [200, 1]; node2: [300].
        assert_eq!(out.expanded, vec![100, 200, 1, 300]);
        assert_eq!(out.node_offsets, vec![0, 1, 3, 4]);
    }

    #[test]
    fn multi_node_masks_land_at_right_expanded_slot() {
        // node0 raw [257, 258] (both kept) vc2 at 257, f128 at 258.
        // node1 raw [100, 259] (100 dropped) -> body [3], no paint.
        let out = run_kernel(
            &[257, 258, 100, 259],
            &[true, false, false, false],
            &[false, true, false, false],
            &[0, 2],
            &[2, 2],
            &[9, 8],
            RESERVED,
        )
        .unwrap();
        // node0: [9, 1, 2]; node1: [8, 3].
        assert_eq!(out.expanded, vec![9, 1, 2, 8, 3]);
        assert_eq!(out.node_offsets, vec![0, 3, 5]);
        // vc2 at node0 body slot 0 (dst 1); f128 at node0 body slot 1 (dst 2).
        assert_eq!(
            out.extra_value_v2_mask,
            vec![false, true, false, false, false]
        );
        assert_eq!(
            out.extra_f128_mask,
            vec![false, false, true, false, false]
        );
    }

    #[test]
    fn empty_input_no_nodes() {
        let out = run_kernel(&[], &[], &[], &[], &[], &[], RESERVED).unwrap();
        assert!(out.expanded.is_empty());
        assert_eq!(out.node_offsets, vec![0]);
    }

    #[test]
    fn adversarial_window_overruns_working_errors() {
        let err = run_kernel(
            &[300],
            &[false],
            &[false],
            &[0],
            &[5], // claims 5 slots but working has 1
            &[1],
            RESERVED,
        )
        .unwrap_err();
        assert!(err.contains("exceeds working length"), "got: {err}");
    }

    #[test]
    fn adversarial_shift_at_threshold_boundary() {
        // working == reserved exactly -> DROPPED (strictly >); reserved+1 ->
        // kept and shifts to 1. Guards the off-by-one at the band edge.
        let out = run_kernel(
            &[256, 257],
            &[false, false],
            &[false, false],
            &[0],
            &[2],
            &[0],
            RESERVED,
        )
        .unwrap();
        assert_eq!(out.expanded, vec![0, 1]); // 256 dropped, 257->1
    }
}
