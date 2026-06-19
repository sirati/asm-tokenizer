//! `build_promote_batched_kernel` — batched VC2 / F128 continuation paint.
//!
//! Single concern: given the flat raw working stream (`working`), the
//! per-position real-token mask (`real_mask`), the inline-digit run-length
//! field (`runlen_number`), the per-position owning-node map (`node_of`),
//! and the per-node CSR (`rec_starts` + `counts`), PAINT the VC2 / F128
//! continuation slots over a fresh copy of `working` and report the two
//! per-position painted masks. This is the GIL-released twin of numpy
//! `_promote_batched` (`_rewrite.py`), mirroring scalar `_promote_vc2` /
//! `_promote_f128` batched.
//!
//! Returns `(working_painted, extra_vc2_raw, extra_f128_raw)` over the raw
//! index space (`working_painted` = the input with the VC2/F128 continuation
//! slots overwritten, the two masks True at the painted slots). It paints the
//! SAME slots the numpy twin does and never crosses a node boundary: the
//! bounds check uses the per-node tail `rec_starts[e] + counts[e]`.
//!
//! ## VC2 promotion (per carrier `p` with `real_mask[p] && working[p] == vc2`)
//!
//! * the carrier needs a `p+1` payload slot inside its node:
//!   `local = p - rec_starts[node] < counts[node] - 1`, else MALFORMED.
//! * `payload_len = runlen_number[p+1]`; `chunk_counts = max(1, (len+7)/8)`.
//! * declared chunks must fit the node tail: `p + chunk_counts <= node_tail`,
//!   else MALFORMED.
//! * paint slots `[p+1, p+chunk_counts)` (that is `chunk_counts - 1` slots)
//!   to `vc2`, set their `extra_vc2_raw` True.
//!
//! ## F128 promotion (per carrier `p` with `real_mask[p] && working[p] == f128`)
//!
//! * ALG-2 needs the high u16 at `p+1, p+2`: `local < counts[node] - 2`, else
//!   MALFORMED.
//! * `high = working[p+1] << 8`, `low = working[p+2]`;
//!   `is_nan_or_inf = ((high | low) & 0x7fff) == 0x7fff`.
//! * for FINITE carriers paint slot `p+1` to `f128`, set `extra_f128_raw`.

use numpy::PyArray1;
use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The three raw-stream outputs in `_promote_batched` return order plus the
/// painted working stream the orchestrator hands straight to the next kernel.
#[cfg_attr(test, derive(Debug))]
struct PromoteBatchedOut {
    working: Vec<u16>,
    extra_vc2_raw: Vec<bool>,
    extra_f128_raw: Vec<bool>,
}

/// Either a structural-input error (mirrors strip_shift_prepend's
/// `PyValueError`) or a malformed-v2-stream guard (mirrors the numpy twin's
/// `AssertionError`); the wrapper maps each to the matching Python exception.
#[cfg_attr(test, derive(Debug))]
enum PromoteError {
    Structural(String),
    Malformed(String),
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it directly.
/// Mirrors numpy `_promote_batched`.
fn run_kernel(
    working: &[u16],
    real_mask: &[bool],
    runlen_number: &[u16],
    node_of: &[i64],
    rec_starts: &[i64],
    counts: &[i64],
    vc2_vocab_id: u16,
    f128_vocab_id: u16,
) -> Result<PromoteBatchedOut, PromoteError> {
    let total = working.len();
    let n_nodes = rec_starts.len();
    if real_mask.len() != total
        || runlen_number.len() != total
        || node_of.len() != total
    {
        return Err(PromoteError::Structural(format!(
            "per-position arrays disagree with working length {total}: \
             real_mask {} runlen_number {} node_of {}",
            real_mask.len(),
            runlen_number.len(),
            node_of.len(),
        )));
    }
    if counts.len() != n_nodes {
        return Err(PromoteError::Structural(format!(
            "per-node arrays disagree on n_nodes ({n_nodes}): rec_starts {} \
             counts {}",
            rec_starts.len(),
            counts.len(),
        )));
    }

    let mut out = PromoteBatchedOut {
        working: working.to_vec(),
        extra_vc2_raw: vec![false; total],
        extra_f128_raw: vec![false; total],
    };
    if total == 0 {
        return Ok(out);
    }

    // --- VC2 promotion ---------------------------------------------------
    for p in 0..total {
        if !(real_mask[p] && working[p] == vc2_vocab_id) {
            continue;
        }
        let e = node_of[p] as usize;
        let local = p as i64 - rec_starts[e];
        if local >= counts[e] - 1 {
            return Err(PromoteError::Malformed(
                "VC2 carrier at the last raw-stream position -- malformed \
                 v2 stream (carrier needs a p+1 slot for the payload \
                 inline-digit run)."
                    .to_string(),
            ));
        }
        // payload_len read at p+1 (guaranteed in-node by the guard above);
        // widen u16 -> i64 before the ceil-div so the arithmetic never wraps.
        let payload_len = runlen_number[p + 1] as i64;
        let chunk_counts = ((payload_len + 7) / 8).max(1);
        let node_tail = rec_starts[e] + counts[e];
        let end = p as i64 + chunk_counts;
        if end > node_tail {
            return Err(PromoteError::Malformed(format!(
                "VC2 carrier at position {p} declares {chunk_counts} chunks \
                 but only {} raw-stream slots remain -- malformed v2 stream.",
                node_tail - p as i64,
            )));
        }
        // Paint the chunk_counts - 1 continuation slots: [p+1, p+chunk_counts).
        for slot in (p + 1)..(p + chunk_counts as usize) {
            out.working[slot] = vc2_vocab_id;
            out.extra_vc2_raw[slot] = true;
        }
    }

    // --- F128 promotion --------------------------------------------------
    // Detect carriers against the INPUT `working` (numpy freezes the carrier
    // set with `np.nonzero` BEFORE any paint), so an F128 carrier's own
    // painted continuation slot can never be re-detected as a new carrier.
    // The carrier test is byte-equivalent to reading the VC2-painted buffer
    // (VC2 paint writes only `vc2_vocab_id`, never `f128_vocab_id`), but the
    // input read avoids the self-re-detection the mutated buffer would cause.
    // The high/low byte READS below stay on `out.working` (post-VC2), matching
    // numpy which evaluates the NaN/Inf test against the VC2-painted stream.
    for p in 0..total {
        if !(real_mask[p] && working[p] == f128_vocab_id) {
            continue;
        }
        let e = node_of[p] as usize;
        let local = p as i64 - rec_starts[e];
        if local >= counts[e] - 2 {
            return Err(PromoteError::Malformed(
                "F128 carrier within 2 positions of the raw-stream tail -- \
                 malformed v2 stream (ALG-2 needs the high u16 of the \
                 binary128 payload at p+1, p+2)."
                    .to_string(),
            ));
        }
        let high = out.working[p + 1] << 8;
        let low = out.working[p + 2];
        let is_nan_or_inf = ((high | low) & 0x7fff) == 0x7fff;
        if !is_nan_or_inf {
            out.working[p + 1] = f128_vocab_id;
            out.extra_f128_raw[p + 1] = true;
        }
    }

    Ok(out)
}

/// PyO3 wrapper: borrow the working stream + per-position state + per-node
/// CSR, run the promotion paint under `py.detach`, and return the painted
/// working stream + the two raw-space masks in `_promote_batched` order
/// (working first, so the orchestrator hands it straight to the next kernel).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_promote_batched_kernel<'py>(
    py: Python<'py>,
    working: numpy::PyReadonlyArray1<'py, u16>,
    real_mask: numpy::PyReadonlyArray1<'py, bool>,
    runlen_number: numpy::PyReadonlyArray1<'py, u16>,
    node_of: numpy::PyReadonlyArray1<'py, i64>,
    rec_starts: numpy::PyReadonlyArray1<'py, i64>,
    counts: numpy::PyReadonlyArray1<'py, i64>,
    vc2_vocab_id: u16,
    f128_vocab_id: u16,
) -> PyResult<Bound<'py, PyTuple>> {
    let working = working.as_slice()?;
    let real_mask = real_mask.as_slice()?;
    let runlen_number = runlen_number.as_slice()?;
    let node_of = node_of.as_slice()?;
    let rec_starts = rec_starts.as_slice()?;
    let counts = counts.as_slice()?;

    let out = py
        .detach(|| {
            run_kernel(
                working,
                real_mask,
                runlen_number,
                node_of,
                rec_starts,
                counts,
                vc2_vocab_id,
                f128_vocab_id,
            )
        })
        .map_err(|e| match e {
            PromoteError::Malformed(m) => PyAssertionError::new_err(m),
            PromoteError::Structural(m) => PyValueError::new_err(m),
        })?;

    let arrays: [Bound<'py, PyAny>; 3] = [
        PyArray1::from_vec(py, out.working).into_any(),
        PyArray1::from_vec(py, out.extra_vc2_raw).into_any(),
        PyArray1::from_vec(py, out.extra_f128_raw).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VC2: u16 = 257;
    const F128: u16 = 263;

    /// Build a single-node `node_of` (all positions owned by node 0).
    fn single_node(total: usize) -> Vec<i64> {
        vec![0i64; total]
    }

    #[test]
    fn no_carrier_leaves_stream_untouched() {
        let working = [264u16, 265, 266];
        let out = run_kernel(
            &working,
            &[true, true, true],
            &[0, 0, 0],
            &single_node(3),
            &[0],
            &[3],
            VC2,
            F128,
        )
        .unwrap();
        assert_eq!(out.working, working);
        assert_eq!(out.extra_vc2_raw, vec![false; 3]);
        assert_eq!(out.extra_f128_raw, vec![false; 3]);
    }

    #[test]
    fn vc2_single_chunk_paints_nothing() {
        // payload_len = 1 -> chunk_counts = max(1, (1+7)/8) = 1 -> 0 painted.
        let working = [VC2, 264u16];
        let out = run_kernel(
            &working,
            &[true, false],
            &[0, 1], // runlen_number[p+1] = 1
            &single_node(2),
            &[0],
            &[2],
            VC2,
            F128,
        )
        .unwrap();
        assert_eq!(out.working, working);
        assert_eq!(out.extra_vc2_raw, vec![false, false]);
    }

    #[test]
    fn vc2_multi_chunk_paints_continuation_slots() {
        // payload_len = 10 -> chunk_counts = max(1, (10+7)/8) = 2 -> 1 painted.
        // carrier at 0, runlen_number[1] = 10; paint slot 1.
        let working = [VC2, 5u16, 6u16];
        let out = run_kernel(
            &working,
            &[true, false, false],
            &[0, 10, 9],
            &single_node(3),
            &[0],
            &[3],
            VC2,
            F128,
        )
        .unwrap();
        assert_eq!(out.working, vec![VC2, VC2, 6]);
        assert_eq!(out.extra_vc2_raw, vec![false, true, false]);
        assert_eq!(out.extra_f128_raw, vec![false; 3]);
    }

    #[test]
    fn f128_finite_paints_p_plus_one() {
        // high u16 = 0x3fff (finite); paint slot 1.
        // high byte 0x3f at p+1, low byte 0xff at p+2.
        let working = [F128, 0x3fu16, 0xffu16, 0u16];
        let out = run_kernel(
            &working,
            &[true, false, false, false],
            &[0, 0, 0, 0],
            &single_node(4),
            &[0],
            &[4],
            VC2,
            F128,
        )
        .unwrap();
        assert_eq!(out.working, vec![F128, F128, 0xff, 0]);
        assert_eq!(out.extra_f128_raw, vec![false, true, false, false]);
        assert_eq!(out.extra_vc2_raw, vec![false; 4]);
    }

    #[test]
    fn f128_painted_slot_is_not_re_detected_as_carrier() {
        // Regression: the F128 loop must detect carriers against the INPUT
        // `working`, not the mutated `out.working`. A finite F128 carrier at
        // p=0 paints slot 1 to F128; if detection read the mutated buffer it
        // would re-detect slot 1 as a new carrier and either spuriously paint
        // slot 2 or false-raise the tail guard. numpy freezes the carrier set
        // before painting, so only slot 1 is ever painted. Here slot 1 carries
        // `real_mask=true` (the trigger) -- a single-carrier 5-slot node, high
        // u16 = 0x3fff (finite).
        let working = [F128, 0x3fu16, 0xffu16, 0u16, 0u16];
        let out = run_kernel(
            &working,
            &[true, true, false, false, false],
            &[0, 0, 0, 0, 0],
            &single_node(5),
            &[0],
            &[5],
            VC2,
            F128,
        )
        .unwrap();
        // ONLY slot 1 painted (the original carrier's continuation); slot 2
        // stays its original byte, never re-painted.
        assert_eq!(out.working, vec![F128, F128, 0xff, 0, 0]);
        assert_eq!(out.extra_f128_raw, vec![false, true, false, false, false]);
        assert_eq!(out.extra_vc2_raw, vec![false; 5]);
    }

    #[test]
    fn f128_nan_or_inf_paints_nothing() {
        // high u16 = 0x7fff (NaN/Inf): high byte 0x7f, low byte 0xff.
        let working = [F128, 0x7fu16, 0xffu16, 0u16];
        let out = run_kernel(
            &working,
            &[true, false, false, false],
            &[0, 0, 0, 0],
            &single_node(4),
            &[0],
            &[4],
            VC2,
            F128,
        )
        .unwrap();
        assert_eq!(out.working, working);
        assert_eq!(out.extra_f128_raw, vec![false; 4]);
    }

    #[test]
    fn vc2_carrier_at_last_position_is_malformed() {
        let err = run_kernel(
            &[264u16, VC2],
            &[true, true],
            &[0, 0],
            &single_node(2),
            &[0],
            &[2],
            VC2,
            F128,
        )
        .unwrap_err();
        match err {
            PromoteError::Malformed(m) => {
                assert!(m.contains("last raw-stream position"), "got: {m}")
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn vc2_declared_chunks_exceed_node_tail_is_malformed() {
        // carrier at 0, payload_len = 16 -> chunk_counts = (16+7)/8 = 2;
        // end = 0 + 2 = 2 == node_tail -> OK at 2 slots. Force overrun with
        // payload_len = 24 -> chunk_counts = (24+7)/8 = 3; end = 3 > tail 2.
        let err = run_kernel(
            &[VC2, 24u16],
            &[true, false],
            &[0, 24],
            &single_node(2),
            &[0],
            &[2],
            VC2,
            F128,
        )
        .unwrap_err();
        match err {
            PromoteError::Malformed(m) => {
                assert!(m.contains("declares"), "got: {m}")
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn f128_within_two_of_tail_is_malformed() {
        // carrier at position 1 of a 2-slot node -> local 1 >= counts 2 - 2.
        let err = run_kernel(
            &[264u16, F128],
            &[true, true],
            &[0, 0],
            &single_node(2),
            &[0],
            &[2],
            VC2,
            F128,
        )
        .unwrap_err();
        match err {
            PromoteError::Malformed(m) => {
                assert!(m.contains("within 2 positions"), "got: {m}")
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn multi_node_paint_stays_within_node_boundary() {
        // node0: [VC2, x, y] payload_len 10 -> 1 painted slot at 1.
        // node1: [VC2, x, y] payload_len 10 -> 1 painted slot at 4.
        // Each node's tail bounds its own paint; no cross-node leak.
        let working = [VC2, 5u16, 6u16, VC2, 7u16, 8u16];
        let node_of = vec![0i64, 0, 0, 1, 1, 1];
        let out = run_kernel(
            &working,
            &[true, false, false, true, false, false],
            &[0, 10, 9, 0, 10, 9],
            &node_of,
            &[0, 3],
            &[3, 3],
            VC2,
            F128,
        )
        .unwrap();
        assert_eq!(out.working, vec![VC2, VC2, 6, VC2, VC2, 8]);
        assert_eq!(
            out.extra_vc2_raw,
            vec![false, true, false, false, true, false]
        );
    }

    #[test]
    fn empty_input_no_positions() {
        let out = run_kernel(&[], &[], &[], &[], &[], &[], VC2, F128).unwrap();
        assert!(out.working.is_empty());
        assert!(out.extra_vc2_raw.is_empty());
        assert!(out.extra_f128_raw.is_empty());
    }
}
