//! `build_carrier_signs_kernel` — the stage-3 per-source sign collection.
//!
//! Single concern: given the FLAT `DenseColumns` columns (full-DFS-axis
//! EXPANDED / RAW arrays + CSR offsets + per-node `surviving` scalar + the
//! kept-node DFS index), identify every surviving NUMBER-band CARRIER in
//! DFS-then-stream encounter order and emit its `(block_idx, sign)` pair as
//! a deterministic GIL-released integer/bool walk.
//!
//! Mirrors the numpy `_batched_carrier_signs` (`_bulk_bytes.py` lines
//! 302-424) EXACTLY — it re-implements NO decode rule beyond reproducing
//! that vectorised arithmetic as a per-segment scalar walk:
//!
//! * body axis of kept node `e` = `expanded[node_slice][1:surviving]`
//!   (slot 0 is the synthetic prepend, dropped); `surviving - 1` slots
//!   when `surviving >= 1`, else 0.
//! * `is_painted = extra_value_v2_mask | extra_f128_mask` over that body;
//!   `is_real = !is_painted`.
//! * `real_idx_local` = within-segment `cumsum(is_real) - 1` (the K-th
//!   non-painted body slot consumes the K-th raw-stream real position).
//! * A NUMBER-band (`band_lo <= shifted < band_hi`) non-painted slot is a
//!   CARRIER; its sign is `is_negative_per_position[raw_slice][
//!   real_positions[real_idx_local]]` and its block is `shifted - band_lo`.
//!
//! The per-segment `real_positions` are recovered by scanning `real_mask`
//! over the node's raw slice (the numpy `np.nonzero(real_mask[raw_slice])`),
//! so the kernel reads the same flat columns the numpy path sliced — no
//! Python per-node loop, no concatenation front-matter.
//!
//! ## Carrier order
//!
//! Kept nodes are walked in `kept_node_index` order (== canonical
//! DFS-then-stream stage-3 linearisation); within a node, body order
//! (ascending). Identical to the numpy boolean-mask gather, which preserves
//! that scan order.

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

/// The kernel's pure-Rust core (no PyO3 in the signature) so unit tests can
/// drive it directly. Returns `(carrier_block_idx, carrier_signs)` in
/// DFS-then-stream encounter order.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    expanded: &[u16],
    extra_value_v2_mask: &[bool],
    extra_f128_mask: &[bool],
    node_offsets: &[i64],
    real_mask: &[bool],
    is_negative_per_position: &[bool],
    raw_offsets: &[i64],
    surviving_token_count: &[i64],
    kept_node_index: &[i64],
    band_lo: i64,
    band_hi: i64,
) -> Result<(Vec<i64>, Vec<bool>), String> {
    let mut block_idx: Vec<i64> = Vec::new();
    let mut signs: Vec<bool> = Vec::new();

    for &e_i64 in kept_node_index {
        if e_i64 < 0 || (e_i64 as usize) >= surviving_token_count.len() {
            return Err(format!(
                "kept_node_index entry {e_i64} out of range for n_nodes {}",
                surviving_token_count.len()
            ));
        }
        let e = e_i64 as usize;
        let surviving = surviving_token_count[e];

        // EXPANDED body axis = node_slice[1:surviving]; empty when
        // surviving <= 1 (only the prepend or nothing survives).
        let (exp_lo, exp_hi) = csr_slice(node_offsets, e);
        let body_len = (surviving - 1).max(0) as usize;
        if body_len == 0 {
            continue;
        }
        let body_start = exp_lo + 1;
        let body_end = body_start + body_len;
        if body_end > exp_hi || body_end > expanded.len() {
            return Err(format!(
                "node {e} surviving {surviving} body [{body_start}, \
                 {body_end}) exceeds expanded slice [{exp_lo}, {exp_hi})"
            ));
        }

        // RAW slice -> real positions (numpy `np.nonzero(real_mask[..])`).
        let (raw_lo, raw_hi) = csr_slice(raw_offsets, e);
        if raw_hi > real_mask.len() || raw_hi > is_negative_per_position.len() {
            return Err(format!(
                "node {e} raw slice [{raw_lo}, {raw_hi}) exceeds flat \
                 real_mask / is_negative arrays"
            ));
        }

        // Within-segment cumsum(is_real) - 1: walk the body, advance the
        // real counter on each non-painted slot, and the real counter's
        // value (minus 1) indexes the node's k-th real position.
        let mut real_running: i64 = 0;
        for j in 0..body_len {
            let painted = extra_value_v2_mask[body_start + j]
                || extra_f128_mask[body_start + j];
            let is_real = !painted;
            if is_real {
                real_running += 1;
            }
            let shifted = expanded[body_start + j] as i64;
            let in_band = shifted >= band_lo && shifted < band_hi;
            if in_band && is_real {
                // real_idx_local = within-seg cumsum(is_real) - 1; the
                // node's k-th real raw position is the k-th True in
                // real_mask over the raw slice (k = real_running - 1).
                let target = (real_running - 1) as usize;
                let mut seen: i64 = 0;
                let mut raw_pos: Option<usize> = None;
                for p in raw_lo..raw_hi {
                    if real_mask[p] {
                        if seen == target as i64 {
                            raw_pos = Some(p);
                            break;
                        }
                        seen += 1;
                    }
                }
                let raw_pos = raw_pos.ok_or_else(|| {
                    format!(
                        "node {e} carrier real index {target} exceeds the \
                         node's real-position count"
                    )
                })?;
                signs.push(is_negative_per_position[raw_pos]);
                block_idx.push(shifted - band_lo);
            }
        }
    }

    Ok((block_idx, signs))
}

/// PyO3 wrapper: borrow the flat `DenseColumns` columns, run the carrier
/// walk under `py.detach`, and hand back `(carrier_block_idx_i64,
/// carrier_signs_bool)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_carrier_signs_kernel<'py>(
    py: Python<'py>,
    expanded: numpy::PyReadonlyArray1<'py, u16>,
    extra_value_v2_mask: numpy::PyReadonlyArray1<'py, bool>,
    extra_f128_mask: numpy::PyReadonlyArray1<'py, bool>,
    node_offsets: numpy::PyReadonlyArray1<'py, i64>,
    real_mask: numpy::PyReadonlyArray1<'py, bool>,
    is_negative_per_position: numpy::PyReadonlyArray1<'py, bool>,
    raw_offsets: numpy::PyReadonlyArray1<'py, i64>,
    surviving_token_count: numpy::PyReadonlyArray1<'py, i64>,
    kept_node_index: numpy::PyReadonlyArray1<'py, i64>,
    band_lo: i64,
    band_hi: i64,
) -> PyResult<Bound<'py, PyTuple>> {
    let expanded = expanded.as_slice()?;
    let extra_value_v2_mask = extra_value_v2_mask.as_slice()?;
    let extra_f128_mask = extra_f128_mask.as_slice()?;
    let node_offsets = node_offsets.as_slice()?;
    let real_mask = real_mask.as_slice()?;
    let is_negative_per_position = is_negative_per_position.as_slice()?;
    let raw_offsets = raw_offsets.as_slice()?;
    let surviving_token_count = surviving_token_count.as_slice()?;
    let kept_node_index = kept_node_index.as_slice()?;

    let (block_idx, signs) = py
        .detach(|| {
            run_kernel(
                expanded,
                extra_value_v2_mask,
                extra_f128_mask,
                node_offsets,
                real_mask,
                is_negative_per_position,
                raw_offsets,
                surviving_token_count,
                kept_node_index,
                band_lo,
                band_hi,
            )
        })
        .map_err(PyValueError::new_err)?;

    let block_arr = PyArray1::from_vec(py, block_idx);
    let signs_arr = PyArray1::from_vec(py, signs);
    PyTuple::new(py, [block_arr.into_any(), signs_arr.into_any()])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference numpy-equivalent: brute-force the carrier walk over a
    /// single segment in the SAME way `_batched_carrier_signs` does, but
    /// straight off per-node Vecs, to cross-check `run_kernel`.
    #[allow(clippy::too_many_arguments)]
    fn one_node(
        expanded_full: Vec<u16>,
        extra_vc2: Vec<bool>,
        extra_f128: Vec<bool>,
        real_mask: Vec<bool>,
        is_neg: Vec<bool>,
        surviving: i64,
    ) -> (Vec<i64>, Vec<bool>) {
        let node_offsets = vec![0, expanded_full.len() as i64];
        let raw_offsets = vec![0, real_mask.len() as i64];
        run_kernel(
            &expanded_full,
            &extra_vc2,
            &extra_f128,
            &node_offsets,
            &real_mask,
            &is_neg,
            &raw_offsets,
            &[surviving],
            &[0],
            1,
            8,
        )
        .unwrap()
    }

    #[test]
    fn single_vc2_carrier_negative() {
        // body = [prepend=9, VC2=1]; surviving=2. raw: [VC2_carrier, b0]
        // real_mask=[T,F]; is_neg at carrier=True.
        let (blocks, signs) = one_node(
            vec![9, 1],
            vec![false, false],
            vec![false, false],
            vec![true, false],
            vec![true, false],
            2,
        );
        assert_eq!(blocks, vec![0]); // VC2 = block 0
        assert_eq!(signs, vec![true]);
    }

    #[test]
    fn painted_slot_is_not_a_carrier() {
        // body = [prepend, VC2 carrier, VC2 painted]; only the carrier
        // contributes (painted borrows its carrier's raw position).
        let (blocks, signs) = one_node(
            vec![9, 1, 1],
            vec![false, false, true], // 2nd body slot painted (vc2)
            vec![false, false, false],
            vec![true, false],
            vec![false, false],
            3,
        );
        assert_eq!(blocks, vec![0]);
        assert_eq!(signs, vec![false]);
    }

    #[test]
    fn identity_band_slot_excluded() {
        // body = [prepend, BLOCK_V2=8 (identity band), F32=4]. id-band slot
        // (shifted 8) is OUT of [1,8); only the F32 carrier counts.
        let (blocks, signs) = one_node(
            vec![9, 8, 4],
            vec![false, false, false],
            vec![false, false, false],
            vec![true, true], // both id + f32 carriers are real
            vec![false, true], // id pos sign F, f32 pos sign T
            3,
        );
        assert_eq!(blocks, vec![3]); // F32 = block 3
        assert_eq!(signs, vec![true]); // reads the SECOND real position
    }

    #[test]
    fn cut_drops_trailing_carriers() {
        // body would be [F32, F64] but surviving=2 -> body=[F32] only.
        let (blocks, signs) = one_node(
            vec![9, 4, 5],
            vec![false, false, false],
            vec![false, false, false],
            vec![true, true],
            vec![true, false],
            2, // surviving cuts after the prepend + F32
        );
        assert_eq!(blocks, vec![3]);
        assert_eq!(signs, vec![true]);
    }

    #[test]
    fn surviving_one_emits_nothing() {
        // Only the prepend survives -> empty body.
        let (blocks, signs) = one_node(
            vec![9, 4],
            vec![false, false],
            vec![false, false],
            vec![true, false],
            vec![true, false],
            1,
        );
        assert!(blocks.is_empty());
        assert!(signs.is_empty());
    }

    #[test]
    fn multi_node_order_and_real_base() {
        // Two nodes, each one carrier; the second node's real position is
        // recovered from ITS OWN raw slice (per-node reset), not a global
        // running index.
        let expanded = vec![9, 1, /* node1 */ 9, 4 /* node2 */];
        let node_offsets = vec![0, 2, 4];
        let real_mask = vec![true, false, /* node1 raw */ true, false];
        let raw_offsets = vec![0, 2, 4];
        let is_neg = vec![true, false, false, false];
        let (blocks, signs) = run_kernel(
            &expanded,
            &[false; 4],
            &[false; 4],
            &node_offsets,
            &real_mask,
            &is_neg,
            &raw_offsets,
            &[2, 2],
            &[0, 1],
            1,
            8,
        )
        .unwrap();
        assert_eq!(blocks, vec![0, 3]); // VC2 then F32, DFS order
        assert_eq!(signs, vec![true, false]);
    }

    #[test]
    fn adversarial_band_off_by_one_changes_carrier_set() {
        // shifted id 8 must be EXCLUDED (band_hi exclusive). A buggy
        // band_hi=9 would wrongly include it.
        let (blocks, _signs) = one_node(
            vec![9, 8],
            vec![false, false],
            vec![false, false],
            vec![true, false],
            vec![false, false],
            2,
        );
        assert!(
            blocks.is_empty(),
            "id-band slot 8 must be excluded from NUMBER carriers"
        );
    }
}
