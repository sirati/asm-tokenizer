//! `build_token_scatter_kernel` — fully-vectorized token scatter into `u16[B, L]`.
//!
//! Single concern: place every SURVIVING expanded node body + each row's
//! variant prefix into a dense `u16[n_rows, seq_len]` token tensor with ONE
//! GIL-released scalar walk — the Rust twin of numpy `scatter_tokens`
//! (`_token_scatter.py`). It re-derives NO decode / cut rule; the per-node
//! `surviving` count (the straddler cut) and `row_of_node` are computed
//! python-side (different concerns, `_surviving.py`) and passed in verbatim.
//!
//! ## Column math (mirrors the numpy, body-free geometry)
//!
//! * `gcum` = exclusive cumsum of `own_length` (size `n_emitted + 1`); node
//!   `e` begins, globally, at column-stream offset `gcum[e]`.
//! * `body_base[row]` = `gcum[row_offsets[row]]` (cumulative own-length before
//!   the row's first node); `body_start[e] = gcum[e] - body_base[row_of[e]]`
//!   is node `e`'s body-relative start column within its row.
//! * node `e` writes its first `surviving[e]` expanded slots
//!   (`expanded[node_offsets[e] .. node_offsets[e] + surviving[e]]`) to row
//!   `row_of[e]` at columns `prefix_len[row_of[e]] + body_start[e] + (0..)`.
//! * row `r`'s prefix writes `min(width_r, seq_len)` ids
//!   (`variant_prefix_tokens[variant_prefix_offsets[r] ..]`) to columns
//!   `0 .. cap - 1`.
//!
//! ## Last-writer-wins ordering (CRITICAL — mirror exactly)
//!
//! numpy concatenates `[prefix_writes, node_writes]` and does a single fancy-
//! index assignment `tokens[rows, cols] = vals`; on a `(row, col)` collision
//! numpy applies writes IN ORDER and the LAST wins, so the node body wins over
//! the prefix. This kernel replays the writes in that SAME order — every
//! prefix slot first, then every node-body slot — into the dense buffer, so
//! the identical last-writer survives. Out-of-`[0, seq_len)` columns are
//! dropped (the numpy `in_bounds` guard); `id == 0` stays wherever no slot
//! writes (the zeroed allocation).

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it directly.
/// Mirrors numpy `scatter_tokens` + its `_flatten_node_writes` /
/// `_flatten_prefix_writes` helpers, returning the flat row-major
/// `n_rows * seq_len` token buffer.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    n_rows: usize,
    seq_len: usize,
    row_offsets: &[i64],
    own_length: &[i64],
    expanded: &[u16],
    node_offsets: &[i64],
    prefix_len: &[i64],
    surviving: &[i64],
    row_of_node: &[i64],
    variant_prefix_tokens: &[u16],
    variant_prefix_offsets: &[i64],
) -> Result<Vec<u16>, String> {
    let mut tokens = vec![0u16; n_rows * seq_len];
    if n_rows == 0 || seq_len == 0 {
        return Ok(tokens);
    }

    let n_emitted = own_length.len();
    if row_offsets.len() != n_rows + 1 {
        return Err(format!(
            "row_offsets has {} entries, expected n_rows + 1 = {}",
            row_offsets.len(),
            n_rows + 1
        ));
    }
    if node_offsets.len() != n_emitted + 1 {
        return Err(format!(
            "node_offsets covers {} nodes but the emission has {n_emitted}",
            node_offsets.len().saturating_sub(1)
        ));
    }
    if surviving.len() != n_emitted || row_of_node.len() != n_emitted {
        return Err(format!(
            "per-node arrays disagree on n_emitted ({n_emitted}): surviving {} \
             row_of_node {}",
            surviving.len(),
            row_of_node.len()
        ));
    }
    if prefix_len.len() != n_rows {
        return Err(format!(
            "prefix_len has {} entries, expected n_rows = {n_rows}",
            prefix_len.len()
        ));
    }
    if variant_prefix_offsets.len() != n_rows + 1 {
        return Err(format!(
            "variant_prefix_offsets has {} entries, expected n_rows + 1 = {}",
            variant_prefix_offsets.len(),
            n_rows + 1
        ));
    }

    // gcum = exclusive cumsum of own_length; gcum[e] = cumulative own-length
    // before node e. body_base[row] = gcum[row_offsets[row]].
    let mut gcum = vec![0i64; n_emitted + 1];
    let mut acc: i64 = 0;
    for e in 0..n_emitted {
        acc += own_length[e];
        gcum[e + 1] = acc;
    }
    let seq_len_i = seq_len as i64;

    // --- prefix writes FIRST (lose to the node body on collision) --------
    for r in 0..n_rows {
        let lo = variant_prefix_offsets[r];
        let hi = variant_prefix_offsets[r + 1];
        if lo < 0 || hi < lo {
            return Err(format!(
                "row {r} prefix window [{lo}, {hi}) is malformed"
            ));
        }
        let width = hi - lo;
        // capped at seq_len (a degenerate prefix wider than the budget is
        // truncated, matching the scalar `min(n_axis, context_len)`).
        let cap = width.min(seq_len_i);
        for k in 0..cap {
            // col = k, always in [0, seq_len) by the cap; no extra guard.
            let src = (lo + k) as usize;
            if src >= variant_prefix_tokens.len() {
                return Err(format!(
                    "row {r} prefix src index {src} exceeds prefix-token \
                     length {}",
                    variant_prefix_tokens.len()
                ));
            }
            tokens[r * seq_len + k as usize] = variant_prefix_tokens[src];
        }
    }

    // --- node body writes SECOND (win on collision) ----------------------
    for e in 0..n_emitted {
        let surv = surviving[e];
        if surv <= 0 {
            continue;
        }
        let row = row_of_node[e];
        if row < 0 || row as usize >= n_rows {
            return Err(format!("node {e} row_of_node {row} out of range"));
        }
        let row_u = row as usize;
        // body_start = gcum[e] - gcum[row_offsets[row]].
        let row_off = row_offsets[row_u];
        if row_off < 0 || row_off as usize >= gcum.len() {
            return Err(format!(
                "row {row_u} row_offsets {row_off} out of range for gcum \
                 length {}",
                gcum.len()
            ));
        }
        let body_base = gcum[row_off as usize];
        let body_start = gcum[e] - body_base;
        let col_base = prefix_len[row_u] + body_start;
        let src_base = node_offsets[e];
        for k in 0..surv {
            let col = col_base + k;
            // in-bounds guard: 0 <= col < seq_len (else drop the write).
            if col < 0 || col >= seq_len_i {
                continue;
            }
            let src = (src_base + k) as usize;
            if src >= expanded.len() {
                return Err(format!(
                    "node {e} expanded src index {src} exceeds expanded \
                     length {}",
                    expanded.len()
                ));
            }
            tokens[row_u * seq_len + col as usize] = expanded[src];
        }
    }

    Ok(tokens)
}

/// PyO3 wrapper: borrow the geometry arrays + expanded stream + per-row
/// prefix, run the scatter under `py.detach`, and return the dense token
/// tensor reshaped to `(n_rows, seq_len)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_token_scatter_kernel<'py>(
    py: Python<'py>,
    n_rows: usize,
    seq_len: usize,
    row_offsets: numpy::PyReadonlyArray1<'py, i64>,
    own_length: numpy::PyReadonlyArray1<'py, i64>,
    expanded: numpy::PyReadonlyArray1<'py, u16>,
    node_offsets: numpy::PyReadonlyArray1<'py, i64>,
    prefix_len: numpy::PyReadonlyArray1<'py, i64>,
    surviving: numpy::PyReadonlyArray1<'py, i64>,
    row_of_node: numpy::PyReadonlyArray1<'py, i64>,
    variant_prefix_tokens: numpy::PyReadonlyArray1<'py, u16>,
    variant_prefix_offsets: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let row_offsets = row_offsets.as_slice()?;
    let own_length = own_length.as_slice()?;
    let expanded = expanded.as_slice()?;
    let node_offsets = node_offsets.as_slice()?;
    let prefix_len = prefix_len.as_slice()?;
    let surviving = surviving.as_slice()?;
    let row_of_node = row_of_node.as_slice()?;
    let variant_prefix_tokens = variant_prefix_tokens.as_slice()?;
    let variant_prefix_offsets = variant_prefix_offsets.as_slice()?;

    let tokens = py
        .detach(|| {
            run_kernel(
                n_rows,
                seq_len,
                row_offsets,
                own_length,
                expanded,
                node_offsets,
                prefix_len,
                surviving,
                row_of_node,
                variant_prefix_tokens,
                variant_prefix_offsets,
            )
        })
        .map_err(PyValueError::new_err)?;

    Ok(PyArray1::from_vec(py, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 rows, a prefix, a straddler cut, and a body-prefix collision.
    /// Row 0: prefix width 2 (ids 90, 91); 2 nodes own_length [3, 2], the 2nd
    /// is the straddler keeping only 1 column (surviving = [3, 1]).
    /// Row 1: prefix width 1 (id 70); 1 node own_length 2, kept whole.
    #[test]
    fn two_rows_prefix_straddler_cut() {
        // expanded stream: node0 row0 [10,11,12], node1 row0 [13,14],
        //                  node2 row1 [20,21].
        let expanded: Vec<u16> = vec![10, 11, 12, 13, 14, 20, 21];
        let node_offsets: Vec<i64> = vec![0, 3, 5, 7];
        let own_length: Vec<i64> = vec![3, 2, 2];
        let row_offsets: Vec<i64> = vec![0, 2, 3]; // row0 nodes [0,2), row1 [2,3)
        let row_of_node: Vec<i64> = vec![0, 0, 1];
        let surviving: Vec<i64> = vec![3, 1, 2]; // node1 straddler -> keep 1
        let prefix_len: Vec<i64> = vec![2, 1];
        let variant_prefix_tokens: Vec<u16> = vec![90, 91, 70];
        let variant_prefix_offsets: Vec<i64> = vec![0, 2, 3];

        // seq_len 6.
        // Row0: prefix cols 0,1 = 90,91. body_start node0 = gcum[0]-gcum[0]=0,
        //   col_base = 2 -> cols 2,3,4 = 10,11,12. node1 body_start =
        //   gcum[1]-gcum[0] = 3, col_base = 5 -> keep 1 col -> col 5 = 13.
        // Row1: prefix col 0 = 70. node2 body_start = gcum[2]-gcum[2]=0,
        //   col_base = 1 -> cols 1,2 = 20,21.
        let out = run_kernel(
            2,
            6,
            &row_offsets,
            &own_length,
            &expanded,
            &node_offsets,
            &prefix_len,
            &surviving,
            &row_of_node,
            &variant_prefix_tokens,
            &variant_prefix_offsets,
        )
        .unwrap();
        let row0 = &out[0..6];
        let row1 = &out[6..12];
        assert_eq!(row0, &[90, 91, 10, 11, 12, 13]);
        assert_eq!(row1, &[70, 20, 21, 0, 0, 0]);
    }

    /// Collision: a prefix wider than the body start so the body overwrites a
    /// prefix column. numpy applies prefix THEN body -> body wins. If the
    /// kernel reversed the order the prefix would win and this asserts catch it.
    #[test]
    fn body_wins_collision_over_prefix() {
        // 1 row, seq_len 4, prefix width 3 (cols 0,1,2 = 90,91,92).
        // 1 node own_length 2, body_start 0, BUT prefix_len = 1 so the body
        // col_base = 1 -> body writes cols 1,2 = 50,51, colliding with prefix
        // cols 1,2. Body must win: result [90,50,51,0].
        let expanded: Vec<u16> = vec![50, 51];
        let node_offsets: Vec<i64> = vec![0, 2];
        let own_length: Vec<i64> = vec![2];
        let row_offsets: Vec<i64> = vec![0, 1];
        let row_of_node: Vec<i64> = vec![0];
        let surviving: Vec<i64> = vec![2];
        let prefix_len: Vec<i64> = vec![1];
        // prefix width 3 exceeds prefix_len 1 on purpose (degenerate); the
        // prefix still writes cols 0,1,2 (capped at seq_len 4), body then wins.
        let variant_prefix_tokens: Vec<u16> = vec![90, 91, 92];
        let variant_prefix_offsets: Vec<i64> = vec![0, 3];

        let out = run_kernel(
            1,
            4,
            &row_offsets,
            &own_length,
            &expanded,
            &node_offsets,
            &prefix_len,
            &surviving,
            &row_of_node,
            &variant_prefix_tokens,
            &variant_prefix_offsets,
        )
        .unwrap();
        assert_eq!(out, vec![90, 50, 51, 0]);
    }

    /// Prefix wider than seq_len is truncated (matches min(width, seq_len)).
    #[test]
    fn prefix_truncated_at_seq_len() {
        let expanded: Vec<u16> = vec![];
        let node_offsets: Vec<i64> = vec![0];
        let own_length: Vec<i64> = vec![];
        let row_offsets: Vec<i64> = vec![0, 0]; // row0 has no nodes
        let row_of_node: Vec<i64> = vec![];
        let surviving: Vec<i64> = vec![];
        let prefix_len: Vec<i64> = vec![3];
        // prefix width 5 but seq_len 2 -> only cols 0,1 written.
        let variant_prefix_tokens: Vec<u16> = vec![1, 2, 3, 4, 5];
        let variant_prefix_offsets: Vec<i64> = vec![0, 5];

        let out = run_kernel(
            1,
            2,
            &row_offsets,
            &own_length,
            &expanded,
            &node_offsets,
            &prefix_len,
            &surviving,
            &row_of_node,
            &variant_prefix_tokens,
            &variant_prefix_offsets,
        )
        .unwrap();
        assert_eq!(out, vec![1, 2]);
    }

    /// Body column past seq_len is dropped (the in_bounds guard).
    #[test]
    fn body_past_seq_len_dropped() {
        // 1 row seq_len 3, no prefix, 1 node own_length 4 surviving 4 at
        // col_base 1 -> cols 1,2,3,4; cols 3,4 are >= 3 -> dropped.
        let expanded: Vec<u16> = vec![10, 11, 12, 13];
        let node_offsets: Vec<i64> = vec![0, 4];
        let own_length: Vec<i64> = vec![4];
        let row_offsets: Vec<i64> = vec![0, 1];
        let row_of_node: Vec<i64> = vec![0];
        let surviving: Vec<i64> = vec![4];
        let prefix_len: Vec<i64> = vec![1];
        let variant_prefix_tokens: Vec<u16> = vec![];
        let variant_prefix_offsets: Vec<i64> = vec![0, 0];

        let out = run_kernel(
            1,
            3,
            &row_offsets,
            &own_length,
            &expanded,
            &node_offsets,
            &prefix_len,
            &surviving,
            &row_of_node,
            &variant_prefix_tokens,
            &variant_prefix_offsets,
        )
        .unwrap();
        // col 0 untouched (0), cols 1,2 = 10,11; 12,13 dropped.
        assert_eq!(out, vec![0, 10, 11]);
    }

    #[test]
    fn empty_no_rows() {
        let out = run_kernel(
            0,
            8,
            &[0],
            &[],
            &[],
            &[0],
            &[],
            &[],
            &[],
            &[],
            &[0],
        )
        .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_seq_len() {
        let out = run_kernel(
            2,
            0,
            &[0, 0, 0],
            &[],
            &[],
            &[0],
            &[0, 0],
            &[],
            &[],
            &[],
            &[0, 0, 0],
        )
        .unwrap();
        assert!(out.is_empty());
    }
}
