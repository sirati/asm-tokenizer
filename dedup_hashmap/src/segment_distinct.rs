//! `segment_distinct_count` — per-segment distinct-value count.
//!
//! Single concern: given a flat array of per-element segment labels
//! (`node`) and a parallel flat array of integer values (`ids`), return
//! `int64[n_nodes]` where `out[s]` is the number of DISTINCT `ids[i]`
//! over the elements `i` whose `node[i] == s`.
//!
//! This is the kernel form of the CSR-grouped distinct count that a
//! global `np.unique(node * 2^16 + ids)` + `bincount` performs in the
//! loader's category-counts path, with two differences that make it
//! faster while staying byte-identical on the output:
//!
//! * No global sort. Each segment accumulates its own values into a
//!   per-segment `hashbrown::HashSet`; the result count is that set's
//!   `len()`. Set membership is order-independent, so the per-segment
//!   distinct count matches `np.unique(ids_of_segment).size` exactly.
//! * The kernel is decode-agnostic: it knows nothing about carriers,
//!   CSR record offsets, or the v2 wire codec. The caller decodes the
//!   `(segment_label, value)` pairs (cheap, vectorized numpy) and hands
//!   them here as two flat `int64` arrays.
//!
//! `node` values are expected in `0 .. n_nodes`; any element whose label
//! is out of that range is a caller contract violation and raises
//! `ValueError` (mirrors the numpy `bincount(minlength=n_nodes)` +
//! pre-clipped node domain — the loader only ever produces in-range
//! labels via `searchsorted` over the CSR offsets).

use hashbrown::HashSet;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Per-segment distinct `ids` count.
///
/// Parameters:
/// - `node`: `int64[m]` — owning segment label per element.
/// - `ids`: `int64[m]` — value per element (caller-decoded).
/// - `n_nodes`: number of segments; output length.
///
/// Returns `int64[n_nodes]`: `out[s]` = `|{ ids[i] : node[i] == s }|`.
#[pyfunction]
pub fn segment_distinct_count<'py>(
    py: Python<'py>,
    node: PyReadonlyArray1<'py, i64>,
    ids: PyReadonlyArray1<'py, i64>,
    n_nodes: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let node_slice = node.as_slice()?;
    let ids_slice = ids.as_slice()?;
    if node_slice.len() != ids_slice.len() {
        return Err(PyValueError::new_err(format!(
            "node and ids must have equal length (got {} vs {})",
            node_slice.len(),
            ids_slice.len()
        )));
    }

    let result: PyResult<Vec<i64>> = py.detach(|| {
        // One distinct-value set per segment. Lazily allocated: a segment
        // with no elements stays an empty set and contributes count 0,
        // matching `bincount`'s zero-fill for absent labels.
        let mut sets: Vec<HashSet<i64>> = Vec::with_capacity(n_nodes);
        sets.resize_with(n_nodes, HashSet::new);

        for (&s, &v) in node_slice.iter().zip(ids_slice.iter()) {
            if s < 0 || (s as usize) >= n_nodes {
                return Err(PyValueError::new_err(format!(
                    "segment label {s} out of range [0, {n_nodes})"
                )));
            }
            sets[s as usize].insert(v);
        }

        Ok(sets.iter().map(|set| set.len() as i64).collect())
    });

    Ok(PyArray1::from_vec(py, result?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: brute-force per-segment distinct count over Rust vecs.
    fn reference(node: &[i64], ids: &[i64], n_nodes: usize) -> Vec<i64> {
        let mut out = vec![0i64; n_nodes];
        for s in 0..n_nodes {
            let mut seen: HashSet<i64> = HashSet::new();
            for (i, &lbl) in node.iter().enumerate() {
                if lbl as usize == s {
                    seen.insert(ids[i]);
                }
            }
            out[s] = seen.len() as i64;
        }
        out
    }

    fn distinct(node: &[i64], ids: &[i64], n_nodes: usize) -> Vec<i64> {
        let mut sets: Vec<HashSet<i64>> = Vec::with_capacity(n_nodes);
        sets.resize_with(n_nodes, HashSet::new);
        for (&s, &v) in node.iter().zip(ids.iter()) {
            sets[s as usize].insert(v);
        }
        sets.iter().map(|set| set.len() as i64).collect()
    }

    #[test]
    fn empty_input() {
        assert_eq!(distinct(&[], &[], 0), Vec::<i64>::new());
        assert_eq!(distinct(&[], &[], 3), vec![0, 0, 0]);
    }

    #[test]
    fn single_segment_dups() {
        // segment 0 has ids {5,5,7,5,7} -> distinct {5,7} -> 2.
        let node = [0, 0, 0, 0, 0];
        let ids = [5, 5, 7, 5, 7];
        assert_eq!(distinct(&node, &ids, 1), vec![2]);
        assert_eq!(distinct(&node, &ids, 1), reference(&node, &ids, 1));
    }

    #[test]
    fn multi_segment_with_gap() {
        // segment 1 empty (no elements) -> 0.
        let node = [0, 0, 2, 2, 2];
        let ids = [9, 9, 1, 2, 1];
        // seg0: {9}=1, seg1: {}=0, seg2: {1,2}=2
        assert_eq!(distinct(&node, &ids, 3), vec![1, 0, 2]);
        assert_eq!(distinct(&node, &ids, 3), reference(&node, &ids, 3));
    }

    #[test]
    fn same_id_different_segments_independent() {
        // id 4 appears in seg0 and seg1; counted once per segment.
        let node = [0, 1, 0, 1];
        let ids = [4, 4, 4, 4];
        assert_eq!(distinct(&node, &ids, 2), vec![1, 1]);
    }

    #[test]
    fn matches_reference_pseudorandom() {
        let n_nodes = 17usize;
        let mut node = Vec::new();
        let mut ids = Vec::new();
        let mut x = 12345u64;
        for _ in 0..5000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            node.push((x % n_nodes as u64) as i64);
            ids.push(((x >> 16) % 40) as i64);
        }
        assert_eq!(
            distinct(&node, &ids, n_nodes),
            reference(&node, &ids, n_nodes)
        );
    }
}
