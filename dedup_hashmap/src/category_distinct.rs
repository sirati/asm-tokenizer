//! `category_distinct_count` — per-node, per-COUNTER-Category distinct
//! caller-local-id counts decoded straight off the flat v2 wire stream.
//!
//! Single concern: decode the per-category carrier ids + owning node from
//! the flat v2 token stream and reduce to per-node distinct counts, with
//! the whole carrier-locate + ALG-5 payload-decode + per-segment distinct
//! reduction running under one `py.detach` (no GIL held).
//!
//! This fuses what the loader previously did as N GIL-held numpy preps
//! (`np.flatnonzero(raw_flat == carrier_id)` carrier-locate,
//! `np.searchsorted(record_offsets, …)` node attribution, masked
//! `np.where` ALG-5 payload-width gathers) feeding the per-segment
//! `segment_distinct_count` kernel — once PER CATEGORY — into a single
//! detached CSR walk over the flat stream that handles ALL categories at
//! once.
//!
//! ## ALG-5 payload-width decode (faithful port)
//!
//! For a carrier at flat position `p` owned by the node spanning
//! `[node_start, node_end)`:
//!
//! * `has_p1 = (p + 1) < node_end` — the flat-space form of the per-node
//!   `p < n - 1` guard. A carrier whose `p + 1` slot would fall in the
//!   NEXT node (or past the stream) has no in-node payload slot.
//! * payload length `L = has_p1 ? runlen_number[p + 1] : 0`.
//! * decoded caller-local id:
//!   - `L == 0`: id = 0 (encoder's reserved caller-local id 0).
//!   - `L == 1`: id = `raw[p + 1]` (low byte; high byte 0).
//!   - `L == 2`: id = `(raw[p + 1] << 8) | raw[p + 2]` (big-endian u16);
//!     the 2-long inline run guarantees `p + 2 < node_end`.
//!   - any other `L`: a v2-codec violation — surfaced as a `ValueError`
//!     carrying the SAME diagnostic the numpy `AssertionError` raised
//!     (offending carrier id + raw positions + declared lengths).
//!
//! ## Distinct reduction
//!
//! Per (category, node) a `hashbrown::HashSet<u16>` collects the decoded
//! ids; the per-node count is that set's `len()`. Set membership is
//! order-independent, so the count is byte-identical to the numpy
//! `np.unique(node * 2^16 + id)` + `bincount` global path it replaces.
//!
//! Output is a flat `int64[n_categories * n_nodes]` row-major grid (one
//! row per carrier id in the supplied order); the caller reshapes it to
//! `[n_categories, n_nodes]` and slices per Category.

use hashbrown::HashSet;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Decode result for one carrier position: the owning node plus the
/// ALG-5 caller-local id, or a width violation.
enum Decoded {
    Id { node: usize, id: u16 },
    BadWidth { pos: usize, length: u16 },
}

/// Per-category, per-node distinct caller-local-id counts.
///
/// Parameters:
/// - `raw_flat`: `u16[total]` — every node's v2 wire-form stream
///   concatenated in node order.
/// - `runlen_number_flat`: `u16[total]` — per-position inline-digit run
///   length, aligned to `raw_flat`.
/// - `record_offsets`: `i64[n_nodes + 1]` — CSR jump table; node `i` owns
///   `raw_flat[record_offsets[i] : record_offsets[i + 1]]`.
/// - `carrier_ids`: `u16[n_categories]` — the identity carrier id per
///   COUNTER Category, in the caller's category order.
/// - `n_nodes`: number of nodes; the per-category output row length.
///
/// Returns `i64[n_categories, n_nodes]`: `out[c][i]` = number of distinct
/// caller-local ids of category `c` decoded in node `i`.
#[pyfunction]
pub fn category_distinct_count<'py>(
    py: Python<'py>,
    raw_flat: PyReadonlyArray1<'py, u16>,
    runlen_number_flat: PyReadonlyArray1<'py, u16>,
    record_offsets: PyReadonlyArray1<'py, i64>,
    carrier_ids: PyReadonlyArray1<'py, u16>,
    n_nodes: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let raw = raw_flat.as_slice()?;
    let runlen = runlen_number_flat.as_slice()?;
    let rec = record_offsets.as_slice()?;
    let carriers = carrier_ids.as_slice()?;

    if raw.len() != runlen.len() {
        return Err(PyValueError::new_err(format!(
            "raw_flat and runlen_number_flat must have equal length \
             (got {} vs {})",
            raw.len(),
            runlen.len()
        )));
    }
    if rec.len() != n_nodes + 1 {
        return Err(PyValueError::new_err(format!(
            "record_offsets length {} must equal n_nodes + 1 = {}",
            rec.len(),
            n_nodes + 1
        )));
    }

    let n_cats = carriers.len();
    let flat = py.detach(|| count_grid(raw, runlen, rec, carriers, n_nodes))?;

    let grid = PyArray2::<i64>::zeros(py, [n_cats, n_nodes], false);
    // SAFETY: freshly allocated, exclusively owned here; `flat` is exactly
    // `n_cats * n_nodes` long (row-major), matching the array's layout.
    unsafe {
        grid.as_slice_mut()?.copy_from_slice(&flat);
    }
    Ok(grid)
}

/// Detached core: CSR-walk the flat stream once, decode every carrier of
/// every category, and reduce to the flat `i64[n_cats * n_nodes]` grid.
fn count_grid(
    raw: &[u16],
    runlen: &[u16],
    rec: &[i64],
    carriers: &[u16],
    n_nodes: usize,
) -> PyResult<Vec<i64>> {
    let n_cats = carriers.len();

    // One distinct-value set per (category, node). Row-major:
    // `sets[c * n_nodes + i]`. Lazily empty sets contribute count 0,
    // matching `bincount`'s zero-fill for absent labels.
    let mut sets: Vec<HashSet<u16>> = Vec::with_capacity(n_cats * n_nodes);
    sets.resize_with(n_cats * n_nodes, HashSet::new);

    for node in 0..n_nodes {
        let node_start = rec[node] as usize;
        let node_end = rec[node + 1] as usize;
        for p in node_start..node_end {
            let value = raw[p];
            // A carrier id may serve at most one category (the COUNTER
            // table is a bijection over identity offsets), but scanning
            // all categories keeps the kernel agnostic to that invariant.
            for (cat, &carrier_id) in carriers.iter().enumerate() {
                if value != carrier_id {
                    continue;
                }
                match decode_carrier(raw, runlen, p, node, node_end) {
                    Decoded::Id { node: n, id } => {
                        sets[cat * n_nodes + n].insert(id);
                    }
                    Decoded::BadWidth { pos, length } => {
                        return Err(PyValueError::new_err(format!(
                            "Identity carrier id {carrier_id} at raw position \
                             {pos} declared payload length {length} -- v2 spec \
                             restricts identity payloads to {{0, 1, 2}} bytes."
                        )));
                    }
                }
            }
        }
    }

    Ok(sets.iter().map(|set| set.len() as i64).collect())
}

/// Faithful ALG-5 payload-width decode for the carrier at flat position
/// `p` owned by `node` (whose exclusive end is `node_end`).
#[inline]
fn decode_carrier(
    raw: &[u16],
    runlen: &[u16],
    p: usize,
    node: usize,
    node_end: usize,
) -> Decoded {
    // `has_p1`: the per-node `p < n - 1` guard in flat space. Absent
    // payload slot -> length 0 -> the 0-byte branch (id 0).
    let length: u16 = if p + 1 < node_end { runlen[p + 1] } else { 0 };
    match length {
        0 => Decoded::Id { node, id: 0 },
        1 => Decoded::Id {
            node,
            id: raw[p + 1],
        },
        2 => {
            // The 2-long inline run that produced `length == 2` lives
            // within the node, so `p + 2 < node_end` holds.
            let hi = raw[p + 1];
            let lo = raw[p + 2];
            Decoded::Id {
                node,
                id: (hi << 8) | lo,
            }
        }
        other => Decoded::BadWidth {
            pos: p,
            length: other,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference: per (category, node) decode every carrier
    /// and count distinct ids, returning the row-major `n_cats * n_nodes`
    /// grid. Mirrors the numpy per-node `np.unique(ids).size` exactly.
    fn reference(
        raw: &[u16],
        runlen: &[u16],
        rec: &[i64],
        carriers: &[u16],
        n_nodes: usize,
    ) -> Vec<i64> {
        let n_cats = carriers.len();
        let mut out = vec![0i64; n_cats * n_nodes];
        for (cat, &carrier_id) in carriers.iter().enumerate() {
            for node in 0..n_nodes {
                let node_start = rec[node] as usize;
                let node_end = rec[node + 1] as usize;
                let mut seen: HashSet<u16> = HashSet::new();
                for p in node_start..node_end {
                    if raw[p] != carrier_id {
                        continue;
                    }
                    let length: u16 =
                        if p + 1 < node_end { runlen[p + 1] } else { 0 };
                    let id = match length {
                        0 => 0,
                        1 => raw[p + 1],
                        2 => (raw[p + 1] << 8) | raw[p + 2],
                        _ => panic!("bad width in reference"),
                    };
                    seen.insert(id);
                }
                out[cat * n_nodes + node] = seen.len() as i64;
            }
        }
        out
    }

    #[test]
    fn zero_byte_reserved_id() {
        // carrier 264 at the node's LAST position -> no p+1 -> length 0 ->
        // id 0. Two such carriers in one node collapse to distinct {0} = 1.
        let raw = [264u16, 264u16];
        let runlen = [0u16, 0u16];
        let rec = [0i64, 2];
        let carriers = [264u16];
        let out = count_grid(&raw, &runlen, &rec, &carriers, 1).unwrap();
        assert_eq!(out, vec![1]);
        assert_eq!(out, reference(&raw, &runlen, &rec, &carriers, 1));
    }

    #[test]
    fn one_byte_decode() {
        // carrier 264 followed by a 1-byte payload (runlen at p+1 == 1).
        // ids: {5, 7, 5} -> distinct 2.
        let raw = [264u16, 5, 264, 7, 264, 5];
        let runlen = [0u16, 1, 0, 1, 0, 1];
        let rec = [0i64, 6];
        let carriers = [264u16];
        let out = count_grid(&raw, &runlen, &rec, &carriers, 1).unwrap();
        assert_eq!(out, vec![2]);
        assert_eq!(out, reference(&raw, &runlen, &rec, &carriers, 1));
    }

    #[test]
    fn two_byte_big_endian() {
        // 2-byte payload: id = (hi << 8) | lo. runlen at p+1 == 2.
        let raw = [264u16, 1, 2, 264, 1, 2];
        let runlen = [0u16, 2, 0, 0, 2, 0];
        let rec = [0i64, 6];
        let carriers = [264u16];
        let out = count_grid(&raw, &runlen, &rec, &carriers, 1).unwrap();
        // both decode to 0x0102 -> distinct {258} -> 1.
        assert_eq!(out, vec![1]);
        assert_eq!(out, reference(&raw, &runlen, &rec, &carriers, 1));
    }

    #[test]
    fn boundary_no_p1_across_node() {
        // carrier at the last slot of node 0; its "p+1" lands in node 1 but
        // the in-node guard forbids the read -> length 0 -> id 0 for node 0.
        // Node 1 has an independent carrier with a 1-byte payload.
        let raw = [264u16, 264, 9];
        let runlen = [0u16, 1, 0];
        let rec = [0i64, 1, 3];
        let carriers = [264u16];
        // node0: carrier at p=0, p+1=1 >= node_end(1) -> id 0 -> {0}=1
        // node1: carrier at p=1, p+1=2 < node_end(3), runlen[2]=0 -> id0 ={0}=1
        let out = count_grid(&raw, &runlen, &rec, &carriers, 2).unwrap();
        assert_eq!(out, vec![1, 1]);
        assert_eq!(out, reference(&raw, &runlen, &rec, &carriers, 2));
    }

    #[test]
    fn multi_category_independent_rows() {
        // Two carriers (264, 268) interleaved; each category gets its own
        // row. Node 0 has 264->{3}, 268->{4,5}.
        let raw = [264u16, 3, 268, 4, 268, 5];
        let runlen = [0u16, 1, 0, 1, 0, 1];
        let rec = [0i64, 6];
        let carriers = [264u16, 268u16];
        let out = count_grid(&raw, &runlen, &rec, &carriers, 1).unwrap();
        // row0 (264): {3}=1; row1 (268): {4,5}=2.
        assert_eq!(out, vec![1, 2]);
        assert_eq!(out, reference(&raw, &runlen, &rec, &carriers, 1));
    }

    #[test]
    fn bad_width_raises() {
        // runlen at p+1 == 3 -> v2-codec violation -> Err.
        let raw = [264u16, 1, 2, 3];
        let runlen = [0u16, 3, 0, 0];
        let rec = [0i64, 4];
        let carriers = [264u16];
        let err = count_grid(&raw, &runlen, &rec, &carriers, 1);
        assert!(err.is_err());
    }

    #[test]
    fn empty_stream_all_zero() {
        let raw: [u16; 0] = [];
        let runlen: [u16; 0] = [];
        let rec = [0i64];
        let carriers = [264u16, 268u16];
        let out = count_grid(&raw, &runlen, &rec, &carriers, 0).unwrap();
        assert_eq!(out, Vec::<i64>::new());
    }

    #[test]
    fn matches_reference_pseudorandom() {
        // Pseudorandom multi-node, multi-category stream vs brute force.
        // Build node by node, emitting carriers with valid 0/1/2-byte
        // payloads so widths never violate ALG-5.
        let carriers = [264u16, 268u16, 269u16];
        let n_nodes = 13usize;
        let mut raw: Vec<u16> = Vec::new();
        let mut runlen: Vec<u16> = Vec::new();
        let mut rec: Vec<i64> = vec![0];
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            x
        };
        for _ in 0..n_nodes {
            let n_emit = (next() % 8) as usize;
            for _ in 0..n_emit {
                // Choose: a carrier (with payload) or filler.
                if next() % 2 == 0 {
                    let cid = carriers[(next() as usize) % carriers.len()];
                    let width = (next() % 3) as u16; // 0, 1, or 2
                    raw.push(cid);
                    runlen.push(0); // carrier's own runlen slot
                    match width {
                        1 => {
                            raw.push((next() % 50) as u16);
                            // The decode reads runlen at p+1 to get width.
                            // Overwrite the just-pushed runlen slot to 1.
                            let last = runlen.len();
                            runlen.push(0);
                            runlen[last] = 1;
                        }
                        2 => {
                            raw.push((next() % 50) as u16);
                            raw.push((next() % 50) as u16);
                            let at = runlen.len();
                            runlen.push(0);
                            runlen.push(0);
                            runlen[at] = 2;
                        }
                        _ => {}
                    }
                } else {
                    raw.push((next() % 50) as u16); // filler digit
                    runlen.push((next() % 3) as u16);
                }
            }
            rec.push(raw.len() as i64);
        }
        let got = count_grid(&raw, &runlen, &rec, &carriers, n_nodes).unwrap();
        let want = reference(&raw, &runlen, &rec, &carriers, n_nodes);
        assert_eq!(got, want);
    }
}
