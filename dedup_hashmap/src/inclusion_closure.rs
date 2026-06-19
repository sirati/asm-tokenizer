//! Inclusion-closure frontier advance — the GIL-released per-level section
//! reachability step.
//!
//! Single concern: given one section frontier, gather every section it
//! reaches in ONE hop through its `ct_function_section_ptr` callee pointers,
//! returning the SORTED-UNIQUE callee-section set. This is the Rust port of
//! the inner per-section loop of
//! `LiveNodeAdjacency.ensure_inclusion_closure`
//! (`sorted_index/_graph_lengths/_adjacency.py`):
//!
//! ```text
//! callee_secs = []
//! for sec in frontier.tolist():
//!     ct_lo = ct_offsets[sec]; ct_hi = ct_offsets[sec + 1]
//!     if ct_hi <= ct_lo: continue
//!     ptrs = ct_function_section_ptr[ct_lo:ct_hi]
//!     ptrs = ptrs[ptrs != 0]                 # #69 explicit-zero -> no section
//!     if ptrs.size == 0: continue
//!     hits = _sec_map.lookup_ndarray(ptrs)
//!     callee_secs.append(hits[hits != _U32_MISS])
//! nxt = np.unique(np.concatenate(callee_secs))   # sorted ascending + dedup
//! ```
//!
//! It owns NEITHER the depth loop, the `reached` set, the
//! `all_sections_resident()` short-circuit, NOR the per-level
//! `ensure_sections` lazy-fill — those stay on the Python side because the
//! lazy catalog must materialise each level's `ct_*` columns BEFORE the
//! kernel reads them, and the `reached`-mask filter is trivial per-batch
//! numpy. The kernel only collapses the inner pointer-gather + `_sec_map`
//! lookup + `np.unique(np.concatenate(...))` reduction that the GIL held.
//!
//! ## State
//!
//! The method lives on `LiveAdjacencyKernel` and reads its `sec_map` — the
//! SAME `function_section_ptr -> section idx` map `resolve_one_parent`'s
//! map-miss gate uses, so the closure pre-pass can never resolve a pointer
//! differently from the real inclusion BFS.

use crate::adjacency_expand::{LiveAdjacencyKernel, U32_MISS};

impl LiveAdjacencyKernel {
    /// One inclusion-closure frontier hop, GIL released.
    ///
    /// For each `sec` in `frontier`, gathers
    /// `ct_function_section_ptr[ct_offsets[sec] : ct_offsets[sec + 1]]`, drops
    /// the #69 explicit-zero pointers, looks each survivor up in `sec_map`
    /// (skipping the all-ones miss), and returns the ascending sorted-unique
    /// set of hit child sections (== `np.unique(np.concatenate(...))`). An
    /// empty frontier — or a frontier whose sections reach nothing — yields
    /// an empty vec, on which the Python depth loop terminates.
    pub(crate) fn advance_inclusion_frontier_impl(
        &self,
        frontier: &[i64],
        ct_offsets: &[i64],
        ct_function_section_ptr: &[u32],
    ) -> Result<Vec<i64>, String> {
        // Accumulate every surviving hit across the frontier, mirroring the
        // numpy `callee_secs.append(...)` then `np.concatenate`. The sort +
        // dedup below reproduces `np.unique` exactly (ascending, deduped).
        let mut hits: Vec<i64> = Vec::new();
        for &sec in frontier.iter() {
            if sec < 0 {
                return Err(format!("frontier section {sec} is negative"));
            }
            let sec_u = sec as usize;
            if sec_u + 1 >= ct_offsets.len() {
                return Err(format!(
                    "frontier section {sec} out of ct_offsets range"
                ));
            }
            let ct_lo = ct_offsets[sec_u];
            let ct_hi = ct_offsets[sec_u + 1];
            // `if ct_hi <= ct_lo: continue` — no call_targets for this section.
            if ct_hi <= ct_lo {
                continue;
            }
            let lo = ct_lo as usize;
            let hi = ct_hi as usize;
            for &ptr in &ct_function_section_ptr[lo..hi] {
                // #69 explicit-zero pointer resolves to no section.
                if ptr == 0 {
                    continue;
                }
                // `_sec_map.lookup_ndarray` returns `_U32_MISS` for a miss;
                // the numpy path keeps `hits[hits != _U32_MISS]`.
                match self.sec_map.get(&ptr) {
                    Some(&v) if v != U32_MISS => hits.push(v as i64),
                    _ => {}
                }
            }
        }
        // `np.unique` == sort ascending + dedup.
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::HashMap;
    use std::sync::Mutex;

    fn kernel(pairs: &[(u32, u32)]) -> LiveAdjacencyKernel {
        let mut sec_map: HashMap<u32, u32> = HashMap::new();
        for &(k, v) in pairs {
            sec_map.insert(k, v);
        }
        LiveAdjacencyKernel::from_parts_for_test(sec_map, Mutex::new(HashMap::new()))
    }

    #[test]
    fn gathers_sorted_unique_callees() {
        // sec0 has ct slots [0,3) -> ptrs 100,200,100 ; sec1 ct slots [3,4)
        // -> ptr 300. Map: 100->5, 200->2, 300->2 (dup callee). Expect the
        // sorted-unique set {2, 5}.
        let ct_offsets = vec![0i64, 3, 4];
        let ptrs = vec![100u32, 200, 100, 300];
        let k = kernel(&[(100, 5), (200, 2), (300, 2)]);
        let out = k
            .advance_inclusion_frontier_impl(&[0i64, 1], &ct_offsets, &ptrs)
            .unwrap();
        assert_eq!(out, vec![2i64, 5]);
    }

    #[test]
    fn drops_zero_ptr_and_map_miss() {
        // ptrs: 0 (#69 zero) , 999 (map miss) , 100 (hit). Only 100 survives.
        let ct_offsets = vec![0i64, 3];
        let ptrs = vec![0u32, 999, 100];
        let k = kernel(&[(100, 7)]);
        let out = k
            .advance_inclusion_frontier_impl(&[0i64], &ct_offsets, &ptrs)
            .unwrap();
        assert_eq!(out, vec![7i64]);
    }

    #[test]
    fn empty_ct_range_and_empty_frontier() {
        let ct_offsets = vec![0i64, 0, 2];
        let ptrs = vec![100u32, 200];
        let k = kernel(&[(100, 1), (200, 3)]);
        // sec0 has an empty ct range -> skipped; frontier {0} reaches nothing.
        let out = k
            .advance_inclusion_frontier_impl(&[0i64], &ct_offsets, &ptrs)
            .unwrap();
        assert!(out.is_empty());
        // Empty frontier -> empty result.
        let out2 = k
            .advance_inclusion_frontier_impl(&[], &ct_offsets, &ptrs)
            .unwrap();
        assert!(out2.is_empty());
    }
}
