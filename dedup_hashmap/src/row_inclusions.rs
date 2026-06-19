//! `compute_row_inclusions_kernel` — the FUSED, single-GIL-released
//! inclusion-BFS kernel (Stage 3).
//!
//! Single concern: run the loader's WHOLE per-root-group + per-depth splice
//! BFS — the subset emission pass and the full-variant-set excluded-pool
//! pass — under ONE `py.detach`, returning the per-row emitted-node / edge-
//! type CSR plus the remembered-excluded pool CSR. This is the Rust port of
//! `vector_batch/_inclusion/_compute.py::compute_row_inclusions` +
//! `_bfs.py::{_bfs_emit, _bfs_full_included}` for the `unmatched_inline=
//! False` production path.
//!
//! ## What it FUSES (and why that is the multi-core unlock)
//!
//! Stage 1 (`LiveAdjacencyKernel`) and Stage 2 (`OnceOnlyInclusionKernel`)
//! each release the GIL per call, but the Python per-group + per-depth loop
//! between them re-acquires the GIL once per level per group. This kernel
//! drives the ENTIRE loop in-Rust — calling the SHARED Stage-1
//! `resolve_frontier` and owning a Stage-2 `DeciderState` directly — so the
//! whole BFS runs under a single `py.detach`. With no per-level GIL hop, N
//! threads each calling this kernel scale across cores.
//!
//! ## In-Rust reuse of the Stage-1/Stage-2 cores (no Python round-trip)
//!
//! * Adjacency: the caller passes the SAME `LiveAdjacencyKernel` instance
//!   the loader's `LiveNodeAdjacency` holds; this kernel calls its
//!   `pub(crate) resolve_frontier` per level, so the gates (#69 zero-ptr /
//!   EXTERN / map-miss), the J-fallback tie-break, and the ascending-unique
//!   slot order are byte-identical to the standalone Stage-1 path AND share
//!   its `sec_map` / `fallback_cache` state (no drift possible).
//! * Decider: this kernel owns one `DeciderState` and drives `begin_root` /
//!   `step_level` exactly as the Python `_bfs` loop did — the columnwise-ALL
//!   FLAG-A, the FLAG-B read-before-write, the first-in-level dedup, all
//!   verbatim from Stage 2.
//!
//! ## ORDER-PRESERVING (the gate decision)
//!
//! The per-level frontier flattening is the EXACT
//! parent-ascending-`parent_pos`, within-parent ascending-call_target-slot
//! order `resolve_frontier` returns — identical to the numpy `_expand_level`
//! concatenation. So the per-BFS-level node SET *and* intra-level sibling
//! order are byte-preserved: the new->reference permutation is the IDENTITY,
//! and the existing byte-identity gate + `test_emit_order_canonicalization`
//! suffice. Intra-level order is NOT reordered.
//!
//! ## Lazy-catalog contract (caller's responsibility)
//!
//! The kernel reads the catalog columns RESIDENT — it never re-acquires the
//! GIL to lazily fill a section. The Python facade
//! (`LiveNodeAdjacency.ensure_inclusion_closure`) materialises the section-
//! reachability closure of the roots up to `max_depth` BEFORE this call, so
//! every section the BFS can touch is filled. Reachability (no inclusion
//! pruning) is a superset of the inclusion-touched sections, so the resident
//! arrays are complete for the whole fused traversal.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::adjacency_expand::{Columns, GateConstants, LiveAdjacencyKernel};
use crate::once_only_inclusion::DeciderState;

/// One batch row's accumulated emission (root + included callees in BFS
/// emission order) — the per-row scratch the subset pass fills.
struct RowEmission {
    nodes: Vec<i64>,
    types: Vec<u8>,
}

/// The fused kernel's flat result, assembled into per-row CSR by the caller.
struct InclusionResult {
    /// `i64[n_rows + 1]` emitted-node CSR offsets.
    emitted_offsets: Vec<i64>,
    /// `i64[sum k]` emitted catalog nodes (root at each row's slot 0).
    emitted_nodes: Vec<i64>,
    /// `u8[sum k]` parallel emitted edge `CallTargetType`.
    emitted_types: Vec<u8>,
    /// `i64[n_rows + 1]` excluded-pool CSR offsets.
    pool_offsets: Vec<i64>,
    /// `i64[sum m]` pool catalog nodes (ascending-unique per row).
    pool_nodes: Vec<i64>,
    /// `u8[sum m]` parallel pool edge `CallTargetType`.
    pool_types: Vec<u8>,
}

/// Drive the WHOLE inclusion BFS for one batch under a single GIL release.
///
/// `adjacency` is the loader's `LiveAdjacencyKernel` (shared `sec_map` /
/// `fallback_cache`); the catalog columns are its flat global backing; the
/// `root_*` arrays are the per-row batch inputs; `consts` carries the
/// EXTERN / MISSING sentinels. Returns the per-row emitted + pool CSR.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn compute_row_inclusions_kernel<'py>(
    py: Python<'py>,
    adjacency: PyRef<'py, LiveAdjacencyKernel>,
    pce_offsets: PyReadonlyArray1<'py, i64>,
    pce_called_idx: PyReadonlyArray1<'py, u16>,
    pce_section_variant_index: PyReadonlyArray1<'py, u16>,
    ct_offsets: PyReadonlyArray1<'py, i64>,
    ct_type: PyReadonlyArray1<'py, u8>,
    ct_function_section_ptr: PyReadonlyArray1<'py, u32>,
    ct_is_matched: PyReadonlyArray1<'py, bool>,
    var_offsets: PyReadonlyArray1<'py, i64>,
    n_call_targets: PyReadonlyArray1<'py, i64>,
    sec_of_var: PyReadonlyArray1<'py, i64>,
    root_sections: PyReadonlyArray1<'py, i64>,
    root_sampled_variants: PyReadonlyArray1<'py, i64>,
    root_groups: PyReadonlyArray1<'py, i64>,
    max_depth: i64,
    need_excluded_pool: bool,
    extern_type: u8,
    missing_variant_index: u16,
    root_edge_type: u8,
    initial_cols: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    if max_depth < 0 {
        return Err(PyValueError::new_err(format!(
            "max_depth must be >= 0; got {max_depth}"
        )));
    }
    let cols = Columns {
        pce_offsets: pce_offsets.as_slice()?,
        pce_called_idx: pce_called_idx.as_slice()?,
        pce_section_variant_index: pce_section_variant_index.as_slice()?,
        ct_offsets: ct_offsets.as_slice()?,
        ct_type: ct_type.as_slice()?,
        ct_function_section_ptr: ct_function_section_ptr.as_slice()?,
        ct_is_matched: ct_is_matched.as_slice()?,
        var_offsets: var_offsets.as_slice()?,
        n_call_targets: n_call_targets.as_slice()?,
        sec_of_var: sec_of_var.as_slice()?,
    };
    let consts = GateConstants {
        extern_type,
        missing_variant_index,
    };
    // Deref the PyRef to a plain `&LiveAdjacencyKernel` BEFORE the detach:
    // a bare Rust reference to a `Sync` pyclass is `Send`/`Ungil` (the
    // sec_map is immutable + the fallback_cache is a `Mutex`), whereas the
    // `PyRef` smart pointer is not. This is the SAME pattern the standalone
    // `expand_batch` uses (it calls `self.resolve_frontier` inside detach).
    let adjacency: &LiveAdjacencyKernel = &adjacency;
    let sec = root_sections.as_slice()?;
    let smp = root_sampled_variants.as_slice()?;
    let grp = root_groups.as_slice()?;
    if sec.len() != smp.len() || sec.len() != grp.len() {
        return Err(PyValueError::new_err(format!(
            "root_sections, root_sampled_variants and root_groups must be \
             parallel; got {} vs {} vs {}",
            sec.len(),
            smp.len(),
            grp.len()
        )));
    }

    let result = py
        .detach(|| {
            run_inclusion_bfs(
                &adjacency,
                &cols,
                &consts,
                sec,
                smp,
                grp,
                max_depth,
                need_excluded_pool,
                root_edge_type,
                initial_cols,
            )
        })
        .map_err(PyValueError::new_err)?;

    let tup = PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, result.emitted_offsets).into_any(),
            PyArray1::from_vec(py, result.emitted_nodes).into_any(),
            PyArray1::from_vec(py, result.emitted_types).into_any(),
            PyArray1::from_vec(py, result.pool_offsets).into_any(),
            PyArray1::from_vec(py, result.pool_nodes).into_any(),
            PyArray1::from_vec(py, result.pool_types).into_any(),
        ],
    )?;
    Ok(tup)
}

/// The GIL-released body: group the rows, run the two BFS passes per group,
/// assemble the per-row CSR. Mirrors `_compute.compute_row_inclusions`.
#[allow(clippy::too_many_arguments)]
fn run_inclusion_bfs(
    adjacency: &LiveAdjacencyKernel,
    cols: &Columns,
    consts: &GateConstants,
    sec: &[i64],
    smp: &[i64],
    grp: &[i64],
    max_depth: i64,
    need_excluded_pool: bool,
    root_edge_type: u8,
    initial_cols: usize,
) -> Result<InclusionResult, String> {
    let n_rows = sec.len();
    // The single reused decider for the whole batch (mirrors the one Python
    // `OnceOnlyInclusion()` instance threaded through every group + pass).
    let mut decider = DeciderState::new(initial_cols.max(1));

    // Per-row emission scratch, filled in row index space (output is indexed
    // by row, so group visitation order is irrelevant — but we group in
    // first-appearance order, like the Python `dict`, regardless).
    let mut emissions: Vec<RowEmission> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        emissions.push(RowEmission {
            nodes: Vec::new(),
            types: Vec::new(),
        });
    }
    // Per-row pool, filled only when needed.
    let mut pools: Vec<(Vec<i64>, Vec<u8>)> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        pools.push((Vec::new(), Vec::new()));
    }

    // Group rows by `root_groups`, preserving first-appearance order (the
    // output is indexed by row, so group order does not affect it; we mirror
    // the Python `dict` insertion order anyway for a faithful drive).
    let mut groups: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut group_pos: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();
    for (r, &g) in grp.iter().enumerate() {
        match group_pos.get(&g) {
            Some(&p) => groups[p].1.push(r),
            None => {
                group_pos.insert(g, groups.len());
                groups.push((g, vec![r]));
            }
        }
    }

    for (_g, batch_rows) in groups.iter() {
        let section_idx = sec[batch_rows[0]];
        for &r in batch_rows.iter() {
            if sec[r] != section_idx {
                return Err(format!(
                    "all rows of a root_groups group must share \
                     root_sections; group spans section {section_idx} and {}",
                    sec[r]
                ));
            }
        }
        let sampled: Vec<i64> = batch_rows.iter().map(|&r| smp[r]).collect();

        // --- subset emission pass (mirrors `_bfs_emit`) ------------------
        bfs_emit(
            adjacency,
            cols,
            consts,
            &mut decider,
            section_idx,
            &sampled,
            batch_rows,
            max_depth,
            root_edge_type,
            &mut emissions,
        )?;

        // --- full-variant-set pass for the pool (mirrors
        //     `_bfs_full_included` + the per-row setdiff) -----------------
        if need_excluded_pool {
            let (included_full, full_edge_type) = bfs_full_included(
                adjacency,
                cols,
                consts,
                &mut decider,
                section_idx,
                max_depth,
            )?;
            for &r in batch_rows.iter() {
                // pool = full-set-included MINUS this row's emitted, ascending
                // unique; pool_types = full_edge_type[pool] (parallel lookup).
                let (pool, pool_types) =
                    diff_pool(&included_full, &full_edge_type, &emissions[r].nodes);
                pools[r] = (pool, pool_types);
            }
        }
    }

    // Assemble per-row CSR in row order.
    let mut emitted_offsets = Vec::with_capacity(n_rows + 1);
    let mut emitted_nodes: Vec<i64> = Vec::new();
    let mut emitted_types: Vec<u8> = Vec::new();
    let mut pool_offsets = Vec::with_capacity(n_rows + 1);
    let mut pool_nodes: Vec<i64> = Vec::new();
    let mut pool_types: Vec<u8> = Vec::new();
    emitted_offsets.push(0);
    pool_offsets.push(0);
    for r in 0..n_rows {
        emitted_nodes.extend_from_slice(&emissions[r].nodes);
        emitted_types.extend_from_slice(&emissions[r].types);
        emitted_offsets.push(emitted_nodes.len() as i64);
        pool_nodes.extend_from_slice(&pools[r].0);
        pool_types.extend_from_slice(&pools[r].1);
        pool_offsets.push(pool_nodes.len() as i64);
    }

    Ok(InclusionResult {
        emitted_offsets,
        emitted_nodes,
        emitted_types,
        pool_offsets,
        pool_nodes,
        pool_types,
    })
}

/// Subset emission BFS for one root group (mirrors `_bfs_emit`): seed each
/// sampled row's root node, then per level resolve the frontier (Stage-1
/// `resolve_frontier`), decide inclusion (Stage-2 `step_level`), append the
/// included callees to their output row's emission, and descend the
/// survivors. `root_edge_type` is the wire `CallTargetType.LOCAL` the Python
/// `_ROOT_EDGE_TYPE` pins (threaded, not restated in Rust).
#[allow(clippy::too_many_arguments)]
fn bfs_emit(
    adjacency: &LiveAdjacencyKernel,
    cols: &Columns,
    consts: &GateConstants,
    decider: &mut DeciderState,
    section_idx: i64,
    sampled: &[i64],
    batch_rows: &[usize],
    max_depth: i64,
    root_edge_type: u8,
    emissions: &mut [RowEmission],
) -> Result<(), String> {
    let n_sampled = sampled.len();
    let v0 = cols.var_offsets[section_idx as usize];
    // FID for begin_root is the catalog section index (the decider keys on
    // child_secs, which are section indices) — matches the Python
    // `decider.begin_root(max(1, n_sampled), section_idx)`.
    decider.begin_root(n_sampled.max(1), section_idx as u32);

    // Seed each sampled row's root node (emission slot 0).
    for (i, &r) in batch_rows.iter().enumerate() {
        emissions[r].nodes.push(v0 + sampled[i]);
        emissions[r].types.push(root_edge_type);
    }

    // Level-0 frontier: one parent per sampled row, expanding its own node.
    // `parent_local` is the SUBSET mask row (0..n_sampled), parallel to the
    // decider's mask rows; map it to the output row via `batch_rows`.
    let mut parent_local: Vec<i64> = (0..n_sampled as i64).collect();
    let mut parent_node: Vec<i64> =
        sampled.iter().map(|&s| v0 + s).collect();

    let mut depth = 1;
    while depth <= max_depth {
        if parent_node.is_empty() {
            break;
        }
        let edges = adjacency.resolve_frontier(&parent_node, cols, consts)?;
        if edges.is_empty() {
            break;
        }
        // rows = parent_local[parent_pos]; fids = child_secs.
        let level_rows: Vec<i64> = edges
            .iter()
            .map(|e| parent_local[e.parent_pos as usize])
            .collect();
        let fids: Vec<u32> = edges.iter().map(|e| e.child_sec).collect();
        let (included, survivor_pairs) = decider.step_level(&level_rows, &fids)?;

        // Emit included pairs in level (pair) order: append to each pair's
        // OUTPUT row. The subset mask row -> output row map is batch_rows.
        for (i, &inc) in included.iter().enumerate() {
            if inc {
                let mask_row = level_rows[i] as usize;
                let out_r = batch_rows[mask_row];
                emissions[out_r].nodes.push(edges[i].child_node);
                emissions[out_r].types.push(edges[i].child_type);
            }
        }

        // Next frontier = survivors (their child_node + their mask row).
        let mut next_local: Vec<i64> = Vec::with_capacity(survivor_pairs.len());
        let mut next_node: Vec<i64> = Vec::with_capacity(survivor_pairs.len());
        for &p in survivor_pairs.iter() {
            next_local.push(level_rows[p as usize]);
            next_node.push(edges[p as usize].child_node);
        }
        parent_local = next_local;
        parent_node = next_node;
        depth += 1;
    }
    Ok(())
}

/// Full-variant-set BFS: the ascending-unique included callee node set + the
/// per-node first-inclusion-wins edge type (mirrors `_bfs_full_included`).
/// Returns `(included_nodes_sorted_unique, node_edge_type)` where
/// `node_edge_type` is keyed by global catalog node index.
fn bfs_full_included(
    adjacency: &LiveAdjacencyKernel,
    cols: &Columns,
    consts: &GateConstants,
    decider: &mut DeciderState,
    section_idx: i64,
    max_depth: i64,
) -> Result<(Vec<i64>, Vec<u8>), String> {
    let n_nodes = *cols.var_offsets.last().unwrap_or(&0) as usize;
    let mut node_edge_type = vec![0u8; n_nodes];
    let sec_u = section_idx as usize;
    let n_variants = cols.var_offsets[sec_u + 1] - cols.var_offsets[sec_u];
    if n_variants <= 0 {
        return Ok((Vec::new(), node_edge_type));
    }
    let v0 = cols.var_offsets[sec_u];
    decider.begin_root(n_variants as usize, section_idx as u32);

    let mut parent_local: Vec<i64> = (0..n_variants).collect();
    let mut parent_node: Vec<i64> = (0..n_variants).map(|j| v0 + j).collect();
    let mut included: Vec<i64> = Vec::new();
    let mut seen = vec![false; n_nodes];

    let mut depth = 1;
    while depth <= max_depth {
        if parent_node.is_empty() {
            break;
        }
        let edges = adjacency.resolve_frontier(&parent_node, cols, consts)?;
        if edges.is_empty() {
            break;
        }
        let level_rows: Vec<i64> = edges
            .iter()
            .map(|e| parent_local[e.parent_pos as usize])
            .collect();
        let fids: Vec<u32> = edges.iter().map(|e| e.child_sec).collect();
        let (inc, survivor_pairs) = decider.step_level(&level_rows, &fids)?;

        for (i, &is_inc) in inc.iter().enumerate() {
            if is_inc {
                let node = edges[i].child_node;
                included.push(node);
                // First-inclusion-wins per node, scanning the level in pair
                // order so the earliest BFS edge is the recorded type.
                let nu = node as usize;
                if !seen[nu] {
                    seen[nu] = true;
                    node_edge_type[nu] = edges[i].child_type;
                }
            }
        }

        let mut next_local: Vec<i64> = Vec::with_capacity(survivor_pairs.len());
        let mut next_node: Vec<i64> = Vec::with_capacity(survivor_pairs.len());
        for &p in survivor_pairs.iter() {
            next_local.push(level_rows[p as usize]);
            next_node.push(edges[p as usize].child_node);
        }
        parent_local = next_local;
        parent_node = next_node;
        depth += 1;
    }

    // np.unique: ascending + de-duplicated.
    included.sort_unstable();
    included.dedup();
    Ok((included, node_edge_type))
}

/// `setdiff1d(included_full, emitted)` ascending-unique + the parallel
/// `full_edge_type[pool]` gather (mirrors the per-row pool flatten). The
/// `emitted` slice is the row's emitted nodes (root first); the pool drops
/// anything the row already emitted. `included_full` is already sorted +
/// unique, so a membership set over `emitted` is the byte-identical diff.
fn diff_pool(
    included_full: &[i64],
    full_edge_type: &[u8],
    emitted: &[i64],
) -> (Vec<i64>, Vec<u8>) {
    if included_full.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let emitted_set: std::collections::HashSet<i64> =
        emitted.iter().copied().collect();
    let mut pool: Vec<i64> = Vec::new();
    let mut pool_types: Vec<u8> = Vec::new();
    for &node in included_full.iter() {
        if !emitted_set.contains(&node) {
            pool.push(node);
            pool_types.push(full_edge_type[node as usize]);
        }
    }
    (pool, pool_types)
}

#[cfg(test)]
mod tests {
    //! The fused kernel's correctness is gated end-to-end by the Python
    //! byte-identity suites (it must reproduce `compute_row_inclusions`
    //! exactly). These in-crate tests pin the fusion-specific helper whose
    //! logic is local to this module (the per-row pool diff), plus an
    //! adversarial mutation proving the diff is load-bearing. The BFS drive
    //! itself is covered by reusing the Stage-1 / Stage-2 cores, each with
    //! their own adversarial unit tests.

    use super::*;

    #[test]
    fn pool_diff_drops_emitted_and_gathers_types() {
        // full-set included = nodes [2,3,5,7] with edge types stamped.
        let included = vec![2i64, 3, 5, 7];
        let mut edge_type = vec![0u8; 8];
        edge_type[2] = 1;
        edge_type[3] = 2;
        edge_type[5] = 1;
        edge_type[7] = 2;
        // row emitted [root=0, 3, 7]; pool = [2,5] with their types.
        let emitted = vec![0i64, 3, 7];
        let (pool, types) = diff_pool(&included, &edge_type, &emitted);
        assert_eq!(pool, vec![2, 5]);
        assert_eq!(types, vec![1u8, 1u8]);
    }

    #[test]
    fn pool_empty_when_all_emitted() {
        let included = vec![2i64, 3];
        let edge_type = vec![0u8; 4];
        let emitted = vec![0i64, 2, 3];
        let (pool, types) = diff_pool(&included, &edge_type, &emitted);
        assert!(pool.is_empty());
        assert!(types.is_empty());
    }

    #[test]
    fn pool_empty_when_no_full_inclusion() {
        let (pool, types) = diff_pool(&[], &[], &[0i64, 1]);
        assert!(pool.is_empty());
        assert!(types.is_empty());
    }

    /// Adversarial: a pool diff that FAILS to drop the row's emitted nodes
    /// would re-inline already-emitted callees (a double-emit bug). Prove
    /// the membership filter changes the output.
    #[test]
    fn adversarial_skipping_emitted_filter_changes_output() {
        let included = vec![2i64, 3, 5];
        let edge_type = vec![0u8; 6];
        let emitted = vec![0i64, 3];
        let (good, _t) = diff_pool(&included, &edge_type, &emitted);
        // BROKEN: no emitted filter -> the whole full set leaks into the pool.
        let broken: Vec<i64> = included.clone();
        assert_ne!(good, broken);
        assert_eq!(good, vec![2, 5]);
    }
}
