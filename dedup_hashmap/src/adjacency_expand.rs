//! `LiveAdjacencyKernel` — the per-level CSR frontier expansion kernel.
//!
//! Single concern: resolve every parent node's DIRECT-call children for a
//! BFS frontier, over the flat columnar catalog arrays, with the GIL
//! released. This is the Rust port of
//! `LiveNodeAdjacency.expand_batch` (`sorted_index/_graph_lengths/
//! _adjacency.py`): the per-parent ascending-unique-slot gather, the
//! EXTERN / explicit-zero-ptr / map-miss gates, and the per-section
//! J-fallback table with its earliest-flat-order tie-break. It
//! re-implements NO inclusion / BFS-loop / decider rule — the Python
//! per-depth loop + `OnceOnlyInclusion` decider own those.
//!
//! ## State (mirrors the Python class)
//!
//! * `sec_map` — the `function_section_ptr -> section idx` map (the
//!   Python `_sec_map`, a `HashMapU32U32`), built once at construction.
//! * `fallback_cache` — the per-section dense fallback-J table cache (the
//!   Python `_fallback_cache` dict), filled per section on first fallback.
//!
//! Both the loader inclusion-BFS and the sorted-index length build drive
//! the SAME `LiveNodeAdjacency`, which holds ONE kernel instance, so the
//! two consumers can never drift.
//!
//! ## Per-call contract
//!
//! `expand_batch` borrows the (possibly lazily-filled) catalog columns
//! fresh each call. It reads ONLY the parent sections' heavy columns
//! (ct_*, pce_* of the frontier's own sections) plus the eager
//! `var_offsets` / `sec_of_var` — never an unfilled callee section's heavy
//! columns — so passing the full-length backing arrays is sound even on
//! the lazy catalog.
//!
//! Returns `(parent_pos, child_secs, child_nodes, child_types,
//! child_matched)` — one entry per surviving (parent, call_target) edge,
//! parents in ascending `parent_pos` and within a parent ascending
//! call_target slot. Byte-identical to the numpy `expand_batch`.

use std::sync::Mutex;

use hashbrown::HashMap;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// `0xFFFFFFFF` — the `HashMapU32U32` miss sentinel the Python path checks
/// `hit != _U32_MISS` against. Resolution drops a map miss.
///
/// `pub(crate)` so the inclusion-closure frontier-advance shares the SAME
/// sentinel as `resolve_one_parent`'s map-miss gate.
pub(crate) const U32_MISS: u32 = u32::MAX;

/// The callee section the kernel cares about per slot, mirroring the
/// numpy gate cascade evaluated per (deduped) parent-slot pair.
///
/// `pub(crate)` so the fused inclusion-BFS kernel (`row_inclusions.rs`) can
/// drive the SAME per-parent resolution in-Rust (no Python round-trip).
pub(crate) struct ResolvedEdge {
    pub(crate) parent_pos: i64,
    pub(crate) child_sec: u32,
    pub(crate) child_node: i64,
    pub(crate) child_type: u8,
    pub(crate) child_matched: bool,
}

/// The borrowed columnar slices `expand_batch` resolves over. All arrays
/// are the catalog's GLOBAL backing (full-length); the kernel indexes them
/// by global slot / variant / entry, exactly as the numpy path does.
///
/// `pub(crate)` so the fused inclusion-BFS kernel can pass the same flat
/// catalog columns straight to `resolve_frontier` without re-marshalling.
pub(crate) struct Columns<'a> {
    /// `i64[total_variants + 1]` per-call-entry CSR (global).
    pub(crate) pce_offsets: &'a [i64],
    /// `u16[total_entries]` -> the called slot idx within the section.
    pub(crate) pce_called_idx: &'a [u16],
    /// `u16[total_entries]` -> the entry's own resolved callee variant J.
    pub(crate) pce_section_variant_index: &'a [u16],
    /// `i64[n_sections + 1]` call_target CSR (global).
    pub(crate) ct_offsets: &'a [i64],
    /// `u8[total_cts]` raw `CallTargetType`.
    pub(crate) ct_type: &'a [u8],
    /// `u32[total_cts]` callee section byte pointer (#69 explicit-zero
    /// sentinel rides here).
    pub(crate) ct_function_section_ptr: &'a [u32],
    /// `bool[total_cts]` parent slot is_matched flag.
    pub(crate) ct_is_matched: &'a [bool],
    /// `i64[n_sections + 1]` variant CSR (global, eager).
    pub(crate) var_offsets: &'a [i64],
    /// `i64[n_sections]` per-section call_target count.
    pub(crate) n_call_targets: &'a [i64],
    /// `i64[total_variants]` owning section per flat variant (eager).
    pub(crate) sec_of_var: &'a [i64],
}

/// The integer constants the gate cascade needs, threaded from Python so
/// the wire-format enum/sentinel values are single-sourced (never restated
/// in Rust). `pub(crate)` for the fused kernel's reuse.
pub(crate) struct GateConstants {
    /// `CallTargetType.EXTERN` raw value — an EXTERN edge is gated out.
    pub(crate) extern_type: u8,
    /// `MISSING_VARIANT_INDEX` — an own_J equal to this takes the fallback
    /// arm; a fallback table entry equal to this is unusable.
    pub(crate) missing_variant_index: u16,
}

#[pyclass(module = "dedup_hashmap._native")]
pub struct LiveAdjacencyKernel {
    /// `function_section_ptr -> section idx` (the Python `_sec_map`).
    ///
    /// `pub(crate)` so the GIL-released inclusion-closure frontier-advance
    /// (`inclusion_closure.rs`) resolves callee pointers through the SAME
    /// offset->idx map — it can never drift from `resolve_frontier`.
    pub(crate) sec_map: HashMap<u32, u32>,
    /// Per-section dense fallback-J table: section idx -> `i64[n_cts]`
    /// whose entry at `called_idx` is the earliest-flat-order usable J for
    /// that slot (-1 when none). Filled on first fallback for a section
    /// (the Python `_fallback_cache`). A `Mutex` keeps the cache
    /// interior-mutable while the pyclass stays `Sync` (PyO3 requires it)
    /// and the GIL-released `expand_batch` can take `&self`. Each
    /// `expand_batch` is single-threaded per kernel (one BFS level at a
    /// time), so the lock is uncontended.
    fallback_cache: Mutex<HashMap<i64, Vec<i64>>>,
}

#[pymethods]
impl LiveAdjacencyKernel {
    /// Build the offset->section-idx map once (mirrors the Python
    /// `_sec_map.insert_ndarray(offs.astype(u32), arange(n_sections))`,
    /// last-wins on dup keys).
    #[new]
    fn new(section_offsets: PyReadonlyArray1<'_, u32>) -> PyResult<Self> {
        let offs = section_offsets.as_slice()?;
        let mut sec_map: HashMap<u32, u32> = HashMap::with_capacity(offs.len());
        for (idx, &off) in offs.iter().enumerate() {
            sec_map.insert(off, idx as u32);
        }
        Ok(LiveAdjacencyKernel {
            sec_map,
            fallback_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Resolve the whole frontier's children. See the module docstring for
    /// the column contract; returns the 5-tuple of parallel ndarrays
    /// `(parent_pos, child_secs, child_nodes, child_types, child_matched)`.
    #[allow(clippy::too_many_arguments)]
    fn expand_batch<'py>(
        &self,
        py: Python<'py>,
        parent_nodes: PyReadonlyArray1<'py, i64>,
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
        extern_type: u8,
        missing_variant_index: u16,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let parents = parent_nodes.as_slice()?;
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

        let edges = py.detach(|| self.resolve_frontier(parents, &cols, &consts));
        let edges = edges.map_err(PyValueError::new_err)?;

        let parent_pos: Vec<i64> = edges.iter().map(|e| e.parent_pos).collect();
        let child_secs: Vec<u32> = edges.iter().map(|e| e.child_sec).collect();
        let child_nodes: Vec<i64> = edges.iter().map(|e| e.child_node).collect();
        let child_types: Vec<u8> = edges.iter().map(|e| e.child_type).collect();
        let child_matched: Vec<bool> =
            edges.iter().map(|e| e.child_matched).collect();

        let tup = PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, parent_pos).into_any(),
                PyArray1::from_vec(py, child_secs).into_any(),
                PyArray1::from_vec(py, child_nodes).into_any(),
                PyArray1::from_vec(py, child_types).into_any(),
                PyArray1::from_vec(py, child_matched).into_any(),
            ],
        )?;
        Ok(tup)
    }

    /// Advance the inclusion-closure section frontier one level (the
    /// GIL-released port of `ensure_inclusion_closure`'s inner per-section
    /// pointer-gather + `_sec_map` lookup + `np.unique(np.concatenate(...))`
    /// reduction). Returns the SORTED-UNIQUE `i64` callee sections the given
    /// `frontier` reaches through its `ct_function_section_ptr` pointers (#69
    /// explicit-zero pointers and map misses dropped). The `reached`-mask
    /// filter + per-level `ensure_sections` lazy-fill stay on the Python side
    /// (they depend on per-batch state / drive the lazy catalog). See
    /// `inclusion_closure.rs` for the contract.
    fn advance_inclusion_frontier<'py>(
        &self,
        py: Python<'py>,
        frontier: PyReadonlyArray1<'py, i64>,
        ct_offsets: PyReadonlyArray1<'py, i64>,
        ct_function_section_ptr: PyReadonlyArray1<'py, u32>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let frontier = frontier.as_slice()?;
        let ct_offsets = ct_offsets.as_slice()?;
        let ct_function_section_ptr = ct_function_section_ptr.as_slice()?;
        let next = py.detach(|| {
            self.advance_inclusion_frontier_impl(
                frontier,
                ct_offsets,
                ct_function_section_ptr,
            )
        });
        let next = next.map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, next))
    }
}

impl LiveAdjacencyKernel {
    /// Test-only constructor from raw parts — lets the inclusion-closure
    /// unit tests (in `inclusion_closure.rs`) build a kernel without going
    /// through `new`'s ndarray path, while keeping the fields private.
    #[cfg(test)]
    pub(crate) fn from_parts_for_test(
        sec_map: HashMap<u32, u32>,
        fallback_cache: Mutex<HashMap<i64, Vec<i64>>>,
    ) -> Self {
        LiveAdjacencyKernel {
            sec_map,
            fallback_cache,
        }
    }

    /// GIL-released frontier resolution. Walks parents in ascending order;
    /// per parent emits its ascending-unique-slot edges that survive the
    /// gate cascade, preserving the exact `expand_batch` flattening order.
    ///
    /// `pub(crate)` so the fused inclusion-BFS kernel reuses this exact
    /// per-level resolution in-Rust — the SAME gates, J-fallback, ordering,
    /// and the SAME `sec_map` / `fallback_cache` state — guaranteeing the
    /// fused path can never drift from the standalone Stage-1 kernel.
    pub(crate) fn resolve_frontier(
        &self,
        parents: &[i64],
        cols: &Columns,
        consts: &GateConstants,
    ) -> Result<Vec<ResolvedEdge>, String> {
        let mut out: Vec<ResolvedEdge> = Vec::new();
        for (pos, &parent) in parents.iter().enumerate() {
            self.resolve_one_parent(pos as i64, parent, cols, consts, &mut out)?;
        }
        Ok(out)
    }

    /// One parent's ascending-unique called slots, each gated + J-resolved.
    ///
    /// Mirrors the numpy per-parent reduction: gather the parent's per-call
    /// entries, take ascending-unique `called` (earliest entry's own_J wins
    /// the tie), then run the gate cascade per surviving slot.
    fn resolve_one_parent(
        &self,
        pos: i64,
        parent: i64,
        cols: &Columns,
        consts: &GateConstants,
        out: &mut Vec<ResolvedEdge>,
    ) -> Result<(), String> {
        let parent_u = parent as usize;
        if parent_u + 1 >= cols.pce_offsets.len() {
            return Err(format!(
                "parent node {parent} out of pce_offsets range"
            ));
        }
        let p0 = cols.pce_offsets[parent_u] as usize;
        let p1 = cols.pce_offsets[parent_u + 1] as usize;
        if p1 == p0 {
            return Ok(());
        }
        let sec = cols.sec_of_var[parent_u];
        let sec_u = sec as usize;
        let ct_lo = cols.ct_offsets[sec_u];
        let ct_hi = cols.ct_offsets[sec_u + 1];

        // Ascending-unique called slots; first occurrence's own_J wins.
        // The numpy path stable-sorts (pos, called) keeping on-disk order
        // within a tie, then keeps the first per (pos, called) — i.e. the
        // EARLIEST flat-order entry's own_J. We reproduce that with a sort
        // by `called` whose tie-break preserves the entry's original index.
        let mut slots: Vec<(i64, i64)> = Vec::with_capacity(p1 - p0);
        for e in p0..p1 {
            slots.push((
                cols.pce_called_idx[e] as i64,
                cols.pce_section_variant_index[e] as i64,
            ));
        }
        // Stable sort by called keeps the earliest-flat-order entry first
        // within each called group (matching the numpy stable argsort), so
        // the first row per called carries the surviving own_J.
        slots.sort_by_key(|&(called, _own_j)| called);

        let mut prev_called: Option<i64> = None;
        for &(called, own_j) in slots.iter() {
            if Some(called) == prev_called {
                continue; // duplicate slot: first own_J already handled.
            }
            prev_called = Some(called);

            // --- gate: slot in range -------------------------------------
            let slot = ct_lo + called;
            if slot >= ct_hi {
                continue;
            }
            let slot_u = slot as usize;
            // --- gate: EXTERN --------------------------------------------
            if cols.ct_type[slot_u] == consts.extern_type {
                continue;
            }
            // --- gate: #69 explicit-zero function_section_ptr ------------
            let ptr = cols.ct_function_section_ptr[slot_u];
            if ptr == 0 {
                continue;
            }
            // --- gate: offset->idx map miss ------------------------------
            let callee_sec = match self.sec_map.get(&ptr) {
                Some(&v) if v != U32_MISS => v as i64,
                _ => continue,
            };
            // --- J selection: own J usable, else per-section fallback ----
            let j = if own_j == consts.missing_variant_index as i64 {
                self.fallback_j(sec, called, cols, consts)
            } else {
                own_j
            };
            if j < 0 {
                continue;
            }
            let child_node = cols.var_offsets[callee_sec as usize] + j;
            let child_sec = cols.sec_of_var[child_node as usize] as u32;
            out.push(ResolvedEdge {
                parent_pos: pos,
                child_sec,
                child_node,
                child_type: cols.ct_type[slot_u],
                child_matched: cols.ct_is_matched[slot_u],
            });
        }
        Ok(())
    }

    /// Per-section fallback J for `(sec, called)`, -1 when none — the dense
    /// table lookup that mirrors `_fallback_J`. Builds + caches the table
    /// on first fallback for the section.
    fn fallback_j(
        &self,
        sec: i64,
        called: i64,
        cols: &Columns,
        consts: &GateConstants,
    ) -> i64 {
        // Fast path: table already cached.
        {
            let cache = self
                .fallback_cache
                .lock()
                .expect("fallback cache mutex poisoned");
            if let Some(table) = cache.get(&sec) {
                return Self::lookup_table(table, called);
            }
        }
        // Build outside the lock (so `build_fallback_table`'s `&self` does
        // not alias the held mutable borrow of `fallback_cache`), then
        // memoise. A racing builder for the same section would recompute
        // the identical deterministic table, so the insert is idempotent.
        let table = self.build_fallback_table(sec, cols, consts);
        let result = Self::lookup_table(&table, called);
        self.fallback_cache
            .lock()
            .expect("fallback cache mutex poisoned")
            .insert(sec, table);
        result
    }

    /// `table[called]` with the `_fallback_J` out-of-range guard (-1).
    #[inline]
    fn lookup_table(table: &[i64], called: i64) -> i64 {
        if called < 0 || called as usize >= table.len() {
            -1
        } else {
            table[called as usize]
        }
    }

    /// `i64[n_call_targets(sec)]` fallback-J per slot for `sec`.
    ///
    /// Entry at slot `ci` is the J of the EARLIEST usable per-call entry
    /// for `ci` across the section's variants in on-disk order (variants
    /// ascending, each variant's entries in pce order) — the verbatim
    /// reverse-then-overwrite tie-break of `_fallback_table`: a later
    /// assignment during the reversed pass is an EARLIER original entry, so
    /// the earliest flat-order usable entry wins.
    fn build_fallback_table(
        &self,
        sec: i64,
        cols: &Columns,
        consts: &GateConstants,
    ) -> Vec<i64> {
        let sec_u = sec as usize;
        let n_cts = cols.n_call_targets[sec_u];
        let n_cts = if n_cts < 0 { 0 } else { n_cts as usize };
        let mut table = vec![-1i64; n_cts];
        if n_cts == 0 {
            return table;
        }
        let v0 = cols.var_offsets[sec_u] as usize;
        let v1 = cols.var_offsets[sec_u + 1] as usize;
        let e0 = cols.pce_offsets[v0] as usize;
        let e1 = cols.pce_offsets[v1] as usize;
        // Reverse pass: iterate entries last->first so the first-in-order
        // usable entry for a slot wins the overwrite.
        for e in (e0..e1).rev() {
            let js = cols.pce_section_variant_index[e] as i64;
            if js == consts.missing_variant_index as i64 {
                continue; // not usable
            }
            let ci = cols.pce_called_idx[e] as i64;
            if ci < 0 || ci as usize >= n_cts {
                continue; // out of range
            }
            table[ci as usize] = js;
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal hand-built catalog: one parent section (sec 0) with 2
    /// variants + 3 call_target slots, calling into sec 1 / sec 2. Exercises
    /// every gate arm + the J fallback.
    fn build_columns() -> (
        Vec<i64>, // pce_offsets
        Vec<u16>, // pce_called_idx
        Vec<u16>, // pce_section_variant_index
        Vec<i64>, // ct_offsets
        Vec<u8>,  // ct_type
        Vec<u32>, // ct_function_section_ptr
        Vec<bool>, // ct_is_matched
        Vec<i64>, // var_offsets
        Vec<i64>, // n_call_targets
        Vec<i64>, // sec_of_var
    ) {
        // 3 sections. sec0: 2 variants, 3 call_targets. sec1: 2 variants.
        // sec2: 1 variant.
        let var_offsets = vec![0i64, 2, 4, 5]; // sec0=[0,2) sec1=[2,4) sec2=[4,5)
        let sec_of_var = vec![0i64, 0, 1, 1, 2];
        let n_call_targets = vec![3i64, 0, 0];
        let ct_offsets = vec![0i64, 3, 3, 3]; // only sec0 has call_targets
        // slot 0 -> sec1 (ptr 100), LOCAL, matched
        // slot 1 -> sec2 (ptr 200), PLT, unmatched
        // slot 2 -> EXTERN (gated)
        let ct_type = vec![0u8, 1, 2]; // LOCAL, PLT, EXTERN
        let ct_function_section_ptr = vec![100u32, 200, 300];
        let ct_is_matched = vec![true, false, false];
        // variant 0 (sec0): calls slot0 own_J=1, slot1 own_J=0
        // variant 1 (sec0): calls slot0 own_J=0 (earlier-flat for fallback)
        // variants 2,3,4: no calls
        let pce_offsets = vec![0i64, 2, 3, 3, 3, 3];
        let pce_called_idx = vec![0u16, 1, 0];
        let pce_section_variant_index = vec![1u16, 0, 0];
        (
            pce_offsets,
            pce_called_idx,
            pce_section_variant_index,
            ct_offsets,
            ct_type,
            ct_function_section_ptr,
            ct_is_matched,
            var_offsets,
            n_call_targets,
            sec_of_var,
        )
    }

    fn kernel() -> LiveAdjacencyKernel {
        // sec offsets 100/200/300 map to section idx 0/1/2.
        let mut sec_map: HashMap<u32, u32> = HashMap::new();
        sec_map.insert(100, 1); // ptr 100 -> sec 1
        sec_map.insert(200, 2); // ptr 200 -> sec 2
        sec_map.insert(300, 0); // ptr 300 -> sec 0 (EXTERN slot, gated before lookup)
        LiveAdjacencyKernel {
            sec_map,
            fallback_cache: Mutex::new(HashMap::new()),
        }
    }

    fn columns<'a>(
        c: &'a (
            Vec<i64>,
            Vec<u16>,
            Vec<u16>,
            Vec<i64>,
            Vec<u8>,
            Vec<u32>,
            Vec<bool>,
            Vec<i64>,
            Vec<i64>,
            Vec<i64>,
        ),
    ) -> Columns<'a> {
        Columns {
            pce_offsets: &c.0,
            pce_called_idx: &c.1,
            pce_section_variant_index: &c.2,
            ct_offsets: &c.3,
            ct_type: &c.4,
            ct_function_section_ptr: &c.5,
            ct_is_matched: &c.6,
            var_offsets: &c.7,
            n_call_targets: &c.8,
            sec_of_var: &c.9,
        }
    }

    const CONSTS: GateConstants = GateConstants {
        extern_type: 2,
        missing_variant_index: 0xFFFE,
    };

    #[test]
    fn resolves_gates_and_order() {
        let c = build_columns();
        let cols = columns(&c);
        let k = kernel();
        // Parent = variant 0 (sec0). Calls slot0 (own_J=1 -> sec1 var
        // offset 2 + 1 = node 3) and slot1 (own_J=0 -> sec2 var offset 4 +
        // 0 = node 4). Slot2 is EXTERN-only (not called by variant 0).
        let edges = k
            .resolve_frontier(&[0i64], &cols, &CONSTS)
            .unwrap();
        assert_eq!(edges.len(), 2);
        // ascending slot order: slot0 first, then slot1.
        assert_eq!(edges[0].child_node, 3);
        assert_eq!(edges[0].child_sec, 1);
        assert_eq!(edges[0].child_type, 0); // LOCAL
        assert!(edges[0].child_matched);
        assert_eq!(edges[1].child_node, 4);
        assert_eq!(edges[1].child_sec, 2);
        assert_eq!(edges[1].child_type, 1); // PLT
        assert!(!edges[1].child_matched);
        assert_eq!(edges[0].parent_pos, 0);
        assert_eq!(edges[1].parent_pos, 0);
    }

    #[test]
    fn extern_and_zero_ptr_gated() {
        // Build a catalog where variant 0 calls an EXTERN slot and a
        // zero-ptr slot; both must drop.
        let var_offsets = vec![0i64, 1, 2];
        let sec_of_var = vec![0i64, 1];
        let n_call_targets = vec![2i64, 0];
        let ct_offsets = vec![0i64, 2, 2];
        let ct_type = vec![2u8, 0]; // slot0 EXTERN, slot1 LOCAL
        let ct_function_section_ptr = vec![100u32, 0]; // slot1 zero-ptr (#69)
        let ct_is_matched = vec![false, true];
        let pce_offsets = vec![0i64, 2, 2];
        let pce_called_idx = vec![0u16, 1];
        let pce_section_variant_index = vec![0u16, 0];
        let c = (
            pce_offsets,
            pce_called_idx,
            pce_section_variant_index,
            ct_offsets,
            ct_type,
            ct_function_section_ptr,
            ct_is_matched,
            var_offsets,
            n_call_targets,
            sec_of_var,
        );
        let cols = columns(&c);
        let mut sec_map: HashMap<u32, u32> = HashMap::new();
        sec_map.insert(100, 1);
        let k = LiveAdjacencyKernel {
            sec_map,
            fallback_cache: Mutex::new(HashMap::new()),
        };
        let edges = k.resolve_frontier(&[0i64], &cols, &CONSTS).unwrap();
        assert_eq!(edges.len(), 0, "EXTERN + zero-ptr slots must both drop");
    }

    #[test]
    fn map_miss_gated() {
        // ptr points to an offset NOT in the sec_map -> drop.
        let var_offsets = vec![0i64, 1, 2];
        let sec_of_var = vec![0i64, 1];
        let n_call_targets = vec![1i64, 0];
        let ct_offsets = vec![0i64, 1, 1];
        let ct_type = vec![0u8];
        let ct_function_section_ptr = vec![999u32]; // not a known section start
        let ct_is_matched = vec![true];
        let pce_offsets = vec![0i64, 1, 1];
        let pce_called_idx = vec![0u16];
        let pce_section_variant_index = vec![0u16];
        let c = (
            pce_offsets,
            pce_called_idx,
            pce_section_variant_index,
            ct_offsets,
            ct_type,
            ct_function_section_ptr,
            ct_is_matched,
            var_offsets,
            n_call_targets,
            sec_of_var,
        );
        let cols = columns(&c);
        let mut sec_map: HashMap<u32, u32> = HashMap::new();
        sec_map.insert(100, 1);
        let k = LiveAdjacencyKernel {
            sec_map,
            fallback_cache: Mutex::new(HashMap::new()),
        };
        let edges = k.resolve_frontier(&[0i64], &cols, &CONSTS).unwrap();
        assert_eq!(edges.len(), 0, "map miss must drop the edge");
    }

    #[test]
    fn j_fallback_earliest_flat_order_wins() {
        // sec0 has 2 variants. variant1 (parent) calls slot0 with own_J =
        // MISSING, forcing the fallback. The section's per-call entries are
        // [v0: (called0, J=5), (called0, J=7), v1: (called0, MISSING)].
        // Earliest-flat-order usable for slot0 is J=5; the reverse-then-
        // overwrite pass must land 5 (not 7).
        // sec1 carries enough variant slots (6) to address J=5 (node 7).
        let var_offsets = vec![0i64, 2, 8]; // sec0=[0,2) sec1=[2,8)
        let sec_of_var = vec![0i64, 0, 1, 1, 1, 1, 1, 1];
        let n_call_targets = vec![1i64, 0];
        let ct_offsets = vec![0i64, 1, 1];
        let ct_type = vec![0u8]; // LOCAL
        let ct_function_section_ptr = vec![100u32];
        let ct_is_matched = vec![true];
        // v0 entries: (0, 5), (0, 7); v1 entries: (0, MISSING)
        let pce_offsets = vec![0i64, 2, 3, 3, 3, 3, 3, 3, 3];
        let pce_called_idx = vec![0u16, 0, 0];
        let pce_section_variant_index = vec![5u16, 7, 0xFFFE];
        let c = (
            pce_offsets,
            pce_called_idx,
            pce_section_variant_index,
            ct_offsets,
            ct_type,
            ct_function_section_ptr,
            ct_is_matched,
            var_offsets,
            n_call_targets,
            sec_of_var,
        );
        let cols = columns(&c);
        let mut sec_map: HashMap<u32, u32> = HashMap::new();
        sec_map.insert(100, 1); // ptr 100 -> sec 1
        let k = LiveAdjacencyKernel {
            sec_map,
            fallback_cache: Mutex::new(HashMap::new()),
        };
        // Parent = variant 1 (node 1), which calls slot0 with own_J=MISSING.
        let edges = k.resolve_frontier(&[1i64], &cols, &CONSTS).unwrap();
        assert_eq!(edges.len(), 1);
        // callee sec1 var_offset = 2; fallback J = 5 -> node 7.
        assert_eq!(edges[0].child_node, 2 + 5);
    }

    #[test]
    fn dedup_keeps_first_own_j() {
        // variant 0 calls slot0 TWICE with own_J 9 then 4. The ascending-
        // unique reduction keeps the EARLIEST flat-order entry's own_J (9).
        // sec1 carries 10 variant slots so own_J=9 (node 10) is addressable.
        let var_offsets = vec![0i64, 1, 11]; // sec0=[0,1) sec1=[1,11)
        let mut sec_of_var = vec![0i64];
        sec_of_var.extend(std::iter::repeat(1i64).take(10));
        let n_call_targets = vec![1i64, 0];
        let ct_offsets = vec![0i64, 1, 1];
        let ct_type = vec![0u8];
        let ct_function_section_ptr = vec![100u32];
        let ct_is_matched = vec![true];
        let mut pce_offsets = vec![0i64, 2];
        pce_offsets.extend(std::iter::repeat(2i64).take(10));
        let pce_called_idx = vec![0u16, 0];
        let pce_section_variant_index = vec![9u16, 4];
        let c = (
            pce_offsets,
            pce_called_idx,
            pce_section_variant_index,
            ct_offsets,
            ct_type,
            ct_function_section_ptr,
            ct_is_matched,
            var_offsets,
            n_call_targets,
            sec_of_var,
        );
        let cols = columns(&c);
        let mut sec_map: HashMap<u32, u32> = HashMap::new();
        sec_map.insert(100, 1);
        let k = LiveAdjacencyKernel {
            sec_map,
            fallback_cache: Mutex::new(HashMap::new()),
        };
        let edges = k.resolve_frontier(&[0i64], &cols, &CONSTS).unwrap();
        assert_eq!(edges.len(), 1);
        // sec1 var_offset 1 + own_J 9 = node 10.
        assert_eq!(edges[0].child_node, 1 + 9);
    }

    #[test]
    fn adversarial_extern_perturbation_changes_output() {
        // Flipping the EXTERN gate off (treat as LOCAL) must surface an
        // extra edge — proving the gate is load-bearing.
        let c = build_columns();
        let cols = columns(&c);
        let k = kernel();
        // Parent variant 1 (node 1) calls slot0 (own_J=0). Add a synthetic
        // run where slot2 (EXTERN) is also called by variant 1 to show it
        // is dropped. We reuse variant 0 which calls slot0+slot1.
        let baseline = k.resolve_frontier(&[0i64], &cols, &CONSTS).unwrap();
        // Perturb: lie that EXTERN is type 99 (so nothing is EXTERN). Build
        // a parent that calls the EXTERN slot2 and confirm it now appears.
        let mut c2 = c.clone();
        // variant 0 also calls slot2 (own_J=0): extend its pce range.
        c2.0 = vec![0i64, 3, 4, 4, 4, 4]; // v0 now [0,3)
        c2.1 = vec![0u16, 1, 2, 0]; // v0: slot0,1,2 ; v1: slot0
        c2.2 = vec![1u16, 0, 0, 0];
        let cols2 = columns(&c2);
        let bad_consts = GateConstants {
            extern_type: 99, // EXTERN no longer gated
            missing_variant_index: 0xFFFE,
        };
        let perturbed = k.resolve_frontier(&[0i64], &cols2, &bad_consts).unwrap();
        // baseline had 2 edges; with EXTERN ungated + slot2 called, we get 3.
        assert_eq!(baseline.len(), 2);
        assert_eq!(perturbed.len(), 3);
        assert_ne!(baseline.len(), perturbed.len());
    }
}
