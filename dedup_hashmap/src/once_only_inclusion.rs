//! `OnceOnlyInclusionKernel` — the per-level once-only / columnwise-ALL
//! inclusion state machine.
//!
//! Single concern: decide, per splice level, which `(variant,
//! callee_function_id)` pairs INCLUDE the callee body and which SURVIVE to
//! the next level — the GIL-released Rust port of `OnceOnlyInclusion`
//! (`splice_inclusion/_state.py`): `begin_root` (root seeded at column 0)
//! and `step_level` (the columnwise-ALL FLAG-A rule, the first-occurrence
//! dedup, the FLAG-B read-before-write once-only test). It re-implements NO
//! adjacency / BFS-loop rule — the Python per-depth loop owns the traversal
//! and feeds this decider one level at a time, exactly as the numpy class
//! it replaces.
//!
//! ## State (mirrors the Python class, reused across roots)
//!
//! * `fid_to_col` — the `function_id -> dense column` map (the Python
//!   `_fid_to_col`, a `HashMapU32U32`), cleared per root by `begin_root`.
//! * `mask` — the flat row-major `[n_variants, n_cols]` inclusion mask (the
//!   Python `_mask`), grown geometrically and zeroed (never re-allocated)
//!   over its used region per root.
//!
//! Both the loader inclusion-BFS and the sorted-index length build drive
//! the SAME `OnceOnlyInclusion` Python class, which holds ONE kernel
//! instance, so the two consumers can never drift.
//!
//! ## The load-bearing per-level ordering (ported verbatim)
//!
//! Per `step_level`, in this exact order:
//!
//! 1. `_assign_columns`: insert-or-get a dense column per callee FID;
//!    genuinely-new FIDs take fresh ascending columns assigned in
//!    ASCENDING-FID order (the numpy `np.unique` sorts), deduped within the
//!    batch so repeats share a column.
//! 2. `pre_cell`: snapshot each pair's mask cell BEFORE marking (FLAG-B
//!    read-before-write — a variant reaching a converging column LATE reads
//!    a True pre-cell here and is excluded).
//! 3. `first_in_level`: True at the FIRST `(variant, col)` in emission
//!    order, False on later repeats within the level (earliest wins).
//! 4. mark `mask[variant, col] = True` for every pair.
//! 5. columnwise-ALL over ONLY the touched columns across every `n_variants`
//!    row: a column all-True is excluded (FLAG-A; a single-variant root's
//!    one row is trivially all-True so it splices nothing).
//! 6. `included = ~pre_cell & first_in_level & ~pair_excluded`.
//! 7. `survivor_pairs = nonzero(included)` in ascending pair index.
//!
//! Returns `(included, survivor_pairs)` — `bool[n_pairs]` index-aligned to
//! the caller's emission-order pair arrays + the ascending survivor index.
//! Byte-identical to the numpy `step_level`.

use std::sync::Mutex;

use hashbrown::HashMap;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// `0xFFFFFFFF` — the `HashMapU32U32` miss sentinel. A column lookup miss
/// means the FID has not yet been assigned a dense column.
const U32_MISS: u32 = u32::MAX;

/// The reusable per-root decider state (mirrors the Python instance
/// fields). Held behind a `Mutex` so the pyclass stays `Sync` (PyO3
/// requires it) while `begin_root` / `step_level` mutate it through
/// `&self` with the GIL released. Each `OnceOnlyInclusion` Python instance
/// owns ONE kernel and is driven single-threaded (one BFS level at a time),
/// so the lock is uncontended.
///
/// `pub(crate)` so the fused inclusion-BFS kernel (`row_inclusions.rs`) can
/// own a `DeciderState` directly and drive `begin_root` / `step_level` for
/// the whole BFS in-Rust — reusing the EXACT once-only / columnwise-ALL
/// state machine, so the fused path can never drift from the standalone
/// Stage-2 decider kernel.
pub(crate) struct DeciderState {
    /// `function_id -> dense column index`, cleared per root.
    fid_to_col: HashMap<u32, u32>,
    /// Flat row-major `[rows_cap, cols_cap]` inclusion mask. The used
    /// region is `[0, n_variants) x [0, n_cols)`; capacity grows
    /// geometrically and is never shrunk.
    mask: Vec<bool>,
    /// Allocated row capacity of `mask`.
    rows_cap: usize,
    /// Allocated column capacity of `mask`.
    cols_cap: usize,
    /// Active variant (mask row) count for the current root.
    n_variants: usize,
    /// Active column count for the current root (cumulative across levels).
    n_cols: usize,
}

impl DeciderState {
    pub(crate) fn new(initial_cols: usize) -> Self {
        let cols_cap = initial_cols.max(1);
        let rows_cap = 1usize;
        DeciderState {
            fid_to_col: HashMap::with_capacity((initial_cols * 2).max(8)),
            mask: vec![false; rows_cap * cols_cap],
            rows_cap,
            cols_cap,
            n_variants: 0,
            n_cols: 0,
        }
    }

    /// `mask[row, col]` flat index over the row-major buffer.
    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols_cap + col
    }

    /// Grow geometrically to fit `(n_variants, n_cols)`, preserving the
    /// already-marked region (mirrors `_ensure_capacity`). Doubling per
    /// deficient axis keeps amortised reallocation O(1) per added row/col.
    fn ensure_capacity(&mut self, n_variants: usize, n_cols: usize) {
        if n_variants <= self.rows_cap && n_cols <= self.cols_cap {
            return;
        }
        let mut new_rows = self.rows_cap;
        while new_rows < n_variants {
            new_rows *= 2;
        }
        let mut new_cols = self.cols_cap;
        while new_cols < n_cols {
            new_cols *= 2;
        }
        let mut grown = vec![false; new_rows * new_cols];
        // Copy the existing used rows into the wider stride.
        for r in 0..self.rows_cap {
            let src = r * self.cols_cap;
            let dst = r * new_cols;
            grown[dst..dst + self.cols_cap]
                .copy_from_slice(&self.mask[src..src + self.cols_cap]);
        }
        self.mask = grown;
        self.rows_cap = new_rows;
        self.cols_cap = new_cols;
    }

    /// Reset for a new root; seed the root body at column 0 for every
    /// variant (mirrors `begin_root`). The root's column is always 0.
    pub(crate) fn begin_root(&mut self, n_variants: usize, root_function_id: u32) {
        self.fid_to_col.clear();
        self.ensure_capacity(n_variants, 1);
        // Zero ONLY the previously-used region (the linearity guard pins
        // the per-root reset to the prior root's used span, never a global
        // high-water column count).
        for r in 0..self.n_variants {
            let base = r * self.cols_cap;
            for c in 0..self.n_cols {
                self.mask[base + c] = false;
            }
        }
        self.n_variants = n_variants;
        self.n_cols = 1;
        for r in 0..n_variants {
            let i = self.idx(r, 0);
            self.mask[i] = true;
        }
        self.fid_to_col.insert(root_function_id, 0);
    }

    /// Insert-or-get dense columns for `fids` (mirrors `_assign_columns`).
    ///
    /// Looks up each FID; genuinely-new FIDs take fresh ascending columns
    /// assigned in ASCENDING-FID order (the numpy `np.unique` sorts the new
    /// FIDs), deduped within the batch so repeats in one level share a
    /// column. Returns the per-pair column index.
    fn assign_columns(&mut self, fids: &[u32]) -> Vec<usize> {
        let mut cols = vec![usize::MAX; fids.len()];
        // First pass: resolve existing FIDs; defer the genuinely-new ones.
        let mut new_uniq: Vec<u32> = Vec::new();
        for (i, &fid) in fids.iter().enumerate() {
            match self.fid_to_col.get(&fid) {
                Some(&c) if c != U32_MISS => {
                    cols[i] = c as usize;
                }
                _ => new_uniq.push(fid),
            }
        }
        if !new_uniq.is_empty() {
            // np.unique: sorted ascending + unique. Fresh columns count up
            // from the current n_cols in that ascending-FID order.
            new_uniq.sort_unstable();
            new_uniq.dedup();
            let base = self.n_cols;
            for (k, &fid) in new_uniq.iter().enumerate() {
                self.fid_to_col.insert(fid, (base + k) as u32);
            }
            self.n_cols = base + new_uniq.len();
            // Second pass: fill the deferred miss positions from the map.
            for (i, &fid) in fids.iter().enumerate() {
                if cols[i] == usize::MAX {
                    cols[i] = *self
                        .fid_to_col
                        .get(&fid)
                        .expect("just-inserted FID column missing")
                        as usize;
                }
            }
        }
        cols
    }

    /// One level's inclusion + survival decision (mirrors `step_level`).
    /// `variant` / `fids` are parallel emission-order pair arrays.
    pub(crate) fn step_level(
        &mut self,
        variant: &[i64],
        fids: &[u32],
    ) -> Result<(Vec<bool>, Vec<i64>), String> {
        let n_pairs = fids.len();
        if variant.len() != n_pairs {
            return Err(format!(
                "variant and callee_function_id must be parallel; got {} vs {}",
                variant.len(),
                n_pairs
            ));
        }
        if n_pairs == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // (1) dense columns, then grow the mask to the (possibly extended)
        // column count — exactly `_assign_columns` then `_ensure_capacity`.
        let cols = self.assign_columns(fids);
        self.ensure_capacity(self.n_variants, self.n_cols);

        // Range-check variant rows (the numpy path would index out of
        // bounds otherwise; surface a clear error instead).
        for &v in variant.iter() {
            if v < 0 || v as usize >= self.n_variants {
                return Err(format!(
                    "variant row {v} out of range for n_variants {}",
                    self.n_variants
                ));
            }
        }

        // (2) pre-mark snapshot (FLAG-B read-before-write).
        let mut pre_cell = vec![false; n_pairs];
        for i in 0..n_pairs {
            let idx = self.idx(variant[i] as usize, cols[i]);
            pre_cell[i] = self.mask[idx];
        }

        // (3) first-occurrence dedup of repeated (variant, col) within this
        // level (earliest in emission order wins). A seen-set pass in
        // original order is byte-identical to the numpy stable-argsort
        // first-per-group (the first in each group is the earliest index).
        let mut first_in_level = vec![false; n_pairs];
        let mut seen: HashMap<(i64, usize), ()> =
            HashMap::with_capacity(n_pairs);
        for i in 0..n_pairs {
            if seen.insert((variant[i], cols[i]), ()).is_none() {
                first_in_level[i] = true;
            }
        }

        // (4) mark every pair's cell True.
        for i in 0..n_pairs {
            let idx = self.idx(variant[i] as usize, cols[i]);
            self.mask[idx] = true;
        }

        // (5) columnwise-ALL over ONLY the touched columns across every
        // variant row (FLAG-A). A touched column all-True over the
        // n_variants rows is excluded; single-variant roots are trivially
        // all-True for every touched column => splice nothing. The numpy
        // path computes `excluded_col[touched]` then gathers
        // `excluded_col[cols]`, so evaluating each DISTINCT column once and
        // gating each pair by its own column is identical.
        let mut col_excluded: HashMap<usize, bool> =
            HashMap::with_capacity(cols.len());
        let mut pair_excluded = vec![false; n_pairs];
        for i in 0..n_pairs {
            let c = cols[i];
            let excluded = *col_excluded.entry(c).or_insert_with(|| {
                let mut all_true = true;
                for r in 0..self.n_variants {
                    if !self.mask[r * self.cols_cap + c] {
                        all_true = false;
                        break;
                    }
                }
                all_true
            });
            pair_excluded[i] = excluded;
        }

        // (6) once-only inclusion: first encounter for this variant (across
        // prior levels AND within this level) AND not excluded. (7)
        // survivors in ascending pair index (np.nonzero order).
        let mut included = vec![false; n_pairs];
        let mut survivor_pairs: Vec<i64> = Vec::new();
        for i in 0..n_pairs {
            let inc = !pre_cell[i] && first_in_level[i] && !pair_excluded[i];
            included[i] = inc;
            if inc {
                survivor_pairs.push(i as i64);
            }
        }
        Ok((included, survivor_pairs))
    }
}

#[pyclass(module = "dedup_hashmap._native")]
pub struct OnceOnlyInclusionKernel {
    state: Mutex<DeciderState>,
}

#[pymethods]
impl OnceOnlyInclusionKernel {
    /// Allocate the reusable decider state (mirrors
    /// `OnceOnlyInclusion.__init__`'s `initial_cols`).
    #[new]
    #[pyo3(signature = (initial_cols = 64))]
    fn new(initial_cols: usize) -> Self {
        OnceOnlyInclusionKernel {
            state: Mutex::new(DeciderState::new(initial_cols)),
        }
    }

    /// Reset for a new root; seed the root body at column 0. Returns 0 (the
    /// root's column), matching the Python `begin_root` return.
    fn begin_root(
        &self,
        py: Python<'_>,
        n_variants: i64,
        root_function_id: u32,
    ) -> PyResult<i64> {
        if n_variants <= 0 {
            return Err(PyValueError::new_err(format!(
                "n_variants must be >= 1; got {n_variants}"
            )));
        }
        let n = n_variants as usize;
        py.detach(|| {
            let mut st = self.state.lock().expect("decider mutex poisoned");
            st.begin_root(n, root_function_id);
        });
        Ok(0)
    }

    /// Resolve one level's pairs into `(included, survivor_pairs)` with the
    /// GIL released; inputs are the caller's emission-order pair arrays.
    fn step_level<'py>(
        &self,
        py: Python<'py>,
        variant: PyReadonlyArray1<'py, i64>,
        callee_function_id: PyReadonlyArray1<'py, u32>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let var = variant.as_slice()?;
        let fids = callee_function_id.as_slice()?;
        let (included, survivor_pairs) = py
            .detach(|| {
                let mut st =
                    self.state.lock().expect("decider mutex poisoned");
                st.step_level(var, fids)
            })
            .map_err(PyValueError::new_err)?;
        let tup = PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, included).into_any(),
                PyArray1::from_vec(py, survivor_pairs).into_any(),
            ],
        )?;
        Ok(tup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decider() -> DeciderState {
        DeciderState::new(64)
    }

    fn step(
        d: &mut DeciderState,
        v: &[i64],
        f: &[u32],
    ) -> (Vec<bool>, Vec<i64>) {
        d.step_level(v, f).unwrap()
    }

    #[test]
    fn root_seeded_blocks_self_recursion() {
        let mut d = decider();
        d.begin_root(2, 100);
        let (inc, surv) = step(&mut d, &[0, 1], &[100, 100]);
        assert_eq!(inc, vec![false, false]);
        assert!(surv.is_empty());
    }

    #[test]
    fn some_not_all_included_all_excluded() {
        let mut d = decider();
        d.begin_root(3, 1);
        // pairs: (v0,200)(v1,200)(v2,999)(v0,300)(v1,300)(v2,300)
        let (inc, _s) = step(
            &mut d,
            &[0, 1, 2, 0, 1, 2],
            &[200, 200, 999, 300, 300, 300],
        );
        assert_eq!(inc, vec![true, true, true, false, false, false]);
    }

    #[test]
    fn dedup_within_level() {
        let mut d = decider();
        d.begin_root(2, 1);
        let (inc, _s) = step(&mut d, &[0, 0, 1], &[200, 200, 300]);
        assert_eq!(inc, vec![true, false, true]);
    }

    #[test]
    fn dedup_across_levels() {
        let mut d = decider();
        d.begin_root(2, 1);
        let (inc1, _s) = step(&mut d, &[0, 1], &[200, 300]);
        assert_eq!(inc1, vec![true, true]);
        let (inc2, _s) = step(&mut d, &[0, 1], &[200, 400]);
        assert_eq!(inc2, vec![false, true]);
    }

    #[test]
    fn flag_a_single_variant_excludes_everything() {
        let mut d = decider();
        d.begin_root(1, 1);
        let (inc, surv) = step(&mut d, &[0, 0], &[200, 300]);
        assert_eq!(inc, vec![false, false]);
        assert!(surv.is_empty());
    }

    #[test]
    fn flag_b_late_convergence_excludes_late_variants() {
        let mut d = decider();
        d.begin_root(2, 1);
        let (inc1, _s) = step(&mut d, &[0], &[200]);
        assert_eq!(inc1, vec![true]);
        // v1 now reaches F too -> column all-True -> excluded.
        let (inc2, surv2) = step(&mut d, &[1], &[200]);
        assert_eq!(inc2, vec![false]);
        assert!(surv2.is_empty());
    }

    #[test]
    fn empty_level_returns_empty() {
        let mut d = decider();
        d.begin_root(2, 1);
        let (inc, surv) = step(&mut d, &[], &[]);
        assert!(inc.is_empty());
        assert!(surv.is_empty());
    }

    #[test]
    fn buffer_reuse_across_many_roots() {
        let mut d = decider();
        for k in 0..100u32 {
            let n_var = 2 + (k % 4);
            d.begin_root(n_var as usize, 1000 + k);
            let rows: Vec<i64> = (0..n_var as i64).collect();
            let fids: Vec<u32> =
                (0..n_var).map(|v| 2000 + k * 10 + v).collect();
            let (inc, _s) = step(&mut d, &rows, &fids);
            assert_eq!(inc, vec![true; n_var as usize]);
        }
        d.begin_root(2, 1);
        let (inc, _s) = step(&mut d, &[0, 1, 0], &[500, 500, 600]);
        assert_eq!(inc, vec![false, false, true]);
    }

    #[test]
    fn mask_grows_without_losing_marks() {
        let mut d = DeciderState::new(2);
        d.begin_root(5, 1);
        let rows: Vec<i64> = (0..5).chain(0..5).collect();
        let mut fids: Vec<u32> = (700..705).collect();
        fids.extend(710..715);
        let (inc, _s) = step(&mut d, &rows, &fids);
        assert_eq!(inc, vec![true; 10]);
        let (inc2, _s) =
            step(&mut d, &[0, 1, 2, 3, 4], &[700, 701, 702, 703, 704]);
        assert_eq!(inc2, vec![false; 5]);
    }

    #[test]
    fn reset_leaves_only_previous_region_zeroed() {
        let mut d = decider();
        d.begin_root(2, 0);
        let big: Vec<u32> = (1..51).collect();
        step(&mut d, &vec![0i64; 50], &big);
        d.begin_root(2, 1000);
        let (inc, _s) = step(&mut d, &[0, 1], &[1, 1]);
        // both reach fid 1 -> all-variants exclusion fires.
        assert_eq!(inc, vec![false, false]);
        let (inc2, _s) = step(&mut d, &[0], &[7]);
        assert!(inc2[0], "first encounter after reset must be included");
    }

    /// Ascending-FID column assignment: new FIDs in one level take columns
    /// in ASCENDING-FID order (np.unique sorts), NOT first-appearance. The
    /// included output is order-invariant here, but column ids drive the
    /// `_dense_columns` counter mint downstream, so pin the order directly.
    #[test]
    fn columns_assigned_in_ascending_fid_order() {
        let mut d = decider();
        d.begin_root(2, 1);
        // Present FIDs out of order: 300 then 200. np.unique => [200, 300]
        // so 200 takes col 1, 300 takes col 2 (col 0 is the root).
        let cols = d.assign_columns(&[300, 200]);
        assert_eq!(cols, vec![2usize, 1usize]);
    }

    /// Adversarial mutation A: break the FLAG-A columnwise-ALL exclusion
    /// (never exclude). A single level where both variants reach the same
    /// callee must flip from [false,false] (faithful) to [true,false].
    #[test]
    fn adversarial_drop_columnwise_all_changes_output() {
        let mut faithful = decider();
        faithful.begin_root(2, 1);
        let (good, _s) = step(&mut faithful, &[0, 1], &[200, 200]);
        assert_eq!(
            good,
            vec![false, false],
            "columnwise-ALL must exclude an all-variant column"
        );

        // Mutated: replicate step_level WITHOUT the columnwise-ALL gate.
        let mut m = decider();
        m.begin_root(2, 1);
        let cols = m.assign_columns(&[200, 200]);
        m.ensure_capacity(m.n_variants, m.n_cols);
        let vr = [0i64, 1];
        let pre: Vec<bool> =
            (0..2).map(|i| m.mask[m.idx(vr[i] as usize, cols[i])]).collect();
        let mut first = vec![false; 2];
        let mut seen: HashMap<(i64, usize), ()> = HashMap::new();
        for i in 0..2 {
            if seen.insert((vr[i], cols[i]), ()).is_none() {
                first[i] = true;
            }
        }
        // BROKEN: omit `& ~pair_excluded`. Both pairs are fresh (distinct
        // variant rows => distinct (variant,col) keys) so both first-occur.
        let bad: Vec<bool> = (0..2).map(|i| !pre[i] && first[i]).collect();
        assert_eq!(bad, vec![true, true]);
        assert_ne!(
            good, bad,
            "dropping columnwise-ALL must change the inclusion output"
        );
    }

    /// Adversarial mutation B: break the FLAG-B read-before-write (mark
    /// BEFORE snapshotting pre_cell). The first occurrence of a fresh
    /// callee must flip from included to excluded, proving the ordering is
    /// load-bearing. Uses a SINGLE-variant-free scenario where the
    /// columnwise-ALL does NOT fire (3 variants, only 1 reaches F), so the
    /// pre_cell read is the sole governing rule.
    #[test]
    fn adversarial_mark_before_snapshot_changes_output() {
        // Faithful: v0 reaches F (col fresh) in a 3-variant root. pre_cell
        // False, column NOT all-True (only 1/3) => included.
        let mut faithful = decider();
        faithful.begin_root(3, 1);
        let (good, _s) = step(&mut faithful, &[0], &[200]);
        assert_eq!(good, vec![true]);

        // Mutated: mark BEFORE snapshotting pre_cell. v0's own write makes
        // pre_cell read True => excluded ([false]) — a DIFFERENT output.
        let mut m = decider();
        m.begin_root(3, 1);
        let cols = m.assign_columns(&[200]);
        m.ensure_capacity(m.n_variants, m.n_cols);
        let vr = [0i64];
        // BROKEN: mark first ...
        for i in 0..1 {
            let idx = m.idx(vr[i] as usize, cols[i]);
            m.mask[idx] = true;
        }
        // ... then snapshot (contaminated by the own write).
        let pre: Vec<bool> =
            (0..1).map(|i| m.mask[m.idx(vr[i] as usize, cols[i])]).collect();
        let mut first = vec![false; 1];
        let mut seen: HashMap<(i64, usize), ()> = HashMap::new();
        for i in 0..1 {
            if seen.insert((vr[i], cols[i]), ()).is_none() {
                first[i] = true;
            }
        }
        // columnwise-ALL: col is 1/3 True -> not excluded.
        let mut all_true = true;
        for r in 0..m.n_variants {
            if !m.mask[r * m.cols_cap + cols[0]] {
                all_true = false;
                break;
            }
        }
        let bad: Vec<bool> =
            (0..1).map(|i| !pre[i] && first[i] && !all_true).collect();
        assert_eq!(bad, vec![false]);
        assert_ne!(
            good, bad,
            "mark-before-snapshot must change the first-inclusion output"
        );
    }
}
