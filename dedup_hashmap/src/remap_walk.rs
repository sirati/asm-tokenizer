//! `apply_remap_walk` — the per-row identity FID/counter remap kernel.
//!
//! Single concern: replicate the batch_decode Stage-4 per-row dedup walk
//! (ALG-3 FUNCTION dedup + ALG-4 COUNTER offset bump + ALG-9 prepend
//! self-counter) as a deterministic INTEGER STATE MACHINE over flat int
//! arrays, mutating the caller-local identity stream in place to
//! variant-global counter ids and emitting the per-row FID inverse map.
//!
//! The Python `apply_per_row_remap` walked a frozen object tree
//! (`Stage3Variant -> Stage3CallTarget -> Stage2CallTarget ->
//! Stage1CallTarget`) reading per-node enum/slice/section state. The
//! caller flattens that tree (vectorized, on the Python side) into the
//! arrays below and hands them here; this kernel re-implements NONE of
//! the band/promotion/expansion rules — it owns ONLY the integer remap.
//!
//! ## Category codes
//!
//! The caller maps every `Category` enum to a small int BEFORE the
//! kernel (collision-checked on the Python side). The kernel only needs
//! the FUNCTION/COUNTER PARTITION and a dense per-category counter space,
//! so it takes the codes as opaque indices `0 .. n_func_cats` (FUNCTION)
//! and `0 .. n_counter_cats` (COUNTER), provided as two parallel column
//! layouts:
//!
//! * FUNCTION categories are addressed by a dense `func_slot` in
//!   `0 .. n_func_cats`. The caller passes the per-node call-target
//!   `ct_func_slot` (the FUNCTION-category dense slot of each call-target
//!   row, or `-1` for non-function rows) and the per-in-stream-position
//!   `instream_func_slot` (`-1` when the in-stream identity is NOT a
//!   FUNCTION-category token). The LOCAL_FUNC slot (the root-seed target)
//!   is `root_func_slot`.
//! * COUNTER categories are addressed by a dense `counter_slot` in
//!   `0 .. n_counter_cats`. Per node the caller passes the per-COUNTER-
//!   category count column `counter_counts[node, counter_slot]` and the
//!   per-in-stream-position `instream_counter_slot` (`-1` when the
//!   in-stream identity is NOT a COUNTER-category token).
//!
//! The encounter category (ALG-9 prepend self-counter lookup) is a
//! FUNCTION category, passed as `node_enc_func_slot`.
//!
//! ## Per-node flat layout
//!
//! All `node_*` arrays are length `n_nodes`, in EMISSION ORDER (the same
//! order the Python walk visited call_targets). `node_row` groups nodes
//! into rows (variants); a row is a maximal run of equal `node_row` —
//! the caller emits rows contiguously. The walk resets its per-row state
//! at each row boundary.

use hashbrown::HashMap;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

const NOT_FOUND_U16: u16 = u16::MAX;

/// Output of one row's walk: the per-FUNCTION-category FID inverse map
/// (`counter_id -> function_name_ptr`, dense in counter-id order).
struct RowFidInverse {
    /// `per_func_slot[func_slot]` is the inverse list for that FUNCTION
    /// category (length == that category's minted counter cardinality).
    per_func_slot: Vec<Vec<u32>>,
}

/// Per-row mutable dedup state (mirrors Python `_RowState` + the three
/// `HashMapU32U16` dedup maps).
struct RowState {
    /// One `fid -> counter_id` map per FUNCTION category slot.
    dedup: Vec<HashMap<u32, u16>>,
    /// Per-FUNCTION-category next dense counter id.
    next_fresh: Vec<u32>,
    /// Per-COUNTER-category running offset.
    counter_offset: Vec<u32>,
    /// Optional `counter_id -> fid` inverse list per FUNCTION category.
    fid_inverse: Option<Vec<Vec<u32>>>,
}

impl RowState {
    fn new(n_func: usize, n_counter: usize, collect_fid: bool) -> Self {
        RowState {
            dedup: (0..n_func).map(|_| HashMap::new()).collect(),
            next_fresh: vec![0u32; n_func],
            counter_offset: vec![0u32; n_counter],
            fid_inverse: if collect_fid {
                Some((0..n_func).map(|_| Vec::new()).collect())
            } else {
                None
            },
        }
    }

    /// Reset for a new row. `clean()`-equivalent: clears entries +
    /// counters but the `HashMap`s keep their allocation.
    fn reset(&mut self) {
        for m in self.dedup.iter_mut() {
            m.clear();
        }
        for v in self.next_fresh.iter_mut() {
            *v = 0;
        }
        for v in self.counter_offset.iter_mut() {
            *v = 0;
        }
        if let Some(inv) = self.fid_inverse.as_mut() {
            for v in inv.iter_mut() {
                v.clear();
            }
        }
    }

    /// ALG-3 + ALG-9 root seed: LOCAL_FUNC root takes counter id 0.
    fn seed_root(&mut self, root_func_slot: usize, root_fid: u32) {
        self.dedup[root_func_slot].insert(root_fid, 0u16);
        self.next_fresh[root_func_slot] = 1;
        if let Some(inv) = self.fid_inverse.as_mut() {
            inv[root_func_slot].push(root_fid);
        }
    }
}

/// Per-node flat inputs, borrowed for the duration of the walk.
struct NodeInputs<'a> {
    row: &'a [i64],
    skip: &'a [bool],
    prepend_pos: &'a [i64],
    fid: &'a [i64],
    enc_func_slot: &'a [i64],
    root_func_slot: usize,
    // call-target section (FUNCTION rows only)
    ct_off: &'a [i64],
    ct_fid: &'a [i64],
    ct_func_slot: &'a [i64],
    // in-stream identity positions
    instream_off: &'a [i64],
    instream_func_slot: &'a [i64],
    instream_counter_slot: &'a [i64],
    // per-node COUNTER counts, shape (n_nodes, n_counter)
    counter_counts: &'a [i64],
    n_counter: usize,
}

/// Mint counters for one node's K call-target FIDs of `func_slot` into
/// the dedup map (ALG-3 step 1, UNCONDITIONAL). Returns the K-length
/// `remap_lookup` (caller-local id `[0,K)` -> counter id, dense in the
/// filtered call-target order).
fn mint_call_target_fids(
    state: &mut RowState,
    func_slot: usize,
    fids: &[u32],
) -> Vec<u16> {
    let mut remap_lookup: Vec<u16> = Vec::with_capacity(fids.len());
    for &fid in fids {
        let existing = state.dedup[func_slot].get(&fid).copied();
        match existing {
            Some(c) => remap_lookup.push(c),
            None => {
                let fresh = state.next_fresh[func_slot];
                let fresh_u16 = fresh as u16;
                state.dedup[func_slot].insert(fid, fresh_u16);
                state.next_fresh[func_slot] = fresh + 1;
                if let Some(inv) = state.fid_inverse.as_mut() {
                    inv[func_slot].push(fid);
                }
                remap_lookup.push(fresh_u16);
            }
        }
    }
    remap_lookup
}

/// Grow `remap_lookup` to cover non-call-target caller-local ids
/// (ids `>= K`). Mirrors Python `_extend_remap_for_non_call_target_ids`:
/// each DISTINCT such id in this node, in ASCENDING id order, gets a
/// fresh dense counter; the FID sidecar records the UNKNOWN sentinel (0).
fn extend_remap_for_non_call_target_ids(
    state: &mut RowState,
    func_slot: usize,
    selected: &[u16],
    remap_lookup: &mut Vec<u16>,
) {
    let k = remap_lookup.len();
    let max_id = selected.iter().copied().max().unwrap_or(0) as usize;
    if max_id < k {
        return;
    }
    // Distinct ids >= K in ascending order (deterministic assignment).
    let mut non_ct: Vec<u16> =
        selected.iter().copied().filter(|&v| (v as usize) >= k).collect();
    non_ct.sort_unstable();
    non_ct.dedup();

    remap_lookup.resize(max_id + 1, NOT_FOUND_U16);
    let mut next = state.next_fresh[func_slot];
    for &id in non_ct.iter() {
        remap_lookup[id as usize] = next as u16;
        next += 1;
    }
    state.next_fresh[func_slot] = next;
    if let Some(inv) = state.fid_inverse.as_mut() {
        for _ in 0..non_ct.len() {
            inv[func_slot].push(0u32); // _UNKNOWN_FID
        }
    }
}

/// One row's walk over `[lo, hi)` nodes. Mutates `identities_flat` in
/// place; returns the per-FUNCTION-category FID inverse (when collected).
#[allow(clippy::too_many_arguments)]
fn walk_row(
    inp: &NodeInputs,
    n_func: usize,
    state: &mut RowState,
    identities_flat: &mut [u16],
    lo: usize,
    hi: usize,
    collect_fid: bool,
) -> PyResult<Option<RowFidInverse>> {
    state.reset();

    // Root seed: LOCAL_FUNC root at counter id 0. The root is the FIRST
    // node of the row (Python reads `call_targets[0]`), regardless of
    // whether it survives.
    let root = lo;
    let root_fid = inp.fid[root];
    if root_fid < 0 {
        return Err(PyValueError::new_err(format!(
            "node {root}: negative function_name_ptr {root_fid}"
        )));
    }
    state.seed_root(inp.root_func_slot, root_fid as u32);

    for e in lo..hi {
        if inp.skip[e] {
            continue;
        }
        // --- ALG-9 prepend: self-counter at identity_slice.start ---
        let enc_slot = inp.enc_func_slot[e];
        if enc_slot < 0 || (enc_slot as usize) >= n_func {
            return Err(PyValueError::new_err(format!(
                "node {e}: encounter func_slot {enc_slot} out of range"
            )));
        }
        let self_fid = inp.fid[e] as u32;
        let self_counter = match state.dedup[enc_slot as usize].get(&self_fid) {
            Some(c) => *c,
            None => {
                return Err(PyValueError::new_err(format!(
                    "node {e}: prepend self-counter missing for \
                     (enc_func_slot={enc_slot}, fid={self_fid}); the \
                     stage-1 walker invariant is violated"
                )));
            }
        };
        let pp = inp.prepend_pos[e];
        if pp < 0 || (pp as usize) >= identities_flat.len() {
            return Err(PyValueError::new_err(format!(
                "node {e}: prepend_pos {pp} out of identities_flat bounds"
            )));
        }
        identities_flat[pp as usize] = self_counter;

        // In-stream slice base: identity_slice.start + 1 == pp + 1.
        let is_lo = inp.instream_off[e] as usize;
        let is_hi = inp.instream_off[e + 1] as usize;
        let instream_len = is_hi - is_lo;
        let stream_base = (pp as usize) + 1;
        if stream_base + instream_len > identities_flat.len() {
            return Err(PyValueError::new_err(format!(
                "node {e}: in-stream slice [{stream_base}, \
                 {}) exceeds identities_flat bounds",
                stream_base + instream_len
            )));
        }

        // --- ALG-3 FUNCTION dedup, per FUNCTION category slot ---
        let ct_lo = inp.ct_off[e] as usize;
        let ct_hi = inp.ct_off[e + 1] as usize;
        for fslot in 0..n_func {
            // Filter this node's call-target rows to `fslot`; the
            // caller-local id within the category is the row's position
            // in this filtered order (encoder grouping invariant).
            let mut fids: Vec<u32> = Vec::new();
            for j in ct_lo..ct_hi {
                if inp.ct_func_slot[j] == fslot as i64 {
                    let f = inp.ct_fid[j];
                    if f < 0 {
                        return Err(PyValueError::new_err(format!(
                            "call-target {j}: negative fid {f}"
                        )));
                    }
                    fids.push(f as u32);
                }
            }
            // Step 1 (unconditional): mint counters for every call-target
            // FID — load-bearing for callee prepend recovery + ALG-3.
            let mut remap_lookup = mint_call_target_fids(state, fslot, &fids);

            // Step 2 (gated on in-stream tokens of this category): gather
            // the caller-local ids of in-stream positions whose category
            // is `fslot`, extend the remap for non-call-target ids, then
            // write back the deduped counter ids in place.
            // First collect the selected caller-local ids + their
            // positions.
            let mut sel_positions: Vec<usize> = Vec::new();
            let mut selected: Vec<u16> = Vec::new();
            for p in 0..instream_len {
                if inp.instream_func_slot[is_lo + p] == fslot as i64 {
                    let cl = identities_flat[stream_base + p];
                    sel_positions.push(stream_base + p);
                    selected.push(cl);
                }
            }
            if selected.is_empty() {
                continue;
            }
            extend_remap_for_non_call_target_ids(
                state,
                fslot,
                &selected,
                &mut remap_lookup,
            );
            for (idx, &pos) in sel_positions.iter().enumerate() {
                let cl = selected[idx] as usize;
                identities_flat[pos] = remap_lookup[cl];
            }
        }

        // --- ALG-4 COUNTER offset bump, per COUNTER category slot ---
        for cslot in 0..inp.n_counter {
            let per_node_count =
                inp.counter_counts[e * inp.n_counter + cslot];
            if per_node_count <= 0 {
                continue;
            }
            let offset = state.counter_offset[cslot];
            if offset > 0 {
                for p in 0..instream_len {
                    if inp.instream_counter_slot[is_lo + p] == cslot as i64 {
                        let pos = stream_base + p;
                        identities_flat[pos] =
                            identities_flat[pos].wrapping_add(offset as u16);
                    }
                }
            }
            state.counter_offset[cslot] = offset + per_node_count as u32;
        }
    }

    if collect_fid {
        let inv = state.fid_inverse.as_ref().unwrap();
        Ok(Some(RowFidInverse {
            per_func_slot: inv.clone(),
        }))
    } else {
        Ok(None)
    }
}

/// Apply the per-row identity FID/counter remap in place.
///
/// See the module docstring for the flat-array contract. Mutates
/// `identities_flat` in place (prepend self-counters + remapped in-stream
/// counter ids). When `collect_fid=True`, returns a Python list of length
/// `n_rows`; each entry is a list of `n_func` `u32` ndarrays (the
/// per-FUNCTION-category `counter_id -> fid` inverse, dense in counter-id
/// order). When `collect_fid=False`, returns `None`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn apply_remap_walk<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_func: usize,
    root_func_slot: usize,
    mut identities_flat: PyReadwriteArray1<'py, u16>,
    node_row: PyReadonlyArray1<'py, i64>,
    node_skip: PyReadonlyArray1<'py, bool>,
    node_prepend_pos: PyReadonlyArray1<'py, i64>,
    node_fid: PyReadonlyArray1<'py, i64>,
    node_enc_func_slot: PyReadonlyArray1<'py, i64>,
    ct_off: PyReadonlyArray1<'py, i64>,
    ct_fid: PyReadonlyArray1<'py, i64>,
    ct_func_slot: PyReadonlyArray1<'py, i64>,
    instream_off: PyReadonlyArray1<'py, i64>,
    instream_func_slot: PyReadonlyArray1<'py, i64>,
    instream_counter_slot: PyReadonlyArray1<'py, i64>,
    counter_counts: PyReadonlyArray2<'py, i64>,
    collect_fid: bool,
) -> PyResult<Option<Py<PyList>>> {
    let identities = identities_flat.as_slice_mut()?;
    let row = node_row.as_slice()?;
    let skip = node_skip.as_slice()?;
    let prepend_pos = node_prepend_pos.as_slice()?;
    let fid = node_fid.as_slice()?;
    let enc_func_slot = node_enc_func_slot.as_slice()?;
    let ct_off_s = ct_off.as_slice()?;
    let ct_fid_s = ct_fid.as_slice()?;
    let ct_func_slot_s = ct_func_slot.as_slice()?;
    let instream_off_s = instream_off.as_slice()?;
    let instream_func_slot_s = instream_func_slot.as_slice()?;
    let instream_counter_slot_s = instream_counter_slot.as_slice()?;
    let counter_counts_arr = counter_counts.as_array();
    let counter_shape = counter_counts_arr.shape();
    let n_counter = counter_shape[1];
    let n_nodes = row.len();

    // Validate parallel lengths up front (a wiring bug otherwise reads
    // a wrong node's state silently).
    for (name, len) in [
        ("node_skip", skip.len()),
        ("node_prepend_pos", prepend_pos.len()),
        ("node_fid", fid.len()),
        ("node_enc_func_slot", enc_func_slot.len()),
    ] {
        if len != n_nodes {
            return Err(PyValueError::new_err(format!(
                "{name} length {len} != n_nodes {n_nodes}"
            )));
        }
    }
    if ct_off_s.len() != n_nodes + 1 || instream_off_s.len() != n_nodes + 1 {
        return Err(PyValueError::new_err(
            "ct_off / instream_off must have length n_nodes + 1".to_string(),
        ));
    }
    if counter_shape[0] != n_nodes {
        return Err(PyValueError::new_err(format!(
            "counter_counts has {} rows != n_nodes {n_nodes}",
            counter_shape[0]
        )));
    }
    if root_func_slot >= n_func {
        return Err(PyValueError::new_err(
            "root_func_slot out of [0, n_func)".to_string(),
        ));
    }

    // `counter_counts` may be non-contiguous; copy into a flat row-major
    // buffer so the kernel can index `[e * n_counter + cslot]` cheaply.
    let mut counter_flat: Vec<i64> = Vec::with_capacity(n_nodes * n_counter);
    for e in 0..n_nodes {
        for c in 0..n_counter {
            counter_flat.push(counter_counts_arr[[e, c]]);
        }
    }

    let inp = NodeInputs {
        row,
        skip,
        prepend_pos,
        fid,
        enc_func_slot,
        root_func_slot,
        ct_off: ct_off_s,
        ct_fid: ct_fid_s,
        ct_func_slot: ct_func_slot_s,
        instream_off: instream_off_s,
        instream_func_slot: instream_func_slot_s,
        instream_counter_slot: instream_counter_slot_s,
        counter_counts: &counter_flat,
        n_counter,
    };

    // Compute row boundaries from `node_row` (rows are contiguous runs).
    // Also collect per-row outputs while the GIL is released.
    let row_outputs: PyResult<Vec<Option<RowFidInverse>>> = py.detach(|| {
        let mut state = RowState::new(n_func, n_counter, collect_fid);
        let mut outputs: Vec<Option<RowFidInverse>> = Vec::with_capacity(n_rows);

        let mut e = 0usize;
        let mut expected_row = 0i64;
        while e < n_nodes {
            let r = inp.row[e];
            if r != expected_row {
                return Err(PyValueError::new_err(format!(
                    "node_row not contiguous/0-based: node {e} has row {r}, \
                     expected {expected_row}"
                )));
            }
            let lo = e;
            while e < n_nodes && inp.row[e] == r {
                e += 1;
            }
            let hi = e;
            let out = walk_row(
                &inp,
                n_func,
                &mut state,
                identities,
                lo,
                hi,
                collect_fid,
            )?;
            outputs.push(out);
            expected_row += 1;
        }
        // Rows with NO nodes never appear in `node_row`; the caller's
        // contract is that every row in `[0, n_rows)` has >= 1 node (a
        // variant always has at least its root call_target). Surface a
        // mismatch as a wiring bug.
        if outputs.len() != n_rows {
            return Err(PyValueError::new_err(format!(
                "walked {} rows but n_rows={n_rows}",
                outputs.len()
            )));
        }
        Ok(outputs)
    });
    let row_outputs = row_outputs?;

    if !collect_fid {
        return Ok(None);
    }

    // Build the Python list of per-row, per-func-slot u32 ndarrays.
    let result = PyList::empty(py);
    for out in row_outputs.into_iter() {
        let inv = out.expect("collect_fid=True yields RowFidInverse");
        let row_list = PyList::empty(py);
        for fslot_inv in inv.per_func_slot.into_iter() {
            let arr = PyArray1::from_vec(py, fslot_inv);
            row_list.append(arr)?;
        }
        result.append(row_list)?;
    }
    Ok(Some(result.unbind()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A self-contained Rust reference walk that mirrors the kernel by a
    // SEPARATE code path (no per-row reset reuse) so a test can catch a
    // state-reset bug.
    fn ref_walk(
        n_rows: usize,
        n_func: usize,
        n_counter: usize,
        root_func_slot: usize,
        identities: &mut Vec<u16>,
        row: &[i64],
        skip: &[bool],
        prepend_pos: &[i64],
        fid: &[i64],
        enc_func_slot: &[i64],
        ct_off: &[i64],
        ct_fid: &[i64],
        ct_func_slot: &[i64],
        instream_off: &[i64],
        instream_func_slot: &[i64],
        instream_counter_slot: &[i64],
        counter_counts: &[i64],
    ) {
        let n_nodes = row.len();
        let mut e = 0usize;
        for _ in 0..n_rows {
            if e >= n_nodes {
                break;
            }
            let r = row[e];
            let lo = e;
            while e < n_nodes && row[e] == r {
                e += 1;
            }
            let hi = e;
            // fresh state per row
            let mut dedup: Vec<HashMap<u32, u16>> =
                (0..n_func).map(|_| HashMap::new()).collect();
            let mut next_fresh = vec![0u32; n_func];
            let mut counter_offset = vec![0u32; n_counter];
            dedup[root_func_slot].insert(fid[lo] as u32, 0u16);
            next_fresh[root_func_slot] = 1;
            for node in lo..hi {
                if skip[node] {
                    continue;
                }
                let enc = enc_func_slot[node] as usize;
                let sc = *dedup[enc].get(&(fid[node] as u32)).unwrap();
                let pp = prepend_pos[node] as usize;
                identities[pp] = sc;
                let is_lo = instream_off[node] as usize;
                let is_hi = instream_off[node + 1] as usize;
                let ilen = is_hi - is_lo;
                let base = pp + 1;
                let ctl = ct_off[node] as usize;
                let cth = ct_off[node + 1] as usize;
                for fslot in 0..n_func {
                    let fids: Vec<u32> = (ctl..cth)
                        .filter(|&j| ct_func_slot[j] == fslot as i64)
                        .map(|j| ct_fid[j] as u32)
                        .collect();
                    let mut remap: Vec<u16> = Vec::new();
                    for &f in &fids {
                        match dedup[fslot].get(&f).copied() {
                            Some(c) => remap.push(c),
                            None => {
                                let fr = next_fresh[fslot];
                                dedup[fslot].insert(f, fr as u16);
                                next_fresh[fslot] = fr + 1;
                                remap.push(fr as u16);
                            }
                        }
                    }
                    let mut positions = Vec::new();
                    let mut selected = Vec::new();
                    for p in 0..ilen {
                        if instream_func_slot[is_lo + p] == fslot as i64 {
                            positions.push(base + p);
                            selected.push(identities[base + p]);
                        }
                    }
                    if selected.is_empty() {
                        continue;
                    }
                    let k = remap.len();
                    let maxid =
                        *selected.iter().max().unwrap() as usize;
                    if maxid >= k {
                        let mut nonct: Vec<u16> = selected
                            .iter()
                            .copied()
                            .filter(|&v| (v as usize) >= k)
                            .collect();
                        nonct.sort_unstable();
                        nonct.dedup();
                        remap.resize(maxid + 1, NOT_FOUND_U16);
                        let mut nx = next_fresh[fslot];
                        for &id in &nonct {
                            remap[id as usize] = nx as u16;
                            nx += 1;
                        }
                        next_fresh[fslot] = nx;
                    }
                    for (i, &pos) in positions.iter().enumerate() {
                        identities[pos] = remap[selected[i] as usize];
                    }
                }
                for cslot in 0..n_counter {
                    let cnt = counter_counts[node * n_counter + cslot];
                    if cnt <= 0 {
                        continue;
                    }
                    let off = counter_offset[cslot];
                    if off > 0 {
                        for p in 0..ilen {
                            if instream_counter_slot[is_lo + p] == cslot as i64 {
                                let pos = base + p;
                                identities[pos] =
                                    identities[pos].wrapping_add(off as u16);
                            }
                        }
                    }
                    counter_offset[cslot] = off + cnt as u32;
                }
            }
        }
    }

    /// Drive the kernel core (`walk_row` loop) without PyO3 and compare
    /// to the reference. Returns the mutated identities.
    #[allow(clippy::too_many_arguments)]
    fn kernel_walk(
        n_rows: usize,
        n_func: usize,
        n_counter: usize,
        root_func_slot: usize,
        identities: &mut Vec<u16>,
        row: &[i64],
        skip: &[bool],
        is_root: &[bool],
        prepend_pos: &[i64],
        fid: &[i64],
        enc_func_slot: &[i64],
        ct_off: &[i64],
        ct_fid: &[i64],
        ct_func_slot: &[i64],
        instream_off: &[i64],
        instream_func_slot: &[i64],
        instream_counter_slot: &[i64],
        counter_counts: &[i64],
    ) {
        let _ = is_root;
        let inp = NodeInputs {
            row,
            skip,
            prepend_pos,
            fid,
            enc_func_slot,
            root_func_slot,
            ct_off,
            ct_fid,
            ct_func_slot,
            instream_off,
            instream_func_slot,
            instream_counter_slot,
            counter_counts,
            n_counter,
        };
        let mut state = RowState::new(n_func, n_counter, false);
        let mut e = 0usize;
        for _ in 0..n_rows {
            if e >= row.len() {
                break;
            }
            let r = row[e];
            let lo = e;
            while e < row.len() && row[e] == r {
                e += 1;
            }
            let hi = e;
            walk_row(&inp, n_func, &mut state, identities, lo, hi, false)
                .unwrap();
        }
    }

    #[test]
    fn matches_reference_pseudorandom() {
        // Build a small pseudo-random batch and assert kernel == ref.
        let n_func = 3usize;
        let n_counter = 5usize;
        let root_func_slot = 0usize; // LOCAL_FUNC slot
        let n_rows = 4usize;

        let mut row = Vec::new();
        let mut skip = Vec::new();
        let mut is_root = Vec::new();
        let mut prepend_pos = Vec::new();
        let mut fid = Vec::new();
        let mut enc = Vec::new();
        let mut ct_off = vec![0i64];
        let mut ct_fid: Vec<i64> = Vec::new();
        let mut ct_fslot: Vec<i64> = Vec::new();
        let mut is_off = vec![0i64];
        let mut is_fslot: Vec<i64> = Vec::new();
        let mut is_cslot: Vec<i64> = Vec::new();
        let mut counter_counts: Vec<i64> = Vec::new();

        let mut x = 99u64;
        let mut next = |m: u64| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (x >> 16) % m
        };

        let mut cursor = 0i64;
        for r in 0..n_rows as i64 {
            let nodes = 1 + next(3) as usize;
            for nidx in 0..nodes {
                row.push(r);
                let s = nidx > 0 && next(5) == 0;
                skip.push(s);
                is_root.push(nidx == 0);
                prepend_pos.push(cursor);
                // root fid stable per row; callees random function fids
                let f = if nidx == 0 {
                    1000 + r
                } else {
                    1000 + (next(6) as i64)
                };
                fid.push(f);
                // encounter category: root LOCAL(0); callees func slot 0..2
                let ec = if nidx == 0 { 0 } else { next(n_func as u64) as i64 };
                enc.push(ec);
                // call-targets: a handful, ensure the callee fid + enc is
                // present so the prepend lookup succeeds.
                let kct = next(4) as usize;
                // guarantee the callee's own (enc, fid) is a call-target of
                // SOME earlier node: simplest is to add it to THIS node's
                // own section under enc category (root's section). To keep
                // the prepend invariant we instead add the next callee's
                // (fid, enc) to the ROOT node's section. For test
                // simplicity, make every non-root node's (enc,fid) appear
                // in the root node's call-target section. We approximate by
                // appending here and patching the root section below.
                for _ in 0..kct {
                    ct_fid.push(1000 + next(6) as i64);
                    ct_fslot.push(next(n_func as u64) as i64);
                }
                ct_off.push(ct_fid.len() as i64);
                // in-stream identity positions: a few, with caller-local
                // ids in a small range so some exceed K (non-call-target).
                let ilen = next(5) as usize;
                for _ in 0..ilen {
                    // category: choose func or counter
                    if next(2) == 0 {
                        is_fslot.push(next(n_func as u64) as i64);
                        is_cslot.push(-1);
                    } else {
                        is_fslot.push(-1);
                        is_cslot.push(next(n_counter as u64) as i64);
                    }
                }
                is_off.push(is_fslot.len() as i64);
                // identities buffer: prepend slot + in-stream caller-local
                // ids
                identities_push(&mut cursor, ilen, &mut next);
                // counter counts per node
                for _ in 0..n_counter {
                    counter_counts.push(next(4) as i64);
                }
            }
        }

        // Patch the root node's call-target section to contain every
        // non-root node's (enc_func_slot, fid) so the prepend invariant
        // holds. Rebuild ct arrays.
        let n_nodes = row.len();
        let mut new_ct_fid: Vec<i64> = Vec::new();
        let mut new_ct_fslot: Vec<i64> = Vec::new();
        let mut new_ct_off = vec![0i64];
        // group non-root (enc, fid) per row
        for node in 0..n_nodes {
            if is_root[node] {
                // collect callees of this row
                let r = row[node];
                for n2 in 0..n_nodes {
                    if row[n2] == r && !is_root[n2] {
                        new_ct_fid.push(fid[n2]);
                        new_ct_fslot.push(enc[n2]);
                    }
                }
            }
            new_ct_off.push(new_ct_fid.len() as i64);
        }

        let mut ids_ref = build_identities(&is_off, &prepend_pos, n_nodes);
        let mut ids_ker = ids_ref.clone();

        ref_walk(
            n_rows, n_func, n_counter, root_func_slot, &mut ids_ref, &row,
            &skip, &prepend_pos, &fid, &enc, &new_ct_off, &new_ct_fid,
            &new_ct_fslot, &is_off, &is_fslot, &is_cslot, &counter_counts,
        );
        kernel_walk(
            n_rows, n_func, n_counter, root_func_slot, &mut ids_ker, &row,
            &skip, &is_root, &prepend_pos, &fid, &enc, &new_ct_off,
            &new_ct_fid, &new_ct_fslot, &is_off, &is_fslot, &is_cslot,
            &counter_counts,
        );
        assert_eq!(ids_ref, ids_ker);
    }

    fn identities_push<F: FnMut(u64) -> u64>(
        cursor: &mut i64,
        ilen: usize,
        _next: &mut F,
    ) {
        // advance the cursor past this node's identity slice (prepend +
        // in-stream)
        *cursor += 1 + ilen as i64;
    }

    fn build_identities(
        is_off: &[i64],
        prepend_pos: &[i64],
        n_nodes: usize,
    ) -> Vec<u16> {
        // total length = last prepend_pos + 1 + its in-stream len
        let last = n_nodes - 1;
        let total = (prepend_pos[last] as usize)
            + 1
            + (is_off[last + 1] - is_off[last]) as usize;
        // fill in-stream slots with small caller-local ids; prepend slots
        // are overwritten by the walk.
        let mut v = vec![0u16; total];
        let mut seed = 7u64;
        for node in 0..n_nodes {
            let base = prepend_pos[node] as usize + 1;
            let ilen = (is_off[node + 1] - is_off[node]) as usize;
            for p in 0..ilen {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                v[base + p] = ((seed >> 16) % 6) as u16; // 0..5
            }
        }
        v
    }

    #[test]
    fn adversarial_swap_breaks_identity() {
        // Perturbing one in-stream caller-local id (at a non-prepend
        // position) that the FUNCTION dedup maps to a DISTINCT counter
        // must change the output. One node (the root), 2 LOCAL
        // call-targets so caller-local ids 1 and 2 map to distinct
        // counters (1 and 2). Perturb an in-stream id 1 -> 2.
        let n_func = 3;
        let n_counter = 5;
        let row = [0i64];
        let skip = [false];
        let is_root = [true];
        let prepend_pos = [0i64];
        let fid = [10i64];
        let enc = [0i64];
        // root section holds 2 LOCAL call-targets (fids 11, 12).
        let ct_off = [0i64, 2];
        let ct_fid = [11i64, 12];
        let ct_fslot = [0i64, 0];
        // 2 in-stream LOCAL identity positions (caller-local ids at
        // identities[1], identities[2]).
        let is_off = [0i64, 2];
        let is_fslot = [0i64, 0];
        let is_cslot = [-1i64, -1];
        let counter_counts = vec![0i64; n_counter];

        // identities: [prepend, in0, in1]; caller-local ids 1 and 2.
        let mut a = vec![0u16, 1, 2];
        let mut b = vec![0u16, 2, 2]; // perturb in0: 1 -> 2

        kernel_walk(
            1, n_func, n_counter, 0, &mut a, &row, &skip, &is_root,
            &prepend_pos, &fid, &enc, &ct_off, &ct_fid, &ct_fslot, &is_off,
            &is_fslot, &is_cslot, &counter_counts,
        );
        kernel_walk(
            1, n_func, n_counter, 0, &mut b, &row, &skip, &is_root,
            &prepend_pos, &fid, &enc, &ct_off, &ct_fid, &ct_fslot, &is_off,
            &is_fslot, &is_cslot, &counter_counts,
        );
        // Root fid 10 -> counter 0; call-targets 11,12 -> counters 1,2.
        // remap_lookup indexes by caller-local id: [counter(11)=1,
        // counter(12)=2]. a: in0 caller-local 1 -> 2; in1 caller-local 2
        // is >= K=2 -> fresh counter 3 => [0, 2, 3].
        // b: in0 caller-local 2 -> fresh 3; in1 caller-local 2 -> same
        // fresh 3 => [0, 3, 3]. Distinct outputs prove the dedup walk is
        // sensitive to the in-stream caller-local id.
        assert_ne!(a, b);
        assert_eq!(a, vec![0u16, 2, 3]);
        assert_eq!(b, vec![0u16, 3, 3]);
    }
}
