//! `build_node_ct_csr_kernel` — per-node call-target-section CSR build.
//!
//! Single concern: given the columnar per-section call_target table (the
//! flat `ct_function_name_ptr` + `ct_type` arrays partitioned by the
//! per-section CSR `ct_offsets`), each emitted node's owning section
//! (`section_of_node`), and the static `CallTargetType -> FUNCTION slot`
//! lookup (`func_slot_lut`), emit each node's OWNING section's call_target
//! table in section order as the flat CSR `(ct_off, ct_fid, ct_func_slot)` —
//! exactly the per-node concatenation the numpy `_build_ct_columns`
//! (`_remap_inputs.py`) performs by re-parsing a `list[CallTarget]` per
//! distinct section and gathering it per node.
//!
//! This re-implements NO decode rule and builds NO `CallTarget` object. It
//! only SLICES the columnar `ct_*` flats per owning section (each section's
//! `[ct_offsets[s], ct_offsets[s + 1])` window) and TRANSLATES `ct_type`
//! through the caller-supplied slot LUT — the same `(fid, func_slot)` pair
//! the numpy twin extracts from each parsed `CallTarget`. The fid is the
//! `ct_function_name_ptr` (u32) widened to i64; the func slot is
//! `func_slot_lut[ct_type]`. Sections repeat across a row's nodes, so the
//! numpy twin caches the per-section `(fid, slot)` arrays once and gathers
//! per node; the gather order here is the SAME node order, so the
//! concatenation is byte-identical.
//!
//! ## Per-node emission (node `e`, owning section `s = section_of_node[e]`)
//!
//! `lo = ct_offsets[s]`, `hi = ct_offsets[s + 1]`; for `k in lo..hi`:
//!   * `ct_fid += ct_function_name_ptr[k] as i64`
//!   * `ct_func_slot += func_slot_lut[ct_type[k] as usize]`
//! and `ct_off[e + 1] = ct_off[e] + (hi - lo)`.
//!
//! ## Order
//!
//! Nodes are walked `0 .. n_nodes`, each appends its section's CT slice in
//! section order — identical to the numpy `np.concatenate(fid_pieces)` over
//! the per-node section slices, which preserves the same node scan order.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The three flat CSR outputs in `_build_ct_columns` return order.
#[cfg_attr(test, derive(Debug))]
struct NodeCtCsrOut {
    ct_off: Vec<i64>,
    ct_fid: Vec<i64>,
    ct_func_slot: Vec<i64>,
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it
/// directly. Mirrors numpy `_build_ct_columns` over the columnar catalog.
fn run_kernel(
    ct_function_name_ptr: &[u32],
    ct_type: &[u8],
    ct_offsets: &[i64],
    section_of_node: &[i64],
    func_slot_lut: &[i64],
) -> Result<NodeCtCsrOut, String> {
    if ct_offsets.is_empty() {
        return Err("ct_offsets must have at least one element".to_string());
    }
    let n_sections = ct_offsets.len() - 1;
    let n_total_ct = *ct_offsets.last().unwrap();
    if n_total_ct < 0 {
        return Err(format!("ct_offsets tail {n_total_ct} is negative"));
    }
    let n_total_ct = n_total_ct as usize;
    if ct_function_name_ptr.len() < n_total_ct || ct_type.len() < n_total_ct {
        return Err(format!(
            "ct columns shorter than ct_offsets tail {n_total_ct}: \
             ct_function_name_ptr {} ct_type {}",
            ct_function_name_ptr.len(),
            ct_type.len(),
        ));
    }

    let n_nodes = section_of_node.len();
    let mut out = NodeCtCsrOut {
        ct_off: vec![0i64; n_nodes + 1],
        ct_fid: Vec::new(),
        ct_func_slot: Vec::new(),
    };

    let mut running: i64 = 0;
    for (e, &s_i64) in section_of_node.iter().enumerate() {
        if s_i64 < 0 || (s_i64 as usize) >= n_sections {
            return Err(format!(
                "section_of_node[{e}] = {s_i64} out of range [0, {n_sections})"
            ));
        }
        let s = s_i64 as usize;
        let lo = ct_offsets[s];
        let hi = ct_offsets[s + 1];
        if lo < 0 || hi < lo || (hi as usize) > n_total_ct {
            return Err(format!(
                "section {s} ct slice [{lo}, {hi}) malformed against tail \
                 {n_total_ct}"
            ));
        }
        let lo = lo as usize;
        let hi = hi as usize;
        for k in lo..hi {
            out.ct_fid.push(ct_function_name_ptr[k] as i64);
            let t = ct_type[k] as usize;
            if t >= func_slot_lut.len() {
                return Err(format!(
                    "ct_type[{k}] = {t} out of LUT range [0, {})",
                    func_slot_lut.len()
                ));
            }
            out.ct_func_slot.push(func_slot_lut[t]);
        }
        running += (hi - lo) as i64;
        out.ct_off[e + 1] = running;
    }

    Ok(out)
}

/// PyO3 wrapper: borrow the columnar CT flats + per-node section + slot
/// LUT, build the CSR under `py.detach`, return `(ct_off, ct_fid,
/// ct_func_slot)` in `_build_ct_columns` order.
#[pyfunction]
pub fn build_node_ct_csr_kernel<'py>(
    py: Python<'py>,
    ct_function_name_ptr: numpy::PyReadonlyArray1<'py, u32>,
    ct_type: numpy::PyReadonlyArray1<'py, u8>,
    ct_offsets: numpy::PyReadonlyArray1<'py, i64>,
    section_of_node: numpy::PyReadonlyArray1<'py, i64>,
    func_slot_lut: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let ct_function_name_ptr = ct_function_name_ptr.as_slice()?;
    let ct_type = ct_type.as_slice()?;
    let ct_offsets = ct_offsets.as_slice()?;
    let section_of_node = section_of_node.as_slice()?;
    let func_slot_lut = func_slot_lut.as_slice()?;

    let out = py
        .detach(|| {
            run_kernel(
                ct_function_name_ptr,
                ct_type,
                ct_offsets,
                section_of_node,
                func_slot_lut,
            )
        })
        .map_err(PyValueError::new_err)?;

    let arrays: [Bound<'py, PyAny>; 3] = [
        PyArray1::from_vec(py, out.ct_off).into_any(),
        PyArray1::from_vec(py, out.ct_fid).into_any(),
        PyArray1::from_vec(py, out.ct_func_slot).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    // LUT: LOCAL(0)->10, PLT(1)->11, EXTERN(2)->12 (stand-ins for the real
    // FUNCTION slot ids).
    const LUT: [i64; 3] = [10, 11, 12];

    #[test]
    fn single_node_single_section() {
        // section 0 has 2 CTs: (fid 100, type 0), (fid 200, type 2).
        let out = run_kernel(
            &[100, 200],
            &[0, 2],
            &[0, 2],
            &[0],
            &LUT,
        )
        .unwrap();
        assert_eq!(out.ct_off, vec![0, 2]);
        assert_eq!(out.ct_fid, vec![100, 200]);
        assert_eq!(out.ct_func_slot, vec![10, 12]);
    }

    #[test]
    fn repeated_section_across_nodes_abuts() {
        // 2 sections. sec0: [fid 1 t0], sec1: [fid 2 t1, fid 3 t2].
        // nodes: [sec1, sec0, sec1] -> the section table is re-emitted per
        // node, in node order.
        let out = run_kernel(
            &[1, 2, 3],
            &[0, 1, 2],
            &[0, 1, 3],
            &[1, 0, 1],
            &LUT,
        )
        .unwrap();
        // node0 sec1 (2 cts), node1 sec0 (1 ct), node2 sec1 (2 cts).
        assert_eq!(out.ct_off, vec![0, 2, 3, 5]);
        assert_eq!(out.ct_fid, vec![2, 3, 1, 2, 3]);
        assert_eq!(out.ct_func_slot, vec![11, 12, 10, 11, 12]);
    }

    #[test]
    fn empty_section_emits_zero_run() {
        // sec0 is empty (offsets 0,0); sec1 has one ct.
        let out = run_kernel(
            &[7],
            &[1],
            &[0, 0, 1],
            &[0, 1, 0],
            &LUT,
        )
        .unwrap();
        assert_eq!(out.ct_off, vec![0, 0, 1, 1]);
        assert_eq!(out.ct_fid, vec![7]);
        assert_eq!(out.ct_func_slot, vec![11]);
    }

    #[test]
    fn no_nodes_yields_single_zero_offset() {
        let out =
            run_kernel(&[1], &[0], &[0, 1], &[], &LUT).unwrap();
        assert_eq!(out.ct_off, vec![0]);
        assert!(out.ct_fid.is_empty());
        assert!(out.ct_func_slot.is_empty());
    }

    #[test]
    fn fid_u32_widens_without_sign_extension() {
        // A high u32 fid must widen to a POSITIVE i64, not sign-extend.
        let big = 0xFFFF_FFFFu32; // 4294967295
        let out =
            run_kernel(&[big], &[0], &[0, 1], &[0], &LUT).unwrap();
        assert_eq!(out.ct_fid, vec![4_294_967_295i64]);
        assert_ne!(out.ct_fid, vec![-1]); // would be the sign-extend bug
    }

    #[test]
    fn adversarial_section_out_of_range_errors() {
        let err = run_kernel(&[1], &[0], &[0, 1], &[5], &LUT)
            .unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn adversarial_ct_type_past_lut_errors() {
        // ct_type 2 but LUT has only 2 entries -> out of range.
        let err = run_kernel(&[1], &[2], &[0, 1], &[0], &[10, 11])
            .unwrap_err();
        assert!(err.contains("LUT range"), "got: {err}");
    }
}
