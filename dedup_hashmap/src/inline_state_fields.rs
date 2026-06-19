//! `build_inline_state_fields_kernel` — fused boundary-aware
//! InlineDecodeState field math over the flat raw stream.
//!
//! Single concern: reproduce, byte-identical, the three pure-numpy
//! boundary-aware passes `_state_fields.py` runs once per decode batch
//! (`_boundary_run_lengths` ×2, `_per_node_digit_cumsum`,
//! `_batched_is_negative`) as ONE GIL-released per-node CSR scalar walk
//! over the flat raw stream. All three numpy passes read the SAME raw
//! stream + per-node CSR (`rec_starts`, `counts`); fusing them releases
//! the GIL once and does a single pass.
//!
//! The kernel OWNS the per-node CSR scalar walk producing the four field
//! arrays; the caller (`_expansion.py`) OWNS deriving its OWN promotion
//! masks + constructing `InlineDecodeState`. The masks the kernel needs
//! are derived INSIDE the detach closure straight from `raw` + the three
//! integer constants (no numpy mask allocs / ascontiguousarray copies).
//!
//! ## Masks (per raw position `p`, derived from `raw[p]`)
//!
//! * `number_mask[p]  = raw[p] <  reserved_digit_count`   (inline digit band)
//! * `real_mask[p]    = raw[p] >  value_negative_id`      (real token)
//! * `value_mask[p]   = !real_mask[p]`                    (digit/sign band)
//! * `carries[p]      = real_mask[p] && raw[p] < eager_block_end`
//!
//! ## Outputs (byte-identical to the numpy trio)
//!
//! * `runlen_number` (`u16[total]`)   — `_boundary_run_lengths(number_mask)`
//! * `runlen_value`  (`u16[total]`)   — `_boundary_run_lengths(value_mask)`
//! * `digit_cumsum`  (`u32[total + n_nodes]`) — packed per-node exclusive
//!   prefix of `number_mask`; node `i`'s `(count_i + 1)`-slot block starts
//!   at `rec_starts[i] + i`, slot `k` = `sum(number_mask[node_start ..
//!   node_start + k])`, with the trailing `(N+1)`-th slot = node digit total.
//! * `is_negative_per_position` (`bool[total]`) — flags a carrier at `p`
//!   whose `p+1` slot is a sign marker (`runlen_value[p+1] !=
//!   runlen_number[p+1]`); NEVER a node's LAST position (no `p+1`).
//!
//! Data dependency: `is_negative` reads `runlen_number`/`runlen_value` at
//! `p+1`, so both run-length passes complete before the is-negative pass —
//! all inside the SAME `py.detach` closure.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The four field arrays in `_expansion.py` consumption order.
#[cfg_attr(test, derive(Debug))]
struct InlineStateFieldsOut {
    runlen_number: Vec<u16>,
    runlen_value: Vec<u16>,
    digit_cumsum: Vec<u32>,
    is_negative_per_position: Vec<bool>,
}

/// Per-node `run_lengths` over the boolean predicate `pred(raw[p])`: a run
/// is a maximal contiguous block of `pred`-true positions WITHIN a node
/// window; its length lands at the run's FIRST position (0 elsewhere).
/// Runs never cross a node boundary, so empty nodes (`count == 0`) collapse
/// to nothing. Mirrors numpy `_boundary_run_lengths`.
fn boundary_run_lengths<F>(
    raw: &[u16],
    rec_starts: &[i64],
    counts: &[i64],
    pred: F,
    out: &mut [u16],
) where
    F: Fn(u16) -> bool,
{
    let n_nodes = rec_starts.len();
    for e in 0..n_nodes {
        let start = rec_starts[e] as usize;
        let cnt = counts[e] as usize;
        let end = start + cnt;
        let mut p = start;
        while p < end {
            if !pred(raw[p]) {
                p += 1;
                continue;
            }
            // Run from `p` to the last consecutive pred-true slot in-window.
            let run_start = p;
            let mut q = p + 1;
            while q < end && pred(raw[q]) {
                q += 1;
            }
            let run_end = q - 1;
            out[run_start] = (run_end - run_start + 1) as u16;
            p = q;
        }
    }
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it
/// directly. Fuses the numpy `_state_fields.py` trio over one CSR walk.
fn run_kernel(
    raw: &[u16],
    rec_starts: &[i64],
    counts: &[i64],
    reserved_digit_count: u16,
    value_negative_id: u16,
    eager_block_end: u16,
) -> Result<InlineStateFieldsOut, String> {
    let n_nodes = rec_starts.len();
    if counts.len() != n_nodes {
        return Err(format!(
            "per-node arrays disagree on n_nodes ({n_nodes}): rec_starts {} \
             counts {}",
            rec_starts.len(),
            counts.len(),
        ));
    }
    let total = raw.len();

    // Validate each node window lands within `raw` (the numpy twin trusts
    // the CSR; the kernel guards the scalar bound walk).
    for e in 0..n_nodes {
        let start = rec_starts[e];
        let cnt = counts[e];
        if start < 0 || cnt < 0 {
            return Err(format!(
                "node {e} window [start={start}, count={cnt}) has a negative \
                 bound"
            ));
        }
        let end = start as usize + cnt as usize;
        if end > total {
            return Err(format!(
                "node {e} window end {end} exceeds raw length {total}"
            ));
        }
    }

    let number_pred = |v: u16| v < reserved_digit_count;
    let value_pred = |v: u16| !(v > value_negative_id); // value_mask = ~real
    let carries = |v: u16| v > value_negative_id && v < eager_block_end;

    // --- run-length passes (both first; is_negative depends on them) ------
    let mut runlen_number = vec![0u16; total];
    let mut runlen_value = vec![0u16; total];
    boundary_run_lengths(raw, rec_starts, counts, number_pred, &mut runlen_number);
    boundary_run_lengths(raw, rec_starts, counts, value_pred, &mut runlen_value);

    // --- per-node digit cumsum (packed CSR, size total + n_nodes) ---------
    // node i's block at dst base `rec_starts[i] + i`, slot k = exclusive
    // prefix of number_mask within the node up to position k; trailing slot
    // (k == count_i) = node digit total.
    let mut digit_cumsum = vec![0u32; total + n_nodes];
    for e in 0..n_nodes {
        let start = rec_starts[e] as usize;
        let cnt = counts[e] as usize;
        let base = start + e; // dst block start
        let mut acc: u32 = 0;
        for k in 0..cnt {
            digit_cumsum[base + k] = acc; // exclusive prefix at position k
            if number_pred(raw[start + k]) {
                acc += 1;
            }
        }
        digit_cumsum[base + cnt] = acc; // trailing (N+1)-th slot = total
    }

    // --- is_negative_per_position -----------------------------------------
    // A carrier at p (not a node's last position) flags negative iff the
    // p+1 slot is a sign marker: runlen_value[p+1] != runlen_number[p+1].
    let mut is_negative_per_position = vec![false; total];
    for e in 0..n_nodes {
        let start = rec_starts[e] as usize;
        let cnt = counts[e] as usize;
        if cnt == 0 {
            continue;
        }
        let end = start + cnt;
        // Exclude the node's LAST position (no p+1 slot).
        for p in start..(end - 1) {
            if carries(raw[p]) {
                is_negative_per_position[p] =
                    runlen_value[p + 1] != runlen_number[p + 1];
            }
        }
    }

    Ok(InlineStateFieldsOut {
        runlen_number,
        runlen_value,
        digit_cumsum,
        is_negative_per_position,
    })
}

/// PyO3 wrapper: borrow the flat raw stream + per-node CSR + the three
/// layout constants, run the fused field walk under `py.detach`, and return
/// `(runlen_number, runlen_value, digit_cumsum, is_negative_per_position)`.
#[pyfunction]
pub fn build_inline_state_fields_kernel<'py>(
    py: Python<'py>,
    raw: numpy::PyReadonlyArray1<'py, u16>,
    rec_starts: numpy::PyReadonlyArray1<'py, i64>,
    counts: numpy::PyReadonlyArray1<'py, i64>,
    reserved_digit_count: u16,
    value_negative_id: u16,
    eager_block_end: u16,
) -> PyResult<Bound<'py, PyTuple>> {
    let raw = raw.as_slice()?;
    let rec_starts = rec_starts.as_slice()?;
    let counts = counts.as_slice()?;

    let out = py
        .detach(|| {
            run_kernel(
                raw,
                rec_starts,
                counts,
                reserved_digit_count,
                value_negative_id,
                eager_block_end,
            )
        })
        .map_err(PyValueError::new_err)?;

    let arrays: [Bound<'py, PyAny>; 4] = [
        PyArray1::from_vec(py, out.runlen_number).into_any(),
        PyArray1::from_vec(py, out.runlen_value).into_any(),
        PyArray1::from_vec(py, out.digit_cumsum).into_any(),
        PyArray1::from_vec(py, out.is_negative_per_position).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canonical unified-vocab constants (see _constants.py).
    const RESERVED: u16 = 256; // _V2_RESERVED_DIGIT_COUNT
    const VALUE_NEG: u16 = 256; // _V2_VALUE_NEGATIVE_TOKEN_ID
    const EAGER_END: u16 = 272; // _V2_EAGER_BLOCK_END
    const VC2: u16 = 257; // _V2_NUMBER_BLOCK_START
    const IDENT: u16 = 264; // _V2_IDENTITY_BLOCK_START
    const SIGN: u16 = 256; // value-negative / sign marker

    fn run(raw: &[u16], rec: &[i64]) -> InlineStateFieldsOut {
        let starts: Vec<i64> = rec[..rec.len() - 1].to_vec();
        let counts: Vec<i64> =
            rec.windows(2).map(|w| w[1] - w[0]).collect();
        run_kernel(raw, &starts, &counts, RESERVED, VALUE_NEG, EAGER_END)
            .unwrap()
    }

    #[test]
    fn single_node_negative_vc2_run() {
        // body: [IDENT, VC2, SIGN, 7, 8, 9]
        //  - number_mask (raw<256): [F, F, F, T, T, T]
        //  - value_mask  (raw<=256): [F, F, T, T, T, T]
        //  - real_mask   (raw>256):  [T, T, F, F, F, F]
        //  - carries (real & raw<272): IDENT@0, VC2@1.
        let raw = [IDENT, VC2, SIGN, 7, 8, 9];
        let out = run(&raw, &[0, 6]);

        // runlen_number: run of digits at slots 3..5 -> length 3 at slot 3.
        assert_eq!(out.runlen_number, vec![0, 0, 0, 3, 0, 0]);
        // runlen_value: run at slots 2..5 -> length 4 at slot 2.
        assert_eq!(out.runlen_value, vec![0, 0, 4, 0, 0, 0]);
        // digit_cumsum block at base 0, size 7 (count 6 + 1):
        //  exclusive prefixes of number_mask [F,F,F,T,T,T]:
        //  k: 0->0 1->0 2->0 3->0 4->1 5->2  trailing(6)->3
        assert_eq!(out.digit_cumsum, vec![0, 0, 0, 0, 1, 2, 3]);
        // is_negative: carrier @0 (IDENT): p+1=1 -> rv[1]=0==rn[1]=0 -> F.
        //   carrier @1 (VC2): p+1=2 -> rv[2]=4 != rn[2]=0 -> TRUE (sign).
        assert_eq!(
            out.is_negative_per_position,
            vec![false, true, false, false, false, false]
        );
    }

    #[test]
    fn multi_node_with_empty_nodes_no_cross_leak() {
        // nodes: [] , [VC2, 0..7] (9 slots), [], [IDENT], []
        // raw = [VC2,0,1,2,3,4,5,6,7, IDENT]
        let raw = [VC2, 0, 1, 2, 3, 4, 5, 6, 7, IDENT];
        // rec offsets: node0 [], node1 [0..9), node2 [], node3 [9..10), node4 []
        let rec = [0i64, 0, 9, 9, 10, 10];
        let out = run(&raw, &rec);

        // node1 number_mask: [F,T,T,T,T,T,T,T,T] -> run at slot1 len 8.
        let mut exp_num = vec![0u16; 10];
        exp_num[1] = 8;
        assert_eq!(out.runlen_number, exp_num);

        // value_mask (raw<=256) node1: VC2>256 F, digits 0..7 <=256 T.
        //  -> [F,T,T,T,T,T,T,T,T] run at slot1 len 8. node3 IDENT>256 -> F.
        let mut exp_val = vec![0u16; 10];
        exp_val[1] = 8;
        assert_eq!(out.runlen_value, exp_val);

        // digit_cumsum: total=10, n_nodes=5 -> size 15.
        //  node0 (start0, base0, cnt0): trailing slot[0]=0.
        //  node1 (start0, base1, cnt9): block [1..11):
        //    exclusive prefixes of number_mask[VC2,0..7]=[F,T,T,T,T,T,T,T,T]:
        //    k:0->0,1->0,2->1,3->2,4->3,5->4,6->5,7->6,8->7 trailing->8
        //  node2 (start9, base11, cnt0): trailing slot[11]=0.
        //  node3 (start9, base12, cnt1): IDENT not number -> [0], trailing 0.
        //    block [12..14): slot12=0, trailing slot13=0.
        //  node4 (start10, base14, cnt0): trailing slot[14]=0.
        let mut exp_cs = vec![0u32; 15];
        // node1 block at base 1: indices 1..=10
        let node1 = [0u32, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        for (i, v) in node1.iter().enumerate() {
            exp_cs[1 + i] = *v;
        }
        assert_eq!(out.digit_cumsum, exp_cs);

        // is_negative: node1 carrier VC2@0: p+1=1 -> rv[1]=8 != rn[1]=8? no,
        //   8==8 -> F. node3 IDENT@9 is node-last -> excluded. -> all False.
        assert_eq!(out.is_negative_per_position, vec![false; 10]);
    }

    #[test]
    fn carrier_at_node_last_position_never_flags() {
        // single node [VC2] — the carrier IS the last position, no p+1.
        let raw = [VC2];
        let out = run(&raw, &[0, 1]);
        assert_eq!(out.is_negative_per_position, vec![false]);
        // digit_cumsum: total 1 + n_nodes 1 = 2; block base0 cnt1:
        //  VC2 not number -> slot0=0, trailing slot1=0.
        assert_eq!(out.digit_cumsum, vec![0, 0]);
    }

    #[test]
    fn empty_batch() {
        let out = run_kernel(&[], &[], &[], RESERVED, VALUE_NEG, EAGER_END)
            .unwrap();
        assert!(out.runlen_number.is_empty());
        assert!(out.runlen_value.is_empty());
        assert!(out.digit_cumsum.is_empty());
        assert!(out.is_negative_per_position.is_empty());
    }

    #[test]
    fn adversarial_window_overruns_raw_errors() {
        let err = run_kernel(
            &[257],
            &[0],
            &[5], // claims 5 slots, raw has 1
            RESERVED,
            VALUE_NEG,
            EAGER_END,
        )
        .unwrap_err();
        assert!(err.contains("exceeds raw length"), "got: {err}");
    }
}
