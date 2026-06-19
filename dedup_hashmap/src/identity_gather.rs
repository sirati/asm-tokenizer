//! `build_identity_carriers_kernel` — stage-3b identity carrier gather.
//!
//! Single concern: given the FLAT `DenseColumns` columns (full-DFS-axis
//! RAW arrays + digit-cumsum CSR + per-node surviving scalars) and stage
//! 3a's per-call_target inline-byte slice starts, locate every SURVIVING
//! in-stream identity carrier in DFS-then-stream encounter order and emit
//! its flat `(first_payload_offset, L, raw_position)` triple as a
//! deterministic GIL-released integer walk.
//!
//! Mirrors the numpy `_gather_identity_carriers` (`_identity_decode.py`
//! lines 186-254) EXACTLY — it re-implements NO decode rule beyond
//! reproducing that vectorised per-node arithmetic as a scalar walk:
//!
//! * skip nodes with `surviving_token_count == 0`;
//! * `in_stream_id_count = surviving_identity_count - 1`; skip when <= 0;
//! * identity carriers = the FIRST `in_stream_id_count` raw positions that
//!   are `real_mask` True AND in the IDENTITY band `[id_lo, id_hi)`;
//! * `L = runlen_number[p + 1]` when `p < n - 1` else 0;
//! * `first_payload_offset = digit_cumsum_node[p + 1] + slice_start`.
//!
//! The ALG-5 row build (`_identity_rows_from_carriers`, 1/2-byte width
//! dispatch) stays in Python — it is already a single vectorised pass over
//! the flat triples and reads no per-node tree.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// CSR slice `[base[e], base[e + 1])` for node `e`.
#[inline]
fn csr_slice(base: &[i64], e: usize) -> (usize, usize) {
    (base[e] as usize, base[e + 1] as usize)
}

/// Pure-Rust core. Returns `(first_payload_offset, L, raw_position)`, all
/// `i64`, in DFS-then-stream carrier order.
#[allow(clippy::too_many_arguments)]
fn run_kernel(
    raw_tokens: &[u16],
    real_mask: &[bool],
    runlen_number: &[u16],
    raw_offsets: &[i64],
    digit_cumsum: &[u32],
    digit_offsets: &[i64],
    surviving_token_count: &[i64],
    surviving_identity_count: &[i64],
    inline_slice_start: &[i64],
    id_lo: i64,
    id_hi: i64,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), String> {
    let n_nodes = surviving_token_count.len();
    if surviving_identity_count.len() != n_nodes
        || inline_slice_start.len() != n_nodes
    {
        return Err(format!(
            "per-node arrays disagree on n_nodes ({n_nodes}): \
             surviving_identity_count {} inline_slice_start {}",
            surviving_identity_count.len(),
            inline_slice_start.len()
        ));
    }

    let mut offsets: Vec<i64> = Vec::new();
    let mut lengths: Vec<i64> = Vec::new();
    let mut positions: Vec<i64> = Vec::new();

    for e in 0..n_nodes {
        if surviving_token_count[e] == 0 {
            continue;
        }
        let in_stream_id_count = surviving_identity_count[e] - 1;
        if in_stream_id_count <= 0 {
            continue;
        }

        let (raw_lo, raw_hi) = csr_slice(raw_offsets, e);
        let n = (raw_hi - raw_lo) as i64;
        if raw_hi > raw_tokens.len() || raw_hi > real_mask.len() {
            return Err(format!(
                "node {e} raw slice [{raw_lo}, {raw_hi}) exceeds flats"
            ));
        }
        // digit-cumsum slice is N + 1 long; node_digit_slice[p + 1] for a
        // raw position p in [0, N) is always in bounds.
        let (dig_lo, dig_hi) = csr_slice(digit_offsets, e);
        if dig_hi > digit_cumsum.len() || (dig_hi - dig_lo) as i64 != n + 1 {
            return Err(format!(
                "node {e} digit slice [{dig_lo}, {dig_hi}) is not N+1 \
                 ({}) or exceeds digit_cumsum",
                n + 1
            ));
        }
        let slice_start = inline_slice_start[e];

        // First `in_stream_id_count` real IDENTITY-band carriers in raw
        // order. The cut chops later carriers but never reorders, so a
        // forward scan that stops after `in_stream_id_count` hits matches
        // `identity_carrier_positions[:in_stream_id_count]`.
        let mut taken: i64 = 0;
        for local in 0..(n as usize) {
            let p_abs = raw_lo + local;
            let tok = i64::from(raw_tokens[p_abs]);
            if real_mask[p_abs] && tok >= id_lo && tok < id_hi {
                let p = local as i64;
                // L = runlen_number[p + 1] when p < n - 1 else 0.
                let l: i64 = if p < n - 1 {
                    let idx = raw_lo + (p as usize) + 1;
                    if idx >= runlen_number.len() {
                        return Err(format!(
                            "node {e} runlen idx {idx} out of bounds"
                        ));
                    }
                    i64::from(runlen_number[idx])
                } else {
                    0
                };
                // first_payload_offset = digit_cumsum_node[p + 1] +
                // slice_start. The node-local digit slice base is dig_lo.
                let off =
                    i64::from(digit_cumsum[dig_lo + (p as usize) + 1]) + slice_start;
                offsets.push(off);
                lengths.push(l);
                positions.push(p);
                taken += 1;
                if taken == in_stream_id_count {
                    break;
                }
            }
        }
    }

    Ok((offsets, lengths, positions))
}

/// PyO3 wrapper: borrow the flat columns, run the gather under
/// `py.detach`, hand back the three `i64` arrays.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_identity_carriers_kernel<'py>(
    py: Python<'py>,
    raw_tokens: numpy::PyReadonlyArray1<'py, u16>,
    real_mask: numpy::PyReadonlyArray1<'py, bool>,
    runlen_number: numpy::PyReadonlyArray1<'py, u16>,
    raw_offsets: numpy::PyReadonlyArray1<'py, i64>,
    digit_cumsum: numpy::PyReadonlyArray1<'py, u32>,
    digit_offsets: numpy::PyReadonlyArray1<'py, i64>,
    surviving_token_count: numpy::PyReadonlyArray1<'py, i64>,
    surviving_identity_count: numpy::PyReadonlyArray1<'py, i64>,
    inline_slice_start: numpy::PyReadonlyArray1<'py, i64>,
    id_lo: i64,
    id_hi: i64,
) -> PyResult<Bound<'py, PyTuple>> {
    let raw_tokens = raw_tokens.as_slice()?;
    let real_mask = real_mask.as_slice()?;
    let runlen_number = runlen_number.as_slice()?;
    let raw_offsets = raw_offsets.as_slice()?;
    let digit_cumsum = digit_cumsum.as_slice()?;
    let digit_offsets = digit_offsets.as_slice()?;
    let surviving_token_count = surviving_token_count.as_slice()?;
    let surviving_identity_count = surviving_identity_count.as_slice()?;
    let inline_slice_start = inline_slice_start.as_slice()?;

    let (offsets, lengths, positions) = py
        .detach(|| {
            run_kernel(
                raw_tokens,
                real_mask,
                runlen_number,
                raw_offsets,
                digit_cumsum,
                digit_offsets,
                surviving_token_count,
                surviving_identity_count,
                inline_slice_start,
                id_lo,
                id_hi,
            )
        })
        .map_err(PyValueError::new_err)?;

    let off_arr = PyArray1::from_vec(py, offsets);
    let len_arr = PyArray1::from_vec(py, lengths);
    let pos_arr = PyArray1::from_vec(py, positions);
    PyTuple::new(
        py,
        [off_arr.into_any(), len_arr.into_any(), pos_arr.into_any()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Identity band [264, 272). Build single-node inputs.
    #[allow(clippy::too_many_arguments)]
    fn one_node(
        raw_tokens: Vec<u16>,
        real_mask: Vec<bool>,
        runlen_number: Vec<u16>,
        digit_cumsum: Vec<u32>,
        surviving_token_count: i64,
        surviving_identity_count: i64,
        slice_start: i64,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let n = raw_tokens.len() as i64;
        run_kernel(
            &raw_tokens,
            &real_mask,
            &runlen_number,
            &[0, n],
            &digit_cumsum,
            &[0, n + 1],
            &[surviving_token_count],
            &[surviving_identity_count],
            &[slice_start],
            264,
            272,
        )
        .unwrap()
    }

    #[test]
    fn two_byte_identity_payload() {
        // raw: [id264, b0, b1]; carrier real at pos 0, payload L=2.
        // digit_cumsum N+1 = [0,0,1,2]; first payload offset =
        // digit_cumsum[1] + slice_start = 0 + 5 = 5.
        let (off, l, pos) = one_node(
            vec![264, 10, 20],
            vec![true, false, false],
            vec![0, 2, 2, 2], // runlen_number[p+1]=runlen[1]=2
            vec![0, 0, 1, 2],
            3, // surviving
            2, // 1 in-stream id (count - 1)
            5,
        );
        assert_eq!(off, vec![5]);
        assert_eq!(l, vec![2]);
        assert_eq!(pos, vec![0]);
    }

    #[test]
    fn terminal_carrier_zero_length() {
        // carrier at the LAST raw slot -> p == n-1 -> L forced to 0.
        let (off, l, pos) = one_node(
            vec![10, 264],
            vec![false, true],
            vec![0, 0],
            vec![0, 0, 0],
            3,
            2,
            1,
        );
        assert_eq!(l, vec![0]);
        assert_eq!(pos, vec![1]);
        // offset = digit_cumsum[p+1=2] + 1 = 0 + 1.
        assert_eq!(off, vec![1]);
    }

    #[test]
    fn cut_limits_in_stream_count() {
        // Two identity carriers but surviving_identity_count - 1 == 1, so
        // only the FIRST is gathered.
        let (off, _l, pos) = one_node(
            vec![264, 264],
            vec![true, true],
            vec![0, 0, 0],
            vec![0, 0, 0],
            3,
            2, // only 1 in-stream id
            1,
        );
        assert_eq!(pos, vec![0]);
        assert_eq!(off.len(), 1);
    }

    #[test]
    fn non_identity_band_real_token_skipped() {
        // A NUMBER-band carrier (257) is real but NOT in the identity
        // band; only the id264 contributes.
        let (_off, _l, pos) = one_node(
            vec![257, 5, 264, 7],
            vec![true, false, true, false],
            vec![0, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 2],
            5,
            2,
            1,
        );
        assert_eq!(pos, vec![2]); // raw position of the id264 carrier
    }

    #[test]
    fn dropped_node_emits_nothing() {
        let (off, l, pos) = one_node(
            vec![264, 1],
            vec![true, false],
            vec![0, 0, 0],
            vec![0, 0, 0],
            0, // surviving_token_count == 0
            0,
            1,
        );
        assert!(off.is_empty() && l.is_empty() && pos.is_empty());
    }

    #[test]
    fn multi_node_slice_start_and_order() {
        // node0 slice_start=1, node1 slice_start=4. Each emits its own
        // carrier with its own base.
        let raw = vec![264, 5, /* node0 */ 264, 6 /* node1 */];
        let real = vec![true, false, true, false];
        let runlen = vec![0, 1, 1, /* node0 N+1=3? */ 0, 1, 1];
        // per-node digit cumsum slices (N+1 each = 3): node0 [0,0,1],
        // node1 [0,0,1].
        let digit = vec![0, 0, 1, 0, 0, 1];
        let (off, _l, pos) = run_kernel(
            &raw,
            &real,
            &runlen,
            &[0, 2, 4],
            &digit,
            &[0, 3, 6],
            &[2, 2],
            &[2, 2],
            &[1, 4],
            264,
            272,
        )
        .unwrap();
        assert_eq!(pos, vec![0, 0]);
        // node0 off = digit[node0][1] + 1 = 0 + 1; node1 = digit[node1][1] + 4 = 0 + 4.
        assert_eq!(off, vec![1, 4]);
    }
}
