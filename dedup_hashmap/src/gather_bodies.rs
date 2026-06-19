//! `build_gather_bodies_kernel` — batched single-pass node-body gather.
//!
//! Single concern: given the per-node `(token_start, token_count)` spans,
//! gather EVERY emitted node's raw u16 token region out of `_data.bin`
//! (passed as the read-only `uint8` mmap view) into one flat `raw` u16
//! array with a CSR jump table — the GIL-free twin of the numpy
//! `gather_node_bodies` (`_body_load.py`).
//!
//! The numpy path views `data_u8` as native-endian `uint16` and gathers at
//! word index `(start >> 1) + within`. `_data.bin` is little-endian, so a
//! native-endian u16 view on an LE buffer reads `data_u8[byte] |
//! data_u8[byte + 1] << 8`. This kernel reads the bytes directly with that
//! explicit LE composition, so the output `raw` is byte-identical to the
//! numpy `data_u16[word_idx]` gather (the build host is LE; the explicit
//! composition keeps it correct regardless of host endianness).
//!
//! This re-implements NO decode rule: it only validates the even-offset
//! (u16-aligned record-tail) invariant, builds the CSR from the counts, and
//! copies each node's `2 * count` body bytes into the flat output as LE
//! words — exactly the gather the numpy twin performs, no run-length, no
//! promotion, no band logic.
//!
//! ## Faithful replication of the numpy contract
//!
//! * Parallel-shape check: `token_starts.len() == token_counts.len()`.
//! * Even-offset validation: any odd `start` raises (same ValueError
//!   semantics, carried as a Rust `Err` string).
//! * `n_nodes == 0` and `total == 0` early returns to empty `raw` +
//!   the all-zero `record_offsets` (`[0]`, resp. `n_nodes + 1` zeros).
//! * `record_offsets = cumsum(counts)` (CSR; `[0] == 0`,
//!   `[-1] == raw.len()`).
//! * Each node `i` reads `data_u8[start_i .. start_i + 2 * count_i]` as
//!   `count_i` LE u16 words, appended in node order.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The two flat outputs in `gather_node_bodies` return order.
#[cfg_attr(test, derive(Debug))]
struct GatheredBodiesOut {
    raw: Vec<u16>,
    record_offsets: Vec<i64>,
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it
/// directly. Mirrors numpy `gather_node_bodies` over the `_data.bin` bytes.
fn run_kernel(
    data_u8: &[u8],
    token_starts: &[i64],
    token_counts: &[i64],
) -> Result<GatheredBodiesOut, String> {
    if token_starts.len() != token_counts.len() {
        return Err(format!(
            "token_starts and token_counts must be parallel; got \
             ({},) vs ({},)",
            token_starts.len(),
            token_counts.len(),
        ));
    }
    let n_nodes = token_starts.len();
    let mut record_offsets = vec![0i64; n_nodes + 1];
    if n_nodes == 0 {
        return Ok(GatheredBodiesOut {
            raw: Vec::new(),
            record_offsets,
        });
    }

    // u16-aligned record-tail offsets: the same evenness invariant the bulk
    // geometry scan validates. An odd offset is a corrupt locator, not a
    // silent mis-gather.
    if token_starts.iter().any(|&s| s & 1 != 0) {
        return Err(
            "gather_node_bodies: token_starts must be even (u16-aligned \
             record-tail offsets); got an odd offset"
                .to_string(),
        );
    }

    // record_offsets = cumsum(counts); guard against a negative count
    // (numpy would silently corrupt the CSR; surfacing it is faithful to the
    // "corrupt locator, not silent mis-gather" stance above).
    let mut running: i64 = 0;
    for (i, &c) in token_counts.iter().enumerate() {
        if c < 0 {
            return Err(format!("token_counts[{i}] = {c} is negative"));
        }
        running += c;
        record_offsets[i + 1] = running;
    }
    let total = running;
    if total == 0 {
        return Ok(GatheredBodiesOut {
            raw: Vec::new(),
            record_offsets,
        });
    }
    let total = total as usize;

    let mut raw: Vec<u16> = Vec::with_capacity(total);
    for (i, (&start, &count)) in
        token_starts.iter().zip(token_counts.iter()).enumerate()
    {
        if start < 0 {
            return Err(format!("token_starts[{i}] = {start} is negative"));
        }
        let byte_lo = start as usize;
        let n_words = count as usize;
        let byte_hi = byte_lo + 2 * n_words;
        if byte_hi > data_u8.len() {
            return Err(format!(
                "node {i} body bytes [{byte_lo}, {byte_hi}) exceed \
                 _data.bin length {}",
                data_u8.len()
            ));
        }
        let mut byte = byte_lo;
        for _ in 0..n_words {
            // LE word: low byte | high byte << 8 — the native-endian u16
            // view on the little-endian `_data.bin` the numpy twin gathers.
            let word = (data_u8[byte] as u16) | ((data_u8[byte + 1] as u16) << 8);
            raw.push(word);
            byte += 2;
        }
    }

    Ok(GatheredBodiesOut {
        raw,
        record_offsets,
    })
}

/// PyO3 wrapper: borrow the `_data.bin` u8 mmap view + the per-node
/// `(start, count)` spans, gather under `py.detach`, return `(raw,
/// record_offsets)` in `gather_node_bodies` return order.
#[pyfunction]
pub fn build_gather_bodies_kernel<'py>(
    py: Python<'py>,
    data_u8: numpy::PyReadonlyArray1<'py, u8>,
    token_starts: numpy::PyReadonlyArray1<'py, i64>,
    token_counts: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let data_u8 = data_u8.as_slice()?;
    let token_starts = token_starts.as_slice()?;
    let token_counts = token_counts.as_slice()?;

    let out = py
        .detach(|| run_kernel(data_u8, token_starts, token_counts))
        .map_err(PyValueError::new_err)?;

    let arrays: [Bound<'py, PyAny>; 2] = [
        PyArray1::from_vec(py, out.raw).into_any(),
        PyArray1::from_vec(py, out.record_offsets).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference: view `data` as native-endian u16 and gather at
    /// `(start >> 1) + within`, exactly the numpy expression.
    fn brute_force(
        data_u8: &[u8],
        starts: &[i64],
        counts: &[i64],
    ) -> (Vec<u16>, Vec<i64>) {
        let mut rec = vec![0i64; starts.len() + 1];
        let mut running = 0i64;
        for (i, &c) in counts.iter().enumerate() {
            running += c;
            rec[i + 1] = running;
        }
        let mut raw = Vec::new();
        for (&start, &count) in starts.iter().zip(counts.iter()) {
            let word0 = (start >> 1) as usize;
            for w in 0..count as usize {
                let byte = (word0 + w) * 2;
                raw.push((data_u8[byte] as u16) | ((data_u8[byte + 1] as u16) << 8));
            }
        }
        (raw, rec)
    }

    #[test]
    fn single_node() {
        // bytes: word0 = 0x0201, word1 = 0x0403 at byte offset 4.
        let data = vec![0, 0, 0, 0, 1, 2, 3, 4];
        let out = run_kernel(&data, &[4], &[2]).unwrap();
        assert_eq!(out.raw, vec![0x0201, 0x0403]);
        assert_eq!(out.record_offsets, vec![0, 2]);
    }

    #[test]
    fn multi_node_csr() {
        let data: Vec<u8> = (0..32u8).collect();
        let starts = [0i64, 8, 20];
        let counts = [3i64, 2, 1];
        let out = run_kernel(&data, &starts, &counts).unwrap();
        let (bf_raw, bf_rec) = brute_force(&data, &starts, &counts);
        assert_eq!(out.raw, bf_raw);
        assert_eq!(out.record_offsets, bf_rec);
        assert_eq!(out.record_offsets, vec![0, 3, 5, 6]);
    }

    #[test]
    fn pseudorandom_vs_brute_force() {
        // Deterministic LCG, no deps.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u32
        };
        let data: Vec<u8> = (0..4096).map(|_| (next() & 0xff) as u8).collect();
        // 50 nodes, even starts, counts so byte_hi stays in range.
        let mut starts = Vec::new();
        let mut counts = Vec::new();
        for _ in 0..50 {
            let start = ((next() as usize % 1000) * 2) as i64; // even, <2000
            let count = (next() % 20) as i64; // <=19 words -> <=38 bytes
            starts.push(start);
            counts.push(count);
        }
        let out = run_kernel(&data, &starts, &counts).unwrap();
        let (bf_raw, bf_rec) = brute_force(&data, &starts, &counts);
        assert_eq!(out.raw, bf_raw);
        assert_eq!(out.record_offsets, bf_rec);
    }

    #[test]
    fn odd_offset_errors() {
        let data = vec![0u8; 16];
        let err = run_kernel(&data, &[3], &[1]).unwrap_err();
        assert!(err.contains("must be even"), "got: {err}");
    }

    #[test]
    fn parallel_shape_mismatch_errors() {
        let data = vec![0u8; 16];
        let err = run_kernel(&data, &[0, 2], &[1]).unwrap_err();
        assert!(err.contains("parallel"), "got: {err}");
    }

    #[test]
    fn empty_nodes() {
        let data = vec![0u8; 16];
        let out = run_kernel(&data, &[], &[]).unwrap();
        assert!(out.raw.is_empty());
        assert_eq!(out.record_offsets, vec![0]);
    }

    #[test]
    fn all_zero_counts_total_zero() {
        let data = vec![0u8; 16];
        let out = run_kernel(&data, &[0, 4, 8], &[0, 0, 0]).unwrap();
        assert!(out.raw.is_empty());
        assert_eq!(out.record_offsets, vec![0, 0, 0, 0]);
    }

    #[test]
    fn out_of_range_body_errors() {
        let data = vec![0u8; 8];
        let err = run_kernel(&data, &[6], &[2]).unwrap_err();
        assert!(err.contains("exceed"), "got: {err}");
    }
}
