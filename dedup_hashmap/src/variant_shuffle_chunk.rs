//! `variant_shuffle_chunk_kernel` — the deterministic validation-sampler core.
//!
//! Single concern: given the per-in-band-section variant counts (in
//! traversal order), the batch size `B`, and the running xoshiro256**
//! stream state threaded across reader calls, produce the validation
//! sampler's flat shuffled+kept variant indices, the per-bunch CSR
//! boundaries, the owning section per bunch, and the advanced RNG state.
//!
//! The ONLY random operation is a per-section in-place Fisher-Yates
//! shuffle drawing unbiased bounded randoms (Lemire reduction) from ONE
//! shared xoshiro256** state that is advanced across the sections in
//! order (NOT reseeded per section, NOT seed+offset). After all sections
//! the advanced state is returned so the caller threads it into the next
//! reader call — a single continuous stream across files.
//!
//! ## Per-section emission (section `i`, variant count `n_i`)
//!
//! Build `idx = [0..n_i)`, Fisher-Yates shuffle it in place against the
//! shared stream, compute `keep = (n_i / B) * B` (integer floor times
//! `B`), append `idx[0..keep]` to the flat `variant_idx` vec, and emit
//! `keep / B` bunches — each pushing one `B`-step into `bunch_offsets`
//! (an exclusive prefix sum, every step exactly `B`) and one `i` into
//! `bunch_section`. A section with `n_i < B` keeps nothing (zero bunches,
//! whole short section dropped); `n_i <= 0` likewise emits zero bunches.
//!
//! ## RNG (hand-rolled — no `rand` crate)
//!
//! splitmix64 (the seed -> 4-word state derivation the Python side uses;
//! also exposed here for any seeding helper), xoshiro256** `next` (the
//! standard rotl-based variant), and Lemire's unbiased bounded
//! `rand_below(k)` via a 128-bit multiply. The pure-Python oracle in
//! `tokenizer/aligned_data/sorted_index/_sampler/_validation_oracle.py`
//! reimplements the SAME math and is the canonical bit-identity target.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// xoshiro256** running state (4 u64 words), threaded across calls.
struct Xoshiro256ss {
    s: [u64; 4],
}

impl Xoshiro256ss {
    fn from_state(s: [u64; 4]) -> Self {
        Xoshiro256ss { s }
    }

    /// Standard xoshiro256** `next`: the rotl(s1*5,7)*9 scrambler plus the
    /// xor/shift/rotl state advance.
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Lemire's unbiased bounded random in `[0, k)` (k >= 1) via a 128-bit
    /// multiply with the canonical rejection of the low-zone bias. For
    /// `k == 0` returns 0 (never drawn: Fisher-Yates always passes `j+1`).
    fn rand_below(&mut self, k: u64) -> u64 {
        if k == 0 {
            return 0;
        }
        let mut x = self.next_u64();
        let mut m = (x as u128) * (k as u128);
        let mut l = m as u64; // low 64 bits
        if l < k {
            // threshold = (2^64 - k) % k == (-k) mod k.
            let t = k.wrapping_neg() % k;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (k as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// splitmix64 step (seed -> next splitmix output + advanced seed). Used to
/// derive the initial 4-word xoshiro256** state from a single seed.
fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive the canonical 4-word xoshiro256** state from one seed (4
/// splitmix64 draws). Exposed for the seeding helper; the kernel itself
/// consumes an already-derived 4-word state.
#[allow(dead_code)]
fn derive_initial_state(seed: u64) -> [u64; 4] {
    let mut s = seed;
    [
        splitmix64(&mut s),
        splitmix64(&mut s),
        splitmix64(&mut s),
        splitmix64(&mut s),
    ]
}

/// The three flat outputs plus the advanced RNG state, in return order.
#[cfg_attr(test, derive(Debug))]
struct VariantShuffleOut {
    variant_idx: Vec<i64>,
    bunch_offsets: Vec<i64>,
    bunch_section: Vec<i64>,
    state_out: [u64; 4],
}

/// Pure-Rust core (no PyO3 in the signature) so unit tests drive it
/// directly. Mirrors the pure-Python oracle's shuffle+chunk+drop exactly.
fn run_kernel(n_variants: &[i64], batch_size: i64, state_in: [u64; 4]) -> VariantShuffleOut {
    let b = batch_size as usize; // batch_size >= 1 validated by the wrapper.
    let mut rng = Xoshiro256ss::from_state(state_in);

    let mut variant_idx: Vec<i64> = Vec::new();
    let mut bunch_offsets: Vec<i64> = vec![0];
    let mut bunch_section: Vec<i64> = Vec::new();

    for (i, &n_i64) in n_variants.iter().enumerate() {
        if n_i64 <= 0 {
            continue;
        }
        let n = n_i64 as usize;

        // idx = [0..n); Fisher-Yates shuffle in place against the shared
        // stream: for j in (1..n).rev() { k = rand_below(j+1); swap(j,k) }.
        let mut idx: Vec<i64> = (0..n as i64).collect();
        for j in (1..n).rev() {
            let k = rng.rand_below((j + 1) as u64) as usize;
            idx.swap(j, k);
        }

        // keep = floor(n / B) * B; emit keep/B bunches of exactly B.
        let n_bunches = n / b;
        let keep = n_bunches * b;
        for v in &idx[..keep] {
            variant_idx.push(*v);
        }
        let mut last = *bunch_offsets.last().unwrap();
        for _ in 0..n_bunches {
            last += b as i64;
            bunch_offsets.push(last);
            bunch_section.push(i as i64);
        }
    }

    VariantShuffleOut {
        variant_idx,
        bunch_offsets,
        bunch_section,
        state_out: rng.s,
    }
}

/// PyO3 wrapper: borrow the per-section variant counts + the 4-word RNG
/// state, run the shuffle+chunk+drop under `py.detach`, and return the
/// tuple `(variant_idx, bunch_offsets, bunch_section, state_out)`.
#[pyfunction]
pub fn variant_shuffle_chunk_kernel<'py>(
    py: Python<'py>,
    n_variants: numpy::PyReadonlyArray1<'py, i64>,
    batch_size: i64,
    state_in: numpy::PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyTuple>> {
    if batch_size < 1 {
        return Err(PyValueError::new_err(format!(
            "batch_size must be >= 1, got {batch_size}"
        )));
    }
    let state_slice = state_in.as_slice()?;
    if state_slice.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "state_in must have length 4, got {}",
            state_slice.len()
        )));
    }
    let state_arr: [u64; 4] = [
        state_slice[0],
        state_slice[1],
        state_slice[2],
        state_slice[3],
    ];
    let n_variants = n_variants.as_slice()?;

    let out = py.detach(|| run_kernel(n_variants, batch_size, state_arr));

    let arrays: [Bound<'py, PyAny>; 4] = [
        PyArray1::from_vec(py, out.variant_idx).into_any(),
        PyArray1::from_vec(py, out.bunch_offsets).into_any(),
        PyArray1::from_vec(py, out.bunch_section).into_any(),
        PyArray1::from_vec(py, out.state_out.to_vec()).into_any(),
    ];
    PyTuple::new(py, arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed seed-derived state used across the determinism teeth.
    fn state(seed: u64) -> [u64; 4] {
        derive_initial_state(seed)
    }

    /// Every bunch_offsets step is exactly B and the last offset equals the
    /// flat variant_idx length (no ragged bunches).
    fn assert_well_formed(out: &VariantShuffleOut, b: i64) {
        for w in out.bunch_offsets.windows(2) {
            assert_eq!(w[1] - w[0], b, "ragged bunch step: {:?}", out.bunch_offsets);
        }
        assert_eq!(out.bunch_offsets[0], 0);
        assert_eq!(
            *out.bunch_offsets.last().unwrap(),
            out.variant_idx.len() as i64,
            "bunch_offsets tail != variant_idx len"
        );
        assert_eq!(out.bunch_section.len(), out.bunch_offsets.len() - 1);
    }

    /// Golden vector: the exact (n_variants, B, state) -> output triple.
    /// Expected values are the bit-identical oracle output, hand-verified:
    /// the shuffle is deterministic from the splitmix64-derived state and
    /// keep = floor(n/B)*B. n = [5, 2, 7], B = 2.
    ///   sec0 n=5 -> keep 4 -> 2 bunches
    ///   sec1 n=2 -> keep 2 -> 1 bunch
    ///   sec2 n=7 -> keep 6 -> 3 bunches
    #[test]
    fn golden_vector() {
        let out = run_kernel(&[5, 2, 7], 2, state(0xDEAD_BEEF));
        assert_well_formed(&out, 2);
        // 6 bunches total -> 7 offsets, 12 flat indices.
        assert_eq!(out.bunch_offsets.len(), 7);
        assert_eq!(out.variant_idx.len(), 12);
        assert_eq!(out.bunch_offsets, vec![0, 2, 4, 6, 8, 10, 12]);
        assert_eq!(out.bunch_section, vec![0, 0, 1, 2, 2, 2]);
        // The flat shuffled+kept indices (golden, frozen from this exact
        // kernel and the pure-Python oracle, which agree bit-for-bit). If
        // the RNG or keep math changes this MUST change with it.
        // sec0 (n=5) keep[0..4], sec1 (n=2) keep[0..2], sec2 (n=7) keep[0..6].
        assert_eq!(out.variant_idx, GOLDEN_VARIANT_IDX.to_vec());
        // The advanced state is non-trivial (stream moved off the seed).
        assert_ne!(out.state_out, state(0xDEAD_BEEF));
        // The advanced state is the frozen golden state.
        assert_eq!(out.state_out, GOLDEN_STATE_OUT);
    }

    /// Wrong-remainder tooth: n_i = 2*B+1 contributes exactly 2 bunches and
    /// 2*B flat indices — the +1 remainder is dropped. An off-by-one keep
    /// (e.g. ceil, or keep = n) fails here.
    #[test]
    fn remainder_dropped() {
        let b = 3i64;
        let n = 2 * b + 1; // 7
        let out = run_kernel(&[n], b, state(1));
        assert_well_formed(&out, b);
        assert_eq!(out.bunch_offsets, vec![0, 3, 6]);
        assert_eq!(out.variant_idx.len() as i64, 2 * b);
        assert_eq!(out.bunch_section, vec![0, 0]);
    }

    /// RNG-advanced tooth: two equal-n sections must produce DIFFERENT
    /// shuffles because the shared stream advanced between them. A
    /// per-section reseed (or seed+offset) bug yields identical halves.
    #[test]
    fn shared_stream_advances_between_sections() {
        let b = 2i64;
        let n = 4i64 * b; // 8, big enough that a shuffle is unlikely to fix-point
        let out = run_kernel(&[n, n], b, state(42));
        assert_well_formed(&out, b);
        let half = out.variant_idx.len() / 2;
        let first = &out.variant_idx[..half];
        let second = &out.variant_idx[half..];
        assert_ne!(
            first, second,
            "equal-n sections shuffled identically => stream not advanced"
        );
        // Both sections are full permutations of [0..n) (kept = n here).
        let mut s0 = first.to_vec();
        s0.sort();
        let mut s1 = second.to_vec();
        s1.sort();
        let ident: Vec<i64> = (0..n).collect();
        assert_eq!(s0, ident);
        assert_eq!(s1, ident);
    }

    /// Drop-short tooth: n_i < B and n_i == 0 both emit zero bunches.
    #[test]
    fn short_and_empty_sections_dropped() {
        let b = 4i64;
        let out = run_kernel(&[3, 0, 4, 1], b, state(7));
        assert_well_formed(&out, b);
        // Only the n=4 section survives -> 1 bunch, owned by section ord 2.
        assert_eq!(out.bunch_offsets, vec![0, 4]);
        assert_eq!(out.bunch_section, vec![2]);
        assert_eq!(out.variant_idx.len(), 4);
    }

    /// Negative n_i is treated as empty (zero bunches), never panics.
    #[test]
    fn negative_count_is_empty() {
        let out = run_kernel(&[-5, 4], 4, state(9));
        assert_well_formed(&out, 4);
        assert_eq!(out.bunch_section, vec![1]);
        assert_eq!(out.variant_idx.len(), 4);
    }

    /// A bunch's flat slice is a contiguous B-window of the section's
    /// shuffled prefix — paired offsets/section align.
    #[test]
    fn offsets_partition_flat_indices() {
        let out = run_kernel(&[5, 7], 2, state(123));
        assert_well_formed(&out, 2);
        // 2 + 3 = 5 bunches, 10 flat indices.
        assert_eq!(out.bunch_section.len(), 5);
        assert_eq!(out.variant_idx.len(), 10);
    }
}

// Frozen golden output for n=[5,2,7], B=2, seed=0xDEADBEEF, printed once
// from this exact kernel (and matched bit-for-bit by the Python oracle).
#[cfg(test)]
const GOLDEN_VARIANT_IDX: [i64; 12] = [4, 2, 0, 1, 0, 1, 2, 0, 6, 3, 1, 5];
#[cfg(test)]
const GOLDEN_STATE_OUT: [u64; 4] = [
    0xa02c_3c53_6c7b_65b4,
    0x8aff_60c0_c2ae_8bea,
    0xf038_3cd6_4353_b256,
    0x287f_1b38_3303_414d,
];
