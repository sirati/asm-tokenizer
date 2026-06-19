"""Pure-Python reference for the deterministic validation-sampler core.

Single concern: the reference math. This reimplements the SAME
splitmix64 seed -> 4-word state derivation, xoshiro256** ``next``,
Lemire unbiased bounded draw, per-section Fisher-Yates shuffle, then the
``floor(n/B)*B`` chunk + short-section drop that the Rust
``variant_shuffle_chunk_kernel`` (dedup_hashmap) implements. It is the
canonical bit-identity target: for the same ``(n_variants, B, seed)`` it
produces output identical to the kernel, word-for-word.

No I/O, no decode imports — only integer math masked to 64 bits.
"""

from __future__ import annotations

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_A = 0xBF58476D1CE4E5B9
_SPLITMIX_B = 0x94D049BB133111EB


def _rotl(x: int, k: int) -> int:
    """64-bit left rotate."""
    return ((x << k) | (x >> (64 - k))) & _MASK64


def _splitmix64_step(seed: int) -> tuple[int, int]:
    """One splitmix64 draw: returns ``(output, advanced_seed)``."""
    seed = (seed + _GOLDEN_GAMMA) & _MASK64
    z = seed
    z = ((z ^ (z >> 30)) * _SPLITMIX_A) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_B) & _MASK64
    z = z ^ (z >> 31)
    return z & _MASK64, seed


def derive_initial_state(seed: int) -> tuple[int, int, int, int]:
    """Derive the canonical 4-word xoshiro256** state from one seed.

    Four successive splitmix64 draws, exactly mirroring the Rust
    ``derive_initial_state``.
    """
    s = seed & _MASK64
    out: list[int] = []
    for _ in range(4):
        word, s = _splitmix64_step(s)
        out.append(word)
    return out[0], out[1], out[2], out[3]


class _Xoshiro256ss:
    """xoshiro256** running state, mirroring the Rust scrambler exactly."""

    __slots__ = ("s",)

    def __init__(self, state: tuple[int, ...]) -> None:
        self.s = [w & _MASK64 for w in state]

    def next_u64(self) -> int:
        s = self.s
        result = (_rotl((s[1] * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s[1] << 17) & _MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        # mask back (xors of masked words stay masked; rotl already masks)
        s[0] &= _MASK64
        s[1] &= _MASK64
        s[2] &= _MASK64
        return result

    def rand_below(self, k: int) -> int:
        """Lemire unbiased bounded draw in ``[0, k)`` (k >= 1)."""
        if k == 0:
            return 0
        x = self.next_u64()
        m = x * k
        low = m & _MASK64
        if low < k:
            t = ((-k) & _MASK64) % k
            while low < t:
                x = self.next_u64()
                m = x * k
                low = m & _MASK64
        return m >> 64


def shuffle_chunk_drop(
    n_variants: list[int],
    batch_size: int,
    state: tuple[int, ...],
) -> tuple[list[int], list[int], list[int], tuple[int, int, int, int]]:
    """Reference shuffle + chunk + drop.

    Returns ``(variant_idx, bunch_offsets, bunch_section, state_out)``,
    bit-identical to the Rust kernel for the same inputs.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    rng = _Xoshiro256ss(state)

    variant_idx: list[int] = []
    bunch_offsets: list[int] = [0]
    bunch_section: list[int] = []

    for i, n in enumerate(n_variants):
        if n <= 0:
            continue
        idx = list(range(n))
        for j in range(n - 1, 0, -1):
            k = rng.rand_below(j + 1)
            idx[j], idx[k] = idx[k], idx[j]
        n_bunches = n // batch_size
        keep = n_bunches * batch_size
        variant_idx.extend(idx[:keep])
        last = bunch_offsets[-1]
        for _ in range(n_bunches):
            last += batch_size
            bunch_offsets.append(last)
            bunch_section.append(i)

    s = rng.s
    return variant_idx, bunch_offsets, bunch_section, (s[0], s[1], s[2], s[3])
