"""Vectorized IEEE-754 binary128 (F128) encoder.

Single concern: per-chunk 8-byte big-endian payload -> f96-shape
normalization for binary128. Consumes the per-chunk layout 3c emits
(``(n_chunks_total, 8)`` u8 -- 2 chunks per finite source INDEPENDENT
of the cut, 1 chunk per NaN/Inf source) and pairs it with the per-
source ``f128_is_nan_or_inf`` sidecar to split chunks back into their
source roles. ``chunks_per_source = where(is_nan_or_inf, 1, 2)`` so
the MSB bytes are always available for the LSB chunk's exponent base
derivation -- mid-cut finite sources still contribute 2 chunks here;
stage 4's per-row sidecar walk drops the invisible MSB at concat
time.

Per-chunk emission formula matches :func:`custom_float._emit_chunk` on
the corresponding ``chunk_value`` -- so for F128 *normals* (whose
effective mantissa has bit 112 set) the per-chunk emission is **byte-
identical** to :func:`custom_float.from_float128`. The divergence is only
on F128 ``+/-0`` and denormals whose ``effective_mantissa.bit_length() <=
64``: the oracle's ``_split_to_chunks`` short-circuits those to 1 chunk,
while the batch path always emits 2 (the high chunk being a signed
zero). This is the **fixed-layout rule** committed by stage 2's ALG-2
``f128_chunk_counts`` -- 3c emits 2 chunks for every finite source.

Branches:

* **NaN / Inf** (``f128_is_nan_or_inf[s] == True``): single chunk per
  source = MSB u64 limb (bytes 0..7). Inf/NaN distinction uses ONLY the
  high mantissa (bits [0, 48) of the MSB limb) -- the ``.rodata-
  robustness`` policy (``custom_float.py:336`` + ``_number_decode.py:
  21-25``): canonical NaN/Inf encoding is fully determined by the
  high 8 bytes, so 3c can safely drop the low limb.
* **Denormal** (``biased_exp == 0``): ``effective_mantissa =
  raw_mantissa``; ``actual_exp = -16382``. Both u64 chunks emitted;
  either may be zero.
* **Normal**: ``effective_mantissa = (1 << 112) | raw_mantissa``;
  ``actual_exp = biased_exp - 16383``. Low chunk = low 64 bits of raw
  mantissa; high chunk = ``(1 << 48) | (top 48 bits of raw mantissa)``.

Per-chunk exponent base (matching the oracle's ``_split_to_chunks`` path):

* chunk 0 (low) base = ``actual_exp - 112``
* chunk 1 (high) base = ``actual_exp - 48``  (= base + 64)

Per-source chunk-position layout in the input ``raw_bytes_2d``:

* Finite source (``f128_is_nan_or_inf[s] == False``): 2 rows --
  ``out_offsets[s] + 0`` = LSB limb (bytes 8..15 of the original 16-
  byte payload), ``out_offsets[s] + 1`` = MSB limb (bytes 0..7).
  Always 2 rows, even for mid-cut finite sources.
* NaN/Inf source (``f128_is_nan_or_inf[s] == True``): 1 row at
  ``out_offsets[s]`` = MSB limb (bytes 0..7).

where ``out_offsets`` is the cumsum of ``chunks_per_source = where(
is_nan_or_inf, 1, 2)``.
"""

from __future__ import annotations

import numpy as np

from ._primitives import emit_chunk_vec, encode_infnan_vec

__all__ = ["normalize_f128"]


def normalize_f128(
    raw_bytes_2d: np.ndarray,
    f128_is_nan_or_inf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized F128 encoder.

    ``raw_bytes_2d``: ``u8[n_chunks_total, 8]`` -- one row per CHUNK (not
    per source). Layout per source: finite sources contribute 2 rows
    (LSB then MSB limb) INDEPENDENT of the per-row cutoff; NaN/Inf
    sources contribute 1 row (MSB limb). Rows appear in source order,
    with each source's chunks consecutive and in LSB-first order for
    finite sources.

    ``f128_is_nan_or_inf``: ``bool[n_sources]`` -- routes the per-source
    dispatch (NaN/Inf path vs finite path) AND drives
    ``chunks_per_source = where(is_nan_or_inf, 1, 2)``. 3c emits the
    full ALG-2 chunk set for every finite source so 3d can read
    ``actual_exp`` from the MSB limb; stage 4's per-row sidecar concat
    drops the trailing invisible MSB chunk for a mid-cut finite source.

    Returns ``(significand, sign_exp)`` arrays of length
    ``n_chunks_total``, in input row order.
    """
    n_sources = int(f128_is_nan_or_inf.shape[0])
    n_chunks_total = int(raw_bytes_2d.shape[0])
    if n_sources == 0:
        if n_chunks_total != 0:
            raise AssertionError(
                "raw_bytes_2d has rows but f128_is_nan_or_inf is empty"
            )
        return (
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.uint32),
        )

    # ---- Per-source chunk-layout map ----
    #
    # chunks_per_source[s] = where(is_nan_or_inf, 1, 2): finite sources
    # contribute 2 chunks (LSB + MSB) INDEPENDENT of the cut; NaN/Inf
    # sources contribute 1 chunk (MSB only).
    # out_offsets[s]: index of source s's first chunk in the per-chunk
    # arrays. out_offsets[n_sources] = total chunk count.
    nan_inf_mask = f128_is_nan_or_inf.astype(bool)
    chunks_per_source = np.where(nan_inf_mask, 1, 2).astype(np.int64)
    out_offsets = np.empty(n_sources + 1, dtype=np.int64)
    out_offsets[0] = 0
    np.cumsum(chunks_per_source, out=out_offsets[1:])
    expected_chunks_total = int(out_offsets[-1])
    if expected_chunks_total != n_chunks_total:
        raise AssertionError(
            f"raw_bytes_2d row count {n_chunks_total} does not match "
            f"is_nan_or_inf-derived chunk count "
            f"{expected_chunks_total}; stage-3c / stage-3d layout drift"
        )

    # Per-chunk view: each row's 8 big-endian bytes -> 1 u64.
    bytes_c = np.ascontiguousarray(raw_bytes_2d)
    chunk_u64 = bytes_c.view(">u8").reshape(-1)  # u64[n_chunks_total]

    # Per-source MSB chunk position = source's LAST chunk row:
    #   NaN/Inf source (chunks=1): source_starts (the only chunk).
    #   Finite source  (chunks=2): source_starts + 1 (the 2nd chunk).
    # 3c always emits both finite chunks (mid-cut finite sources still
    # contribute 2 chunks here), so the MSB position is unambiguous.
    source_starts = out_offsets[:n_sources]
    msb_chunk_pos = source_starts + (chunks_per_source - np.int64(1))
    msb_u64 = chunk_u64[msb_chunk_pos]

    # Field extraction from the MSB u64 (per source):
    #   sign at bit 63
    #   biased_exp at bits 48..62 (15 bits)
    #   high-mantissa at bits 0..47 (48 bits)
    sign_bit = (msb_u64 >> np.uint64(63)) & np.uint64(1)
    is_negative = sign_bit.astype(bool)
    biased_exp = (msb_u64 >> np.uint64(48)) & np.uint64(0x7FFF)
    high_mantissa = msb_u64 & np.uint64((1 << 48) - 1)

    # Sanity: f128_is_nan_or_inf must agree with the bit pattern.
    expected_nan_or_inf = biased_exp == np.uint64(0x7FFF)
    if (expected_nan_or_inf != nan_inf_mask).any():
        raise AssertionError(
            "f128_is_nan_or_inf disagrees with bit-pattern detection; "
            "stage-2 sidecar drift from the actual payload"
        )

    # ---- NaN/Inf branch (per source -- emit 1 chunk each) ----
    # .rodata-robustness policy: classify Inf vs NaN using only the high
    # mantissa (3c does not transmit the low limb for NaN/Inf sources).
    # See module docstring + _number_decode.py:21-25 for the contract.
    is_inf = high_mantissa == np.uint64(0)
    infnan_sig_per_source, infnan_sign_exp_per_source = encode_infnan_vec(
        is_negative, is_inf
    )

    # ---- Finite branch (per source -- emit 2 chunks each) ----
    # Per-source LSB u64 lives at the source-start offset for finite
    # sources. For NaN/Inf sources the LSB position is undefined (the
    # row doesn't exist); we still index safely by clamping to the MSB
    # position (the value is gathered but discarded by the finite-mask
    # write below).
    lsb_chunk_pos = np.where(nan_inf_mask, msb_chunk_pos, source_starts)
    low_u64 = chunk_u64[lsb_chunk_pos]

    # Effective mantissa: bit 112 set for normals; absent for denormals.
    is_denormal = biased_exp == np.uint64(0)
    bias = 16383
    actual_exp = np.where(
        is_denormal,
        np.int64(1 - bias),
        biased_exp.astype(np.int64) - np.int64(bias),
    )

    # Per chunk:
    #   low_chunk  = bits [0, 64) of effective_mantissa = low_u64
    #   high_chunk = bits [64, 128) of effective_mantissa.
    #                For normals: (1 << 48) | high_mantissa (implicit-1 sits
    #                at bit 112 of effective_mantissa, i.e. bit 48 of
    #                high_chunk).
    #                For denormals: high_mantissa.
    high_chunk = np.where(
        is_denormal,
        high_mantissa,
        high_mantissa | np.uint64(1 << 48),
    ).astype(np.uint64)
    low_chunk = low_u64

    # Per chunk exponent_base:
    #   _encode_fp_normalized's base_exponent_unbiased = actual_exp -
    #     mantissa_bits = actual_exp - 112.
    #   _split_to_chunks adds 64*k to the base for chunk k.
    chunk0_base = actual_exp - np.int64(112)
    chunk1_base = chunk0_base + np.int64(64)

    finite_chunk0_sig, finite_chunk0_sign_exp = emit_chunk_vec(
        low_chunk, is_negative, chunk0_base
    )
    finite_chunk1_sig, finite_chunk1_sign_exp = emit_chunk_vec(
        high_chunk, is_negative, chunk1_base
    )

    # ---- Scatter per-source results back to per-chunk output ----
    out_sig = np.empty(n_chunks_total, dtype=np.uint64)
    out_sign_exp = np.empty(n_chunks_total, dtype=np.uint32)

    # Finite sources: chunk 0 (LSB) at source_starts and chunk 1
    # (MSB) at source_starts + 1.
    finite_idx = np.nonzero(~nan_inf_mask)[0]
    if finite_idx.size > 0:
        finite_starts = source_starts[finite_idx]
        out_sig[finite_starts] = finite_chunk0_sig[finite_idx]
        out_sign_exp[finite_starts] = finite_chunk0_sign_exp[finite_idx]
        out_sig[finite_starts + 1] = finite_chunk1_sig[finite_idx]
        out_sign_exp[finite_starts + 1] = finite_chunk1_sign_exp[finite_idx]

    nan_inf_idx = np.nonzero(nan_inf_mask)[0]
    if nan_inf_idx.size > 0:
        nan_inf_starts = source_starts[nan_inf_idx]
        out_sig[nan_inf_starts] = infnan_sig_per_source[nan_inf_idx]
        out_sign_exp[nan_inf_starts] = infnan_sign_exp_per_source[nan_inf_idx]

    return out_sig, out_sign_exp
