"""Vectorized IEEE-754 binary128 (F128) encoder.

Single concern: 16-byte big-endian payload -> f96-shape normalization for
binary128. Emits **2 chunks per finite source** (low + high u64 limb of the
113-bit effective mantissa) and **1 chunk per NaN/Inf source** -- this is
the **fixed-layout rule** committed by the batch pipeline's stage 2 (ALG-2
``f128_chunk_counts``).

Per-chunk normalization formula is identical to :func:`custom_float._emit_chunk`
on the corresponding ``chunk_value`` -- so for F128 *normals* (whose
effective mantissa has bit 112 set) the per-chunk emission is **byte-
identical** to :func:`custom_float.from_float128`. The divergence is only
on F128 ``+/-0`` and denormals whose ``effective_mantissa.bit_length() <=
64``: the oracle's ``_split_to_chunks`` short-circuits those to 1 chunk,
while the batch path emits 2 (with the high chunk being a signed zero).

Branches:

* **NaN / Inf** (``biased_exp == 0x7FFF``, per ``f128_is_nan_or_inf[k] ==
  True``): single-chunk sentinel via :func:`encode_infnan_vec`. Inf detected
  by ``high_mantissa == 0 AND low_u64 == 0``.
* **Denormal** (``biased_exp == 0``): ``effective_mantissa = raw_mantissa``;
  ``actual_exp = -16382``. Both u64 chunks emitted; either may be zero.
* **Normal**: ``effective_mantissa = (1 << 112) | raw_mantissa``;
  ``actual_exp = biased_exp - 16383``. Low chunk = low 64 bits of raw
  mantissa; high chunk = ``(1 << 48) | (top 48 bits of raw mantissa)``.

Per-chunk exponent base (matching the oracle's ``_split_to_chunks`` path):

* chunk 0 (low) base = ``actual_exp - 112``
* chunk 1 (high) base = ``actual_exp - 48``  (= base + 64)
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

    ``raw_bytes_2d``: ``u8[n_sources, 16]`` -- one row per source, 16 big-
    endian bytes per row.

    ``f128_is_nan_or_inf``: ``bool[n_sources]`` -- aligned with the rows of
    ``raw_bytes_2d``. Per stage 2's ALG-2 -- determines whether each source
    contributes 1 chunk (NaN/Inf) or 2 chunks (finite).

    Returns ``(significand, sign_exp)`` arrays of total length =
    ``2 * (~is_nan_or_inf).sum() + is_nan_or_inf.sum()``, ordered per source
    in input order with finite sources contributing ``[chunk0, chunk1]``
    consecutively and NaN/Inf sources contributing ``[infnan]``.
    """
    n_sources = raw_bytes_2d.shape[0]
    if n_sources == 0:
        return (
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.uint32),
        )

    bytes_c = np.ascontiguousarray(raw_bytes_2d)
    # u64[n_sources, 2]; limb 0 = HIGH (bytes 0..7), limb 1 = LOW (bytes 8..15).
    u64_limbs = bytes_c.view(">u8").reshape(-1, 2)
    high_u64 = u64_limbs[:, 0]
    low_u64 = u64_limbs[:, 1]

    # Field extraction from the high u64:
    #   sign at bit 63
    #   biased_exp at bits 48..62 (15 bits)
    #   high-mantissa at bits 0..47 (48 bits)
    sign_bit = (high_u64 >> np.uint64(63)) & np.uint64(1)
    is_negative = sign_bit.astype(bool)
    biased_exp = (high_u64 >> np.uint64(48)) & np.uint64(0x7FFF)
    high_mantissa = high_u64 & np.uint64((1 << 48) - 1)

    # Sanity: f128_is_nan_or_inf must agree with the bit pattern.
    expected_nan_or_inf = biased_exp == np.uint64(0x7FFF)
    if (expected_nan_or_inf != f128_is_nan_or_inf).any():
        raise AssertionError(
            "f128_is_nan_or_inf disagrees with bit-pattern detection; "
            "stage-2 sidecar drift from the actual payload"
        )

    # ---- NaN/Inf branch (per source -- emit 1 chunk each) ----
    nan_inf_mask = f128_is_nan_or_inf.astype(bool)
    # Inf is "all mantissa bits zero" (after stripping sign+exp); NaN is
    # anything else.
    is_inf = (high_mantissa == np.uint64(0)) & (low_u64 == np.uint64(0))
    infnan_sig_all, infnan_sign_exp_all = encode_infnan_vec(
        is_negative, is_inf
    )

    # ---- Finite branch (per source -- emit 2 chunks each) ----
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

    # ---- Assemble output in per-source chunk order ----
    chunks_per_source = np.where(
        nan_inf_mask, np.int64(1), np.int64(2)
    )
    out_offsets = np.empty(n_sources + 1, dtype=np.int64)
    out_offsets[0] = 0
    np.cumsum(chunks_per_source, out=out_offsets[1:])
    total_chunks = int(out_offsets[-1])

    out_sig = np.empty(total_chunks, dtype=np.uint64)
    out_sign_exp = np.empty(total_chunks, dtype=np.uint32)

    # Finite-source rows: place chunk0 at out_offsets[k] + 0, chunk1 at + 1.
    finite_idx = np.nonzero(~nan_inf_mask)[0]
    if finite_idx.size > 0:
        starts_finite = out_offsets[finite_idx]
        out_sig[starts_finite] = finite_chunk0_sig[finite_idx]
        out_sign_exp[starts_finite] = finite_chunk0_sign_exp[finite_idx]
        out_sig[starts_finite + 1] = finite_chunk1_sig[finite_idx]
        out_sign_exp[starts_finite + 1] = finite_chunk1_sign_exp[finite_idx]

    nan_inf_idx = np.nonzero(nan_inf_mask)[0]
    if nan_inf_idx.size > 0:
        starts_nan = out_offsets[nan_inf_idx]
        out_sig[starts_nan] = infnan_sig_all[nan_inf_idx]
        out_sign_exp[starts_nan] = infnan_sign_exp_all[nan_inf_idx]

    return out_sig, out_sign_exp
