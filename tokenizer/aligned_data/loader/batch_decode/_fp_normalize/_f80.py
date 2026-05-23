"""Vectorized x87 80-bit extended-precision (F80) encoder.

Single concern: 10-byte big-endian payload -> f96-shape normalization for
the x87 explicit-leading-bit layout. Always emits exactly one chunk per
source.

Matches :func:`custom_float._encode_fp_normalized` with
``has_explicit_leading_bit=True``, ``mantissa_bits=64``, ``exponent_bits=15``,
``bias=16383``.

Layout (big-endian on the wire):

* bytes [0..1] -- u16 sign+exponent word: ``sign : 1 | biased_exp : 15``
  from MSB downward.
* bytes [2..9] -- 64-bit mantissa with EXPLICIT leading 1 at bit 63 for
  normal values.

Branches:

* **NaN / Inf** (``biased_exp == 0x7FFF``): single-chunk sentinel. Inf is
  ``raw_mantissa == (1 << 63)``; NaN is anything else (matches the oracle's
  ``has_explicit_leading_bit`` Inf detection).
* **Denormal / Unnormal** (``biased_exp == 0`` OR ``explicit_leading == 0``):
  strip the (now-zero or absent) explicit leading bit and renormalize via
  pure left-shift. This matches :func:`custom_float.from_float80`'s policy
  of treating x87 pseudo-denormals + unnormals (invalid in HW since the
  Pentium) as raw-fraction signals -- the asm-tokenizer encounters them as
  ``.rodata`` byte patterns and a crash would be unacceptable.
* **Normal**: ``effective_mantissa = raw_mantissa``;
  ``actual_exp = biased_exp - 16383``. Leading 1 sits at bit 63 of
  ``effective_mantissa`` -- ``shift`` evaluates to 0 in
  :func:`emit_chunk_vec`.

``leading_bit_position = mantissa_bits - 1 = 63`` (explicit-leading path),
so ``chunk_exponent_base = actual_exp - 63``.
"""

from __future__ import annotations

import numpy as np

from ._primitives import emit_chunk_vec, encode_infnan_vec

__all__ = ["normalize_f80"]


def normalize_f80(
    raw_bytes_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized F80 encoder.

    ``raw_bytes_2d``: ``u8[n_sources, 10]`` -- one row per source, 10 big-
    endian bytes per row.

    Returns ``(significand: u64[n_sources], sign_exp: u32[n_sources])``.
    """
    # View as big-endian u16 limbs. Strides may not align for arbitrary input
    # -- if not C-contiguous, ascontiguousarray copies once.
    bytes_c = np.ascontiguousarray(raw_bytes_2d)
    limbs = bytes_c.view(">u2").reshape(-1, 5)  # u16[n_sources, 5]

    sign_exp_word = limbs[:, 0].astype(np.int64)
    # Reassemble 64-bit mantissa from 4 big-endian u16 limbs.
    mantissa_u64 = (
        (limbs[:, 1].astype(np.uint64) << np.uint64(48))
        | (limbs[:, 2].astype(np.uint64) << np.uint64(32))
        | (limbs[:, 3].astype(np.uint64) << np.uint64(16))
        | limbs[:, 4].astype(np.uint64)
    )

    sign_bit = (sign_exp_word >> np.int64(15)) & np.int64(1)
    is_negative = sign_bit.astype(bool)
    biased_exp = sign_exp_word & np.int64(0x7FFF)

    is_nan_or_inf = biased_exp == np.int64(0x7FFF)
    # F80 Inf is raw_mantissa == (1 << 63); anything else is NaN.
    is_inf = is_nan_or_inf & (mantissa_u64 == np.uint64(1 << 63))

    # Denormal / unnormal branch -- biased_exp == 0 OR explicit_leading == 0:
    #   effective_mantissa = raw_mantissa & ((1 << 63) - 1)  (strip top bit)
    #   actual_exp = 1 - bias = -16382
    # Normal branch:
    #   effective_mantissa = raw_mantissa, actual_exp = biased_exp - 16383.
    explicit_leading = (mantissa_u64 >> np.uint64(63)) & np.uint64(1)
    use_denormal_path = (biased_exp == np.int64(0)) | (
        explicit_leading == np.uint64(0)
    )
    bias = 16383
    effective_mantissa = np.where(
        use_denormal_path,
        mantissa_u64 & np.uint64((1 << 63) - 1),
        mantissa_u64,
    ).astype(np.uint64)
    actual_exp = np.where(
        use_denormal_path,
        np.int64(1 - bias),
        biased_exp - np.int64(bias),
    )

    # F80 has has_explicit_leading_bit=True with mantissa_bits=64, so
    # leading_bit_position = mantissa_bits - 1 = 63.
    chunk_exponent_base = actual_exp - np.int64(63)

    finite_sig, finite_sign_exp = emit_chunk_vec(
        effective_mantissa, is_negative, chunk_exponent_base
    )
    infnan_sig, infnan_sign_exp = encode_infnan_vec(is_negative, is_inf)

    sig = np.where(is_nan_or_inf, infnan_sig, finite_sig).astype(np.uint64)
    sign_exp = np.where(
        is_nan_or_inf, infnan_sign_exp, finite_sign_exp
    ).astype(np.uint32)
    return sig, sign_exp
