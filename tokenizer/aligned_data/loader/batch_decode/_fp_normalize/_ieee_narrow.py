"""Vectorized IEEE-754 narrow-width encoder for F16 / BF16 / F32 / F64.

Single concern: bit-pattern -> f96-shape normalization for IEEE widths whose
effective mantissa (``mantissa_bits + 1``) fits in u64 (53 bits worst case
for F64). Always emits exactly one chunk per source.

Matches :func:`custom_float._encode_fp_normalized` with
``has_explicit_leading_bit=False`` for ``effective_width <= 64``.
"""

from __future__ import annotations

import numpy as np

from ._primitives import emit_chunk_vec, encode_infnan_vec

__all__ = ["normalize_ieee_narrow"]


def normalize_ieee_narrow(
    bits: np.ndarray,
    *,
    mantissa_bits: int,
    exponent_bits: int,
    bias: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized IEEE-754 narrow-width encoder.

    ``bits``: integer dtype array of raw bit patterns (u16 / u32 / u64). Each
    element holds ``[sign : 1 | biased_exp : exponent_bits | mantissa :
    mantissa_bits]`` from MSB downward.

    Per-format parameters (matching the per-source encoders in
    :mod:`custom_float`):

    * F16:  ``mantissa_bits=10, exponent_bits=5,  bias=15``
    * BF16: ``mantissa_bits=7,  exponent_bits=8,  bias=127``
    * F32:  ``mantissa_bits=23, exponent_bits=8,  bias=127``
    * F64:  ``mantissa_bits=52, exponent_bits=11, bias=1023``

    Branches handled vectorized:

    * **NaN / Inf** (``biased_exp == all-ones``): single-chunk sentinel via
      :func:`encode_infnan_vec`. Inf detected by ``raw_mantissa == 0``;
      anything else is NaN.
    * **Denormal** (``biased_exp == 0``): ``effective_mantissa =
      raw_mantissa``; ``actual_exp = 1 - bias``. The mantissa's leading 1
      may sit anywhere in ``[0, mantissa_bits - 1]``; ``emit_chunk_vec``
      finds it via the leading-zero ladder.
    * **Normal**: ``effective_mantissa = (1 << mantissa_bits) | raw_mantissa``;
      ``actual_exp = biased_exp - bias``. Leading 1 always at bit
      ``mantissa_bits`` of effective_mantissa.

    ``chunk_exponent_base = actual_exp - mantissa_bits`` (the
    ``base_exponent_unbiased`` from the oracle's IEEE path).
    """
    # Promote to int64 for safe arithmetic across the field extraction. The
    # widest input is u64 (F64) -- still fits since the sign bit is extracted
    # before any arithmetic that could overflow.
    b = bits.astype(np.int64)
    mantissa_mask = (1 << mantissa_bits) - 1
    exp_field_mask = (1 << exponent_bits) - 1

    raw_mantissa = b & np.int64(mantissa_mask)
    biased_exp = (b >> np.int64(mantissa_bits)) & np.int64(exp_field_mask)
    sign_bit = (b >> np.int64(mantissa_bits + exponent_bits)) & np.int64(1)
    is_negative = sign_bit.astype(bool)

    is_nan_or_inf = biased_exp == np.int64(exp_field_mask)
    # IEEE path: Inf has raw_mantissa == 0; NaN has any non-zero mantissa.
    is_inf = is_nan_or_inf & (raw_mantissa == np.int64(0))

    # Denormal path: biased_exp == 0. effective_mantissa = raw_mantissa,
    # actual_exp = 1 - bias. (Per oracle.)
    # Normal path: effective_mantissa = (1 << mantissa_bits) | raw_mantissa,
    # actual_exp = biased_exp - bias.
    is_denormal = biased_exp == np.int64(0)
    effective_mantissa = np.where(
        is_denormal,
        raw_mantissa,
        np.int64(1 << mantissa_bits) | raw_mantissa,
    )
    actual_exp = np.where(
        is_denormal,
        np.int64(1 - bias),
        biased_exp - np.int64(bias),
    )

    # base_exponent_unbiased = actual_exp - leading_bit_position;
    # leading_bit_position = mantissa_bits (IEEE path).
    chunk_exponent_base = actual_exp - np.int64(mantissa_bits)

    # Emit normalized chunk for finite values; replace NaN/Inf with the
    # encode_infnan sentinel post-hoc via np.where.
    finite_sig, finite_sign_exp = emit_chunk_vec(
        effective_mantissa.astype(np.uint64),
        is_negative,
        chunk_exponent_base,
    )
    infnan_sig, infnan_sign_exp = encode_infnan_vec(is_negative, is_inf)

    sig = np.where(is_nan_or_inf, infnan_sig, finite_sig).astype(np.uint64)
    sign_exp = np.where(
        is_nan_or_inf, infnan_sign_exp, finite_sign_exp
    ).astype(np.uint32)
    return sig, sign_exp
