"""Shared vectorized bit-level primitives for the FP normalizer kernels.

Single concern: the bit-level operations every per-TokenType kernel needs --
leading-1 position for a u64 array, vectorized ``pack_sign_exp``, vectorized
``_emit_chunk`` (chunk normalization to the f96 sidecar shape), and
vectorized ``_encode_infnan`` (NaN/Inf sentinel emission).

These mirror :mod:`tokenizer.aligned_data.loader.decoded.custom_float`'s
:func:`pack_sign_exp` + :func:`_emit_chunk` + :func:`_encode_infnan`,
re-expressed for batch numpy arrays.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.decoded.custom_float import (
    INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS,
)

__all__ = [
    "leading_one_position_u64",
    "pack_sign_exp_vec",
    "emit_chunk_vec",
    "encode_infnan_vec",
    "U31_EXP_MASK",
    "U32_SIGN_BIT",
]


# u31 mask -- the biased exponent occupies the low 31 bits of sign_exp.
U31_EXP_MASK = np.uint32((1 << 31) - 1)
U32_SIGN_BIT = np.uint32(1 << 31)
# Pre-packed NaN/Inf biased exponent (unbiased == INFNAN_EXPONENT_UNBIASED,
# biased == 0x7FFFFFFF).
_INFNAN_BIASED_EXP = np.uint32(INFNAN_EXPONENT_UNBIASED + TARGET_EXPONENT_BIAS)


def leading_one_position_u64(x: np.ndarray) -> np.ndarray:
    """Vectorized position of leading 1 for u64 elements.

    Returns ``bit_length(x) - 1`` for ``x > 0`` (i.e. an integer in
    ``[0, 63]``). Returns ``0`` for ``x == 0`` (callers MUST mask that case
    via the signed-zero branch -- this function does not special-case zero).

    Implementation: binary-search shift-and-mask ladder. Avoids
    ``np.log2(x.astype(np.float64))`` -- f64 carries only 53 mantissa bits,
    so log2 on values >= 2**53 loses the exact leading-bit position. The
    ladder runs in O(log2(64)) = 6 vectorized shift+mask passes.
    """
    pos = np.zeros(x.shape, dtype=np.int32)
    y = x.copy()
    # 32-bit step.
    mask = y >= np.uint64(1 << 32)
    pos = np.where(mask, pos + 32, pos)
    y = np.where(mask, y >> np.uint64(32), y)
    # 16-bit step.
    mask = y >= np.uint64(1 << 16)
    pos = np.where(mask, pos + 16, pos)
    y = np.where(mask, y >> np.uint64(16), y)
    # 8-bit step.
    mask = y >= np.uint64(1 << 8)
    pos = np.where(mask, pos + 8, pos)
    y = np.where(mask, y >> np.uint64(8), y)
    # 4-bit step.
    mask = y >= np.uint64(1 << 4)
    pos = np.where(mask, pos + 4, pos)
    y = np.where(mask, y >> np.uint64(4), y)
    # 2-bit step.
    mask = y >= np.uint64(1 << 2)
    pos = np.where(mask, pos + 2, pos)
    y = np.where(mask, y >> np.uint64(2), y)
    # 1-bit step.
    mask = y >= np.uint64(1 << 1)
    pos = np.where(mask, pos + 1, pos)
    return pos


def pack_sign_exp_vec(
    is_negative: np.ndarray, exponent_unbiased: np.ndarray
) -> np.ndarray:
    """Vectorized :func:`custom_float.pack_sign_exp`.

    ``is_negative``: bool array (True -> sign bit set in bit 31).
    ``exponent_unbiased``: int array of unbiased exponents.

    Returns u32 array with sign at bit 31 + biased exponent in the low 31
    bits. Raises if any biased exponent overflows the u31 field (matches the
    per-source ``pack_sign_exp`` assertion).
    """
    biased = exponent_unbiased.astype(np.int64) + np.int64(TARGET_EXPONENT_BIAS)
    if biased.size and (biased < 0).any():
        raise OverflowError(
            "biased exponent underflows u31 (negative unbiased exceeds bias)"
        )
    if biased.size and (biased > int(U31_EXP_MASK)).any():
        raise OverflowError("biased exponent overflows u31 range")
    sign_word = np.where(is_negative, U32_SIGN_BIT, np.uint32(0))
    return (sign_word | biased.astype(np.uint32)).astype(np.uint32)


def emit_chunk_vec(
    chunk_u64: np.ndarray,
    is_negative: np.ndarray,
    chunk_exponent_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`custom_float._emit_chunk`.

    Per the oracle: zero chunks emit ``(sig=0, sign_exp=pack(sign, base))``;
    non-zero chunks normalize so the leading 1 sits at bit 63, with
    ``exponent_unbiased = base - shift`` where ``shift = 63 - leading_pos``.

    Inputs:
      * ``chunk_u64``: u64 array of chunk values.
      * ``is_negative``: bool array (same shape).
      * ``chunk_exponent_base``: int64 array of base exponents
        (e.g. ``actual_exp - mantissa_leading_bit_position`` for the FP
        path; ``64 * chunk_index_within_source`` for the multi-chunk-int
        path).
    """
    is_zero = chunk_u64 == np.uint64(0)
    leading_pos = leading_one_position_u64(chunk_u64)
    shift = np.where(is_zero, np.int32(0), np.int32(63) - leading_pos)
    # Normalized significand: left-shift to bit 63; for zero chunks shift is 0
    # and chunk_u64 is already 0 -- normalized stays 0. The shift may be 0
    # (already at bit 63 -- e.g. F64 normal with the implicit leading-1
    # inserted) or up to 63 (denormals + small VC2 chunks). u64 left-shift by
    # 0..63 stays in-range.
    normalized = (
        chunk_u64.astype(np.uint64) << shift.astype(np.uint64)
    ) & np.uint64((1 << 64) - 1)
    # For zero chunks the exponent_unbiased == base (no shift); for non-zero
    # it's base - shift.
    exp_unbiased = chunk_exponent_base.astype(np.int64) - shift.astype(np.int64)
    sign_exp = pack_sign_exp_vec(is_negative, exp_unbiased)
    return normalized, sign_exp


def encode_infnan_vec(
    is_negative: np.ndarray, mantissa_is_zero: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`custom_float._encode_infnan`.

    Inf -> sig=0, NaN -> sig=1 (canonical; source NaN payload is dropped).
    Both share the reserved ``INFNAN_EXPONENT_UNBIASED`` exponent.
    """
    sig = np.where(mantissa_is_zero, np.uint64(0), np.uint64(1))
    sign_word = np.where(is_negative, U32_SIGN_BIT, np.uint32(0))
    sign_exp = (sign_word | _INFNAN_BIASED_EXP).astype(np.uint32)
    return sig.astype(np.uint64), sign_exp
