"""Hex-form rendering for NUMBER-band chunk pairs.

Single concern: turn a single ``(significand, sign_exponent)`` chunk
(produced by the custom-float kernel in
:mod:`tokenizer.aligned_data.loader.decoded.custom_float`) into the
``"<basename>:<hex>"`` text shape the inspector emits as
:class:`AsmLine` -- mirroring
:meth:`token_manager._V2FloatInner.to_asm_like` /
:meth:`token_manager.ValuedConstV2Inner.to_asm_like` so the
BatchDecodeBackend's rendering aligns with the FtlBackend's
``Inner.to_asm_like`` text shape.

Phase 1 limitation (plan v2 decisions #17 + #18 + audit B-HIGH-5):
multi-chunk sources (F128 finite, K>1 VC2) render only the MSB chunk's
contribution; trailing chunks are emitted as :class:`AsmLine("...")`
by the walker (one ``AsmLine`` per trailing chunk). The walker --
NOT this module -- drives the trailing-chunk emission.

For each ``TokenType``, this module inverts the encoder's per-chunk
normalization (see ``custom_float._emit_chunk`` /
``_encode_fp_normalized``) to recover the source ``bits`` pattern.
The recovered ``bits`` are zero-padded to the source type's natural
width (``2 * width_bytes`` hex chars) and prefixed with the basename
matching the encoder's ``_get_basename`` output. Phase-2 follow-up
(plan section 11 + decision #17): real multi-chunk reconstruction
moves here via the chunk-count sidecar; for now the MSB-only render
documents the truncation by emitting the ``...`` tail tokens.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tokenizer.aligned_data.loader.decoded.custom_float import (
    INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS,
    reconstruct_chunks,
    unpack_chunk,
)
from tokenizer.tokens import TokenType


__all__ = ["chunks_to_hex_bits", "format_number"]


# Per-IEEE-754 layout for the float TokenTypes that round-trip through a
# single chunk (or the MSB chunk in Phase-1 multi-chunk rendering).
# ``has_explicit_leading_bit`` mirrors the encoder's two mantissa
# conventions (False for IEEE, True for x87 f80).
_IEEE_LAYOUTS: dict[TokenType, Tuple[int, int, int, bool, str]] = {
    # token_type: (mantissa_bits, exponent_bits, bias, has_explicit_leading, basename)
    TokenType.FLOAT16: (10, 5, 15, False, "float16"),
    TokenType.BFLOAT16: (7, 8, 127, False, "bfloat16"),
    TokenType.FLOAT32: (23, 8, 127, False, "float32"),
    TokenType.FLOAT64: (52, 11, 1023, False, "float64"),
    TokenType.FLOAT80: (64, 15, 16383, True, "float80"),
    TokenType.FLOAT128: (112, 15, 16383, False, "float128"),
}


# Source bit-width (in bytes) for each float TokenType; matches the
# encoder's per-Inner ``width_bytes`` classvar. The rendered hex is
# zero-padded to ``2 * width_bytes`` so the visual width matches the
# FtlBackend's ``Inner.to_asm_like`` output.
_WIDTH_BYTES: dict[TokenType, int] = {
    TokenType.FLOAT16: 2,
    TokenType.BFLOAT16: 2,
    TokenType.FLOAT32: 4,
    TokenType.FLOAT64: 8,
    TokenType.FLOAT80: 10,
    TokenType.FLOAT128: 16,
}


def _ieee_bits_from_chunk(
    sig: int,
    sign: int,
    exp_unbiased: int,
    *,
    mantissa_bits: int,
    exponent_bits: int,
    bias: int,
    has_explicit_leading_bit: bool,
) -> int:
    """Recover the source IEEE-754 (or x87 f80) ``bits`` pattern.

    Inverts the encoder's ``_encode_fp_normalized``: from the
    normalized ``(sig, sign, exp_unbiased)`` chunk, recover the raw
    sign/biased-exp/mantissa fields that the encoder consumed.

    Signed zero (``sig == 0``) round-trips to ``+0`` / ``-0`` in source
    layout. NaN/Inf is handled at the dispatch layer above; this
    helper sees only finite-source chunks.
    """
    sign_bit = 0 if sign >= 0 else 1
    if sig == 0:
        return sign_bit << (mantissa_bits + exponent_bits)

    # The encoder normalised the chunk so the leading 1 lands at bit 63.
    # For finite floats the source's effective-mantissa leading 1 sits
    # at a fixed bit position (``mantissa_bits`` for IEEE; ``mantissa_bits-1``
    # for f80 explicit-leading) -- so the encoder's left-shift is fixed.
    leading_bit_position = (
        mantissa_bits - 1 if has_explicit_leading_bit else mantissa_bits
    )
    shift = 63 - leading_bit_position
    if shift < 0:
        raise ValueError(
            f"unsupported leading_bit_position={leading_bit_position} "
            f"(would require sig wider than 64 bits)"
        )
    effective_mantissa = sig >> shift
    actual_exp = exp_unbiased + leading_bit_position + shift

    smallest_normal_exp = 1 - bias
    # Denormal detection: the encoder's denormal/unnormal branch fixes
    # ``actual_exp = 1 - bias`` regardless of input, so the chunk's
    # ``base_exponent_unbiased = actual_exp - leading_bit_position`` is
    # also fixed. The encoder's per-chunk normalization shift is then
    # ``shift = base_exponent_unbiased - exp_unbiased`` (recovered from
    # the stored chunk exponent), and ``raw_mantissa = sig >> shift``.
    denorm_base_exponent = smallest_normal_exp - leading_bit_position
    is_denormal = actual_exp < smallest_normal_exp
    if is_denormal:
        denorm_shift = denorm_base_exponent - exp_unbiased
        if denorm_shift < 0 or denorm_shift > 63:
            raise ValueError(
                f"unexpected denormal shift {denorm_shift} for "
                f"layout(m={mantissa_bits}, e={exponent_bits}, "
                f"bias={bias}, explicit={has_explicit_leading_bit})"
            )
        raw_mantissa = int(sig) >> denorm_shift
        biased_exp = 0
    else:
        biased_exp = actual_exp + bias
        # IEEE: clear the implicit leading 1 from the effective mantissa
        # (top bit at position ``mantissa_bits``). x87: the explicit
        # leading 1 lives at position ``mantissa_bits-1`` and IS part of
        # the stored field, so mask to the full ``mantissa_bits`` width.
        raw_mantissa = effective_mantissa & ((1 << mantissa_bits) - 1)

    raw_mantissa &= (1 << mantissa_bits) - 1
    biased_exp &= (1 << exponent_bits) - 1
    return (
        (sign_bit << (mantissa_bits + exponent_bits))
        | (biased_exp << mantissa_bits)
        | raw_mantissa
    )


def _ieee_nan_inf_bits(
    sign: int,
    sig: int,
    *,
    mantissa_bits: int,
    exponent_bits: int,
    has_explicit_leading_bit: bool,
) -> int:
    """Source bits for the NaN/Inf sentinel chunk.

    Encoder convention (``custom_float._encode_infnan``): ``sig == 0``
    is Inf, ``sig == 1`` is NaN (canonical). Source NaN payload was
    dropped by the encoder; we reconstruct a canonical quiet-NaN bit
    pattern (top mantissa bit set) for the NaN case.
    """
    sign_bit = 0 if sign >= 0 else 1
    biased_exp = (1 << exponent_bits) - 1
    if sig == 0:
        # Inf: mantissa is zero (IEEE) or has just the explicit leading 1 (f80).
        raw_mantissa = (1 << (mantissa_bits - 1)) if has_explicit_leading_bit else 0
    else:
        # NaN: canonical quiet-NaN bit pattern -- top mantissa bit set.
        raw_mantissa = 1 << (mantissa_bits - 1)
    return (
        (sign_bit << (mantissa_bits + exponent_bits))
        | (biased_exp << mantissa_bits)
        | raw_mantissa
    )


def chunks_to_hex_bits(
    token_type: TokenType,
    sig: np.uint64,
    se: np.uint32,
) -> str:
    """Render a single ``(sig, se)`` chunk as ``"<basename>:<hex>"``.

    The MSB-only Phase-1 contract (plan #17): callers pass a single
    chunk pair -- the MSB for multi-chunk sources. For F128 finite
    callers MUST pass the MSB chunk (chunk index 1); the resulting
    hex is the MSB half of the source bits, padded to the source type's
    natural width. Trailing chunks are NOT this helper's concern;
    the walker emits ``AsmLine("...")`` per trailing chunk.

    NaN/Inf (encoder sentinel exponent) and VC2 are dispatched off the
    chunk's unbiased exponent + the token_type; finite IEEE floats go
    through ``_ieee_bits_from_chunk``.
    """
    sig_int, sign, exp_unbiased = unpack_chunk((sig, se))
    if token_type is TokenType.VALUED_CONST_V2:
        # MSB-only render: reconstruct the chunk's contribution to the
        # source integer. For single-chunk VC2 this round-trips the
        # source value exactly; for multi-chunk VC2 (K>1) this is the
        # MSB chunk's contribution (signed-integer value ``sig << exp``).
        value = reconstruct_chunks([(sig, se)])
        # ``reconstruct_chunks`` returns a Fraction; VC2 is non-negative
        # by encoder contract (sign is carried via a separate
        # ``value_negative`` marker upstream) so the magnitude is an
        # integer.
        magnitude = int(value) if value.denominator == 1 else int(value)
        # Mirror the encoder's :meth:`ValuedConstV2Inner.to_asm_like` shape.
        return f"v2:{magnitude:x}"

    layout = _IEEE_LAYOUTS.get(token_type)
    if layout is None:
        raise ValueError(
            f"chunks_to_hex_bits: unsupported NUMBER token_type {token_type!r}"
        )
    mantissa_bits, exponent_bits, bias, has_explicit, basename = layout
    width_bytes = _WIDTH_BYTES[token_type]
    if exp_unbiased == INFNAN_EXPONENT_UNBIASED:
        bits = _ieee_nan_inf_bits(
            sign,
            sig_int,
            mantissa_bits=mantissa_bits,
            exponent_bits=exponent_bits,
            has_explicit_leading_bit=has_explicit,
        )
    else:
        bits = _ieee_bits_from_chunk(
            sig_int,
            sign,
            exp_unbiased,
            mantissa_bits=mantissa_bits,
            exponent_bits=exponent_bits,
            bias=bias,
            has_explicit_leading_bit=has_explicit,
        )
    return f"{basename}:{bits:0{width_bytes * 2}x}"


def format_number(
    token_type: TokenType,
    sig: np.uint64,
    se: np.uint32,
) -> str:
    """Convenience wrapper -- public surface mirroring the plan name.

    Plan section 6 names the helper ``format_number``; we keep both the
    plan-named entry point and the more descriptive
    :func:`chunks_to_hex_bits` so callers can pick the name matching
    their concern (rendering vs. chunk-bit packing).
    """
    return chunks_to_hex_bits(token_type, sig, se)
