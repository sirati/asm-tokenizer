"""Tests for :func:`chunks_to_hex_bits` (NUMBER-band hex rendering).

Cross-backend parity: the BatchDecodeBackend's ``AsmLine`` text shape
MUST match the FtlBackend's ``Inner.to_asm_like`` output for every
NUMBER ``TokenType`` -- pins decision #18 (hex form
``"<basename>:<bits>"``). For F128 finite the Phase-1 fallback
returns the ``"float128:..."`` placeholder; F128 NaN/Inf renders full
hex via :func:`_ieee_nan_inf_bits`.

Plan reference: ``inspector-render-backends.md`` §6 + decisions #17 +
#18 + audit B-MED-9.

These tests live with the rendering helper they pin (the helper lives
in :mod:`tokenizer.aligned_data.loader.decoded.number_hex_format`; the
BatchDecodeBackend's row walker is the sole consumer, so the
backend-tests subdir owns this contract).
"""

from __future__ import annotations

import pytest

from tokenizer.aligned_data.loader.decoded.custom_float import (
    from_bfloat16,
    from_float16,
    from_float32,
    from_float64,
    from_float80,
    from_float128,
    from_int,
)
from tokenizer.aligned_data.loader.decoded.number_hex_format import (
    chunks_to_hex_bits,
)
from tokenizer.tokens import TokenType


def _expected_inner_text(basename: str, bits: int, width_bytes: int) -> str:
    """Reproduce :meth:`_V2FloatInner.to_asm_like` (token_manager.py
    line 1616 + 1620): ``f"{basename}:{bits:0{width*2}x}"``.

    Reproduced inline (not imported) so the parity test would CATCH a
    regression if either side's text formatting drifts -- importing
    the encoder string would short-circuit the cross-backend assertion.
    """
    return f"{basename}:{bits:0{width_bytes * 2}x}"


# ---------------------------------------------------------------------------
# IEEE single-chunk round-trip parity vs ``Inner.to_asm_like``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token_type, encoder, basename, width_bytes, source_bits",
    [
        # FLOAT16 +1.0 = 0x3c00 (sign 0, biased_exp 15, mantissa 0)
        (TokenType.FLOAT16, from_float16, "float16", 2, 0x3C00),
        # FLOAT16 -2.5 = 0xc100 (sign 1, biased_exp 16, mantissa 0x200)
        (TokenType.FLOAT16, from_float16, "float16", 2, 0xC100),
        # BFLOAT16 +1.0 = 0x3f80
        (TokenType.BFLOAT16, from_bfloat16, "bfloat16", 2, 0x3F80),
        # FLOAT32 +1.0 = 0x3f800000
        (TokenType.FLOAT32, from_float32, "float32", 4, 0x3F800000),
        # FLOAT32 -1.5 = 0xbfc00000
        (TokenType.FLOAT32, from_float32, "float32", 4, 0xBFC00000),
        # FLOAT64 +1.0 = 0x3ff0000000000000
        (TokenType.FLOAT64, from_float64, "float64", 8, 0x3FF0000000000000),
        # FLOAT64 pi-ish = 0x400921fb54442d18
        (TokenType.FLOAT64, from_float64, "float64", 8, 0x400921FB54442D18),
    ],
)
def test_ieee_float_round_trip_matches_inner_text(
    token_type: TokenType,
    encoder,
    basename: str,
    width_bytes: int,
    source_bits: int,
) -> None:
    """Encode source ``bits`` through the chunker, render via
    :func:`chunks_to_hex_bits`, assert the text equals the encoder's
    ``Inner.to_asm_like`` output.
    """
    chunks = encoder(source_bits)
    assert len(chunks) == 1, (
        f"single-chunk float {basename} should emit 1 chunk, got {len(chunks)}"
    )
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(token_type, sig, se)
    expected = _expected_inner_text(basename, source_bits, width_bytes)
    assert rendered == expected


def test_float80_finite_hits_placeholder() -> None:
    """f80 mantissa width (64 bits, explicit-leading convention) sits
    AT the helper's ``> 63`` placeholder gate; finite f80 sources
    therefore render as ``"float80:..."`` under Phase-1 (the encoder
    drops one bit of resolution through the chunker, so a single
    chunk pair cannot reconstruct the full 80-bit pattern losslessly).
    This pins the contract so a Phase-2 promotion that adds the
    sidecar reconstruction will deliberately flip this test.
    """
    bits = 0x3FFF8000000000000000  # +1.0 in f80
    chunks = from_float80(bits)
    assert len(chunks) == 1
    sig, se = chunks[0]
    assert chunks_to_hex_bits(TokenType.FLOAT80, sig, se) == "float80:..."


def test_float80_inf_renders_full_hex() -> None:
    """f80 ``+Inf`` -> :func:`_ieee_nan_inf_bits` path -> full 20-hex-
    char bit pattern. Pins that the NaN/Inf dispatch runs BEFORE the
    mantissa-too-wide placeholder gate.
    """
    bits = (0x7FFF << 64) | (1 << 63)  # exp all-ones, explicit-leading set
    chunks = from_float80(bits)
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.FLOAT80, sig, se)
    assert rendered == _expected_inner_text("float80", bits, 10)


# ---------------------------------------------------------------------------
# F128 -- Phase-1 placeholder for finite, full hex for NaN/Inf
# ---------------------------------------------------------------------------


def test_float128_finite_returns_placeholder() -> None:
    """F128 finite source mantissa is 112 bits, wider than 64-bit
    chunk sig; :func:`chunks_to_hex_bits` returns the
    ``"float128:..."`` placeholder per Phase-1 fallback (plan #17).
    Both chunks (lead + MSB) hit the same placeholder dispatch.
    """
    finite_bits = 0x3FFF0000000000000000000000000000  # 1.0 in F128
    chunks = from_float128(finite_bits)
    assert len(chunks) == 2
    lead_sig, lead_se = chunks[0]
    msb_sig, msb_se = chunks[1]
    assert chunks_to_hex_bits(TokenType.FLOAT128, lead_sig, lead_se) == "float128:..."
    assert chunks_to_hex_bits(TokenType.FLOAT128, msb_sig, msb_se) == "float128:..."


def test_float128_nan_full_hex() -> None:
    """F128 NaN -> single-chunk INFNAN sentinel -> full 32-hex-char
    bit pattern via :func:`_ieee_nan_inf_bits`. Encoder canonicalises
    NaN to quiet-NaN (top mantissa bit set).
    """
    # Source NaN: sign=0, biased_exp=0x7fff, top mantissa bit set
    source_nan = (0x7FFF << 112) | (1 << 111)
    chunks = from_float128(source_nan)
    assert len(chunks) == 1
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.FLOAT128, sig, se)
    # Canonical quiet-NaN: sign=0, exp=all-ones (15 bits), mantissa
    # = top bit set => bits = 0x7fff_8000_..._0000
    expected_bits = (0x7FFF << 112) | (1 << 111)
    assert rendered == _expected_inner_text("float128", expected_bits, 16)


def test_float128_inf_full_hex() -> None:
    """F128 +Inf -> single-chunk INFNAN sentinel -> full 32-hex-char
    bit pattern.
    """
    source_inf = 0x7FFF << 112
    chunks = from_float128(source_inf)
    assert len(chunks) == 1
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.FLOAT128, sig, se)
    expected_bits = 0x7FFF << 112  # exp all-ones, mantissa zero
    assert rendered == _expected_inner_text("float128", expected_bits, 16)


def test_float128_negative_inf_full_hex() -> None:
    """Negative-sign carries through to the rendered bit pattern."""
    source_neg_inf = (1 << 127) | (0x7FFF << 112)
    chunks = from_float128(source_neg_inf)
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.FLOAT128, sig, se)
    expected_bits = (1 << 127) | (0x7FFF << 112)
    assert rendered == _expected_inner_text("float128", expected_bits, 16)


# ---------------------------------------------------------------------------
# VC2 -- single-chunk round-trip vs ``ValuedConstV2Inner.to_asm_like``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 0x42, 0xDEADBEEF, 0xFFFFFFFFFFFFFFFF])
def test_valued_const_v2_single_chunk_matches_inner_text(value: int) -> None:
    """VC2 single-chunk values render as ``"v2:<value_hex>"`` -- exact
    parity with :meth:`ValuedConstV2Inner.to_asm_like` (token_manager.py
    line 1551). The encoder's ``from_int`` produces one chunk for
    values < 2**64; the helper recovers the magnitude via
    :func:`reconstruct_chunks`.
    """
    chunks = from_int(value)
    assert len(chunks) == 1
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.VALUED_CONST_V2, sig, se)
    expected = f"v2:{value:x}"  # mirrors ValuedConstV2Inner.to_asm_like
    assert rendered == expected


# ---------------------------------------------------------------------------
# NaN/Inf for the smaller IEEE float types
# ---------------------------------------------------------------------------


def test_float32_inf_renders_full_hex() -> None:
    """F32 Inf -> ``_ieee_nan_inf_bits`` path -> full hex (not a
    placeholder). Pins that single-chunk INFNAN dispatch lands on
    the bit-pattern recovery, not the mantissa-too-wide branch.
    """
    source_inf = 0x7F800000
    chunks = from_float32(source_inf)
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.FLOAT32, sig, se)
    assert rendered == _expected_inner_text("float32", source_inf, 4)


def test_float64_quiet_nan_renders_canonical() -> None:
    """F64 NaN -> canonical quiet-NaN top-mantissa-bit-set pattern."""
    source_nan = (0x7FF << 52) | (1 << 51)  # sign 0, exp all-ones, msb set
    chunks = from_float64(source_nan)
    sig, se = chunks[0]
    rendered = chunks_to_hex_bits(TokenType.FLOAT64, sig, se)
    expected_bits = (0x7FF << 52) | (1 << 51)
    assert rendered == _expected_inner_text("float64", expected_bits, 8)


# ---------------------------------------------------------------------------
# Defensive: unknown TokenType
# ---------------------------------------------------------------------------


def test_unsupported_token_type_raises() -> None:
    """Passing a non-NUMBER ``TokenType`` (e.g. an identity token type)
    raises :class:`ValueError`; the helper is concerned ONLY with the
    NUMBER block.
    """
    chunks = from_float32(0x3F800000)
    sig, se = chunks[0]
    with pytest.raises(ValueError, match="unsupported NUMBER token_type"):
        chunks_to_hex_bits(TokenType.UNRESOLVED, sig, se)
