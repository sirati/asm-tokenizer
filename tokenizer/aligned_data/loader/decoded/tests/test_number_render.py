"""Tests for :mod:`_number_render` + :mod:`_number_render_collector`.

Pin contract:

* :func:`render_vc2_short` formats ``v:HEX (decimal)`` with sign
  carried by ``value_negative``.
* :func:`render_vc2_full` reconstructs multi-chunk VC2 and renders
  the full magnitude.
* :func:`render_float_short` formats ``f<N>:0.<digits> E<exp>`` at
  AT-MOST :data:`_SHORT_MANTISSA_DIGITS` precision; NaN / Inf /
  signed-zero short-circuit.
* :func:`render_float_full` uses IEEE-754 round-trip-minimum digit
  counts (auto-derived from :data:`_IEEE_LAYOUTS`).
* :func:`needs_precision_expand` returns ``False`` for values that
  fit losslessly in the short precision (the natural-precision
  formatter makes this a simple text compare).
* :class:`_NumberAccumulator` groups K-chunk sources by shifted id,
  auto-flushes on band switch, exposes ``has_pending`` for the
  walker's intra-instruction invariant assert.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
    _FULL_PRECISION_DIGITS,
    _SHORT_MANTISSA_DIGITS,
    _TOKEN_TYPE_TO_PREFIX,
    needs_precision_expand,
    render_float_full,
    render_float_short,
    render_vc2_full,
    render_vc2_short,
)
from tokenizer.aligned_data.loader.decoded._number_render_collector import (
    AccumulatorEmission,
    _NumberAccumulator,
)
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
    _IEEE_LAYOUTS,
)
from tokenizer.tokens import TokenType


# ---------------------------------------------------------------------------
# Module-load tripwires
# ---------------------------------------------------------------------------


def test_full_precision_digits_match_ieee_round_trip_minimum() -> None:
    """Pin the IEEE-754 round-trip-minimum digit count per format.

    Formula: ``ceil(precision_bits * log10(2)) + 1`` where
    ``precision_bits = mantissa_bits + 1`` for IEEE formats and
    ``mantissa_bits`` for x87 f80 (explicit-leading convention).
    Values: F16=5, BF16=4, F32=9, F64=17, F80=21, F128=36.
    """
    expected = {
        TokenType.FLOAT16: 5,
        TokenType.BFLOAT16: 4,
        TokenType.FLOAT32: 9,
        TokenType.FLOAT64: 17,
        TokenType.FLOAT80: 21,
        TokenType.FLOAT128: 36,
    }
    assert _FULL_PRECISION_DIGITS == expected


def test_full_precision_table_covers_all_ieee_layouts() -> None:
    """Adding a new FP TokenType to ``_IEEE_LAYOUTS`` must update the
    full-precision table; the module-load assert pins this contract.
    """
    assert set(_FULL_PRECISION_DIGITS) == set(_IEEE_LAYOUTS)
    assert set(_TOKEN_TYPE_TO_PREFIX) == set(_IEEE_LAYOUTS)


def test_short_mantissa_digits_is_four() -> None:
    """Pin the short-form width constant; if it changes, callers
    that depend on the visual budget need a coordinated update."""
    assert _SHORT_MANTISSA_DIGITS == 4


def test_short_prefixes_match_basenames() -> None:
    """Short prefix = drop ``float`` + prepend ``f``;
    ``bfloat16 -> bf16`` is the one explicit override."""
    assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT16] == "f16"
    assert _TOKEN_TYPE_TO_PREFIX[TokenType.BFLOAT16] == "bf16"
    assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT32] == "f32"
    assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT64] == "f64"
    assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT80] == "f80"
    assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT128] == "f128"


# ---------------------------------------------------------------------------
# VC2 short / full
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, negative, expected",
    [
        (0, False, "v:0 (0)"),
        (0, True, "v:0 (0)"),  # zero is unsigned regardless of marker
        (0x7B, False, "v:7B (123)"),
        (0x7B, True, "v:-7B (-123)"),
        (0xFF, False, "v:FF (255)"),
        (0xDEADBEEF, False, "v:DEADBEEF (3735928559)"),
        (0xDEADBEEF, True, "v:-DEADBEEF (-3735928559)"),
    ],
)
def test_render_vc2_short_signed(value: int, negative: bool, expected: str) -> None:
    assert render_vc2_short(value, negative) == expected


def test_render_vc2_short_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        render_vc2_short(-1, False)


def test_render_vc2_full_single_chunk_matches_short() -> None:
    """A single-chunk VC2 source's full text equals its short text;
    the predicate yields ``False`` (no expansion needed)."""
    chunks = tuple(from_int(0xDEAD))
    short = render_vc2_short(0xDEAD, False)
    full = render_vc2_full(chunks, False)
    assert short == full == "v:DEAD (57005)"
    assert not needs_precision_expand(short, full)


def test_render_vc2_full_multi_chunk() -> None:
    """A 2-chunk VC2 reconstructs the wide magnitude losslessly."""
    big = (0xDEADBEEF << 64) | 0xCAFEBABE_DEAFBEEF
    chunks = tuple(from_int(big))
    assert len(chunks) == 2
    full = render_vc2_full(chunks, False)
    assert full == f"v:{big:X} ({big})"
    # Multi-chunk VC2 conservative (W3-15): the expanded view equals
    # the unbounded magnitude; the "short" form derived from the
    # entire chunk sequence is the SAME text, so no expansion entry.
    assert not needs_precision_expand(full, full)


def test_render_vc2_full_negative_marker() -> None:
    """``value_negative`` flips both the hex and decimal sign."""
    chunks = tuple(from_int(0x100))
    assert render_vc2_full(chunks, True) == "v:-100 (-256)"


# ---------------------------------------------------------------------------
# Float short
# ---------------------------------------------------------------------------


def _f64_bits(x: float) -> int:
    return int.from_bytes(struct.pack(">d", x), "big")


def _f32_bits(x: float) -> int:
    return int.from_bytes(struct.pack(">f", x), "big")


@pytest.mark.parametrize(
    "encoder, token_type, bits, expected_short",
    [
        # IEEE bits for ``1.0`` -- exponent biased=bias, mantissa=0.
        # The natural-precision short form is ``0.1 E1`` regardless of width.
        (from_float16, TokenType.FLOAT16, 0x3C00, "f16:0.1 E1"),
        (from_float32, TokenType.FLOAT32, 0x3F800000, "f32:0.1 E1"),
        (from_float64, TokenType.FLOAT64, 0x3FF0000000000000, "f64:0.1 E1"),
        # F16 ``1.5`` -- 2-digit mantissa fits in 4 digits.
        (from_float16, TokenType.FLOAT16, 0x3E00, "f16:0.15 E1"),
        # F32 ``-2.5``
        (from_float32, TokenType.FLOAT32, 0xC0200000, "f32:-0.25 E1"),
        # BF16 ``2.0`` -- 1-digit mantissa.
        (from_bfloat16, TokenType.BFLOAT16, 0x4000, "bf16:0.2 E1"),
    ],
)
def test_render_float_short_representative(
    encoder, token_type: TokenType, bits: int, expected_short: str
) -> None:
    chunks = tuple(encoder(bits))
    assert render_float_short(token_type, chunks) == expected_short


def test_render_float_short_caps_at_four_digits() -> None:
    """``pi`` exceeds 4 sig digits; short truncates to exactly 4."""
    chunks = tuple(from_float64(_f64_bits(math.pi)))
    assert render_float_short(TokenType.FLOAT64, chunks) == "f64:0.3142 E1"


def test_render_float_short_unsupported_token_type_raises() -> None:
    with pytest.raises(ValueError):
        render_float_short(TokenType.VALUED_CONST_V2, ((np.uint64(0), np.uint32(0)),))


# ---------------------------------------------------------------------------
# Float full -- IEEE-min round-trip
# ---------------------------------------------------------------------------


def test_render_float_full_f80_finite_at_round_trip_precision() -> None:
    """F80 ``1.0`` -- full form shows the natural compact ``0.1 E1``
    (the value's natural precision is 1 digit; the f80 IEEE-min
    digit count of 21 only caps; it does not pad)."""
    # Build the bit pattern via struct.pack of a Python float, then
    # transcribe to f80's explicit-leading layout via an f64 round-
    # trip (1.0 is exactly representable). Bits: sign=0, biased
    # exp=0x3FFF (=16383, bias), mantissa=top bit set (explicit
    # leading 1) + zeros.
    bits_f80 = (0x3FFF << 64) | (1 << 63)
    chunks = tuple(from_float80(bits_f80))
    assert render_float_full(TokenType.FLOAT80, chunks) == "f80:0.1 E1"


def test_render_float_full_f128_finite_at_round_trip_precision() -> None:
    """F128 ``1.0`` is a 2-chunk source; full form reconstructs the
    full magnitude (this validates the reconstruction kernel goes
    through both chunks rather than just the lead)."""
    bits_f128 = 0x3FFF0000000000000000000000000000
    chunks = tuple(from_float128(bits_f128))
    assert len(chunks) == 2  # multi-chunk source
    assert render_float_full(TokenType.FLOAT128, chunks) == "f128:0.1 E1"


def test_render_float_full_f64_pi_at_seventeen_digits() -> None:
    """F64 ``pi`` -- full form uses 17 digits (IEEE round-trip min).
    Decimal's compact form may emit fewer when trailing digits are
    zero, but pi has no trailing zeros so we see the full 17."""
    chunks = tuple(from_float64(_f64_bits(math.pi)))
    full = render_float_full(TokenType.FLOAT64, chunks)
    assert full == "f64:0.31415926535897931 E1"
    # The decimal portion sans prefix has exactly 17 digits.
    decimal_part = full.split(":")[1].split(" ")[0]  # "0.31415..."
    assert len(decimal_part.replace("0.", "")) == 17


def test_render_float_full_short_diverge_for_irrational() -> None:
    """For a value with more than 4 sig digits, short truncates and
    full preserves; the predicate detects this."""
    chunks = tuple(from_float64(_f64_bits(math.pi)))
    short = render_float_short(TokenType.FLOAT64, chunks)
    full = render_float_full(TokenType.FLOAT64, chunks)
    assert short != full
    assert needs_precision_expand(short, full)


# ---------------------------------------------------------------------------
# Specials: NaN / Inf / signed-zero
# ---------------------------------------------------------------------------


def test_render_float_nan_unsigned() -> None:
    """NaN renders ``f<N>:NaN`` -- sign is dropped (W3-15 / Q4).
    Encoder canonicalises NaN sig to 1 regardless of source sign."""
    # Quiet NaN: sign=0, exp=all-ones, top mantissa bit set.
    nan_pos = (0x7FF << 52) | (1 << 51)
    nan_neg = (1 << 63) | nan_pos
    chunks_pos = tuple(from_float64(nan_pos))
    chunks_neg = tuple(from_float64(nan_neg))
    assert render_float_short(TokenType.FLOAT64, chunks_pos) == "f64:NaN"
    assert render_float_short(TokenType.FLOAT64, chunks_neg) == "f64:NaN"
    assert render_float_full(TokenType.FLOAT64, chunks_pos) == "f64:NaN"


def test_render_float_inf_signed() -> None:
    inf_pos = 0x7FF << 52
    inf_neg = (1 << 63) | inf_pos
    chunks_pos = tuple(from_float64(inf_pos))
    chunks_neg = tuple(from_float64(inf_neg))
    assert render_float_short(TokenType.FLOAT64, chunks_pos) == "f64:Inf"
    assert render_float_short(TokenType.FLOAT64, chunks_neg) == "f64:-Inf"
    assert render_float_full(TokenType.FLOAT64, chunks_neg) == "f64:-Inf"


def test_render_float_signed_zero_collapses_to_unsigned() -> None:
    """``+0`` and ``-0`` both render as ``f<N>:0`` (W3-15 zero short)."""
    chunks_pos = tuple(from_float64(0))
    chunks_neg = tuple(from_float64(1 << 63))
    assert render_float_short(TokenType.FLOAT64, chunks_pos) == "f64:0"
    assert render_float_short(TokenType.FLOAT64, chunks_neg) == "f64:0"
    assert render_float_full(TokenType.FLOAT64, chunks_pos) == "f64:0"
    # Predicate: zero never expands (short == full).
    assert not needs_precision_expand("f64:0", "f64:0")


# ---------------------------------------------------------------------------
# `needs_precision_expand` predicate
# ---------------------------------------------------------------------------


def test_needs_precision_expand_false_for_integer_floats() -> None:
    """Integer-valued floats fit in 1 sig digit -- short == full,
    no expansion entry needed."""
    for x in [1.0, 2.0, 8.0, 256.0, -1.0]:
        chunks = tuple(from_float64(_f64_bits(x)))
        short = render_float_short(TokenType.FLOAT64, chunks)
        full = render_float_full(TokenType.FLOAT64, chunks)
        assert short == full, f"x={x}: short={short!r} full={full!r}"
        assert not needs_precision_expand(short, full)


def test_needs_precision_expand_true_for_pi() -> None:
    chunks = tuple(from_float64(_f64_bits(math.pi)))
    short = render_float_short(TokenType.FLOAT64, chunks)
    full = render_float_full(TokenType.FLOAT64, chunks)
    assert needs_precision_expand(short, full)


# ---------------------------------------------------------------------------
# `_NumberAccumulator` -- K-chunk grouping + flush semantics
# ---------------------------------------------------------------------------


def _f64_chunk(x: float):
    chunks = from_float64(_f64_bits(x))
    assert len(chunks) == 1
    return chunks[0]


def test_accumulator_empty_flush_returns_none() -> None:
    acc = _NumberAccumulator()
    assert not acc.has_pending()
    assert acc.flush() is None


def test_accumulator_single_chunk_float_flush() -> None:
    """One f64 chunk feeds + flushes: short + (no expand, since 1.0
    is lossless at 4 digits)."""
    acc = _NumberAccumulator()
    prior = acc.feed(
        token_type=TokenType.FLOAT64,
        shifted_id=42,
        chunk=_f64_chunk(1.0),
    )
    assert prior is None
    assert acc.has_pending()
    emission = acc.flush()
    assert isinstance(emission, AccumulatorEmission)
    assert emission.short_text == "f64:0.1 E1"
    assert emission.precision_entry is None  # short == full
    assert not acc.has_pending()


def test_accumulator_multi_chunk_vc2_groups_by_shifted_id() -> None:
    """Two consecutive VC2 chunks with the same shifted id form ONE
    multi-chunk source; the accumulator emits ONE text at flush."""
    big = (0xDEADBEEF << 64) | 0xCAFEBABE
    chunks = from_int(big)
    assert len(chunks) == 2

    acc = _NumberAccumulator()
    for chunk in chunks:
        prior = acc.feed(
            token_type=TokenType.VALUED_CONST_V2,
            shifted_id=7,
            chunk=chunk,
            value_negative=False,
        )
        assert prior is None  # same shifted id -> no auto-flush
    assert acc.has_pending()

    emission = acc.flush()
    assert isinstance(emission, AccumulatorEmission)
    assert emission.short_text == f"v:{big:X} ({big})"
    # VC2 multi-chunk: short == full (W3-15 conservative), no entry.
    assert emission.precision_entry is None


def test_accumulator_auto_flush_on_shifted_id_change() -> None:
    """A feed with a different shifted id auto-flushes the prior
    source and starts a new one; the prior emission is returned so
    the caller can append it before continuing."""
    acc = _NumberAccumulator()
    acc.feed(
        token_type=TokenType.FLOAT64,
        shifted_id=7,
        chunk=_f64_chunk(1.0),
    )
    prior = acc.feed(
        token_type=TokenType.FLOAT64,
        shifted_id=8,  # different id -> auto-flush
        chunk=_f64_chunk(2.0),
    )
    assert isinstance(prior, AccumulatorEmission)
    assert prior.short_text == "f64:0.1 E1"
    # The new source is now in flight; flush yields it.
    second = acc.flush()
    assert isinstance(second, AccumulatorEmission)
    assert second.short_text == "f64:0.2 E1"


def test_accumulator_produces_precision_entry_when_short_truncates() -> None:
    """A value that truncates in the short form attaches an
    :class:`InlineNumberPrecisionEntry` payload."""
    acc = _NumberAccumulator()
    acc.feed(
        token_type=TokenType.FLOAT64,
        shifted_id=9,
        chunk=_f64_chunk(math.pi),
    )
    emission = acc.flush()
    assert emission is not None
    assert emission.precision_entry is not None
    assert isinstance(emission.precision_entry, InlineNumberPrecisionEntry)
    assert emission.precision_entry.token_type is TokenType.FLOAT64
    assert emission.precision_entry.full_text == "f64:0.31415926535897931 E1"
    assert len(emission.precision_entry.chunks) == 1


def test_accumulator_idempotent_flush() -> None:
    """Calling :meth:`flush` twice in a row on an empty buffer yields
    ``None`` both times -- callers can flush at every instruction
    boundary without guarding on :meth:`has_pending`."""
    acc = _NumberAccumulator()
    acc.feed(
        token_type=TokenType.FLOAT64,
        shifted_id=1,
        chunk=_f64_chunk(0.5),
    )
    first = acc.flush()
    assert first is not None
    assert acc.flush() is None
    assert acc.flush() is None
