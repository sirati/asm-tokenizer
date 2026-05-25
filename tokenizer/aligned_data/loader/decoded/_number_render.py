"""Inspector-display rendering for NUMBER-band chunk pairs.

Single concern: turn the wire-form ``(sig, sign_exp)`` chunk pairs
(from :mod:`custom_float`) into the two inspector display forms:

* **Short** (one-row default): ``v:HEX (decimal)`` for VC2 (sign via
  the ``value_negative`` marker); ``f<N>:[-]0.<digits> E<exp>`` for
  floats at AT-MOST :data:`_SHORT_MANTISSA_DIGITS` significant digits.
* **Full** (expansion ``Openable``): same shape, mantissa at the
  IEEE-754 round-trip-minimum digit count
  (:data:`_FULL_PRECISION_DIGITS` -- auto-derived from
  ``_IEEE_LAYOUTS[t].mantissa_bits``).
* **Specials**: ``f<N>:NaN`` (unsigned); ``f<N>:Inf`` / ``f<N>:-Inf``;
  ``f<N>:0`` for signed zero. None of these carry an expansion entry.

The expansion entry (:class:`InlineNumberPrecisionEntry`) is only
attached when :func:`needs_precision_expand` says the short form
lost precision -- natural-precision rendering keeps that predicate a
trivial string compare.

This module is the sole renderer consumed by the live
BatchDecodeBackend NUMBER-band path; the legacy
:func:`number_hex_format.chunks_to_hex_bits` stays put as the
FtlBackend-parity debug helper (W3-14). Module boundary: no inspector
imports; depends only on :mod:`custom_float` + :mod:`number_hex_format`
+ :mod:`tokenizer.tokens`. The inspector's ``_render/_protocol.py``
re-exports :class:`InlineNumberPrecisionEntry`; the import direction
stays one-way (inspector -> renderer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.decoded.custom_float import (
    INFNAN_EXPONENT_UNBIASED,
    reconstruct_chunks,
    unpack_chunk,
)
from tokenizer.aligned_data.loader.decoded.number_hex_format import (
    _IEEE_LAYOUTS,
)
from tokenizer.tokens import TokenType


__all__ = [
    "InlineNumberPrecisionEntry",
    "render_vc2_short",
    "render_vc2_full",
    "render_float_short",
    "render_float_full",
    "needs_precision_expand",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


# Mantissa width of the short ("default display") form. Single source
# of truth — the float-short renderer formats to exactly this many
# significant digits, and :func:`needs_precision_expand` uses this
# value to decide whether the short form lost precision.
_SHORT_MANTISSA_DIGITS: int = 4


def _round_trip_decimal_digits(layout) -> int:
    """IEEE-754 round-trip minimum decimal digits for ``layout``.

    Formula: ``ceil(precision_bits * log10(2)) + 1`` where
    ``precision_bits`` includes the implicit / explicit leading bit:

    * IEEE-754 formats (implicit leading bit, ``has_explicit_leading_bit
      == False``): ``precision_bits = mantissa_bits + 1``.
    * x87 f80 (explicit leading bit stored in the mantissa field,
      ``has_explicit_leading_bit == True``): ``precision_bits =
      mantissa_bits``.

    Produces (verified): F16=5, BF16=4, F32=9, F64=17, F80=21, F128=36
    — the standard ``digits10_round_trip`` for each format. This is
    the minimum digit count s.t. binary -> decimal -> binary
    round-trips exactly for every representable value in the format.
    """
    if layout.has_explicit_leading_bit:
        precision_bits = layout.mantissa_bits
    else:
        precision_bits = layout.mantissa_bits + 1
    return math.ceil(precision_bits * math.log10(2)) + 1


# Full-precision digit table, auto-derived from `_IEEE_LAYOUTS`.
# Module-load tripwire (next assert) catches any FP TokenType added to
# `_IEEE_LAYOUTS` without a matching short-prefix entry.
_FULL_PRECISION_DIGITS: dict[TokenType, int] = {
    t: _round_trip_decimal_digits(layout) for t, layout in _IEEE_LAYOUTS.items()
}


def _derive_short_prefix(basename: str) -> str:
    """Short-form prefix from the layout's basename.

    Rule: drop the ``float`` prefix and prepend ``f``; ``bfloat16``
    has the one explicit override to ``bf16`` (so ``BFLOAT16`` does
    not collide with ``FLOAT16``'s ``f16``).
    """
    if basename == "bfloat16":
        return "bf16"
    return "f" + basename.removeprefix("float")


_TOKEN_TYPE_TO_PREFIX: dict[TokenType, str] = {
    t: _derive_short_prefix(layout.basename) for t, layout in _IEEE_LAYOUTS.items()
}


# Module-load tripwires (W3-15): every FP TokenType in `_IEEE_LAYOUTS`
# MUST have a matching `_FULL_PRECISION_DIGITS` + `_TOKEN_TYPE_TO_PREFIX`
# entry. Adding a new float width to `_IEEE_LAYOUTS` without updating
# the prefix override (if needed) fails loud at import.
assert set(_FULL_PRECISION_DIGITS) == set(_IEEE_LAYOUTS), (
    "full-precision digit table out of sync with _IEEE_LAYOUTS"
)
assert set(_TOKEN_TYPE_TO_PREFIX) == set(_IEEE_LAYOUTS), (
    "short-prefix table out of sync with _IEEE_LAYOUTS"
)
# Spot-check the short prefixes (sanity, not exhaustive):
assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT16] == "f16"
assert _TOKEN_TYPE_TO_PREFIX[TokenType.BFLOAT16] == "bf16"
assert _TOKEN_TYPE_TO_PREFIX[TokenType.FLOAT128] == "f128"


# ---------------------------------------------------------------------------
# Openable payload
# ---------------------------------------------------------------------------


# Type aliases for chunk pairs (keep imports light at call sites).
_ChunkPair = Tuple[np.uint64, np.uint32]


@dataclass(frozen=True)
class InlineNumberPrecisionEntry:
    """Full-precision expansion payload for a NUMBER-band row.

    The frozen-dataclass identity discriminates this from
    :class:`InlineCallEntry` / :class:`InlineJumpEntry` in the
    inspector's :data:`Openable` union (no enum tag — dispatch via
    ``isinstance`` / ``match``).

    Carried alongside the short-form row whenever
    :func:`needs_precision_expand` returns ``True``; the tree-model
    reads :attr:`full_text` verbatim. :attr:`chunks` carries the
    original wire chunks for future forensic-decomposition use cases
    (W3-1 — kept as the producer's owned shape).
    """

    token_type: TokenType
    chunks: Tuple[_ChunkPair, ...]
    full_text: str


# ---------------------------------------------------------------------------
# VC2 renderers
# ---------------------------------------------------------------------------


def render_vc2_short(value: int, value_negative: bool) -> str:
    """VC2 short form: ``v:HEX (decimal)``, sign-aware.

    The encoder carries the sign via a separate ``value_negative``
    postfix marker (token id 256, ``VALUE_NEGATIVE``); the renderer
    consumes both as inputs and applies the sign in both the hex and
    decimal parts. ``value`` is the (non-negative) magnitude.
    """
    if value < 0:
        raise ValueError(
            f"render_vc2_short: value must be non-negative magnitude, got {value}"
        )
    sign = "-" if value_negative and value != 0 else ""
    return f"v:{sign}{value:X} ({sign}{value})"


def render_vc2_full(
    chunks: Tuple[_ChunkPair, ...], value_negative: bool
) -> str:
    """VC2 full-precision form: reconstruct multi-chunk magnitude, render.

    Same text shape as :func:`render_vc2_short`; the "full" form
    differs only in that it reconstructs the entire multi-chunk
    magnitude (lead + trailing chunks) rather than just the lead
    chunk's contribution. For a single-chunk VC2 source this returns
    identical text to :func:`render_vc2_short`.

    ``reconstruct_chunks`` returns a Fraction; VC2 magnitudes are
    integers by encoder contract (no sub-bit storage; chunks compose
    a wide integer).
    """
    fraction = reconstruct_chunks(list(chunks))
    if fraction.denominator != 1:
        raise ValueError(
            f"render_vc2_full: VC2 reconstruction yielded non-integer "
            f"fraction {fraction}"
        )
    magnitude = int(fraction)
    if magnitude < 0:
        # reconstruct_chunks carries the chunk-level sign; for VC2 the
        # convention is that sign is external (value_negative). The
        # chunk's own sign should already be +; defensive abs() so we
        # don't double-up.
        magnitude = -magnitude
    return render_vc2_short(magnitude, value_negative)


# ---------------------------------------------------------------------------
# Float renderers
# ---------------------------------------------------------------------------


def _is_nan_chunk(chunk: _ChunkPair) -> bool:
    """The chunk is the encoder's NaN sentinel (exp = INFNAN, sig != 0)."""
    sig_int, _sign, exp_unbiased = unpack_chunk(chunk)
    return exp_unbiased == INFNAN_EXPONENT_UNBIASED and sig_int != 0


def _is_inf_chunk(chunk: _ChunkPair) -> bool:
    """The chunk is the encoder's Inf sentinel (exp = INFNAN, sig == 0)."""
    sig_int, _sign, exp_unbiased = unpack_chunk(chunk)
    return exp_unbiased == INFNAN_EXPONENT_UNBIASED and sig_int == 0


def _float_special_text(
    chunks: Tuple[_ChunkPair, ...], prefix: str
) -> Optional[str]:
    """Render NaN / Inf / signed-zero, or ``None`` for finite chunks.

    Specials route the same way in both short and full form: NaN /
    Inf / zero have no precision to lose, so no expansion entry is
    offered (the short form already captures all information).
    """
    if not chunks:
        raise ValueError("float renderer: empty chunk sequence")
    lead = chunks[0]
    if _is_nan_chunk(lead):
        # NaN sign is unconditionally dropped (W3-15: encoder
        # canonicalises NaN sig to 1; sign-bit-only NaN is
        # forensically marginal).
        return f"{prefix}:NaN"
    if _is_inf_chunk(lead):
        _sig, sign, _exp = unpack_chunk(lead)
        return f"{prefix}:-Inf" if sign < 0 else f"{prefix}:Inf"
    # Signed-zero short-circuit: all chunks have sig == 0. The encoder
    # emits a signed-zero chunk per stride position for genuine zero
    # values, so we inspect the entire chunk sequence rather than
    # peeking at the lead alone.
    if all(unpack_chunk(c)[0] == 0 for c in chunks):
        return f"{prefix}:0"
    return None


def _reconstruct_value_decimal(
    chunks: Tuple[_ChunkPair, ...], digits: int
) -> Decimal:
    """Exact ``Fraction`` -> ``Decimal`` at ``digits`` significant digits.

    The reconstruction is exact (``Fraction`` carries the full
    multi-chunk magnitude); the Decimal conversion is the lossy step
    — but we set the precision to ``digits`` significant digits, so
    the resulting Decimal IS the canonical round-trip-minimum
    rendering. For short form (``_SHORT_MANTISSA_DIGITS``) this
    truncates / rounds intentionally; for full form (IEEE
    round-trip-minimum) this preserves all source information.
    """
    fraction = reconstruct_chunks(list(chunks))
    # `Decimal(Fraction)` is exact but unbounded; quantize via the
    # context precision after the numerator/denominator divide.
    ctx = getcontext().copy()
    ctx.prec = digits
    numerator = Decimal(fraction.numerator)
    denominator = Decimal(fraction.denominator)
    return ctx.divide(numerator, denominator)


def _format_scientific(dec: Decimal, digits: int) -> str:
    """Format ``dec`` as ``[-]0.<digits> E<exp>`` at AT-MOST ``digits`` precision.

    The mantissa is ``0.<digits>`` (leading zero before the decimal
    point, NOT ``d.ddd``); the exponent is adjusted so
    ``value == mantissa * 10**exp``. This is the "engineering" form
    most asm pretty-printers use.

    The mantissa is the value's natural compact digit string — no
    trailing-zero padding to a fixed width. The ``digits`` parameter
    has already capped the Decimal's precision context upstream, so
    the compact form already carries AT MOST ``digits`` significant
    digits. Natural-precision rendering keeps the short / full
    comparison trivial: identical text iff the value fit losslessly
    in the short precision.

    Special-cases zero: a zero mantissa renders as ``0`` (no
    exponent), matching the signed-zero short-circuit in
    :func:`_float_special_text`.
    """
    if dec.is_zero():
        return "0"
    sign, digit_tuple, exponent_attr = dec.as_tuple()
    # `Decimal` carries the digits compactly. The scientific-form
    # exponent (with mantissa ``0.<digits>``) is computed from the
    # compact digit count: value == 0.<compact> * 10**(exp_attr + len).
    compact_digit_str = "".join(str(d) for d in digit_tuple)
    # Defensive cap (Decimal context already capped at `digits`).
    if len(compact_digit_str) > digits:
        compact_digit_str = compact_digit_str[:digits]
    exp = int(exponent_attr) + len(compact_digit_str)
    sign_str = "-" if sign else ""
    return f"{sign_str}0.{compact_digit_str} E{exp}"


def render_float_short(
    token_type: TokenType, chunks: Tuple[_ChunkPair, ...]
) -> str:
    """Float short form: ``f<N>:[-]0.dddd E<exp>``.

    The ``f<N>`` prefix derives from :data:`_IEEE_LAYOUTS` ``basename``
    via :data:`_TOKEN_TYPE_TO_PREFIX`. Specials (NaN / Inf / signed
    zero) short-circuit via :func:`_float_special_text`. Finite
    values reconstruct via :func:`_reconstruct_value_decimal` at
    :data:`_SHORT_MANTISSA_DIGITS` precision.
    """
    prefix = _TOKEN_TYPE_TO_PREFIX.get(token_type)
    if prefix is None:
        raise ValueError(
            f"render_float_short: unsupported token_type {token_type!r}"
        )
    special = _float_special_text(chunks, prefix)
    if special is not None:
        return special
    dec = _reconstruct_value_decimal(chunks, _SHORT_MANTISSA_DIGITS)
    return f"{prefix}:{_format_scientific(dec, _SHORT_MANTISSA_DIGITS)}"


def render_float_full(
    token_type: TokenType, chunks: Tuple[_ChunkPair, ...]
) -> str:
    """Float full-precision form: same shape, more mantissa digits.

    Uses :data:`_FULL_PRECISION_DIGITS` (auto-derived from each
    format's mantissa bits) so binary -> decimal -> binary round-
    trips losslessly for every representable value in the format.
    """
    prefix = _TOKEN_TYPE_TO_PREFIX.get(token_type)
    if prefix is None:
        raise ValueError(
            f"render_float_full: unsupported token_type {token_type!r}"
        )
    special = _float_special_text(chunks, prefix)
    if special is not None:
        return special
    digits = _FULL_PRECISION_DIGITS[token_type]
    dec = _reconstruct_value_decimal(chunks, digits)
    return f"{prefix}:{_format_scientific(dec, digits)}"


# ---------------------------------------------------------------------------
# Expansion predicate
# ---------------------------------------------------------------------------


def needs_precision_expand(short_text: str, full_text: str) -> bool:
    """``True`` iff the short form lost precision relative to the full form.

    The expansion entry is only offered when the full form would
    surface NEW information. For values that fit losslessly in the
    short form (e.g. integer-valued floats, simple powers of two),
    the two strings are identical and no expansion is needed
    (``can_expand = False`` at the tree-model layer).

    The decision lives here (next to the renderers) so callers
    construct :class:`InlineNumberPrecisionEntry` only when it adds
    information. This is the W3-13 / W4-amended predicate, lifted
    into a simple string compare because both renderers go through
    the same formatter — same value at different precisions yields
    the same text iff the short precision was already lossless.
    """
    return short_text != full_text
