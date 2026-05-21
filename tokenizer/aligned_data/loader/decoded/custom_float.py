"""Custom-float kernel for number-carrying tokens.

Single concern: encode any numeric source value (wide integer, f16, bf16, f32,
f64, f80, f128, ...) into a list of ``(significand: u64, sign_exponent: u32)``
chunks under a single shared representation.

Representation
--------------
Each emitted pair represents the value
``(-1)^sign * sig * 2**exponent_unbiased`` with ``sig`` interpreted as an
unsigned 64-bit integer (i.e. NOT a fractional 1.xxx significand).  Non-zero
chunks are emitted *normalized*: the leading 1 of ``sig`` always sits at bit
63 of the u64 (so ``sig`` is in ``[2**63, 2**64)``).  Signed-zero chunks
carry ``sig == 0`` and the sign bit; their exponent is the chunk's stride
position and is not value-bearing on its own.

The ``sign_exponent`` word packs the sign bit at MSB (bit 31) and the
*biased* exponent in the low 31 bits.  Bias = ``TARGET_EXPONENT_BIAS``
(mid-range of u31).

Multi-chunk values
------------------
Magnitudes wider than 64 bits split into ``ceil(width/64)`` u64 chunks; the
emitted list is ordered low-chunk first (``chunks[0]`` holds bits ``[0, 64)``
of the magnitude, ``chunks[k]`` holds bits ``[64*k, 64*(k+1))``).  Each chunk
is normalized independently so its leading 1 lands at bit 63; the per-chunk
exponent records the chunk's stride ``64*k`` minus the normalization shift.
An all-zero chunk emits a *signed zero* (sign carried, sig=0, exponent =
``base + 64*k``).

This same kernel services wide integers (``from_int``) and wide floats
(``from_float128`` today; any future wider FP type follows the same path).
No rounding anywhere: a future source format that cannot fit raises rather
than silently rounding.

NaN / Inf
---------
Source NaN and Inf map to a reserved unbiased exponent
``INFNAN_EXPONENT_UNBIASED`` (biases back to the max u31 ``0x7FFFFFFF``):

* ``+Inf`` / ``-Inf``: ``(sig=0, sign_exp=pack_sign_exp(sign, INFNAN))``.
* ``NaN``: ``(sig=1, sign_exp=pack_sign_exp(sign, INFNAN))`` — payload not
  preserved (canonical sig).

NaN / Inf always emit a single chunk regardless of source width; detection
runs before ``_split_to_chunks`` so f128 NaN / Inf is one chunk, not two.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TARGET_SIGNIFICAND_BITS",
    "TARGET_EXPONENT_BITS",
    "TARGET_EXPONENT_BIAS",
    "INFNAN_EXPONENT_UNBIASED",
    "pack_sign_exp",
    "from_int",
    "from_float16",
    "from_bfloat16",
    "from_float32",
    "from_float64",
    "from_float80",
    "from_float128",
]

TARGET_SIGNIFICAND_BITS = 64
TARGET_EXPONENT_BITS = 31
TARGET_EXPONENT_BIAS = 0x4000_0000
# Reserved unbiased exponent for NaN / Inf; biases back to the max u31 word
# (0x7FFFFFFF).  All finite values land in [0, 0x7FFFFFFE] biased.
INFNAN_EXPONENT_UNBIASED = (1 << TARGET_EXPONENT_BITS) - 1 - TARGET_EXPONENT_BIAS

_U64_MASK = (1 << 64) - 1
_SIGN_BIT = 1 << 31
_EXP_MASK = (1 << TARGET_EXPONENT_BITS) - 1


def pack_sign_exp(sign: int, exponent_unbiased: int) -> np.uint32:
    """Pack ``(sign, exponent_unbiased)`` into the wire u32.

    ``sign >= 0`` clears the sign bit; ``sign < 0`` sets it (bit 31).
    The exponent is biased by ``TARGET_EXPONENT_BIAS`` and asserted to fit
    the u31 field.
    """
    biased = exponent_unbiased + TARGET_EXPONENT_BIAS
    if not (0 <= biased <= _EXP_MASK):
        raise OverflowError(
            f"biased exponent {biased} out of u31 range "
            f"(unbiased={exponent_unbiased}, bias={TARGET_EXPONENT_BIAS})"
        )
    sign_bit = 0 if sign >= 0 else _SIGN_BIT
    return np.uint32(sign_bit | biased)


def _encode_infnan(*, sign: int, mantissa_is_zero: bool) -> tuple[np.uint64, np.uint32]:
    """Single-chunk NaN / Inf sentinel.

    Inf -> sig=0, NaN -> sig=1 (canonical; source NaN payload is dropped).
    """
    sig = np.uint64(0 if mantissa_is_zero else 1)
    return sig, pack_sign_exp(sign, INFNAN_EXPONENT_UNBIASED)


def _emit_chunk(
    chunk_value: int,
    *,
    sign: int,
    chunk_exponent_base: int,
) -> tuple[np.uint64, np.uint32]:
    """Normalize one u64 chunk and emit its ``(sig, sign_exp)`` pair.

    ``chunk_exponent_base`` is the exponent the chunk would have if its
    leading 1 were already at bit 63 (i.e. ``base + 64*k`` for chunk k).
    An all-zero chunk emits signed-zero at that base exponent.
    """
    if chunk_value == 0:
        return np.uint64(0), pack_sign_exp(sign, chunk_exponent_base)
    leading_bit_pos = chunk_value.bit_length() - 1
    shift = 63 - leading_bit_pos
    normalized = (chunk_value << shift) & _U64_MASK
    exponent_unbiased = chunk_exponent_base - shift
    return np.uint64(normalized), pack_sign_exp(sign, exponent_unbiased)


def _split_to_chunks(
    magnitude: int,
    *,
    sign: int,
    base_exponent_unbiased: int,
) -> list[tuple[np.uint64, np.uint32]]:
    """Split a non-negative ``magnitude`` into u64 chunks and emit pairs.

    The returned list is ordered low-chunk first: ``chunks[k]`` holds bits
    ``[64*k, 64*(k+1))`` of ``magnitude``.  Each chunk's exponent base is
    ``base_exponent_unbiased + 64*k``; per-chunk normalization subtracts the
    left-shift count from that base.  All-zero chunks emit signed zero at
    the chunk's base exponent.

    ``magnitude`` must be non-negative (sign is carried via the ``sign``
    parameter).  ``magnitude == 0`` emits a single signed-zero chunk.
    """
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative; sign is separate")
    bit_length = magnitude.bit_length()
    chunk_count = max(1, (bit_length + 63) // 64)
    chunks: list[tuple[np.uint64, np.uint32]] = []
    for k in range(chunk_count):
        chunk_value = (magnitude >> (64 * k)) & _U64_MASK
        chunk_exponent_base = base_exponent_unbiased + 64 * k
        chunks.append(
            _emit_chunk(
                chunk_value, sign=sign, chunk_exponent_base=chunk_exponent_base
            )
        )
    return chunks


def from_int(
    value: int, *, sign: int = +1
) -> list[tuple[np.uint64, np.uint32]]:
    """Encode a non-negative integer ``value``; ``sign`` carries the sign.

    Single-chunk values (``value < 2**64``) emit one pair; wider values emit
    ``ceil(bit_length / 64)`` pairs via the shared chunk kernel.  See module
    docstring for the per-chunk exponent rule.
    """
    if value < 0:
        raise ValueError(
            "from_int requires non-negative value; pass sign=-1 for negatives"
        )
    return _split_to_chunks(value, sign=sign, base_exponent_unbiased=0)


def _encode_fp_normalized(
    bits: int,
    *,
    mantissa_bits: int,
    exponent_bits: int,
    bias: int,
) -> list[tuple[np.uint64, np.uint32]]:
    """Generic IEEE-754-style FP -> custom-float kernel.

    ``bits`` is the raw bit pattern of the source value, interpreted as
    ``[sign : 1 | biased_exp : exponent_bits | mantissa : mantissa_bits]``
    from MSB downward.  ``mantissa_bits`` does NOT count the implicit
    leading 1; sources without an implicit leading 1 (e.g. x87 f80, which
    stores the leading bit explicitly) should still pass the *fractional*
    width here and let the explicit-leading-bit be picked up via the wider
    raw mantissa (see ``from_float80``).

    The encoder:
      * splits sign / biased_exp / mantissa via bit-masks;
      * maps source NaN / Inf (biased_exp = all-ones) to the single-chunk
        ``INFNAN_EXPONENT_UNBIASED`` sentinel (Inf -> sig=0; NaN -> sig=1);
      * emits a single signed-zero chunk for source ``±0``;
      * renormalizes denormals via pure left-shift (lossless);
      * for sources with ``mantissa_bits + 1 <= TARGET_SIGNIFICAND_BITS``,
        emits one chunk; for wider sources (f128) delegates to
        ``_split_to_chunks`` for the multi-chunk path.
    """
    total_bits = 1 + exponent_bits + mantissa_bits
    if not (0 <= bits < (1 << total_bits)):
        raise ValueError(
            f"bits=0x{bits:x} out of range for "
            f"sign+{exponent_bits}+{mantissa_bits} layout"
        )

    mantissa_mask = (1 << mantissa_bits) - 1
    exp_field_mask = (1 << exponent_bits) - 1

    raw_mantissa = bits & mantissa_mask
    biased_exp = (bits >> mantissa_bits) & exp_field_mask
    sign_bit = (bits >> (mantissa_bits + exponent_bits)) & 1
    sign = -1 if sign_bit else +1

    if biased_exp == exp_field_mask:
        # Source NaN / Inf collapse to a single chunk regardless of source
        # width; multi-chunk delegation never runs on these sources.
        return [_encode_infnan(sign=sign, mantissa_is_zero=raw_mantissa == 0)]

    is_denormal = biased_exp == 0
    if is_denormal:
        effective_mantissa = raw_mantissa
        actual_exp = 1 - bias
    else:
        effective_mantissa = (1 << mantissa_bits) | raw_mantissa
        actual_exp = biased_exp - bias

    base_exponent_unbiased = actual_exp - mantissa_bits

    if mantissa_bits + 1 <= TARGET_SIGNIFICAND_BITS:
        return [
            _emit_chunk(
                effective_mantissa,
                sign=sign,
                chunk_exponent_base=base_exponent_unbiased,
            )
        ]
    return _split_to_chunks(
        effective_mantissa,
        sign=sign,
        base_exponent_unbiased=base_exponent_unbiased,
    )


def from_float16(bits: int) -> list[tuple[np.uint64, np.uint32]]:
    """IEEE-754 binary16: 1 / 5 / 10 bits, bias 15."""
    return _encode_fp_normalized(
        bits, mantissa_bits=10, exponent_bits=5, bias=15
    )


def from_bfloat16(bits: int) -> list[tuple[np.uint64, np.uint32]]:
    """bfloat16: 1 / 8 / 7 bits, bias 127 (f32 truncated mantissa)."""
    return _encode_fp_normalized(
        bits, mantissa_bits=7, exponent_bits=8, bias=127
    )


def from_float32(bits: int) -> list[tuple[np.uint64, np.uint32]]:
    """IEEE-754 binary32: 1 / 8 / 23 bits, bias 127."""
    return _encode_fp_normalized(
        bits, mantissa_bits=23, exponent_bits=8, bias=127
    )


def from_float64(bits: int) -> list[tuple[np.uint64, np.uint32]]:
    """IEEE-754 binary64: 1 / 11 / 52 bits, bias 1023."""
    return _encode_fp_normalized(
        bits, mantissa_bits=52, exponent_bits=11, bias=1023
    )


def from_float80(bits: int) -> list[tuple[np.uint64, np.uint32]]:
    """x87 80-bit extended precision: 1 sign + 15 exponent + 64 mantissa.

    Unlike IEEE-754, the leading mantissa bit is stored explicitly (not
    implicit) — it occupies bit 63 of the 64-bit mantissa field for normal
    values, leaving 63 fractional bits.  We expose the *fractional* width
    (63) to the generic kernel and synthesise the same value by passing
    ``mantissa_bits=63``; the explicit leading 1 of normals then collides
    with the implicit one the kernel reconstructs, which is correct.

    Denormals (biased_exp == 0) and pseudo-denormals / unnormals (biased_exp
    > 0 but explicit-leading-bit == 0) are both treated as raw fractional
    mantissas via the denormal renormalization path — pure left-shift, no
    rounding.
    """
    total_bits = 80
    if not (0 <= bits < (1 << total_bits)):
        raise ValueError(f"bits=0x{bits:x} out of range for f80 layout")
    raw_mantissa = bits & ((1 << 64) - 1)
    biased_exp = (bits >> 64) & ((1 << 15) - 1)
    sign_bit = (bits >> 79) & 1
    sign = -1 if sign_bit else +1
    bias = 16383

    explicit_leading = (raw_mantissa >> 63) & 1
    fractional = raw_mantissa & ((1 << 63) - 1)

    if biased_exp == (1 << 15) - 1:
        # x87 f80: biased_exp=0x7FFF + explicit_leading=1 + fractional=0 is
        # the canonical Inf encoding; everything else with this exponent is
        # a NaN (quiet/signaling) or a pseudo-* encoding (invalid on modern
        # CPUs).  We collapse all pseudo-* forms into NaN, matching the
        # "drop payload, canonical sig=1" convention.
        is_inf = explicit_leading == 1 and fractional == 0
        return [_encode_infnan(sign=sign, mantissa_is_zero=is_inf)]

    if biased_exp == 0 or explicit_leading == 0:
        actual_exp = 1 - bias
        effective_mantissa = fractional
    else:
        actual_exp = biased_exp - bias
        effective_mantissa = (1 << 63) | fractional

    base_exponent_unbiased = actual_exp - 63
    return [
        _emit_chunk(
            effective_mantissa,
            sign=sign,
            chunk_exponent_base=base_exponent_unbiased,
        )
    ]


def from_float128(bits: int) -> list[tuple[np.uint64, np.uint32]]:
    """IEEE-754 binary128: 1 / 15 / 112 bits, bias 16383.

    Lossless: the 113-bit effective mantissa (1 implicit + 112 stored)
    splits into two u64 chunks via ``_split_to_chunks``; the same sign
    rides both chunks; the low chunk carries the trailing 49 bits
    left-aligned at bit 63.
    """
    return _encode_fp_normalized(
        bits, mantissa_bits=112, exponent_bits=15, bias=16383
    )
