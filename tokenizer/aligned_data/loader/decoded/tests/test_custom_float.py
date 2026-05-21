"""Tests for the custom-float kernel.

Covers per-source-type encoders, multi-chunk integer chunking, signed-zero
chunks, denormal renormalization, FLOAT128 bit-exact lossless round-trip,
and NaN / Inf sentinel emission per the locked-in design.

Value representation (verbatim from ``custom_float`` module docstring):
    value = sign * sig * 2**exponent_unbiased
with ``sig`` interpreted as an unsigned 64-bit integer.  Tests reconstruct
the source value via this rule rather than asserting internal exponent
values directly, except where the spawn brief calls out specific exponents.
"""

from __future__ import annotations

import random
import struct
from fractions import Fraction

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.custom_float import (
    INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS,
    TARGET_EXPONENT_BITS,
    TARGET_SIGNIFICAND_BITS,
    _encode_fp_normalized,
    from_bfloat16,
    from_float16,
    from_float32,
    from_float64,
    from_float80,
    from_float128,
    from_int,
    pack_sign_exp,
)


# ----- helpers -----------------------------------------------------------------

def _unpack(pair: tuple[np.uint64, np.uint32]) -> tuple[int, int, int]:
    """Return (sig:int, sign:+1/-1, exponent_unbiased:int)."""
    sig, sign_exp = pair
    sign_exp_int = int(sign_exp)
    sign = -1 if (sign_exp_int >> 31) & 1 else +1
    biased = sign_exp_int & ((1 << 31) - 1)
    return int(sig), sign, biased - TARGET_EXPONENT_BIAS


def _reconstruct(chunks: list[tuple[np.uint64, np.uint32]]) -> Fraction:
    """Reconstruct the signed value (exact Fraction) from chunks.

    Convention: value = sign * sig * 2**exp.  Signed-zero chunks contribute
    nothing to the magnitude; non-zero chunks must share the same sign.
    NaN / Inf sentinels are NOT handled here — those tests inspect the
    sentinel exponent directly.
    """
    magnitude = Fraction(0)
    sign_seen: int | None = None
    for sig, sign, exp in (_unpack(c) for c in chunks):
        if sig == 0:
            continue
        contribution = (
            Fraction(sig << exp, 1) if exp >= 0 else Fraction(sig, 1 << (-exp))
        )
        magnitude += contribution
        if sign_seen is None:
            sign_seen = sign
        else:
            assert sign == sign_seen, (
                f"chunks disagree on sign: {sign_seen} vs {sign}"
            )
    return sign_seen * magnitude if sign_seen is not None else Fraction(0)


# ----- pack_sign_exp -----------------------------------------------------------

def test_pack_sign_exp_zero_positive():
    assert int(pack_sign_exp(+1, 0)) == TARGET_EXPONENT_BIAS


def test_pack_sign_exp_zero_negative():
    assert int(pack_sign_exp(-1, 0)) == TARGET_EXPONENT_BIAS | (1 << 31)


def test_pack_sign_exp_positive_exponent():
    assert int(pack_sign_exp(+1, 5)) == TARGET_EXPONENT_BIAS + 5


def test_pack_sign_exp_negative_exponent():
    assert int(pack_sign_exp(+1, -7)) == TARGET_EXPONENT_BIAS - 7


def test_pack_sign_exp_extreme_negative():
    assert int(pack_sign_exp(+1, -TARGET_EXPONENT_BIAS)) == 0


def test_pack_sign_exp_max_finite_exponent():
    assert (
        int(pack_sign_exp(+1, INFNAN_EXPONENT_UNBIASED))
        == (1 << TARGET_EXPONENT_BITS) - 1
    )


def test_pack_sign_exp_overflow_raises():
    with pytest.raises(OverflowError):
        pack_sign_exp(+1, INFNAN_EXPONENT_UNBIASED + 1)


def test_pack_sign_exp_underflow_raises():
    with pytest.raises(OverflowError):
        pack_sign_exp(+1, -TARGET_EXPONENT_BIAS - 1)


def test_pack_sign_exp_zero_sign_is_positive():
    # sign=0 is treated as positive per the ``sign >= 0`` rule.
    assert int(pack_sign_exp(0, 3)) == TARGET_EXPONENT_BIAS + 3


# ----- from_int ----------------------------------------------------------------

def test_from_int_zero_positive():
    chunks = from_int(0, sign=+1)
    assert chunks == [(np.uint64(0), pack_sign_exp(+1, 0))]


def test_from_int_zero_negative_signed_zero():
    chunks = from_int(0, sign=-1)
    assert chunks == [(np.uint64(0), pack_sign_exp(-1, 0))]


def test_from_int_one():
    chunks = from_int(1, sign=+1)
    assert len(chunks) == 1
    sig, sign, exp = _unpack(chunks[0])
    assert sig == 1 << 63
    assert sign == +1
    assert exp == -63
    assert _reconstruct(chunks) == 1


def test_from_int_two():
    chunks = from_int(2, sign=+1)
    sig, _, exp = _unpack(chunks[0])
    assert sig == 1 << 63 and exp == -62
    assert _reconstruct(chunks) == 2


def test_from_int_negative_value_raises():
    with pytest.raises(ValueError):
        from_int(-1)


def test_from_int_max_u64():
    chunks = from_int((1 << 64) - 1, sign=+1)
    assert len(chunks) == 1
    sig, _, exp = _unpack(chunks[0])
    assert sig == (1 << 64) - 1
    assert exp == 0
    assert _reconstruct(chunks) == (1 << 64) - 1


def test_from_int_2_pow_64_low_chunk_signed_zero():
    chunks = from_int(1 << 64, sign=+1)
    assert len(chunks) == 2
    low_sig, low_sign, low_exp = _unpack(chunks[0])
    assert low_sig == 0
    assert low_sign == +1
    assert low_exp == 0
    high_sig, high_sign, high_exp = _unpack(chunks[1])
    assert high_sig == 1 << 63
    assert high_sign == +1
    assert high_exp == 1
    assert _reconstruct(chunks) == 1 << 64


def test_from_int_2_pow_64_negative_sign_on_both_chunks():
    chunks = from_int(1 << 64, sign=-1)
    for pair in chunks:
        _, sign, _ = _unpack(pair)
        assert sign == -1
    assert _reconstruct(chunks) == -(1 << 64)


def test_from_int_2_pow_128_minus_1():
    chunks = from_int((1 << 128) - 1, sign=+1)
    assert len(chunks) == 2
    low_sig, _, low_exp = _unpack(chunks[0])
    high_sig, _, high_exp = _unpack(chunks[1])
    assert low_sig == (1 << 64) - 1
    assert low_exp == 0
    assert high_sig == (1 << 64) - 1
    assert high_exp == 64
    assert _reconstruct(chunks) == (1 << 128) - 1


def test_from_int_random_u192_roundtrip():
    rng = random.Random(0xDEAD_BEEF)
    for _ in range(40):
        value = rng.randrange(1 << 192)
        chunks = from_int(value, sign=+1)
        assert len(chunks) == max(1, (value.bit_length() + 63) // 64)
        assert _reconstruct(chunks) == value


def test_from_int_random_signed_u192_roundtrip():
    rng = random.Random(0xBAAD_F00D)
    for _ in range(20):
        magnitude = rng.randrange(1, 1 << 192)
        chunks = from_int(magnitude, sign=-1)
        assert _reconstruct(chunks) == -magnitude


def test_from_int_zero_chunk_in_middle():
    # value = (1<<128) | 1 -> 3 chunks: low=1, mid=0 (signed zero), high=1.
    value = (1 << 128) | 1
    chunks = from_int(value, sign=-1)
    assert len(chunks) == 3
    low_sig, low_sign, low_exp = _unpack(chunks[0])
    mid_sig, mid_sign, mid_exp = _unpack(chunks[1])
    high_sig, high_sign, high_exp = _unpack(chunks[2])
    assert low_sig == 1 << 63 and low_exp == -63 and low_sign == -1
    assert mid_sig == 0 and mid_sign == -1 and mid_exp == 64
    assert high_sig == 1 << 63 and high_exp == 128 - 63 and high_sign == -1
    assert _reconstruct(chunks) == -value


# ----- FP helpers --------------------------------------------------------------

def _bits_from_format(value: float, fmt: str) -> int:
    raw = struct.pack(fmt, value)
    return int.from_bytes(raw, "little")


def _f32_bits(value: float) -> int:
    return _bits_from_format(value, "<f")


def _f64_bits(value: float) -> int:
    return _bits_from_format(value, "<d")


def _f128_bits(sign: int, biased_exp: int, mantissa112: int) -> int:
    return (
        (sign & 1) << 127
        | (biased_exp & ((1 << 15) - 1)) << 112
        | (mantissa112 & ((1 << 112) - 1))
    )


def _f80_bits(sign: int, biased_exp: int, mantissa64: int) -> int:
    return (
        (sign & 1) << 79
        | (biased_exp & ((1 << 15) - 1)) << 64
        | (mantissa64 & ((1 << 64) - 1))
    )


def _f16_bits(sign: int, biased_exp: int, mantissa10: int) -> int:
    return (
        (sign & 1) << 15
        | (biased_exp & ((1 << 5) - 1)) << 10
        | (mantissa10 & ((1 << 10) - 1))
    )


def _bf16_bits(sign: int, biased_exp: int, mantissa7: int) -> int:
    return (
        (sign & 1) << 15
        | (biased_exp & ((1 << 8) - 1)) << 7
        | (mantissa7 & ((1 << 7) - 1))
    )


def _is_infnan_sentinel(pair: tuple[np.uint64, np.uint32]) -> bool:
    _, _, exp = _unpack(pair)
    return exp == INFNAN_EXPONENT_UNBIASED


# ----- from_float32 ------------------------------------------------------------

def test_from_float32_positive_zero():
    chunks = from_float32(_f32_bits(0.0))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == +1


def test_from_float32_negative_zero():
    chunks = from_float32(_f32_bits(-0.0))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == -1


def test_from_float32_one():
    chunks = from_float32(_f32_bits(1.0))
    assert _reconstruct(chunks) == 1
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1 << 63  # normalized


def test_from_float32_negative_one():
    chunks = from_float32(_f32_bits(-1.0))
    assert _reconstruct(chunks) == -1


def test_from_float32_smallest_normal():
    # 2**-126
    chunks = from_float32(1 << 23)
    assert _reconstruct(chunks) == Fraction(1, 1 << 126)


def test_from_float32_smallest_denormal():
    # 2**-149
    chunks = from_float32(1)
    assert _reconstruct(chunks) == Fraction(1, 1 << 149)


def test_from_float32_largest_finite():
    bits = (254 << 23) | ((1 << 23) - 1)
    chunks = from_float32(bits)
    # largest f32 = (2**24 - 1) * 2**104
    assert _reconstruct(chunks) == ((1 << 24) - 1) * (1 << 104)


def test_from_float32_positive_inf():
    chunks = from_float32(255 << 23)
    assert _is_infnan_sentinel(chunks[0])
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == +1


def test_from_float32_negative_inf():
    chunks = from_float32((1 << 31) | (255 << 23))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == -1
    assert _is_infnan_sentinel(chunks[0])


def test_from_float32_quiet_nan_collapses_to_sig_one():
    chunks = from_float32((255 << 23) | (1 << 22))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 1 and sign == +1
    assert _is_infnan_sentinel(chunks[0])


def test_from_float32_signalling_nan_collapses_to_sig_one():
    chunks = from_float32((255 << 23) | 1)
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1
    assert _is_infnan_sentinel(chunks[0])


def test_from_float32_negative_nan():
    chunks = from_float32((1 << 31) | (255 << 23) | (1 << 22))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 1 and sign == -1
    assert _is_infnan_sentinel(chunks[0])


def test_from_float32_random_normals_lossless():
    rng = random.Random(0x1234_5678)
    for _ in range(200):
        biased_exp = rng.randrange(1, 255)
        mantissa = rng.randrange(0, 1 << 23)
        sign_bit = rng.randint(0, 1)
        bits = (sign_bit << 31) | (biased_exp << 23) | mantissa
        chunks = from_float32(bits)
        expected_mantissa = (1 << 23) | mantissa
        expected_exp = biased_exp - 127 - 23
        expected_sign = -1 if sign_bit else +1
        if expected_exp >= 0:
            expected_value = Fraction(expected_sign * expected_mantissa * (1 << expected_exp), 1)
        else:
            expected_value = Fraction(expected_sign * expected_mantissa, 1 << (-expected_exp))
        assert _reconstruct(chunks) == expected_value


# ----- from_float64 ------------------------------------------------------------

def test_from_float64_positive_zero():
    sig, sign, _ = _unpack(from_float64(_f64_bits(0.0))[0])
    assert sig == 0 and sign == +1


def test_from_float64_negative_zero():
    sig, sign, _ = _unpack(from_float64(_f64_bits(-0.0))[0])
    assert sig == 0 and sign == -1


def test_from_float64_one():
    assert _reconstruct(from_float64(_f64_bits(1.0))) == 1


def test_from_float64_negative_one():
    assert _reconstruct(from_float64(_f64_bits(-1.0))) == -1


def test_from_float64_smallest_normal():
    assert _reconstruct(from_float64(1 << 52)) == Fraction(1, 1 << 1022)


def test_from_float64_smallest_denormal():
    assert _reconstruct(from_float64(1)) == Fraction(1, 1 << 1074)


def test_from_float64_largest_finite():
    bits = (2046 << 52) | ((1 << 52) - 1)
    # largest f64 = (2**53 - 1) * 2**971
    assert _reconstruct(from_float64(bits)) == ((1 << 53) - 1) * (1 << 971)


def test_from_float64_positive_inf():
    chunks = from_float64(2047 << 52)
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == +1
    assert _is_infnan_sentinel(chunks[0])


def test_from_float64_nan_collapses_to_sig_one():
    chunks = from_float64((2047 << 52) | 1)
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1
    assert _is_infnan_sentinel(chunks[0])


# ----- from_float16 ------------------------------------------------------------

def test_from_float16_one():
    assert _reconstruct(from_float16(15 << 10)) == 1


def test_from_float16_negative_zero():
    sig, sign, _ = _unpack(from_float16(1 << 15)[0])
    assert sig == 0 and sign == -1


def test_from_float16_smallest_denormal():
    # 2**-24
    assert _reconstruct(from_float16(1)) == Fraction(1, 1 << 24)


def test_from_float16_smallest_normal():
    # 2**-14
    assert _reconstruct(from_float16(1 << 10)) == Fraction(1, 1 << 14)


def test_from_float16_largest_finite():
    bits = (30 << 10) | ((1 << 10) - 1)
    # largest f16 = (2**11 - 1) * 2**5
    assert _reconstruct(from_float16(bits)) == ((1 << 11) - 1) * (1 << 5)


def test_from_float16_inf():
    chunks = from_float16(31 << 10)
    sig, _, _ = _unpack(chunks[0])
    assert sig == 0 and _is_infnan_sentinel(chunks[0])


def test_from_float16_nan():
    chunks = from_float16((31 << 10) | 1)
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1 and _is_infnan_sentinel(chunks[0])


# ----- from_bfloat16 -----------------------------------------------------------

def test_from_bfloat16_one():
    assert _reconstruct(from_bfloat16(127 << 7)) == 1


def test_from_bfloat16_smallest_denormal():
    # 2**-133
    assert _reconstruct(from_bfloat16(1)) == Fraction(1, 1 << 133)


def test_from_bfloat16_smallest_normal():
    # 2**-126
    assert _reconstruct(from_bfloat16(1 << 7)) == Fraction(1, 1 << 126)


def test_from_bfloat16_largest_finite():
    bits = (254 << 7) | ((1 << 7) - 1)
    # largest bf16 = (2**8 - 1) * 2**120
    assert _reconstruct(from_bfloat16(bits)) == ((1 << 8) - 1) * (1 << 120)


def test_from_bfloat16_nan():
    chunks = from_bfloat16((255 << 7) | 1)
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1 and _is_infnan_sentinel(chunks[0])


# ----- from_float80 ------------------------------------------------------------

def test_from_float80_one():
    # 1.0: sign=0, biased_exp=16383, mantissa=1<<63 (explicit leading bit)
    chunks = from_float80(_f80_bits(0, 16383, 1 << 63))
    assert _reconstruct(chunks) == 1


def test_from_float80_negative_one():
    chunks = from_float80(_f80_bits(1, 16383, 1 << 63))
    assert _reconstruct(chunks) == -1


def test_from_float80_positive_zero():
    sig, sign, _ = _unpack(from_float80(_f80_bits(0, 0, 0))[0])
    assert sig == 0 and sign == +1


def test_from_float80_negative_zero():
    sig, sign, _ = _unpack(from_float80(_f80_bits(1, 0, 0))[0])
    assert sig == 0 and sign == -1


def test_from_float80_smallest_denormal():
    # explicit leading bit=0, fractional=1, biased_exp=0 -> value = 2**-16445
    chunks = from_float80(_f80_bits(0, 0, 1))
    assert _reconstruct(chunks) == Fraction(1, 1 << 16445)


def test_from_float80_smallest_normal():
    # biased_exp=1, mantissa=1<<63 -> value = 2**-16382
    chunks = from_float80(_f80_bits(0, 1, 1 << 63))
    assert _reconstruct(chunks) == Fraction(1, 1 << 16382)


def test_from_float80_largest_finite():
    bits = _f80_bits(0, 0x7FFE, (1 << 64) - 1)
    # value = (2**64 - 1) * 2**(16383 - 63) = (2**64 - 1) * 2**16320
    assert _reconstruct(chunks=from_float80(bits)) == ((1 << 64) - 1) * (1 << 16320)


def test_from_float80_inf():
    chunks = from_float80(_f80_bits(0, 0x7FFF, 1 << 63))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == +1
    assert _is_infnan_sentinel(chunks[0])


def test_from_float80_nan():
    chunks = from_float80(_f80_bits(0, 0x7FFF, (1 << 63) | 1))
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1 and _is_infnan_sentinel(chunks[0])


# ----- from_float128 -----------------------------------------------------------

def test_from_float128_positive_zero():
    chunks = from_float128(_f128_bits(0, 0, 0))
    assert len(chunks) == 1
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == +1


def test_from_float128_negative_zero():
    chunks = from_float128(_f128_bits(1, 0, 0))
    assert len(chunks) == 1
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == -1


def test_from_float128_one_two_chunks_signed_zero_low():
    chunks = from_float128(_f128_bits(0, 16383, 0))
    assert len(chunks) == 2
    low_sig, low_sign, low_exp = _unpack(chunks[0])
    # Low chunk: mantissa low 64 bits are 0 -> signed zero at exp = -112.
    assert low_sig == 0
    assert low_sign == +1
    assert low_exp == -112
    high_sig, _, high_exp = _unpack(chunks[1])
    # High chunk: bit 48 set after >>64 -> normalize_shift=15, base=-112+64=-48.
    # exp = -48 - 15 = -63.  sig=1<<63.  sig*2**exp = 2**63 * 2**-63 = 1.
    assert high_sig == 1 << 63
    assert high_exp == -63
    assert _reconstruct(chunks) == 1


def test_from_float128_inf():
    chunks = from_float128(_f128_bits(0, 0x7FFF, 0))
    assert len(chunks) == 1
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == +1 and _is_infnan_sentinel(chunks[0])


def test_from_float128_negative_inf():
    chunks = from_float128(_f128_bits(1, 0x7FFF, 0))
    sig, sign, _ = _unpack(chunks[0])
    assert sig == 0 and sign == -1 and _is_infnan_sentinel(chunks[0])


def test_from_float128_nan():
    chunks = from_float128(_f128_bits(0, 0x7FFF, 0xDEAD_BEEF))
    assert len(chunks) == 1
    sig, _, _ = _unpack(chunks[0])
    assert sig == 1 and _is_infnan_sentinel(chunks[0])


def test_from_float128_smallest_denormal():
    # mantissa=1, biased_exp=0 -> value = 2**-16494
    chunks = from_float128(_f128_bits(0, 0, 1))
    assert _reconstruct(chunks) == Fraction(1, 1 << 16494)


def test_from_float128_smallest_normal():
    # biased_exp=1, mantissa=0 -> value = 2**-16382
    chunks = from_float128(_f128_bits(0, 1, 0))
    assert _reconstruct(chunks) == Fraction(1, 1 << 16382)


def test_from_float128_largest_finite():
    bits = _f128_bits(0, 32766, (1 << 112) - 1)
    # largest f128 = (2**113 - 1) * 2**16271
    assert _reconstruct(from_float128(bits)) == (2 ** 113 - 1) * (2 ** 16271)


def test_from_float128_random_normals_lossless():
    """For each random 113-bit-mantissa value, the two-chunk emission must
    reassemble to the EXACT original value (bit-for-bit at the rational
    level)."""
    rng = random.Random(0xCAFE_BABE)
    for _ in range(50):
        biased_exp = rng.randrange(1, 32767)
        mantissa_raw = rng.randrange(0, 1 << 112)
        sign_bit = rng.randint(0, 1)
        bits = _f128_bits(sign_bit, biased_exp, mantissa_raw)
        chunks = from_float128(bits)
        # Expected exact value = (1<<112 | mantissa_raw) * 2**(biased_exp - 16383 - 112)
        eff = (1 << 112) | mantissa_raw
        eff_exp = biased_exp - 16383 - 112
        sign = -1 if sign_bit else +1
        if eff_exp >= 0:
            expected = Fraction(sign * eff * (1 << eff_exp), 1)
        else:
            expected = Fraction(sign * eff, 1 << (-eff_exp))
        assert _reconstruct(chunks) == expected


def test_from_float128_random_denormals_lossless():
    rng = random.Random(0xFEED_FACE)
    for _ in range(20):
        mantissa_raw = rng.randrange(1, 1 << 112)
        sign_bit = rng.randint(0, 1)
        bits = _f128_bits(sign_bit, 0, mantissa_raw)
        chunks = from_float128(bits)
        sign = -1 if sign_bit else +1
        expected = Fraction(sign * mantissa_raw, 1 << 16494)
        assert _reconstruct(chunks) == expected


# ----- _encode_fp_normalized with a synthetic f24 layout -----------------------

def test_encode_fp_normalized_synthetic_f24_one():
    """A hypothetical f24 (1 sign + 7 exp + 16 mantissa, bias 63)."""
    mantissa_bits, exponent_bits, bias = 16, 7, 63
    bits = bias << mantissa_bits  # f24(1.0)
    chunks = _encode_fp_normalized(
        bits, mantissa_bits=mantissa_bits, exponent_bits=exponent_bits, bias=bias
    )
    assert _reconstruct(chunks) == 1


def test_encode_fp_normalized_synthetic_f24_smallest_denormal():
    mantissa_bits, exponent_bits, bias = 16, 7, 63
    chunks = _encode_fp_normalized(
        1, mantissa_bits=mantissa_bits, exponent_bits=exponent_bits, bias=bias
    )
    # smallest f24 denormal = 2**(1 - bias - mantissa_bits) = 2**-78
    assert _reconstruct(chunks) == Fraction(1, 1 << 78)


def test_encode_fp_normalized_synthetic_f24_inf():
    mantissa_bits, exponent_bits, bias = 16, 7, 63
    bits = ((1 << exponent_bits) - 1) << mantissa_bits
    chunks = _encode_fp_normalized(
        bits, mantissa_bits=mantissa_bits, exponent_bits=exponent_bits, bias=bias
    )
    sig, _, _ = _unpack(chunks[0])
    assert sig == 0 and _is_infnan_sentinel(chunks[0])


def test_encode_fp_normalized_bits_out_of_range_raises():
    with pytest.raises(ValueError):
        _encode_fp_normalized(
            1 << 32, mantissa_bits=23, exponent_bits=8, bias=127
        )


# ----- module-level constants sanity ------------------------------------------

def test_constants_consistency():
    biased = INFNAN_EXPONENT_UNBIASED + TARGET_EXPONENT_BIAS
    assert biased == (1 << TARGET_EXPONENT_BITS) - 1
    assert TARGET_SIGNIFICAND_BITS == 64
    assert TARGET_EXPONENT_BITS == 31
    assert TARGET_EXPONENT_BIAS == 0x4000_0000
