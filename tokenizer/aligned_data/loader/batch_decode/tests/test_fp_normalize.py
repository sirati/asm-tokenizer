"""Byte-equivalence tests for the vectorized FP normalizer.

Each TokenType branch (F16 / BF16 / F32 / F64 / F80 / F128 / VC2) is exercised
against the per-source Python-loop oracle in
``tokenizer.aligned_data.loader.decoded.custom_float``. The vectorized output
MUST be 1:1 identical to the oracle's ``(significand, sign_exp)`` chunks --
this pins the bit-for-bit replacement.

Test fixtures construct ``inline_bytes`` + ``idx_2d`` arrays directly (no
stage-1 / stage-2 / stage-3 plumbing) to isolate the normalization layer.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._fp_normalize import (
    normalize_per_token_type,
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
from tokenizer.tokens import TokenType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_inline_bytes(payloads: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Build a flat ``inline_bytes`` buffer + an ``idx_2d`` indexer.

    Layout: index 0 is the leading zero pad. Each payload starts at the next
    free offset; ``idx_2d`` rows give the per-source byte offsets.
    """
    total = 1 + sum(len(p) for p in payloads)
    buf = np.zeros(total, dtype=np.uint8)
    if not payloads:
        return buf, np.zeros((0, 0), dtype=np.uint32)
    payload_size = len(payloads[0])
    assert all(len(p) == payload_size for p in payloads), (
        "all payloads in a single idx_2d must be the same width"
    )
    idx_2d = np.zeros((len(payloads), payload_size), dtype=np.uint32)
    offset = 1
    for i, p in enumerate(payloads):
        buf[offset : offset + payload_size] = np.frombuffer(p, dtype=np.uint8)
        idx_2d[i] = np.arange(offset, offset + payload_size, dtype=np.uint32)
        offset += payload_size
    return buf, idx_2d


def _oracle_pairs_to_arrays(
    pairs_per_source: list[list[tuple[np.uint64, np.uint32]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the oracle's per-source chunk lists into parallel arrays."""
    if not pairs_per_source:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint32)
    sigs: list[np.uint64] = []
    sign_exps: list[np.uint32] = []
    for chunks in pairs_per_source:
        for sig, sign_exp in chunks:
            sigs.append(sig)
            sign_exps.append(sign_exp)
    return (
        np.array(sigs, dtype=np.uint64),
        np.array(sign_exps, dtype=np.uint32),
    )


def _run_normalize_simple(
    token_type: TokenType,
    payloads: list[bytes],
) -> tuple[np.ndarray, np.ndarray]:
    """Run normalize for single-chunk-per-source types (F16/BF16/F32/F64/F80).

    Sign comes from the bit pattern; ``is_negative_per_source`` is provided
    but expected to be ignored for these types (we still pass it
    consistently with what stage 2 would set).
    """
    inline_bytes, idx_2d = _build_inline_bytes(payloads)
    n = len(payloads)
    # For these types, sign is in the bit pattern; pass a False placeholder
    # (matching what stage 2 derives -- VC2-style postfix sign is the only
    # value-bearing per-source sign info).
    is_neg = np.zeros(n, dtype=bool)
    out = normalize_per_token_type(
        idx_2d_per_type={token_type: idx_2d},
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=np.zeros(0, dtype=bool),
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        is_negative_per_source_per_type={token_type: is_neg},
    )
    return out[token_type]


def _u16_to_be_bytes(bits: int) -> bytes:
    return struct.pack(">H", bits & 0xFFFF)


def _u32_to_be_bytes(bits: int) -> bytes:
    return struct.pack(">I", bits & 0xFFFFFFFF)


def _u64_to_be_bytes(bits: int) -> bytes:
    return struct.pack(">Q", bits & ((1 << 64) - 1))


def _f80_to_be_bytes(bits: int) -> bytes:
    # 10 bytes big-endian: sign+exp word (2) + mantissa (8).
    sign_exp_word = (bits >> 64) & 0xFFFF
    mantissa = bits & ((1 << 64) - 1)
    return struct.pack(">H", sign_exp_word) + struct.pack(">Q", mantissa)


def _f128_to_be_bytes(bits: int) -> bytes:
    high = (bits >> 64) & ((1 << 64) - 1)
    low = bits & ((1 << 64) - 1)
    return struct.pack(">Q", high) + struct.pack(">Q", low)


def _assert_arrays_equal(
    got: tuple[np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray],
) -> None:
    got_sig, got_sign_exp = got
    exp_sig, exp_sign_exp = expected
    np.testing.assert_array_equal(got_sig, exp_sig)
    np.testing.assert_array_equal(got_sign_exp, exp_sign_exp)


# ---------------------------------------------------------------------------
# F16
# ---------------------------------------------------------------------------


def _f16_bits(value: float) -> int:
    return int(np.frombuffer(np.float16(value).tobytes(), dtype=np.uint16)[0])


F16_NORMAL = _f16_bits(1.5)  # positive normal
F16_NEG_NORMAL = _f16_bits(-3.25)
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_NAN_QUIET = 0x7E00
F16_NAN_SIGNALING = 0x7C01
F16_SMALLEST_NORMAL = 0x0400  # 2^-14
F16_LARGEST_FINITE = 0x7BFF  # 65504
F16_SMALL_DENORMAL = 0x0001  # 2^-24 (smallest positive denormal)
F16_BIG_DENORMAL = 0x03FF  # largest denormal


@pytest.mark.parametrize(
    "bits, label",
    [
        (F16_NORMAL, "normal_pos"),
        (F16_NEG_NORMAL, "normal_neg"),
        (F16_POS_ZERO, "pos_zero"),
        (F16_NEG_ZERO, "neg_zero"),
        (F16_POS_INF, "pos_inf"),
        (F16_NEG_INF, "neg_inf"),
        (F16_NAN_QUIET, "nan_quiet"),
        (F16_NAN_SIGNALING, "nan_signaling"),
        (F16_SMALLEST_NORMAL, "smallest_normal"),
        (F16_LARGEST_FINITE, "largest_finite"),
        (F16_SMALL_DENORMAL, "small_denormal"),
        (F16_BIG_DENORMAL, "big_denormal"),
    ],
)
def test_f16_round_trip_vs_oracle(bits: int, label: str) -> None:
    payloads = [_u16_to_be_bytes(bits)]
    got = _run_normalize_simple(TokenType.FLOAT16, payloads)
    expected = _oracle_pairs_to_arrays([from_float16(bits)])
    _assert_arrays_equal(got, expected)


def test_f16_random_sample_vs_oracle() -> None:
    rng = np.random.default_rng(0xDEADBEEF)
    sample = rng.integers(0, 1 << 16, size=100, dtype=np.uint16)
    payloads = [_u16_to_be_bytes(int(b)) for b in sample]
    got = _run_normalize_simple(TokenType.FLOAT16, payloads)
    expected = _oracle_pairs_to_arrays(
        [from_float16(int(b)) for b in sample]
    )
    _assert_arrays_equal(got, expected)


def test_f16_empty_input() -> None:
    inline_bytes, _ = _build_inline_bytes([])
    idx_2d = np.zeros((0, 2), dtype=np.uint32)
    out = normalize_per_token_type(
        idx_2d_per_type={TokenType.FLOAT16: idx_2d},
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=np.zeros(0, dtype=bool),
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        is_negative_per_source_per_type={
            TokenType.FLOAT16: np.zeros(0, dtype=bool)
        },
    )
    sig, sign_exp = out[TokenType.FLOAT16]
    assert sig.shape == (0,)
    assert sign_exp.shape == (0,)


# ---------------------------------------------------------------------------
# BF16
# ---------------------------------------------------------------------------

# Build BF16 patterns manually: 1 sign + 8 exp + 7 mantissa.
BF16_POS_NORMAL = 0x3F80  # 1.0
BF16_NEG_NORMAL = 0xBF80  # -1.0
BF16_POS_ZERO = 0x0000
BF16_NEG_ZERO = 0x8000
BF16_POS_INF = 0x7F80
BF16_NEG_INF = 0xFF80
BF16_NAN = 0x7FC0
BF16_SMALLEST_NORMAL = 0x0080  # 2^-126
BF16_LARGEST_FINITE = 0x7F7F
BF16_SMALL_DENORMAL = 0x0001
BF16_BIG_DENORMAL = 0x007F


@pytest.mark.parametrize(
    "bits",
    [
        BF16_POS_NORMAL,
        BF16_NEG_NORMAL,
        BF16_POS_ZERO,
        BF16_NEG_ZERO,
        BF16_POS_INF,
        BF16_NEG_INF,
        BF16_NAN,
        BF16_SMALLEST_NORMAL,
        BF16_LARGEST_FINITE,
        BF16_SMALL_DENORMAL,
        BF16_BIG_DENORMAL,
    ],
)
def test_bf16_round_trip_vs_oracle(bits: int) -> None:
    payloads = [_u16_to_be_bytes(bits)]
    got = _run_normalize_simple(TokenType.BFLOAT16, payloads)
    expected = _oracle_pairs_to_arrays([from_bfloat16(bits)])
    _assert_arrays_equal(got, expected)


def test_bf16_random_sample_vs_oracle() -> None:
    rng = np.random.default_rng(0xCAFEBABE)
    sample = rng.integers(0, 1 << 16, size=100, dtype=np.uint16)
    payloads = [_u16_to_be_bytes(int(b)) for b in sample]
    got = _run_normalize_simple(TokenType.BFLOAT16, payloads)
    expected = _oracle_pairs_to_arrays(
        [from_bfloat16(int(b)) for b in sample]
    )
    _assert_arrays_equal(got, expected)


# ---------------------------------------------------------------------------
# F32
# ---------------------------------------------------------------------------


def _f32_bits(value: float) -> int:
    return int(np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0])


F32_NORMAL = _f32_bits(3.14159)
F32_NEG_NORMAL = _f32_bits(-2.71828)
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_NAN_QUIET = 0x7FC00000
F32_NAN_SIGNALING = 0x7F800001
F32_SMALLEST_NORMAL = 0x00800000  # 2^-126
F32_LARGEST_FINITE = 0x7F7FFFFF
F32_SMALL_DENORMAL = 0x00000001
F32_BIG_DENORMAL = 0x007FFFFF


@pytest.mark.parametrize(
    "bits",
    [
        F32_NORMAL,
        F32_NEG_NORMAL,
        F32_POS_ZERO,
        F32_NEG_ZERO,
        F32_POS_INF,
        F32_NEG_INF,
        F32_NAN_QUIET,
        F32_NAN_SIGNALING,
        F32_SMALLEST_NORMAL,
        F32_LARGEST_FINITE,
        F32_SMALL_DENORMAL,
        F32_BIG_DENORMAL,
    ],
)
def test_f32_round_trip_vs_oracle(bits: int) -> None:
    payloads = [_u32_to_be_bytes(bits)]
    got = _run_normalize_simple(TokenType.FLOAT32, payloads)
    expected = _oracle_pairs_to_arrays([from_float32(bits)])
    _assert_arrays_equal(got, expected)


def test_f32_random_sample_vs_oracle() -> None:
    rng = np.random.default_rng(0x12345678)
    sample = rng.integers(0, 1 << 32, size=100, dtype=np.uint32)
    payloads = [_u32_to_be_bytes(int(b)) for b in sample]
    got = _run_normalize_simple(TokenType.FLOAT32, payloads)
    expected = _oracle_pairs_to_arrays(
        [from_float32(int(b)) for b in sample]
    )
    _assert_arrays_equal(got, expected)


# ---------------------------------------------------------------------------
# F64
# ---------------------------------------------------------------------------


def _f64_bits(value: float) -> int:
    return int(np.frombuffer(np.float64(value).tobytes(), dtype=np.uint64)[0])


F64_NORMAL = _f64_bits(3.141592653589793)
F64_NEG_NORMAL = _f64_bits(-2.718281828459045)
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_NAN_QUIET = 0x7FF8000000000000
F64_NAN_SIGNALING = 0x7FF0000000000001
F64_SMALLEST_NORMAL = 0x0010000000000000
F64_LARGEST_FINITE = 0x7FEFFFFFFFFFFFFF
F64_SMALL_DENORMAL = 0x0000000000000001
F64_BIG_DENORMAL = 0x000FFFFFFFFFFFFF


@pytest.mark.parametrize(
    "bits",
    [
        F64_NORMAL,
        F64_NEG_NORMAL,
        F64_POS_ZERO,
        F64_NEG_ZERO,
        F64_POS_INF,
        F64_NEG_INF,
        F64_NAN_QUIET,
        F64_NAN_SIGNALING,
        F64_SMALLEST_NORMAL,
        F64_LARGEST_FINITE,
        F64_SMALL_DENORMAL,
        F64_BIG_DENORMAL,
    ],
)
def test_f64_round_trip_vs_oracle(bits: int) -> None:
    payloads = [_u64_to_be_bytes(bits)]
    got = _run_normalize_simple(TokenType.FLOAT64, payloads)
    expected = _oracle_pairs_to_arrays([from_float64(bits)])
    _assert_arrays_equal(got, expected)


def test_f64_random_sample_vs_oracle() -> None:
    rng = np.random.default_rng(0x87654321)
    sample = rng.integers(0, 1 << 64, size=100, dtype=np.uint64)
    payloads = [_u64_to_be_bytes(int(b)) for b in sample]
    got = _run_normalize_simple(TokenType.FLOAT64, payloads)
    expected = _oracle_pairs_to_arrays(
        [from_float64(int(b)) for b in sample]
    )
    _assert_arrays_equal(got, expected)


# ---------------------------------------------------------------------------
# F80
# ---------------------------------------------------------------------------

# F80 patterns: 80 bits = sign(1) + exp(15) + mantissa(64).
F80_POS_ZERO = 0x0000_0000000000000000
F80_NEG_ZERO = 0x8000_0000000000000000
# 1.0: biased_exp = 16383, explicit-leading = 1, raw_mantissa = 1 << 63.
F80_ONE = (16383 << 64) | (1 << 63)
F80_NEG_ONE = (1 << 79) | (16383 << 64) | (1 << 63)
# +Inf: biased_exp = 0x7FFF, mantissa = 1 << 63.
F80_POS_INF = (0x7FFF << 64) | (1 << 63)
F80_NEG_INF = (1 << 79) | (0x7FFF << 64) | (1 << 63)
# NaN: biased_exp = 0x7FFF, mantissa != 1 << 63.
F80_NAN = (0x7FFF << 64) | ((1 << 63) | 0x1234)
# Smallest normal: biased_exp = 1, explicit-leading = 1.
F80_SMALLEST_NORMAL = (1 << 64) | (1 << 63)
# Largest finite: biased_exp = 0x7FFE.
F80_LARGEST_FINITE = (0x7FFE << 64) | ((1 << 64) - 1)
# Pseudo-denormal: biased_exp = 0, mantissa < (1 << 63).
F80_DENORMAL_SMALL = 0x0000_0000000000000001
F80_DENORMAL_BIG = 0x0000_7FFFFFFFFFFFFFFF
# Unnormal: biased_exp > 0, explicit-leading = 0 (invalid per x87 but oracle
# routes through denormal path).
F80_UNNORMAL = (10 << 64) | 0x1234567890ABCDEF  # explicit_leading bit = 0


@pytest.mark.parametrize(
    "bits, label",
    [
        (F80_POS_ZERO, "pos_zero"),
        (F80_NEG_ZERO, "neg_zero"),
        (F80_ONE, "one"),
        (F80_NEG_ONE, "neg_one"),
        (F80_POS_INF, "pos_inf"),
        (F80_NEG_INF, "neg_inf"),
        (F80_NAN, "nan"),
        (F80_SMALLEST_NORMAL, "smallest_normal"),
        (F80_LARGEST_FINITE, "largest_finite"),
        (F80_DENORMAL_SMALL, "denormal_small"),
        (F80_DENORMAL_BIG, "denormal_big"),
        (F80_UNNORMAL, "unnormal"),
    ],
)
def test_f80_round_trip_vs_oracle(bits: int, label: str) -> None:
    payloads = [_f80_to_be_bytes(bits)]
    got = _run_normalize_simple(TokenType.FLOAT80, payloads)
    expected = _oracle_pairs_to_arrays([from_float80(bits)])
    _assert_arrays_equal(got, expected)


def test_f80_random_sample_vs_oracle() -> None:
    rng = np.random.default_rng(0xF80F80F8)
    # F80 is 80 bits; sample by drawing 80 bits via two integer draws.
    n = 100
    high_words = rng.integers(0, 1 << 16, size=n, dtype=np.uint64)
    low_qwords = rng.integers(0, 1 << 64, size=n, dtype=np.uint64)
    sample = [(int(h) << 64) | int(low) for h, low in zip(high_words, low_qwords)]
    payloads = [_f80_to_be_bytes(b) for b in sample]
    got = _run_normalize_simple(TokenType.FLOAT80, payloads)
    expected = _oracle_pairs_to_arrays([from_float80(b) for b in sample])
    _assert_arrays_equal(got, expected)


# ---------------------------------------------------------------------------
# F128
# ---------------------------------------------------------------------------


def _f128_finite_bits(
    sign: int, biased_exp: int, raw_mantissa: int
) -> int:
    assert 0 <= biased_exp < (1 << 15)
    assert 0 <= raw_mantissa < (1 << 112)
    return (
        (sign << 127) | (biased_exp << 112) | raw_mantissa
    )


F128_POS_ZERO = 0
F128_NEG_ZERO = 1 << 127
F128_ONE = _f128_finite_bits(0, 16383, 0)
F128_NEG_ONE = _f128_finite_bits(1, 16383, 0)
F128_POS_INF = _f128_finite_bits(0, 0x7FFF, 0)
F128_NEG_INF = _f128_finite_bits(1, 0x7FFF, 0)
F128_NAN = _f128_finite_bits(0, 0x7FFF, 0xDEAD)
F128_SMALLEST_NORMAL = _f128_finite_bits(0, 1, 0)
F128_LARGEST_FINITE = _f128_finite_bits(0, 0x7FFE, (1 << 112) - 1)
F128_SMALL_DENORMAL = _f128_finite_bits(0, 0, 1)
F128_BIG_DENORMAL = _f128_finite_bits(0, 0, (1 << 112) - 1)
# Mix of high+low chunks for a finite normal value:
F128_MIXED = _f128_finite_bits(
    1, 16400, 0xCAFEBABE_DEADBEEF_BADC0FFEE_1234_5678 & ((1 << 112) - 1)
)


def _f128_is_nan_or_inf_from_bits(bits: int) -> bool:
    return ((bits >> 112) & 0x7FFF) == 0x7FFF


def _f128_batch_expected_chunks(
    bits: int,
) -> list[tuple[np.uint64, np.uint32]]:
    """Per-source expected ``(sig, sign_exp)`` chunks under the BATCH
    pipeline's fixed-layout rule.

    The batch pipeline ALWAYS emits 2 chunks per finite F128 source (one per
    u64 limb -- low + high), unlike the ``from_float128`` oracle which short-
    circuits to a single chunk when the effective_mantissa fits in u64 (e.g.
    F128 +/-0, denormals with bit_length <= 64). This helper expresses the
    plan's fixed-layout rule; for finite values, both chunks share the same
    sign + the per-chunk normalization is identical to the oracle's
    ``_emit_chunk`` formula.

    NaN/Inf: 1 chunk; Inf vs NaN classification uses ONLY the high-
    mantissa bits (.rodata-robustness policy -- 3c drops the low limb
    for NaN/Inf sources so 3d can't see it). This mirrors the new
    per-chunk normalize_f128 contract.
    """
    from tokenizer.aligned_data.loader.decoded.custom_float import (
        _emit_chunk,
        _encode_infnan,
    )

    sign_bit = (bits >> 127) & 1
    biased_exp = (bits >> 112) & 0x7FFF
    raw_mantissa = bits & ((1 << 112) - 1)
    sign = -1 if sign_bit else +1
    if biased_exp == 0x7FFF:
        # NaN/Inf: 1 chunk via _encode_infnan(sign, mantissa_is_zero).
        # High mantissa = bits [64, 112) of raw_mantissa = top 48 bits.
        high_mantissa = (raw_mantissa >> 64) & ((1 << 48) - 1)
        is_inf = high_mantissa == 0
        return [_encode_infnan(sign=sign, mantissa_is_zero=is_inf)]
    # Finite path: ALWAYS 2 chunks (low + high).
    bias = 16383
    if biased_exp == 0:
        actual_exp = 1 - bias
        effective_mantissa = raw_mantissa
    else:
        actual_exp = biased_exp - bias
        effective_mantissa = (1 << 112) | raw_mantissa
    base = actual_exp - 112  # leading_bit_position = mantissa_bits = 112
    low_chunk = effective_mantissa & ((1 << 64) - 1)
    high_chunk = (effective_mantissa >> 64) & ((1 << 64) - 1)
    return [
        _emit_chunk(low_chunk, sign=sign, chunk_exponent_base=base),
        _emit_chunk(high_chunk, sign=sign, chunk_exponent_base=base + 64),
    ]


def _build_f128_per_chunk_fixture(
    bits_list: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``inline_bytes`` + per-chunk ``idx_2d`` for the F128 layout.

    Per-source layout in ``idx_2d`` (matches 3c's emission contract):

    * Finite source (``biased_exp != 0x7FFF``): 2 rows -- row 0 = LSB
      limb (bytes 8..15 of the original 16-byte payload), row 1 = MSB
      limb (bytes 0..7).
    * NaN/Inf source (``biased_exp == 0x7FFF``): 1 row = MSB limb
      (bytes 0..7).

    Returns ``(inline_bytes, idx_2d, f128_is_nan_or_inf)``.
    """
    # Lay out each source's 16 bytes contiguously in inline_bytes after
    # the leading pad. idx_2d's per-chunk rows then point into that
    # contiguous region at the right 8-byte windows.
    total_payload_bytes = 16 * len(bits_list)
    inline_bytes = np.zeros(1 + total_payload_bytes, dtype=np.uint8)
    rows: list[np.ndarray] = []
    is_nan_or_inf_per_source: list[bool] = []
    offset = 1  # index 0 = leading zero pad
    for bits in bits_list:
        payload = _f128_to_be_bytes(bits)
        assert len(payload) == 16
        inline_bytes[offset : offset + 16] = np.frombuffer(payload, dtype=np.uint8)
        msb_byte_offset = offset  # bytes 0..7 of the payload
        lsb_byte_offset = offset + 8  # bytes 8..15
        nan_or_inf = _f128_is_nan_or_inf_from_bits(bits)
        is_nan_or_inf_per_source.append(nan_or_inf)
        if nan_or_inf:
            # NaN/Inf source: 1 row = MSB limb.
            rows.append(
                np.arange(msb_byte_offset, msb_byte_offset + 8, dtype=np.uint32)
            )
        else:
            # Finite source: 2 rows, LSB then MSB.
            rows.append(
                np.arange(lsb_byte_offset, lsb_byte_offset + 8, dtype=np.uint32)
            )
            rows.append(
                np.arange(msb_byte_offset, msb_byte_offset + 8, dtype=np.uint32)
            )
        offset += 16
    if rows:
        idx_2d = np.stack(rows, axis=0)
    else:
        idx_2d = np.zeros((0, 8), dtype=np.uint32)
    return (
        inline_bytes,
        idx_2d,
        np.array(is_nan_or_inf_per_source, dtype=bool),
    )


def _run_f128(bits_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
    inline_bytes, idx_2d, nan_or_inf = _build_f128_per_chunk_fixture(bits_list)
    is_neg = np.zeros(len(bits_list), dtype=bool)  # F128 sign in bit pattern
    out = normalize_per_token_type(
        idx_2d_per_type={TokenType.FLOAT128: idx_2d},
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=nan_or_inf,
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        is_negative_per_source_per_type={TokenType.FLOAT128: is_neg},
    )
    return out[TokenType.FLOAT128]


@pytest.mark.parametrize(
    "bits, label",
    [
        (F128_POS_ZERO, "pos_zero"),
        (F128_NEG_ZERO, "neg_zero"),
        (F128_ONE, "one"),
        (F128_NEG_ONE, "neg_one"),
        (F128_POS_INF, "pos_inf"),
        (F128_NEG_INF, "neg_inf"),
        (F128_NAN, "nan"),
        (F128_SMALLEST_NORMAL, "smallest_normal"),
        (F128_LARGEST_FINITE, "largest_finite"),
        (F128_SMALL_DENORMAL, "small_denormal"),
        (F128_BIG_DENORMAL, "big_denormal"),
        (F128_MIXED, "mixed_finite_normal"),
    ],
)
def test_f128_round_trip_vs_oracle(bits: int, label: str) -> None:
    """Per-source byte-equivalence against the batch pipeline's fixed-layout
    expected output.

    The batch pipeline always emits 2 chunks per finite F128 source (1 per
    NaN/Inf source); the ``from_float128`` oracle short-circuits finite ones
    with effective_mantissa.bit_length() <= 64 down to a single chunk. The
    per-chunk normalization (``_emit_chunk`` formula) is identical between
    the two paths -- this test pins that equivalence over the full
    representative coverage.
    """
    got = _run_f128([bits])
    expected = _oracle_pairs_to_arrays([_f128_batch_expected_chunks(bits)])
    _assert_arrays_equal(got, expected)


def test_f128_finite_emits_two_chunks_nanorinf_emits_one() -> None:
    # Mix: 1 finite (2 chunks) + 1 NaN (1 chunk) + 1 finite (2 chunks).
    got = _run_f128([F128_ONE, F128_NAN, F128_NEG_ONE])
    assert got[0].shape == (5,)
    assert got[1].shape == (5,)
    expected = _oracle_pairs_to_arrays(
        [
            _f128_batch_expected_chunks(F128_ONE),
            _f128_batch_expected_chunks(F128_NAN),
            _f128_batch_expected_chunks(F128_NEG_ONE),
        ]
    )
    _assert_arrays_equal(got, expected)


def test_f128_normal_byte_equivalent_to_from_float128() -> None:
    """For F128 *normals* (effective_mantissa has bit 112 set), the batch
    pipeline's 2-chunk emission is byte-identical to ``from_float128``.

    This pins the per-chunk normalization equivalence on the values that
    actually overlap between the two paths.
    """
    rng = np.random.default_rng(0xF128AA88)
    n = 50
    bits_list = []
    for _ in range(n):
        # biased_exp in [1, 0x7FFE] -> normal (not denormal, not NaN/Inf)
        biased_exp = int(rng.integers(1, 0x7FFF))
        # Build a 112-bit mantissa from two u64 draws (numpy can't draw a
        # 112-bit integer directly).
        m_high = int(rng.integers(0, 1 << 48, dtype=np.uint64))  # top 48 bits
        m_low = int(rng.integers(0, 1 << 64, dtype=np.uint64))   # low 64 bits
        mantissa = (m_high << 64) | m_low
        sign = int(rng.integers(0, 2))
        bits_list.append((sign << 127) | (biased_exp << 112) | mantissa)
    got = _run_f128(bits_list)
    expected = _oracle_pairs_to_arrays([from_float128(b) for b in bits_list])
    _assert_arrays_equal(got, expected)


def test_f128_random_sample_vs_batch_layout() -> None:
    """Full-coverage random sample including denormals + zero. Uses the
    batch-layout expected builder."""
    rng = np.random.default_rng(0xF128F128)
    n = 100
    high_qwords = rng.integers(0, 1 << 64, size=n, dtype=np.uint64)
    low_qwords = rng.integers(0, 1 << 64, size=n, dtype=np.uint64)
    sample = [(int(h) << 64) | int(low) for h, low in zip(high_qwords, low_qwords)]
    got = _run_f128(sample)
    expected = _oracle_pairs_to_arrays(
        [_f128_batch_expected_chunks(b) for b in sample]
    )
    _assert_arrays_equal(got, expected)


def test_f128_sidecar_mismatch_raises() -> None:
    """Defensive check: f128_is_nan_or_inf sidecar must agree with the bit
    pattern. A wrong sidecar value is a stage-2 invariant violation.

    We hand-build a fixture where the sidecar says NaN/Inf (so 3c would
    have emitted only the MSB row) but we instead supply the MSB row of
    a FINITE F128 value. The normalizer's bit-pattern check should fire.
    """
    payload = _f128_to_be_bytes(F128_ONE)
    # 3c-style fixture: claim NaN/Inf -> 1 chunk = MSB limb (bytes 0..7).
    inline_bytes = np.zeros(1 + 8, dtype=np.uint8)
    inline_bytes[1:9] = np.frombuffer(payload[:8], dtype=np.uint8)
    idx_2d = np.arange(1, 9, dtype=np.uint32)[np.newaxis, :]
    # F128_ONE is finite (biased_exp == 16383), but we claim NaN/Inf.
    nan_or_inf = np.array([True], dtype=bool)
    with pytest.raises(AssertionError, match="f128_is_nan_or_inf"):
        normalize_per_token_type(
            idx_2d_per_type={TokenType.FLOAT128: idx_2d},
            inline_bytes=inline_bytes,
            f128_is_nan_or_inf=nan_or_inf,
            vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
            is_negative_per_source_per_type={
                TokenType.FLOAT128: np.zeros(1, dtype=bool)
            },
        )


# ---------------------------------------------------------------------------
# VC2 (valued_const_v2)
# ---------------------------------------------------------------------------


def _vc2_chunks_for_int(value: int) -> list[int]:
    """Split a non-negative integer into low-first u64 chunks (matches
    custom_float._split_to_chunks)."""
    if value == 0:
        return [0]
    chunks = []
    v = value
    while v > 0:
        chunks.append(v & ((1 << 64) - 1))
        v >>= 64
    return chunks


def _build_vc2_fixture(
    sources: list[tuple[int, int]],  # (value, sign:+1/-1)
) -> tuple[
    np.ndarray,  # inline_bytes
    np.ndarray,  # idx_2d (per-chunk rows, 8 bytes each)
    np.ndarray,  # vc2_chunk_exponent_sidecar
    np.ndarray,  # is_negative_per_source
]:
    """Construct a VC2 fixture: per-source values get split into chunks; each
    chunk is laid out as 8 bytes big-endian in inline_bytes; idx_2d has one
    row per chunk."""
    # Build chunks per source.
    all_chunks: list[bytes] = []
    sidecar: list[int] = []
    is_neg: list[bool] = []
    for value, sign in sources:
        chunks = _vc2_chunks_for_int(value)
        is_neg.append(sign < 0)
        for i, c in enumerate(chunks):
            all_chunks.append(_u64_to_be_bytes(c))
            sidecar.append(i)
    inline_bytes, idx_2d = _build_inline_bytes(all_chunks)
    return (
        inline_bytes,
        idx_2d,
        np.array(sidecar, dtype=np.uint32),
        np.array(is_neg, dtype=bool),
    )


def _run_vc2(
    sources: list[tuple[int, int]],
) -> tuple[
    tuple[np.ndarray, np.ndarray], list[list[tuple[np.uint64, np.uint32]]]
]:
    """Run VC2 normalize + collect oracle chunks for comparison."""
    inline_bytes, idx_2d, sidecar, is_neg = _build_vc2_fixture(sources)
    out = normalize_per_token_type(
        idx_2d_per_type={TokenType.VALUED_CONST_V2: idx_2d},
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=np.zeros(0, dtype=bool),
        vc2_chunk_exponent_sidecar=sidecar,
        is_negative_per_source_per_type={
            TokenType.VALUED_CONST_V2: is_neg,
        },
    )
    oracle_chunks = [from_int(v, sign=s) for v, s in sources]
    return out[TokenType.VALUED_CONST_V2], oracle_chunks


def test_vc2_single_chunk_small_positive() -> None:
    got, oracle = _run_vc2([(42, +1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


def test_vc2_single_chunk_small_negative() -> None:
    got, oracle = _run_vc2([(42, -1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


def test_vc2_zero_pos_sign() -> None:
    """+0 chunk emits sig=0 with sign-bit clear, exponent=base
    (pack_sign_exp(sign, 0))."""
    got, oracle = _run_vc2([(0, +1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)
    # Sanity:
    assert got[0][0] == 0
    # sign_exp == pack_sign_exp(+1, 0) == TARGET_EXPONENT_BIAS.
    from tokenizer.aligned_data.loader.decoded.custom_float import (
        TARGET_EXPONENT_BIAS,
    )
    assert got[1][0] == np.uint32(TARGET_EXPONENT_BIAS)


def test_vc2_zero_neg_sign() -> None:
    """-0 chunk emits sig=0 with sign-bit set."""
    got, oracle = _run_vc2([(0, -1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


def test_vc2_full_u64_at_bit_63() -> None:
    """Chunk with leading 1 already at bit 63 -- shift = 0."""
    got, oracle = _run_vc2([(1 << 63, +1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


def test_vc2_max_u64() -> None:
    got, oracle = _run_vc2([((1 << 64) - 1, +1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


def test_vc2_three_chunk_integer() -> None:
    """A 3-chunk integer: each chunk normalized; chunk exponent_base
    correct."""
    value = (0xABCD_EF12 << 128) | (0x3456_7890 << 64) | 0xCAFE_BABE
    got, oracle = _run_vc2([(value, +1)])
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)
    # Sanity: should be 3 chunks.
    assert got[0].shape == (3,)


def test_vc2_multi_source_mixed_sign() -> None:
    sources = [
        (1234567890, +1),
        (0, -1),
        ((1 << 70), +1),  # 2 chunks
        ((1 << 64) | 0xABC, -1),  # 2 chunks
        (1, +1),
        (0, +1),
    ]
    got, oracle = _run_vc2(sources)
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


def test_vc2_random_sample_vs_oracle() -> None:
    rng = np.random.default_rng(0x1C20232)
    n = 100
    sources: list[tuple[int, int]] = []
    for _ in range(n):
        # mix of widths: 1..3 chunks
        nchunks = int(rng.integers(1, 4))
        value = 0
        for k in range(nchunks):
            limb = int(rng.integers(0, 1 << 64, dtype=np.uint64))
            value |= limb << (64 * k)
        sign = +1 if rng.integers(0, 2) == 0 else -1
        sources.append((value, sign))
    got, oracle = _run_vc2(sources)
    expected = _oracle_pairs_to_arrays(oracle)
    _assert_arrays_equal(got, expected)


# ---------------------------------------------------------------------------
# Cross-cutting / API sanity
# ---------------------------------------------------------------------------


def test_normalize_rejects_non_number_token_type() -> None:
    """Identity-band TokenTypes shouldn't reach this layer."""
    inline_bytes = np.zeros(1, dtype=np.uint8)
    idx_2d = np.array([[0, 0]], dtype=np.uint32)
    with pytest.raises(ValueError, match="NUMBER-band"):
        normalize_per_token_type(
            idx_2d_per_type={TokenType.BLOCK_V2: idx_2d},
            inline_bytes=inline_bytes,
            f128_is_nan_or_inf=np.zeros(0, dtype=bool),
            vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
            is_negative_per_source_per_type={
                TokenType.BLOCK_V2: np.zeros(1, dtype=bool)
            },
        )


def test_normalize_handles_multiple_token_types_in_one_call() -> None:
    """API should be reentrant across types within a single call."""
    f16_payloads = [_u16_to_be_bytes(F16_NORMAL), _u16_to_be_bytes(F16_POS_INF)]
    f64_payloads = [_u64_to_be_bytes(F64_NORMAL)]
    f16_bytes, f16_idx = _build_inline_bytes(f16_payloads)
    # Re-use the same inline_bytes; append F64 payloads after F16.
    # Easier: build a single inline_bytes containing both blobs sequentially.
    total = 1 + sum(len(p) for p in f16_payloads) + sum(
        len(p) for p in f64_payloads
    )
    inline_bytes = np.zeros(total, dtype=np.uint8)
    offset = 1
    f16_idx_2d = np.zeros((2, 2), dtype=np.uint32)
    for i, p in enumerate(f16_payloads):
        inline_bytes[offset : offset + 2] = np.frombuffer(p, dtype=np.uint8)
        f16_idx_2d[i] = np.arange(offset, offset + 2, dtype=np.uint32)
        offset += 2
    f64_idx_2d = np.zeros((1, 8), dtype=np.uint32)
    for i, p in enumerate(f64_payloads):
        inline_bytes[offset : offset + 8] = np.frombuffer(p, dtype=np.uint8)
        f64_idx_2d[i] = np.arange(offset, offset + 8, dtype=np.uint32)
        offset += 8

    out = normalize_per_token_type(
        idx_2d_per_type={
            TokenType.FLOAT16: f16_idx_2d,
            TokenType.FLOAT64: f64_idx_2d,
        },
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=np.zeros(0, dtype=bool),
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        is_negative_per_source_per_type={
            TokenType.FLOAT16: np.zeros(2, dtype=bool),
            TokenType.FLOAT64: np.zeros(1, dtype=bool),
        },
    )
    expected_f16 = _oracle_pairs_to_arrays(
        [from_float16(F16_NORMAL), from_float16(F16_POS_INF)]
    )
    expected_f64 = _oracle_pairs_to_arrays([from_float64(F64_NORMAL)])
    _assert_arrays_equal(out[TokenType.FLOAT16], expected_f16)
    _assert_arrays_equal(out[TokenType.FLOAT64], expected_f64)
