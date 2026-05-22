"""Tests for ``decoded.extract.decode_raw_tokens``.

Covers the single-function decode pass end-to-end against hand-built v2
streams.  The fixture ``_make_id_maps`` is the single source of truth
for the synthetic vocab-id values used in this file -- no test inlines
a literal token-id, so renumbering happens in one place if the dispatch
contract ever changes.
"""

from __future__ import annotations

import struct
from typing import Dict, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.custom_float import (
    from_float32,
    from_float64,
    from_float128,
    from_int,
)
from tokenizer.aligned_data.loader.decoded.extract import decode_raw_tokens
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Fixture: deterministic synthetic vocab-id maps.  Values are chosen so
# every id sits above the digit-slot range (>= 256) and is unique across
# the union of the two dicts.
# ---------------------------------------------------------------------------


def _make_id_maps() -> Tuple[Dict[Category, int], Dict[TokenType, int]]:
    id_token_ids: Dict[Category, int] = {
        Category.BLOCK: 300,
        Category.LOCAL_FUNC: 301,
        Category.PLT_FUNC: 302,
        Category.EXT_FUNC: 303,
        Category.RO_DATA_PTR: 304,
        Category.RW_DATA_PTR: 305,
        Category.STRING_PTR: 306,
        Category.JUMP_TABLE: 307,
    }
    number_token_ids: Dict[TokenType, int] = {
        TokenType.VALUED_CONST_V2: 400,
        TokenType.FLOAT16: 401,
        TokenType.BFLOAT16: 402,
        TokenType.FLOAT32: 403,
        TokenType.FLOAT64: 404,
        TokenType.FLOAT80: 405,
        TokenType.FLOAT128: 406,
    }
    return id_token_ids, number_token_ids


def _u16(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.uint16)


def _decode(raw, **overrides) -> "decode_raw_tokens":
    id_token_ids, number_token_ids = _make_id_maps()
    kwargs = dict(
        id_token_ids=id_token_ids,
        number_token_ids=number_token_ids,
        func_name="t",
    )
    kwargs.update(overrides)
    return decode_raw_tokens(raw, **kwargs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_stream_returns_all_empty_arrays():
    out = _decode(_u16())
    assert out.real_tokens.shape == (0,)
    assert out.real_tokens.dtype == np.uint16
    assert set(out.identities.keys()) == set(Category)
    for c in Category:
        assert out.identities[c].shape == (0,)
        assert out.identities[c].dtype == np.uint16
    assert out.numbers_significant.shape == (0,)
    assert out.numbers_sign_exponent.shape == (0,)


def test_first_position_inline_byte_raises():
    raw = _u16(42, 300)  # 42 is in the digit-slot range -> illegal stream
    with pytest.raises(AssertionError):
        _decode(raw)


# ---------------------------------------------------------------------------
# Identity arm
# ---------------------------------------------------------------------------


def test_single_real_token_no_inline_yields_identity_zero():
    """A BLOCK_V2 token alone at position 0 has 0 trailing inline bytes;
    plan decision 7 + identity decoder treat empty payload as identity=0
    (sentinel only fires on > 0xFFFE, not on empty)."""
    raw = _u16(300)  # Category.BLOCK type-id
    out = _decode(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(300))
    np.testing.assert_array_equal(out.identities[Category.BLOCK], _u16(0))
    for c in Category:
        if c is Category.BLOCK:
            continue
        assert out.identities[c].shape == (0,)
    assert out.numbers_significant.shape == (0,)


def test_single_identity_inline_value_42():
    raw = _u16(301, 42)  # LOCAL_FUNC, inline-byte 42
    out = _decode(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(301))
    np.testing.assert_array_equal(out.identities[Category.LOCAL_FUNC], _u16(42))
    assert out.numbers_significant.shape == (0,)


def test_identity_overflow_clips_to_0xFFFF_sentinel():
    # value 0x10000 (3 bytes: 0x01 0x00 0x00) exceeds 0xFFFE -> sentinel.
    raw = _u16(301, 0x01, 0x00, 0x00)
    out = _decode(raw)
    np.testing.assert_array_equal(
        out.identities[Category.LOCAL_FUNC], _u16(0xFFFF)
    )


def test_identity_at_max_non_sentinel_stays_intact():
    # value 0xFFFE -> stored as-is; only > 0xFFFE clips.
    raw = _u16(301, 0xFF, 0xFE)
    out = _decode(raw)
    np.testing.assert_array_equal(
        out.identities[Category.LOCAL_FUNC], _u16(0xFFFE)
    )


def test_multiple_same_category_identities_in_stream_order():
    # Three LOCAL_FUNC tokens with identities 5, 100, 0x100.
    raw = _u16(
        301, 5,
        301, 100,
        301, 0x01, 0x00,
    )
    out = _decode(raw)
    np.testing.assert_array_equal(
        out.identities[Category.LOCAL_FUNC], _u16(5, 100, 0x100)
    )
    # Real-token stream keeps three real tokens; inline bytes stripped.
    np.testing.assert_array_equal(out.real_tokens, _u16(301, 301, 301))


def test_two_different_categories_interleaved():
    raw = _u16(
        301, 7,   # LOCAL_FUNC = 7
        304, 9,   # RO_DATA_PTR = 9
        301, 8,   # LOCAL_FUNC = 8
    )
    out = _decode(raw)
    np.testing.assert_array_equal(
        out.identities[Category.LOCAL_FUNC], _u16(7, 8)
    )
    np.testing.assert_array_equal(
        out.identities[Category.RO_DATA_PTR], _u16(9)
    )
    for c in Category:
        if c in (Category.LOCAL_FUNC, Category.RO_DATA_PTR):
            continue
        assert out.identities[c].shape == (0,)


def test_all_eight_categories_present_each_with_one_occurrence():
    id_token_ids, number_token_ids = _make_id_maps()
    tokens = []
    for category, type_id in id_token_ids.items():
        # Each gets identity = ord(name[0]) so they differ.
        tokens.append(type_id)
        tokens.append(ord(category.name[0]))
    raw = np.array(tokens, dtype=np.uint16)
    out = decode_raw_tokens(
        raw,
        id_token_ids=id_token_ids,
        number_token_ids=number_token_ids,
        func_name="t",
    )
    for category in Category:
        assert out.identities[category].shape == (1,)
        assert int(out.identities[category][0]) == ord(category.name[0])
    assert out.numbers_significant.shape == (0,)


# ---------------------------------------------------------------------------
# Number arm: single-chunk floats
# ---------------------------------------------------------------------------


def test_single_float32_inline_value_matches_from_float32():
    # Pick the bit pattern for the float 3.14: 0x4048f5c3.
    bits = struct.unpack(">I", struct.pack(">f", 3.14))[0]
    payload_bytes = bits.to_bytes(4, byteorder="big", signed=False)
    raw = _u16(403, *payload_bytes)  # FLOAT32 = 403
    out = _decode(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(403))
    expected_chunks = from_float32(bits)
    assert out.numbers_significant.shape == (len(expected_chunks),)
    for idx, (sig, sign_exp) in enumerate(expected_chunks):
        assert int(out.numbers_significant[idx]) == int(sig)
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp)


# ---------------------------------------------------------------------------
# Number arm: multi-chunk integer + FLOAT128 promotion
# ---------------------------------------------------------------------------


def test_valued_const_u128_multi_chunk_promotion_and_alignment():
    # value = 2**100 + 7 -> bit_length 101 -> 2 chunks via from_int.
    value = (1 << 100) + 7
    payload_bytes = value.to_bytes(
        (value.bit_length() + 7) // 8, byteorder="big", signed=False
    )
    inline_len = len(payload_bytes)
    # ceil(inline_len / 8) chunks expected.
    expected_chunk_count = (inline_len + 7) // 8
    raw = _u16(400, *payload_bytes)  # VALUED_CONST_V2 = 400
    out = _decode(raw)
    assert expected_chunk_count >= 2  # sanity for the test design
    # Strip-and-promote: real_tokens has chunk_count consecutive VALUED_CONST_V2
    # tokens; all inline bytes are stripped.
    np.testing.assert_array_equal(
        out.real_tokens, np.full(expected_chunk_count, 400, dtype=np.uint16)
    )
    expected = from_int(value, sign=+1)
    assert len(expected) == expected_chunk_count
    for idx, (sig, sign_exp) in enumerate(expected):
        assert int(out.numbers_significant[idx]) == int(sig)
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp)


def test_float128_finite_two_chunk_lossless_promotion():
    # Build a finite f128 bit pattern: mantissa = 0x1234_5678_9ABC_DEF0_1122,
    # exponent biased = 16383 (unbiased 0), sign = 0.
    bias = 16383
    biased_exp = bias  # value with exponent 0
    # 112-bit mantissa (stored without leading implicit-1).
    mantissa_low = 0x1234_5678_9ABC_DEF0
    mantissa_high_top = 0x1122
    mantissa = (mantissa_high_top << 64) | mantissa_low  # 112-bit value
    assert mantissa < (1 << 112)
    sign = 0
    bits = (sign << 127) | (biased_exp << 112) | mantissa
    payload_bytes = bits.to_bytes(16, byteorder="big", signed=False)
    raw = _u16(406, *payload_bytes)  # FLOAT128 = 406
    out = _decode(raw)
    # 2 chunks per plan decision 15.
    np.testing.assert_array_equal(
        out.real_tokens, _u16(406, 406)
    )
    expected = from_float128(bits)
    assert len(expected) == 2
    assert int(out.numbers_significant[0]) == int(expected[0][0])
    assert int(out.numbers_sign_exponent[0]) == int(expected[0][1])
    assert int(out.numbers_significant[1]) == int(expected[1][0])
    assert int(out.numbers_sign_exponent[1]) == int(expected[1][1])


def test_float128_positive_infinity_single_chunk():
    # f128 +Inf: biased_exp all-ones (0x7FFF), mantissa 0, sign 0.
    bias = 16383
    biased_exp = (1 << 15) - 1  # 0x7FFF
    bits = (biased_exp << 112)
    payload_bytes = bits.to_bytes(16, byteorder="big", signed=False)
    raw = _u16(406, *payload_bytes)
    out = _decode(raw)
    # Plan decision 14: NaN/Inf always single chunk regardless of source
    # width; promotion path bypassed.  Real_tokens has ONE FLOAT128 token.
    np.testing.assert_array_equal(out.real_tokens, _u16(406))
    expected = from_float128(bits)
    assert len(expected) == 1
    assert int(out.numbers_significant[0]) == int(expected[0][0])
    assert int(out.numbers_sign_exponent[0]) == int(expected[0][1])


# ---------------------------------------------------------------------------
# Mixed identity + number streams
# ---------------------------------------------------------------------------


def test_mixed_identity_and_number_streams():
    # BLOCK (id=1) + VALUED_CONST_V2 (value=0x42, u64-fits) + LOCAL_FUNC
    # (id=5) + FLOAT64 (some bits).
    block_inline = (0x01,)
    vc_value = 0x42
    vc_bytes = (vc_value,)
    lf_inline = (5,)
    f64_bits = struct.unpack(">Q", struct.pack(">d", 1.5))[0]
    f64_bytes = tuple(f64_bits.to_bytes(8, byteorder="big", signed=False))
    raw = _u16(
        300, *block_inline,    # BLOCK = 1
        400, *vc_bytes,        # VALUED_CONST_V2 = 0x42
        301, *lf_inline,       # LOCAL_FUNC = 5
        404, *f64_bytes,       # FLOAT64 = 1.5
    )
    out = _decode(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(300, 400, 301, 404))
    np.testing.assert_array_equal(out.identities[Category.BLOCK], _u16(1))
    np.testing.assert_array_equal(out.identities[Category.LOCAL_FUNC], _u16(5))
    expected_vc = from_int(vc_value, sign=+1)
    expected_f64 = from_float64(f64_bits)
    expected = expected_vc + expected_f64
    assert out.numbers_significant.shape == (len(expected),)
    for idx, (sig, sign_exp) in enumerate(expected):
        assert int(out.numbers_significant[idx]) == int(sig)
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp)


def test_number_entries_in_stream_position_order_not_clustered_by_type():
    """Regression: side-array entries follow stream position order, NOT
    a "process FLOAT64 occurrences, then VALUED_CONST_V2 occurrences"
    clustering.  This is critical for Phase 3 splicing because consumers
    map a real-token's position to a side-array index by counting
    number-token occurrences up to that position."""
    bits_b = struct.unpack(">Q", struct.pack(">d", 2.5))[0]
    bits_d = struct.unpack(">Q", struct.pack(">d", 4.5))[0]
    b_bytes = tuple(bits_b.to_bytes(8, byteorder="big", signed=False))
    d_bytes = tuple(bits_d.to_bytes(8, byteorder="big", signed=False))
    vc_value = 9
    raw = _u16(
        404, *b_bytes,    # FLOAT64
        400, vc_value,    # VALUED_CONST_V2
        404, *d_bytes,    # FLOAT64
    )
    out = _decode(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(404, 400, 404))
    expected = (
        from_float64(bits_b)
        + from_int(vc_value, sign=+1)
        + from_float64(bits_d)
    )
    assert out.numbers_significant.shape == (3,)
    for idx, (sig, sign_exp) in enumerate(expected):
        assert int(out.numbers_significant[idx]) == int(sig)
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp)


# ---------------------------------------------------------------------------
# Working-buffer isolation
# ---------------------------------------------------------------------------


def test_caller_raw_tokens_unchanged_after_decode():
    """Promotion writes into a working copy; caller's array must not be
    mutated (the splicer holds the raw stream across multiple decode
    calls and depends on this)."""
    value = (1 << 100) + 7  # forces multi-chunk promotion
    payload_bytes = value.to_bytes(
        (value.bit_length() + 7) // 8, byteorder="big", signed=False
    )
    raw = _u16(400, *payload_bytes)
    snapshot = raw.copy()
    _decode(raw)
    np.testing.assert_array_equal(raw, snapshot)


def test_interleaved_multi_chunk_sources_preserve_per_source_chunk_contiguity():
    """A stream with [VALUED_CONST_V2 u128, FLOAT64, VALUED_CONST_V2 u128] must:

    - emit 5 number-side-array entries (2 + 1 + 2),
    - in stream-position order (NOT clustered: f64 entry between the two
      u128 chunk groups, not after both),
    - with each u128's two chunks contiguous (NO interleaving WITHIN one
      source).

    Locks the invariant that the splicer relies on: side-array index
    range ``[k, k + chunk_count_k)`` belongs to exactly one source.
    """
    # Distinct u128 values: every chunk (low + high) differs between
    # the two sources so any cross-source mixup would be detectable
    # via a mismatched chunk pair.
    u128_a = 0x0123456789ABCDEF_FEDCBA9876543210
    u128_b = 0xDEADBEEFCAFEBABE_1234567890ABCDEF
    a_bytes = u128_a.to_bytes(16, byteorder="big", signed=False)
    b_bytes = u128_b.to_bytes(16, byteorder="big", signed=False)
    # A recognisable f64 value (1.0) between them.
    f64_bits = struct.unpack(">Q", struct.pack(">d", 1.0))[0]
    f64_bytes = f64_bits.to_bytes(8, byteorder="big", signed=False)
    raw = _u16(
        400, *a_bytes,    # VALUED_CONST_V2 = u128_a -> 2 chunks
        404, *f64_bytes,  # FLOAT64 = 1.0 -> 1 chunk
        400, *b_bytes,    # VALUED_CONST_V2 = u128_b -> 2 chunks
    )
    out = _decode(raw)

    # Five side-array entries: 2 + 1 + 2.
    assert out.numbers_significant.shape == (5,)
    assert out.numbers_sign_exponent.shape == (5,)

    # Real-token stream after strip-and-promote: u128_a paints two
    # VALUED_CONST_V2 slots; FLOAT64 is single-token; u128_b paints two
    # VALUED_CONST_V2 slots.  Order: [VC, VC, F64, VC, VC].
    np.testing.assert_array_equal(
        out.real_tokens, _u16(400, 400, 404, 400, 400)
    )

    # Reconstruct each source from its chunk range and compare against
    # the canonical encoder output.  This catches BOTH cross-source
    # mixup (a's chunks mixed with b's) AND intra-source interleaving
    # (a's high chunk swapped with the f64 chunk, etc.).
    expected_a = from_int(u128_a, sign=+1)
    expected_f64 = from_float64(f64_bits)
    expected_b = from_int(u128_b, sign=+1)
    assert len(expected_a) == 2
    assert len(expected_f64) == 1
    assert len(expected_b) == 2

    expected = expected_a + expected_f64 + expected_b
    for idx, (sig, sign_exp) in enumerate(expected):
        assert int(out.numbers_significant[idx]) == int(sig), (
            f"significand mismatch at side-array index {idx}: "
            f"expected {int(sig):#x}, got {int(out.numbers_significant[idx]):#x}"
        )
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp), (
            f"sign_exp mismatch at side-array index {idx}: "
            f"expected {int(sign_exp):#x}, got {int(out.numbers_sign_exponent[idx]):#x}"
        )


# ---------------------------------------------------------------------------
# Partial vocab: missing TokenTypes are silently absent from the id maps;
# the decoder must still return a DecodedFunction with all 8 Category keys.
# ---------------------------------------------------------------------------


def test_partial_vocab_decode_succeeds_with_empty_identity_arrays_for_missing_types():
    """A vocab that only exposes LOCAL_FUNC (identity) + FLOAT64 (number)
    yields ``id_token_ids`` / ``number_token_ids`` dicts smaller than the
    canonical 8 / 7.  ``decode_raw_tokens`` must:

    - keep all 8 ``Category`` keys in ``identities`` (the
      ``DecodedFunction.__post_init__`` invariant);
    - populate LOCAL_FUNC with the decoded values;
    - leave the other 7 Category arrays empty (length 0);
    - emit number side-array entries ONLY for FLOAT64 (the only
      number-token type the partial vocab advertises).

    Crucially the stream contains a real-token id (303) that the partial
    vocab does NOT recognise.  Because nothing in the partial map points
    at 303, that token is preserved verbatim in the real-token stream
    but never appears in any identity / number side array — vocab
    incompleteness leaves stream content untouched, only the side-
    channel decoding is partial.
    """
    # Build the partial id maps directly — bypassing the resolver — so
    # this test isolates ``decode_raw_tokens``'s handling of partial maps
    # from the resolver's "absent TokenType -> omitted key" behaviour
    # (which is covered in test_category_tokens.py).
    id_token_ids = {Category.LOCAL_FUNC: 301}
    number_token_ids = {TokenType.FLOAT64: 404}

    # Stream: LOCAL_FUNC(5), <unknown real-token 303>, FLOAT64(2.5), FLOAT64(4.5).
    bits_a = struct.unpack(">Q", struct.pack(">d", 2.5))[0]
    bits_b = struct.unpack(">Q", struct.pack(">d", 4.5))[0]
    a_bytes = tuple(bits_a.to_bytes(8, byteorder="big", signed=False))
    b_bytes = tuple(bits_b.to_bytes(8, byteorder="big", signed=False))
    raw = _u16(
        301, 5,               # LOCAL_FUNC = 5
        303,                  # unknown real-token (id 303 — partial vocab ignores it)
        404, *a_bytes,        # FLOAT64 = 2.5
        404, *b_bytes,        # FLOAT64 = 4.5
    )

    out = decode_raw_tokens(
        raw,
        id_token_ids=id_token_ids,
        number_token_ids=number_token_ids,
        func_name="t",
    )

    # All 8 Category keys are present (DecodedFunction invariant).
    assert set(out.identities.keys()) == set(Category)

    # LOCAL_FUNC carries the decoded value.
    np.testing.assert_array_equal(
        out.identities[Category.LOCAL_FUNC], _u16(5)
    )

    # Every other Category is empty.
    for category in Category:
        if category is Category.LOCAL_FUNC:
            continue
        assert out.identities[category].shape == (0,), (
            f"{category.name} should be empty but has shape "
            f"{out.identities[category].shape}"
        )

    # FLOAT64 side-array entries: one per occurrence.
    expected_numbers = from_float64(bits_a) + from_float64(bits_b)
    assert out.numbers_significant.shape == (len(expected_numbers),)
    assert out.numbers_sign_exponent.shape == (len(expected_numbers),)
    for idx, (sig, sign_exp) in enumerate(expected_numbers):
        assert int(out.numbers_significant[idx]) == int(sig)
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp)

    # Real-token stream preserves the unknown id 303 verbatim alongside
    # LOCAL_FUNC + the two FLOAT64 occurrences.
    np.testing.assert_array_equal(out.real_tokens, _u16(301, 303, 404, 404))
