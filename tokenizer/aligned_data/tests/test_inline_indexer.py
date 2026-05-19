"""Tests for ``tokenizer.aligned_data.inline_indexer``.

Inline indexer encodes a v1 entry as a 16-hex-char string that lives
inline in ``matched_sections.csv`` variant rows -- no separate index
file. The codec is a thin wrapper over the writer-side packer
(``pack_v1_entry``) and the reader-side decoder
(``index_format.decode_index_entry``); these tests pin the round-trip
contract and the wrong-input error contract.
"""

from __future__ import annotations

import random

import pytest

from tokenizer.aligned_data._writers import pack_v1_entry
from tokenizer.aligned_data.index_format import MAX_NORMAL_REAL_LENGTH
from tokenizer.aligned_data.inline_indexer import (
    decode_inline_indexer,
    encode_inline_indexer,
)

_MAX_OVERLONG_REAL_LENGTH = 0xFFFFFF << 2  # 67_108_860 bytes ≈ 64 MiB
_MAX_OFFSET_SHIFTED = (1 << 40) - 1
_MAX_OFFSET = _MAX_OFFSET_SHIFTED << 2


# ---------------------------------------------------------------------------
# Round-trip — fuzz with 1000 random aligned pairs
# ---------------------------------------------------------------------------


def test_round_trip_random_pairs():
    """1000 random ``(offset, length)`` round-trips.

    For normal records (``length <= MAX_NORMAL_REAL_LENGTH``) the decoded
    length equals the input and ``is_overlong`` is ``False``. For
    overlong records the decoded length is ``0`` (sentinel) and
    ``is_overlong`` is ``True`` -- the real length lives in the data
    record's overlong field, which is not part of the inline encoding.
    """
    rng = random.Random(20260519)
    for _ in range(1000):
        offset = rng.randint(0, _MAX_OFFSET_SHIFTED) << 2
        length = rng.randint(1, _MAX_OVERLONG_REAL_LENGTH >> 2) << 2

        hex_str = encode_inline_indexer(offset, length)
        assert len(hex_str) == 16

        start, decoded_length, is_overlong = decode_inline_indexer(hex_str)
        assert start == offset

        if length <= MAX_NORMAL_REAL_LENGTH:
            assert not is_overlong
            assert decoded_length == length
        else:
            assert is_overlong
            assert decoded_length == 0


# ---------------------------------------------------------------------------
# Hex length invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset, length",
    [
        (0, 4),
        (4, 8),
        (1024, 65536),
        (_MAX_OFFSET, 4),
        (0, MAX_NORMAL_REAL_LENGTH),
        (0, MAX_NORMAL_REAL_LENGTH + 4),
        (0, _MAX_OVERLONG_REAL_LENGTH),
    ],
)
def test_hex_length_always_sixteen(offset: int, length: int):
    """Every inline encoding is exactly 16 hex characters."""
    assert len(encode_inline_indexer(offset, length)) == 16


# ---------------------------------------------------------------------------
# Layering — encode is byte-equivalent to pack_v1_entry(..., avg=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset, length",
    [
        (0, 4),
        (12, 100),
        (4096, MAX_NORMAL_REAL_LENGTH),
        (8, MAX_NORMAL_REAL_LENGTH + 4),
        (0, _MAX_OVERLONG_REAL_LENGTH),
    ],
)
def test_encode_is_pack_v1_entry_hex(offset: int, length: int):
    """The encoder must be a thin wrapper over ``pack_v1_entry(offset, length, 0)``.

    Pins the single-source-of-truth invariant: layout knowledge lives
    in the packer, the codec only hex-wraps it.
    """
    assert bytes.fromhex(encode_inline_indexer(offset, length)) == pack_v1_entry(
        offset, length, 0
    )


# ---------------------------------------------------------------------------
# Reserved trailing byte is zero
# ---------------------------------------------------------------------------


def test_reserved_byte_is_zero():
    """Inline encoding writes 0 in the reserved u8 byte (per design)."""
    encoded = bytes.fromhex(encode_inline_indexer(0, 4))
    assert encoded[7] == 0


# ---------------------------------------------------------------------------
# Decode error contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "xx", "abcd", "a" * 15, "a" * 17, "a" * 32])
def test_decode_wrong_length_raises_value_error(bad: str):
    """Anything other than a 16-character string is a ValueError."""
    with pytest.raises(ValueError):
        decode_inline_indexer(bad)


def test_decode_non_hex_raises_value_error():
    """Non-hex characters trip ValueError even at the right length."""
    with pytest.raises(ValueError):
        decode_inline_indexer("ZZZZZZZZZZZZZZZZ")
