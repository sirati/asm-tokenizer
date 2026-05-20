"""Tests for ``tokenizer.aligned_data.inline_indexer``.

The inline indexer encodes a single ``_data.bin`` offset as an 8-hex-
char string that lives inline in ``matched_sections.csv`` variant
rows -- no separate index file, no length, no overlong sentinel.
The codec is a thin wrapper over the layout single-source-of-truth in
``index_format`` (``pack_index_entry`` and ``decode_index_entry``);
these tests pin the round-trip contract, the hex-length invariant,
and the hard-cutover error contract for the legacy 16-char layout.
"""

from __future__ import annotations

import random

import pytest

from tokenizer.aligned_data.index_format import pack_index_entry
from tokenizer.aligned_data.inline_indexer import (
    decode_inline_indexer,
    encode_inline_indexer,
)

_ALIGNMENT_BYTES = 16  # records align to 16 bytes; offsets divisible by 16
_MAX_OFFSET_SHIFTED = (1 << 32) - 1
_MAX_OFFSET = _MAX_OFFSET_SHIFTED << 4


# ---------------------------------------------------------------------------
# Round-trip -- fuzz with 1000 random aligned offsets
# ---------------------------------------------------------------------------


def test_round_trip_random_offsets():
    """1000 random 16-byte-aligned offsets round-trip exactly."""
    rng = random.Random(20260520)
    for _ in range(1000):
        offset = rng.randint(0, _MAX_OFFSET_SHIFTED) << 4

        hex_str = encode_inline_indexer(offset)
        assert len(hex_str) == 8

        decoded = decode_inline_indexer(hex_str)
        assert decoded == offset


# ---------------------------------------------------------------------------
# Hex length invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [
        0,
        _ALIGNMENT_BYTES,
        1024,
        _ALIGNMENT_BYTES * 1234,
        _MAX_OFFSET,
    ],
)
def test_hex_length_always_eight(offset: int):
    """Every inline encoding is exactly 8 hex characters."""
    assert len(encode_inline_indexer(offset)) == 8


# ---------------------------------------------------------------------------
# Layering -- encode is byte-equivalent to pack_index_entry(offset)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [
        0,
        _ALIGNMENT_BYTES,
        4096,
        _ALIGNMENT_BYTES * 9999,
        _MAX_OFFSET,
    ],
)
def test_encode_is_pack_index_entry_hex(offset: int):
    """The encoder must be a thin wrapper over ``pack_index_entry(offset)``.

    Pins the single-source-of-truth invariant: layout knowledge lives
    in the packer, the codec only hex-wraps it.
    """
    assert bytes.fromhex(encode_inline_indexer(offset)) == pack_index_entry(offset)


# ---------------------------------------------------------------------------
# Hard-cutover error contract: legacy 16-char layout
# ---------------------------------------------------------------------------


def test_legacy_sixteen_char_raises_migration_error():
    """A 16-char input is the legacy v1-cleanup layout and must trip a
    migration-pointing ValueError."""
    legacy = "a" * 16
    with pytest.raises(ValueError, match="re-run memmap_builder"):
        decode_inline_indexer(legacy)


# ---------------------------------------------------------------------------
# Decode error contract: wrong-length / non-hex inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "xx", "abcd", "a" * 7, "a" * 9, "a" * 32])
def test_decode_wrong_length_raises_value_error(bad: str):
    """Anything other than an 8-character string (or the legacy 16) is a ValueError."""
    with pytest.raises(ValueError):
        decode_inline_indexer(bad)


def test_decode_non_hex_raises_value_error():
    """Non-hex characters trip ValueError even at the right length."""
    with pytest.raises(ValueError):
        decode_inline_indexer("ZZZZZZZZ")
