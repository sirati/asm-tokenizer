"""Tests for ``tokenizer.aligned_data.index_format.pack_index_entry``.

The packer is the pure-function single source of truth for the on-wire
4-byte ``_index.bin`` entry. Records in ``_data.bin`` are self-describing
(their header carries ``insn_len``, ``block_word_count``, ``token_count``)
so the index entry collapses to ``u32 offset_shifted``: 4 bytes, one
field, no length, no avg_len_bucket, no sentinel.

The downstream ``write_index_entry`` (in ``_writers.py``, owned by a
sibling subtask) is expected to be a thin file-writing wrapper over
``pack_index_entry``; that equivalence is pinned by its own dedicated
suite and is not re-covered here.
"""

from __future__ import annotations

import struct

import pytest

from tokenizer.aligned_data.binary_format import IndexEntrySkip
from tokenizer.aligned_data.index_format import (
    ALIGNMENT_SHIFT,
    INDEX_ENTRY_SIZE,
    decode_index_entry,
    pack_index_entry,
)

_ALIGN: int = 1 << ALIGNMENT_SHIFT  # 16
_MAX_OFFSET: int = ((1 << 32) - 1) << ALIGNMENT_SHIFT  # ~64 GiB cap


# ---------------------------------------------------------------------------
# Wire width + bit-exact byte layout for the simplest valid record.
# ---------------------------------------------------------------------------


def test_entry_width_is_four_bytes():
    """Pin the contract: every packed entry is exactly :data:`INDEX_ENTRY_SIZE`."""
    assert INDEX_ENTRY_SIZE == 4
    assert len(pack_index_entry(0)) == INDEX_ENTRY_SIZE
    assert len(pack_index_entry(_MAX_OFFSET)) == INDEX_ENTRY_SIZE


def test_minimum_valid_entry():
    """``offset=0`` → ``offset_shifted=0`` → four zero bytes."""
    assert pack_index_entry(0) == b"\x00\x00\x00\x00"


def test_bit_exact_layout_at_one_alignment_step():
    """``offset=16`` → ``offset_shifted=1`` → ``\\x01\\x00\\x00\\x00``."""
    assert pack_index_entry(_ALIGN) == b"\x01\x00\x00\x00"


def test_bit_exact_layout_at_high_offset():
    """High offset: ``offset_shifted`` lays out little-endian across 4 bytes."""
    # offset_shifted = 0xCAFEBABE
    offset = 0xCAFEBABE << ALIGNMENT_SHIFT
    assert pack_index_entry(offset) == b"\xbe\xba\xfe\xca"


# ---------------------------------------------------------------------------
# Cap-overflow IndexEntrySkip propagation.
# ---------------------------------------------------------------------------


def test_offset_cap_overflow_raises():
    """One alignment step past the u32 cap raises :class:`IndexEntrySkip`."""
    over_cap = _MAX_OFFSET + _ALIGN  # still aligned, but above the cap
    with pytest.raises(IndexEntrySkip) as info:
        pack_index_entry(over_cap)
    assert info.value.reason == "offset_overflow"
    assert info.value.value == over_cap


def test_offset_at_cap_does_not_raise():
    """The exact u32 cap is the LAST acceptable offset; no raise."""
    packed = pack_index_entry(_MAX_OFFSET)
    (offset_shifted,) = struct.unpack("<I", packed)
    assert offset_shifted == (1 << 32) - 1


# ---------------------------------------------------------------------------
# Alignment programmer-error assertions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("misaligned", [1, 2, 4, 8, 15, 17, 31])
def test_unaligned_offset_asserts(misaligned: int):
    """Any offset not a multiple of 16 reveals a writer bug; assert fires."""
    with pytest.raises(AssertionError):
        pack_index_entry(misaligned)


# ---------------------------------------------------------------------------
# Round-trip through decode_index_entry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [
        0,
        _ALIGN,
        _ALIGN * 2,
        1024,
        1 << 20,
        1 << 28,
        _MAX_OFFSET - _ALIGN,
        _MAX_OFFSET,
    ],
)
def test_pack_decode_round_trip(offset: int):
    """``decode_index_entry(pack_index_entry(o)) == o`` for every aligned offset."""
    packed = pack_index_entry(offset)
    assert decode_index_entry(packed) == offset
