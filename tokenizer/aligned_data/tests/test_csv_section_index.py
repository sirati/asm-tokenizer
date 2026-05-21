"""Tests for the ``<binary>_matched_index.bin`` codec.

Covers the 8-byte packed entry (u40 bin_offset>>2 + u24 bin_length>>2),
the 4-byte alignment assertion, empty + missing + truncated file
handling, cap-overflow propagation, and parity between the vectorised
reader and the streaming iterator.
"""

from __future__ import annotations

import io
import random
import struct
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import IndexEntrySkip
from tokenizer.aligned_data.csv_section_index import (
    ENTRY_SIZE,
    MAX_BIN_LENGTH,
    MAX_BIN_OFFSET,
    iter_csv_section_index_entries,
    pack_csv_section_index_entry,
    read_csv_section_index_arrays,
    unpack_csv_section_index_entry,
    write_csv_section_index_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aligned(rng: random.Random, upper_exclusive: int) -> int:
    """Random integer in ``[0, upper_exclusive)`` that is 4-byte aligned."""
    # ``upper_exclusive`` is already a multiple of 4 (the format caps).
    return (rng.randint(0, (upper_exclusive >> 2) - 1)) << 2


def _write_entries(path: Path, entries):
    with open(path, "wb") as fh:
        for bin_offset, bin_section_length in entries:
            write_csv_section_index_entry(fh, bin_offset, bin_section_length)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_pack_returns_eight_bytes():
    """Single entry is exactly :data:`ENTRY_SIZE` bytes on the wire."""
    raw = pack_csv_section_index_entry(0, 0)
    assert isinstance(raw, bytes)
    assert len(raw) == ENTRY_SIZE


def test_pack_unpack_round_trip_smallest():
    """The all-zeros entry round-trips through pack + unpack."""
    raw = pack_csv_section_index_entry(0, 0)
    assert unpack_csv_section_index_entry(raw) == (0, 0)


def test_pack_unpack_round_trip_largest():
    """The largest representable values (cap - 4) round-trip cleanly."""
    bin_offset = MAX_BIN_OFFSET - 4
    bin_length = MAX_BIN_LENGTH - 4
    raw = pack_csv_section_index_entry(bin_offset, bin_length)
    assert unpack_csv_section_index_entry(raw) == (bin_offset, bin_length)


def test_round_trip_1000_random_pairs():
    """1000 random (bin_offset, bin_length) pairs round-trip through
    pack + unpack without drift in either field."""
    rng = random.Random(0xC0FFEE)
    for _ in range(1000):
        bin_offset = _aligned(rng, MAX_BIN_OFFSET)
        bin_length = _aligned(rng, MAX_BIN_LENGTH)
        raw = pack_csv_section_index_entry(bin_offset, bin_length)
        assert len(raw) == ENTRY_SIZE
        assert unpack_csv_section_index_entry(raw) == (bin_offset, bin_length)


def test_bit_exact_byte_layout():
    """Pin a known pair to its on-wire bytes so any endianness or
    field-order regression is caught immediately.

    ``bin_offset = 0x1F`` → not valid (alignment); use multiples of 4.
    ``bin_offset = 0x10`` → stored 0x04 in low 40 bits.
    ``bin_length = 0x20`` → stored 0x08 shifted up by 40 bits.

    Combined u64 = (0x08 << 40) | 0x04 = 0x0000_0800_0000_0004.
    Little-endian bytes: ``\\x04\\x00\\x00\\x00\\x00\\x08\\x00\\x00``.
    """
    raw = pack_csv_section_index_entry(0x10, 0x20)
    assert raw == b"\x04\x00\x00\x00\x00\x08\x00\x00"


def test_pack_matches_hand_rolled_struct():
    """Belt-and-suspenders: packer agrees with a direct u64 LE pack."""
    bin_offset = 0x0123_4560
    bin_length = 0x0098_7654
    stored_offset = bin_offset >> 2
    stored_length = bin_length >> 2
    packed = stored_offset | (stored_length << 40)
    assert pack_csv_section_index_entry(bin_offset, bin_length) == struct.pack(
        "<Q", packed
    )


# ---------------------------------------------------------------------------
# Alignment assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bin_offset", [1, 2, 3, 5, 7, 0x1001])
def test_unaligned_bin_offset_raises(bin_offset: int):
    """Non-4-aligned ``bin_offset`` violates the format invariant."""
    with pytest.raises(AssertionError, match="bin_offset"):
        pack_csv_section_index_entry(bin_offset, 0)


@pytest.mark.parametrize("bin_length", [1, 2, 3, 5, 7, 0x1001])
def test_unaligned_bin_length_raises(bin_length: int):
    """Non-4-aligned ``bin_section_length`` violates the format invariant."""
    with pytest.raises(AssertionError, match="bin_section_length"):
        pack_csv_section_index_entry(0, bin_length)


# ---------------------------------------------------------------------------
# Cap overflows
# ---------------------------------------------------------------------------


def test_pack_bin_offset_overflow_raises():
    """``bin_offset >= MAX_BIN_OFFSET`` overflows the u40 stored field."""
    with pytest.raises(IndexEntrySkip) as info:
        pack_csv_section_index_entry(MAX_BIN_OFFSET, 4)
    assert info.value.reason == "bin_offset_overflow"
    assert info.value.value == MAX_BIN_OFFSET


def test_pack_bin_length_overflow_raises():
    """``bin_section_length >= MAX_BIN_LENGTH`` overflows the u24 stored field."""
    with pytest.raises(IndexEntrySkip) as info:
        pack_csv_section_index_entry(0, MAX_BIN_LENGTH)
    assert info.value.reason == "bin_length_overflow"
    assert info.value.value == MAX_BIN_LENGTH


def test_write_with_error_log_swallows_skip(tmp_path):
    """An ``error_log`` handle turns overflow into a log line + skip;
    nothing is written to the index file."""
    buf = io.BytesIO()
    log = io.StringIO()
    write_csv_section_index_entry(
        buf,
        MAX_BIN_OFFSET,
        4,
        func_name="bigfn",
        error_log=log,
    )
    assert buf.tell() == 0
    fields = log.getvalue().rstrip("\n").split("\t")
    assert fields[0] == "bin_offset_overflow"
    assert fields[1] == "bigfn"
    assert int(fields[2]) == MAX_BIN_OFFSET


def test_write_without_error_log_propagates_skip():
    """Without ``error_log`` the overflow propagates; nothing written."""
    buf = io.BytesIO()
    with pytest.raises(IndexEntrySkip):
        write_csv_section_index_entry(buf, 0, MAX_BIN_LENGTH)
    assert buf.tell() == 0


# ---------------------------------------------------------------------------
# Reader: empty / missing / truncated / stride mismatch
# ---------------------------------------------------------------------------


def test_empty_file_returns_empty_arrays(tmp_path):
    """A zero-byte index file is well-formed: zero entries."""
    path = tmp_path / "matched_index.bin"
    path.write_bytes(b"")
    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    bin_starts, bin_lengths = arrays
    assert bin_starts.shape == (0,)
    assert bin_lengths.shape == (0,)
    assert bin_starts.dtype == np.int64
    assert bin_lengths.dtype == np.uint32


def test_nonexistent_path_returns_none(tmp_path):
    """A missing index file means "no matched arm written yet"; the
    caller decides how to handle that — we don't crash."""
    assert read_csv_section_index_arrays(tmp_path / "does_not_exist.bin") is None


def test_size_not_multiple_of_eight_raises_with_migration_hint(tmp_path):
    """A size that isn't a multiple of :data:`ENTRY_SIZE` is a legacy
    stride mismatch — surface it loudly with a re-build pointer rather
    than guessing where the valid prefix ends."""
    path = tmp_path / "matched_index.bin"
    path.write_bytes(b"\x00" * (ENTRY_SIZE + 3))
    with pytest.raises(ValueError, match="re-run memmap_builder"):
        read_csv_section_index_arrays(path)


# ---------------------------------------------------------------------------
# Round-trip via tmp_path file
# ---------------------------------------------------------------------------


def test_read_csv_section_index_arrays_round_trip(tmp_path):
    """Write a batch of entries to a real file; the reader recovers
    both columns with the documented dtypes and real (not stored)
    values."""
    rng = random.Random(0xBEEF)
    entries = [
        (_aligned(rng, MAX_BIN_OFFSET), _aligned(rng, MAX_BIN_LENGTH))
        for _ in range(50)
    ]

    path = tmp_path / "matched_index.bin"
    _write_entries(path, entries)

    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    bin_starts, bin_lengths = arrays
    assert bin_starts.dtype == np.int64
    assert bin_lengths.dtype == np.uint32

    expected_starts = np.array([e[0] for e in entries], dtype=np.int64)
    expected_lengths = np.array([e[1] for e in entries], dtype=np.uint32)
    np.testing.assert_array_equal(bin_starts, expected_starts)
    np.testing.assert_array_equal(bin_lengths, expected_lengths)


# ---------------------------------------------------------------------------
# Streaming iterator parity
# ---------------------------------------------------------------------------


def test_iter_matches_constructed_sequence(tmp_path):
    """The streaming iterator yields the exact pairs that were written,
    in order."""
    entries = [
        (0, 4),
        (0x10, 0x100),
        (0x1_0000, 0x4000),
        (MAX_BIN_OFFSET - 4, MAX_BIN_LENGTH - 4),
    ]
    path = tmp_path / "matched_index.bin"
    _write_entries(path, entries)

    yielded = list(iter_csv_section_index_entries(path))
    assert yielded == entries


def test_vectorised_read_matches_iter_for_1000_entries(tmp_path):
    """Vectorised column extraction must agree with the per-entry
    iterator for a 1000-entry file — catches any off-by-one or
    endianness regression in the vectorisation path."""
    rng = random.Random(0xBADF00D)
    entries = [
        (_aligned(rng, MAX_BIN_OFFSET), _aligned(rng, MAX_BIN_LENGTH))
        for _ in range(1000)
    ]
    path = tmp_path / "matched_index.bin"
    _write_entries(path, entries)

    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    bin_starts, bin_lengths = arrays

    iterated = list(iter_csv_section_index_entries(path))
    iter_starts = np.array([t[0] for t in iterated], dtype=np.int64)
    iter_lengths = np.array([t[1] for t in iterated], dtype=np.uint32)

    np.testing.assert_array_equal(bin_starts, iter_starts)
    np.testing.assert_array_equal(bin_lengths, iter_lengths)


# ---------------------------------------------------------------------------
# Unpack input shape
# ---------------------------------------------------------------------------


def test_unpack_wrong_size_raises():
    """``unpack`` is strict about input width to catch caller bugs early."""
    with pytest.raises(ValueError, match="bytes"):
        unpack_csv_section_index_entry(b"\x00" * 7)
    with pytest.raises(ValueError, match="bytes"):
        unpack_csv_section_index_entry(b"\x00" * 9)
