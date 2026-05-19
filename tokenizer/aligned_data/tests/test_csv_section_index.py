"""Tests for the pre-v1 ``matched_index.bin`` codec.

Covers the on-disk byte layout (round-trip), the deliberate absence of
4-byte alignment on ``csv_offset`` (the whole point of the pre-v1
layout: text-file byte positions are not 4-aligned), empty + truncated
+ absent file handling, encode-time caps, and parity between the
vectorised reader and the streaming iterator.
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
    MAX_CSV_LENGTH,
    MAX_CSV_OFFSET,
    iter_csv_section_index_entries,
    pack_csv_section_index_entry,
    read_csv_section_index_arrays,
    write_csv_section_index_entry,
)


def _expected_avg_byte(avg_len: int) -> int:
    """Mirror the writer's clamp so test expectations stay in one place."""
    return min(avg_len >> 4, 255)


def _write_entries(path: Path, entries):
    with open(path, "wb") as fh:
        for csv_offset, csv_len, avg_len in entries:
            write_csv_section_index_entry(fh, csv_offset, csv_len, avg_len)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_100_random_entries(tmp_path):
    """100 random entries pack + read back column-by-column unchanged."""
    rng = random.Random(0xC0FFEE)
    entries = []
    for _ in range(100):
        csv_offset = rng.randint(0, MAX_CSV_OFFSET)
        csv_len = rng.randint(1, MAX_CSV_LENGTH)
        avg_len = rng.randint(0, 4096)
        entries.append((csv_offset, csv_len, avg_len))

    path = tmp_path / "matched_index.bin"
    _write_entries(path, entries)

    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    csv_starts, csv_lengths, avg_lengths = arrays

    expected_starts = np.array([e[0] for e in entries], dtype=np.int64)
    expected_lengths = np.array([e[1] for e in entries], dtype=np.uint32)
    expected_avgs = np.array([_expected_avg_byte(e[2]) for e in entries], dtype=np.uint8)

    np.testing.assert_array_equal(csv_starts, expected_starts)
    np.testing.assert_array_equal(csv_lengths, expected_lengths)
    np.testing.assert_array_equal(avg_lengths, expected_avgs)


def test_writer_advances_exactly_eight_bytes():
    """Each entry must consume exactly :data:`ENTRY_SIZE` bytes on the wire."""
    buf = io.BytesIO()
    before = buf.tell()
    write_csv_section_index_entry(buf, 0, 1, 0)
    assert buf.tell() - before == ENTRY_SIZE


def test_bit_exact_byte_layout():
    """Pin a known triple to its on-wire bytes so structural mistakes
    (endianness, field order, u24 padding direction) are caught
    immediately.

    ``csv_offset=0x04030201`` → bytes 0-3 ``\\x01\\x02\\x03\\x04``
    ``csv_len=0x070605``      → bytes 4-6 ``\\x05\\x06\\x07``
    ``avg_len=32``            → clamp     ``\\x02``
    """
    buf = io.BytesIO()
    write_csv_section_index_entry(buf, 0x04030201, 0x00070605, 32)
    assert buf.getvalue() == b"\x01\x02\x03\x04\x05\x06\x07\x02"


# ---------------------------------------------------------------------------
# Pre-v1 layout: csv_offset NOT 4-aligned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("csv_offset", [0, 1, 2, 3, 17, 19, 4097, 0xFFFFFFFE, MAX_CSV_OFFSET])
def test_non_4_aligned_csv_offset_round_trips(tmp_path, csv_offset: int):
    """The whole point of the pre-v1 layout: text-file byte positions
    are NOT 4-aligned. Any u32 value must round-trip without complaint;
    a "let's reintroduce alignment" regression is caught here."""
    path = tmp_path / "matched_index.bin"
    _write_entries(path, [(csv_offset, 1, 0)])

    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    csv_starts, csv_lengths, _ = arrays
    assert csv_starts[0] == csv_offset
    assert csv_lengths[0] == 1


# ---------------------------------------------------------------------------
# Empty / missing / truncated file
# ---------------------------------------------------------------------------


def test_empty_file_returns_empty_arrays(tmp_path):
    """A zero-byte index file is well-formed: zero entries."""
    path = tmp_path / "matched_index.bin"
    path.write_bytes(b"")
    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    csv_starts, csv_lengths, avg_lengths = arrays
    assert csv_starts.shape == (0,)
    assert csv_lengths.shape == (0,)
    assert avg_lengths.shape == (0,)
    assert csv_starts.dtype == np.int64
    assert csv_lengths.dtype == np.uint32
    assert avg_lengths.dtype == np.uint8


def test_nonexistent_path_returns_none(tmp_path):
    """A missing index file means "no matched arm written yet"; the
    caller decides how to handle that — we don't crash."""
    assert read_csv_section_index_arrays(tmp_path / "does_not_exist.bin") is None


def test_truncated_file_raises(tmp_path):
    """A size that isn't a multiple of :data:`ENTRY_SIZE` indicates
    corruption; surface it loudly rather than guessing where the
    valid prefix ends."""
    path = tmp_path / "matched_index.bin"
    path.write_bytes(b"\x00" * (ENTRY_SIZE + 3))
    with pytest.raises(ValueError, match="not a multiple"):
        read_csv_section_index_arrays(path)


# ---------------------------------------------------------------------------
# Cap overflows
# ---------------------------------------------------------------------------


def test_pack_csv_offset_overflow_raises():
    """``csv_offset > MAX_CSV_OFFSET`` overflows the u32 field."""
    with pytest.raises(IndexEntrySkip) as info:
        pack_csv_section_index_entry(MAX_CSV_OFFSET + 1, 4, 0)
    assert info.value.reason == "csv_offset_overflow"
    assert info.value.value == MAX_CSV_OFFSET + 1


def test_pack_csv_length_overflow_raises():
    """``csv_len > MAX_CSV_LENGTH`` overflows the u24 field."""
    with pytest.raises(IndexEntrySkip) as info:
        pack_csv_section_index_entry(0, MAX_CSV_LENGTH + 1, 0)
    assert info.value.reason == "csv_length_overflow"
    assert info.value.value == MAX_CSV_LENGTH + 1


def test_write_with_error_log_swallows_skip():
    """An ``error_log`` handle turns overflow into a log line + skip;
    nothing is written to the index file."""
    buf = io.BytesIO()
    log = io.StringIO()
    write_csv_section_index_entry(
        buf, MAX_CSV_OFFSET + 1, 4, 0, func_name="bigfn", error_log=log
    )
    assert buf.tell() == 0
    fields = log.getvalue().rstrip("\n").split("\t")
    assert fields[0] == "csv_offset_overflow"
    assert fields[1] == "bigfn"
    assert int(fields[2]) == MAX_CSV_OFFSET + 1


def test_write_without_error_log_propagates_skip():
    """Without ``error_log`` the overflow propagates; nothing written."""
    buf = io.BytesIO()
    with pytest.raises(IndexEntrySkip):
        write_csv_section_index_entry(buf, 0, MAX_CSV_LENGTH + 1, 0)
    assert buf.tell() == 0


# ---------------------------------------------------------------------------
# Streaming iterator parity
# ---------------------------------------------------------------------------


def test_iter_matches_constructed_sequence(tmp_path):
    """The streaming iterator yields the exact triples that were
    written, in order, with ``avg_len`` reflecting the writer's clamp."""
    entries = [(0, 1, 0), (1, 17, 32), (2, 256, 4096), (MAX_CSV_OFFSET, MAX_CSV_LENGTH, 16)]
    path = tmp_path / "matched_index.bin"
    _write_entries(path, entries)

    yielded = list(iter_csv_section_index_entries(path))
    expected = [
        (csv_offset, csv_len, _expected_avg_byte(avg_len))
        for csv_offset, csv_len, avg_len in entries
    ]
    assert yielded == expected


def test_vectorised_read_matches_iter_for_1000_entries(tmp_path):
    """Vectorised column extraction must agree with the per-entry
    iterator for a 1000-entry file — catches any off-by-one or
    endianness regression in the vectorisation path."""
    rng = random.Random(0xBEEF)
    entries = [
        (
            rng.randint(0, MAX_CSV_OFFSET),
            rng.randint(1, MAX_CSV_LENGTH),
            rng.randint(0, 4096),
        )
        for _ in range(1000)
    ]
    path = tmp_path / "matched_index.bin"
    _write_entries(path, entries)

    arrays = read_csv_section_index_arrays(path)
    assert arrays is not None
    csv_starts, csv_lengths, avg_lengths = arrays

    iterated = list(iter_csv_section_index_entries(path))
    iter_starts = np.array([t[0] for t in iterated], dtype=np.int64)
    iter_lengths = np.array([t[1] for t in iterated], dtype=np.uint32)
    iter_avgs = np.array([t[2] for t in iterated], dtype=np.uint8)

    np.testing.assert_array_equal(csv_starts, iter_starts)
    np.testing.assert_array_equal(csv_lengths, iter_lengths)
    np.testing.assert_array_equal(avg_lengths, iter_avgs)


# ---------------------------------------------------------------------------
# Sanity: hand-rolled struct parity
# ---------------------------------------------------------------------------


def test_pack_matches_hand_rolled_struct():
    """One more belt-and-suspenders check: the packer agrees with a
    hand-rolled little-endian struct.pack sequence."""
    csv_offset = 0xDEADBEEF & 0xFFFFFFFF
    csv_len = 0x123456
    avg_len = 48
    raw = pack_csv_section_index_entry(csv_offset, csv_len, avg_len)
    expected = (
        struct.pack("<I", csv_offset)
        + struct.pack("<I", csv_len)[:3]
        + struct.pack("B", _expected_avg_byte(avg_len))
    )
    assert raw == expected
