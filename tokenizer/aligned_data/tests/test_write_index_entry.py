"""Tests for ``tokenizer.aligned_data._writers.write_index_entry``.

Each entry is one ``u32 = offset >> ALIGNMENT_SHIFT`` (4 bytes total).
Records are self-describing in ``_data.bin`` so no length, no avg_len,
no overlong sentinel. The writer is a thin wrapper over
:func:`tokenizer.aligned_data.index_format.pack_index_entry` and shares
its cap (~64 GiB per binary at the current 16-byte record alignment)
and its alignment invariant (offsets must be 16-byte aligned).

Cap violations raise :class:`IndexEntrySkip`; when an ``error_log`` is
supplied the exception is logged and the entry is skipped. Alignment
violations are programmer errors and raise :class:`AssertionError`
without logging (same policy as the data-record writer).
"""

from __future__ import annotations

import io
import struct

import pytest

from tokenizer.aligned_data.binary_format import IndexEntrySkip, RECORD_ALIGNMENT
from tokenizer.aligned_data.index_format import (
    ALIGNMENT_SHIFT,
    INDEX_ENTRY_SIZE,
)
from tokenizer.aligned_data._writers import write_index_entry

# Convenience cap mirroring the codec's private constant. Kept here so
# test failures point at intent ("at the 64 GiB boundary") not at raw
# hex; pinning to the same arithmetic also makes the test independent
# of how the codec spells its private constant.
MAX_OFFSET = ((1 << 32) - 1) << ALIGNMENT_SHIFT  # ~64 GiB


def _unpack_entry(raw: bytes) -> int:
    """Decode the 4-byte entry to its stored ``offset_shifted`` field."""
    assert len(raw) == INDEX_ENTRY_SIZE
    (offset_shifted,) = struct.unpack("<I", raw)
    return offset_shifted


# ---------------------------------------------------------------------------
# Normal round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [0, RECORD_ALIGNMENT, RECORD_ALIGNMENT * 2, 1024, MAX_OFFSET],
)
def test_round_trip(offset: int):
    """Every aligned offset packs back to its ``offset >> ALIGNMENT_SHIFT``."""
    buf = io.BytesIO()
    write_index_entry(buf, offset)
    offset_shifted = _unpack_entry(buf.getvalue())
    assert offset_shifted == offset >> ALIGNMENT_SHIFT


def test_writer_advances_exactly_four_bytes():
    """Pin the on-wire entry width so a future "optimisation" can't
    silently regress it."""
    buf = io.BytesIO()
    before = buf.tell()
    write_index_entry(buf, RECORD_ALIGNMENT)
    assert buf.tell() - before == INDEX_ENTRY_SIZE
    assert INDEX_ENTRY_SIZE == 4


# ---------------------------------------------------------------------------
# Cap violations — propagation vs logging
# ---------------------------------------------------------------------------


def test_offset_cap_raise_without_error_log():
    """One ``RECORD_ALIGNMENT`` step past the cap raises
    :class:`IndexEntrySkip` when no log handle is supplied."""
    over_cap = MAX_OFFSET + RECORD_ALIGNMENT
    buf = io.BytesIO()
    with pytest.raises(IndexEntrySkip) as info:
        write_index_entry(buf, over_cap)
    assert info.value.reason == "offset_overflow"
    assert info.value.value == over_cap
    # No partial write: the packer raises BEFORE writing.
    assert buf.tell() == 0


def test_offset_cap_logs_when_error_log_supplied():
    """With an ``error_log`` handle, the exception is logged and the
    writer returns ``None`` without advancing the index file."""
    over_cap = MAX_OFFSET + RECORD_ALIGNMENT
    buf = io.BytesIO()
    log = io.StringIO()
    result = write_index_entry(
        buf, over_cap, func_name="bigfn", error_log=log,
    )
    assert result is None
    assert buf.tell() == 0

    log_text = log.getvalue()
    assert log_text.count("\n") == 1
    fields = log_text.rstrip("\n").split("\t")
    assert fields[0] == "offset_overflow"
    assert fields[1] == "bigfn"
    assert int(fields[2]) == over_cap


# ---------------------------------------------------------------------------
# Programmer-error assertions (never logged)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_offset", [1, 3, 5, 7, 15])
def test_unaligned_offset_asserts(bad_offset: int):
    """Misaligned offsets reveal a writer bug; raise unconditionally."""
    buf = io.BytesIO()
    with pytest.raises(AssertionError):
        write_index_entry(buf, bad_offset)


def test_alignment_error_not_logged_even_with_error_log():
    """Programmer errors must bubble up even when an ``error_log`` is
    available -- they're never appropriate to log + skip silently."""
    buf = io.BytesIO()
    log = io.StringIO()
    with pytest.raises(AssertionError):
        write_index_entry(buf, 5, func_name="bug", error_log=log)
    assert log.getvalue() == ""


# ---------------------------------------------------------------------------
# Bit-exact byte layout
# ---------------------------------------------------------------------------


def test_bit_exact_byte_layout():
    """Pin a known offset to its on-wire bytes so a structural mistake
    (endianness, slice off-by-one) is caught immediately.

    ``offset=16`` → ``offset_shifted=1`` → ``\\x01\\x00\\x00\\x00``
    """
    buf = io.BytesIO()
    write_index_entry(buf, RECORD_ALIGNMENT)
    assert buf.getvalue() == b"\x01\x00\x00\x00"
