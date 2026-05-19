"""Tests for ``tokenizer.aligned_data.io.write_index_entry``.

Covers the v1 index-entry packing — u40 ``offset_shifted`` plus u16
``length_shifted`` plus u8 ``avg_len`` — and the sentinel transition at
the 256 KiB / 64 MiB boundaries. Cap violations raise
:class:`IndexEntrySkip`; when an ``error_log`` is supplied the
exception is logged and the entry is skipped. Alignment / zero-length
violations are programmer errors and raise :class:`AssertionError`
without logging.
"""

from __future__ import annotations

import io
import struct

import pytest

from tokenizer.aligned_data.binary_format import IndexEntrySkip
from tokenizer.aligned_data.index_format import SENTINEL_LENGTH
from tokenizer.aligned_data.io import write_index_entry

# Convenience caps mirroring the writer's private constants. Kept here
# so test failures point at intent ("at the 256 KiB boundary") not at
# raw hex.
MAX_NORMAL_LENGTH = 0xFFFF << 2  # 262_140 bytes ≈ 256 KiB
OVERLONG_CAP = 0xFFFFFF << 2     # 67_108_860 bytes ≈ 64 MiB
MAX_OFFSET = ((1 << 40) - 1) << 2  # ~4 TiB


def _unpack_entry(raw: bytes) -> tuple[int, int, int]:
    """Decode the 8-byte entry into ``(offset_shifted, length_field, avg_len)``.

    The u40 is read by re-padding the low-5-byte slice to a full u64.
    """
    assert len(raw) == 8
    offset_padded = raw[:5] + b"\x00\x00\x00"
    offset_shifted = struct.unpack("<Q", offset_padded)[0]
    length_field = struct.unpack("<H", raw[5:7])[0]
    avg_len = raw[7]
    return offset_shifted, length_field, avg_len


# ---------------------------------------------------------------------------
# Normal round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0, 4, 8, 1024])
@pytest.mark.parametrize("length", [8, 64, 100_000, MAX_NORMAL_LENGTH])
@pytest.mark.parametrize("avg_len", [0, 16, 4096])
def test_normal_round_trip(start: int, length: int, avg_len: int):
    """Every combination in the normal regime packs back to its inputs."""
    buf = io.BytesIO()
    write_index_entry(buf, start, length, avg_len)

    offset_shifted, length_field, avg_byte = _unpack_entry(buf.getvalue())

    assert offset_shifted == start >> 2
    assert length_field == length >> 2
    assert length_field != SENTINEL_LENGTH  # all picked values stay in normal regime
    assert avg_byte == min(avg_len >> 4, 255)


def test_writer_advances_exactly_eight_bytes():
    """The entry width is fixed at 8 bytes; the writer must not over- or
    under-write. The internal post-write assertion guards this in
    production; the test pins the externally-observable consequence so
    a future "optimisation" doesn't regress the on-wire size."""
    buf = io.BytesIO()
    before = buf.tell()
    write_index_entry(buf, 4, 8, 16)
    assert buf.tell() - before == 8


# ---------------------------------------------------------------------------
# Sentinel transitions
# ---------------------------------------------------------------------------


def test_max_normal_length_is_not_sentinel():
    """``length == 0xFFFF << 2`` is the LAST normal value: it packs into
    ``length_field = 0xFFFF`` which is distinct from the
    ``SENTINEL_LENGTH == 0x0000`` flag."""
    buf = io.BytesIO()
    write_index_entry(buf, 0, MAX_NORMAL_LENGTH, 0)
    _, length_field, _ = _unpack_entry(buf.getvalue())
    assert length_field == 0xFFFF
    assert length_field != SENTINEL_LENGTH


def test_first_overlong_length_triggers_sentinel():
    """The next 4-aligned step above ``0xFFFF << 2`` crosses into the
    overlong regime, stamping the sentinel in ``length_field``."""
    overlong_length = MAX_NORMAL_LENGTH + 4
    buf = io.BytesIO()
    write_index_entry(buf, 0, overlong_length, 0)
    _, length_field, _ = _unpack_entry(buf.getvalue())
    assert length_field == SENTINEL_LENGTH


def test_one_mib_length_is_overlong_sentinel():
    """A real length of 1 MiB sits well inside the overlong regime."""
    buf = io.BytesIO()
    write_index_entry(buf, 0, 1 << 20, 0)
    _, length_field, _ = _unpack_entry(buf.getvalue())
    assert length_field == SENTINEL_LENGTH


def test_overlong_at_cap_is_sentinel():
    """The exact 64 MiB-shifted cap is the LAST acceptable overlong
    length; it must encode as the sentinel without raising."""
    buf = io.BytesIO()
    write_index_entry(buf, 0, OVERLONG_CAP, 0)
    _, length_field, _ = _unpack_entry(buf.getvalue())
    assert length_field == SENTINEL_LENGTH


# ---------------------------------------------------------------------------
# Cap violations — propagation vs logging
# ---------------------------------------------------------------------------


def test_overlong_cap_raise_without_error_log():
    """One 4-byte step past the overlong cap raises
    :class:`IndexEntrySkip` when no log handle is supplied."""
    over_cap = OVERLONG_CAP + 4
    buf = io.BytesIO()
    with pytest.raises(IndexEntrySkip) as info:
        write_index_entry(buf, 0, over_cap, 0)
    assert info.value.reason == "overlong_length_overflow"
    assert info.value.value == over_cap
    # No partial write: the entry-encoding raises BEFORE writing.
    assert buf.tell() == 0


def test_overlong_cap_logs_when_error_log_supplied():
    """With an ``error_log`` handle, the exception is logged and the
    writer returns ``None`` without advancing the index file."""
    over_cap = OVERLONG_CAP + 4
    buf = io.BytesIO()
    log = io.StringIO()
    result = write_index_entry(
        buf, 0, over_cap, 0, func_name="myfunc", error_log=log,
    )
    assert result is None
    assert buf.tell() == 0  # no bytes appended

    log_text = log.getvalue()
    assert log_text.count("\n") == 1
    fields = log_text.rstrip("\n").split("\t")
    assert fields[0] == "overlong_length_overflow"
    assert fields[1] == "myfunc"
    assert int(fields[2]) == over_cap


def test_offset_cap_raise_without_error_log():
    """A start above ``(2**40 - 1) << 2`` overflows the u40 offset and
    raises :class:`IndexEntrySkip`."""
    over_offset = 1 << 44  # > MAX_OFFSET (~4 TiB), still 4-aligned
    assert over_offset > MAX_OFFSET
    buf = io.BytesIO()
    with pytest.raises(IndexEntrySkip) as info:
        write_index_entry(buf, over_offset, 8, 0)
    assert info.value.reason == "offset_overflow"
    assert info.value.value == over_offset
    assert buf.tell() == 0


def test_offset_cap_logs_when_error_log_supplied():
    """With ``error_log`` supplied, the offset-cap path logs and skips
    without writing."""
    over_offset = 1 << 44
    buf = io.BytesIO()
    log = io.StringIO()
    result = write_index_entry(
        buf, over_offset, 8, 0, func_name="bigfn", error_log=log,
    )
    assert result is None
    assert buf.tell() == 0

    fields = log.getvalue().rstrip("\n").split("\t")
    assert fields[0] == "offset_overflow"
    assert fields[1] == "bigfn"
    assert int(fields[2]) == over_offset


# ---------------------------------------------------------------------------
# Programmer-error assertions (never logged)
# ---------------------------------------------------------------------------


def test_unaligned_start_asserts():
    """Misaligned offsets reveal a writer bug; raise unconditionally."""
    buf = io.BytesIO()
    with pytest.raises(AssertionError):
        write_index_entry(buf, 5, 8, 0)


def test_unaligned_length_asserts():
    """Misaligned lengths reveal a writer bug; raise unconditionally."""
    buf = io.BytesIO()
    with pytest.raises(AssertionError):
        write_index_entry(buf, 0, 7, 0)


def test_zero_length_asserts():
    """A real length of 0 is impossible: the smallest padded record is
    8 bytes. Treat as a programmer error."""
    buf = io.BytesIO()
    with pytest.raises(AssertionError):
        write_index_entry(buf, 0, 0, 0)


def test_alignment_error_not_logged_even_with_error_log():
    """Programmer errors must bubble up even when an ``error_log`` is
    available — they're never appropriate to log + skip silently."""
    buf = io.BytesIO()
    log = io.StringIO()
    with pytest.raises(AssertionError):
        write_index_entry(buf, 5, 8, 0, func_name="bug", error_log=log)
    assert log.getvalue() == ""


# ---------------------------------------------------------------------------
# Bit-exact byte layout
# ---------------------------------------------------------------------------


def test_bit_exact_byte_layout():
    """Pin a known triple to its on-wire bytes so a structural mistake
    (endianness, field order, slice off-by-one) is caught immediately.

    ``start=4``  → ``offset_shifted=1``  → ``\\x01\\x00\\x00\\x00\\x00``
    ``length=8`` → ``length_field=2``    → ``\\x02\\x00``
    ``avg=32``   → ``avg_clamped=2``     → ``\\x02``
    """
    buf = io.BytesIO()
    write_index_entry(buf, 4, 8, 32)
    assert buf.getvalue() == b"\x01\x00\x00\x00\x00\x02\x00\x02"
