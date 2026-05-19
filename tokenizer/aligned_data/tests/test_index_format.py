"""Unit tests for the ``_index.bin`` prelude wire format.

Covers round-trip, hard-cutover rejection (missing magic + wrong
format_version), truncation, the overlong-length sentinel constant, and the
``INDEX_HEADER_SIZE`` advertised width matching the writer's actual byte
count.
"""

from __future__ import annotations

import io
import struct

import pytest

from tokenizer.aligned_data.index_format import (
    ALIGNMENT_SHIFT,
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
    IndexPrelude,
    SENTINEL_LENGTH,
    read_index_prelude,
    write_index_prelude,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION


def test_prelude_round_trip():
    buf = io.BytesIO()
    write_index_prelude(buf)
    buf.seek(0)
    prelude = read_index_prelude(buf)
    assert prelude == IndexPrelude(
        format_version=MEMMAP_FORMAT_VERSION,
        alignment_shift=ALIGNMENT_SHIFT,
    )


def test_write_advances_exactly_header_size():
    buf = io.BytesIO()
    write_index_prelude(buf)
    assert buf.tell() == INDEX_HEADER_SIZE


def test_wrong_magic_raises_with_migration_hint():
    buf = io.BytesIO(b"XXXX" + struct.pack("<III", MEMMAP_FORMAT_VERSION, ALIGNMENT_SHIFT, 0))
    with pytest.raises(ValueError) as excinfo:
        read_index_prelude(buf)
    assert "re-run memmap_builder" in str(excinfo.value)


def test_wrong_format_version_raises_mentioning_version():
    bogus_version = MEMMAP_FORMAT_VERSION + 1
    buf = io.BytesIO(INDEX_MAGIC + struct.pack("<III", bogus_version, ALIGNMENT_SHIFT, 0))
    with pytest.raises(ValueError) as excinfo:
        read_index_prelude(buf)
    msg = str(excinfo.value)
    assert str(bogus_version) in msg
    assert str(MEMMAP_FORMAT_VERSION) in msg


def test_truncated_prelude_raises():
    buf = io.BytesIO(b"\x00" * (INDEX_HEADER_SIZE // 2))
    with pytest.raises(struct.error):
        read_index_prelude(buf)


def test_sentinel_constant_round_trips_through_u16():
    packed = struct.pack("<H", SENTINEL_LENGTH)
    (unpacked,) = struct.unpack("<H", packed)
    assert unpacked == SENTINEL_LENGTH
    assert SENTINEL_LENGTH == 0x0000


def test_reserved_field_is_not_checked():
    # Writer stamps reserved=0, but the reader must tolerate arbitrary values
    # so future bumps that repurpose the field stay additive.
    raw = INDEX_MAGIC + struct.pack(
        "<III", MEMMAP_FORMAT_VERSION, ALIGNMENT_SHIFT, 0xDEADBEEF
    )
    buf = io.BytesIO(raw)
    prelude = read_index_prelude(buf)
    assert prelude.format_version == MEMMAP_FORMAT_VERSION
    assert prelude.alignment_shift == ALIGNMENT_SHIFT
