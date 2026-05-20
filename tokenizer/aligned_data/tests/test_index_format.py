"""Unit tests for the ``_index.bin`` prelude + 4-byte entry stream.

Covers prelude round-trip, hard-cutover rejection (missing magic + wrong
format_version + legacy 8-byte entry stride), truncation, the
``INDEX_HEADER_SIZE``/``INDEX_ENTRY_SIZE`` advertised widths matching the
writer's actual byte count, and the vectorised :func:`read_index_arrays`
matching the per-entry :func:`iter_index_entries` decode across a
1000-entry corpus.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import IndexEntrySkip
from tokenizer.aligned_data.index_format import (
    ALIGNMENT_SHIFT,
    INDEX_ENTRY_SIZE,
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
    IndexPrelude,
    decode_index_entry,
    iter_index_entries,
    pack_index_entry,
    read_index_arrays,
    read_index_prelude,
    write_index_prelude,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_ALIGN: int = 1 << ALIGNMENT_SHIFT  # 16
_MAX_OFFSET: int = ((1 << 32) - 1) << ALIGNMENT_SHIFT  # ~64 GiB cap
_LEGACY_ENTRY_SIZE: int = 8  # the pre-iteration v1 entry stride


# ---------------------------------------------------------------------------
# Prelude round-trip + structural invariants.
# ---------------------------------------------------------------------------


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


def test_prelude_advertised_constants_match_iteration_layout():
    """Pin the post-iteration on-wire constants: 16-byte alignment, 4-byte entries."""
    assert ALIGNMENT_SHIFT == 4
    assert INDEX_ENTRY_SIZE == 4
    assert INDEX_HEADER_SIZE == 16


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


# ---------------------------------------------------------------------------
# 1000-entry round-trip through pack/read.
# ---------------------------------------------------------------------------


def _write_synthetic_index(path, offsets):
    """Write a v1 ``_index.bin`` containing the given offsets in order."""
    with open(path, "wb") as fh:
        write_index_prelude(fh)
        for offset in offsets:
            fh.write(pack_index_entry(offset))


def test_round_trip_1000_random_aligned_offsets(tmp_path):
    """1000 random aligned offsets in ``[0, _MAX_OFFSET)`` round-trip through
    ``pack_index_entry`` → file → ``read_index_arrays`` and through the
    streaming ``iter_index_entries`` path.

    Both paths must return identical int64 offsets bit-for-bit, and the
    returned ndarray dtype must match the documented contract.
    """
    # Deterministic seed for reproducibility.
    rng = np.random.default_rng(0xC0FFEE)
    # ``integers`` upper bound is exclusive; mask to alignment.
    raw = rng.integers(0, _MAX_OFFSET, size=1000, dtype=np.int64, endpoint=False)
    offsets = [int(o) & ~(_ALIGN - 1) for o in raw]

    path = tmp_path / "round_trip_index.bin"
    _write_synthetic_index(path, offsets)

    # Vectorised path.
    offsets_vec = read_index_arrays(path)
    assert offsets_vec is not None
    assert offsets_vec.dtype == np.int64
    assert list(offsets_vec) == offsets

    # Streaming path -- the reference implementation for the decode loop.
    offsets_ref = list(iter_index_entries(path))
    assert offsets_ref == offsets


def test_read_index_arrays_returns_single_ndarray(tmp_path):
    """The reader returns a SINGLE int64 ndarray (no tuple) per the
    self-describing-record design -- the length lives in the data record
    header, not the index entry."""
    path = tmp_path / "single_arr.bin"
    _write_synthetic_index(path, [_ALIGN, _ALIGN * 2, _ALIGN * 3])
    result = read_index_arrays(path)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert list(result) == [_ALIGN, _ALIGN * 2, _ALIGN * 3]


def test_read_index_arrays_missing_file_returns_none(tmp_path):
    assert read_index_arrays(tmp_path / "nope.bin") is None


def test_read_index_arrays_empty_entries_returns_none(tmp_path):
    """Prelude-only file (no entries) is treated as the empty-corpus case."""
    path = tmp_path / "empty.bin"
    with open(path, "wb") as fh:
        write_index_prelude(fh)
    assert read_index_arrays(path) is None


# ---------------------------------------------------------------------------
# Cap-overflow + alignment guards re-pinned through the round-trip path.
# ---------------------------------------------------------------------------


def test_pack_raises_index_entry_skip_above_cap():
    """An offset above the u32 cap raises :class:`IndexEntrySkip`; the
    encode-time cap policy ("log + skip + continue") propagates through
    the writer layer."""
    with pytest.raises(IndexEntrySkip) as info:
        pack_index_entry(_MAX_OFFSET + _ALIGN)
    assert info.value.reason == "offset_overflow"


def test_pack_raises_assertion_on_misalignment():
    """Misaligned offsets reveal a writer bug; assert fires unconditionally."""
    with pytest.raises(AssertionError):
        pack_index_entry(1)


# ---------------------------------------------------------------------------
# Hard-cutover smoke: legacy 8-byte entry stride raises with migration hint.
# ---------------------------------------------------------------------------


def test_entry_region_not_multiple_of_four_raises_with_migration_hint(tmp_path):
    """Hard-cutover migration smoke: any ``_index.bin`` whose entry region
    size is not a multiple of :data:`INDEX_ENTRY_SIZE` raises with the
    canonical migration hint. This is the canonical signature of a
    truncated / wrong-stride producer, including (some) stale legacy
    8-byte-stride files where the file was interrupted mid-entry.

    The plan's hard-cutover spec requires the phrase ``re-run
    memmap_builder`` to appear in the message so downstream smokes can
    pin the migration hint.
    """
    # The cleanest synthetic shape: prelude + 1 byte. Size-divisible
    # legacy files that happen to align to 4 are caught downstream by
    # the inline-indexer 8-vs-16 hex-length check (sibling subtask).
    path = tmp_path / "legacy_index.bin"
    with open(path, "wb") as fh:
        write_index_prelude(fh)
        fh.write(b"\x00")  # 1-byte partial entry -> stride check fires
    with pytest.raises(ValueError) as excinfo:
        read_index_arrays(path)
    assert "re-run memmap_builder" in str(excinfo.value)


def test_legacy_8_byte_stride_three_entries_is_misinterpreted(tmp_path):
    """Three 8-byte legacy entries → 24-byte region. ``24 % 4 == 0`` so
    the size-mismatch path does NOT fire -- the file is silently
    reinterpreted as six 4-byte entries. Document this trade-off:
    the index-format reader cannot distinguish a legacy 8-byte stride
    when the entry count makes the region a multiple of 4 (which is
    every legacy 8-byte stride, since ``8 * N`` is always ``% 4 == 0``).
    The legacy-vs-current discriminator therefore lands on the inline
    indexer's 16-vs-8 hex-length check (sibling subtask) and the
    record header parser in ``_data.bin``.
    """
    # We assert the design trade-off explicitly so a future hardening
    # pass that adds a real legacy-stride sniff (e.g. by reading the
    # first 8 bytes and noticing the high-order zeros pattern of a
    # legacy offset_shifted+length_shifted) has a failing test to
    # update.
    path = tmp_path / "legacy_3entry.bin"
    with open(path, "wb") as fh:
        write_index_prelude(fh)
        fh.write(b"\x00" * 24)  # 3 legacy 8-byte entries = 24 bytes
    # Reader sees 6 four-byte entries; does NOT raise on stride.
    offsets = read_index_arrays(path)
    assert offsets is not None
    assert len(offsets) == 6


# ---------------------------------------------------------------------------
# Truncated-entry guard inside the streaming reader.
# ---------------------------------------------------------------------------


def test_iter_index_entries_raises_on_truncated_tail(tmp_path):
    path = tmp_path / "truncated.bin"
    with open(path, "wb") as fh:
        write_index_prelude(fh)
        fh.write(pack_index_entry(_ALIGN))
        fh.write(b"\x00")  # 1-byte partial entry
    with pytest.raises(ValueError) as excinfo:
        list(iter_index_entries(path))
    assert "truncated" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# decode_index_entry handles diverse buffer-like inputs.
# ---------------------------------------------------------------------------


def test_decode_accepts_bytes_bytearray_memoryview():
    packed = pack_index_entry(_ALIGN * 7)
    assert decode_index_entry(packed) == _ALIGN * 7
    assert decode_index_entry(bytearray(packed)) == _ALIGN * 7
    assert decode_index_entry(memoryview(packed)) == _ALIGN * 7
