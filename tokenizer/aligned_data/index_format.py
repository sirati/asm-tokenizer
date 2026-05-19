"""Binary serialization of the ``_index.bin`` 16-byte file-level prelude.

Single concern: the wire format of the prelude that precedes the index-entry
stream, plus the shared constants describing it (magic, header size, the
initial ``alignment_shift`` writers stamp, and the per-entry length sentinel
that flags an overlong record).

Layout (little-endian):

    byte 0-3   magic              b"IDX1"
    byte 4-7   u32  format_version  (== MEMMAP_FORMAT_VERSION)
    byte 8-11  u32  alignment_shift (records aligned to ``1 << shift`` bytes)
    byte 12-15 u32  reserved        (write 0; readers ignore)

The reader returns ``format_version`` and ``alignment_shift`` parsed from the
file so future bumps stay additive; callers consult the parsed shift, not the
``ALIGNMENT_SHIFT`` constant exported here (that constant is the value the
current writer stamps, nothing more).

``SENTINEL_LENGTH`` is the per-entry ``length_shifted`` value (u16) that flags
an overlong record whose real length lives in a u24 field immediately after
the per-record header inside ``_data.bin``. ``0x0000`` is a natural sentinel
because no real record can have length 0 — the minimum padded record is well
above zero bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

INDEX_MAGIC: bytes = b"IDX1"
INDEX_HEADER_SIZE: int = 16
INDEX_ENTRY_SIZE: int = 8
ALIGNMENT_SHIFT: int = 2
SENTINEL_LENGTH: int = 0x0000

# Largest real record length the u16 length_shifted field can carry without
# tripping the overlong sentinel. ``length_shifted`` max is 0xFFFF; the
# alignment shift (<<2) lifts that to 0xFFFF << 2 = 262140 bytes (~256 KiB).
# Records strictly above this switch to the overlong layout where the real
# length lives in the u24 overlong field of the data record. Single source
# of truth — writer (`_writers`), index-decoding helper (`_index_decoding`),
# and validator (`_v1_checks`) all import this constant.
MAX_NORMAL_REAL_LENGTH: int = 0xFFFF << ALIGNMENT_SHIFT

_PRELUDE_STRUCT = struct.Struct("<4sIII")
assert _PRELUDE_STRUCT.size == INDEX_HEADER_SIZE


@dataclass(frozen=True)
class IndexPrelude:
    """Parsed contents of the ``_index.bin`` 16-byte prelude.

    Magic and reserved fields are validation-only and not surfaced here.
    """

    format_version: int
    alignment_shift: int


def write_index_prelude(file_handle: BinaryIO) -> None:
    """Write the 16-byte ``_index.bin`` prelude at the current file position."""
    file_handle.write(
        _PRELUDE_STRUCT.pack(
            INDEX_MAGIC,
            MEMMAP_FORMAT_VERSION,
            ALIGNMENT_SHIFT,
            0,
        )
    )


def read_index_prelude(file_handle: BinaryIO) -> IndexPrelude:
    """Read and validate the 16-byte ``_index.bin`` prelude.

    Raises :class:`ValueError` with a migration-pointing message on missing
    magic or unsupported format_version. Reserved field is read but not
    checked.
    """
    raw = file_handle.read(INDEX_HEADER_SIZE)
    magic, format_version, alignment_shift, _reserved = _PRELUDE_STRUCT.unpack(raw)
    if magic != INDEX_MAGIC:
        raise ValueError(
            "_index.bin missing magic header; re-run memmap_builder on the per-binary CSVs to regenerate"
        )
    if format_version != MEMMAP_FORMAT_VERSION:
        raise ValueError(
            f"_index.bin format_version must be {MEMMAP_FORMAT_VERSION}; got {format_version}; re-run memmap_builder on the per-binary CSVs to regenerate"
        )
    return IndexPrelude(format_version=format_version, alignment_shift=alignment_shift)


def decode_index_entry(
    entry_bytes, shift: int
) -> Tuple[int, int, int, bool]:
    """Decode one 8-byte index entry, applying the alignment ``shift``.

    Returns ``(start, length, avg_len, is_overlong)``. ``start`` is the
    real byte offset in ``_data.bin`` (``offset_shifted << shift``).
    ``length`` is the real record byte length (``length_shifted <<
    shift``) for normal entries; for the sentinel entry it is ``0`` and
    ``is_overlong`` is ``True`` (the caller resolves the real length
    from the data record's overlong field).
    """
    offset_shifted = int.from_bytes(bytes(entry_bytes[0:5]), "little")
    length_shifted = int.from_bytes(bytes(entry_bytes[5:7]), "little")
    avg_len = int(entry_bytes[7])
    start = offset_shifted << shift
    if length_shifted == SENTINEL_LENGTH:
        return start, 0, avg_len, True
    return start, length_shifted << shift, avg_len, False


def read_index_arrays(
    index_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a v1 ``_index.bin`` into three column ndarrays.

    Returns ``(starts, lengths, avg_lengths)`` with dtypes
    ``int64``/``uint32``/``uint8``. ``starts`` are real byte offsets;
    ``lengths`` carry the real record length for normal entries and ``0``
    for sentinel (overlong) entries -- callers detect the marker and
    resolve the real length from the data record's overlong field.
    Returns ``None`` when the file does not exist. Raises
    :class:`ValueError` on missing prelude / wrong version / malformed
    entry region.
    """
    if not index_path.exists():
        return None
    filesize = index_path.stat().st_size
    if filesize < INDEX_HEADER_SIZE:
        raise ValueError(
            f"{index_path}: file size {filesize} smaller than the 16-byte "
            f"prelude; re-run memmap_builder on the per-binary CSVs to regenerate"
        )
    entries_size = filesize - INDEX_HEADER_SIZE
    if entries_size % INDEX_ENTRY_SIZE != 0:
        raise ValueError(
            f"{index_path}: entry region size {entries_size} is not a multiple "
            f"of {INDEX_ENTRY_SIZE}"
        )
    n_entries = entries_size // INDEX_ENTRY_SIZE
    with open(index_path, "rb") as fh:
        prelude = read_index_prelude(fh)
    shift = prelude.alignment_shift
    if n_entries == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint8),
        )
    # ``offset=`` skips the prelude bytes natively.
    index_memmap = np.memmap(
        index_path,
        dtype=np.uint8,
        mode="r",
        offset=INDEX_HEADER_SIZE,
        shape=(n_entries, INDEX_ENTRY_SIZE),
    )
    try:
        # Vectorised entry decode -- bit-for-bit equivalent to the
        # per-entry ``decode_index_entry`` loop, but built from column
        # slices over the ``(n_entries, INDEX_ENTRY_SIZE)`` matrix so the
        # work happens in numpy rather than Python. The pad-and-view
        # trick avoids per-row ``int.from_bytes`` calls.
        #
        # u40 offset_shifted (bytes 0-4): pad the row with 3 zero bytes
        # to land on 8 and view as little-endian u64.
        zero_pad_3 = np.zeros((n_entries, 3), dtype=np.uint8)
        offset_padded = np.concatenate(
            [index_memmap[:, 0:5], zero_pad_3], axis=1
        )
        offset_shifted = np.ascontiguousarray(offset_padded).view(
            np.uint64
        ).reshape(n_entries)
        starts = (offset_shifted.astype(np.int64) << shift)
        # u16 length_shifted (bytes 5-6): assemble little-endian by hand
        # from the two byte columns. Keeps the work in uint16 so the
        # sentinel comparison stays bit-exact.
        length_shifted = (
            index_memmap[:, 5].astype(np.uint16)
            | (index_memmap[:, 6].astype(np.uint16) << 8)
        )
        sentinel_mask = length_shifted == SENTINEL_LENGTH
        lengths = np.where(
            sentinel_mask,
            np.uint32(0),
            length_shifted.astype(np.uint32) << shift,
        ).astype(np.uint32)
        # u8 avg_len (byte 7): direct copy so the returned array owns
        # its buffer once the memmap is dropped on exit.
        avg_lengths = np.array(index_memmap[:, 7], dtype=np.uint8)
    finally:
        # Drop the memmap deterministically: ``read_index_arrays`` is a
        # leaf reader, so the file mapping should not outlive this call
        # even if the consumer raises mid-iteration on the returned
        # arrays (avg_lengths already owns its bytes; starts/lengths
        # were built from arithmetic on derived arrays).
        del index_memmap
    return starts, lengths, avg_lengths


def iter_index_entries(index_path: Path):
    """Yield ``(start, length, avg_len)`` per entry of a v1 ``_index.bin``.

    Convenience for streaming readers (the iterator does not materialise
    the column arrays). ``length`` is ``0`` for sentinel entries.
    """
    with open(index_path, "rb") as fh:
        prelude = read_index_prelude(fh)
        shift = prelude.alignment_shift
        while True:
            raw = fh.read(INDEX_ENTRY_SIZE)
            if not raw:
                return
            if len(raw) != INDEX_ENTRY_SIZE:
                raise ValueError(
                    f"{index_path}: truncated entry of {len(raw)} bytes; "
                    f"expected {INDEX_ENTRY_SIZE}"
                )
            start, length, avg_len, _ = decode_index_entry(raw, shift)
            yield start, length, avg_len
