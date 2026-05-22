"""Binary serialization of the ``_index.bin`` 16-byte prelude + 4-byte entries.

Single concern: the on-wire format of the per-binary ``_index.bin`` file --
the 16-byte ``IDX1`` prelude plus the homogeneous stream of fixed-width
entries that follows. Records are self-describing in ``_data.bin`` (the
record header carries ``insn_len``, ``block_word_count``, ``token_count``)
so the index entry is now just an offset locator: 4 bytes per entry,
``u32 offset_shifted`` with ``real_offset = stored << ALIGNMENT_SHIFT``.

Layout (little-endian):

    byte 0-3   magic              b"IDX1"
    byte 4-7   u32  format_version  (== MEMMAP_FORMAT_VERSION)
    byte 8-11  u32  alignment_shift (records aligned to ``1 << shift`` bytes)
    byte 12-15 u32  reserved        (write 0; readers ignore)

    byte 16+N*4    u32  offset_shifted   (real_offset = stored << shift)

The reader returns ``format_version`` and ``alignment_shift`` parsed from
the file so future bumps stay additive; callers consult the parsed shift,
not the ``ALIGNMENT_SHIFT`` constant exported here (that constant is the
value the current writer stamps, nothing more).

There is no sentinel and no overlong escape -- ``_data.bin`` records carry
their own geometry in the record header. The maximum addressable offset is
``((1 << 32) - 1) << ALIGNMENT_SHIFT`` -- ~64 GiB per binary at the
current 16-byte record alignment.

The codec (``pack_index_entry`` / ``decode_index_entry``) lives here so the
inline-indexer in ``matched_sections.csv`` and the on-disk ``_index.bin``
share one source of truth for the byte layout. Encoders in other modules
(``_writers.py``, ``inline_indexer.py``) wrap these helpers; they do not
re-derive the layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import numpy as np

from tokenizer.aligned_data.binary_format import IndexEntrySkip
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

INDEX_MAGIC: bytes = b"IDX1"
INDEX_HEADER_SIZE: int = 16
INDEX_ENTRY_SIZE: int = 4
ALIGNMENT_SHIFT: int = 4

# Largest real offset the u32 ``offset_shifted`` field can carry.
# ``offset_shifted`` max is ``(1 << 32) - 1``; the alignment shift (<<4)
# lifts that to ``((1 << 32) - 1) << 4`` ≈ 64 GiB. Any larger offset
# raises :class:`IndexEntrySkip` from :func:`pack_index_entry`.
_MAX_OFFSET: int = ((1 << 32) - 1) << ALIGNMENT_SHIFT

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


def pack_index_entry(offset: int) -> bytes:
    """Pack one 4-byte ``_index.bin`` entry encoding ``offset >> ALIGNMENT_SHIFT``.

    Returns exactly 4 little-endian bytes. ``offset`` must be a multiple
    of ``1 << ALIGNMENT_SHIFT`` (16 bytes) -- a violation reveals a
    writer bug and asserts unconditionally.

    On cap overflow (``offset > _MAX_OFFSET``) raises
    :class:`IndexEntrySkip` with reason ``"offset_overflow"`` so the
    encode-time cap policy ("log + skip + continue") propagates uniformly
    through the writer layer.
    """
    assert offset % (1 << ALIGNMENT_SHIFT) == 0, (
        f"index entry offset {offset} must be aligned to {1 << ALIGNMENT_SHIFT} bytes"
    )
    if offset > _MAX_OFFSET:
        raise IndexEntrySkip("offset_overflow", offset)
    return struct.pack("<I", offset >> ALIGNMENT_SHIFT)


def decode_index_entry(entry_bytes) -> int:
    """Decode one 4-byte index entry to its real ``_data.bin`` byte offset.

    ``entry_bytes`` may be any 4-byte buffer (``bytes``/``bytearray``/
    ``memoryview``/numpy row). The decoder reads the stored
    ``offset_shifted`` and returns ``stored << ALIGNMENT_SHIFT`` -- the
    real byte offset into ``_data.bin``.
    """
    (offset_shifted,) = struct.unpack("<I", bytes(entry_bytes))
    return offset_shifted << ALIGNMENT_SHIFT


def read_index_arrays(index_path: Path) -> Optional[np.ndarray]:
    """Load a v1 ``_index.bin`` into a single ndarray of real byte offsets.

    Returns an ``int64`` ndarray of real offsets (``stored << shift``).
    Returns ``None`` when the file does not exist OR when it contains the
    16-byte prelude but no entries (the empty-corpus case the rest of the
    loader treats as "no functions on this arm" -- callers fan out on
    ``None`` rather than special-casing a zero-row array).

    Raises :class:`ValueError` with a migration-pointing message on:

    * missing / wrong-version prelude (delegated to
      :func:`read_index_prelude`),
    * file size smaller than the 16-byte prelude,
    * entry region size not a multiple of :data:`INDEX_ENTRY_SIZE` -- the
      canonical signature of a stale legacy 8-byte-stride file from the
      pre-iteration v1 layout. The message contains the phrase
      ``re-run memmap_builder`` so hard-cutover smokes can pin it.
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
            f"of {INDEX_ENTRY_SIZE} -- this is the legacy index entry stride; "
            f"re-run memmap_builder on the per-binary CSVs to regenerate"
        )
    n_entries = entries_size // INDEX_ENTRY_SIZE
    with open(index_path, "rb") as fh:
        prelude = read_index_prelude(fh)
    if n_entries == 0:
        return None
    shift = prelude.alignment_shift
    # ``offset=`` skips the prelude bytes natively; a uint32 view over the
    # entry region performs the column decode in numpy. The result is
    # promoted to int64 with a left shift to recover real byte offsets.
    index_memmap = np.memmap(
        index_path,
        dtype=np.uint32,
        mode="r",
        offset=INDEX_HEADER_SIZE,
        shape=(n_entries,),
    )
    try:
        offsets = index_memmap.astype(np.int64) << shift
    finally:
        # Drop the memmap deterministically: ``read_index_arrays`` is a
        # leaf reader, so the file mapping should not outlive this call
        # even if the consumer raises mid-iteration on the returned
        # array (``offsets`` already owns its bytes after the astype).
        del index_memmap
    return offsets


def iter_index_entries(index_path: Path) -> Iterator[int]:
    """Yield per-entry real ``_data.bin`` byte offsets from a v1 ``_index.bin``.

    Streaming convenience for readers that do not want to materialise the
    full offsets ndarray. Decodes via :func:`decode_index_entry` so the
    streaming + vectorised paths share one codec.
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
            (offset_shifted,) = struct.unpack("<I", raw)
            yield offset_shifted << shift
