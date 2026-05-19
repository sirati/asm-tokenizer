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
from typing import BinaryIO

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

INDEX_MAGIC: bytes = b"IDX1"
INDEX_HEADER_SIZE: int = 16
ALIGNMENT_SHIFT: int = 2
SENTINEL_LENGTH: int = 0x0000

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
