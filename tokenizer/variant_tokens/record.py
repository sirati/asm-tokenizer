"""Byte-level handle I/O for ``<bin>_variants.bin`` records.

Single concern: bridge the encoder's ``np.ndarray[uint16]`` to / from
file handles. Callers own handle lifecycle — there is intentionally no
path-based ``open + write + close`` variant (the memmap-builder writes
many records into one handle; the dataloader memmaps once per
session).

Record layout (matches ``encoder.encode_record`` output)::

    +-----+-----+-----+-----+-----+-----+----- ... -----+
    | u16 | u16 | u16 | u16 | u16 | u16 ...              |
    | n   |arch | comp| cver| opt | metadata k/v tokens  |
    +-----+-----+-----+-----+-----+-----+----- ... -----+

Total bytes = ``2 + 2*n``. The leading u16 is both the record's size
header (how many bytes follow) and the token count, so the reader
does not need a side table to walk records sequentially.
"""

from __future__ import annotations

from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

from .encoder import encode_record

# Width (in bytes) of one wire-format token. Centralised so a future
# bin-layout bump that widens IDs (already gated by the format_version
# co-versioning rule in the plan) only edits this constant.
_U16_BYTES = 2


def write_record(handle: BinaryIO, version_info: Any, vocab: Any) -> int:
    """Encode and append one variant record to ``handle``.

    Returns the byte offset at which the record begins — section CSVs
    cite this offset (hex-formatted) as ``variant_ref``. ``handle``
    must be opened in binary write/append mode; callers are responsible
    for the open/close lifecycle.
    """
    offset = handle.tell()
    record = encode_record(version_info, vocab)
    # ``ndarray.tobytes()`` gives the platform-native byte order. The
    # whole pipeline (writer + reader + memmap consumer) runs on the
    # same machine class — every existing ``_data.bin`` site likewise
    # writes raw ndarray bytes without an endianness preamble — so we
    # match that convention rather than introduce a new one here.
    handle.write(record.tobytes())
    return offset


def read_record(memmap: npt.NDArray[np.uint8], offset: int) -> npt.NDArray[np.uint16]:
    """Slice one variant record from an open ``_variants.bin`` memmap.

    ``memmap`` is the caller's open ``np.memmap`` (or any uint8-array
    view); ``offset`` is the starting byte offset (as supplied by
    ``write_record`` and stored in the section CSV's ``variant_ref``).

    Reads the leading u16 size header to learn ``n_tokens``, then
    returns a uint16 view of the full ``[n_tokens, *ids]`` slice —
    matches the layout ``encoder.decode_record`` expects.

    Returned view aliases the underlying memmap memory; copy with
    ``np.array(view, copy=True)`` if the caller intends to keep it
    after the memmap closes.
    """
    # Read the size header as one u16 (avoid struct.unpack overhead;
    # this is on the dataloader hot path).
    header_view = memmap[offset:offset + _U16_BYTES].view(dtype=np.uint16)
    assert header_view.size == 1, (
        f"read_record: cannot read u16 header at offset {offset}; "
        f"memmap slice is {memmap[offset:offset + _U16_BYTES].size} "
        "bytes (expected 2)"
    )
    n_tokens = int(header_view[0])
    # Full record bytes = header (2) + payload (2 * n_tokens). Slice
    # then reinterpret as uint16 so the caller gets
    # ``[n_tokens, *ids]`` in one array — matches ``encode_record``
    # output and ``decode_record`` input.
    total_bytes = _U16_BYTES + n_tokens * _U16_BYTES
    raw = memmap[offset:offset + total_bytes]
    assert raw.size == total_bytes, (
        f"read_record: short read at offset {offset}; got {raw.size} "
        f"bytes, expected {total_bytes} (n_tokens={n_tokens})"
    )
    return raw.view(dtype=np.uint16)
