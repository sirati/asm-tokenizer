"""Inline hex codec for the v1 8-byte index entry.

Single concern: encode/decode a v1 index entry as a 16-hex-character
string so it can sit inline in ``matched_sections.csv`` variant rows
(no separate index file, no extra I/O). Layout knowledge is NOT
duplicated here -- packing delegates to ``_writers.pack_v1_entry`` and
decoding delegates to ``index_format.decode_index_entry``; this module
is a thin codec wrapper over those two single sources of truth.

The trailing u8 byte of the entry is reserved for inline use: the
encoder writes ``avg_len=0`` (per-variant entries have no obvious
per-record metric) and the decoder discards it.
"""

from __future__ import annotations

from typing import Tuple

from ._writers import pack_v1_entry
from .index_format import ALIGNMENT_SHIFT, decode_index_entry

_INLINE_HEX_LEN: int = 16  # 8 bytes * 2 hex chars per byte


def encode_inline_indexer(offset: int, length: int) -> str:
    """Return the 16-hex-character encoding of a v1 entry for ``(offset, length)``.

    Calls :func:`pack_v1_entry` with ``avg_len=0`` (the reserved u8 byte)
    and hex-encodes the resulting 8 bytes. Always returns a string of
    length 16. Raises :class:`IndexEntrySkip` on cap overflow and
    :class:`AssertionError` on alignment violations (same contract as
    ``pack_v1_entry``).
    """
    return pack_v1_entry(offset, length, 0).hex()


def decode_inline_indexer(hex_str: str) -> Tuple[int, int, bool]:
    """Decode a 16-hex-character inline entry into ``(start, length, is_overlong)``.

    Reuses :func:`index_format.decode_index_entry` for the actual byte
    interpretation -- so writer-side and reader-side share one layout
    definition. The trailing u8 (avg_len) is discarded because the
    inline encoding reserves that byte. ``length`` is ``0`` for an
    overlong (sentinel) entry; the caller resolves the real length from
    the data record's overlong field.

    Raises :class:`ValueError` when ``hex_str`` is not exactly 16
    characters or contains non-hex characters.
    """
    if len(hex_str) != _INLINE_HEX_LEN:
        raise ValueError(
            f"inline indexer must be exactly {_INLINE_HEX_LEN} hex chars; got {len(hex_str)}"
        )
    try:
        entry_bytes = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise ValueError(f"inline indexer is not valid hex: {hex_str!r}") from exc
    start, length, _avg_len, is_overlong = decode_index_entry(
        entry_bytes, shift=ALIGNMENT_SHIFT
    )
    return start, length, is_overlong
