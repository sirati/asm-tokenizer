"""Inline hex codec for a single ``_data.bin`` offset.

Single concern: encode/decode a ``_data.bin`` byte offset as an 8-hex-
character string so it can sit inline in ``matched_sections.csv``
variant rows (no separate index file, no extra I/O). Layout knowledge
is NOT duplicated here -- packing delegates to
``index_format.pack_index_entry`` and decoding delegates to
``index_format.decode_index_entry``; this module is a thin codec
wrapper over those two single sources of truth.

Records in ``_data.bin`` are 16-byte aligned, so the on-disk entry is
one ``u32 = offset >> 4`` (four bytes, eight hex characters). The
record is self-describing -- the header at ``offset`` carries every
geometry field a reader needs -- so the inline encoding no longer
carries a length, an overlong sentinel, or any per-record metric.
"""

from __future__ import annotations

from .index_format import decode_index_entry, pack_index_entry

_INLINE_HEX_LEN: int = 8  # 4 bytes * 2 hex chars per byte
_LEGACY_INLINE_HEX_LEN: int = 16  # old 8-byte entry (offset + length + avg_len)


def encode_inline_indexer(offset: int) -> str:
    """Return the 8-hex-character encoding of ``offset`` into ``_data.bin``.

    Calls :func:`pack_index_entry` and hex-encodes the resulting 4
    bytes. Always returns a string of length 8. Raises
    :class:`IndexEntrySkip` on cap overflow and :class:`AssertionError`
    on alignment violations (same contract as ``pack_index_entry``).
    """
    return pack_index_entry(offset).hex()


def decode_inline_indexer(hex_str: str) -> int:
    """Decode an 8-hex-character inline entry into the ``_data.bin`` offset.

    Reuses :func:`index_format.decode_index_entry` for the actual byte
    interpretation -- so writer-side and reader-side share one layout
    definition.

    Raises :class:`ValueError` when ``hex_str`` is not exactly 8
    characters or contains non-hex characters. A 16-character input is
    the legacy pre-cutover layout; this raises a migration-pointing
    ValueError instructing the caller to re-run ``memmap_builder``.
    """
    if len(hex_str) == _LEGACY_INLINE_HEX_LEN:
        raise ValueError(
            f"inline indexer is {_LEGACY_INLINE_HEX_LEN} hex chars (legacy "
            f"8-byte entry); expected {_INLINE_HEX_LEN}; re-run memmap_builder "
            f"on the per-binary CSVs to regenerate"
        )
    if len(hex_str) != _INLINE_HEX_LEN:
        raise ValueError(
            f"inline indexer must be exactly {_INLINE_HEX_LEN} hex chars; got {len(hex_str)}"
        )
    try:
        entry_bytes = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise ValueError(f"inline indexer is not valid hex: {hex_str!r}") from exc
    return decode_index_entry(entry_bytes)
