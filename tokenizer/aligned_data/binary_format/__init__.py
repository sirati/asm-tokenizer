"""Self-describing per-record header for ``_data.bin`` (public re-exports).

Submodules:

* :mod:`_header` -- format enum + dataclass + parse/encode + caps.
* :mod:`_pad` -- pad placement rule + total record size.
* :mod:`_body` -- zero-copy (insn, block, tokens) slicing.

This package is the sole source of truth for on-wire record geometry.
All other modules in the project consume only the names re-exported
here -- nothing imports a submodule directly.
"""

from __future__ import annotations

from ._body import extract_arrays_from_data
from ._header import (
    BLOCK_WORD_SIZE,
    ENTRY_IDX_SIZE,
    MAX_HEADER_BYTES,
    NORMAL_BLOCK_WORD_CAP,
    NORMAL_INSN_CAP,
    NORMAL_PREFIX_BYTES,
    NORMAL_TOKEN_CAPS,
    RECORD_ALIGNMENT,
    ULTRASHORT_BLOCK_CAP,
    ULTRASHORT_INSN_CAP,
    ULTRASHORT_PREFIX_BYTES,
    ULTRASHORT_TOKENS_CAP,
    BinaryHeader,
    BinaryHeaderFormat,
    IndexEntrySkip,
    determine_block_encoding,
    encode_binary_header,
    parse_binary_header,
    prefix_bytes_for_header,
)
from ._pad import derive_pad_placement, record_total_size


def record_token_count(header: BinaryHeader) -> int:
    """Return ``header.token_count`` (one-line convenience for callers)."""
    return header.token_count


def record_token_count_from_memmap(data_mmap, offset: int) -> int:
    """Parse the header at ``data_mmap[offset:]`` and return its token count.

    Reads at most :data:`MAX_HEADER_BYTES` bytes from the memmap; the
    real header may be shorter (ultrashort or a narrower token width
    tag) but :func:`parse_binary_header` reads only the bytes it needs.
    Used by the unmatched-arm loader for per-record token counts
    without materialising the whole record body.
    """
    end = min(offset + MAX_HEADER_BYTES, len(data_mmap))
    header, _ = parse_binary_header(data_mmap[offset:end])
    return header.token_count


__all__ = [
    # Constants
    "BLOCK_WORD_SIZE",
    "ENTRY_IDX_SIZE",
    "MAX_HEADER_BYTES",
    "NORMAL_BLOCK_WORD_CAP",
    "NORMAL_INSN_CAP",
    "NORMAL_PREFIX_BYTES",
    "NORMAL_TOKEN_CAPS",
    "RECORD_ALIGNMENT",
    "ULTRASHORT_BLOCK_CAP",
    "ULTRASHORT_INSN_CAP",
    "ULTRASHORT_PREFIX_BYTES",
    "ULTRASHORT_TOKENS_CAP",
    # Header dataclass + format + skip exception
    "BinaryHeader",
    "BinaryHeaderFormat",
    "IndexEntrySkip",
    # Encoder/decoder
    "determine_block_encoding",
    "encode_binary_header",
    "parse_binary_header",
    "prefix_bytes_for_header",
    # Pad + geometry
    "derive_pad_placement",
    "record_total_size",
    "record_token_count",
    "record_token_count_from_memmap",
    # Body slicing
    "extract_arrays_from_data",
]
