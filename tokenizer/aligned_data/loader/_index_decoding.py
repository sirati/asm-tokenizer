"""Per-record length / token-count helpers shared by the reader chain.

Single concern: bridge the sentinel marker the index reader hands the
caller (``stored_length == 0`` flags an overlong record) and the real
record geometry needed to slice ``_data.bin``. Both ``metadata_loader``
(``load_unmatched_lengths``) and ``session`` (record slicing) need the
same resolution path; this module owns it once.

The wire layout itself lives in :mod:`tokenizer.aligned_data.binary_format`
and :mod:`tokenizer.aligned_data.index_format` -- this module imports the
sizes and parsers; it never duplicates them.
"""

from __future__ import annotations

from typing import Tuple

from tokenizer.aligned_data.binary_format import (
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
    BinaryHeader,
    parse_binary_header,
)

# Cached per-arithmetic sum so the hot resolve path stays branch-free.
_OVERLONG_PREFIX = HEADER_BYTES + OVERLONG_FIELD_BYTES

# Real-length threshold above which the writer always picks the overlong
# layout. Mirrors ``tokenizer.aligned_data._writers._MAX_NORMAL_REAL_LENGTH``;
# kept private here so the wire-format detail is not duplicated across
# the reader chain.
_MAX_NORMAL_REAL_LENGTH = 0xFFFF << 2


def resolve_record_length(
    data_memmap, start: int, stored_length: int
) -> Tuple[int, bool]:
    """Resolve a record's real byte length + overlong flag from one stored value.

    Two callers cross this helper: the unmatched arm passes the index
    reader's ``stored_length`` (sentinel ``0`` flags overlong, real
    length lives in the data record's u24 field); the matched arm
    passes ``data_len`` straight from the sections CSV (always the real
    length, never sentinel). Both converge through the same rule below
    so layout knowledge stays in one place.

    * ``stored_length == 0``  -> sentinel; read the overlong field.
    * ``stored_length > _MAX_NORMAL_REAL_LENGTH`` -> the writer chose
      the overlong layout because the record could not fit the index's
      u16 cap; return ``(stored_length, True)``.
    * else -> normal record; ``(stored_length, False)``.
    """
    if stored_length == 0:
        field = data_memmap[start + HEADER_BYTES : start + _OVERLONG_PREFIX]
        return int.from_bytes(bytes(field), "little") << 2, True
    if stored_length > _MAX_NORMAL_REAL_LENGTH:
        return stored_length, True
    return stored_length, False


def record_token_count(
    data_memmap, start: int, stored_length: int
) -> int:
    """Real token count for the record at ``start``.

    Resolves the overlong sentinel if present, parses the header, and
    derives the token count from the tail after insn + pad + block.
    Token entries are uint16 so the body remainder is halved.
    """
    real_length, is_overlong = resolve_record_length(
        data_memmap, start, stored_length
    )
    body_prefix = _OVERLONG_PREFIX if is_overlong else HEADER_BYTES
    header: BinaryHeader = parse_binary_header(
        data_memmap[start : start + HEADER_BYTES]
    )
    body_bytes = (
        real_length
        - body_prefix
        - header.insn_len
        - header.pad_size
        - header.block_len
    )
    return body_bytes // 2
