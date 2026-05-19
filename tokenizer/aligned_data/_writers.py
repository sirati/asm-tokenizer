"""Binary-format writers split out of ``io.py`` to keep that module
under the 300 LOC cap.

Single concern: encode one function record into ``_data.bin`` and one
8-byte index entry into ``_index.bin``, including the pad-and-overlong
layout decisions and the cap-overflow / error-log handling. Re-exported
from ``tokenizer.aligned_data.io`` so external callers keep the existing
import path.
"""
import struct

import numpy as np

from .binary_format import (
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
    IndexEntrySkip,
    compute_pad,
    determine_block_encoding,
    encode_binary_header,
    pack_avg_len_bucket,
)
from .index_format import MAX_NORMAL_REAL_LENGTH, SENTINEL_LENGTH

# Per-record body prefix sizes for the writer's total-length arithmetic
# (single-source-of-truth shared with binary_format's reader path).
_NORMAL_PREFIX_BYTES = HEADER_BYTES
_OVERLONG_PREFIX_BYTES = HEADER_BYTES + OVERLONG_FIELD_BYTES

# Caps derived from the on-wire entry layout (see ``index_format.py``).
# offset is stored as the low 5 bytes of a u64 (u40); length is u16 of
# the shifted real length, with the value 0x0000 reserved as the sentinel
# that flags an overlong record (real length carried in ``_data.bin``).
_MAX_OFFSET_SHIFTED = (1 << 40) - 1
_MAX_NORMAL_LENGTH_SHIFTED = 0xFFFF
_MAX_OVERLONG_REAL_LENGTH = 0xFFFFFF << 2  # 67,108,860 bytes (~64 MiB)
_INDEX_ENTRY_SIZE = 8


def write_function_binary_data(
    file2,
    tokens,
    block_runlength,
    insn_runlength,
    dedup_cache=None,
    *,
    func_name: str = "",
    error_log=None,
):
    """Write one function record (pad-aligned, overlong-aware).

    Returns ``(data_offset, data_len)`` on success. On
    :class:`IndexEntrySkip` from the encode path the partial write is
    truncated and ``None`` is returned so the caller skips the index
    entry; if ``error_log`` is provided the skip reason is logged.
    """
    cache_key = None
    if dedup_cache is not None:
        cache_key = (
            tokens.tobytes(),
            block_runlength.tobytes(),
            insn_runlength.tobytes(),
        )
        if cache_key in dedup_cache:
            return dedup_cache[cache_key]

    data_offset = file2.tell()
    insn_bytes = insn_runlength.astype(np.uint8).tobytes()
    block_enc = determine_block_encoding(block_runlength)
    block_bytes = block_runlength.astype(
        [np.uint8, np.uint16, np.uint32][block_enc]
    ).tobytes()
    insn_len = len(insn_bytes)
    block_len = len(block_bytes)
    token_count = len(tokens)

    try:
        # Pick normal vs overlong layout from the would-be total length.
        pad_normal = compute_pad(insn_len, block_len, token_count, is_overlong=False)
        total_normal = _NORMAL_PREFIX_BYTES + insn_len + pad_normal + block_len + 2 * token_count
        if total_normal <= MAX_NORMAL_REAL_LENGTH:
            is_overlong = False
            pad_size = pad_normal
            total = total_normal
        else:
            pad_long = compute_pad(insn_len, block_len, token_count, is_overlong=True)
            total_long = _OVERLONG_PREFIX_BYTES + insn_len + pad_long + block_len + 2 * token_count
            if total_long > _MAX_OVERLONG_REAL_LENGTH:
                raise IndexEntrySkip("overlong_length_overflow", total_long)
            is_overlong = True
            pad_size = pad_long
            total = total_long

        header = encode_binary_header(insn_len, block_enc, block_len, pad_size=pad_size)
        file2.write(header)
        if is_overlong:
            file2.write(struct.pack("<I", total >> 2)[0:3])
        file2.write(insn_bytes)
        file2.write(b"\x00" * pad_size)
        file2.write(block_bytes)
        file2.write(tokens.tobytes())

        data_len = file2.tell() - data_offset
        assert data_len == total, (data_len, total)
        assert data_len % 4 == 0, data_len
    except IndexEntrySkip as exc:
        file2.seek(data_offset)
        file2.truncate()
        if error_log is not None:
            # Lazy import: ``tokenizer.memmap_builder`` package init pulls
            # back into ``aligned_data``; a top-level import would cycle.
            from tokenizer.memmap_builder.error_log import write_error_log_entry
            write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return None

    result = (data_offset, data_len)
    if dedup_cache is not None:
        dedup_cache[cache_key] = result
    return result


def pack_v1_entry(offset: int, length: int, avg_len: int) -> bytes:
    """Pack one 8-byte v1 index entry: u40 offset_shifted, u16 length_shifted, u8 avg_len.

    Pure function. Single source of truth for the on-wire 8-byte v1
    entry layout. Both ``write_index_entry`` (writes to ``_index.bin``)
    and ``inline_indexer.encode_inline_indexer`` (hex-embeds the entry
    inline in ``matched_sections.csv``) call into this packer so the
    layout exists in exactly one place.

    ``offset`` and ``length`` are shifted right by 2 (4-byte record
    alignment is a writer invariant). A real length above
    ``MAX_NORMAL_REAL_LENGTH`` (~256 KiB) is encoded with sentinel
    ``length_shifted == SENTINEL_LENGTH``; the real length then lives
    in the u24-shifted overlong field of the matching ``_data.bin``
    record (cap ~64 MiB). On cap violation :class:`IndexEntrySkip` is
    raised so callers can decide between logging-and-skipping vs
    propagating. Alignment violations are programmer errors and raise
    :class:`AssertionError`.
    """
    assert offset % 4 == 0, f"index entry offset must be 4-byte aligned; got {offset}"
    assert length % 4 == 0, f"index entry length must be 4-byte aligned; got {length}"
    assert length > 0, "index entry length must be > 0 (minimum padded record is 8 bytes)"

    offset_shifted = offset >> 2
    if offset_shifted > _MAX_OFFSET_SHIFTED:
        raise IndexEntrySkip("offset_overflow", offset)

    length_shifted = length >> 2
    if length_shifted <= _MAX_NORMAL_LENGTH_SHIFTED:
        length_field = length_shifted
    else:
        if length > _MAX_OVERLONG_REAL_LENGTH:
            raise IndexEntrySkip("overlong_length_overflow", length)
        length_field = SENTINEL_LENGTH

    avg_len_clamped = pack_avg_len_bucket(avg_len)
    # Low 5 bytes of a u64 LE = u40 LE on the wire.
    return (
        struct.pack("<Q", offset_shifted)[:5]
        + struct.pack("<H", length_field)
        + struct.pack("B", avg_len_clamped)
    )


def write_index_entry(
    file3,
    start: int,
    length: int,
    avg_len: int,
    *,
    func_name: str = "",
    error_log=None,
) -> None:
    """Write one 8-byte index entry; thin wrapper over :func:`pack_v1_entry`.

    On cap violation :class:`IndexEntrySkip` propagates; when
    ``error_log`` is supplied the exception is logged and the function
    returns ``None`` (no entry written). ``func_name`` is logged so the
    offending function is recoverable. Alignment violations are
    programmer errors and raise :class:`AssertionError` (never logged).
    """
    try:
        entry_bytes = pack_v1_entry(start, length, avg_len)
    except IndexEntrySkip as exc:
        if error_log is None:
            raise
        # Lazy import: ``tokenizer.memmap_builder`` package init pulls
        # back into ``aligned_data``; a top-level import would cycle.
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return

    before = file3.tell()
    file3.write(entry_bytes)
    assert file3.tell() - before == _INDEX_ENTRY_SIZE, (
        f"index entry wrote {file3.tell() - before} bytes; expected {_INDEX_ENTRY_SIZE}"
    )
