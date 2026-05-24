"""Pure record-encoding helpers for ``_data.bin``.

Single concern: take the three per-function ndarrays (insn runlength,
block runlength, tokens) and produce the canonical record bytes
(self-describing header + body) that go on disk. The actual writing
happens at a higher layer — see
:mod:`tokenizer.memmap_builder._dedup` for the content-addressed dedup
pipeline that owns the memmap writer.

The per-function ``IndexEntrySkip`` policy is preserved: when the
encoder raises (insn_len / block_word_count / token_count cap overflow),
the offending field is logged into ``error.log`` (if a handle was
supplied) and ``None`` is returned. With no ``error_log`` the exception
re-raises so the caller can choose a different policy.

Re-exported from :mod:`tokenizer.aligned_data.io` so external callers
keep the existing import path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .binary_format import (
    BinaryHeader,
    BinaryHeaderFormat,
    IndexEntrySkip,
    RECORD_ALIGNMENT,
    derive_pad_placement,
    determine_block_encoding,
    encode_binary_header,
    parse_binary_header,
    record_total_size,
)
from .index_format import INDEX_ENTRY_SIZE, pack_index_entry


def assemble_function_record(
    tokens: np.ndarray,
    block_runlength: np.ndarray,
    insn_runlength: np.ndarray,
    *,
    entry_idx: int,
    func_name: str = "",
    error_log=None,
) -> Optional[bytes]:
    """Assemble the on-disk bytes for one function record.

    Layout matches :func:`derive_pad_placement`: self-describing header,
    optional pre-pad, block words, optional post-pad, tokens. Total is
    always a multiple of :data:`RECORD_ALIGNMENT` (= 16).

    ``entry_idx`` is the record's encounter-order ordinal within its
    containing ``_data.bin`` file (first written record = 0, second = 1,
    ...). It is stamped into the on-wire header so the loader can later
    cross-check the per-arm index against the data-bin contents.

    Returns the raw record bytes on success.

    On :class:`IndexEntrySkip` (``insn_len_overflow`` /
    ``block_word_count_overflow`` / ``token_count_overflow``) — with
    ``error_log`` supplied the offending field is logged and ``None`` is
    returned; with ``error_log=None`` the exception re-raises so the
    caller can decide a different policy.
    """
    insn_bytes = insn_runlength.astype(np.uint8).tobytes()
    block_enc = determine_block_encoding(block_runlength)
    block_dtype = (np.uint8, np.uint16, np.uint32)[block_enc]
    block_bytes = block_runlength.astype(block_dtype).tobytes()
    tokens_bytes = tokens.tobytes()

    insn_len = len(insn_bytes)
    block_word_count = len(block_runlength)
    token_count = len(tokens)

    # Header.format is informational; encode_binary_header re-derives the
    # canonical (shortest) form via the strict ultrashort predicate.
    header = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=block_enc,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
        entry_idx=entry_idx,
    )

    try:
        header_bytes = encode_binary_header(header)
        canonical_header, _prefix_bytes = parse_binary_header(header_bytes)
        pre_pad, post_pad = derive_pad_placement(canonical_header)
        total = record_total_size(canonical_header)

        parts = [header_bytes, insn_bytes]
        if pre_pad:
            parts.append(b"\x00" * pre_pad)
        parts.append(block_bytes)
        if post_pad:
            parts.append(b"\x00" * post_pad)
        parts.append(tokens_bytes)
        record_bytes = b"".join(parts)

        assert len(record_bytes) == total, (len(record_bytes), total)
        assert len(record_bytes) % RECORD_ALIGNMENT == 0, len(record_bytes)
        return record_bytes
    except IndexEntrySkip as exc:
        if error_log is None:
            raise
        # Lazy import: ``tokenizer.memmap_builder`` package init pulls
        # back into ``aligned_data``; a top-level import would cycle.
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return None


def write_function_binary_data(
    file2,
    tokens: np.ndarray,
    block_runlength: np.ndarray,
    insn_runlength: np.ndarray,
    *,
    entry_idx: int,
    func_name: str = "",
    error_log=None,
):
    """Append one function record to a file-like object at its current position.

    Thin wrapper over :func:`assemble_function_record` for ad-hoc /
    test usage. The production memmap-builder pipeline writes via
    :mod:`tokenizer.memmap_builder._dedup` and does NOT go through
    this helper — the dedup helper assembles the same record bytes
    and routes them through the content-addressed dedup map.

    ``entry_idx`` is the record's encounter-order ordinal within the
    file; the caller is responsible for sequencing it (first record
    written = 0, second = 1, ...).

    Returns ``(data_offset, total_record_bytes)`` on success, ``None``
    on an :class:`IndexEntrySkip` cap overflow (logged into
    ``error_log`` when supplied).
    """
    data_offset = file2.tell()
    record_bytes = assemble_function_record(
        tokens,
        block_runlength,
        insn_runlength,
        entry_idx=entry_idx,
        func_name=func_name,
        error_log=error_log,
    )
    if record_bytes is None:
        return None
    file2.write(record_bytes)
    return (data_offset, len(record_bytes))


def write_index_entry(
    file3,
    offset: int,
    *,
    func_name: str = "",
    error_log=None,
) -> None:
    """Write one 4-byte index entry; thin wrapper over :func:`pack_index_entry`.

    On :class:`IndexEntrySkip` (offset above the ~64 GiB cap): with
    ``error_log`` supplied the exception is logged and the function
    returns without writing; without ``error_log`` it propagates.
    Alignment violations (``offset`` not a multiple of
    :data:`RECORD_ALIGNMENT`) are programmer errors and raise
    :class:`AssertionError` unconditionally -- never logged.
    """
    try:
        entry_bytes = pack_index_entry(offset)
    except IndexEntrySkip as exc:
        if error_log is None:
            raise
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return

    before = file3.tell()
    file3.write(entry_bytes)
    assert file3.tell() - before == INDEX_ENTRY_SIZE, (
        f"index entry wrote {file3.tell() - before} bytes; "
        f"expected {INDEX_ENTRY_SIZE}"
    )
