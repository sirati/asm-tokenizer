"""Binary-format writers split out of ``io.py`` to keep that module focused.

Single concern: encode ONE function record into ``_data.bin`` and ONE
4-byte index entry into ``_index.bin``. Each record is self-describing
via :class:`~tokenizer.aligned_data.binary_format.BinaryHeader` so the
index entry shrinks to a 16-byte-aligned offset only; length and the
former overlong escape are gone. Cap-overflow handling mirrors the
project's "log + skip + continue build" policy (see
:func:`tokenizer.memmap_builder.error_log.write_error_log_entry`).

Re-exported from :mod:`tokenizer.aligned_data.io` so external callers
keep the existing import path.
"""

from __future__ import annotations

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
    """Write one function record at the current file position.

    The record body is laid out exactly as
    :func:`~tokenizer.aligned_data.binary_format.derive_pad_placement`
    prescribes (self-describing header, optional pre-pad, block words,
    optional post-pad, tokens) so the total is always a multiple of
    :data:`RECORD_ALIGNMENT` (= 16). The writer never asks the caller to
    pre-compute the geometry -- it builds a :class:`BinaryHeader`,
    encodes it, derives pad placement, and writes the body in one shot.

    Returns ``(data_offset, total_record_bytes)`` on success.
    ``total_record_bytes`` is what the writer wrote -- the index
    layer never needs it (records are self-describing), but the
    matched-arm inlining-cell wire format embeds per-callee
    ``[idx, start, length, is_matched]`` quads, so callers building
    that cell need the length and would otherwise have to re-parse
    the encoded header to recover it.

    On :class:`IndexEntrySkip` (insn_len / block_word_count /
    token_count cap overflow from
    :func:`~tokenizer.aligned_data.binary_format.encode_binary_header`)
    the partial write is truncated back to the pre-call position and
    ``None`` is returned; ``error_log`` (when supplied) receives one
    TSV row naming the offending field, the function name, and the
    offending value. With ``error_log=None`` the exception re-raises so
    the caller can decide a different policy.

    ``dedup_cache`` (optional ``dict``) is keyed by ``(insn_bytes,
    block_bytes, tokens_bytes)`` and short-circuits to the cached
    ``(offset, total)`` on hit; skipped writes are never cached.
    """
    insn_bytes = insn_runlength.astype(np.uint8).tobytes()
    block_enc = determine_block_encoding(block_runlength)
    block_dtype = (np.uint8, np.uint16, np.uint32)[block_enc]
    block_bytes = block_runlength.astype(block_dtype).tobytes()
    tokens_bytes = tokens.tobytes()

    cache_key = None
    if dedup_cache is not None:
        cache_key = (insn_bytes, block_bytes, tokens_bytes)
        cached = dedup_cache.get(cache_key)
        if cached is not None:
            return cached

    data_offset = file2.tell()
    insn_len = len(insn_bytes)
    block_word_count = len(block_runlength)
    token_count = len(tokens)

    # Header.format is informational; encode_binary_header re-derives the
    # canonical (shortest) form via the strict ultrashort predicate so
    # handing it ``Normal`` unconditionally still emits ultrashort when
    # eligible. Keeping that dispatch in ONE place (the encoder) means
    # this writer never reimplements the predicate.
    header = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=block_enc,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
    )

    try:
        header_bytes = encode_binary_header(header)
        # ``encode_binary_header`` canonicalises the form (a Normal-tagged
        # header that fits ultrashort is emitted in ultrashort form), so the
        # parsed-back header is the one whose ``format`` field matches the
        # bytes on disk. Geometry helpers (``derive_pad_placement`` /
        # ``record_total_size``) key on ``header.format``; using the parsed
        # version keeps writer and reader on the same dataclass instance --
        # one source of truth for the on-disk geometry.
        canonical_header, _prefix_bytes = parse_binary_header(header_bytes)
        pre_pad, post_pad = derive_pad_placement(canonical_header)
        total = record_total_size(canonical_header)

        file2.write(header_bytes)
        file2.write(insn_bytes)
        if pre_pad:
            file2.write(b"\x00" * pre_pad)
        file2.write(block_bytes)
        if post_pad:
            file2.write(b"\x00" * post_pad)
        file2.write(tokens_bytes)

        written = file2.tell() - data_offset
        assert written == total, (written, total)
        assert written % RECORD_ALIGNMENT == 0, written
    except IndexEntrySkip as exc:
        file2.seek(data_offset)
        file2.truncate()
        if error_log is None:
            raise
        # Lazy import: ``tokenizer.memmap_builder`` package init pulls
        # back into ``aligned_data``; a top-level import would cycle.
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return None

    result = (data_offset, total)
    if dedup_cache is not None:
        dedup_cache[cache_key] = result
    return result


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
        # Lazy import: ``tokenizer.memmap_builder`` package init pulls
        # back into ``aligned_data``; a top-level import would cycle.
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return

    before = file3.tell()
    file3.write(entry_bytes)
    assert file3.tell() - before == INDEX_ENTRY_SIZE, (
        f"index entry wrote {file3.tell() - before} bytes; "
        f"expected {INDEX_ENTRY_SIZE}"
    )
