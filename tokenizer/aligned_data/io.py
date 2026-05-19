import csv
import struct

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from .binary_format import (
    IndexEntrySkip,
    compute_pad,
    determine_block_encoding,
    encode_binary_header,
    extract_arrays_from_data,
    parse_binary_header,
)
from .csv_format import format_inlining_dict, format_variant_refs

# Largest real record length the index entry can carry without the sentinel.
# `length_shifted` is u16; multiplied by the alignment shift (<<2) gives the
# cap. Records strictly above this switch to the overlong layout with a u24
# length field stored inside the data record (cap 0xFFFFFF << 2 bytes).
_MAX_NORMAL_REAL_LENGTH = 0xFFFF << 2
_MAX_OVERLONG_REAL_LENGTH = 0xFFFFFF << 2


def decode_and_translate_tokens(row, mapping=None):
    # `mapping` is the per-binary local-ID → unified-ID lookup written by
    # `vocab_unifier`. Under format_version=2 the unifier identity-maps IDs
    # 0..255 (the protocol-reserved digit slots), so inline-digit
    # continuations survive `mapping[tokens]` byte-for-byte: digit 0x42
    # in the per-binary stream stays digit 0x42 in the unified stream.
    # No v2-specific branch needed here — the fancy-indexing semantics
    # do the right thing as long as the unifier emits identity for that
    # range (see `tokenizer/vocab_unifier/unifier.py`).
    tokens = base64_to_ndarray_vec(row["tokens_base64"])
    if mapping is not None:
        tokens = mapping[tokens]
    return tokens.astype(np.uint16)


def decode_runlengths(row):
    block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
    insn_runlength = base64_to_ndarray_vec(row["instruction_runlength_base64"])
    return block_runlength, insn_runlength


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
        total_normal = 6 + insn_len + pad_normal + block_len + 2 * token_count
        if total_normal <= _MAX_NORMAL_REAL_LENGTH:
            is_overlong = False
            pad_size = pad_normal
            total = total_normal
        else:
            pad_long = compute_pad(insn_len, block_len, token_count, is_overlong=True)
            total_long = 9 + insn_len + pad_long + block_len + 2 * token_count
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
            from tokenizer.memmap_builder.error_log import write_error_log_entry
            write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return None

    result = (data_offset, data_len)
    if dedup_cache is not None:
        dedup_cache[cache_key] = result
    return result


def write_index_entry(file3, start, length, avg_len):
    file3.write(struct.pack("<I", start))
    file3.write(struct.pack("<I", length)[0:3])
    avg_len_clamped = min(avg_len >> 4, 255)
    file3.write(struct.pack("B", avg_len_clamped))


def write_function_section_csv(
    writer,
    variant_ref,
    inlining_list,
    data_offset,
    data_len,
):
    """Write one matched-section row.

    ``variant_ref`` is the ``0x<hex>`` row index into the per-group
    ``<binary>_variants.csv`` sidecar (see
    ``tokenizer.memmap_builder.variants.VariantRegistry``). The
    4-axis canonical tuple (arch, compiler, version, opt) and any
    sidecar ``extra_metadata`` are recoverable via that ref;
    keeping them out of the section CSV avoids the per-row repetition
    that conflated variants sharing the canonical-4 axes.
    """
    inlining_str = format_inlining_dict(inlining_list)
    writer.writerow(
        [
            variant_ref,
            inlining_str,
            f"{data_offset:x}",
            f"{data_len:x}",
        ]
    )


def write_unmatched_section_csv(
    writer,
    func_name,
    variant_refs,
    called_functions_str,
    inlining_data_str,
    data_offset,
    data_len,
):
    """Write one unmatched-section row.

    ``variant_refs`` is the ordered list of ``0x<hex>`` row indices
    (one per version present for this unmatched function). Encoded
    semicolon-joined into a single cell, mirroring the structure of
    the legacy ``compiler_sets`` cell so the column count stays
    constant across the section CSV.
    """
    variants_str = format_variant_refs(variant_refs)
    writer.writerow(
        [
            func_name,
            variants_str,
            called_functions_str,
            inlining_data_str,
            f"{data_offset:x}",
            f"{data_len:x}",
        ]
    )


def read_index_file(index_path):
    """Read the index file and yield (start, length, avg_len) for each function."""
    with open(index_path, "rb") as f:
        while True:
            start_bytes = f.read(4)
            if not start_bytes or len(start_bytes) < 4:
                break
            start = int.from_bytes(start_bytes, "little")
            length_bytes = f.read(3)
            if not length_bytes or len(length_bytes) < 3:
                break
            length = int.from_bytes(length_bytes, "little")
            avg_len_byte = f.read(1)
            if not avg_len_byte or len(avg_len_byte) < 1:
                break
            avg_len = int.from_bytes(avg_len_byte, "little")
            yield (start, length, avg_len)


def read_sections_file(sections_path):
    """Read the sections CSV file and yield (func_name, [rows]) for each function section."""
    with open(sections_path, newline="", encoding="ascii") as f:
        reader = csv.reader(f)
        func_name = None
        rows = []
        for row in reader:
            if not row or (len(row) == 1 and row[0]):
                # New section or blank line
                if func_name is not None and rows:
                    yield (func_name, rows)
                func_name = row[0] if row and row[0] else None
                rows = []
            elif func_name:
                rows.append(row)
        if func_name and rows:
            yield (func_name, rows)


def read_data_file(data_path, offset, length):
    """Read the binary data for a function from the data file given offset and length."""
    with open(data_path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
        header = parse_binary_header(data)
        return extract_arrays_from_data(data, header)


def parse_function_data_memmap(memmap_handle, offset, length):
    """Slice one function record from an already-open ``_data.bin`` view.

    ``memmap_handle`` is the caller's already-open ``np.memmap`` (or any
    1-D uint8 ndarray view) of the WHOLE ``_data.bin`` file; ``offset``
    and ``length`` are the byte range of one function record within it
    (the same values stored as ``data_offset`` / ``data_len`` in the
    section CSV).

    Pure parsing: no file I/O, no handle lifecycle. Mirrors the
    ``variant_tokens.record.read_record`` pattern so a future session
    owns one open handle per bin file and slices many records out of
    it without re-opening per call.

    Returns ``(insn_runlength, block_runlength, tokens)``.
    """
    data = memmap_handle[offset:offset + length]
    return parse_function_data_header(data)


def read_function_data_memmap(data_path, offset, length):
    """
    Read the binary data for a function from the data file using numpy.memmap for random access.
    Returns: insn_runlength, block_runlength, tokens

    Thin wrapper that opens the full ``_data.bin`` as a uint8 memmap,
    delegates to :func:`parse_function_data_memmap`, and lets the
    memmap close when the local reference drops. Use the open-handle
    form directly inside a session that needs many records out of the
    same bin file (avoids per-call mmap overhead).
    """
    data = np.memmap(data_path, dtype=np.uint8, mode="r")
    return parse_function_data_memmap(data, offset, length)


def parse_function_data_header(data_bytes):
    """
    Parse the header and return (insn_runlength, block_runlength, tokens) ndarrays.
    data_bytes: bytes or 1D uint8 array
    """
    header = parse_binary_header(data_bytes)
    return extract_arrays_from_data(data_bytes, header)
