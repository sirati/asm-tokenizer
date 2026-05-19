import csv

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from .binary_format import (
    extract_arrays_from_data,
    parse_binary_header,
)
from .csv_format import format_inlining_dict, format_variant_refs
from ._writers import write_function_binary_data, write_index_entry  # re-export

__all__ = (
    "decode_and_translate_tokens",
    "decode_runlengths",
    "parse_function_data_header",
    "parse_function_data_memmap",
    "read_data_file",
    "read_function_data_memmap",
    "read_index_file",
    "read_sections_file",
    "write_function_binary_data",
    "write_function_section_csv",
    "write_index_entry",
    "write_unmatched_section_csv",
)


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


def read_data_file(data_path, offset, length, is_overlong: bool = False):
    """Read the binary data for a function from the data file given offset and length.

    ``is_overlong`` is forwarded to the parser so the body offset shifts
    past the 3-byte overlong-length field when the caller already
    resolved the real length via the index sentinel.
    """
    with open(data_path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
        header = parse_binary_header(data)
        return extract_arrays_from_data(data, header, is_overlong=is_overlong)


def parse_function_data_memmap(memmap_handle, offset, length, is_overlong: bool = False):
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

    ``is_overlong`` is forwarded to the parser; the session layer sets
    it after observing the index-entry sentinel.

    Returns ``(insn_runlength, block_runlength, tokens)``.
    """
    data = memmap_handle[offset:offset + length]
    return parse_function_data_header(data, is_overlong=is_overlong)


def read_function_data_memmap(data_path, offset, length, is_overlong: bool = False):
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
    return parse_function_data_memmap(data, offset, length, is_overlong=is_overlong)


def parse_function_data_header(data_bytes, is_overlong: bool = False):
    """
    Parse the header and return (insn_runlength, block_runlength, tokens) ndarrays.
    data_bytes: bytes or 1D uint8 array

    ``is_overlong`` is forwarded to :func:`extract_arrays_from_data`
    so the body offset accounts for the optional 3-byte overlong-length
    field that precedes the body in overlong records.
    """
    header = parse_binary_header(data_bytes)
    return extract_arrays_from_data(data_bytes, header, is_overlong=is_overlong)
