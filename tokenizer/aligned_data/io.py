import csv

import numpy as np

from .binary_format import (
    MAX_HEADER_BYTES,
    extract_arrays_from_data,
    parse_binary_header,
    record_total_size,
)
from .csv_format import format_inlining_dict, format_variant_refs
from .index_format import iter_index_entries
from ._writers import (  # re-export
    assemble_function_record,
    write_function_binary_data,
    write_index_entry,
)

__all__ = (
    "assemble_function_record",
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




def write_function_section_csv(
    writer,
    variant_ref,
    inlining_list,
    indexer_hex,
):
    """Write one matched-section variant row (3 cells).

    ``variant_ref`` is the ``0x<hex>`` row index into the per-group
    ``<binary>_variants.csv`` sidecar (see
    ``tokenizer.memmap_builder.variants.VariantRegistry``). The
    4-axis canonical tuple (arch, compiler, version, opt) and any
    sidecar ``extra_metadata`` are recoverable via that ref;
    keeping them out of the section CSV avoids the per-row repetition
    that conflated variants sharing the canonical-4 axes.

    ``indexer_hex`` is the 8-hex-char inline encoding of the v1
    4-byte index entry for this variant's ``_data.bin`` record
    (one ``u32 = offset >> 4``; records are self-describing so no
    length / overlong marker rides alongside the offset). Callers
    compute it via
    :func:`tokenizer.aligned_data.inline_indexer.encode_inline_indexer`;
    this writer treats it as an opaque string and emits it verbatim,
    so the writer stays unaware of the entry layout.
    """
    inlining_str = format_inlining_dict(inlining_list)
    writer.writerow(
        [
            variant_ref,
            inlining_str,
            indexer_hex,
        ]
    )


def write_unmatched_section_csv(
    writer,
    line_no_b64,
    variant_refs,
    called_functions_str,
    inlining_data_str,
    indexer_hex,
):
    """Write one unmatched-section row (5 cells).

    ``line_no_b64`` is the compact urlsafe-base64 of this function's
    1-indexed line number in the ``<binary>_function_names.txt``
    sidecar; callers compute it via the registry. ``called_functions_str``
    likewise carries comma-joined base64 line nos (NOT raw function
    names) produced by the caller -- the writer is unaware of either
    indirection.

    ``variant_refs`` is the ordered list of ``0x<hex>`` row indices
    (one per version present for this unmatched function). Encoded
    semicolon-joined into a single cell, mirroring the structure of
    the legacy ``compiler_sets`` cell so the column count stays
    constant across the section CSV.

    ``indexer_hex`` is the 8-hex-char inline encoding of the v1
    4-byte index entry for this function's first variant ``_data.bin``
    record (one ``u32 = offset >> 4``; records are self-describing so
    no length / overlong marker rides alongside the offset). Callers
    compute it via
    :func:`tokenizer.aligned_data.inline_indexer.encode_inline_indexer`;
    this writer treats it as an opaque string and emits it verbatim.
    """
    variants_str = format_variant_refs(variant_refs)
    writer.writerow(
        [
            line_no_b64,
            variants_str,
            called_functions_str,
            inlining_data_str,
            indexer_hex,
        ]
    )


def read_index_file(index_path):
    """Yield ``start`` (real byte offset) for each entry in a v1 ``_index.bin``.

    Thin wrapper over :func:`tokenizer.aligned_data.index_format.iter_index_entries`
    so external callers keep the existing import path. Records are
    self-describing in ``_data.bin`` (their headers carry the geometry)
    so the index entry shrinks to an offset only. Raises
    :class:`ValueError` on missing prelude / wrong format version.
    """
    yield from iter_index_entries(index_path)


def read_sections_file(sections_path):
    """Yield ``(func_name, [variant_rows])`` for each function section.

    Routes through :func:`tokenizer.aligned_data.loader.metadata_loader.open_sections_csv`
    so the ``# format=N`` prelude is consumed (and validated) before
    the ``csv.reader`` sees the stream. Without this routing, the
    prelude line would appear as a phantom section row and silently
    corrupt the iteration. The import is local to avoid a circular
    dependency between ``io`` and ``loader``.

    Section layout (matched arm; what pass-2 writes):
    ``header_row`` (``[func_name, unique_called_str]`` -- 2 cells)
    followed by zero or more variant rows (3 cells each), terminated
    by a blank row. The first row of the file (after the prelude) is
    a header; every blank row marks the boundary before the next
    header. ``variant_rows`` is the list of variant rows between the
    header and its trailing blank; ``func_name`` is the header's
    first cell.
    """
    from .loader.metadata_loader import open_sections_csv

    handle, _ = open_sections_csv(sections_path)
    try:
        reader = csv.reader(handle)
        func_name = None
        rows = []
        expecting_header = True
        for row in reader:
            if not row:
                # Blank row = end of current section. Emit and reset.
                if func_name is not None:
                    yield (func_name, rows)
                func_name = None
                rows = []
                expecting_header = True
                continue
            if expecting_header:
                func_name = row[0]
                rows = []
                expecting_header = False
            else:
                rows.append(row)
        if func_name is not None:
            yield (func_name, rows)
    finally:
        handle.close()


def read_data_file(data_path, offset):
    """Read one self-describing function record from ``_data.bin``.

    The record at ``offset`` carries its own geometry in the header --
    :func:`parse_binary_header` returns the header dataclass + the
    prefix-byte count the header occupied on disk, and
    :func:`record_total_size` yields the total record byte count; no
    companion length / overlong flag is needed at the boundary.
    """
    with open(data_path, "rb") as f:
        f.seek(offset)
        prefix_window = f.read(MAX_HEADER_BYTES)
        header, prefix_bytes = parse_binary_header(prefix_window)
        total = record_total_size(header)
        f.seek(offset)
        data = f.read(total)
        return extract_arrays_from_data(data, header, prefix_bytes)


def parse_function_data_memmap(memmap_handle, offset):
    """Slice one function record from an already-open ``_data.bin`` view.

    ``memmap_handle`` is the caller's already-open ``np.memmap`` (or any
    1-D uint8 ndarray view) of the WHOLE ``_data.bin`` file; ``offset``
    is the byte position of one self-describing record within it. The
    record header carries every geometry field a reader needs, so no
    companion length or overlong flag crosses this boundary.

    Pure parsing: no file I/O, no handle lifecycle. Mirrors the
    ``variant_tokens.record.read_record`` pattern so a session can own
    one open handle per bin file and slice many records out of it
    without re-opening per call.

    Returns ``(insn_runlength, block_runlength, tokens)``.
    """
    header, prefix_bytes = parse_binary_header(
        memmap_handle[offset : offset + MAX_HEADER_BYTES]
    )
    total = record_total_size(header)
    data = memmap_handle[offset : offset + total]
    return extract_arrays_from_data(data, header, prefix_bytes)


def read_function_data_memmap(data_path, offset):
    """Read one self-describing record from ``_data.bin`` via ``np.memmap``.

    Thin wrapper that opens the full ``_data.bin`` as a uint8 memmap
    and delegates to :func:`parse_function_data_memmap`; the memmap
    closes when the local reference drops. Use the open-handle form
    directly inside a session that needs many records out of the same
    bin file (avoids per-call mmap overhead).
    """
    data = np.memmap(data_path, dtype=np.uint8, mode="r")
    return parse_function_data_memmap(data, offset)


def parse_function_data_header(data_bytes):
    """Parse one self-describing record from in-memory bytes.

    ``data_bytes`` must start at the record's header byte 0 and span
    the full record (the header tells the parser how much to consume).
    Returns ``(insn_runlength, block_runlength, tokens)``.
    """
    header, prefix_bytes = parse_binary_header(data_bytes)
    return extract_arrays_from_data(data_bytes, header, prefix_bytes)
