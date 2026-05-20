"""Pass-2 section + index emitters (matched + unmatched).

Pass 1 (``tokenizer.memmap_builder.passes``) writes ``_data.bin``
records and collects per-function metadata; pass 2 emits the section
CSVs and per-arm index files.

Layout responsibilities:

* Matched header rows + unmatched first cell carry the base64 of the
  function's line number in ``<binary>_function_names.txt``; called-
  funcs cells are comma-joined base64 line numbers. Raw function names
  no longer appear in either section CSV.
* Matched variant rows carry a single ``indexer_hex`` cell (8 hex chars
  encoding ``offset >> 4`` into the variant's ``matched_data.bin``
  record). No per-variant entry in any index file. Record geometry is
  recovered from the self-describing record header, so the inline
  encoding no longer carries a length.
* ``matched_index.bin`` is the function-to-CSV-section locator
  (u40 ``csv_offset >> 2`` + u24 ``csv_section_length >> 2``, 8 bytes).
  Both quantities are 4-byte aligned: after every section's blank-row
  terminator the writer pads with raw ``\\n`` bytes until the next
  section start lands on a 4-byte boundary AND ≥ 1 fully empty line of
  separation is guaranteed.
* ``unmatched_index.bin`` is one entry per version's data-bin record
  (16-byte aligned offsets, 4-byte entries) and continues using
  ``write_index_entry``.
* Inlining-data cells keep LOCAL ``called_func_id`` indices.
* All ``csv.writer`` callsites in this module pin
  ``lineterminator='\\n'`` so byte counts (and therefore the
  matched-arm pad arithmetic) are deterministic.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List

from ..aligned_data.csv_section_index import write_csv_section_index_entry
from ..aligned_data.inline_indexer import encode_inline_indexer
from ..aligned_data.io import (
    write_function_section_csv,
    write_index_entry,
    write_unmatched_section_csv,
)
from ..aligned_data.csv_format import (
    format_function_line_no,
    format_function_line_nos_csv,
)
from .function_names import FunctionNamesRegistry
from .variants import VariantRegistry, write_warn_log_entry

# Matched-section CSV alignment:
#
# * Each section must start at a byte offset that is a multiple of
#   ``_SECTION_ALIGN``.
# * Each section must be separated from the previous one by ≥ 1 fully
#   empty line, i.e. ≥ 2 ``\n`` bytes between the last content byte of
#   section N and the first byte of section N+1. ``csv.writer.writerow([])``
#   already emits one ``\n``; this module writes 1-``_SECTION_ALIGN``
#   additional raw ``\n`` bytes after it.
#
# The csv writer is pinned to ``lineterminator='\n'`` so the byte
# arithmetic below is platform-independent.
_SECTION_ALIGN: int = 4
_SECTION_LINE_TERMINATOR: str = "\n"


def _pad_section_to_alignment(sections_file, content_offset: int) -> int:
    """Pad a just-closed matched section to ``_SECTION_ALIGN`` + ≥ 1 empty line.

    Caller invariant: ``writer.writerow([])`` has just emitted one
    ``\n``, so the current file position is at ``end0`` relative to
    ``content_offset``. This helper computes the smallest
    ``next_start >= end0 + 1`` that is a multiple of
    :data:`_SECTION_ALIGN` and writes ``next_start - end0`` raw ``\n``
    characters straight through the underlying text stream (bypassing
    the csv writer for these bytes -- they are inter-section structure,
    not row content). Returns the new section_end (== ``next_start``)
    relative to ``content_offset``.
    """
    end0 = sections_file.tell() - content_offset
    # Smallest 4-multiple strictly > end0. That single bump satisfies
    # BOTH constraints (>= end0 + 1 AND multiple of _SECTION_ALIGN), so
    # pad always lands in 1..._SECTION_ALIGN inclusive. Combined with
    # the two ``\n``s already on disk (last variant row's terminator +
    # ``writerow([])``'s blank-row terminator), the run of ``\n`` bytes
    # after the last non-``\n`` content byte spans
    # ``2 + pad`` chars → 3..(_SECTION_ALIGN + 2) inclusive.
    next_start = ((end0 // _SECTION_ALIGN) + 1) * _SECTION_ALIGN
    pad = next_start - end0
    sections_file.write(_SECTION_LINE_TERMINATOR * pad)
    return sections_file.tell() - content_offset


def write_matched_sections_pass2(
    matched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
    variants: VariantRegistry,
    registry: FunctionNamesRegistry,
    *,
    error_log=None,
):
    """Pass 2: matched sections CSV + pre-v1 ``matched_index.bin``.

    ``registry`` is the FINALISED ``FunctionNamesRegistry``; the caller
    (builder.py) must finalise + write the sidecar BEFORE pass 2.
    ``index_file`` is the pre-v1 layout matched-index handle (no v1
    16-byte prelude, no alignment shift). ``error_log`` is forwarded
    to the section-index writer so per-entry cap overflows are logged
    and the entry skipped.

    The CSV-prelude bytes already written by the caller are excluded
    from every stored ``section_start`` by snapshotting the current
    file position once at entry -- the reader compensates by adding
    its ``content_offset`` back at load time.
    """
    content_offset = sections_file.tell()
    writer = csv.writer(sections_file, lineterminator=_SECTION_LINE_TERMINATOR)
    # (func_name, section_start, section_len) per function in encounter
    # order. The previous avg-len-bucket sort fed
    # ``select_random_function_by_length`` (now a NotImplementedError
    # stub), so the index no longer carries an avg_len column and the
    # writer no longer reorders entries: readers do not depend on entry
    # order matching CSV row order.
    pending_index_entries: List = []

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        unique_called = entry["unique_called"]
        version_data = entry["version_data"]

        section_start = sections_file.tell() - content_offset
        line_no_b64 = format_function_line_no(registry.line_no(func_name))
        called_line_nos_b64 = format_function_line_nos_csv(
            [registry.line_no(name) for name in unique_called]
        )
        writer.writerow([line_no_b64, called_line_nos_b64])

        for vdata in version_data:
            vkey = vdata["vkey"]
            called = vdata["called"]
            data_offset = vdata["data_offset"]

            variant_ref = variants.ref(vkey)
            inlining_data = {}
            for called_func in called:
                called_idx = unique_called.index(called_func)
                lookup_key = (called_func, vkey)
                if lookup_key in function_lookup:
                    func_offset, func_len, is_matched = function_lookup[lookup_key]
                    inlining_data[called_idx] = (func_offset, func_len, is_matched)
                else:
                    write_warn_log_entry(warn_log, func_name, variant_ref, called_func)

            inlining_list = [
                [idx, start, length, is_matched]
                for idx, (start, length, is_matched) in sorted(inlining_data.items())
            ]

            indexer_hex = encode_inline_indexer(data_offset)
            write_function_section_csv(writer, variant_ref, inlining_list, indexer_hex)

        writer.writerow([])
        section_end = _pad_section_to_alignment(sections_file, content_offset)
        pending_index_entries.append(
            (func_name, section_start, section_end - section_start)
        )

    for func_name, section_start, section_len in pending_index_entries:
        write_csv_section_index_entry(
            index_file,
            csv_offset=section_start,
            csv_section_length=section_len,
            func_name=func_name,
            error_log=error_log,
        )


def group_unmatched_entries_by_function(
    unmatched_data_entries: List[dict],
) -> Dict[str, dict]:
    """Group unmatched entries by function name.

    Per-function aggregator: collects the vkeys in encounter order
    (their list position is the ``comp_set_id`` referenced from
    ``called_by_version`` and the inlining-data merge). The pass-2
    writer resolves vkeys to ``0x<hex>`` variant refs via the
    ``VariantRegistry``; this stage stays unaware of the on-disk
    encoding.
    """
    unmatched_by_func = {}
    for entry in unmatched_data_entries:
        func_name = entry["func_name"]
        vkey = entry["vkey"]

        if func_name not in unmatched_by_func:
            unmatched_by_func[func_name] = {
                "all_called": set(),
                "version_data_list": [],
                "called_by_version": [],
                "vkeys": [],
            }

        unmatched_by_func[func_name]["all_called"].update(entry["called"])
        unmatched_by_func[func_name]["version_data_list"].append(
            (entry["data_offset"], entry["data_len"], entry["token_len"])
        )
        comp_set_id = len(unmatched_by_func[func_name]["vkeys"])
        unmatched_by_func[func_name]["vkeys"].append(vkey)
        unmatched_by_func[func_name]["called_by_version"].append((comp_set_id, entry["called"]))

    return unmatched_by_func


def _build_unmatched_inlining_data_list(
    called_by_version,
    unique_called_list: List[str],
    vkeys: List,
    function_lookup: dict,
    warn_log,
    func_name: str,
    variants: VariantRegistry,
) -> List:
    """Resolve each unmatched call site into the on-disk inlining tuple."""
    inlining_data_list = []
    for comp_set_id, called_funcs in called_by_version:
        for called_func in called_funcs:
            called_func_id = unique_called_list.index(called_func)
            lookup_key = (called_func, vkeys[comp_set_id])
            if lookup_key in function_lookup:
                func_offset, func_len, is_matched = function_lookup[lookup_key]
                inlining_data_list.append(
                    [called_func_id, comp_set_id, func_offset, func_len, is_matched]
                )
            else:
                write_warn_log_entry(
                    warn_log, func_name, variants.ref(vkeys[comp_set_id]), called_func
                )
    return inlining_data_list


def _format_unmatched_inlining_str(inlining_data_list: List) -> str:
    """Merge per-version tuples + format the on-disk inlining-data cell.

    ``called_func_id`` is a LOCAL index into the section's
    ``unique_called`` list (NOT a function name). Duplicate
    ``(called_func_id, offset, length, is_matched)`` tuples appearing
    across multiple versions collapse into one entry whose
    ``comp_set_id`` field underscore-joins the contributing version IDs.
    """
    grouped = defaultdict(list)
    for called_func_id, comp_set_id, offset, length, is_matched in inlining_data_list:
        grouped[(called_func_id, offset, length, is_matched)].append(comp_set_id)

    merged_entries = []
    for (called_func_id, offset, length, is_matched), comp_set_ids in sorted(grouped.items()):
        comp_set_str = "_".join(map(str, sorted(comp_set_ids)))
        merged_entries.append(
            f"{called_func_id}-{comp_set_str},{offset:x},{length:x},{is_matched}"
        )
    return ";".join(merged_entries)


def write_unmatched_sections_pass2(
    unmatched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
    variants: VariantRegistry,
    registry: FunctionNamesRegistry,
    *,
    error_log=None,
):
    """Pass 2: unmatched sections CSV + v1 ``unmatched_index.bin``.

    ``registry`` provides line numbers for the first-cell + called-funcs
    cell base64 indirection. Inlining-data cells keep LOCAL
    ``called_func_id`` indices unchanged.

    ``index_file`` is the v1 unmatched-index handle; one entry per
    version's data-bin record (16-byte aligned, 4-byte u32 ``offset >> 4``
    entries -- record geometry comes from the self-describing record
    header, no length / sentinel / overlong machinery). The matched-arm
    restructuring does NOT touch this path. ``error_log`` is forwarded
    so per-version cap overflows are logged and the entry skipped (no
    abort).

    No CSV-byte alignment is required on this arm -- it has no
    section-locator index that would benefit from a shifted offset --
    so the section writer just emits rows back-to-back with the pinned
    ``\\n`` line terminator.
    """
    writer = csv.writer(sections_file, lineterminator=_SECTION_LINE_TERMINATOR)
    unmatched_by_func = group_unmatched_entries_by_function(unmatched_data_entries)

    for func_name, data in unmatched_by_func.items():
        all_called = data["all_called"]
        version_data_list = data["version_data_list"]
        called_by_version = data["called_by_version"]
        vkeys = data["vkeys"]

        if not version_data_list:
            continue

        unique_called_list = sorted(all_called)
        first_offset = version_data_list[0][0]

        inlining_data_list = _build_unmatched_inlining_data_list(
            called_by_version,
            unique_called_list,
            vkeys,
            function_lookup,
            warn_log,
            func_name,
            variants,
        )

        variant_refs = [variants.ref(vkey) for vkey in vkeys]
        line_no_b64 = format_function_line_no(registry.line_no(func_name))
        called_line_nos_b64 = format_function_line_nos_csv(
            [registry.line_no(name) for name in unique_called_list]
        )
        inlining_data_str = _format_unmatched_inlining_str(inlining_data_list)
        indexer_hex = encode_inline_indexer(first_offset)

        write_unmatched_section_csv(
            writer,
            line_no_b64,
            variant_refs,
            called_line_nos_b64,
            inlining_data_str,
            indexer_hex,
        )

        for data_offset, _data_len, _token_len in version_data_list:
            write_index_entry(
                index_file,
                data_offset,
                func_name=func_name,
                error_log=error_log,
            )
