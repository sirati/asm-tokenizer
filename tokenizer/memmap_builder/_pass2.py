"""Pass-2 section + index emitters (matched + unmatched).

Pass 1 (in :mod:`tokenizer.memmap_builder.passes`) walks the per-binary
CSVs, writes ``_data.bin`` records, and collects metadata into per-
function dicts. Pass 2 reads those dicts back, emits section CSV rows
and ``_index.bin`` entries, and threads the ``<binary>.error.log``
handle to ``write_index_entry`` so cap-overflows skip the entry rather
than aborting the build.

Split from ``passes.py`` to keep both files under the 300 LOC cap.
"""

from __future__ import annotations

import csv
from typing import Dict, List

from ..aligned_data.csv_format import format_unique_called
from ..aligned_data.io import write_function_section_csv, write_index_entry
from .variants import VariantRegistry, write_warn_log_entry
from .writers import (
    build_inlining_data_for_unmatched,
    finalize_index_file,
    write_unmatched_function_section,
)


def write_matched_sections_pass2(
    matched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
    variants: VariantRegistry,
    *,
    error_log=None,
):
    """Pass 2: Write matched sections CSV with resolved inlining data.

    ``error_log`` is forwarded to the index-entry writer so a per-entry
    cap overflow gets logged and the entry is skipped from
    ``_index.bin``.

    The CSV-prelude bytes already written by the caller (builder.py)
    are excluded from every stored ``section_start`` by snapshotting
    the current file position once at entry — the reader compensates
    by adding its ``content_offset`` back at load time, so the index
    stays prelude-agnostic.
    """
    content_offset = sections_file.tell()
    writer = csv.writer(sections_file)
    index_entries = []

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        unique_called = entry["unique_called"]
        version_data = entry["version_data"]

        section_start = sections_file.tell() - content_offset
        unique_called_str = format_unique_called(unique_called)
        writer.writerow([func_name, unique_called_str])

        total_len = 0
        for vdata in version_data:
            vkey = vdata["vkey"]
            called = vdata["called"]
            data_offset = vdata["data_offset"]
            data_len = vdata["data_len"]
            token_len = vdata["token_len"]

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

            write_function_section_csv(
                writer,
                variant_ref,
                inlining_list,
                data_offset,
                data_len,
            )
            total_len += token_len

        avg_len = total_len // len(version_data) if version_data else 0
        section_end = sections_file.tell() - content_offset
        index_entries.append(
            (func_name, section_start, section_end - section_start, avg_len)
        )
        writer.writerow([])

    finalize_index_file(
        index_file, index_entries, sort_by_avg_len=True, error_log=error_log
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


def write_unmatched_sections_pass2(
    unmatched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
    variants: VariantRegistry,
    *,
    error_log=None,
):
    """Pass 2: Write unmatched sections CSV with resolved inlining data.

    ``error_log`` is forwarded to ``write_index_entry`` so per-version
    cap overflows get logged + skipped (no index entry written) rather
    than raising. ``func_name`` is forwarded so the skipped entry is
    traceable in the log.
    """
    writer = csv.writer(sections_file)
    unmatched_by_func = group_unmatched_entries_by_function(unmatched_data_entries)

    for func_name, data in unmatched_by_func.items():
        all_called = data["all_called"]
        version_data_list = data["version_data_list"]
        called_by_version = data["called_by_version"]
        vkeys = data["vkeys"]

        if not version_data_list:
            continue

        unique_called_list = sorted(all_called)
        first_offset, first_len = version_data_list[0][0], version_data_list[0][1]

        inlining_data_list = build_inlining_data_for_unmatched(
            called_by_version,
            unique_called_list,
            vkeys,
            function_lookup,
            warn_log,
            func_name,
            variants,
        )

        variant_refs = [variants.ref(vkey) for vkey in vkeys]

        write_unmatched_function_section(
            writer,
            func_name,
            variant_refs,
            unique_called_list,
            inlining_data_list,
            first_offset,
            first_len,
        )

        for data_offset, data_len, token_len in version_data_list:
            write_index_entry(
                index_file,
                data_offset,
                data_len,
                token_len,
                func_name=func_name,
                error_log=error_log,
            )
