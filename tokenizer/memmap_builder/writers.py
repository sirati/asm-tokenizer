from typing import List, Tuple

from ..aligned_data.csv_format import format_unique_called
from ..aligned_data.io import (
    write_function_section_csv,
    write_index_entry,
    write_unmatched_section_csv,
)
from .variants import VariantRegistry, write_warn_log_entry


def write_matched_function_section(
    writer,
    func_name: str,
    unique_called: List[str],
    version_data_list: List[dict],
    function_lookup: dict,
    warn_log,
    variants: VariantRegistry,
) -> Tuple[int, int]:
    """
    Write a complete matched function section (header + version rows).
    Returns (section_start, section_length) for index.
    """
    unique_called_str = format_unique_called(unique_called)
    writer.writerow([func_name, unique_called_str])

    total_len = 0
    for vdata in version_data_list:
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
            [idx, start, length, is_matched] for idx, (start, length, is_matched) in sorted(inlining_data.items())
        ]

        write_function_section_csv(
            writer,
            variant_ref,
            inlining_list,
            data_offset,
            data_len,
        )
        total_len += token_len

    writer.writerow([])
    return total_len


def write_unmatched_function_section(
    writer,
    func_name: str,
    variant_refs: List[str],
    unique_called_list: List[str],
    inlining_data_list: List,
    first_offset: int,
    first_len: int,
):
    """Write a single-line unmatched function section.

    ``variant_refs`` is the ordered list of ``0x<hex>`` refs (one per
    version present for this unmatched function), in the same order
    as the inlining data's ``comp_set_id`` references it.
    """
    called_str = format_unique_called(unique_called_list)

    from collections import defaultdict

    grouped = defaultdict(list)
    for called_func_id, comp_set_id, offset, length, is_matched in inlining_data_list:
        key = (called_func_id, offset, length, is_matched)
        grouped[key].append(comp_set_id)

    merged_entries = []
    for (called_func_id, offset, length, is_matched), comp_set_ids in sorted(grouped.items()):
        comp_set_str = "_".join(map(str, sorted(comp_set_ids)))
        merged_entries.append(f"{called_func_id}-{comp_set_str},{offset:x},{length:x},{is_matched}")

    inlining_data_str = ";".join(merged_entries)

    write_unmatched_section_csv(
        writer,
        func_name,
        variant_refs,
        called_str,
        inlining_data_str,
        first_offset,
        first_len,
    )


def finalize_index_file(index_file, index_entries: List[Tuple[int, int, int]], sort_by_avg_len: bool = True):
    """Write sorted index entries to index file."""
    sorted_entries = sorted(index_entries, key=lambda x: x[2]) if sort_by_avg_len else index_entries
    for start, length, avg_len in sorted_entries:
        write_index_entry(index_file, start, length, avg_len)


def build_inlining_data_for_unmatched(
    called_by_version: List[Tuple[int, set]],
    unique_called_list: List[str],
    vkeys: List,
    function_lookup: dict,
    warn_log,
    func_name: str,
    variants: VariantRegistry,
) -> List:
    """Build inlining data list for unmatched functions with compiler_set_id-called_func_id format."""
    inlining_data_list = []
    for comp_set_id, called_funcs in called_by_version:
        for called_func in called_funcs:
            called_func_id = unique_called_list.index(called_func)
            lookup_key = (called_func, vkeys[comp_set_id])
            if lookup_key in function_lookup:
                func_offset, func_len, is_matched = function_lookup[lookup_key]
                inlining_data_list.append(
                    [
                        called_func_id,
                        comp_set_id,
                        func_offset,
                        func_len,
                        is_matched,
                    ]
                )
            else:
                write_warn_log_entry(
                    warn_log, func_name, variants.ref(vkeys[comp_set_id]), called_func
                )
    return inlining_data_list
