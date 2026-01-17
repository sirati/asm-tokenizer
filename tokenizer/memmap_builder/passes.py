import csv
from typing import Dict, List

from ..aligned_data.csv_format import format_unique_called
from ..aligned_data.io import write_function_section_csv, write_index_entry
from .helpers import (
    get_called_functions_from_row,
    process_function_binary_data,
    should_skip_function_for_matched,
    should_skip_function_for_unmatched,
)
from .writers import (
    build_inlining_data_for_unmatched,
    finalize_index_file,
    write_unmatched_function_section,
)


def process_matched_function_pass1(
    func_name: str,
    rows: List,
    version_keys: List,
    mapping_dict: Dict,
    data_file,
) -> dict:
    """Process a matched function in pass 1: write binary data and collect metadata."""
    if func_name.startswith(".L"):
        return None

    if should_skip_function_for_matched(rows):
        return None

    dedup_cache = {}
    all_called_by_vkey = {}

    for vkey, row in zip(version_keys, rows):
        if row is not None:
            all_called_by_vkey[vkey] = get_called_functions_from_row(row)

    unique_called = sorted(set(fn for called_list in all_called_by_vkey.values() for fn in called_list))

    version_data = []
    for vkey, row in zip(version_keys, rows):
        if row is None:
            continue

        called = all_called_by_vkey[vkey]
        mapping = mapping_dict.get(vkey)
        binary_data = process_function_binary_data(row, mapping, data_file, dedup_cache)

        version_data.append(
            {
                "vkey": vkey,
                "called": called,
                "data_offset": binary_data.data_offset,
                "data_len": binary_data.data_len,
                "token_len": binary_data.token_len,
            }
        )

    unique_offsets = set(vdata["data_offset"] for vdata in version_data)
    if len(unique_offsets) == 1:
        return None

    return {
        "func_name": func_name,
        "unique_called": unique_called,
        "version_data": version_data,
    }


def process_unmatched_function_pass1(
    func_name: str,
    rows: List,
    version_keys: List,
    mapping_dict: Dict,
    data_file,
) -> List[dict]:
    """Process an unmatched function in pass 1: write binary data and collect metadata."""
    if func_name.startswith(".L"):
        return []

    unmatched_entries = []
    for vkey, row in zip(version_keys, rows):
        if row is None:
            continue

        if should_skip_function_for_unmatched(row):
            continue

        dedup_cache = {}
        called = get_called_functions_from_row(row)
        mapping = mapping_dict.get(vkey)

        try:
            binary_data = process_function_binary_data(row, mapping, data_file, dedup_cache)

            platform_tuple = (
                vkey.arch,
                vkey.compiler,
                vkey.compilerversion,
                vkey.opt,
            )

            unmatched_entries.append(
                {
                    "func_name": func_name,
                    "vkey": vkey,
                    "data_offset": binary_data.data_offset,
                    "data_len": binary_data.data_len,
                    "token_len": binary_data.token_len,
                    "platform_tuple": platform_tuple,
                    "called": set(called),
                }
            )
        except Exception:
            pass

    return unmatched_entries


def build_function_lookup_table(matched_data_entries: List[dict], unmatched_data_entries: List[dict]) -> dict:
    """Build lookup table: {(func_name, vkey): (offset, length, is_matched)}."""
    function_lookup = {}

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        version_data = entry["version_data"]
        for vdata in version_data:
            vkey = vdata["vkey"]
            function_lookup[(func_name, vkey)] = (
                vdata["data_offset"],
                vdata["data_len"],
                1,
            )

    for entry in unmatched_data_entries:
        func_name = entry["func_name"]
        vkey = entry["vkey"]
        function_lookup[(func_name, vkey)] = (
            entry["data_offset"],
            entry["data_len"],
            0,
        )

    return function_lookup


def write_matched_sections_pass2(
    matched_data_entries: List[dict],
    function_lookup: dict,
    sections_file,
    index_file,
    warn_log,
):
    """Pass 2: Write matched sections CSV with resolved inlining data."""
    writer = csv.writer(sections_file)
    index_entries = []

    for entry in matched_data_entries:
        func_name = entry["func_name"]
        unique_called = entry["unique_called"]
        version_data = entry["version_data"]

        section_start = sections_file.tell()
        unique_called_str = format_unique_called(unique_called)
        writer.writerow([func_name, unique_called_str])

        total_len = 0
        for vdata in version_data:
            vkey = vdata["vkey"]
            called = vdata["called"]
            data_offset = vdata["data_offset"]
            data_len = vdata["data_len"]
            token_len = vdata["token_len"]

            inlining_data = {}
            for called_func in called:
                called_idx = unique_called.index(called_func)
                lookup_key = (called_func, vkey)
                if lookup_key in function_lookup:
                    func_offset, func_len, is_matched = function_lookup[lookup_key]
                    inlining_data[called_idx] = (func_offset, func_len, is_matched)
                else:
                    warn_log.write(
                        f"{func_name},{vkey.arch},{vkey.compiler},{vkey.compilerversion},{vkey.opt},{called_func}\n"
                    )

            inlining_list = [
                [idx, start, length, is_matched] for idx, (start, length, is_matched) in sorted(inlining_data.items())
            ]

            write_function_section_csv(
                writer,
                vkey.arch,
                vkey.compiler,
                vkey.compilerversion,
                vkey.opt,
                inlining_list,
                data_offset,
                data_len,
            )
            total_len += token_len

        avg_len = total_len // len(version_data) if version_data else 0
        index_entries.append((section_start, sections_file.tell() - section_start, avg_len))
        writer.writerow([])

    finalize_index_file(index_file, index_entries, sort_by_avg_len=True)


def group_unmatched_entries_by_function(
    unmatched_data_entries: List[dict],
) -> Dict[str, dict]:
    """Group unmatched entries by function name."""
    unmatched_by_func = {}
    for entry in unmatched_data_entries:
        func_name = entry["func_name"]
        vkey = entry["vkey"]

        if func_name not in unmatched_by_func:
            unmatched_by_func[func_name] = {
                "platform_tuples": [],
                "all_called": set(),
                "version_data_list": [],
                "called_by_version": [],
                "vkeys": [],
            }

        unmatched_by_func[func_name]["platform_tuples"].append(entry["platform_tuple"])
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
):
    """Pass 2: Write unmatched sections CSV with resolved inlining data."""
    writer = csv.writer(sections_file)
    unmatched_by_func = group_unmatched_entries_by_function(unmatched_data_entries)

    for func_name, data in unmatched_by_func.items():
        platform_tuples = data["platform_tuples"]
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
        )

        write_unmatched_function_section(
            writer,
            func_name,
            platform_tuples,
            unique_called_list,
            inlining_data_list,
            first_offset,
            first_len,
        )

        for data_offset, data_len, token_len in version_data_list:
            write_index_entry(index_file, data_offset, data_len, token_len)
