import csv
import json
import os
import re
import struct
from pathlib import Path

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec
from tokenizer.data_loader import load_vocab_manager

from .io import (
    decode_and_translate_tokens,
    decode_runlengths,
    write_function_binary_data,
    write_function_section_csv,
    write_index_entry,
)
from .match import is_vocab_row, lockstep_function_match

# Regex for parsing filenames like: x86-gcc-5-O3_minigzipsh_output.csv
FILENAME_RE = re.compile(
    r"^(?P<arch>x86|x64|arm32|arm64)-(?P<compiler>gcc|clang)-(?P<compilerversion>[\d\.]+)-(?P<opt>O\d)_(?P<binary>.+?)_output\\.csv$"
)


def parse_filename(filename):
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return match.groupdict()


def collect_binaries(output_path):
    """Walk output_path, group files by binary name, collect metadata."""
    binaries = {}
    for file in os.listdir(output_path):
        if not file.endswith("_output.csv"):
            continue
        meta = parse_filename(file)
        if not meta:
            continue
        binary = meta["binary"]
        binaries.setdefault(binary, []).append(
            {
                "file": os.path.join(output_path, file),
                "arch": meta["arch"],
                "compiler": meta["compiler"],
                "compilerversion": meta["compilerversion"],
                "opt": meta["opt"],
            }
        )
    return binaries


def load_function_data(csv_path):
    """Load function data from a csv file."""
    functions = {}
    with open(csv_path, newline="", encoding="ascii") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["function_name"]
            if name.startswith(".L"):
                continue
            block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
            if block_runlength.sum() >= 4096:
                continue
            functions.setdefault(name, []).append(row)
    return functions


# --- Utility functions ---
def get_vocab_and_mapping(csv_path, mapping_path=None):
    from tokenizer.compact_base64_utils import base64_to_ndarray_vec

    vocab = load_vocab_manager(Path(csv_path))
    mapping = None
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="ascii") as f:
            mapping = base64_to_ndarray_vec(f.read())
    return vocab, mapping


def get_function_names_across_versions(versions):
    """Return set of function names present in all versions (excluding .L*)."""
    name_sets = []
    for v in versions:
        with open(v["file"], newline="", encoding="ascii") as f:
            reader = csv.DictReader(f)
            names = set()
            for row in reader:
                name = row["function_name"]
                if not name.startswith(".L"):
                    block_runlength = base64_to_ndarray_vec(
                        row["block_runlength_base64"]
                    )
                    if block_runlength.sum() < 4096:
                        names.add(name)
            name_sets.append(names)
    if not name_sets:
        return set()
    return set.intersection(*name_sets)


def load_all_function_data(versions, function_names):
    """Return dict: {version_tuple: {func_name: row}} for all versions and function_names."""
    data = {}
    for v in versions:
        key = (v["arch"], v["compiler"], v["compilerversion"], v["opt"])
        data[key] = {}
        with open(v["file"], newline="", encoding="ascii") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["function_name"]
                if name in function_names:
                    data[key][name] = row
    return data


def get_called_functions(row):
    """Extract called function names from opaque_metadata field if possible."""
    opaque_metadata = row.get("opaque_metadata", "")
    try:
        # opaque_metadata is a string representation of a list (see low_level.py)
        # e.g. str(repr(meta_result)), where meta_result is a list
        # We expect a list of dicts, each possibly with a 'calls' key
        import ast

        meta = ast.literal_eval(opaque_metadata)
        called = set()
        for entry in meta:
            if isinstance(entry, dict):
                for key in ("calls", "called", "callees", "call_targets"):
                    if key in entry and isinstance(entry[key], list):
                        called.update(str(x) for x in entry[key])
        return sorted(called)
    except Exception:
        return []


def find_inlined_functions(tokens_a, tokens_b):
    # Naive: find all subsequences of tokens_b in tokens_a (not robust, but placeholder)
    # Returns list of (start_idx, length) in tokens_a where tokens_b appears
    from numpy.lib.stride_tricks import sliding_window_view

    if len(tokens_b) == 0 or len(tokens_a) < len(tokens_b):
        return []
    windows = sliding_window_view(tokens_a, len(tokens_b))
    matches = np.where(np.all(windows == tokens_b, axis=1))[0]
    return [(int(idx), len(tokens_b)) for idx in matches]


def compute_avg_function_length(tokens, inlining_map, all_tokens_by_vkey):
    # Sum own length and all inlined function lengths (non-overlapping)
    total = len(tokens)
    for vkey, inlined_list in inlining_map.items():
        for inlined in inlined_list:
            # Use the length of the inlined function in the other version
            other_tokens = all_tokens_by_vkey.get(vkey)
            if other_tokens is not None:
                total += len(other_tokens)
    return total


def write_function_sections(
    out_path,
    function_names,
    version_keys,
    all_data,
    mapping_dict,
    binary_name,
    unmatched=False,
):
    """Write the function sections and binary data files as specified."""
    prefix = f"{binary_name}_unmatched" if unmatched else binary_name
    file1 = open(f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii")
    file2 = open(f"{out_path}/{prefix}_data.bin", "wb")
    file3 = open(f"{out_path}/{prefix}_index.bin", "wb")

    writer = csv.writer(file1)
    index_entries = []
    func_lengths = []

    for func_name in function_names:
        section_start = file1.tell()
        writer.writerow([func_name])
        func_section_len = 0
        func_avg_len = 0
        all_tokens_by_vkey = {}
        inlining_maps_by_vkey = {}
        for vkey in version_keys:
            row = all_data[vkey].get(func_name)
            if not row:
                continue
            arch, compiler, compilerversion, opt = vkey
            called = get_called_functions(row)
            mapping = mapping_dict.get(vkey)
            tokens = decode_and_translate_tokens(row, mapping)
            block_runlength, insn_runlength = decode_runlengths(row)
            data_offset, data_len = write_function_binary_data(
                file2, tokens, block_runlength, insn_runlength
            )
            all_tokens_by_vkey[vkey] = tokens
            inlining_map = {}
            for other_vkey in version_keys:
                if other_vkey == vkey:
                    continue
                other_row = all_data[other_vkey].get(func_name)
                if not other_row:
                    continue
                other_tokens = decode_and_translate_tokens(
                    other_row, mapping_dict.get(other_vkey)
                )
                inlined = find_inlined_functions(tokens, other_tokens)
                if inlined:
                    inlining_map[str(other_vkey)] = [
                        {"start": s, "length": l} for (s, l) in inlined
                    ]
            inlining_maps_by_vkey[vkey] = inlining_map
            write_function_section_csv(
                writer,
                func_name,
                arch,
                compiler,
                compilerversion,
                opt,
                called,
                inlining_map,
                data_offset,
                data_len,
            )
            func_section_len += 1
        # Compute average length including inlined functions (across all vkeys)
        avg_len = 0
        if func_section_len:
            for vkey in version_keys:
                tokens = all_tokens_by_vkey.get(vkey)
                inlining_map = inlining_maps_by_vkey.get(vkey, {})
                if tokens is not None:
                    avg_len += compute_avg_function_length(
                        tokens, inlining_map, all_tokens_by_vkey
                    )
            avg_len //= func_section_len
        index_entries.append((section_start, file1.tell() - section_start, avg_len))
        func_lengths.append(avg_len)
        writer.writerow([])  # Blank line after each function section
    for start, length, avg_len in sorted(index_entries, key=lambda x: x[2]):
        write_index_entry(file3, start, length, avg_len)
    file1.close()
    file2.close()
    file3.close()


def write_unmatched_files(
    out_path, function_names, version_keys, all_data, mapping_dict, binary_name
):
    prefix = f"{binary_name}_unmatched"
    file1 = open(f"{out_path}/{prefix}_data.bin", "wb")
    file2 = open(f"{out_path}/{prefix}_index.bin", "wb")
    for func_name in function_names:
        for vkey in version_keys:
            row = all_data[vkey].get(func_name)
            if not row:
                continue
            mapping = mapping_dict.get(vkey)
            tokens = decode_and_translate_tokens(row, mapping)
            block_runlength, insn_runlength = decode_runlengths(row)
            data_offset, data_len = write_function_binary_data(
                file1, tokens, block_runlength, insn_runlength
            )
            write_index_entry(file2, data_offset, data_len, 0, always_zero=True)
    file1.close()
    file2.close()


# --- Split out helpers and refactor for clarity ---


def get_all_function_names(versions):
    """Return set of all function names (excluding .L*) in all versions."""
    all_names = set()
    for v in versions:
        with open(v["file"], newline="", encoding="ascii") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["function_name"]
                if not name.startswith(".L"):
                    all_names.add(name)
    return all_names


def process_unmatched_too_long(
    csv_paths, version_keys, mapping_dict, out_path, binary, matched_set
):
    """Process unmatched/too-long functions in a streaming, memory-efficient way."""
    prefix = f"{binary}_unmatched"
    file1 = open(f"{out_path}/{prefix}_data.bin", "wb")
    file2 = open(f"{out_path}/{prefix}_index.bin", "wb")
    files = [open(p, newline="", encoding="ascii") for p in csv_paths]
    readers = [csv.reader(f) for f in files]
    headers = [next(reader) for reader in readers]
    for i, reader in enumerate(readers):
        for row in reader:
            if is_vocab_row(row):
                continue
            func_name = row[0]
            if func_name.startswith(".L") or func_name in matched_set:
                continue
            block_runlength = row[3]
            try:
                from tokenizer.compact_base64_utils import base64_to_ndarray_vec

                if base64_to_ndarray_vec(block_runlength).sum() < 4096:
                    continue
            except Exception:
                continue
            mapping = mapping_dict[version_keys[i]]
            row_dict = {k: v for k, v in zip(headers[i], row)}
            tokens = decode_and_translate_tokens(row_dict, mapping)
            block_runlength_arr, insn_runlength = decode_runlengths(row_dict)
            data_offset, data_len = write_function_binary_data(
                file1, tokens, block_runlength_arr, insn_runlength
            )
            write_index_entry(file2, data_offset, data_len, 0, always_zero=True)
    for f in files:
        f.close()
    file1.close()
    file2.close()


def export_matched_and_unmatched_sets(binaries, output_path):
    for binary, versions in binaries.items():
        print(f"Binary: {binary}")
        mapping_dict = {}
        csv_paths = []
        version_keys = []
        for v in versions:
            mapping_path = v["file"].replace("_output.csv", ".mapping.b64c")
            _, mapping = get_vocab_and_mapping(v["file"], mapping_path)
            vkey = (v["arch"], v["compiler"], v["compilerversion"], v["opt"])
            mapping_dict[vkey] = mapping
            csv_paths.append(v["file"])
            version_keys.append(vkey)
        out_path = os.path.dirname(versions[0]["file"])
        # Matched set: process in streaming lockstep
        prefix = binary
        file1 = open(
            f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii"
        )
        file2 = open(f"{out_path}/{prefix}_data.bin", "wb")
        file3 = open(f"{out_path}/{prefix}_index.bin", "wb")
        writer = csv.writer(file1)
        index_entries = []
        func_lengths = []
        for func_name, rows in lockstep_function_match(csv_paths):
            section_start = file1.tell()
            writer.writerow([func_name])
            func_section_len = 0
            func_avg_len = 0
            all_tokens_by_vkey = {}
            inlining_maps_by_vkey = {}
            for vkey, row, header in zip(
                version_keys,
                rows,
                [
                    next(csv.reader(open(p, newline="", encoding="ascii")))
                    for p in csv_paths
                ],
            ):
                row_dict = {k: v for k, v in zip(header, row)}
                arch, compiler, compilerversion, opt = vkey
                called = get_called_functions(row_dict)
                mapping = mapping_dict.get(vkey)
                tokens = decode_and_translate_tokens(row_dict, mapping)
                block_runlength, insn_runlength = decode_runlengths(row_dict)
                data_offset, data_len = write_function_binary_data(
                    file2, tokens, block_runlength, insn_runlength
                )
                all_tokens_by_vkey[vkey] = tokens
                inlining_map = {}
                for other_vkey, other_row, other_header in zip(
                    version_keys,
                    rows,
                    [
                        next(csv.reader(open(p, newline="", encoding="ascii")))
                        for p in csv_paths
                    ],
                ):
                    if other_vkey == vkey:
                        continue
                    other_row_dict = {k: v for k, v in zip(other_header, other_row)}
                    other_tokens = decode_and_translate_tokens(
                        other_row_dict, mapping_dict.get(other_vkey)
                    )
                    inlined = find_inlined_functions(tokens, other_tokens)
                    if inlined:
                        inlining_map[str(other_vkey)] = [
                            {"start": s, "length": l} for (s, l) in inlined
                        ]
                inlining_maps_by_vkey[vkey] = inlining_map
                write_function_section_csv(
                    writer,
                    func_name,
                    arch,
                    compiler,
                    compilerversion,
                    opt,
                    called,
                    inlining_map,
                    data_offset,
                    data_len,
                )
                func_section_len += 1
            avg_len = 0
            if func_section_len:
                for vkey in version_keys:
                    tokens = all_tokens_by_vkey.get(vkey)
                    inlining_map = inlining_maps_by_vkey.get(vkey, {})
                    if tokens is not None:
                        avg_len += compute_avg_function_length(
                            tokens, inlining_map, all_tokens_by_vkey
                        )
                avg_len //= func_section_len
            index_entries.append((section_start, file1.tell() - section_start, avg_len))
            func_lengths.append(avg_len)
            writer.writerow([])
        for start, length, avg_len in sorted(index_entries, key=lambda x: x[2]):
            write_index_entry(file3, start, length, avg_len)
        file1.close()
        file2.close()
        file3.close()
        # Unmatched/too-long: process all files line by line, skipping matched
        matched_set = set()
        for func_name, _ in lockstep_function_match(csv_paths):
            matched_set.add(func_name)
        process_unmatched_too_long(
            csv_paths, version_keys, mapping_dict, out_path, binary, matched_set
        )


def run_alignment_export(output_path):
    binaries = collect_binaries(output_path)
    export_matched_and_unmatched_sets(binaries, output_path)


def main(output_path):
    run_alignment_export(output_path)


if __name__ == "__main__":
    main(Path("../tokenizer").resolve())
