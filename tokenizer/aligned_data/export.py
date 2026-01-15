import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

from tokenizer.compact_base64_utils import base64_to_ndarray_vec
from tokenizer.data_loader import load_vocab_manager

from .io import (
    decode_and_translate_tokens,
    decode_runlengths,
    format_unique_called,
    write_function_binary_data,
    write_function_section_csv,
    write_index_entry,
    write_unmatched_section_csv,
)
from .match import is_vocab_row, lockstep_function_match


@dataclass(frozen=True)
class VersionKey:
    """Represents a unique version of a binary."""

    arch: str
    compiler: str
    compilerversion: str
    opt: str


# Regex for parsing filenames like: x86-gcc-5-O3_minigzipsh_output.csv
FILENAME_RE = re.compile(
    r"^(?P<arch>x86|x64|arm32|arm64)-(?P<compiler>gcc|clang)-(?P<compilerversion>[\d\.]+)-(?P<opt>O\d)_(?P<binary>.+?)_output\.csv$"
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
def get_mapping(mapping_path):
    from tokenizer.compact_base64_utils import base64_to_ndarray_vec

    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="ascii") as f:
            return base64_to_ndarray_vec(f.read())
    return None


def get_vocab_and_mapping(csv_path, mapping_path=None):
    vocab = load_vocab_manager(Path(csv_path))
    mapping = get_mapping(mapping_path)
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
    """Extract called function names from opaque_metadata field."""
    opaque_metadata = row.get("opaque_metadata", "")
    try:
        import ast

        meta = ast.literal_eval(opaque_metadata)
        called = set()
        for entry in meta:
            if isinstance(entry, tuple) and len(entry) >= 5:
                name = entry[2]
                type_field = entry[3]
                if type_field == "local_function":
                    called.add(name)
        return sorted(called)
    except Exception:
        return []


def compute_avg_function_length(tokens):
    """Compute average function length."""
    return len(tokens)


def write_function_sections(
    out_path,
    function_names,
    version_keys,
    all_data,
    mapping_dict,
    binary_name,
    function_lookup,
    warn_log,
    unmatched=False,
):
    """Write the function sections and binary data files as specified."""
    prefix = f"{binary_name}_unmatched" if unmatched else binary_name
    file1 = open(f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii")
    file2 = open(f"{out_path}/{prefix}_data.bin", "wb")
    file3 = open(f"{out_path}/{prefix}_index.bin", "wb")

    writer = csv.writer(file1)
    index_entries = []

    for func_name in function_names:
        dedup_cache = {}
        section_start = file1.tell()

        all_called_by_vkey = {}
        for vkey in version_keys:
            row = all_data[vkey].get(func_name)
            if row:
                all_called_by_vkey[vkey] = get_called_functions(row)

        unique_called = sorted(
            set(fn for called_list in all_called_by_vkey.values() for fn in called_list)
        )
        unique_called_str = format_unique_called(unique_called)
        writer.writerow([func_name, unique_called_str])

        func_section_len = 0
        total_len = 0

        for vkey in version_keys:
            row = all_data[vkey].get(func_name)
            if not row:
                continue

            called = all_called_by_vkey[vkey]

            mapping = mapping_dict.get(vkey)
            tokens = decode_and_translate_tokens(row, mapping)
            block_runlength, insn_runlength = decode_runlengths(row)
            data_offset, data_len = write_function_binary_data(
                file2, tokens, block_runlength, insn_runlength, dedup_cache
            )

            inlining_data = {}
            for called_func in called:
                called_idx = unique_called.index(called_func)
                lookup_key = (called_func, vkey)
                if lookup_key in function_lookup:
                    func_offset, func_len, is_matched = function_lookup[lookup_key]
                    inlining_data[called_idx] = (func_offset, func_len, is_matched)
                else:
                    warn_log.write(
                        f"unknown local func called: {func_name},{vkey.arch},{vkey.compiler},{vkey.compilerversion},{vkey.opt},{called_func}\n"
                    )

            inlining_list = [
                [idx, start, length, is_matched]
                for idx, (start, length, is_matched) in sorted(inlining_data.items())
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
            func_section_len += 1
            total_len += len(tokens)

        avg_len = total_len // func_section_len if func_section_len > 0 else 0
        index_entries.append((section_start, file1.tell() - section_start, avg_len))
        writer.writerow([])

    for start, length, avg_len in sorted(index_entries, key=lambda x: x[2]):
        write_index_entry(file3, start, length, avg_len)

    file1.close()
    file2.close()
    file3.close()


def write_unmatched_files(
    out_path,
    function_names,
    version_keys,
    all_data,
    mapping_dict,
    binary_name,
    function_lookup,
    warn_log,
):
    prefix = f"{binary_name}_unmatched"
    sections_file = open(
        f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii"
    )
    data_file = open(f"{out_path}/{prefix}_data.bin", "wb")
    index_file = open(f"{out_path}/{prefix}_index.bin", "wb")
    sections_writer = csv.writer(sections_file)

    for func_name in function_names:
        dedup_cache = {}
        platform_tuples = []
        all_called = set()
        version_data_list = []
        called_by_version = []

        compiler_set_id = 0
        for vkey in version_keys:
            row = all_data[vkey].get(func_name)
            if not row:
                continue

            called = get_called_functions(row)
            all_called.update(called)

            mapping = mapping_dict.get(vkey)
            tokens = decode_and_translate_tokens(row, mapping)
            block_runlength, insn_runlength = decode_runlengths(row)
            data_offset, data_len = write_function_binary_data(
                data_file, tokens, block_runlength, insn_runlength, dedup_cache
            )

            platform_tuples.append(
                (vkey.arch, vkey.compiler, vkey.compilerversion, vkey.opt)
            )
            version_data_list.append((data_offset, data_len, len(tokens)))
            called_by_version.append((compiler_set_id, called))
            compiler_set_id += 1

        if version_data_list:
            unique_called_list = sorted(all_called)
            called_str = format_unique_called(unique_called_list)
            first_offset, first_len = version_data_list[0][0], version_data_list[0][1]

            inlining_data_list = []
            for comp_set_id, called_funcs in called_by_version:
                for called_func in called_funcs:
                    called_func_id = unique_called_list.index(called_func)
                    lookup_key = (called_func, version_keys[comp_set_id])
                    if lookup_key in function_lookup:
                        func_offset, func_len, is_matched = function_lookup[lookup_key]
                        inlining_data_list.append(
                            [
                                f"{comp_set_id}-{called_func_id}",
                                func_offset,
                                func_len,
                                is_matched,
                            ]
                        )
                    else:
                        vkey = version_keys[comp_set_id]
                        warn_log.write(
                            f"{func_name},{vkey.arch},{vkey.compiler},{vkey.compilerversion},{vkey.opt},{called_func}\n"
                        )

            from .io import write_unmatched_section_csv

            inlining_data_str = ";".join(
                f"{idx},{offset:x},{length:x},{is_matched}"
                for idx, offset, length, is_matched in inlining_data_list
            )

            write_unmatched_section_csv(
                sections_writer,
                func_name,
                platform_tuples,
                called_str,
                inlining_data_str,
                first_offset,
                first_len,
            )

            for data_offset, data_len, token_len in version_data_list:
                write_index_entry(index_file, data_offset, data_len, token_len)

    sections_file.close()
    data_file.close()
    index_file.close()


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
            write_index_entry(file2, data_offset, data_len, len(tokens))
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
            mapping = get_mapping(mapping_path)
            vkey = VersionKey(
                arch=v["arch"],
                compiler=v["compiler"],
                compilerversion=v["compilerversion"],
                opt=v["opt"],
            )
            mapping_dict[vkey] = mapping
            csv_paths.append(v["file"])
            version_keys.append(vkey)
        out_path = os.path.dirname(versions[0]["file"])

        prefix = binary
        unmatched_prefix = f"{binary}_unmatched"

        # Pass 1: Write all binary data and build lookup tables
        matched_data_entries = []
        unmatched_data_entries = []
        matched_functions = set()

        matched_data_file = open(f"{out_path}/{prefix}_data.bin", "wb")
        unmatched_data_file = open(f"{out_path}/{unmatched_prefix}_data.bin", "wb")

        for match_data in lockstep_function_match(csv_paths):
            func_name = match_data["function_name"]
            rows = match_data["rows"]
            count = match_data["count"]

            if count >= 2:
                if func_name.startswith(".L"):
                    continue
                try:
                    from tokenizer.compact_base64_utils import base64_to_ndarray_vec

                    passes_filter = True
                    for row in rows:
                        if row is not None and "block_runlength" in row:
                            if (
                                base64_to_ndarray_vec(row["block_runlength"]).sum()
                                >= 4096
                            ):
                                passes_filter = False
                                break

                    if not passes_filter:
                        continue
                except Exception:
                    pass

                matched_functions.add(func_name)
                matched_dedup_cache = {}

                all_called_by_vkey = {}
                for vkey, row in zip(version_keys, rows):
                    if row is not None:
                        all_called_by_vkey[vkey] = get_called_functions(row)

                unique_called = sorted(
                    set(
                        fn
                        for called_list in all_called_by_vkey.values()
                        for fn in called_list
                    )
                )

                version_data = []
                for vkey, row in zip(version_keys, rows):
                    if row is None:
                        continue

                    called = all_called_by_vkey[vkey]

                    mapping = mapping_dict.get(vkey)
                    tokens = decode_and_translate_tokens(row, mapping)
                    block_runlength, insn_runlength = decode_runlengths(row)
                    data_offset, data_len = write_function_binary_data(
                        matched_data_file,
                        tokens,
                        block_runlength,
                        insn_runlength,
                        matched_dedup_cache,
                    )

                    version_data.append(
                        {
                            "vkey": vkey,
                            "called": called,
                            "data_offset": data_offset,
                            "data_len": data_len,
                            "token_len": len(tokens),
                        }
                    )

                matched_data_entries.append(
                    {
                        "func_name": func_name,
                        "unique_called": unique_called,
                        "version_data": version_data,
                    }
                )

            elif count == 1:
                if func_name.startswith(".L"):
                    continue
                unmatched_dedup_cache = {}
                platform_tuples = []
                all_called = set()
                version_data_list = []

                for vkey, row in zip(version_keys, rows):
                    if row is None:
                        continue
                    try:
                        from tokenizer.compact_base64_utils import base64_to_ndarray_vec

                        if "block_runlength" in row:
                            if (  # todo I do not understand why we have to do this here, i believe we want all functoins including longer ones in the unmatched data
                                base64_to_ndarray_vec(row["block_runlength"]).sum()
                                < 4096
                            ):
                                continue

                        called = get_called_functions(row)
                        all_called.update(called)

                        mapping = mapping_dict.get(vkey)
                        tokens = decode_and_translate_tokens(row, mapping)
                        block_runlength, insn_runlength = decode_runlengths(row)
                        data_offset, data_len = write_function_binary_data(
                            unmatched_data_file,
                            tokens,
                            block_runlength,
                            insn_runlength,
                            unmatched_dedup_cache,
                        )

                        platform_tuples.append(
                            (vkey.arch, vkey.compiler, vkey.compilerversion, vkey.opt)
                        )
                        version_data_list.append((data_offset, data_len, len(tokens)))

                        unmatched_data_entries.append(
                            {
                                "func_name": func_name,
                                "vkey": vkey,
                                "data_offset": data_offset,
                                "data_len": data_len,
                                "token_len": len(tokens),
                                "platform_tuples": platform_tuples,
                                "called": all_called,
                                "version_data_list": version_data_list,
                            }
                        )
                    except Exception:
                        pass

        matched_data_file.close()
        unmatched_data_file.close()

        # Build lookup table: {(func_name, vkey): (offset, length, is_matched)}
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

        # Pass 2: Write CSV files with actual values
        matched_file1 = open(
            f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii"
        )
        matched_file3 = open(f"{out_path}/{prefix}_index.bin", "wb")
        warn_log = open(f"{out_path}/{binary}.warn.log", "w", encoding="ascii")
        matched_writer = csv.writer(matched_file1)
        matched_index_entries = []

        for entry in matched_data_entries:
            func_name = entry["func_name"]
            unique_called = entry["unique_called"]
            version_data = entry["version_data"]

            section_start = matched_file1.tell()
            unique_called_str = format_unique_called(unique_called)
            matched_writer.writerow([func_name, unique_called_str])

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
                    [idx, start, length, is_matched]
                    for idx, (start, length, is_matched) in sorted(
                        inlining_data.items()
                    )
                ]

                write_function_section_csv(
                    matched_writer,
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
            matched_index_entries.append(
                (section_start, matched_file1.tell() - section_start, avg_len)
            )
            matched_writer.writerow([])

        for start, length, avg_len in sorted(matched_index_entries, key=lambda x: x[2]):
            write_index_entry(matched_file3, start, length, avg_len)

        matched_file1.close()
        matched_file3.close()
        warn_log.close()

        # Write unmatched sections and index
        unmatched_sections_file = open(
            f"{out_path}/{unmatched_prefix}_sections.csv",
            "w",
            newline="",
            encoding="ascii",
        )
        unmatched_file2 = open(f"{out_path}/{unmatched_prefix}_index.bin", "wb")
        unmatched_writer = csv.writer(unmatched_sections_file)

        # Group unmatched entries by function name
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
            unmatched_by_func[func_name]["platform_tuples"].extend(
                entry.get("platform_tuples", [])
            )
            unmatched_by_func[func_name]["all_called"].update(
                entry.get("called", set())
            )
            unmatched_by_func[func_name]["version_data_list"].extend(
                entry.get("version_data_list", [])
            )
            comp_set_id = len(unmatched_by_func[func_name]["vkeys"])
            unmatched_by_func[func_name]["vkeys"].append(vkey)
            unmatched_by_func[func_name]["called_by_version"].append(
                (comp_set_id, entry.get("called", set()))
            )

        for func_name, data in unmatched_by_func.items():
            platform_tuples = data["platform_tuples"]
            all_called = data["all_called"]
            version_data_list = data["version_data_list"]
            called_by_version = data["called_by_version"]
            vkeys = data["vkeys"]

            if version_data_list:
                unique_called_list = sorted(all_called)
                called_str = format_unique_called(unique_called_list)
                first_offset, first_len = (
                    version_data_list[0][0],
                    version_data_list[0][1],
                )

                inlining_data_list = []
                for comp_set_id, called_funcs in called_by_version:
                    for called_func in called_funcs:
                        called_func_id = unique_called_list.index(called_func)
                        lookup_key = (called_func, vkeys[comp_set_id])
                        if lookup_key in function_lookup:
                            func_offset, func_len, is_matched = function_lookup[
                                lookup_key
                            ]
                            inlining_data_list.append(
                                [
                                    f"{comp_set_id}-{called_func_id}",
                                    func_offset,
                                    func_len,
                                    is_matched,
                                ]
                            )
                        else:
                            vkey = vkeys[comp_set_id]
                            warn_log.write(
                                f"{func_name},{vkey.arch},{vkey.compiler},{vkey.compilerversion},{vkey.opt},{called_func}\n"
                            )

                inlining_data_str = ";".join(
                    f"{idx},{offset:x},{length:x},{is_matched}"
                    for idx, offset, length, is_matched in inlining_data_list
                )

                write_unmatched_section_csv(
                    unmatched_writer,
                    func_name,
                    platform_tuples,
                    called_str,
                    inlining_data_str,
                    first_offset,
                    first_len,
                )

                for data_offset, data_len, token_len in version_data_list:
                    write_index_entry(unmatched_file2, data_offset, data_len, token_len)

        unmatched_sections_file.close()
        unmatched_file2.close()


def run_alignment_export(output_path):
    binaries = collect_binaries(output_path)
    export_matched_and_unmatched_sets(binaries, output_path)


def main(output_path):
    run_alignment_export(output_path)


if __name__ == "__main__":
    main(Path("./out/zlib/").resolve())
