import csv
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tokenizer.compact_base64_utils import base64_to_ndarray_vec
from tokenizer.memmap_builder.helpers import (
    build_inlining_data,
    collect_unique_called_functions,
    format_inlining_list,
    get_called_functions_from_row,
    process_function_binary_data,
)
from tokenizer.memmap_builder.variants import VariantRegistry
from tokenizer.memmap_builder.writers import finalize_index_file
from tokenizer.vocab_unifier import load_vocab_manager

from .csv_format import format_unique_called
from .io import (
    decode_and_translate_tokens,
    decode_runlengths,
    write_function_binary_data,
    write_function_section_csv,
    write_index_entry,
)
from .match import is_vocab_row, lockstep_function_match

logger = logging.getLogger(__name__)


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


@contextmanager
def _open_versioned_dictreader(csv_path):
    """Open a tokenizer-output CSV as a ``csv.DictReader``, transparently
    handling both v1 and v2 wire formats.

    v2 files prefix the data with a single-cell prelude row
    ``["version=2"]`` (see ``tokenizer/main_loop.py``). v1 files start
    directly with the header row whose first cell is ``function_name``.

    This helper peeks the first row to detect the format, consumes the
    prelude row when present, and yields a ``DictReader`` configured with
    explicit ``fieldnames`` (the next row, which is the actual header) so
    that the version dispatch is fully local to this opener -- callers
    iterate rows as dicts keyed by the file's own column names. The
    column-name difference between v1 (``opaque_metadata``) and v2
    (``metadata``) therefore surfaces in row keys and is handled by
    downstream consumers (e.g. ``get_called_functions_from_row``).
    """
    with open(csv_path, newline="", encoding="ascii") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            # Empty file: yield an empty DictReader-shaped iterator.
            yield iter(())
            return
        if first_row and first_row[0].startswith("version="):
            # v2+: prelude consumed; the next row is the real header.
            header = next(reader, None)
            if header is None:
                yield iter(())
                return
        else:
            # v1: ``first_row`` *is* the header.
            header = first_row
        dict_reader = csv.DictReader(f, fieldnames=header)
        yield dict_reader


def load_function_data(csv_path):
    """Load function data from a csv file."""
    functions = {}
    with _open_versioned_dictreader(csv_path) as reader:
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
    if vocab is None:
        logger.error(f"Failed to load vocabulary from {csv_path}")
        raise RuntimeError(f"Failed to load vocabulary from {csv_path}")
    mapping = get_mapping(mapping_path)
    return vocab, mapping


def get_function_names_across_versions(versions):
    """Return set of function names present in all versions (excluding .L*)."""
    name_sets = []
    for v in versions:
        with _open_versioned_dictreader(v["file"]) as reader:
            names = set()
            for row in reader:
                name = row["function_name"]
                if not name.startswith(".L"):
                    block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
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
        with _open_versioned_dictreader(v["file"]) as reader:
            for row in reader:
                name = row["function_name"]
                if name in function_names:
                    data[key][name] = row
    return data


def get_called_functions(row):
    """Extract called function names from opaque_metadata field."""
    return get_called_functions_from_row(row)


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
    variants,
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

        unique_called = collect_unique_called_functions(all_called_by_vkey, version_keys)
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
            binary_data = process_function_binary_data(row, mapping, file2, dedup_cache)

            inlining_list = build_inlining_data(
                called, unique_called, vkey, function_lookup, warn_log, func_name, variants
            )

            write_function_section_csv(
                writer,
                variants.ref(vkey),
                format_inlining_list(inlining_list),
                binary_data.data_offset,
                binary_data.data_len,
            )
            func_section_len += 1
            total_len += binary_data.token_len

        avg_len = total_len // func_section_len if func_section_len > 0 else 0
        index_entries.append((section_start, file1.tell() - section_start, avg_len))
        writer.writerow([])

    finalize_index_file(file3, index_entries, sort_by_avg_len=True)

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
    variants,
):
    from tokenizer.memmap_builder.writers import (
        build_inlining_data_for_unmatched,
        write_unmatched_function_section,
    )

    prefix = f"{binary_name}_unmatched"
    sections_file = open(f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii")
    data_file = open(f"{out_path}/{prefix}_data.bin", "wb")
    index_file = open(f"{out_path}/{prefix}_index.bin", "wb")
    sections_writer = csv.writer(sections_file)

    for func_name in function_names:
        dedup_cache = {}
        per_func_vkeys = []
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
            binary_data = process_function_binary_data(row, mapping, data_file, dedup_cache)

            per_func_vkeys.append(vkey)
            version_data_list.append((binary_data.data_offset, binary_data.data_len, binary_data.token_len))
            called_by_version.append((compiler_set_id, called))
            compiler_set_id += 1

        if version_data_list:
            unique_called_list = sorted(all_called)
            first_offset, first_len = version_data_list[0][0], version_data_list[0][1]

            inlining_data_list = build_inlining_data_for_unmatched(
                called_by_version,
                unique_called_list,
                version_keys,
                function_lookup,
                warn_log,
                func_name,
                variants,
            )

            variant_refs = [variants.ref(vkey) for vkey in per_func_vkeys]

            write_unmatched_function_section(
                sections_writer,
                func_name,
                variant_refs,
                unique_called_list,
                inlining_data_list,
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


def process_unmatched_too_long(csv_paths, version_keys, mapping_dict, out_path, binary, matched_set):
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
            data_offset, data_len = write_function_binary_data(file1, tokens, block_runlength_arr, insn_runlength)
            write_index_entry(file2, data_offset, data_len, len(tokens))
    for f in files:
        f.close()
    file1.close()
    file2.close()


def export_matched_and_unmatched_sets(binaries, output_path):
    from tokenizer.memmap_builder.passes import (
        build_function_lookup_table,
        process_matched_function_pass1,
        process_unmatched_function_pass1,
        write_matched_sections_pass2,
        write_unmatched_sections_pass2,
    )

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

        matched_data_entries = []
        unmatched_data_entries = []

        matched_data_file = open(f"{out_path}/{prefix}_data.bin", "wb")
        unmatched_data_file = open(f"{out_path}/{unmatched_prefix}_data.bin", "wb")

        for match_data in lockstep_function_match(csv_paths):
            func_name = match_data["function_name"]
            rows = match_data["rows"]
            count = match_data["count"]

            if count >= 2:
                entry = process_matched_function_pass1(func_name, rows, version_keys, mapping_dict, matched_data_file)
                if entry is not None:
                    matched_data_entries.append(entry)
                else:
                    entries = process_unmatched_function_pass1(
                        func_name, rows, version_keys, mapping_dict, unmatched_data_file
                    )
                    unmatched_data_entries.extend(entries)

            elif count == 1:
                entries = process_unmatched_function_pass1(
                    func_name, rows, version_keys, mapping_dict, unmatched_data_file
                )
                unmatched_data_entries.extend(entries)

        matched_data_file.close()
        unmatched_data_file.close()

        function_lookup = build_function_lookup_table(matched_data_entries, unmatched_data_entries)

        # Legacy entry-point: no per-variant metadata, so the registry
        # captures only the canonical-4 axes; ``filename``/``pkg``
        # default to the binary name and ``flags`` is empty. The
        # ``_variants.csv`` sidecar is still emitted so the section
        # CSVs are self-describing through their 0x<hex> refs.
        variants = VariantRegistry.from_vkeys(version_keys, filename=binary, pkg=binary)
        variants.write_sidecar(Path(out_path), binary)

        matched_sections_file = open(f"{out_path}/{prefix}_sections.csv", "w", newline="", encoding="ascii")
        matched_index_file = open(f"{out_path}/{prefix}_index.bin", "wb")
        warn_log = open(f"{out_path}/{binary}.warn.log", "w", encoding="ascii")

        write_matched_sections_pass2(
            matched_data_entries,
            function_lookup,
            matched_sections_file,
            matched_index_file,
            warn_log,
            variants,
        )

        matched_sections_file.close()
        matched_index_file.close()

        unmatched_sections_file = open(
            f"{out_path}/{unmatched_prefix}_sections.csv",
            "w",
            newline="",
            encoding="ascii",
        )
        unmatched_index_file = open(f"{out_path}/{unmatched_prefix}_index.bin", "wb")

        write_unmatched_sections_pass2(
            unmatched_data_entries,
            function_lookup,
            unmatched_sections_file,
            unmatched_index_file,
            warn_log,
            variants,
        )

        unmatched_sections_file.close()
        unmatched_index_file.close()
        warn_log.close()


def run_alignment_export(output_path):
    binaries = collect_binaries(output_path)
    export_matched_and_unmatched_sets(binaries, output_path)


def main(output_path):
    run_alignment_export(output_path)


if __name__ == "__main__":
    main(Path("./out/zlib/").resolve())
