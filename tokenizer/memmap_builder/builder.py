import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from ..aligned_data.match import lockstep_function_match
from .passes import (
    build_function_lookup_table,
    process_matched_function_pass1,
    process_unmatched_function_pass1,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionKey:
    """Represents a unique version of a binary."""

    arch: str
    compiler: str
    compilerversion: str
    opt: str


@dataclass
class BinaryVersionInfo:
    """Information about a specific binary version."""

    path: Path
    mapping_path: Path
    arch: str
    compiler: str
    compilerversion: str
    opt: str


def get_mapping(mapping_path: Path):
    """Load mapping file if it exists."""
    if mapping_path and mapping_path.exists():
        with open(mapping_path, "r", encoding="ascii") as f:
            return base64_to_ndarray_vec(f.read())
    return None


def build_memmap_files(versions: List[BinaryVersionInfo], output_dir: Path, binary_name: str) -> None:
    """Build memory-mapped binary files from aligned CSV data."""

    mapping_dict = {}
    csv_paths = []
    version_keys = []

    for version in versions:
        mapping = get_mapping(version.mapping_path)

        vkey = VersionKey(
            arch=version.arch,
            compiler=version.compiler,
            compilerversion=version.compilerversion,
            opt=version.opt,
        )

        mapping_dict[vkey] = mapping
        csv_paths.append(str(version.path))
        version_keys.append(vkey)

    prefix = binary_name
    unmatched_prefix = f"{binary_name}_unmatched"

    matched_data_entries = []
    unmatched_data_entries = []

    matched_data_file = open(output_dir / f"{prefix}_data.bin", "wb")
    unmatched_data_file = open(output_dir / f"{unmatched_prefix}_data.bin", "wb")

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
            entries = process_unmatched_function_pass1(func_name, rows, version_keys, mapping_dict, unmatched_data_file)
            unmatched_data_entries.extend(entries)

    matched_data_file.close()
    unmatched_data_file.close()

    function_lookup = build_function_lookup_table(matched_data_entries, unmatched_data_entries)

    matched_sections_file = open(output_dir / f"{prefix}_sections.csv", "w", newline="", encoding="ascii")
    matched_index_file = open(output_dir / f"{prefix}_index.bin", "wb")
    warn_log = open(output_dir / f"{binary_name}.warn.log", "w", encoding="ascii")

    write_matched_sections_pass2(
        matched_data_entries,
        function_lookup,
        matched_sections_file,
        matched_index_file,
        warn_log,
    )

    matched_sections_file.close()
    matched_index_file.close()

    unmatched_sections_file = open(
        output_dir / f"{unmatched_prefix}_sections.csv",
        "w",
        newline="",
        encoding="ascii",
    )
    unmatched_index_file = open(output_dir / f"{unmatched_prefix}_index.bin", "wb")

    write_unmatched_sections_pass2(
        unmatched_data_entries,
        function_lookup,
        unmatched_sections_file,
        unmatched_index_file,
        warn_log,
    )

    unmatched_sections_file.close()
    unmatched_index_file.close()
    warn_log.close()
