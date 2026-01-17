import logging
import sys
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

    logger.info(f"  Output directory: {output_dir}")

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

    matched_data_path = output_dir / f"{prefix}_data.bin"
    unmatched_data_path = output_dir / f"{unmatched_prefix}_data.bin"
    logger.info(f"  Creating: {matched_data_path}")
    logger.info(f"  Creating: {unmatched_data_path}")
    matched_data_file = open(matched_data_path, "wb")
    unmatched_data_file = open(unmatched_data_path, "wb")

    progress_callback = None
    pbar = None
    if sys.stdout.isatty():
        try:
            from tqdm import tqdm

            total_size = sum(Path(csv_path).stat().st_size for csv_path in csv_paths)
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Processing {binary_name}", leave=False)
            progress_callback = pbar.update
            last_bytes = [0]

            def progress_wrapper(current_bytes):
                delta = current_bytes - last_bytes[0]
                last_bytes[0] = current_bytes
                pbar.update(delta)

            progress_callback = progress_wrapper
        except ImportError:
            pass

    for match_data in lockstep_function_match(csv_paths, progress_callback):
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

    if pbar is not None:
        pbar.close()

    matched_data_file.close()
    logger.info(f"  Closed: {matched_data_path}")
    unmatched_data_file.close()
    logger.info(f"  Closed: {unmatched_data_path}")

    function_lookup = build_function_lookup_table(matched_data_entries, unmatched_data_entries)

    matched_sections_path = output_dir / f"{prefix}_sections.csv"
    matched_index_path = output_dir / f"{prefix}_index.bin"
    warn_log_path = output_dir / f"{binary_name}.warn.log"
    logger.info(f"  Creating: {matched_sections_path}")
    logger.info(f"  Creating: {matched_index_path}")
    logger.info(f"  Creating: {warn_log_path}")
    matched_sections_file = open(matched_sections_path, "w", newline="", encoding="ascii")
    matched_index_file = open(matched_index_path, "wb")
    warn_log = open(warn_log_path, "w", encoding="ascii")

    write_matched_sections_pass2(
        matched_data_entries,
        function_lookup,
        matched_sections_file,
        matched_index_file,
        warn_log,
    )

    matched_sections_file.close()
    logger.info(f"  Closed: {matched_sections_path}")
    matched_index_file.close()
    logger.info(f"  Closed: {matched_index_path}")

    unmatched_sections_path = output_dir / f"{unmatched_prefix}_sections.csv"
    unmatched_index_path = output_dir / f"{unmatched_prefix}_index.bin"
    logger.info(f"  Creating: {unmatched_sections_path}")
    logger.info(f"  Creating: {unmatched_index_path}")
    unmatched_sections_file = open(
        unmatched_sections_path,
        "w",
        newline="",
        encoding="ascii",
    )
    unmatched_index_file = open(unmatched_index_path, "wb")

    write_unmatched_sections_pass2(
        unmatched_data_entries,
        function_lookup,
        unmatched_sections_file,
        unmatched_index_file,
        warn_log,
    )

    unmatched_sections_file.close()
    logger.info(f"  Closed: {unmatched_sections_path}")
    unmatched_index_file.close()
    logger.info(f"  Closed: {unmatched_index_path}")
    warn_log.close()
    logger.info(f"  Closed: {warn_log_path}")
