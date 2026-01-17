import argparse
import logging
from collections import defaultdict
from pathlib import Path

from shared import (
    add_selection_arguments,
    find_matching_binaries,
    format_binary_info,
    normalize_opt_levels,
    print_selection_summary,
    process_selection_arguments,
)

from .builder import BinaryVersionInfo, build_memmap_files

logger = logging.getLogger(__name__)


def group_binaries_by_name(binaries):
    """Group binaries by their binary name."""
    grouped = defaultdict(list)
    for binary in binaries:
        grouped[binary.binary_name].append(binary)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memory-mapped binary files from aligned CSV data.")

    add_selection_arguments(parser)

    parser.set_defaults(file_format="platform-compiler-version-optimisationlevel_binaryname_output.csv")
    parser.set_defaults(source="./out/")

    args = parser.parse_args()

    config = process_selection_arguments(args)

    display_opt_levels = None
    if config.opt_levels:
        normalized = normalize_opt_levels(config.opt_levels, config.opt_regex)
        display_opt_levels = normalized.display_values

    print_selection_summary(config, display_opt_levels)

    file_names_parsed = find_matching_binaries(
        source_dir=config.source_dir,
        platforms=config.platforms,
        compiler=config.compiler,
        compiler_versions=config.compiler_versions,
        opt_levels=config.opt_levels,
        format_string=config.file_format,
        version_regex=config.version_regex,
        opt_regex=config.opt_regex,
        name_regex=config.name_regex,
        exclude_subfolders=config.exclude_subfolders,
    )

    if config.list_files:
        print(f"Found {len(file_names_parsed)} CSV files:")
        for csv_file in file_names_parsed:
            print(format_binary_info(csv_file, config.source_dir))
        return

    print(f"Found {len(file_names_parsed)} CSV files to process")

    binaries_by_name = group_binaries_by_name(file_names_parsed)

    for binary_name, binaries in binaries_by_name.items():
        print(f"\nProcessing binary: {binary_name}")
        print(f"  Versions: {len(binaries)}")

        versions = [
            BinaryVersionInfo(
                path=binary.path,
                arch=binary.platform,
                compiler=binary.compiler,
                compilerversion=binary.version,
                opt=binary.opt_level,
            )
            for binary in binaries
        ]

        build_memmap_files(versions, config.output_dir, binary_name)

        print(f"  Completed: {binary_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
