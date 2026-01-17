import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from shared import (
    add_selection_arguments,
    find_matching_binaries,
    format_binary_info,
    increase_csv_field_size_limit,
    normalize_opt_levels,
    print_selection_summary,
    process_selection_arguments,
)

from .builder import BinaryVersionInfo, build_memmap_files


def group_binaries_by_name(binaries):
    """Group binaries by their binary name."""
    grouped = defaultdict(list)
    for binary in binaries:
        grouped[binary.binary_name].append(binary)
    return grouped


def match_csv_to_mapping(csv_binaries, mapping_binaries):
    """Match each CSV binary to its corresponding mapping file.

    Returns dict mapping csv BinaryInfo to mapping BinaryInfo.
    """
    matched = {}
    unmatched_csv = []

    for csv_bin in csv_binaries:
        found = False
        for map_bin in mapping_binaries:
            if csv_bin.identifier == map_bin.identifier:
                matched[csv_bin.identifier] = map_bin
                found = True
                break
        if not found:
            unmatched_csv.append(csv_bin)

    return matched, unmatched_csv


def main() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    increase_csv_field_size_limit()

    parser = argparse.ArgumentParser(description="Build memory-mapped binary files from aligned CSV data.")

    add_selection_arguments(parser)

    parser.add_argument(
        "--vocab-source",
        type=str,
        default=None,
        help="Source directory for vocabulary and mapping files. If not specified, uses the same as --source.",
    )

    parser.set_defaults(source="./out/")

    args = parser.parse_args()

    config = process_selection_arguments(args)

    vocab_source_dir = Path(args.vocab_source).resolve() if args.vocab_source else config.source_dir
    logger.info(f"Vocab source directory: {vocab_source_dir}")

    if not vocab_source_dir.exists():
        logger.error(f"Vocab source directory does not exist: {vocab_source_dir}")
        sys.exit(1)

    unified_vocab_path = vocab_source_dir / "unified_vocab.csv"
    if not unified_vocab_path.exists():
        logger.error(f"unified_vocab.csv not found in vocab source directory: {vocab_source_dir}")
        sys.exit(1)

    display_opt_levels = None
    if config.opt_levels:
        normalized = normalize_opt_levels(config.opt_levels, config.opt_regex)
        display_opt_levels = normalized.display_values

    print_selection_summary(config, display_opt_levels)
    if args.vocab_source:
        logger.info(f"Vocab source directory: {vocab_source_dir}")

    csv_output_format = config.file_format + "_out\\put.\\csv"
    mapping_format = config.file_format + "_out\\put.ma\\p\\ping.b64\\c"

    csv_binaries = find_matching_binaries(
        source_dir=config.source_dir,
        platforms=config.platforms,
        compiler=config.compiler,
        compiler_versions=config.compiler_versions,
        opt_levels=config.opt_levels,
        format_string=csv_output_format,
        version_regex=config.version_regex,
        opt_regex=config.opt_regex,
        name_regex=config.name_regex,
        exclude_subfolders=config.exclude_subfolders,
    )

    mapping_binaries = find_matching_binaries(
        source_dir=vocab_source_dir,
        platforms=config.platforms,
        compiler=config.compiler,
        compiler_versions=config.compiler_versions,
        opt_levels=config.opt_levels,
        format_string=mapping_format,
        version_regex=config.version_regex,
        opt_regex=config.opt_regex,
        name_regex=config.name_regex,
        exclude_subfolders=config.exclude_subfolders,
    )

    logger.info(f"Found {len(csv_binaries)} output.csv and {len(mapping_binaries)} mapping.bin files")

    matched_pairs, unmatched_csv = match_csv_to_mapping(csv_binaries, mapping_binaries)

    csv_by_identifier = {csv_bin.identifier: csv_bin for csv_bin in csv_binaries}

    if unmatched_csv:
        logger.warning(f"{len(unmatched_csv)} CSV file(s) have no matching mapping file:")
        for csv_bin in unmatched_csv:
            logger.warning(f"  {format_binary_info(csv_bin, config.source_dir)}")

    if config.list_files:
        logger.info(f"Found {len(matched_pairs)} matched CSV/mapping file pairs:")
        for identifier, map_bin in matched_pairs.items():
            csv_bin = csv_by_identifier[identifier]
            csv_info = format_binary_info(csv_bin, config.source_dir)
            map_info = format_binary_info(map_bin, vocab_source_dir)
            logger.info(f"  CSV: {csv_info}")
            logger.info(f"  MAP: {map_info}")
        return

    logger.info(f"Found {len(matched_pairs)} matched CSV/mapping file pairs to process")

    csv_binaries_matched = [csv_by_identifier[identifier] for identifier in matched_pairs.keys()]
    binaries_by_name = group_binaries_by_name(csv_binaries_matched)

    for binary_name, binaries in binaries_by_name.items():
        logger.info(f"\nProcessing binary: {binary_name}")
        logger.info(f"  Versions: {len(binaries)}")

        versions = [
            BinaryVersionInfo(
                path=binary.path,
                mapping_path=matched_pairs[binary.identifier].path,
                arch=binary.platform,
                compiler=binary.compiler,
                compilerversion=binary.version,
                opt=binary.opt_level,
            )
            for binary in binaries
        ]

        build_memmap_files(versions, config.output_dir, binary_name)

        logger.info(f"  Completed: {binary_name}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
