import argparse
import logging

from shared import (
    add_selection_arguments,
    find_matching_binaries,
    format_binary_info,
    normalize_opt_levels,
    print_selection_summary,
    process_selection_arguments,
)

from .unifier import unify_vocab


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify vocabularies from multiple CSV files into a single unified vocabulary."
    )

    add_selection_arguments(parser)

    parser.add_argument(
        "--out-unified-vocab",
        type=str,
        default="unified_vocab.csv",
        help="Name of the output unified vocabulary file (default: unified_vocab.csv)",
    )

    parser.add_argument(
        "--insert-neg-value",
        action="store_true",
        help=(
            "Legacy-compat DEFAULT era for per-binary CSVs generated "
            "BEFORE value_negative was reserved at slot 256 (256 reserved "
            "slots, digits only). NOTE: each CSV's era is now auto-detected "
            "per-file from its token-stream coherence; this flag is only the "
            "tiebreak when detection is inconclusive (modern CSVs always "
            "self-detect as 257-reserved regardless). Pass it for "
            "MIXED-era corpora (legacy untouched + modern re-tokenized): "
            "modern files self-upgrade while legacy files keep this default. "
            "register_on_vocab_manager remaps legacy real tokens (256+) into "
            "canonical-layout unified ids; the emitted unified vocab always "
            "has the canonical 257-reserved layout (value_negative at slot 256)."
        ),
    )

    parser.add_argument(
        "--raw-logs",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="PREFIX",
        help="Disable log formatting (no level, timestamp, etc - only raw messages). Optionally specify a prefix",
    )

    parser.set_defaults(source="./out/")

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.raw_logs is not None:
        log_format = f"{args.raw_logs}%(message)s" if args.raw_logs else "%(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    config = process_selection_arguments(args)

    display_opt_levels = None
    if config.opt_levels:
        normalized = normalize_opt_levels(config.opt_levels, config.opt_regex)
        display_opt_levels = normalized.display_values

    print_selection_summary(config, display_opt_levels)

    csv_output_format = config.file_format + "_out\\put.\\csv"

    file_names_parsed = find_matching_binaries(
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

    if config.list_files:
        logger.info(f"Found {len(file_names_parsed)} CSV files:")
        for csv_file in file_names_parsed:
            logger.info(format_binary_info(csv_file, config.source_dir))
        return

    logger.info(f"Found {len(file_names_parsed)} CSV files to unify")

    unified_vocab_file = config.output_dir / args.out_unified_vocab

    csv_paths = [binary.path for binary in file_names_parsed]
    unify_vocab(
        csv_paths,
        unified_vocab_file,
        insert_value_negative=args.insert_neg_value,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
