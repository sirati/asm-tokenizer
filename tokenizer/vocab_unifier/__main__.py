import argparse

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

    parser.set_defaults(source="./out/")

    args = parser.parse_args()

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
        print(f"Found {len(file_names_parsed)} CSV files:")
        for csv_file in file_names_parsed:
            print(format_binary_info(csv_file, config.source_dir))
        return

    print(f"Found {len(file_names_parsed)} CSV files to unify")

    unified_vocab_file = config.output_dir / args.out_unified_vocab

    csv_paths = [binary.path for binary in file_names_parsed]
    unify_vocab(csv_paths, unified_vocab_file)

    print("Done!")


if __name__ == "__main__":
    main()
