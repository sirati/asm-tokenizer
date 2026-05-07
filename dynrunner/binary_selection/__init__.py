"""Asm-binary corpus selection helpers, vendored out of the framework
in `dynamic_runner` commit 6c65bb7. The framework's `_shared/` retains
only the generic `TaskInfo` / `BinaryIdentifier` / `format_size`; the
asm-specific filename parser, filter compiler, walker, and the
`--platform`/`--compiler`/etc. argparse flags are owned by the
consumer (us) and live here.

Public surface — what TaskDefinitions in `dynrunner/{tokenize,
unify_vocab, build_memmap}/` import from:

  - filename parsing: `parse_binary_filename`,
    `build_binary_filename_format`, `BinaryFilenameFormat`,
    `FieldRegexes`, `FIELD_MAPPING`, `REQUIRED_FIELDS`
  - filter compilation + per-filename matching:
    `SelectionFilters`, `compile_selection_filters`,
    `match_filename`, `is_excluded_subfolder`
  - whole-tree os.walk discovery (legacy):
    `find_matching_binaries`
  - argparse + post-parse: `add_asm_selection_arguments`,
    `process_selection_arguments`, `print_selection_summary`,
    `normalize_opt_levels`, `SelectionConfig`,
    `NormalizedOptLevels`
  - re-exported framework types: `BinaryIdentifier`, `TaskInfo`,
    `format_size`, `format_binary_info`
"""

from .binary_info import (
    BinaryFilenameFormat,
    BinaryIdentifier,
    FIELD_MAPPING,
    FieldRegexes,
    REQUIRED_FIELDS,
    TaskInfo,
    build_binary_filename_format,
    build_field_regexes,
    format_binary_info,
    format_size,
    parse_binary_filename,
)
from .binary_selector import (
    SelectionFilters,
    compile_selection_filters,
    find_matching_binaries,
    is_excluded_subfolder,
    match_filename,
)
from .selection_args import (
    NormalizedOptLevels,
    SelectionConfig,
    add_asm_selection_arguments,
    normalize_opt_levels,
    print_selection_summary,
    process_selection_arguments,
)

__all__ = [
    "BinaryFilenameFormat",
    "BinaryIdentifier",
    "FIELD_MAPPING",
    "FieldRegexes",
    "NormalizedOptLevels",
    "REQUIRED_FIELDS",
    "SelectionConfig",
    "SelectionFilters",
    "TaskInfo",
    "add_asm_selection_arguments",
    "build_binary_filename_format",
    "build_field_regexes",
    "compile_selection_filters",
    "find_matching_binaries",
    "format_binary_info",
    "format_size",
    "is_excluded_subfolder",
    "match_filename",
    "normalize_opt_levels",
    "parse_binary_filename",
    "print_selection_summary",
    "process_selection_arguments",
]
