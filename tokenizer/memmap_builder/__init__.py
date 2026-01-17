"""Memory-mapped binary file builder for aligned assembly data.

This module provides functionality to build memory-mapped binary files from
aligned CSV data, organizing functions by matched/unmatched status and creating
efficient index structures for random access.
"""

from .builder import BinaryVersionInfo, VersionKey, build_memmap_files
from .helpers import (
    FunctionBinaryData,
    InliningEntry,
    build_inlining_data,
    collect_unique_called_functions,
    format_inlining_list,
    get_called_functions_from_row,
    process_function_binary_data,
    should_skip_function,
    should_skip_function_for_matched,
    should_skip_function_for_unmatched,
)
from .passes import (
    build_function_lookup_table,
    group_unmatched_entries_by_function,
    process_matched_function_pass1,
    process_unmatched_function_pass1,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)
from .writers import (
    build_inlining_data_for_unmatched,
    finalize_index_file,
    write_matched_function_section,
    write_unmatched_function_section,
)

__all__ = [
    "BinaryVersionInfo",
    "VersionKey",
    "build_memmap_files",
    "FunctionBinaryData",
    "InliningEntry",
    "build_inlining_data",
    "collect_unique_called_functions",
    "format_inlining_list",
    "get_called_functions_from_row",
    "process_function_binary_data",
    "should_skip_function",
    "should_skip_function_for_matched",
    "should_skip_function_for_unmatched",
    "build_function_lookup_table",
    "group_unmatched_entries_by_function",
    "process_matched_function_pass1",
    "process_unmatched_function_pass1",
    "write_matched_sections_pass2",
    "write_unmatched_sections_pass2",
    "build_inlining_data_for_unmatched",
    "finalize_index_file",
    "write_matched_function_section",
    "write_unmatched_function_section",
]
