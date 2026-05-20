"""Memory-mapped binary file builder for aligned assembly data.

This module provides functionality to build memory-mapped binary files from
aligned CSV data, organizing functions by matched/unmatched status and creating
efficient index structures for random access.
"""

from .builder import BinaryVersionInfo, VersionKey, build_memmap_files
from .helpers import (
    should_skip_for_matched,
    should_skip_for_unmatched,
)
from .passes import (
    build_function_lookup_table,
    group_unmatched_entries_by_function,
    process_matched_function,
    process_unmatched_function,
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)

__all__ = [
    "BinaryVersionInfo",
    "VersionKey",
    "build_memmap_files",
    "should_skip_for_matched",
    "should_skip_for_unmatched",
    "build_function_lookup_table",
    "group_unmatched_entries_by_function",
    "process_matched_function",
    "process_unmatched_function",
    "write_matched_sections_pass2",
    "write_unmatched_sections_pass2",
]
