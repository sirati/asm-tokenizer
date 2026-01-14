"""
Aligned data module for function alignment export and processing.

This module provides utilities for:
- Loading and processing function alignment data
- Exporting matched and unmatched function sets
- Reading/writing binary data formats for aligned functions
- Index management for fast function lookup by length
"""

from .export import (
    collect_binaries,
    compute_avg_function_length,
    export_matched_and_unmatched_sets,
    find_inlined_functions,
    get_all_function_names,
    get_called_functions,
    get_function_names_across_versions,
    get_vocab_and_mapping,
    load_all_function_data,
    load_function_data,
    parse_filename,
    process_unmatched_too_long,
    run_alignment_export,
    write_function_sections,
    write_unmatched_files,
)
from .index import (
    create_length_lookup_map,
    extract_avg_lengths,
    load_index_memmap,
    read_index_entry,
    select_random_function_by_length,
)
from .io import (
    decode_and_translate_tokens,
    decode_runlengths,
    parse_function_data_header,
    read_data_file,
    read_function_data_memmap,
    read_index_file,
    read_sections_file,
    write_function_binary_data,
    write_function_section_csv,
    write_index_entry,
)
from .match import (
    is_vocab_row,
    lockstep_function_match,
    open_csv_skip_vocab,
)
from .metadata import (
    extract_all_metadata_from_section_rows,
    extract_metadata_from_section_row,
)
from .sections import (
    read_function_section,
)

__all__ = [
    # io
    "decode_and_translate_tokens",
    "decode_runlengths",
    "write_function_binary_data",
    "write_index_entry",
    "write_function_section_csv",
    "read_index_file",
    "read_sections_file",
    "read_data_file",
    "read_function_data_memmap",
    "parse_function_data_header",
    # export
    "parse_filename",
    "collect_binaries",
    "load_function_data",
    "get_vocab_and_mapping",
    "get_function_names_across_versions",
    "load_all_function_data",
    "get_called_functions",
    "find_inlined_functions",
    "compute_avg_function_length",
    "write_function_sections",
    "write_unmatched_files",
    "get_all_function_names",
    "process_unmatched_too_long",
    "export_matched_and_unmatched_sets",
    "run_alignment_export",
    # match
    "is_vocab_row",
    "open_csv_skip_vocab",
    "lockstep_function_match",
    # metadata
    "extract_metadata_from_section_row",
    "extract_all_metadata_from_section_rows",
    # sections
    "read_function_section",
    # index
    "load_index_memmap",
    "extract_avg_lengths",
    "create_length_lookup_map",
    "select_random_function_by_length",
    "read_index_entry",
]
