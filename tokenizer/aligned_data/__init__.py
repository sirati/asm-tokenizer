"""
Aligned data module for function alignment export and processing.

This module provides utilities for:
- Loading and processing function alignment data
- Exporting matched and unmatched function sets
- Reading/writing binary data formats for aligned functions
- Index management for fast function lookup by length
"""

from .io import (
    assemble_function_record,
    format_call_targets_dict,
    format_variant_refs,
    parse_function_data_header,
    parse_function_data_memmap,
    read_data_file,
    read_function_data_memmap,
    read_index_file,
    read_sections_file,
    write_function_binary_data,
    write_function_section_csv,
    write_index_entry,
    write_unmatched_section_csv,
)
from .loader import (
    AlignedDataLoader,
    BinaryDataset,
    FunctionData,
    MatchedFunction,
    load_single_matched_function,
)
from .match import (
    is_vocab_row,
    open_csv_skip_vocab,
)
from .metadata import (
    extract_metadata_from_variant_block,
    parse_call_targets,
    parse_called_line_nos_typed,
)
from .parsed_record_iter import (
    Matched,
    ParsedRecord,
    Unmatched,
    lockstep_records,
    open_parsed_record_iter,
)
from .sections import (
    read_function_section,
)

__all__ = [
    # data_loader
    "AlignedDataLoader",
    "BinaryDataset",
    "FunctionData",
    "MatchedFunction",
    "load_single_matched_function",
    # io
    "assemble_function_record",
    "format_call_targets_dict",
    "format_variant_refs",
    "write_function_binary_data",
    "write_index_entry",
    "write_function_section_csv",
    "write_unmatched_section_csv",
    "read_index_file",
    "read_sections_file",
    "read_data_file",
    "read_function_data_memmap",
    "parse_function_data_memmap",
    "parse_function_data_header",
    # match
    "is_vocab_row",
    "open_csv_skip_vocab",
    # parsed_record_iter
    "ParsedRecord",
    "Matched",
    "Unmatched",
    "open_parsed_record_iter",
    "lockstep_records",
    # metadata
    "extract_metadata_from_variant_block",
    "parse_call_targets",
    "parse_called_line_nos_typed",
    # sections
    "read_function_section",
]
