"""
Aligned data module for function alignment export and processing.

This module provides utilities for:
- Loading and processing function alignment data
- Exporting matched and unmatched function sets
- Reading/writing binary data formats for aligned functions
- Index management for fast function lookup by length
"""

from .io import (
    decode_and_translate_tokens,
    decode_runlengths,
    format_inlining_dict,
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
    # data_loader
    "AlignedDataLoader",
    "BinaryDataset",
    "FunctionData",
    "MatchedFunction",
    "load_single_matched_function",
    # io
    "decode_and_translate_tokens",
    "decode_runlengths",
    "format_inlining_dict",
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
    "lockstep_function_match",
    # metadata
    "extract_metadata_from_section_row",
    "extract_all_metadata_from_section_rows",
    # sections
    "read_function_section",
]
