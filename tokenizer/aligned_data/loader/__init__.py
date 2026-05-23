"""
Loader subpackage for aligned function data.

This package provides classes and utilities for efficiently loading and sampling
function data from aligned data files, with support for multiple binaries,
length-based filtering, and ML-safe memory management.

Key Classes:
- FunctionData: Single function variant
- MatchedFunction: Function with multiple compilation variants
- BinaryDataset: Manages data for a single binary
- AlignedDataLoader: Main interface for loading from multiple binaries

Design Principles:
- No memmap caching (prevents memory leaks in ML training)
- Pre-computed edge indices for O(1) length-based lookups
- Efficient random sampling without searching
"""

from .aligned_data_loader import AlignedDataLoader
from .binary_dataset import BinaryDataset
from .decoded import (
    Category,
    INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS,
    TARGET_EXPONENT_BITS,
    TARGET_SIGNIFICAND_BITS,
    resolve_category_token_ids,
    resolve_number_token_ids,
    resolve_value_negative_token_id,
)
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .utils import load_single_matched_function

# Intentionally NOT re-exporting the batch_decode subpackage's entries
# from this namespace -- `from .batch_decode import batch_decode` would
# shadow the ``batch_decode`` submodule attribute with the function of
# the same name, breaking ``import loader.batch_decode.<submodule>``
# for consumers. Consumers import the public batch entry via the
# subpackage path:
#   from tokenizer.aligned_data.loader.batch_decode import (
#       BatchDecodeResult, SectionPointerSpec, VariantPadding, batch_decode,
#   )

__all__ = [
    "AlignedDataLoader",
    "BinaryDataset",
    "Category",
    "FunctionData",
    "INFNAN_EXPONENT_UNBIASED",
    "MatchedFunction",
    "TARGET_EXPONENT_BIAS",
    "TARGET_EXPONENT_BITS",
    "TARGET_SIGNIFICAND_BITS",
    "load_single_matched_function",
    "resolve_category_token_ids",
    "resolve_number_token_ids",
    "resolve_value_negative_token_id",
]
