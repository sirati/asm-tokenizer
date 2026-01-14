"""
Loader subpackage for aligned function data.

This package provides classes and utilities for efficiently loading and sampling
function data from aligned data files, with support for multiple binaries,
length-based filtering, and ML-safe memory management.

Key Classes:
- FunctionData: Single function version
- MatchedFunction: Function with multiple compilation versions
- BinaryDataset: Manages data for a single binary
- AlignedDataLoader: Main interface for loading from multiple binaries

Design Principles:
- No memmap caching (prevents memory leaks in ML training)
- Pre-computed edge indices for O(1) length-based lookups
- Efficient random sampling without searching
"""

from .aligned_data_loader import AlignedDataLoader
from .binary_dataset import BinaryDataset
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .utils import load_single_matched_function

__all__ = [
    "AlignedDataLoader",
    "BinaryDataset",
    "FunctionData",
    "MatchedFunction",
    "load_single_matched_function",
]
