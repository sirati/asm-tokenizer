"""
Utility functions for the aligned data loader.

This module provides convenience functions for common loading tasks.
"""

from pathlib import Path
from typing import Optional, Union

from .binary_dataset import BinaryDataset
from .matched_function import MatchedFunction


def load_single_matched_function(
    base_path: Union[str, Path], binary_name: str, func_name: str
) -> Optional[MatchedFunction]:
    """
    Convenience function to load a specific matched function by name.

    Args:
        base_path: Directory containing aligned data files
        binary_name: Name of the binary
        func_name: Name of the function to load

    Returns:
        MatchedFunction if found, None otherwise
    """
    dataset = BinaryDataset(base_path, binary_name)

    # Index file is sorted by length, not by name order in sections file
    # Must scan through all functions to find the one with matching name
    for idx in range(dataset.matched_count):
        matched_func = dataset.load_matched_function(idx)
        if matched_func.func_name == func_name:
            return matched_func

    return None
