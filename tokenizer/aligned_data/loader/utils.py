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

    # Find function in the sections file
    try:
        idx = dataset.matched_func_names.index(func_name)
        return dataset.load_matched_function(idx)
    except (ValueError, IndexError):
        return None
