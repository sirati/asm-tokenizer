"""Memory-mapped output validation module.

This module provides functionality to validate that the memory-mapped binary files
produced by memmap_builder contain the same data as the original CSV files.
"""

from .validator import ValidationStats, ValidatorConfig, VersionInfo, validate_memmap_output

__all__ = [
    "ValidatorConfig",
    "VersionInfo",
    "ValidationStats",
    "validate_memmap_output",
]
