"""
AlignedDataLoader class for loading function data across multiple binaries.

This module provides the main interface for loading and sampling function data
from aligned data files, supporting multiple binaries with flexible filtering
and sampling strategies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .binary_dataset import BinaryDataset
from .function_data import FunctionData
from .matched_function import MatchedFunction


class AlignedDataLoader:
    """
    Main data loader for aligned function data across multiple binaries.

    Supports:
    - Loading matched functions (same function across multiple compilation versions)
    - Loading unmatched functions (single version functions)
    - Filtering by token length
    - Random sampling with various strategies
    - No memmap caching (safe for ML training)
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        binary_names: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize data loader.

        Args:
            base_path: Directory containing aligned data files
            binary_names: List of binary names to load
            min_length: Minimum token length (inclusive), None for no limit
            max_length: Maximum token length (inclusive), None for no limit
            seed: Random seed for reproducibility
        """
        self.base_path = Path(base_path)
        self.binary_names = binary_names
        self.min_length = min_length if min_length is not None else 0
        self.max_length = max_length if max_length is not None else 1_000_000
        self.rng = np.random.default_rng(seed)

        # Load datasets
        self.datasets = {name: BinaryDataset(base_path, name) for name in binary_names}

        # Build global indices
        self._build_indices()

    def _build_indices(self):
        """Build global indices across all binaries."""
        # Matched functions: (binary_name, idx)
        self.matched_indices = []
        for binary_name, dataset in self.datasets.items():
            indices = dataset.get_matched_indices_in_range(
                self.min_length, self.max_length
            )
            for idx in indices:
                self.matched_indices.append((binary_name, int(idx)))

        # Unmatched functions: (binary_name, idx)
        self.unmatched_indices = []
        for binary_name, dataset in self.datasets.items():
            indices = dataset.get_unmatched_indices_in_range(
                self.min_length, self.max_length
            )
            for idx in indices:
                self.unmatched_indices.append((binary_name, int(idx)))

    def load_matched_functions(
        self, n: int, target_length: Optional[int] = None
    ) -> List[MatchedFunction]:
        """
        Load N random matched functions of the same or similar length.

        Uses pre-computed edge indices for O(1) lookup without searching.

        Args:
            n: Number of functions to load
            target_length: Target token length, or None to pick random length

        Returns:
            List of MatchedFunction objects
        """
        if len(self.matched_indices) == 0:
            return []

        # Determine target length
        if target_length is None:
            # Pick a random length from available lengths
            all_counts = []
            all_lengths = []
            for dataset in self.datasets.values():
                for length, count in enumerate(dataset.matched_count_per_length):
                    if count > 0 and self.min_length <= length <= self.max_length:
                        all_lengths.append(length)
                        all_counts.append(count)

            if not all_lengths:
                return []

            # Weight by count for better sampling
            total = sum(all_counts)
            probs = [c / total for c in all_counts]
            target_length = self.rng.choice(all_lengths, p=probs)

        # Collect candidates from all datasets
        candidates = []
        # Ensure target_length is set
        if target_length is None:
            target_length = self.min_length

        for binary_name, dataset in self.datasets.items():
            indices = dataset.get_matched_indices_by_length(target_length, min_count=1)
            for idx in indices:
                candidates.append((binary_name, int(idx)))

        if len(candidates) == 0:
            # Fallback to any available
            candidates = self.matched_indices

        # Sample N functions
        n = min(n, len(candidates))
        if n == 0:
            return []

        selected_idx = self.rng.choice(len(candidates), size=n, replace=False)

        functions = []
        for idx in selected_idx:
            binary_name, func_idx = candidates[idx]
            func = self.datasets[binary_name].load_matched_function(func_idx)
            functions.append(func)

        return functions

    def load_unmatched_functions(self, n: int) -> List[FunctionData]:
        """
        Load N random unmatched functions.

        Args:
            n: Number of functions to load

        Returns:
            List of FunctionData objects
        """
        if len(self.unmatched_indices) == 0:
            return []

        n = min(n, len(self.unmatched_indices))
        if n == 0:
            return []

        selected_idx = self.rng.choice(
            len(self.unmatched_indices), size=n, replace=False
        )

        functions = []
        for idx in selected_idx:
            binary_name, func_idx = self.unmatched_indices[idx]
            func = self.datasets[binary_name].load_unmatched_function(func_idx)
            functions.append(func)

        return functions

    def load_random_sections(
        self, n: int
    ) -> List[Union[FunctionData, MatchedFunction]]:
        """
        Load N random function sections (mixed matched and unmatched).

        Each matched function is treated as a single unit. The split between matched
        and unmatched is chosen randomly to avoid bias.

        Args:
            n: Number of sections to load

        Returns:
            List containing FunctionData (for unmatched) and MatchedFunction (for matched) objects
        """
        total_available = len(self.matched_indices) + len(self.unmatched_indices)
        if total_available == 0:
            return []

        n = min(n, total_available)

        # Randomly decide how many matched vs unmatched
        # Use binomial distribution based on proportion
        if total_available > 0:
            p_matched = len(self.matched_indices) / total_available
            n_matched = self.rng.binomial(n, p_matched)
            n_unmatched = n - n_matched
        else:
            n_matched = 0
            n_unmatched = 0

        # Load both types
        matched = self.load_matched_functions(n_matched) if n_matched > 0 else []
        unmatched = (
            self.load_unmatched_functions(n_unmatched) if n_unmatched > 0 else []
        )

        # Combine and shuffle
        all_functions = matched + unmatched
        self.rng.shuffle(all_functions)

        return all_functions

    def get_statistics(self) -> Dict:
        """Get statistics about the loaded datasets."""
        stats = {
            "total_binaries": len(self.binary_names),
            "total_matched_functions": len(self.matched_indices),
            "total_unmatched_functions": len(self.unmatched_indices),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "binaries": {},
        }

        for binary_name, dataset in self.datasets.items():
            matched_in_range = len(
                dataset.get_matched_indices_in_range(self.min_length, self.max_length)
            )
            unmatched_in_range = len(
                dataset.get_unmatched_indices_in_range(self.min_length, self.max_length)
            )

            stats["binaries"][binary_name] = {
                "total_matched": dataset.matched_count,
                "total_unmatched": dataset.unmatched_count,
                "matched_in_range": matched_in_range,
                "unmatched_in_range": unmatched_in_range,
            }

        return stats
