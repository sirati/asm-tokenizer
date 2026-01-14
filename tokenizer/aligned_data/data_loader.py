"""
Data loader for aligned function data.

This module provides efficient loading and sampling of function data from the aligned data format,
supporting multiple binaries with filtering by token length and various sampling strategies.

Key design principles:
- No memmap caching (prevents memory leaks in ML training)
- Pre-computed edge indices for O(1) length-based lookups
- Efficient random sampling without searching
"""

#todo optimially we would like two more capabilities
# 1. for a given lengths know how many functions we can load without OOM
# 2. allow biased sampling to target length, i.e. increase batch variaty by not only taking the target lengths,
#    but also sampling shorter functions with some bias against them

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .io import read_function_data_memmap
from .metadata import extract_metadata_from_section_row


class FunctionData:
    """Represents a single function version with its data."""

    def __init__(
        self,
        func_name: str,
        metadata: Dict,
        tokens: np.ndarray,
        insn_runlength: np.ndarray,
        block_runlength: np.ndarray,
    ):
        self.func_name = func_name
        self.metadata = metadata
        self.tokens = tokens
        self.insn_runlength = insn_runlength
        self.block_runlength = block_runlength

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"FunctionData({self.func_name}, {self.metadata.get('arch')}-{self.metadata.get('compiler')}-{self.metadata.get('opt')}, {len(self)} tokens)"


class MatchedFunction:
    """Represents a matched function with multiple versions."""

    def __init__(self, func_name: str, versions: List[FunctionData]):
        self.func_name = func_name
        self.versions = versions

    def __len__(self):
        """Return average token count across all versions."""
        return int(np.mean([len(v) for v in self.versions]))

    def __repr__(self):
        return f"MatchedFunction({self.func_name}, {len(self.versions)} versions, avg {len(self)} tokens)"


class BinaryDataset:
    """Manages data for a single binary (matched + unmatched functions)."""

    def __init__(self, base_path: Union[str, Path], binary_name: str):
        """
        Initialize dataset for a binary.

        Args:
            base_path: Directory containing the aligned data files
            binary_name: Name of the binary (e.g., 'minigzipsh')
        """
        self.base_path = Path(base_path)
        self.binary_name = binary_name

        # File paths
        self.matched_sections = self.base_path / f"{binary_name}_sections.csv"
        self.matched_data = self.base_path / f"{binary_name}_data.bin"
        self.matched_index = self.base_path / f"{binary_name}_index.bin"

        self.unmatched_sections = (
            self.base_path / f"{binary_name}_unmatched_sections.csv"
        )
        self.unmatched_data = self.base_path / f"{binary_name}_unmatched_data.bin"
        self.unmatched_index = self.base_path / f"{binary_name}_unmatched_index.bin"

        # Load metadata (NO memmap caching)
        self._load_metadata()

    def _load_index_once(self, index_path: Path):
        """Load index file, extract data, close memmap immediately."""
        if not index_path.exists():
            return None, None, None

        # Open memmap temporarily
        filesize = index_path.stat().st_size
        if filesize % 8 != 0:
            raise ValueError(f"Index file size {filesize} is not a multiple of 8")

        n_entries = filesize // 8
        index_memmap = np.memmap(
            index_path, dtype=np.uint8, mode="r", shape=(n_entries, 8)
        )

        # Extract all needed data before closing
        starts = np.zeros(n_entries, dtype=np.uint32)
        lengths = np.zeros(n_entries, dtype=np.uint32)
        avg_lengths = np.zeros(n_entries, dtype=np.uint8)

        for i in range(n_entries):
            entry = index_memmap[i]
            starts[i] = int.from_bytes(entry[0:4].tobytes(), "little")
            lengths[i] = int.from_bytes(entry[4:7].tobytes(), "little")
            avg_lengths[i] = entry[7]

        # Close memmap by deleting reference
        del index_memmap

        return starts, lengths, avg_lengths

    def _build_length_lookup_tables(
        self, avg_lengths: np.ndarray, scale_factor: int = 16
    ):
        """
        Build edge indices and count arrays for efficient length-based lookup.

        Args:
            avg_lengths: Array of average lengths (sorted)
            scale_factor: Scale factor for lengths (avg_lengths are scaled down)

        Returns:
            edge_indices: edge_indices[L] = first index where length >= L
            count_per_length: count_per_length[L] = number of functions with length L
        """
        if len(avg_lengths) == 0:
            return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)

        # Scale back to actual lengths
        actual_lengths = avg_lengths.astype(np.int32) * scale_factor

        max_length = int(actual_lengths.max())

        # Allocate arrays (up to max_length + 1)
        edge_indices = np.zeros(max_length + 2, dtype=np.int32)
        count_per_length = np.zeros(max_length + 1, dtype=np.int32)

        # Count occurrences
        for length in actual_lengths:
            count_per_length[length] += 1

        # Build edge indices (first index where length >= L)
        # Since sorted, we can build this efficiently
        current_idx = 0
        for length in range(max_length + 2):
            # Find first index with actual_lengths[i] >= length
            while (
                current_idx < len(actual_lengths)
                and actual_lengths[current_idx] < length
            ):
                current_idx += 1
            edge_indices[length] = current_idx

        return edge_indices, count_per_length

    def _load_unmatched_lengths(self):
        """Load actual token counts for unmatched functions (no memmap caching)."""
        if not self.unmatched_index.exists() or not self.unmatched_data.exists():
            return np.array([], dtype=np.int32)

        # Read index
        starts, lengths, _ = self._load_index_once(self.unmatched_index)
        if starts is None:
            return np.array([], dtype=np.int32)

        # Open data memmap temporarily to read headers
        data_memmap = np.memmap(str(self.unmatched_data), dtype=np.uint8, mode="r")

        token_counts = []
        if starts is None or lengths is None:
            return np.array([], dtype=np.int32)

        for i in range(len(starts)):
            start = int(starts[i])
            length = int(lengths[i])
            # Read just the header
            insn_len = data_memmap[start]
            block_len = int.from_bytes(
                data_memmap[start + 2 : start + 4].tobytes(), "little"
            )
            token_bytes = length - 4 - insn_len - block_len
            token_count = token_bytes // 2  # uint16 tokens
            token_counts.append(token_count)

        # Close memmap
        del data_memmap

        return np.array(token_counts, dtype=np.int32)

    def _load_metadata(self):
        """Load index files and build metadata structures (NO memmap caching)."""
        # Matched functions
        if self.matched_index.exists():
            self.matched_starts, self.matched_lengths, matched_avg_lengths = (
                self._load_index_once(self.matched_index)
            )

            if self.matched_starts is not None and matched_avg_lengths is not None:
                self.matched_count = len(self.matched_starts)
                if len(matched_avg_lengths) > 0:
                    self.matched_edge_indices, self.matched_count_per_length = (
                        self._build_length_lookup_tables(
                            matched_avg_lengths, scale_factor=16
                        )
                    )
                else:
                    self.matched_edge_indices = np.zeros(1, dtype=np.int32)
                    self.matched_count_per_length = np.zeros(1, dtype=np.int32)

                # Build function name index from sections file
                self.matched_func_names = []
                with open(
                    self.matched_sections, "r", newline="", encoding="ascii"
                ) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) == 1:  # Section header
                            self.matched_func_names.append(row[0])
            else:
                self.matched_count = 0
                self.matched_starts = np.array([], dtype=np.uint32)
                self.matched_lengths = np.array([], dtype=np.uint32)
                self.matched_edge_indices = np.zeros(1, dtype=np.int32)
                self.matched_count_per_length = np.zeros(1, dtype=np.int32)
                self.matched_func_names = []
        else:
            self.matched_count = 0
            self.matched_starts = np.array([], dtype=np.uint32)
            self.matched_lengths = np.array([], dtype=np.uint32)
            self.matched_edge_indices = np.zeros(1, dtype=np.int32)
            self.matched_count_per_length = np.zeros(1, dtype=np.int32)
            self.matched_func_names = []

        # Unmatched functions
        if self.unmatched_index.exists():
            self.unmatched_starts, self.unmatched_lengths, _ = self._load_index_once(
                self.unmatched_index
            )

            if self.unmatched_starts is not None:
                self.unmatched_count = len(self.unmatched_starts)
                # Load actual token counts (not cached as memmap)
                unmatched_token_counts = self._load_unmatched_lengths()
                if len(unmatched_token_counts) > 0:
                    self.unmatched_edge_indices, self.unmatched_count_per_length = (
                        self._build_length_lookup_tables(
                            unmatched_token_counts, scale_factor=1
                        )
                    )
                else:
                    self.unmatched_edge_indices = np.zeros(1, dtype=np.int32)
                    self.unmatched_count_per_length = np.zeros(1, dtype=np.int32)
            else:
                self.unmatched_count = 0
                self.unmatched_starts = np.array([], dtype=np.uint32)
                self.unmatched_lengths = np.array([], dtype=np.uint32)
                self.unmatched_edge_indices = np.zeros(1, dtype=np.int32)
                self.unmatched_count_per_length = np.zeros(1, dtype=np.int32)
        else:
            self.unmatched_count = 0
            self.unmatched_starts = np.array([], dtype=np.uint32)
            self.unmatched_lengths = np.array([], dtype=np.uint32)
            self.unmatched_edge_indices = np.zeros(1, dtype=np.int32)
            self.unmatched_count_per_length = np.zeros(1, dtype=np.int32)

    def get_matched_indices_by_length(
        self, target_length: int, min_count: int = 1
    ) -> np.ndarray:
        """
        Get indices of matched functions at or near target length.

        Expands search range if not enough functions at exact length.

        Args:
            target_length: Target token length
            min_count: Minimum number of functions needed

        Returns:
            Array of function indices
        """
        if self.matched_count == 0:
            return np.array([], dtype=np.int32)

        max_available_length = len(self.matched_count_per_length) - 1

        # Start with exact length
        search_radius = 0
        max_radius = 1000  # Maximum expansion

        while search_radius <= max_radius:
            min_len = max(0, target_length - search_radius)
            max_len = min(max_available_length, target_length + search_radius)

            # Get range using pre-computed edges
            start_idx = (
                self.matched_edge_indices[min_len]
                if min_len < len(self.matched_edge_indices)
                else self.matched_count
            )
            end_idx = (
                self.matched_edge_indices[max_len + 1]
                if max_len + 1 < len(self.matched_edge_indices)
                else self.matched_count
            )

            count = end_idx - start_idx

            if count >= min_count:
                return np.arange(start_idx, end_idx, dtype=np.int32)

            # Expand search
            if search_radius == 0:
                search_radius = 16  # Start with ±16 tokens (1 scaled unit)
            else:
                search_radius = int(search_radius * 1.5)

        # Return all if we couldn't find enough
        return np.arange(self.matched_count, dtype=np.int32)

    def get_unmatched_indices_by_length(
        self, target_length: int, min_count: int = 1
    ) -> np.ndarray:
        """Get indices of unmatched functions at or near target length."""
        if self.unmatched_count == 0:
            return np.array([], dtype=np.int32)

        max_available_length = len(self.unmatched_count_per_length) - 1

        search_radius = 0
        max_radius = 1000

        while search_radius <= max_radius:
            min_len = max(0, target_length - search_radius)
            max_len = min(max_available_length, target_length + search_radius)

            start_idx = (
                self.unmatched_edge_indices[min_len]
                if min_len < len(self.unmatched_edge_indices)
                else self.unmatched_count
            )
            end_idx = (
                self.unmatched_edge_indices[max_len + 1]
                if max_len + 1 < len(self.unmatched_edge_indices)
                else self.unmatched_count
            )

            count = end_idx - start_idx

            if count >= min_count:
                return np.arange(start_idx, end_idx, dtype=np.int32)

            if search_radius == 0:
                search_radius = 16
            else:
                search_radius = int(search_radius * 1.5)

        return np.arange(self.unmatched_count, dtype=np.int32)

    def get_matched_indices_in_range(self, min_len: int, max_len: int) -> np.ndarray:
        """Get all matched function indices within length range."""
        if self.matched_count == 0:
            return np.array([], dtype=np.int32)

        max_available = len(self.matched_edge_indices) - 2
        min_len = max(0, min(min_len, max_available))
        max_len = max(0, min(max_len, max_available))

        start_idx = self.matched_edge_indices[min_len]
        end_idx = (
            self.matched_edge_indices[max_len + 1]
            if max_len + 1 < len(self.matched_edge_indices)
            else self.matched_count
        )

        return np.arange(start_idx, end_idx, dtype=np.int32)

    def get_unmatched_indices_in_range(self, min_len: int, max_len: int) -> np.ndarray:
        """Get all unmatched function indices within length range."""
        if self.unmatched_count == 0:
            return np.array([], dtype=np.int32)

        max_available = len(self.unmatched_edge_indices) - 2
        min_len = max(0, min(min_len, max_available))
        max_len = max(0, min(max_len, max_available))

        start_idx = self.unmatched_edge_indices[min_len]
        end_idx = (
            self.unmatched_edge_indices[max_len + 1]
            if max_len + 1 < len(self.unmatched_edge_indices)
            else self.unmatched_count
        )

        return np.arange(start_idx, end_idx, dtype=np.int32)

    def load_matched_function(self, idx: int) -> MatchedFunction:
        """Load a matched function by index (opens files fresh, no caching)."""
        # Get section location from stored arrays
        if (
            self.matched_starts is None
            or self.matched_lengths is None
            or idx >= len(self.matched_starts)
        ):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        start = int(self.matched_starts[idx])
        length = int(self.matched_lengths[idx])

        # Read section from CSV
        with open(self.matched_sections, "r", newline="", encoding="ascii") as f:
            f.seek(start)
            section_data = f.read(length)

        lines = section_data.strip().split("\n")
        func_name = lines[0]

        # Parse versions
        versions = []
        reader = csv.reader(lines[1:])
        header = [
            "function_name",
            "arch",
            "compiler",
            "compilerversion",
            "opt",
            "called",
            "inlining_map",
            "data_offset",
            "data_len",
        ]

        for row in reader:
            if not row or len(row) == 0:
                continue

            metadata = extract_metadata_from_section_row(row, header)

            # Load binary data (memmap opened and closed inside read_function_data_memmap)
            data_offset = metadata["data_offset"]
            data_len = metadata["data_len"]
            insn_rl, block_rl, tokens = read_function_data_memmap(
                str(self.matched_data), data_offset, data_len
            )

            versions.append(
                FunctionData(func_name, metadata, tokens, insn_rl, block_rl)
            )

        return MatchedFunction(func_name, versions)

    def load_unmatched_function(self, idx: int) -> FunctionData:
        """Load an unmatched function by index (opens files fresh, no caching)."""
        # Get data location from stored arrays
        if (
            self.unmatched_starts is None
            or self.unmatched_lengths is None
            or idx >= len(self.unmatched_starts)
        ):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        start = int(self.unmatched_starts[idx])
        length = int(self.unmatched_lengths[idx])

        # Load binary data (memmap opened and closed inside)
        insn_rl, block_rl, tokens = read_function_data_memmap(
            str(self.unmatched_data), start, length
        )

        # Minimal metadata for unmatched
        metadata = {
            "arch": "unknown",
            "compiler": "unknown",
            "compilerversion": "unknown",
            "opt": "unknown",
            "called": [],
            "inlining_map": {},
            "data_offset": start,
            "data_len": length,
        }

        return FunctionData(f"unmatched_{idx}", metadata, tokens, insn_rl, block_rl)


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
