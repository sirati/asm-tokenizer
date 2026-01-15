"""
BinaryDataset class for managing data for a single binary.

This module handles loading and indexing of matched and unmatched functions
for a single binary, with efficient length-based lookups using pre-computed
edge indices.
"""

import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..io import read_function_data_memmap
from ..metadata import extract_metadata_from_section_row
from .function_data import FunctionData
from .matched_function import MatchedFunction


class BinaryDataset:
    """Manages data for a single binary (matched + unmatched functions)."""

    def __init__(self, base_path: Path, binary_name: str):
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

        self.unmatched_sections = self.base_path / f"{binary_name}_unmatched_sections.csv"
        self.unmatched_data = self.base_path / f"{binary_name}_unmatched_data.bin"
        self.unmatched_index = self.base_path / f"{binary_name}_unmatched_index.bin"

        # Load metadata (NO memmap caching)
        self._load_metadata()

    def _load_index_once(
        self, index_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load index file, extract data, close memmap immediately."""
        if not index_path.exists():
            return None, None, None

        # Open memmap temporarily
        filesize = index_path.stat().st_size
        if filesize % 8 != 0:
            raise ValueError(f"Index file size {filesize} is not a multiple of 8")

        n_entries = filesize // 8
        index_memmap = np.memmap(index_path, dtype=np.uint8, mode="r", shape=(n_entries, 8))

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            while current_idx < len(actual_lengths) and actual_lengths[current_idx] < length:
                current_idx += 1
            edge_indices[length] = current_idx

        return edge_indices, count_per_length

    def _load_unmatched_lengths(self) -> np.ndarray:
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
            insn_len = int.from_bytes(data_memmap[start : start + 3].tobytes(), "little")
            block_len = int.from_bytes(data_memmap[start + 4 : start + 6].tobytes(), "little")
            token_bytes = length - 6 - insn_len - block_len
            token_count = token_bytes // 2  # uint16 tokens
            token_counts.append(token_count)

        # Close memmap
        del data_memmap

        return np.array(token_counts, dtype=np.int32)

    def _load_metadata(self):
        """Load index files and build metadata structures (NO memmap caching)."""
        # Matched functions
        if self.matched_index.exists():
            self.matched_starts, self.matched_lengths, matched_avg_lengths = self._load_index_once(self.matched_index)

            if self.matched_starts is not None and matched_avg_lengths is not None:
                self.matched_count = len(self.matched_starts)
                if len(matched_avg_lengths) > 0:
                    self.matched_edge_indices, self.matched_count_per_length = self._build_length_lookup_tables(
                        matched_avg_lengths, scale_factor=16
                    )
                else:
                    self.matched_edge_indices = np.zeros(1, dtype=np.int32)
                    self.matched_count_per_length = np.zeros(1, dtype=np.int32)

                # Build function name index from sections file
                self.matched_func_names = []
                with open(self.matched_sections, "r", newline="", encoding="ascii") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) == 2:  # Section header: func_name, called_functions
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
            self.unmatched_starts, self.unmatched_lengths, _ = self._load_index_once(self.unmatched_index)

            if self.unmatched_starts is not None:
                self.unmatched_count = len(self.unmatched_starts)
                # Load actual token counts (not cached as memmap)
                unmatched_token_counts = self._load_unmatched_lengths()
                if len(unmatched_token_counts) > 0:
                    self.unmatched_edge_indices, self.unmatched_count_per_length = self._build_length_lookup_tables(
                        unmatched_token_counts, scale_factor=1
                    )
                else:
                    self.unmatched_edge_indices = np.zeros(1, dtype=np.int32)
                    self.unmatched_count_per_length = np.zeros(1, dtype=np.int32)

                # Build function names from unmatched sections file
                self.unmatched_func_names = []
                if self.unmatched_sections.exists():
                    with open(self.unmatched_sections, "r", newline="", encoding="ascii") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if (
                                row and len(row) == 6
                            ):  # Unmatched format: func_name, compiler_sets, called_funcs, inlining_data, offset, len
                                self.unmatched_func_names.append(row[0])
                else:
                    self.unmatched_func_names = []
            else:
                self.unmatched_count = 0
                self.unmatched_starts = np.array([], dtype=np.uint32)
                self.unmatched_lengths = np.array([], dtype=np.uint32)
                self.unmatched_edge_indices = np.zeros(1, dtype=np.int32)
                self.unmatched_count_per_length = np.zeros(1, dtype=np.int32)
                self.unmatched_func_names = []
        else:
            self.unmatched_count = 0
            self.unmatched_starts = np.array([], dtype=np.uint32)
            self.unmatched_lengths = np.array([], dtype=np.uint32)
            self.unmatched_edge_indices = np.zeros(1, dtype=np.int32)
            self.unmatched_count_per_length = np.zeros(1, dtype=np.int32)
            self.unmatched_func_names = []

    def get_matched_indices_by_length(self, target_length: int, min_count: int = 1) -> np.ndarray:
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
                self.matched_edge_indices[min_len] if min_len < len(self.matched_edge_indices) else self.matched_count
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

    def get_unmatched_indices_by_length(self, target_length: int, min_count: int = 1) -> np.ndarray:
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
        if self.matched_starts is None or self.matched_lengths is None or idx >= len(self.matched_starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        start = int(self.matched_starts[idx])
        length = int(self.matched_lengths[idx])

        # Read section from CSV
        with open(self.matched_sections, "r", newline="", encoding="ascii") as f:
            f.seek(start)
            section_data = f.read(length)

        lines = section_data.strip().split("\n")

        # Parse first line: func_name, called_functions_str
        first_line = list(csv.reader([lines[0]]))[0]
        func_name = first_line[0]

        # Parse versions (remaining lines except empty line at end)
        versions = []
        reader = csv.reader(lines[1:])
        header = [
            "arch",
            "compiler",
            "compilerversion",
            "opt",
            "inlining_data",
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
            insn_rl, block_rl, tokens = read_function_data_memmap(str(self.matched_data), data_offset, data_len)

            versions.append(FunctionData(func_name, metadata, tokens, insn_rl, block_rl))

        return MatchedFunction(func_name, versions)

    def load_unmatched_function(self, idx: int) -> FunctionData:
        """Load an unmatched function by index (opens files fresh, no caching)."""
        # Get data location from stored arrays
        if self.unmatched_starts is None or self.unmatched_lengths is None or idx >= len(self.unmatched_starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        start = int(self.unmatched_starts[idx])
        length = int(self.unmatched_lengths[idx])

        # Load binary data (memmap opened and closed inside)
        insn_rl, block_rl, tokens = read_function_data_memmap(str(self.unmatched_data), start, length)

        # Try to load metadata from sections file if available
        func_name = f"unmatched_{idx}"
        platform_info = "unknown"
        called = []

        inlining_data = []
        data_offset_from_csv = start
        data_len_from_csv = length

        if self.unmatched_sections.exists() and idx < len(self.unmatched_func_names):
            func_name = self.unmatched_func_names[idx]
            # Parse sections file for this function
            with open(self.unmatched_sections, "r", newline="", encoding="ascii") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == idx and row and len(row) == 6:
                        platform_info = row[1]
                        called_str = row[2]
                        inlining_str = row[3]
                        data_offset_from_csv = int(row[4], 16)
                        data_len_from_csv = int(row[5], 16)

                        # Parse called functions (escaped commas)
                        if called_str:
                            called = [name.replace("\\,", ",") for name in called_str.split(",") if name]

                        # Parse inlining data
                        from ..metadata import parse_inlining_data

                        inlining_data = parse_inlining_data(inlining_str)
                        break

        metadata = {
            "arch": "unknown",
            "compiler": "unknown",
            "compilerversion": "unknown",
            "opt": "unknown",
            "platform_info": platform_info,
            "called": called,
            "inlining_data": inlining_data,
            "data_offset": data_offset_from_csv,
            "data_len": data_len_from_csv,
        }

        return FunctionData(func_name, metadata, tokens, insn_rl, block_rl)
