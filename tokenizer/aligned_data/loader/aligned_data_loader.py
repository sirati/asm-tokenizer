"""
AlignedDataLoader class for loading function data across multiple binaries.

Receiver contracts (satisfied by sibling sub-tasks of this batch):

* ``BinaryDataset(base_path, binary_name, vocab_manager=...)`` — accepts the
  unified ``VocabularyManager`` so variant-axis token IDs resolve against
  the corpus-wide ID space (5G shell-integrator wires receipt).
* ``BinaryDataset.open_session() -> BinarySession`` — context manager that
  lazily opens (and on ``__exit__`` closes) the three per-binary file
  handles (sections CSV, ``_data.bin`` memmap, ``_variants.bin`` memmap),
  exposing ``load_matched(idx)`` and ``load_unmatched(idx)`` methods (5C
  ``session.py`` wires receipt).

Per-binary batching rationale: today's public ``load_matched_function(idx)``
opens-and-closes all three handles per call. Grouping batch candidates by
``binary_name`` before entering any session, then doing every load for that
group inside one ``with binary.open_session()`` block, collapses that to
one open-and-close set per touched binary. ML-leak discipline is preserved
because each top-level batch call still drops the handles on exit.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .binary_dataset import BinaryDataset
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .unified_vocab_gate import (
    load_and_validate_unified_vocab,
    resolve_unified_vocab_path,
)


class AlignedDataLoader:
    """
    Main data loader for aligned function data across multiple binaries.

    Supports:
    - Loading matched functions (same function across multiple compilation versions)
    - Loading unmatched functions (single version functions)
    - Filtering by token length
    - Random sampling with various strategies
    - No memmap caching across batches (safe for ML training); within a single
      batch call, file handles are reused per-binary via ``open_session``.
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        binary_names: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        seed: Optional[int] = None,
        unified_vocab_path: Optional[Path] = None,
    ):
        """
        Initialize data loader.

        Args:
            base_path: Directory containing aligned data files
            binary_names: List of binary names to load
            min_length: Minimum token length (inclusive), None for no limit
            max_length: Maximum token length (inclusive), None for no limit
            seed: Random seed for reproducibility
            unified_vocab_path: Path to the corpus-wide ``unified_vocab.csv``.
                Defaults to the first existing candidate from
                :func:`resolve_unified_vocab_path` — alongside the memmap
                bins, or at the corpus-root parent when the bins live in
                a subdirectory. Loaded once here and threaded into every
                ``BinaryDataset`` so variant-axis tokens decode through the
                same ID space the memmap_builder wrote.

        Raises:
            ValueError: If the unified vocab is missing, unparseable, or its
                ``format_version`` does not equal the memmap-chain version
                (see ``unified_vocab_gate.REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION``,
                currently v1). Hard cutover — see ``unified_vocab_gate`` for
                the rationale.
        """
        self.base_path = Path(base_path)
        self.binary_names = binary_names
        self.min_length = min_length if min_length is not None else 0
        self.max_length = max_length if max_length is not None else 1_000_000
        self.rng = np.random.default_rng(seed)

        # Resolve unified vocab path: explicit caller-supplied path wins;
        # otherwise search the policy candidates (vocab alongside the
        # memmap bins or one level up at the corpus root). Keep the
        # resolved path on the instance for diagnostics.
        self.unified_vocab_path = (
            Path(unified_vocab_path)
            if unified_vocab_path is not None
            else resolve_unified_vocab_path(self.base_path)
        )

        # Load + gate the unified vocab BEFORE constructing any BinaryDataset.
        # Failing here means no per-binary state ever materialises, so the
        # caller sees a clean ValueError without partially-initialised state.
        self.vocab_manager = load_and_validate_unified_vocab(
            self.unified_vocab_path
        )

        # Pass the validated vocab into each BinaryDataset (receiver contract
        # wired by the shell-integrator subtask).
        self.datasets = {
            name: BinaryDataset(
                self.base_path, name, vocab_manager=self.vocab_manager
            )
            for name in binary_names
        }

        # Build global indices
        self._build_indices()

    def _build_indices(self):
        """Build global indices across all binaries."""
        # Matched functions: (binary_name, idx)
        self.matched_indices = []
        for binary_name, dataset in self.datasets.items():
            indices = dataset.get_matched_indices_in_range(self.min_length, self.max_length)
            for idx in indices:
                self.matched_indices.append((binary_name, int(idx)))

        # Unmatched functions: (binary_name, idx)
        self.unmatched_indices = []
        for binary_name, dataset in self.datasets.items():
            indices = dataset.get_unmatched_indices_in_range(self.min_length, self.max_length)
            for idx in indices:
                self.unmatched_indices.append((binary_name, int(idx)))

    # ------------------------------------------------------------------
    # Per-binary batching helper
    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_binary(
        candidates: Sequence[Tuple[str, int]],
    ) -> Dict[str, List[int]]:
        """Group ``(binary_name, func_idx)`` pairs by ``binary_name``.

        Insertion-ordered dict (Python 3.7+ guarantee) preserves the random
        sampling order across binaries; per-binary index lists keep the
        order they were sampled in, so deterministic seeds stay
        deterministic.
        """
        grouped: Dict[str, List[int]] = {}
        for binary_name, func_idx in candidates:
            grouped.setdefault(binary_name, []).append(int(func_idx))
        return grouped

    def load_matched_functions(self, n: int, target_length: Optional[int] = None) -> List[MatchedFunction]:
        """
        Load N random matched functions of the same or similar length.

        Uses pre-computed edge indices for O(1) lookup without searching.
        Groups selected candidates by ``binary_name`` before entering any
        session; iterates groups serially with one ``binary.open_session()``
        per group, so per-binary file handles are reused across every load
        in the group.
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
        candidates: List[Tuple[str, int]] = []
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
        selected: List[Tuple[str, int]] = [candidates[i] for i in selected_idx]

        # Group BEFORE entering any session so we open at most one session
        # per touched binary; each group's loads share three file handles.
        grouped = self._group_by_binary(selected)
        functions: List[MatchedFunction] = []
        for binary_name, func_indices in grouped.items():
            binary = self.datasets[binary_name]
            with binary.open_session() as sess:
                for func_idx in func_indices:
                    functions.append(sess.load_matched(func_idx))

        return functions

    def load_unmatched_functions(self, n: int) -> List[FunctionData]:
        """Load N random unmatched functions.

        Same per-binary batching as ``load_matched_functions``.
        """
        if len(self.unmatched_indices) == 0:
            return []

        n = min(n, len(self.unmatched_indices))
        if n == 0:
            return []

        selected_idx = self.rng.choice(len(self.unmatched_indices), size=n, replace=False)
        selected: List[Tuple[str, int]] = [self.unmatched_indices[i] for i in selected_idx]

        grouped = self._group_by_binary(selected)
        functions: List[FunctionData] = []
        for binary_name, func_indices in grouped.items():
            binary = self.datasets[binary_name]
            with binary.open_session() as sess:
                for func_idx in func_indices:
                    functions.append(sess.load_unmatched(func_idx))

        return functions

    def load_random_sections(self, n: int) -> List[Union[FunctionData, MatchedFunction]]:
        """Load N random function sections (mixed matched and unmatched).

        Each matched function is treated as a single unit. The matched and
        unmatched batch helpers each do their own per-binary grouping, so
        the open-session count is bounded by the number of distinct
        binaries touched across both halves (not by ``n``).
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
        unmatched = self.load_unmatched_functions(n_unmatched) if n_unmatched > 0 else []

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
            matched_in_range = len(dataset.get_matched_indices_in_range(self.min_length, self.max_length))
            unmatched_in_range = len(dataset.get_unmatched_indices_in_range(self.min_length, self.max_length))

            stats["binaries"][binary_name] = {
                "total_matched": dataset.matched_count,
                "total_unmatched": dataset.unmatched_count,
                "matched_in_range": matched_in_range,
                "unmatched_in_range": unmatched_in_range,
            }

        return stats
