"""Per-binary dataset shell.

Single concern: own the per-binary paths and metadata arms, and hand callers
a ``BinarySession`` that bundles the three handle-lifetime concerns
(sections CSV + ``_data.bin`` memmap + ``_variants.bin`` memmap).

Loading concerns live elsewhere:
    * SectionArm assembly ........ ``metadata_loader``
    * Slim-CSV decode ............ ``variant_resolver.load_variants_offset_to_filename``
    * Function-names sidecar ..... ``function_names_loader.load_function_names``
    * Handle lifecycle + slicing . ``session.BinarySession``
    * Parser glue ................ ``_session_parsers``

The legacy ``_versions.json`` sidecar plumbing is gone — variant identity
is now resolved through the ``_variants.bin`` + slim ``_variants.csv`` pair
via ``BinarySession.get_variant_by_ref``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .function_data import FunctionData
from .function_names_loader import load_function_names
from .matched_function import MatchedFunction
from .metadata_loader import (
    BinaryArmPaths,
    SectionArm,
    SectionKind,
    load_section_arm,
)
from .session import BinarySession
from .variant_resolver import load_variants_offset_to_filename


class BinaryDataset:
    """Manages data for a single binary (matched + unmatched arms).

    Holds derived metadata (length-banded edge indices, per-row offsets)
    eagerly so length-based candidate sampling is O(1); defers file-handle
    work to ``open_session()`` so the three handles stay open only for the
    duration of an in-progress batch.
    """

    def __init__(
        self,
        base_path: Path,
        binary_name: str,
        vocab_manager: Optional[Any] = None,
    ):
        self.base_path = Path(base_path)
        self.binary_name = binary_name
        self.vocab_manager = vocab_manager

        # Public path attributes consumed by callers
        # (validator: ``unmatched_sections.exists()`` etc.).
        self.matched_sections = self.base_path / f"{binary_name}_sections.csv"
        self.matched_data = self.base_path / f"{binary_name}_data.bin"
        self.matched_index = self.base_path / f"{binary_name}_index.bin"
        self.unmatched_sections = self.base_path / f"{binary_name}_unmatched_sections.csv"
        self.unmatched_data = self.base_path / f"{binary_name}_unmatched_data.bin"
        self.unmatched_index = self.base_path / f"{binary_name}_unmatched_index.bin"
        self.variants_sidecar = self.base_path / f"{binary_name}_variants.csv"
        self.function_names_sidecar = (
            self.base_path / f"{binary_name}_function_names.txt"
        )

        # Function-names sidecar: hard cutover. Required whenever either
        # arm exists; both arms' section CSVs reference names through
        # base64 line numbers into this file. A missing or bad-prelude
        # sidecar raises ValueError with a migration-pointing message
        # via ``load_function_names``. The all-arms-empty path skips it
        # to keep the loader usable on a brand-new (empty) output dir.
        if self.matched_index.exists() or self.unmatched_index.exists():
            self.name_to_line, self.line_to_name = load_function_names(
                self.function_names_sidecar
            )
        else:
            self.name_to_line, self.line_to_name = {}, {}

        # Build both section arms via the shared dispatch (single
        # implementation in ``metadata_loader``).
        self._matched_arm: SectionArm = load_section_arm(
            SectionKind.MATCHED,
            self._arm_paths(SectionKind.MATCHED),
            self.line_to_name,
        )
        self._unmatched_arm: SectionArm = load_section_arm(
            SectionKind.UNMATCHED,
            self._arm_paths(SectionKind.UNMATCHED),
            self.line_to_name,
        )
        self._publish_arm("matched", self._matched_arm)
        self._publish_arm("unmatched", self._unmatched_arm)

        # Slim ``_variants.csv`` is small (variants per binary count in the
        # dozens-to-hundreds); read once and cache the offset→filename map
        # so every session shares one parse.
        self._offset_to_filename: Dict[int, str] = (
            load_variants_offset_to_filename(self.variants_sidecar)
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _arm_paths(self, kind: SectionKind) -> BinaryArmPaths:
        """Bundle the per-arm path triple. Single switch point — the rest
        of the shell only sees ``BinaryArmPaths``.
        """
        if kind is SectionKind.MATCHED:
            return BinaryArmPaths(
                sections_csv=self.matched_sections,
                index_bin=self.matched_index,
                data_bin=self.matched_data,
            )
        return BinaryArmPaths(
            sections_csv=self.unmatched_sections,
            index_bin=self.unmatched_index,
            data_bin=self.unmatched_data,
        )

    def _publish_arm(self, attr_prefix: str, arm: SectionArm) -> None:
        """Mirror a SectionArm onto the legacy ``self.<prefix>_*`` public
        attributes that the validator + utils + tests read directly. New
        consumers should read from ``self._matched_arm`` / ``_unmatched_arm``
        instead.

        Post self-describing record header, ``<prefix>_starts`` is the
        per-RECORD ``_data.bin`` offset (matched = per-variant flat,
        unmatched = per-function); records are self-describing so no
        companion ``_lengths`` / ``_is_overlong`` / ``_avg_lengths``
        array exists. The ``<prefix>_csv_starts`` / ``_csv_lengths``
        mirror the per-function CSV-section locator for the matched
        arm (empty on the unmatched arm where rows are single-line).
        """
        setattr(self, f"{attr_prefix}_starts", arm.starts)
        setattr(self, f"{attr_prefix}_edge_indices", arm.edge_indices)
        setattr(self, f"{attr_prefix}_count_per_length", arm.count_per_length)
        setattr(self, f"{attr_prefix}_func_names", arm.func_names)
        setattr(self, f"{attr_prefix}_count", arm.count)
        setattr(self, f"{attr_prefix}_section_starts", arm.section_starts)
        setattr(self, f"{attr_prefix}_csv_starts", arm.csv_starts)
        setattr(self, f"{attr_prefix}_csv_lengths", arm.csv_lengths)

    # ------------------------------------------------------------------
    # Session API
    # ------------------------------------------------------------------
    def open_session(self) -> BinarySession:
        """Return a fresh ``BinarySession`` bound to this binary.

        The session lazily opens (and on ``__exit__`` closes) the
        three per-binary handles. Batches that touch many functions on
        one binary enter ONE session and call ``load_matched`` /
        ``load_unmatched`` / ``get_variant_by_ref`` inside it; the three
        handles are shared across every call.
        """
        return BinarySession(
            base_path=self.base_path,
            binary_name=self.binary_name,
            vocab_manager=self.vocab_manager,
            metadata={
                "matched_arm": self._matched_arm,
                "unmatched_arm": self._unmatched_arm,
                "offset_to_filename": self._offset_to_filename,
                "line_to_name": self.line_to_name,
                "name_to_line": self.name_to_line,
            },
        )

    # ------------------------------------------------------------------
    # One-shot wrappers — preserve single-call semantics for notebook /
    # script callers; each enters and exits a session of its own.
    # ------------------------------------------------------------------
    def load_matched_function(self, idx: int) -> MatchedFunction:
        with self.open_session() as sess:
            return sess.load_matched(idx)

    def load_unmatched_function(self, idx: int) -> FunctionData:
        with self.open_session() as sess:
            return sess.load_unmatched(idx)

    def get_variant_by_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        with self.open_session() as sess:
            return sess.get_variant_by_ref(ref)

    # ------------------------------------------------------------------
    # Length-banded candidate sampling
    # ------------------------------------------------------------------
    def get_matched_indices_by_length(
        self, target_length: int, min_count: int = 1
    ) -> np.ndarray:
        """Indices of matched functions at or near ``target_length``.

        Expands the search radius until at least ``min_count`` candidates
        are found, capped at radius=1000 to prevent runaway scans on a
        very-narrow length distribution.
        """
        return _expand_length_band(
            self._matched_arm, target_length, min_count
        )

    def get_unmatched_indices_by_length(
        self, target_length: int, min_count: int = 1
    ) -> np.ndarray:
        return _expand_length_band(
            self._unmatched_arm, target_length, min_count
        )

    def get_matched_indices_in_range(
        self, min_len: int, max_len: int
    ) -> np.ndarray:
        return _band_indices_in_range(self._matched_arm, min_len, max_len)

    def get_unmatched_indices_in_range(
        self, min_len: int, max_len: int
    ) -> np.ndarray:
        return _band_indices_in_range(self._unmatched_arm, min_len, max_len)


# --------------------------------------------------------------------------
# Length-band primitives. Both arms use the same edge-indices table shape,
# so the slicing math is shared here instead of duplicated per arm.
# --------------------------------------------------------------------------
def _expand_length_band(
    arm: SectionArm, target_length: int, min_count: int
) -> np.ndarray:
    """O(1) lookup expanded geometrically until ``min_count`` is met."""
    if arm.count == 0:
        return np.array([], dtype=np.int32)

    max_available_length = len(arm.count_per_length) - 1
    search_radius = 0
    max_radius = 1000
    while search_radius <= max_radius:
        min_len = max(0, target_length - search_radius)
        max_len = min(max_available_length, target_length + search_radius)
        start_idx, end_idx = _band_edges(arm, min_len, max_len)
        count = end_idx - start_idx
        if count >= min_count:
            return np.arange(start_idx, end_idx, dtype=np.int32)
        search_radius = 16 if search_radius == 0 else int(search_radius * 1.5)
    return np.arange(arm.count, dtype=np.int32)


def _band_indices_in_range(
    arm: SectionArm, min_len: int, max_len: int
) -> np.ndarray:
    if arm.count == 0:
        return np.array([], dtype=np.int32)
    max_available = len(arm.edge_indices) - 2
    min_len = max(0, min(min_len, max_available))
    max_len = max(0, min(max_len, max_available))
    start_idx, end_idx = _band_edges(arm, min_len, max_len)
    return np.arange(start_idx, end_idx, dtype=np.int32)


def _band_edges(arm: SectionArm, min_len: int, max_len: int) -> tuple[int, int]:
    """Edge-indices slice for ``[min_len, max_len]`` clamped to arm bounds."""
    edges = arm.edge_indices
    count = arm.count
    start_idx = int(edges[min_len]) if min_len < len(edges) else count
    end_idx = int(edges[max_len + 1]) if max_len + 1 < len(edges) else count
    return start_idx, end_idx
