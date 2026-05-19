"""
BinaryDataset class for managing data for a single binary.

This module handles loading and indexing of matched and unmatched functions
for a single binary, with efficient length-based lookups using pre-computed
edge indices.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tokenizer.variant_info import VariantInfo

from ..io import read_function_data_memmap
from ..metadata import extract_metadata_from_section_row
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .metadata_loader import (
    BinaryArmPaths,
    SectionArm,
    SectionKind,
    build_length_lookup_tables,
    load_index_once,
    load_section_arm,
    load_unmatched_lengths,
    open_sections_csv,
)


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

        self.versions_sidecar = self.base_path / f"{binary_name}_versions.json"
        self.variants_sidecar = self.base_path / f"{binary_name}_variants.csv"

        # Load per-version variant metadata sidecar (legacy fallback: empty list).
        self._versions: List[VariantInfo] = self._load_versions_sidecar()

        # Load the per-group ``_variants.csv`` sidecar that the section
        # CSVs reference by ``0x<hex>`` row index. Positional list of
        # dicts (one per variant); empty when the sidecar is absent
        # (legacy datasets written before the variant-ref schema). The
        # row index is the canonical reference.
        self._variants: List[Dict[str, str]] = self._load_variants_sidecar()

        # Load metadata (NO memmap caching)
        self._load_metadata()

    def _load_versions_sidecar(self) -> List[VariantInfo]:
        """Read ``<binary>_versions.json`` and rebuild a positional list
        of ``VariantInfo`` (list index == ``version_idx``).

        Missing sidecar yields an empty list — legacy datasets without
        the sidecar simply have no per-variant metadata available, and
        ``get_variant_metadata`` returns ``None`` for every index.
        """
        if not self.versions_sidecar.exists():
            return []

        with open(self.versions_sidecar, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Sidecar entries are written in version_idx order; positional
        # reconstruction matches the writer's contract.
        return [
            VariantInfo(
                arch=entry["arch"],
                compiler=entry["compiler"],
                compiler_version=entry["compiler_version"],
                opt=entry["opt"],
                pkg=entry["pkg"],
                variant_id=entry["variant_id"],
                extra_metadata=entry["extra_metadata"],
            )
            for entry in payload
        ]

    def get_variant_metadata(self, version_idx: int) -> Optional[VariantInfo]:
        """Return the ``VariantInfo`` for a given ``version_idx``.

        Returns ``None`` when the sidecar is absent (legacy datasets) or
        when ``version_idx`` is out of range. Negative indices are
        rejected so callers cannot accidentally wrap around to a valid
        end-of-list entry.
        """
        if version_idx < 0 or version_idx >= len(self._versions):
            return None
        return self._versions[version_idx]

    def _load_variants_sidecar(self) -> List[Dict[str, str]]:
        """Read ``<binary>_variants.csv`` and return one dict per row.

        Columns are read from the CSV header; row index in the returned
        list is the canonical ``variant_ref`` line number (parsed from
        ``0x<hex>`` cells in section CSVs). Missing sidecar yields an
        empty list — legacy datasets predate this schema and have no
        variant-ref resolution.
        """
        if not self.variants_sidecar.exists():
            return []
        with open(self.variants_sidecar, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def get_variant_by_ref(
        self, variant_ref: str
    ) -> Optional[Dict[str, str]]:
        """Resolve a section-CSV ``variant_ref`` (``0x<hex>``) to the
        variant's full metadata dict from ``_variants.csv``.

        Returns ``None`` if the sidecar is absent or the parsed line
        index is out of range. Caller is responsible for handling the
        legacy-dataset fallback.
        """
        if not self._variants:
            return None
        try:
            idx = int(variant_ref, 16)
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(self._variants):
            return None
        return self._variants[idx]

    def _open_sections_csv(self, path: Path):
        """Open a ``_sections.csv`` file, transparently consuming a
        ``version=N`` wire-format prelude if one is present.

        Background: the v2 wire format prepends a single-cell
        ``version=2`` row before the header. The per-section byte-offset
        index (``*_index.bin``) is written by the memmap-builder pipeline
        in lockstep with the CSV; today that writer (``passes.py``) does
        not emit a prelude into ``_sections.csv`` and the loader's seek
        offsets land correctly without intervention.

        This helper future-proofs the loader against a v2 ``_sections.csv``
        whose index file was inherited from a v1 layout — i.e. offsets
        are relative to the *content* (post-prelude) start while the CSV
        on disk now begins with a prelude row. The peek consumes that
        prelude and reports its byte width as a *content offset* that
        seek-based callers add to stored offsets. Iteration-based callers
        simply continue reading from the returned handle's current
        position (which is already past the prelude).

        Contract for writers: if the memmap-builder is ever updated to
        emit a v2 prelude into ``_sections.csv``, it MUST keep stored
        offsets relative to the post-prelude content start (i.e. compute
        ``section_start = sections_file.tell()`` BEFORE the first
        function row but AFTER the prelude row). Mixing a
        prelude-stamped CSV with a post-prelude-tell()-based index would
        double-bias every seek. The two valid configurations are:
          1. v1 CSV (no prelude) + v1 index (no bias). Current state.
          2. v2 CSV (with prelude) + v1-content-relative index (prelude
             skipped here, then content offsets applied verbatim).

        Returns:
            (file_handle, content_offset_bytes). ``content_offset_bytes``
            is ``0`` for v1 (no prelude) and the prelude row's byte
            width for v2.
        """
        return open_sections_csv(path)

    def _load_index_once(
        self, index_path: Path
    ):
        """Shim → ``metadata_loader.load_index_once``; retained so
        external callers / tests that still use the dataset method
        name continue to work. The implementation lives next to the
        other index-bin helpers in ``metadata_loader``.
        """
        return load_index_once(index_path)

    def _build_length_lookup_tables(
        self, avg_lengths: np.ndarray, scale_factor: int = 16
    ):
        """Shim → ``metadata_loader.build_length_lookup_tables``."""
        return build_length_lookup_tables(avg_lengths, scale_factor=scale_factor)

    def _load_unmatched_lengths(self) -> np.ndarray:
        """Shim → ``metadata_loader.load_unmatched_lengths``.

        Re-loads the unmatched index inside the helper for the legacy
        signature; the arm-driven ``_load_metadata`` path calls
        ``load_unmatched_lengths`` directly with already-loaded
        ``starts`` / ``lengths`` to avoid a second index pass.
        """
        if not self.unmatched_index.exists():
            return np.array([], dtype=np.int32)
        starts, lengths, _ = load_index_once(self.unmatched_index)
        if starts is None or lengths is None:
            return np.array([], dtype=np.int32)
        paths = BinaryArmPaths(
            sections_csv=self.unmatched_sections,
            index_bin=self.unmatched_index,
            data_bin=self.unmatched_data,
        )
        return load_unmatched_lengths(paths, starts, lengths)

    def _arm_paths(self, kind: SectionKind) -> BinaryArmPaths:
        """Bundle the three on-disk file paths for the requested arm.

        Single switch-point between matched and unmatched path triples;
        the rest of ``_load_metadata`` reads ``kind`` only through this
        helper so no per-arm ``if`` cascade leaks into the loader body.
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
        """Mirror a ``SectionArm`` onto the legacy ``self.<prefix>_*``
        public attributes that downstream consumers read directly
        (``AlignedDataLoader``, ``validator.py``,
        ``utils.load_single_matched_function``). Naming uses
        ``attr_prefix`` ("matched" / "unmatched") so the six fields
        land at their established names with no per-arm branch.
        """
        setattr(self, f"{attr_prefix}_starts", arm.starts)
        setattr(self, f"{attr_prefix}_lengths", arm.lengths)
        setattr(self, f"{attr_prefix}_edge_indices", arm.edge_indices)
        setattr(self, f"{attr_prefix}_count_per_length", arm.count_per_length)
        setattr(self, f"{attr_prefix}_func_names", arm.func_names)
        setattr(self, f"{attr_prefix}_count", arm.count)
        # ``section_starts`` is a new field; only the unmatched arm
        # populates it (matched seeks from ``starts`` directly).
        # Batch 5G consumes it to drop the linear-scan in
        # ``load_unmatched_function``.
        setattr(self, f"{attr_prefix}_section_starts", arm.section_starts)

    def _load_metadata(self):
        """Build matched + unmatched ``SectionArm``s and publish them.

        Delegates per-arm computation to
        ``metadata_loader.load_section_arm`` (single implementation
        shared by both kinds). This method's only remaining concern
        is binding the arms onto the legacy public attribute names.
        """
        matched_arm = load_section_arm(
            SectionKind.MATCHED, self._arm_paths(SectionKind.MATCHED)
        )
        unmatched_arm = load_section_arm(
            SectionKind.UNMATCHED, self._arm_paths(SectionKind.UNMATCHED)
        )
        self._publish_arm("matched", matched_arm)
        self._publish_arm("unmatched", unmatched_arm)
        # Stash the arms so future session code (Batch 5G) can pass
        # them through without re-loading; harmless attribute for
        # current callers.
        self._matched_arm = matched_arm
        self._unmatched_arm = unmatched_arm

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
        f, content_offset = self._open_sections_csv(self.matched_sections)
        try:
            f.seek(start + content_offset)
            section_data = f.read(length)
        finally:
            f.close()

        lines = section_data.strip().split("\n")

        # Parse first line: func_name, called_functions_str
        first_line = list(csv.reader([lines[0]]))[0]
        func_name = first_line[0]

        # Parse versions (remaining lines except empty line at end).
        # New per-row schema collapses the 4-axis canonical tuple
        # into a single ``variant_ref`` ``0x<hex>`` index that points
        # into the per-group ``<binary>_variants.csv`` sidecar.
        versions = []
        reader = csv.reader(lines[1:])
        header = [
            "variant_ref",
            "inlining_data",
            "data_offset",
            "data_len",
        ]

        for row in reader:
            if not row or len(row) == 0:
                continue

            metadata = extract_metadata_from_section_row(row, header)

            # Resolve the 0x<hex> ``variant_ref`` against the per-group
            # ``_variants.csv`` sidecar so callers see the variant's
            # canonical-4 axes / filename / flags directly on the
            # metadata dict (instead of having to round-trip the ref).
            # The raw ``variant_ref`` stays present for consumers that
            # need the index identity. Sidecar-absent (legacy datasets)
            # is a benign no-op.
            variant_row = self.get_variant_by_ref(metadata["variant_ref"])
            if variant_row is not None:
                for k, v in variant_row.items():
                    metadata.setdefault(k, v)

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
            f, _content_offset = self._open_sections_csv(self.unmatched_sections)
            try:
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
            finally:
                f.close()

        # Unmatched section rows carry ``;``-joined variant refs in
        # column 1 (one function present in multiple variants but
        # without a matched canonical form). Resolve every ref against
        # the per-group ``_variants.csv`` sidecar so callers get the
        # list of variants this unmatched function appears in. The
        # raw ``platform_info`` string is kept for consumers that need
        # the unresolved form.
        variant_refs = [r for r in platform_info.split(";") if r] if platform_info else []
        variants = [
            v for v in (self.get_variant_by_ref(r) for r in variant_refs) if v is not None
        ]

        metadata = {
            "arch": "unknown",
            "compiler": "unknown",
            "compilerversion": "unknown",
            "opt": "unknown",
            "platform_info": platform_info,
            "variant_refs": variant_refs,
            "variants": variants,
            "called": called,
            "inlining_data": inlining_data,
            "data_offset": data_offset_from_csv,
            "data_len": data_len_from_csv,
        }

        return FunctionData(func_name, metadata, tokens, insn_rl, block_rl)
