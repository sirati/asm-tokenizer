"""Per-arm catalog read for the realized-lengths pass.

Single concern: surface, per arm, the two columns the length pass needs
-- the per-section CSR ``var_offsets`` (so the writer can emit the CSR
jump table directly) and the flat per-variant ``var_data_offset_shifted``
column (so the dedup compute can resolve every variant's record offset)
-- by reusing the EXISTING columnar parser
(:func:`tokenizer.aligned_data.matched_sections_columnar.
parse_sections_columnar`) for both arms.

Boundary contract (the design-first sentence):

  *Given the memmap directory + a binary name + an arm, return that
  arm's per-section variant CSR offsets + flat per-variant data-bin
  pointers, parsed once from ``<binary>_sections.bin`` -- no parallel
  parser, no length compute, no file writing.*

The matched arm reuses :func:`...sorted_index._prepass.
read_section_variant_info` (matched-region columnar pre-pass). The
unmatched arm walks the unmatched region of the shared
``<binary>_sections.bin`` for its section starts (catalog order, via the
shared structural walk :func:`...loader._sections_bin_walk.
walk_parsed_sections`) and feeds those starts through the SAME
``parse_sections_columnar`` -- the parser is region-agnostic, so the
unmatched arm needs no bespoke decoder.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader._sections_bin_walk import (
    read_sections_bin_blob,
    unmatched_region_start,
    walk_parsed_sections,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index._prepass import (
    read_section_variant_info,
)

from ._format import MATCHED_ARM, UNMATCHED_ARM, RealizedLengthsArm


__all__ = ["ArmCatalog", "SectionRegion", "read_region_catalog", "read_arm_catalog"]


class SectionRegion(enum.Enum):
    """Which region of the shared ``_sections.bin`` a catalog read covers.

    The two sidecar families (lengths + geometry) both dedup against the
    same two regions; the region -- not the sidecar family -- is the only
    discriminator the catalog read needs, so it is the typed selector
    threaded into :func:`read_region_catalog`.
    """

    MATCHED = "matched"
    UNMATCHED = "unmatched"


#: Same record-offset shift the matched/unmatched arm loaders use:
#: ``real_offset = data_offset_shifted << 4`` (16-byte record alignment).
_DATA_OFFSET_SHIFT = 4


@dataclass(frozen=True)
class ArmCatalog:
    """One arm's per-section CSR + per-variant record offsets.

    ``var_offsets`` is ``int64[n_sections + 1]`` (CSR; ``var_offsets[s]
    : var_offsets[s + 1]`` are section ``s``'s variants in catalog
    order). ``record_offsets`` is ``int64[total_variants]`` -- the real
    ``_data.bin`` byte offset per variant (``data_offset_shifted << 4``).
    """

    var_offsets: np.ndarray
    record_offsets: np.ndarray

    @property
    def n_sections(self) -> int:
        return max(0, int(self.var_offsets.size) - 1)


def _empty_catalog() -> ArmCatalog:
    return ArmCatalog(
        var_offsets=np.zeros(1, dtype=np.int64),
        record_offsets=np.zeros(0, dtype=np.int64),
    )


def read_region_catalog(
    base_path: Path, binary_name: str, region: SectionRegion
) -> ArmCatalog:
    """Read ``region``'s per-section CSR + per-variant record offsets.

    Dispatches to the matched or unmatched region read; both funnel
    through :func:`parse_sections_columnar` so the on-disk variant
    layout is decoded by exactly one parser. Region-keyed so both
    sidecar families (lengths + geometry) share one catalog read.
    """
    base_path = Path(base_path)
    if region is SectionRegion.MATCHED:
        return _read_matched(base_path, binary_name)
    if region is SectionRegion.UNMATCHED:
        return _read_unmatched(base_path, binary_name)
    raise ValueError(f"unknown section region: {region!r}")


#: Maps a length-arm to the region it dedups against. The geometry arms
#: carry their region directly (a :class:`SectionRegion` field), so only
#: the legacy length arm needs this name->region bridge.
_ARM_REGION = {
    MATCHED_ARM.name: SectionRegion.MATCHED,
    UNMATCHED_ARM.name: SectionRegion.UNMATCHED,
}


def read_arm_catalog(
    base_path: Path, binary_name: str, arm: RealizedLengthsArm
) -> ArmCatalog:
    """Read ``arm``'s per-section CSR + per-variant record offsets.

    Thin region-bridge over :func:`read_region_catalog` for the legacy
    realized-length arm (preserves the original API + callers).
    """
    try:
        region = _ARM_REGION[arm.name]
    except KeyError:
        raise ValueError(f"unknown realized-lengths arm: {arm!r}")
    return read_region_catalog(base_path, binary_name, region)


def _from_columns(
    var_offsets: np.ndarray, var_data_offset_shifted: np.ndarray
) -> ArmCatalog:
    record_offsets = (
        np.asarray(var_data_offset_shifted, dtype=np.int64) << _DATA_OFFSET_SHIFT
    )
    return ArmCatalog(
        var_offsets=np.asarray(var_offsets, dtype=np.int64),
        record_offsets=record_offsets,
    )


def _read_matched(base_path: Path, binary_name: str) -> ArmCatalog:
    """Matched-region columnar pre-pass (reused from sorted_index)."""
    info = read_section_variant_info(base_path, binary_name)
    if info.counts.size == 0:
        return _empty_catalog()
    cols = info.cols
    return _from_columns(cols.var_offsets, cols.var_data_offset_shifted)


def _read_unmatched(base_path: Path, binary_name: str) -> ArmCatalog:
    """Unmatched-region columnar read.

    Walks the unmatched region of the shared ``_sections.bin`` for its
    section starts (catalog order) via the shared structural walk, then
    decodes those starts through the shared columnar parser. No
    function-name resolution is involved -- that is the loader's concern,
    not the length pass's.
    """
    matched_index = base_path / f"{binary_name}_index.bin"
    sections_bin = base_path / f"{binary_name}_sections.bin"
    if not sections_bin.exists():
        return _empty_catalog()
    region_start = unmatched_region_start(matched_index)
    raw, blob_view = read_sections_bin_blob(sections_bin)
    # This arm decodes the region with the vectorised columnar parser, so
    # it needs only the section START offsets from the structural walk; the
    # ``Section`` the walk threads out is consumed by the columnar pass's
    # caller, not here. (The walk's own ``parse_section_bin`` is the single
    # boundary-finding parse; the columnar parser below is a separate,
    # vectorised decoder over the same starts.)
    section_starts = [start for start, _section in
                      walk_parsed_sections(blob_view, region_start)]
    if not section_starts:
        return _empty_catalog()
    # ``parse_sections_columnar`` wants a uint8 ndarray; ``raw`` is the
    # ``np.memmap(uint8)`` backing the section-start offsets the walk just
    # produced, so ``np.asarray`` is a zero-copy view that keeps the read
    # lazy (the columnar parser pages in only the bytes it indexes).
    blob = np.asarray(raw)
    cols = parse_sections_columnar(
        blob, np.asarray(section_starts, dtype=np.int64)
    )
    return _from_columns(cols.var_offsets, cols.var_data_offset_shifted)
