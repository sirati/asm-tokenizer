"""Open the body-free + body geometry handles for the vectorized path.

Single concern: bundle, for one binary's ONE arm (matched OR unmatched),
the handles the geometry prepass (plan C1) + the fused scatter (plan C2)
consume -- the columnar ``sections.bin`` catalog of that arm's region +
its ``section_offsets``, the RLG3 realized-geometry reader for that arm,
the shared ``_variants.bin`` prefix bytes, and the arm's ``_data.bin``
body bytes -- opened through the SAME readers the index build uses, never
a bespoke BIN parse. A second helper (:func:`open_vector_batch_arm_set`)
opens BOTH arms at once into one keyed :class:`VectorBatchArmSet` for the
entry orchestrator's per-arm dispatch.

The handles are all lazy / mmap views (house rules); :meth:`close`
releases the geometry reader + the memmaps deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict

import numpy as np

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader._sections_bin_walk import SectionRegion
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.realized_lengths import RealizedGeometryReader
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_ARMS,
    GEOMETRY_MATCHED_ARM,
    RealizedGeometryArm,
)
from tokenizer.aligned_data.sorted_index._prepass import (
    read_region_section_variant_info_lazy,
)


__all__ = [
    "VectorBatchHandles",
    "VectorBatchArmSet",
    "open_vector_batch_handles",
    "open_vector_batch_arm_set",
]


#: The two geometry arms keyed by the :class:`SectionKind` a sampled
#: pointer carries -- the single bridge between the sampler's arm tag and
#: the geometry-arm's filename suffixes / region. ``MATCHED`` pointers use
#: the matched arm's sidecars, ``UNMATCHED`` pointers the unmatched arm's.
_ARM_BY_KIND: Dict[SectionKind, RealizedGeometryArm] = {
    SectionKind.MATCHED: next(
        a for a in GEOMETRY_ARMS if a.region is SectionRegion.MATCHED
    ),
    SectionKind.UNMATCHED: next(
        a for a in GEOMETRY_ARMS if a.region is SectionRegion.UNMATCHED
    ),
}


@dataclass(frozen=True)
class VectorBatchHandles:
    """The opened geometry + body handles for one binary, one arm.

    ``cols`` / ``section_offsets`` index by THIS arm's section position
    (the region's catalog order, parallel to the RLG3 axes and to
    :class:`BinarySession`'s ``load_matched`` / ``load_unmatched`` index
    space). ``geometry`` is this arm's RLG3 reader; ``variants_u8`` is the
    SHARED ``_variants.bin`` (both arms' ``var_ref_offset`` index it);
    ``data_u8`` is this arm's ``_data.bin`` region, all read-only uint8
    memmap views.
    """

    cols: ColumnarSections
    section_offsets: np.ndarray
    geometry: RealizedGeometryReader
    variants_u8: np.ndarray
    data_u8: np.ndarray

    @cached_property
    def adjacency(self) -> LiveNodeAdjacency:
        """The arm's splice adjacency, built once per binary.

        The adjacency (its offset->idx hashmap + the per-binary MISSING
        inventory scan) is a pure function of ``cols`` / ``section_offsets``
        -- both fixed for this binary's lifetime -- so it is constructed
        once here and reused across every batch's geometry prepass rather
        than rebuilt (and re-scanned) per :func:`compute_batch_geometry`
        call. Memoised on first access; the frozen handle's fields stay
        immutable (the cache lands in ``__dict__``).
        """
        return LiveNodeAdjacency(
            self.cols, self.section_offsets, self.cols.sec_of_var
        )

    def close(self) -> None:
        """Release the geometry reader + the memmap views."""
        self.geometry.close()
        for arr in (self.variants_u8, self.data_u8, self.section_offsets):
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "VectorBatchHandles":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@dataclass(frozen=True)
class VectorBatchArmSet:
    """Both arms' handle bundles for one binary, keyed by :class:`SectionKind`.

    The entry orchestrator groups its resolved batch rows by arm and
    runs the geometry -> scatter -> dense pass once per arm against the
    matching bundle. A bundle whose arm is absent from the corpus simply
    yields an empty geometry for that arm's (empty) row group -- the
    grouping never dispatches a non-existent arm.
    """

    by_kind: Dict[SectionKind, VectorBatchHandles]

    def __getitem__(self, kind: SectionKind) -> VectorBatchHandles:
        return self.by_kind[kind]

    def close(self) -> None:
        for handles in self.by_kind.values():
            handles.close()

    def __enter__(self) -> "VectorBatchArmSet":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_vector_batch_handles(
    base_path: Path,
    binary_name: str,
    arm: RealizedGeometryArm = GEOMETRY_MATCHED_ARM,
) -> VectorBatchHandles:
    """Open ``arm``'s geometry + body handles for ``binary_name``.

    Parameters
    ----------
    base_path / binary_name:
        The memmap directory + binary stem (the same keys the index
        build + :class:`BinarySession` use).
    arm:
        Which geometry arm to open (``GEOMETRY_MATCHED_ARM`` /
        ``GEOMETRY_UNMATCHED_ARM``). Carries the region the columnar
        catalog is read over AND the ``_data.bin`` suffix this arm
        dedups against, so the matched-vs-unmatched filename pair is
        never hand-rolled here. Defaults to the matched arm.

    Returns
    -------
    VectorBatchHandles
        The arm's columnar catalog + section offsets, the arm's RLG3
        reader, the SHARED ``_variants.bin`` uint8 memmap, and the arm's
        ``_data.bin`` uint8 memmap.
    """
    base_path = Path(base_path)
    # Open the catalog LAZILY (matched arm: section-bounded on-demand fill;
    # unmatched arm: eager, it is cheap). The decode path touches <=5% of
    # sections, so the full matched columnar parse (~1.9 s on z3) is bounded
    # to the sampled set's BFS closure instead of paid in full at open.
    info = read_region_section_variant_info_lazy(
        base_path, binary_name, arm.region
    )
    geometry = RealizedGeometryReader.open(base_path, binary_name, arm)
    # An ABSENT ``_variants.bin`` is valid: the session's variant resolver
    # treats it as "no variant-prefix records" (empty ``variant_tokens``).
    # Hand the prefix readers an empty buffer so they mirror that exactly
    # (see ``_prefix`` / ``_prefix_values`` empty-buffer handling). The
    # file is SHARED across arms -- both arms' ``var_ref_offset`` index it.
    variants_path = base_path / f"{binary_name}_variants.bin"
    variants_u8 = (
        np.memmap(variants_path, dtype=np.uint8, mode="r")
        if variants_path.exists()
        else np.zeros(0, dtype=np.uint8)
    )
    data_u8 = np.memmap(
        arm.data_path(base_path, binary_name), dtype=np.uint8, mode="r"
    )
    return VectorBatchHandles(
        cols=info.cols,
        section_offsets=np.asarray(info.section_offsets, dtype=np.int64),
        geometry=geometry,
        variants_u8=variants_u8,
        data_u8=data_u8,
    )


def open_vector_batch_arm_set(
    base_path: Path, binary_name: str
) -> VectorBatchArmSet:
    """Open BOTH arms' handle bundles, keyed by :class:`SectionKind`.

    The entry orchestrator's per-arm dispatch consumes this; each arm's
    bundle is opened via :func:`open_vector_batch_handles` so there is
    one columnar-read + one geometry-open concern per arm. Closing the
    set closes every bundle.
    """
    base_path = Path(base_path)
    by_kind = {
        kind: open_vector_batch_handles(base_path, binary_name, arm)
        for kind, arm in _ARM_BY_KIND.items()
    }
    return VectorBatchArmSet(by_kind=by_kind)
