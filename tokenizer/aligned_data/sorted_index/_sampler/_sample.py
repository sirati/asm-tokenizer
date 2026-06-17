"""Cross-binary unbiased sampler over per-binary sorted-index readers.

:class:`MultiBinarySortedIndexSampler` and the free
:func:`sample_section_pointers` implement a Reading-A unbiased without-
replacement sample over per-binary :class:`SortedIndexReader` urns
(plan ALG-3 + D6). :class:`CrossSpecSortedIndexSampler` is the strict
generalisation whose urn cells are ``(binary, spec)`` over EVERY
configured spec's readers, so one draw is unbiased across both the
binary AND the depth axis at once.

Binary ordering is canonical alphabetical -- the sampler's
:attr:`MultiBinarySortedIndexSampler.binary_names` and internal per-
binary iteration order ALL use alphabetical order so the downstream
per-row ``binary_id`` numbering is stable across runs.

The without-replacement urn draw itself (sum the per-cell pool sizes,
draw one ``multivariate_hypergeometric`` across all cells) is the SAME
math regardless of what keys the cells are -- binary, or ``(binary,
spec)``. It lives in exactly one place, :func:`_draw_per_cell_counts`,
which both samplers call; widening the cell key is a pure generalisation,
never a second copy of the hypergeometric logic.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from .._reader import SortedIndexReader
from .._types import IndexSpec, MultiBinarySectionPointer


__all__ = [
    "MultiBinarySortedIndexSampler",
    "CrossSpecSortedIndexSampler",
    "sample_section_pointers",
]


# ---------------------------------------------------------------------------
# Shared without-replacement urn draw (the ONE hypergeometric step)
# ---------------------------------------------------------------------------


def _draw_per_cell_counts(
    counts_by_key: Dict[Hashable, int],
    k: int,
    rng: np.random.Generator,
) -> Dict[Hashable, int]:
    """Exact without-replacement draw of ``k`` items across labelled urns.

    ``counts_by_key`` maps each urn cell (an arbitrary hashable key --
    a binary name, or a ``(binary, spec)`` pair) to its pool size. Draws
    ``min(k, total)`` items across all cells via a single
    :meth:`numpy.random.Generator.multivariate_hypergeometric` (the exact
    without-replacement multivariate draw), and returns the per-cell draw
    count keyed the same way. Cells that drew zero are omitted.

    Empty total (every cell size 0) returns ``{}``. The cell iteration
    order is ``counts_by_key``'s insertion order, so a caller wanting
    cross-run-stable output passes a canonically-ordered dict (both
    samplers do). This is the SINGLE owner of the hypergeometric step; no
    caller re-implements it.
    """
    total = sum(counts_by_key.values())
    if total == 0:
        return {}
    draw_count = min(k, total)
    keys = list(counts_by_key)
    counts_arr = np.array(
        [counts_by_key[key] for key in keys], dtype=np.int64,
    )
    drawn = rng.multivariate_hypergeometric(counts_arr, draw_count)
    return {
        key: int(draw)
        for key, draw in zip(keys, drawn)
        if int(draw) > 0
    }


# ---------------------------------------------------------------------------
# Cross-binary unbiased sampler (Reading A, plan ALG-3 + D6)
# ---------------------------------------------------------------------------


class MultiBinarySortedIndexSampler:
    """Stateful Reading-A sampler binding a fixed set of per-binary readers.

    Construction canonicalises the per-binary order to alphabetical
    ``binary_name`` so downstream consumers (notably :func:`_concat_results`
    and :func:`open_length_bucketed_batch`) can rely on stable per-row
    ``binary_id`` numbering across runs.

    The class is intentionally thin: every per-call sampling decision is
    delegated to :func:`sample_section_pointers` so the algorithm is
    testable in isolation without a sampler instance.
    """

    def __init__(self, readers: Dict[str, SortedIndexReader]) -> None:
        ordered_names = sorted(readers)
        # Preserve only the alphabetical-order dict so per-binary
        # iteration in :func:`sample_section_pointers` and
        # :func:`open_length_bucketed_batch` matches the ordering
        # exposed by :attr:`binary_names`.
        self._readers: Dict[str, SortedIndexReader] = {
            name: readers[name] for name in ordered_names
        }
        self._binary_names: List[str] = ordered_names

    @property
    def binary_names(self) -> List[str]:
        """Alphabetical ``binary_name`` list -- the ``binary_id`` reverse map."""
        return list(self._binary_names)

    def count_at(self, target_length: int) -> int:
        """Pool size at ``target_length`` summed over every binary."""
        return sum(r.count_at(target_length) for r in self._readers.values())

    def count_in_band(self, lo: int, hi: int) -> int:
        """Pool size for lengths in ``[lo, hi]`` summed over every binary."""
        return sum(r.count_in_band(lo, hi) for r in self._readers.values())

    def sample_section_pointers(
        self,
        target_length: int,
        count: int,
        rng: np.random.Generator,
        *,
        band: Optional[Tuple[int, int]] = None,
    ) -> List[MultiBinarySectionPointer]:
        """Delegate to :func:`sample_section_pointers` with our readers.

        When ``band=(lo, hi)`` is provided, eligible sections are those
        whose index key falls in ``[lo, hi]`` rather than exactly at
        ``target_length`` (length-band sampling).
        """
        return sample_section_pointers(
            self._readers, target_length, count, rng, band=band,
        )


class CrossSpecSortedIndexSampler:
    """Unbiased Reading-A sampler over ``(binary, spec)`` urn cells.

    The strict generalisation of :class:`MultiBinarySortedIndexSampler`:
    instead of one urn cell per binary, there is one cell per
    ``(binary_name, spec)`` over EVERY configured spec's readers. A single
    without-replacement draw (:func:`_draw_per_cell_counts`) therefore
    spans the binary AND the depth axis at once, so a row is no more
    likely to come from one depth than another beyond what each cell's
    pool size warrants -- the design's "no d3 dominance" property.

    Each emitted :class:`MultiBinarySectionPointer` carries the ``spec``
    it was drawn from, so the cross-depth load path can read each row's
    ``max_depth`` straight off ``spec.depth``.

    Canonical cell order = alphabetical ``binary_name``, then
    :meth:`IndexSpec.sort_key` -- mirroring
    :class:`MultiBinarySortedIndexSampler`'s alphabetical canon and the
    collection's :func:`._spec.sorted_specs` order, so the draw is
    deterministic across runs for a fixed seed.
    """

    def __init__(
        self,
        readers_by_spec: Dict[IndexSpec, Dict[str, SortedIndexReader]],
    ) -> None:
        # Canonical cell order: alphabetical binary, then spec sort_key.
        # Membership is uniform across specs (every binary carries every
        # spec's .idx), but we union the names defensively so a partially
        # populated dict still orders cleanly.
        specs = sorted(readers_by_spec, key=IndexSpec.sort_key)
        names = sorted(
            {name for readers in readers_by_spec.values() for name in readers}
        )
        # Cells ordered binary-major, spec-minor.
        self._cells: List[Tuple[str, IndexSpec]] = [
            (name, spec)
            for name in names
            for spec in specs
            if name in readers_by_spec[spec]
        ]
        self._readers_by_spec = readers_by_spec
        self._specs: List[IndexSpec] = specs
        self._binary_names: List[str] = names

    @property
    def binary_names(self) -> List[str]:
        """Alphabetical ``binary_name`` list (union across specs)."""
        return list(self._binary_names)

    @property
    def specs(self) -> List[IndexSpec]:
        """Configured specs in canonical :meth:`IndexSpec.sort_key` order."""
        return list(self._specs)

    def _reader(self, name: str, spec: IndexSpec) -> SortedIndexReader:
        return self._readers_by_spec[spec][name]

    def _cell_count(
        self,
        name: str,
        spec: IndexSpec,
        target_length: int,
        band: Optional[Tuple[int, int]],
    ) -> int:
        reader = self._reader(name, spec)
        if band is not None:
            lo, hi = band
            return reader.count_in_band(lo, hi)
        return reader.count_at(target_length)

    def count_at(self, target_length: int) -> int:
        """Pool size at ``target_length`` summed over EVERY ``(binary, spec)``."""
        return sum(
            self._cell_count(name, spec, target_length, None)
            for name, spec in self._cells
        )

    def count_in_band(self, lo: int, hi: int) -> int:
        """Pool size in ``[lo, hi]`` summed over EVERY ``(binary, spec)``."""
        return sum(
            self._cell_count(name, spec, 0, (lo, hi))
            for name, spec in self._cells
        )

    def sample_section_pointers(
        self,
        target_length: int,
        count: int,
        rng: np.random.Generator,
        *,
        band: Optional[Tuple[int, int]] = None,
    ) -> List[MultiBinarySectionPointer]:
        """Unbiased without-replacement sample across all ``(binary, spec)`` cells.

        Per-cell pool sizes feed the SHARED :func:`_draw_per_cell_counts`
        urn draw; each drawn cell then contributes a uniform within-cell
        index draw (:func:`_sample_cell_indices`), and every emitted
        pointer is stamped with its ``spec``. The combined output is
        shuffled (same anti-clustering as the per-binary sampler; here it
        also de-blocks the depth axis).
        """
        counts_by_cell = {
            cell: self._cell_count(cell[0], cell[1], target_length, band)
            for cell in self._cells
        }
        drawn = _draw_per_cell_counts(counts_by_cell, count, rng)

        out: List[MultiBinarySectionPointer] = []
        for (name, spec), draw_int in drawn.items():
            idxs = _sample_cell_indices(
                self._reader(name, spec),
                target_length,
                draw_int,
                rng,
                band=band,
            )
            out.extend(
                MultiBinarySectionPointer(
                    binary_name=name,
                    section_pointer=SectionPointerSpec(
                        arm=SectionKind.MATCHED, idx=int(i),
                    ),
                    spec=spec,
                )
                for i in idxs
            )
        rng.shuffle(out)
        return out


def sample_section_pointers(
    readers: Dict[str, SortedIndexReader],
    target_length: int,
    count: int,
    rng: np.random.Generator,
    *,
    band: Optional[Tuple[int, int]] = None,
) -> List[MultiBinarySectionPointer]:
    """Reading-A unbiased sample over per-binary urns.

    Each ``(binary, section_idx)`` pair at ``target_length`` is equally
    likely; larger binaries contribute proportionally more samples
    (plan D6). Implementation:

    1. Sum each binary's ``count_at(target_length)`` into a per-binary
       urn size vector.
    2. Draw a per-binary count vector via
       :meth:`numpy.random.Generator.multivariate_hypergeometric` --
       exact without-replacement draw across all urns.
    3. Per binary, call ``sample_section_indices`` with the drawn
       count and build :class:`MultiBinarySectionPointer` rows.
    4. Shuffle the combined output to break per-binary row clustering
       (otherwise downstream batches would have a deterministic
       per-binary block layout that leaks the sampler's per-urn order).

    Empty pool (``total == 0``) returns ``[]``; the helper does NOT
    raise -- the caller (:func:`open_length_bucketed_batch`) is the
    layer that raises :class:`ValueError` to surface this to training
    loops.

    ``readers`` is iterated in dict-insertion order; callers wanting
    stable cross-run output should pass an alphabetically-canonical
    dict (:class:`MultiBinarySortedIndexSampler` does so internally).

    When ``band=(lo, hi)`` is provided, sections with index key in
    ``[lo, hi]`` (inclusive) are eligible rather than exactly
    ``target_length``.  The per-binary urn sizes use
    :meth:`SortedIndexReader.count_in_band`; sampling draws via
    :meth:`SortedIndexReader.sample_section_indices_in_band`.
    """
    if band is not None:
        lo, hi = band
        per_binary_counts = {
            name: rdr.count_in_band(lo, hi) for name, rdr in readers.items()
        }
    else:
        per_binary_counts = {
            name: rdr.count_at(target_length) for name, rdr in readers.items()
        }
    drawn = _draw_per_cell_counts(per_binary_counts, count, rng)

    out: List[MultiBinarySectionPointer] = []
    for name, draw_int in drawn.items():
        idxs = _sample_cell_indices(
            readers[name], target_length, draw_int, rng, band=band,
        )
        out.extend(
            MultiBinarySectionPointer(
                binary_name=name,
                section_pointer=SectionPointerSpec(
                    arm=SectionKind.MATCHED, idx=int(i),
                ),
            )
            for i in idxs
        )

    rng.shuffle(out)
    return out


def _sample_cell_indices(
    reader: SortedIndexReader,
    target_length: int,
    count: int,
    rng: np.random.Generator,
    *,
    band: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Draw ``count`` section indices from one reader's pool.

    Routes to :meth:`SortedIndexReader.sample_section_indices_in_band`
    when a ``band`` is given, else :meth:`sample_section_indices` at the
    exact ``target_length`` -- the per-reader within-cell uniform draw
    both samplers share once the cross-cell counts are decided.
    """
    if band is not None:
        lo, hi = band
        return reader.sample_section_indices_in_band(lo, hi, count, rng)
    return reader.sample_section_indices(target_length, count, rng)
