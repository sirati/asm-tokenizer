"""Cross-binary unbiased sampler over per-binary sorted-index readers.

:class:`MultiBinarySortedIndexSampler` and the free
:func:`sample_section_pointers` implement a Reading-A unbiased without-
replacement sample over per-binary :class:`SortedIndexReader` urns
(plan ALG-3 + D6).

Binary ordering is canonical alphabetical -- the sampler's
:attr:`MultiBinarySortedIndexSampler.binary_names` and internal per-
binary iteration order ALL use alphabetical order so the downstream
per-row ``binary_id`` numbering is stable across runs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from .._reader import SortedIndexReader
from .._types import MultiBinarySectionPointer


__all__ = [
    "MultiBinarySortedIndexSampler",
    "sample_section_pointers",
]


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
    total = sum(per_binary_counts.values())
    if total == 0:
        return []
    k = min(count, total)

    binary_names = list(per_binary_counts)
    counts_arr = np.array(
        [per_binary_counts[n] for n in binary_names], dtype=np.int64,
    )
    drawn = rng.multivariate_hypergeometric(counts_arr, k)

    out: List[MultiBinarySectionPointer] = []
    for name, draw in zip(binary_names, drawn):
        draw_int = int(draw)
        if draw_int == 0:
            continue
        if band is not None:
            lo, hi = band
            idxs = readers[name].sample_section_indices_in_band(
                lo, hi, draw_int, rng,
            )
        else:
            idxs = readers[name].sample_section_indices(
                target_length, draw_int, rng,
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
