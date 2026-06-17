"""Typed dataclasses + enums for the sorted-index module.

This file defines the typed surface used across the sorted-index
pipeline (build, read, sample, cross-binary batch helper). The types
here are intentionally pure dataclasses + enums with no I/O, no
file-format awareness, and no batch-decode pipeline imports beyond the
two already-typed pieces (``SectionPointerSpec``, ``BatchDecodeResult``)
they extend.

Concerns owned by this module:

- :class:`ReductionKind` + :class:`LengthReduction` -- the typed
  parameter that controls how per-variant lengths collapse to a single
  per-section key length (plan D2).
- :class:`MultiBinarySectionPointer` -- typed cross-binary section
  pointer; replaces ad-hoc ``(binary_name, section_idx)`` tuples
  (plan D7).
- :class:`MultiBinaryBatchDecodeResult` -- extension of
  :class:`BatchDecodeResult` with a per-row ``binary_id`` sidecar so
  cross-binary batches retain provenance (plan D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
)


__all__ = [
    "ReductionKind",
    "LengthReduction",
    "IndexSpec",
    "MultiBinarySectionPointer",
    "MultiBinaryBatchDecodeResult",
]


class ReductionKind(Enum):
    """Discriminator for :class:`LengthReduction`.

    ``MAX`` collapses per-variant lengths via :func:`numpy.ndarray.max`;
    ``PERCENTILE`` collapses via :func:`numpy.percentile` with
    ``method="lower"`` so the result is always one of the input values.
    """

    MAX = "max"
    PERCENTILE = "percentile"


@dataclass(frozen=True)
class LengthReduction:
    """Per-section length aggregator across N variants (plan D2).

    A typed parameter understood by every layer of the sorted-index
    pipeline. ``filename_tag`` produces the canonical CLI/filename
    spelling (``"max"``, ``"p05"``, ``"p95"``) used by the wire-format
    regex; ``reduce`` performs the actual aggregation over a length
    vector for one section.

    Construction validation (:meth:`__post_init__`):

    - :attr:`ReductionKind.MAX` requires ``percentile is None``.
    - :attr:`ReductionKind.PERCENTILE` requires ``percentile`` in
      ``[1, 99]``. ``p100`` collapses to ``MAX`` at the parser
      boundary (:func:`._modes.parse_reduction`) -- direct
      construction with ``percentile=100`` is rejected so the
      canonical form is the only representation.
    """

    kind: ReductionKind
    percentile: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind is ReductionKind.MAX:
            if self.percentile is not None:
                raise ValueError(
                    f"MAX reduction takes no percentile; got {self.percentile!r}",
                )
            return
        # PERCENTILE
        if self.percentile is None:
            raise ValueError("PERCENTILE reduction requires a percentile value")
        if not (1 <= self.percentile <= 99):
            raise ValueError(
                f"percentile must be in [1, 99]; got {self.percentile!r}",
            )

    def filename_tag(self) -> str:
        """Canonical CLI/filename spelling.

        ``"max"`` for :attr:`ReductionKind.MAX`; zero-padded
        ``"p{percentile:02d}"`` for :attr:`ReductionKind.PERCENTILE` so
        files lexsort by percentile.
        """
        if self.kind is ReductionKind.MAX:
            return "max"
        # __post_init__ guarantees percentile is not None for PERCENTILE.
        assert self.percentile is not None
        return f"p{self.percentile:02d}"

    def reduce_segmented(
        self, values: np.ndarray, seg_offsets: np.ndarray
    ) -> np.ndarray:
        """Vectorized :meth:`reduce` over CSR segments.

        ``values`` is the concatenated per-variant lengths;
        ``seg_offsets`` (``int64[S + 1]``, exclusive prefix sums) ties
        each segment (= section) to its slice. Returns ``int64[S]``
        with one reduced key per segment; empty segments yield 0 --
        exactly :meth:`reduce`'s contract, applied per segment (the
        scalar method remains the source of truth; the equivalence
        test pins the two).
        """
        counts = np.diff(seg_offsets)
        out = np.zeros(counts.size, dtype=np.int64)
        nonempty = counts > 0
        if not bool(nonempty.any()):
            return out
        starts = seg_offsets[:-1][nonempty]
        if self.kind is ReductionKind.MAX:
            # Consecutive non-empty starts span exactly their own
            # elements (empty segments hold none), so reduceat over the
            # non-empty starts is safe.
            out[nonempty] = np.maximum.reduceat(values, starts)
            return out
        assert self.percentile is not None
        seg_ids = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        ordered = values[np.lexsort((values, seg_ids))]
        # np.percentile(..., method="lower"): index = floor((n-1) * q/100).
        pick = np.floor(
            (counts[nonempty] - 1) * (self.percentile / 100.0)
        ).astype(np.int64)
        out[nonempty] = ordered[starts + pick]
        return out

    def reduce_grouped_segmented(
        self,
        values: np.ndarray,
        group_keys: np.ndarray,
        seg_offsets: np.ndarray,
    ) -> np.ndarray:
        """Vectorized :meth:`reduce_groups` over CSR segments.

        Within each segment, items sharing a ``group_keys`` value form
        one duplicate-group; each group collapses to its representative
        (group max for MAX, group mean for PERCENTILE -- mirroring
        :meth:`_group_representative`), then the representatives reduce
        per segment like :meth:`reduce_segmented`. Returns
        ``int64[S]``; empty segments yield 0.
        """
        counts = np.diff(seg_offsets)
        n_seg = counts.size
        if values.size == 0:
            return np.zeros(n_seg, dtype=np.int64)
        seg_ids = np.repeat(np.arange(n_seg, dtype=np.int64), counts)
        order = np.lexsort((group_keys, seg_ids))
        sv = values[order].astype(np.float64)
        sk = group_keys[order]
        ss = seg_ids[order]
        is_group_start = np.ones(sv.size, dtype=bool)
        is_group_start[1:] = (sk[1:] != sk[:-1]) | (ss[1:] != ss[:-1])
        group_starts = np.nonzero(is_group_start)[0]
        group_seg = ss[group_starts]
        if self.kind is ReductionKind.MAX:
            reps = np.maximum.reduceat(sv, group_starts)
        else:
            sums = np.add.reduceat(sv, group_starts)
            group_counts = np.diff(
                np.concatenate([group_starts, [sv.size]])
            )
            reps = sums / group_counts
        groups_per_seg = np.bincount(group_seg, minlength=n_seg)
        rep_offsets = np.zeros(n_seg + 1, dtype=np.int64)
        np.cumsum(groups_per_seg, out=rep_offsets[1:])
        out = np.zeros(n_seg, dtype=np.int64)
        nonempty = groups_per_seg > 0
        starts = rep_offsets[:-1][nonempty]
        if self.kind is ReductionKind.MAX:
            # int(float_max) truncates toward zero, same as astype.
            out[nonempty] = np.maximum.reduceat(reps, starts).astype(
                np.int64
            )
            return out
        assert self.percentile is not None
        ordered = reps[np.lexsort((reps, group_seg))]
        pick = np.floor(
            (groups_per_seg[nonempty] - 1) * (self.percentile / 100.0)
        ).astype(np.int64)
        out[nonempty] = ordered[starts + pick].astype(np.int64)
        return out

    def reduce(self, lengths: np.ndarray) -> int:
        """Collapse a vector of per-variant lengths to one int key.

        Empty input -> ``0`` (0-variant sections stamp 0 directly;
        see plan ALG-1).
        """
        if lengths.size == 0:
            return 0
        if self.kind is ReductionKind.MAX:
            return int(lengths.max())
        # __post_init__ guarantees percentile is not None for PERCENTILE.
        assert self.percentile is not None
        return int(np.percentile(lengths, self.percentile, method="lower"))

    def _group_representative(self, group: np.ndarray) -> float:
        """Collapse one duplicate-group to its single representative length.

        The duplicate-aware path (``--adjust-for-duplicates``) treats a
        set of variants sharing one data-bin pointer as ONE item; this
        method is the per-kind rule for that item's representative
        length:

        * :attr:`ReductionKind.MAX` -> the group MAX.
        * :attr:`ReductionKind.PERCENTILE` -> the group AVERAGE (may be
          fractional; the final cast to ``int`` happens in
          :meth:`reduce_groups`).

        A singleton group's representative is its lone value under both
        rules, so :meth:`reduce_groups` over singleton groups is
        identical to :meth:`reduce` over the flat vector.
        """
        if self.kind is ReductionKind.MAX:
            return float(group.max())
        return float(group.mean())

    def reduce_groups(self, groups: Sequence[np.ndarray]) -> int:
        """Collapse duplicate-grouped per-variant lengths to one int key.

        Each entry in ``groups`` is the length vector of one
        duplicate-group (variants sharing a data-bin pointer). Every
        group collapses to a single representative via
        :meth:`_group_representative`; the representatives are then
        reduced the same way :meth:`reduce` reduces a flat vector.

        When every group is a singleton (no duplicates), the result is
        identical to :meth:`reduce` over the concatenated lengths -- so
        the duplicate-aware path is a strict generalisation of the plain
        path, not a parallel code branch.

        Empty input (no groups, or every group empty) -> ``0``.
        """
        representatives = np.fromiter(
            (
                self._group_representative(group)
                for group in groups
                if group.size > 0
            ),
            dtype=np.float64,
        )
        if representatives.size == 0:
            return 0
        if self.kind is ReductionKind.MAX:
            return int(representatives.max())
        # __post_init__ guarantees percentile is not None for PERCENTILE.
        assert self.percentile is not None
        return int(
            np.percentile(representatives, self.percentile, method="lower")
        )


@dataclass(frozen=True)
class IndexSpec:
    """One ``(reduction, depth)`` output identity for the sorted index.

    Replaces the ad-hoc ``(LengthReduction, depth)`` tuple used to key
    the multi-(mode, depth) compute results + to drive one ``.idx``
    filename per pair. ``reduction`` is the per-section length
    aggregator; ``depth`` is the splice depth encoded in the filename's
    ``_d<NNN>`` tag. Hashable + frozen so it can key the result dict and
    round-trip through :func:`._builder.write_sorted_index_files`.
    """

    reduction: LengthReduction
    depth: int

    def sort_key(self) -> "tuple[str, int]":
        """Canonical cross-run ordering key: ``(filename_tag, depth)``.

        The SINGLE source of truth for spec ordering -- both the
        collection's display order (``._spec.sorted_specs``) and the
        cross-(binary x spec) sampler's canonical cell order key off this
        so a spec sorts identically everywhere (mirroring the ``.idx``
        filename's lexsort-by-percentile tag).
        """
        return (self.reduction.filename_tag(), self.depth)


@dataclass(frozen=True)
class MultiBinarySectionPointer:
    """Typed cross-binary section pointer (plan D7).

    Replaces ad-hoc ``(binary_name, section_idx)`` tuples used at the
    cross-binary sampler boundary. ``section_pointer`` carries the
    single-binary :class:`SectionPointerSpec` (which already encodes
    ``SectionKind.MATCHED`` + per-arm idx); ``binary_name`` selects
    which per-binary session to open.

    ``spec`` is the OPTIONAL ``(reduction, depth)`` :class:`IndexSpec`
    the pointer was drawn from. It is ``None`` for every single-spec /
    per-binary draw (the historical path, which never tags a spec); the
    cross-(binary x spec) sampler stamps it so the cross-depth load path
    can derive each row's ``max_depth`` from ``spec.depth``. Declared
    LAST with a ``None`` default so every existing construction is
    unchanged.
    """

    binary_name: str
    section_pointer: SectionPointerSpec
    spec: Optional[IndexSpec] = None


@dataclass(frozen=True)
class MultiBinaryBatchDecodeResult:
    """Extension of :class:`BatchDecodeResult` with cross-binary identity.

    The inner :class:`BatchDecodeResult` is the concatenated per-binary
    result (see plan ALG-6); ``binary_id_per_row`` is a per-row index
    into ``binary_names`` so each batch row can be traced back to its
    source binary (plan D7).
    """

    inner: BatchDecodeResult
    binary_id_per_row: np.ndarray
    """``u32[batch_size]`` -- index into :attr:`binary_names`."""

    binary_names: List[str]
    """``binary_id -> binary_name`` reverse map; alphabetical order
    matching :attr:`MultiBinarySortedIndexSampler.binary_names`."""
