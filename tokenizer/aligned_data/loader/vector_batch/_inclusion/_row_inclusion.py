"""The per-row inclusion record + the flat-CSR lazy row view.

Single concern: the per-row inclusion VALUE -- one batch row's ordered
emitted nodes + remembered-excluded backfill pool, with the parallel
per-node edge :class:`CallTargetType` arrays -- in two forms that read the
SAME bytes: the immutable :class:`RowInclusion` record, and the lazy
:class:`RowInclusionView` sequence that carves it on demand out of the
fused kernel's flat :class:`InclusionCSR` WITHOUT materialising one Python
object per row. The BFS traversal that fills the CSR lives in :mod:`._bfs`;
the per-row drive (which returns the CSR) in :mod:`._compute`.

WHY a lazy view (not a ``list[RowInclusion]``): the production consumer
(:mod:`.._geometry`) reads the rows straight back into the flat CSR shape
the kernel already returns, so materialising B per-row objects only to
re-concatenate them is pure waste (and GIL-held). The view lets tests /
one-shot callers index ``view[r]`` ergonomically (each access slices the
CSR -- a parent-state read, never a stored copy) while the hot path
consumes the CSR arrays directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    InclusionCSR,
)


__all__ = ["RowInclusion", "RowInclusionView"]


@dataclass(frozen=True)
class RowInclusion:
    """One batch row's ordered emitted nodes + remembered-excluded pool.

    ``emitted_nodes`` is the BFS emission order (root first, then the
    included callees level by level); ``emitted_edge_types`` is the
    parallel per-emitted-node :class:`CallTargetType` of the edge that
    reached it (root seeded ``CallTargetType.LOCAL``); ``excluded_nodes``
    is the sampled-subset-pruned / full-set-included backfill pool (de-
    duplicated, ascending), with ``excluded_edge_types`` the parallel
    per-pool-node :class:`CallTargetType` of the FULL-set EDGE that
    reached it (the very ct_type this callee would have carried had the
    subset not pruned it -- so a re-inlined pool node keeps its true
    parent-slot edge type, never a default). ``emitted_nodes`` /
    ``excluded_nodes`` are catalog NODE indices (``var_offsets``-major).
    """

    emitted_nodes: np.ndarray  # int64[k] -- BFS emission order, root at [0]
    emitted_edge_types: np.ndarray  # uint8[k] -- edge CallTargetType per node
    excluded_nodes: np.ndarray  # int64[m] -- remembered backfill pool
    excluded_edge_types: np.ndarray  # uint8[m] -- edge CallTargetType per pool node


@dataclass(frozen=True)
class RowInclusionView:
    """A lazy, indexable sequence of per-row inclusions over one CSR.

    Wraps the fused kernel's flat :class:`InclusionCSR` (the canonical
    cross-boundary contract) and yields a :class:`RowInclusion` per row ON
    ACCESS by slicing the parallel value arrays at the row's CSR offsets --
    the slices are numpy VIEWS into the CSR's current arrays (no copy, no
    stored per-row object). ``len(view)`` is the batch row count; ``view[r]``
    carves row ``r``. The hot path consumes ``view.csr`` arrays directly and
    never indexes; only ergonomic / test callers materialise a row.
    """

    csr: InclusionCSR

    def __len__(self) -> int:
        return int(self.csr.emitted_offsets.size - 1)

    def __getitem__(self, r: int) -> RowInclusion:
        csr = self.csr
        e0, e1 = int(csr.emitted_offsets[r]), int(csr.emitted_offsets[r + 1])
        p0, p1 = int(csr.pool_offsets[r]), int(csr.pool_offsets[r + 1])
        return RowInclusion(
            emitted_nodes=csr.emitted_nodes[e0:e1],
            emitted_edge_types=csr.emitted_types[e0:e1],
            excluded_nodes=csr.pool_nodes[p0:p1],
            excluded_edge_types=csr.pool_types[p0:p1],
        )

    def __iter__(self):
        for r in range(len(self)):
            yield self[r]
