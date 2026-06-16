"""The per-row inclusion result record.

Single concern: the immutable :class:`RowInclusion` value -- one batch
row's ordered emitted nodes + remembered-excluded backfill pool, with the
parallel per-node edge :class:`CallTargetType` arrays. The BFS traversal
that fills it lives in :mod:`._bfs`; the per-row drive in :mod:`._compute`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["RowInclusion"]


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
