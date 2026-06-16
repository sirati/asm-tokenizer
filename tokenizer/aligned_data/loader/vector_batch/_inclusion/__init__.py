"""Body-free BFS emission-ORDER + remembered-excluded pool per batch row.

Single concern: drive the SHARED once-only inclusion decider
(:mod:`...splice_inclusion.OnceOnlyInclusion`) over the catalog
adjacency (:class:`...sorted_index._graph_lengths.LiveNodeAdjacency`) to
produce, for each sampled ``(root section, sampled variant)`` batch row:

1. the ORDERED list of emitted callee NODES in BFS emission order (root,
   then per level the included callees in parent-then-ascending-
   call_target-slot order), and
2. the REMEMBERED extra-excluded pool -- callee nodes the SAMPLED-subset
   BFS pruned that the FULL variant-set BFS would have included.

WHY two BFS passes per root section: the columnwise-ALL exclusion (FLAG-A
in :mod:`...splice_inclusion._state`) is a property of the mask's variant
ROWS. The post-T0 decode path feeds the decider the SAMPLED subset, so
this prepass does too (the two converge by construction). The FULL-set
pass is run ONLY to learn which callees the subset's narrower mask
pruned that the full mask would have kept -- the remembered-excluded
backfill candidates. Both passes reuse the same shared decider + the same
adjacency; neither touches ``_data.bin``.

Package layout:

* :mod:`._row_inclusion` -- the immutable :class:`RowInclusion` result.
* :mod:`._bfs` -- the level-synchronous subset / full-set BFS passes.
* :mod:`._compute` -- the per-row drive (:func:`compute_row_inclusions`).
"""

from ._compute import compute_row_inclusions
from ._row_inclusion import RowInclusion


__all__ = ["RowInclusion", "compute_row_inclusions"]
