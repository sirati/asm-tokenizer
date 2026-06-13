"""Shared once-only + all-variants-equivalence splice inclusion.

The single owner of the splice-tree inclusion decision: given a root's
variants and, per level, the resolved ``(variant, callee_function_id)``
pairs, it answers which pairs INCLUDE the callee body at this level and
which SURVIVE to the next -- enforcing once-only-per-root dedup and the
columnwise-ALL "reached by every variant => excluded + pruned" rule.

Both consumers drive the SAME implementation so their inclusion
semantics can never drift:

* the dataloader callee walk
  (:mod:`...loader.batch_decode._callee_walk`) -- emits tokens for
  included call targets, level by level;
* the sorted-index graph-lengths build
  (:mod:`...sorted_index._graph_lengths`) -- sums included body lengths
  per variant per level.

See :mod:`._state` for the algorithm + the buffer-reuse discipline.
"""

from ._state import LevelResult, OnceOnlyInclusion


__all__ = ["LevelResult", "OnceOnlyInclusion"]
