"""Splice-graph spliced-length compute for the matched-arm sorted index.

Public surface: :func:`compute_node_lengths` -- per-(section, variant,
depth) spliced token lengths straight from the columnar catalog + the
INJECTED per-node body lengths (the matched-arm realized-length
sidecar), with no token body decoded.

Three single-concern submodules:

* :mod:`._resolve` -- the no-cutoff length budget constant the BFS
  asserts against;
* :mod:`._adjacency` -- per-node splice children read LIVE from the
  catalog memmap (the J fallback chain, EXTERN/unresolved/unknown-offset
  gates), with no precomputed graph structure -- the catalog IS the
  adjacency;
* :mod:`._bfs` -- the per-root level-synchronous BFS that drives the
  SHARED once-only inclusion decider
  (:mod:`...splice_inclusion`) -- the same instance the dataloader
  callee walk uses, so spliced-length semantics never drift from the
  decode path.
"""

from ._bfs import LARGE_CONTEXT_LEN, compute_node_lengths


__all__ = ["LARGE_CONTEXT_LEN", "compute_node_lengths"]
