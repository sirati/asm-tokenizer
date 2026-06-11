"""Cycle flagging + exact-DFS fallback for the graph length compute.

Single concern: the ACTIVE-PATH semantics of the stage-1 callee walk.
The walker (:mod:`...loader.batch_decode._callee_walk._walker`) skips a
callee whose section is already on the current DFS path (cycle key =
section offset, discarded on backtrack). The vectorized depth-DP in
:mod:`._graph_lengths` cannot express that path-dependence, so:

* :func:`flag_cycle_roots` over-approximates the set of root sections
  whose depth-bounded splice tree COULD revisit a section (only those
  roots' DP results may be wrong);
* :func:`exact_lengths_for_root` replays the walker's DFS exactly
  (preorder, per-call-target order, visited-add before descent /
  discard after) for one root node, summing precomputed own-lengths.

A root needs the exact path iff some path of length <= max_depth from
it revisits a section. Such a path must reach a node that lies on a
directed cycle, using at least one step to come back -- so flagging
every root that can reach a cycle-member section within
``max_depth - 1`` steps is a sound over-approximation (cycle members
found via SCC: size > 1 or an explicit self-loop). Roots outside the
flag set provably have revisit-free trees, where the DP equals the
walk.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


__all__ = ["flag_cycle_roots", "exact_lengths_for_root"]


def _scc_ids(n: int, adj_off: np.ndarray, adj_dst: np.ndarray) -> np.ndarray:
    """Iterative Tarjan over a CSR digraph; returns component id per node."""
    UNSET = -1
    index = np.full(n, UNSET, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    on_stack = np.zeros(n, dtype=bool)
    comp = np.full(n, UNSET, dtype=np.int64)
    stack: List[int] = []
    next_index = 0
    next_comp = 0

    for start in range(n):
        if index[start] != UNSET:
            continue
        # Each work-stack frame: [node, next-edge cursor].
        work: List[List[int]] = [[start, int(adj_off[start])]]
        index[start] = low[start] = next_index
        next_index += 1
        stack.append(start)
        on_stack[start] = True

        while work:
            node, cursor = work[-1]
            if cursor < adj_off[node + 1]:
                work[-1][1] += 1
                dst = int(adj_dst[cursor])
                if index[dst] == UNSET:
                    index[dst] = low[dst] = next_index
                    next_index += 1
                    stack.append(dst)
                    on_stack[dst] = True
                    work.append([dst, int(adj_off[dst])])
                elif on_stack[dst]:
                    low[node] = min(low[node], index[dst])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp[w] = next_comp
                        if w == node:
                            break
                    next_comp += 1
    return comp


def flag_cycle_roots(
    n_sections: int,
    edge_src_sec: np.ndarray,
    edge_dst_sec: np.ndarray,
    max_depth: int,
) -> np.ndarray:
    """``bool[n_sections]`` -- True where a root MAY hit the walk's
    active-path cycle skip within ``max_depth``.

    ``edge_src_sec`` / ``edge_dst_sec`` are the section-level
    projections of every resolved splice edge (duplicates fine).
    """
    flagged = np.zeros(n_sections, dtype=bool)
    if max_depth <= 0 or edge_src_sec.size == 0:
        return flagged

    # Unique section-level edges, CSR by source.
    key = edge_src_sec.astype(np.int64) * n_sections + edge_dst_sec
    uniq = np.unique(key)
    src = (uniq // n_sections).astype(np.int64)
    dst = (uniq % n_sections).astype(np.int64)
    counts = np.bincount(src, minlength=n_sections)
    adj_off = np.zeros(n_sections + 1, dtype=np.int64)
    np.cumsum(counts, out=adj_off[1:])

    comp = _scc_ids(n_sections, adj_off, dst)
    comp_sizes = np.bincount(comp)
    on_cycle = comp_sizes[comp] > 1
    on_cycle[src[src == dst]] = True  # explicit self-loops

    if not bool(on_cycle.any()):
        return flagged

    # Reverse BFS from the cycle set, depth-limited to max_depth - 1.
    rev_order = np.argsort(dst, kind="stable")
    rev_src = dst[rev_order]
    rev_dst = src[rev_order]
    rev_counts = np.bincount(rev_src, minlength=n_sections)
    rev_off = np.zeros(n_sections + 1, dtype=np.int64)
    np.cumsum(rev_counts, out=rev_off[1:])

    flagged |= on_cycle
    frontier = np.nonzero(on_cycle)[0]
    for _ in range(max_depth - 1):
        if frontier.size == 0:
            break
        # Gather all predecessors of the frontier in one shot.
        spans = [
            rev_dst[rev_off[s] : rev_off[s + 1]] for s in frontier.tolist()
        ]
        if not spans:
            break
        preds = np.unique(np.concatenate(spans)) if spans else frontier[:0]
        fresh = preds[~flagged[preds]]
        flagged[fresh] = True
        frontier = fresh
    return flagged


def exact_lengths_for_root(
    root_node: int,
    root_sec: int,
    *,
    own: np.ndarray,
    adj_off: np.ndarray,
    adj_child: np.ndarray,
    child_sec: np.ndarray,
    max_depth: int,
    depths: Sequence[int],
) -> Dict[int, int]:
    """Replay the walker's DFS for one root; per-depth spliced lengths.

    ``adj_*`` is the NODE-level CSR adjacency of resolved edges in
    per-call-target order (the walker's emission order). Returns
    ``{depth -> length}`` for every requested depth, where the depth-d
    length sums ``own`` over the root + every callee whose path depth
    is <= d -- exactly :func:`.._length_compute`'s historical
    ``_variant_lengths_at_depth`` contract.
    """
    buckets = [0] * (max_depth + 1)
    buckets[0] = int(own[root_node])
    visited = {int(root_sec)}

    def _descend(node: int, depth: int) -> None:
        if depth >= max_depth:
            return
        for e in range(int(adj_off[node]), int(adj_off[node + 1])):
            sec = int(child_sec[e])
            if sec in visited:
                continue
            child = int(adj_child[e])
            buckets[depth + 1] += int(own[child])
            visited.add(sec)
            try:
                _descend(child, depth + 1)
            finally:
                visited.discard(sec)

    _descend(root_node, 0)

    out: Dict[int, int] = {}
    running = 0
    by_depth = {}
    for d, b in enumerate(buckets):
        running += b
        by_depth[d] = running
    for d in depths:
        out[d] = by_depth[min(d, max_depth)]
    return out
