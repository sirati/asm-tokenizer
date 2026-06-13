"""Per-root BFS length aggregation over the live catalog adjacency.

Single concern: turn the per-node live splice children
(:mod:`._adjacency`) + per-node own-lengths into per-(section, variant,
depth) spliced lengths, by driving the SHARED once-only inclusion
decider (:mod:`...splice_inclusion`) one level at a time per root.

A "root" is one matched section together with ALL its variants (the
mask rows). The BFS is level-synchronous: level 0 is each variant's own
body; level ``d`` resolves every surviving parent node's children
(:class:`._adjacency.LiveNodeAdjacency`), feeds the
``(variant, callee_section)`` pairs to the shared decider, sums the
INCLUDED children's own-lengths into that variant's depth-``d`` bucket,
and threads the included pairs forward as the next level's parents. The
shared decider enforces once-only-per-root dedup (self / mutual
recursion, diamonds, branch-shared callees) and the columnwise-ALL
"reached by every variant => excluded + pruned" rule -- the SAME
instance the dataloader walk uses, so the two can never drift.

The once-only key is the callee SECTION index (function identity in the
splice graph -- same-FID sibling sections are distinct callees, exactly
as the walker's ``(arm, section_offset)`` cycle key treated them). The
root's own section is seeded at column 0, so a self/mutual-recursive
edge back to the root is already-included and never re-spliced.

Depth-0 lengths are byte-identical to the legacy build: ``own = 1
self-token + contributing body length`` per :func:`._resolve._body_lengths`.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion

from ._adjacency import LiveNodeAdjacency
from ._resolve import LARGE_CONTEXT_LEN, _body_lengths


__all__ = ["compute_node_lengths", "LARGE_CONTEXT_LEN"]


def compute_node_lengths(
    cols: ColumnarSections,
    section_offsets: np.ndarray,
    data_u8: np.ndarray,
    depths: List[int],
) -> Dict[int, np.ndarray]:
    """Depth-``d`` spliced length per (section, variant) node.

    Returns ``{depth -> int64[total_vars]}`` for every requested depth.
    Raises :class:`AssertionError` if any length reaches
    :data:`LARGE_CONTEXT_LEN` (the legacy build's no-cutoff guarantee
    -- plan D-2.2).

    Each requested depth is materialised from ONE max-depth BFS per
    root (a shallower depth is the cumulative prefix of the deeper
    walk). The shared :class:`OnceOnlyInclusion` instance is reused
    across every root -- its hashmap + mask are cleared, never
    re-allocated, per section.
    """
    if not depths or any(d < 0 for d in depths):
        raise ValueError(f"depths must be non-empty and >= 0; got {depths!r}")
    max_depth = max(depths)
    total_vars = int(cols.var_n_calls.size)
    sec_of_var = np.repeat(
        np.arange(cols.n_variants.size, dtype=np.int64), cols.n_variants
    )

    if total_vars == 0:
        return {d: np.zeros(0, dtype=np.int64) for d in depths}

    # Own length = 1 self-token + contributing body length (the
    # variant-token row prefix stays outside the sums -- see
    # :func:`._resolve._body_lengths`). Depth-0 is byte-identical to base.
    own = _body_lengths(cols, data_u8) + 1

    # ``cum[d]`` is the cumulative depth-``d`` length per node (own +
    # included bodies up to level d). cum[0] = own.
    cum = np.tile(own.astype(np.int64), (max_depth + 1, 1))

    if max_depth > 0:
        adjacency = LiveNodeAdjacency(cols, section_offsets, sec_of_var)
        decider = OnceOnlyInclusion()
        for sec in range(int(cols.n_variants.size)):
            v0 = int(cols.var_offsets[sec])
            v1 = int(cols.var_offsets[sec + 1])
            if v1 == v0:
                continue
            _bfs_one_root(
                sec=sec,
                v0=v0,
                n_variants=v1 - v0,
                own=own,
                adjacency=adjacency,
                decider=decider,
                max_depth=max_depth,
                cum=cum,
            )

    results: Dict[int, np.ndarray] = {d: cum[d].copy() for d in depths}
    _assert_under_budget(results, cols, sec_of_var)
    return results


def _bfs_one_root(
    *,
    sec: int,
    v0: int,
    n_variants: int,
    own: np.ndarray,
    adjacency: LiveNodeAdjacency,
    decider: OnceOnlyInclusion,
    max_depth: int,
    cum: np.ndarray,
) -> None:
    """Level-synchronous BFS for one root section; writes ``cum``.

    ``cum[d][v0 + r]`` accumulates root variant ``r``'s cumulative
    depth-``d`` spliced length. The shared ``decider`` is reset for this
    root (``begin_root`` seeds the root's own section at column 0).
    """
    decider.begin_root(n_variants, sec)

    # Each level's parents: parallel arrays of the mask ROW (0..n_variants)
    # and the resolved NODE (flat variant index) to expand. Level 0's
    # parents are the root variants themselves.
    parent_row = np.arange(n_variants, dtype=np.int64)
    parent_node = v0 + parent_row

    for depth in range(1, max_depth + 1):
        if parent_node.size == 0:
            break
        rows, fids, child_nodes = _expand_children(
            parent_row, parent_node, adjacency
        )
        if child_nodes.size == 0:
            break
        result = decider.step_level(rows, fids)
        inc = result.included
        if bool(inc.any()):
            # Add each included child's own-length to its variant's
            # depth-``depth`` cumulative length; every deeper depth
            # inherits it (a depth-d inclusion is present at every d'>=d).
            inc_rows = rows[inc]
            inc_len = own[child_nodes[inc]].astype(np.int64)
            add = np.zeros(n_variants, dtype=np.int64)
            np.add.at(add, inc_rows, inc_len)
            cum[depth:, v0 : v0 + n_variants] += add

        surv = result.survivor_pairs
        parent_row = rows[surv]
        parent_node = child_nodes[surv]


def _expand_children(
    parent_row: np.ndarray,
    parent_node: np.ndarray,
    adjacency: LiveNodeAdjacency,
) -> tuple:
    """Flatten every parent's resolved children into level pair arrays.

    Returns ``(rows, callee_secs, child_nodes)`` parallel arrays: one
    entry per (parent, resolved call_target). ``rows`` is the mask row
    (variant) the child belongs to; ``callee_secs`` is the once-only
    key; ``child_nodes`` is the flat callee variant index whose body
    length is summed when the pair is included.

    Children are derived per node live from the catalog memmap (never a
    graph-wide adjacency); the order within a parent is ascending
    call_target-slot order, parents in their level order.
    """
    row_chunks: List[np.ndarray] = []
    sec_chunks: List[np.ndarray] = []
    node_chunks: List[np.ndarray] = []
    for row, node in zip(parent_row.tolist(), parent_node.tolist()):
        children, child_secs = adjacency(int(node))
        if children.size == 0:
            continue
        row_chunks.append(np.full(children.size, row, dtype=np.int64))
        sec_chunks.append(np.asarray(child_secs, dtype=np.uint32))
        node_chunks.append(np.asarray(children, dtype=np.int64))
    if not row_chunks:
        e_i = np.zeros(0, dtype=np.int64)
        return e_i, np.zeros(0, dtype=np.uint32), e_i.copy()
    return (
        np.concatenate(row_chunks),
        np.concatenate(sec_chunks),
        np.concatenate(node_chunks),
    )


def _assert_under_budget(
    results: Dict[int, np.ndarray],
    cols: ColumnarSections,
    sec_of_var: np.ndarray,
) -> None:
    """Raise if any spliced length reaches the legacy cutoff budget."""
    for d, arr in results.items():
        if arr.size and int(arr.max()) >= LARGE_CONTEXT_LEN:
            node = int(arr.argmax())
            raise AssertionError(
                f"sorted-index length compute: depth-{d} length "
                f"{int(arr.max())} at section_idx={int(sec_of_var[node])} "
                f"variant_idx={int(node - cols.var_offsets[sec_of_var[node]])} "
                f"reaches LARGE_CONTEXT_LEN ({LARGE_CONTEXT_LEN}); the "
                "legacy walk's cutoff would have fired here"
            )
