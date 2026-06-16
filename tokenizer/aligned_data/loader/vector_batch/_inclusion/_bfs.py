"""Level-synchronous splice BFS over the catalog adjacency.

Single concern: the level-by-level traversal that records emission ORDER
(the length-only ``compute_node_lengths`` discards it) -- the subset pass
(:func:`_bfs_emit`) and the FULL-variant-set pass
(:func:`_bfs_full_included`) that learns the remembered-excluded pool,
plus the shared per-level :func:`_expand_level` child flattening.

WHY reuse, not re-implement: the inclusion semantics (once-only-per-root
dedup, the columnwise-ALL "reached by every variant => excluded + pruned"
rule) live in :class:`OnceOnlyInclusion`; the per-node splice children
(the ``choose_callee_variant`` J fallback chain, EXTERN / unresolved-
pointer / unknown-offset gates) live in :class:`LiveNodeAdjacency`. This
module owns ONLY the level-synchronous traversal that records emission
ORDER + the sampled-vs-full pool difference; neither pass touches
``_data.bin``. The per-row drive over both passes lives in
:mod:`._compute`.

The single localized position-assignment choice (the owner's "one-line
strategy swap" mandate for a future ``--match-legacy-order`` toggle) is
the BFS frontier ordering here; it is BFS order and nowhere else.
"""

from __future__ import annotations

from typing import List

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion
from tokenizer.aligned_data.splice_inclusion._unmatched_expand import (
    Edge,
    expand_unmatched_edges,
)


__all__ = [
    "_bfs_emit",
    "_bfs_full_included",
    "_expand_level",
    "_ROOT_EDGE_TYPE",
]


#: The edge type seeded for every row's ROOT node. The decode path's root
#: self-token category is ``LOCAL_FUNC`` (see
#: ``batch_decode._callee_walk._walker``), which is the ``CallTargetType.
#: LOCAL`` edge mapping -- so the root is recorded as a LOCAL edge.
_ROOT_EDGE_TYPE: int = int(CallTargetType.LOCAL)



def _bfs_emit(
    *,
    section_idx: int,
    sampled_variants: np.ndarray,
    cols: ColumnarSections,
    adjacency: LiveNodeAdjacency,
    decider: OnceOnlyInclusion,
    max_depth: int,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
):
    """Subset BFS: per sampled row, the ORDERED emitted node list.

    Returns ``(emitted_per_row, emitted_types_per_row, included_union)``:

    * ``emitted_per_row`` -- ``list[int64[k]]`` parallel to
      ``sampled_variants``; each is ``[root, callee, ...]`` in BFS
      emission order.
    * ``emitted_types_per_row`` -- ``list[uint8[k]]`` parallel to
      ``emitted_per_row``; each entry is the :class:`CallTargetType` of
      the EDGE that reached the emitted node (the root is seeded as
      ``CallTargetType.LOCAL`` -- the decode path's root encounter
      category is ``LOCAL_FUNC``). The scatter maps this edge type to
      the inlined-callee self-token category.
    * ``included_union`` -- ``int64[]`` the de-duplicated union of every
      node any sampled row included (root excluded -- the pool diff is
      against callees only). Unused by the caller for the subset pass but
      kept symmetric with :func:`_bfs_full_included`.

    Mask rows are the SAMPLED variants only (the subset). ``begin_root``
    sizes the mask to ``len(sampled_variants)`` rows; the root column 0
    seeds every sampled row's own section, so self / mutual recursion
    never re-splices.
    """
    n_sampled = int(sampled_variants.size)
    v0 = int(cols.var_offsets[section_idx])
    # The decider's mask rows are 0..n_sampled-1 (the subset), each
    # standing for one sampled variant. ``begin_root`` seeds the root
    # section at column 0 for every row.
    decider.begin_root(max(1, n_sampled), section_idx)

    emitted_per_row: List[List[int]] = [
        [v0 + int(sampled_variants[i])] for i in range(n_sampled)
    ]
    # Per emitted node, the edge CallTargetType that reached it. The root
    # is seeded LOCAL (its decode encounter category is LOCAL_FUNC).
    emitted_types_per_row: List[List[int]] = [
        [_ROOT_EDGE_TYPE] for _ in range(n_sampled)
    ]
    included: List[int] = []

    # Level-0 parents: one per sampled row, expanding ITS OWN sampled
    # variant node. ``mask_row`` is the subset row (0..n_sampled-1);
    # ``node`` is the flat catalog node it expands.
    parent_row = np.arange(n_sampled, dtype=np.int64)
    parent_node = v0 + sampled_variants.astype(np.int64)

    for _depth in range(1, max_depth + 1):
        if parent_node.size == 0:
            break
        rows, fids, child_nodes, child_types, child_matched = _expand_level(
            parent_row, parent_node, adjacency
        )
        if unmatched_inline:
            rows, fids, child_nodes, child_types = _apply_unmatched_inline(
                rows, fids, child_nodes, child_types, child_matched,
                adjacency, unmatched_inline_depth,
            )
        if child_nodes.size == 0:
            break
        result = decider.step_level(rows, fids)
        inc = result.included
        if bool(inc.any()):
            inc_rows = rows[inc]
            inc_nodes = child_nodes[inc]
            inc_types = child_types[inc]
            # Emission order WITHIN this level follows the pair order
            # (parents in frontier order, each parent's children in
            # ascending call_target slot) -- exactly the order
            # ``_expand_level`` concatenated them.
            for mask_row, node, edge_type in zip(
                inc_rows.tolist(), inc_nodes.tolist(), inc_types.tolist()
            ):
                emitted_per_row[mask_row].append(node)
                emitted_types_per_row[mask_row].append(edge_type)
                included.append(node)
        surv = result.survivor_pairs
        parent_row = rows[surv]
        parent_node = child_nodes[surv]

    return (
        [np.asarray(e, dtype=np.int64) for e in emitted_per_row],
        [np.asarray(t, dtype=np.uint8) for t in emitted_types_per_row],
        np.unique(np.asarray(included, dtype=np.int64))
        if included
        else np.zeros(0, dtype=np.int64),
    )


def _bfs_full_included(
    *,
    section_idx: int,
    cols: ColumnarSections,
    adjacency: LiveNodeAdjacency,
    decider: OnceOnlyInclusion,
    max_depth: int,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
):
    """Full variant-set BFS: INCLUDED callee node set + per-node edge type.

    Order is irrelevant for the node SET (the pool is a set difference);
    mask rows = EVERY variant of the section, so the columnwise-ALL
    exclusion uses the full mask the legacy index build used.

    Returns ``(included_nodes, node_edge_type)``:

    * ``included_nodes`` -- the ascending-unique set of callee nodes any
      variant included (root excluded).
    * ``node_edge_type`` -- ``uint8[n_nodes]`` keyed by catalog node index:
      ``node_edge_type[n]`` is the :class:`CallTargetType` of the EDGE the
      full-set BFS reached node ``n`` by (the FIRST inclusion in BFS order
      wins when the same node is reached via several edges -- deterministic
      and the natural "the edge that first put it in the pool" choice).
      Entries for never-included nodes are unset (the caller only ever
      gathers included / pool nodes, never others). This is the verbatim
      parent-slot ct_type, NOT a default -- it is the very edge attribute a
      re-inlined pool node must carry through backfill.
    """
    n_nodes = int(cols.var_offsets[-1])
    node_edge_type = np.zeros(n_nodes, dtype=np.uint8)
    n_variants = int(cols.n_variants[section_idx])
    if n_variants <= 0:
        return np.zeros(0, dtype=np.int64), node_edge_type
    v0 = int(cols.var_offsets[section_idx])
    decider.begin_root(n_variants, section_idx)

    parent_row = np.arange(n_variants, dtype=np.int64)
    parent_node = v0 + parent_row
    included: List[np.ndarray] = []
    seen = np.zeros(n_nodes, dtype=bool)

    for _depth in range(1, max_depth + 1):
        if parent_node.size == 0:
            break
        rows, fids, child_nodes, child_types, child_matched = _expand_level(
            parent_row, parent_node, adjacency
        )
        if unmatched_inline:
            rows, fids, child_nodes, child_types = _apply_unmatched_inline(
                rows, fids, child_nodes, child_types, child_matched,
                adjacency, unmatched_inline_depth,
            )
        if child_nodes.size == 0:
            break
        result = decider.step_level(rows, fids)
        inc = result.included
        if bool(inc.any()):
            inc_nodes = child_nodes[inc]
            inc_types = child_types[inc]
            included.append(inc_nodes)
            # First-inclusion-wins per node: only stamp the edge type for a
            # node not already seen, scanning the level in pair order so the
            # earliest BFS edge is the recorded one.
            for n, t in zip(inc_nodes.tolist(), inc_types.tolist()):
                if not seen[n]:
                    seen[n] = True
                    node_edge_type[n] = t
        surv = result.survivor_pairs
        parent_row = rows[surv]
        parent_node = child_nodes[surv]

    if not included:
        return np.zeros(0, dtype=np.int64), node_edge_type
    return np.unique(np.concatenate(included)).astype(np.int64), node_edge_type


def _expand_level(
    parent_row: np.ndarray,
    parent_node: np.ndarray,
    adjacency: LiveNodeAdjacency,
):
    """Flatten every parent's resolved children into level pair arrays.

    Returns ``(rows, callee_secs, child_nodes, child_types, child_matched)``
    -- one entry per (parent, resolved call_target), parents in frontier
    order and each parent's children in ascending call_target slot (the
    order :meth:`LiveNodeAdjacency.__call__` returns). ``rows`` is the mask
    row; ``callee_secs`` is the once-only key the decider dedups on;
    ``child_nodes`` is the flat callee node; ``child_types`` is the
    parent slot's :class:`CallTargetType` (uint8) per child -- the edge
    attribute the scatter turns into the inlined-callee self-token
    category; ``child_matched`` is the parent slot's ``is_matched`` flag
    (bool) per child, read only by the opt-in unmatched-outline transform.
    Mirrors ``...._graph_lengths._bfs._expand_children`` (the length twin)
    so the BFS frontier order matches the index build's.
    """
    row_chunks: List[np.ndarray] = []
    sec_chunks: List[np.ndarray] = []
    node_chunks: List[np.ndarray] = []
    type_chunks: List[np.ndarray] = []
    matched_chunks: List[np.ndarray] = []
    for row, node in zip(parent_row.tolist(), parent_node.tolist()):
        children, child_secs, child_types, child_matched = adjacency(int(node))
        if children.size == 0:
            continue
        row_chunks.append(np.full(children.size, row, dtype=np.int64))
        sec_chunks.append(np.asarray(child_secs, dtype=np.uint32))
        node_chunks.append(np.asarray(children, dtype=np.int64))
        type_chunks.append(np.asarray(child_types, dtype=np.uint8))
        matched_chunks.append(np.asarray(child_matched, dtype=bool))
    if not row_chunks:
        e_i = np.zeros(0, dtype=np.int64)
        return (
            e_i,
            np.zeros(0, dtype=np.uint32),
            e_i.copy(),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=bool),
        )
    return (
        np.concatenate(row_chunks),
        np.concatenate(sec_chunks),
        np.concatenate(node_chunks),
        np.concatenate(type_chunks),
        np.concatenate(matched_chunks),
    )


def _apply_unmatched_inline(
    rows: np.ndarray,
    fids: np.ndarray,
    child_nodes: np.ndarray,
    child_types: np.ndarray,
    child_matched: np.ndarray,
    adjacency: LiveNodeAdjacency,
    cap: int,
):
    """Surface matched edges behind unmatched edges for one level.

    Drives the SHARED :func:`expand_unmatched_edges` transform over the
    level's flat edge arrays, with a resolver that resolves an unmatched
    node's OWN children via :func:`_expand_level` (the same per-node
    resolution the level used). Returns the surfaced matched level arrays
    ``(rows, fids, child_nodes, child_types)`` -- the same shape the BFS
    feeds the decider, with the ``is_matched`` axis consumed away.

    The payload carried through the transform is the per-edge
    ``(row, fid, node, edge_type)`` tuple (the once-only key + the
    emission attributes), so a surfaced matched edge keeps its own
    parent-slot node + edge type verbatim.
    """

    def _to_edge(row, fid, node, edge_type, matched) -> Edge:
        return Edge(
            mask_row=int(row),
            dedup_key=int(fid),
            is_matched=bool(matched),
            payload=(int(row), int(fid), int(node), int(edge_type)),
        )

    def _resolve_children(edge: Edge) -> List[Edge]:
        row, _fid, node, _etype = edge.payload
        # one-parent expansion: resolve THIS unmatched node's children.
        c_rows, c_fids, c_nodes, c_types, c_matched = _expand_level(
            np.asarray([row], dtype=np.int64),
            np.asarray([node], dtype=np.int64),
            adjacency,
        )
        return [
            _to_edge(c_rows[i], c_fids[i], c_nodes[i], c_types[i], c_matched[i])
            for i in range(c_rows.size)
        ]

    edges = [
        _to_edge(
            rows[i], fids[i], child_nodes[i], child_types[i], child_matched[i]
        )
        for i in range(rows.size)
    ]
    surfaced = expand_unmatched_edges(edges, _resolve_children, max_unmatched_depth=cap)
    if not surfaced:
        e_i = np.zeros(0, dtype=np.int64)
        return e_i, np.zeros(0, dtype=np.uint32), e_i.copy(), np.zeros(0, dtype=np.uint8)
    s_rows = np.asarray([e.payload[0] for e in surfaced], dtype=np.int64)
    s_fids = np.asarray([e.payload[1] for e in surfaced], dtype=np.uint32)
    s_nodes = np.asarray([e.payload[2] for e in surfaced], dtype=np.int64)
    s_types = np.asarray([e.payload[3] for e in surfaced], dtype=np.uint8)
    return s_rows, s_fids, s_nodes, s_types
