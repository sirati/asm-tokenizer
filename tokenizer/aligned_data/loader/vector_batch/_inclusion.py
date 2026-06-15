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

WHY reuse, not re-implement: the inclusion semantics (once-only-per-root
dedup, the columnwise-ALL "reached by every variant => excluded + pruned"
rule) live in :class:`OnceOnlyInclusion`; the per-node splice children
(the ``choose_callee_variant`` J fallback chain, EXTERN / unresolved-
pointer / unknown-offset gates) live in :class:`LiveNodeAdjacency`. This
module owns ONLY the level-synchronous traversal that records emission
ORDER (the length-only :func:`...compute_node_lengths` discards order) +
the sampled-vs-full pool difference.

WHY two BFS passes per root section: the columnwise-ALL exclusion (FLAG-A
in :mod:`...splice_inclusion._state`) is a property of the mask's variant
ROWS. The post-T0 decode path feeds the decider the SAMPLED subset, so
this prepass does too (the two converge by construction). The FULL-set
pass is run ONLY to learn which callees the subset's narrower mask
pruned that the full mask would have kept -- the remembered-excluded
backfill candidates. Both passes reuse the same shared decider + the same
adjacency; neither touches ``_data.bin``.

The single localized position-assignment choice (the owner's "one-line
strategy swap" mandate for a future ``--match-legacy-order`` toggle) is
the BFS frontier ordering here; it is BFS order and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion


__all__ = ["RowInclusion", "compute_row_inclusions"]


#: The edge type seeded for every row's ROOT node. The decode path's root
#: self-token category is ``LOCAL_FUNC`` (see
#: ``batch_decode._callee_walk._walker``), which is the ``CallTargetType.
#: LOCAL`` edge mapping -- so the root is recorded as a LOCAL edge.
_ROOT_EDGE_TYPE: int = int(CallTargetType.LOCAL)


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


def compute_row_inclusions(
    cols: ColumnarSections,
    section_offsets: np.ndarray,
    *,
    root_sections: np.ndarray,
    root_sampled_variants: np.ndarray,
    max_depth: int,
    need_excluded_pool: bool = True,
) -> List[RowInclusion]:
    """Per-row ordered emitted nodes + remembered-excluded pool.

    Parameters
    ----------
    cols:
        The columnar ``sections.bin`` catalog (:func:`parse_sections_
        columnar` output) -- the body-free splice graph.
    section_offsets:
        ``int[n_sections]`` byte offsets parallel to ``cols`` (the
        :class:`LiveNodeAdjacency` offset->idx key source).
    root_sections:
        ``int[B]`` -- the per-row root SECTION index.
    root_sampled_variants:
        ``int[B]`` -- the per-row sampled VARIANT index WITHIN the root
        section (``0 <= v < n_variants[section]``). One batch row per
        ``(root_sections[r], root_sampled_variants[r])`` pair.
    max_depth:
        Splice-tree BFS depth cap (``>= 0``).
    need_excluded_pool:
        Whether the remembered-excluded backfill pool is needed. The pool
        is the FULL-set-included MINUS subset-emitted diff -- it feeds
        ONLY backfill (:mod:`._backfill`), which runs only when the caller
        passes an ``augment_geometry`` hook. When ``False`` the per-row
        ``excluded_nodes`` / ``excluded_edge_types`` are left empty and the
        per-section FULL-variant-set BFS (:func:`_bfs_full_included`) is
        skipped entirely -- the dominant cost of the prepass when backfill
        is off. Byte-identity-safe: the emitted nodes are unchanged; only
        the (then-unused) pool is suppressed.

    Returns
    -------
    list[RowInclusion]
        One per batch row, in input order. ``emitted_nodes[0]`` is always
        the root node ``var_offsets[section] + sampled_variant``.

    Notes
    -----
    Body-free: only ``cols`` (sections.bin) + ``section_offsets`` are
    read; no ``_data.bin`` byte is touched (the adjacency + decider are
    metadata-only).
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0; got {max_depth}")
    sec = np.asarray(root_sections, dtype=np.int64).reshape(-1)
    smp = np.asarray(root_sampled_variants, dtype=np.int64).reshape(-1)
    if sec.shape != smp.shape:
        raise ValueError(
            "root_sections and root_sampled_variants must be parallel; got "
            f"{sec.shape} vs {smp.shape}"
        )
    n_rows = sec.size
    sec_of_var = np.repeat(
        np.arange(cols.n_variants.size, dtype=np.int64), cols.n_variants
    )
    adjacency = LiveNodeAdjacency(cols, section_offsets, sec_of_var)
    decider = OnceOnlyInclusion()

    # Group batch rows by root SECTION: every row sharing a section drives
    # ONE decider pass whose mask rows are exactly that section's SAMPLED
    # variants (the subset). begin_root needs the full mask-row count, so
    # we size the mask to the section's full variant count but only seed /
    # walk the sampled rows -- non-sampled rows of the section are absent
    # from the subset mask, so they neither exclude (FLAG-A) nor emit.
    out: List[RowInclusion] = [None] * n_rows  # type: ignore[list-item]
    rows_by_section: Dict[int, List[int]] = {}
    for r in range(n_rows):
        rows_by_section.setdefault(int(sec[r]), []).append(r)

    for section_idx, batch_rows in rows_by_section.items():
        sampled = smp[batch_rows]
        # Subset emission + inclusion mask membership.
        emitted_per_row, emitted_types_per_row, included_subset = _bfs_emit(
            section_idx=section_idx,
            sampled_variants=sampled,
            cols=cols,
            adjacency=adjacency,
            decider=decider,
            max_depth=max_depth,
        )
        # Full-set inclusion membership (order discarded) for the pool diff,
        # plus the EDGE ct_type each full-set callee was reached by -- the
        # provenance a re-inlined pool node carries through backfill. Skipped
        # when the pool is not needed (backfill off): this FULL-variant-set
        # BFS is the prepass's dominant cost and feeds ONLY the (then-unused)
        # pool.
        if need_excluded_pool:
            included_full, full_edge_type = _bfs_full_included(
                section_idx=section_idx,
                cols=cols,
                adjacency=adjacency,
                decider=decider,
                max_depth=max_depth,
            )
        for local, r in enumerate(batch_rows):
            emitted = emitted_per_row[local]
            emitted_types = emitted_types_per_row[local]
            if need_excluded_pool:
                # Remembered-excluded = full-set-included MINUS subset-emitted
                # (the callees the narrower subset mask pruned). De-duplicated
                # ascending; excludes anything this row already emitted.
                pool = np.setdiff1d(
                    included_full, emitted, assume_unique=False
                ).astype(np.int64)
                # Carry each pool node's FULL-set edge ct_type verbatim (the
                # ct_type it would have had as an inlined callee). full_edge_type
                # is keyed by node so the gather is a parallel lookup, never a
                # default.
                pool_types = (
                    full_edge_type[pool]
                    if pool.size
                    else np.zeros(0, dtype=np.uint8)
                )
            else:
                pool = np.zeros(0, dtype=np.int64)
                pool_types = np.zeros(0, dtype=np.uint8)
            out[r] = RowInclusion(
                emitted_nodes=emitted,
                emitted_edge_types=emitted_types,
                excluded_nodes=pool,
                excluded_edge_types=pool_types,
            )
    return out  # type: ignore[return-value]


def _bfs_emit(
    *,
    section_idx: int,
    sampled_variants: np.ndarray,
    cols: ColumnarSections,
    adjacency: LiveNodeAdjacency,
    decider: OnceOnlyInclusion,
    max_depth: int,
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
        rows, fids, child_nodes, child_types = _expand_level(
            parent_row, parent_node, adjacency
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
        rows, fids, child_nodes, child_types = _expand_level(
            parent_row, parent_node, adjacency
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

    Returns ``(rows, callee_secs, child_nodes, child_types)`` -- one
    entry per (parent, resolved call_target), parents in frontier order
    and each parent's children in ascending call_target slot (the order
    :meth:`LiveNodeAdjacency.__call__` returns). ``rows`` is the mask
    row; ``callee_secs`` is the once-only key the decider dedups on;
    ``child_nodes`` is the flat callee node; ``child_types`` is the
    parent slot's :class:`CallTargetType` (uint8) per child -- the edge
    attribute the scatter turns into the inlined-callee self-token
    category. Mirrors ``...._graph_lengths._bfs._expand_children`` (the
    length twin) so the BFS frontier order matches the index build's.
    """
    row_chunks: List[np.ndarray] = []
    sec_chunks: List[np.ndarray] = []
    node_chunks: List[np.ndarray] = []
    type_chunks: List[np.ndarray] = []
    for row, node in zip(parent_row.tolist(), parent_node.tolist()):
        children, child_secs, child_types = adjacency(int(node))
        if children.size == 0:
            continue
        row_chunks.append(np.full(children.size, row, dtype=np.int64))
        sec_chunks.append(np.asarray(child_secs, dtype=np.uint32))
        node_chunks.append(np.asarray(children, dtype=np.int64))
        type_chunks.append(np.asarray(child_types, dtype=np.uint8))
    if not row_chunks:
        e_i = np.zeros(0, dtype=np.int64)
        return (
            e_i,
            np.zeros(0, dtype=np.uint32),
            e_i.copy(),
            np.zeros(0, dtype=np.uint8),
        )
    return (
        np.concatenate(row_chunks),
        np.concatenate(sec_chunks),
        np.concatenate(node_chunks),
        np.concatenate(type_chunks),
    )
