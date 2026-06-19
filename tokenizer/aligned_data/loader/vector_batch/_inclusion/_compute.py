"""Per-row drive over the subset + full-set splice BFS passes.

Single concern: group the batch rows by DECIDER-ROOT (NOT catalog
section), run the subset emission pass (:func:`._bfs._bfs_emit`) and --
when the backfill pool is needed -- the FULL-variant-set pass
(:func:`._bfs._bfs_full_included`) per group, and assemble each row's
:class:`._row_inclusion.RowInclusion` (emitted nodes + the full-set-minus-
subset remembered-excluded pool). The level-synchronous traversal itself
lives in :mod:`._bfs`; the result record in :mod:`._row_inclusion`.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion

from ._bfs import _ROOT_EDGE_TYPE, _bfs_emit, _bfs_full_included
from ._row_inclusion import RowInclusion


__all__ = ["RowInclusion", "compute_row_inclusions"]


def compute_row_inclusions(
    cols: ColumnarSections,
    section_offsets: np.ndarray,
    *,
    root_sections: np.ndarray,
    root_sampled_variants: np.ndarray,
    root_groups: np.ndarray,
    max_depth: int,
    need_excluded_pool: bool = True,
    adjacency: LiveNodeAdjacency | None = None,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
) -> List[RowInclusion]:
    """Per-row ordered emitted nodes + remembered-excluded pool.

    Parameters
    ----------
    cols:
        The columnar ``sections.bin`` catalog (:func:`parse_sections_
        columnar` output) -- the body-free splice graph.
    section_offsets:
        ``int[n_sections]`` byte offsets parallel to ``cols`` (the
        :class:`LiveNodeAdjacency` offset->idx key source). Used ONLY to
        construct the adjacency when ``adjacency`` is not injected.
    adjacency:
        A pre-built :class:`LiveNodeAdjacency` over the SAME ``cols`` /
        ``section_offsets``. The adjacency (its offset->idx hashmap + the
        per-binary MISSING inventory scan) is cols-invariant, so callers
        with a stable per-binary catalog (the dataloader's handles) pass
        the once-built instance instead of forcing a fresh build -- and
        the full-array inventory scan -- on every batch. When ``None`` it
        is built here from ``cols`` / ``section_offsets`` (the only extra
        argument the build needs), so test / one-shot callers need not
        thread it.
    root_sections:
        ``int[B]`` -- the per-row root SECTION index.
    root_sampled_variants:
        ``int[B]`` -- the per-row sampled VARIANT index WITHIN the root
        section (``0 <= v < n_variants[section]``). One batch row per
        ``(root_sections[r], root_sampled_variants[r])`` pair.
    root_groups:
        ``int[B]`` -- the per-row DECIDER-ROOT group id. Rows sharing a
        group id are the co-sampled variants of ONE root (one
        ``begin_root`` call whose mask spans exactly those rows -- the
        columnwise-ALL exclusion is a property of THIS root's rows only).
        This is the originating ``batch_decode`` ``walk_section_callees_
        pending`` unit (one resolved section pointer), NOT the catalog
        section: distinct batch rows that collide on the same physical
        section but came from different roots carry DIFFERENT group ids,
        so each is its own decider root (``n_variants`` = its own group's
        row count) and a single-variant root splices nothing (FLAG-A).
        Every row in a group MUST share the same ``root_sections`` value
        (a group is one section pointer's sampled variants).
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
    grp = np.asarray(root_groups, dtype=np.int64).reshape(-1)
    if not (sec.shape == smp.shape == grp.shape):
        raise ValueError(
            "root_sections, root_sampled_variants and root_groups must be "
            f"parallel; got {sec.shape} vs {smp.shape} vs {grp.shape}"
        )
    n_rows = sec.size
    if adjacency is None:
        adjacency = LiveNodeAdjacency(cols, section_offsets, cols.sec_of_var)

    # The whole per-group + per-depth BFS runs in ONE GIL-released Rust
    # kernel (the fused Stage-3 port), reusing this adjacency's Stage-1/2
    # cores in-Rust. The opt-in unmatched-outline inlining is a recursive,
    # Python-callback-driven graph transform (:mod:`...splice_inclusion.
    # _unmatched_expand`) that cannot run under the single GIL release, so
    # that path keeps the per-group Python drive below; the production
    # default (``unmatched_inline=False``) takes the fused kernel.
    if not unmatched_inline:
        return _fused_row_inclusions(
            adjacency,
            sec=sec,
            smp=smp,
            grp=grp,
            max_depth=max_depth,
            need_excluded_pool=need_excluded_pool,
        )

    decider = OnceOnlyInclusion()

    # Group batch rows by DECIDER-ROOT group, NOT by catalog section. Every
    # row sharing a group is one root's co-sampled variants and drives ONE
    # decider pass whose mask rows are exactly those rows (the subset);
    # begin_root sizes the mask to the GROUP's row count, so the
    # columnwise-ALL exclusion (FLAG-A) sees only this root's rows -- a
    # single-variant root splices nothing. Two batch rows that collide on
    # the same physical section but came from different roots carry
    # DIFFERENT group ids, so each is its own root and never conflates the
    # other's variants into its mask (the #67 fix).
    out: List[RowInclusion] = [None] * n_rows  # type: ignore[list-item]
    rows_by_group: Dict[int, List[int]] = {}
    for r in range(n_rows):
        rows_by_group.setdefault(int(grp[r]), []).append(r)

    for batch_rows in rows_by_group.values():
        group_secs = sec[batch_rows]
        section_idx = int(group_secs[0])
        if bool((group_secs != section_idx).any()):
            raise ValueError(
                "all rows of a root_groups group must share root_sections; "
                f"group spans sections {np.unique(group_secs).tolist()}"
            )
        sampled = smp[batch_rows]
        # Subset emission + inclusion mask membership.
        emitted_per_row, emitted_types_per_row, included_subset = _bfs_emit(
            section_idx=section_idx,
            sampled_variants=sampled,
            cols=cols,
            adjacency=adjacency,
            decider=decider,
            max_depth=max_depth,
            unmatched_inline=unmatched_inline,
            unmatched_inline_depth=unmatched_inline_depth,
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
                unmatched_inline=unmatched_inline,
                unmatched_inline_depth=unmatched_inline_depth,
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


def _fused_row_inclusions(
    adjacency: LiveNodeAdjacency,
    *,
    sec: np.ndarray,
    smp: np.ndarray,
    grp: np.ndarray,
    max_depth: int,
    need_excluded_pool: bool,
) -> List[RowInclusion]:
    """Slice the fused GIL-released kernel's CSR into per-row records.

    Delegates the WHOLE per-group + per-depth BFS to the adjacency's fused
    kernel (:meth:`LiveNodeAdjacency.compute_row_inclusions_csr`), which runs
    it under one GIL release reusing the shared Stage-1/2 cores, then carves
    the returned :class:`InclusionCSR` into one :class:`RowInclusion` per
    batch row. The CSR slices ARE the row's emitted-node / pool views (root
    at emitted slot 0; the pool is ascending-unique with parallel edge
    types), so the assembly is a pure boundary translation -- no inclusion
    logic lives here.
    """
    csr = adjacency.compute_row_inclusions_csr(
        root_sections=sec,
        root_sampled_variants=smp,
        root_groups=grp,
        max_depth=max_depth,
        need_excluded_pool=need_excluded_pool,
        root_edge_type=_ROOT_EDGE_TYPE,
    )
    e_off = csr.emitted_offsets
    p_off = csr.pool_offsets
    out: List[RowInclusion] = []
    for r in range(sec.size):
        e0, e1 = int(e_off[r]), int(e_off[r + 1])
        p0, p1 = int(p_off[r]), int(p_off[r + 1])
        out.append(
            RowInclusion(
                emitted_nodes=csr.emitted_nodes[e0:e1],
                emitted_edge_types=csr.emitted_types[e0:e1],
                excluded_nodes=csr.pool_nodes[p0:p1],
                excluded_edge_types=csr.pool_types[p0:p1],
            )
        )
    return out
