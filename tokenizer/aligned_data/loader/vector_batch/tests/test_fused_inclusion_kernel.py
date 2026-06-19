"""Fused inclusion-BFS kernel: identity vs the per-level reference + teeth.

The Stage-3 fused kernel (:func:`compute_row_inclusions` with
``unmatched_inline=False``) runs the WHOLE per-group + per-depth splice BFS
under one GIL release. This module gates it directly against a faithful
per-level Python reference drive over the SAME production
:meth:`LiveNodeAdjacency.expand_batch` + :class:`OnceOnlyInclusion` (the
two engines the kernel reuses in-Rust), asserting per-row byte-identity of
the emitted nodes / edge types AND the remembered-excluded pool.

Teeth (the order-preserving contract is load-bearing): a reference variant
that REVERSES each level's intra-level sibling order produces a DIFFERENT
emission than the fused kernel -- proving the kernel's parent-ascending,
slot-ascending flattening is not vacuously reproducible by any order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.loader.vector_batch._inclusion import (
    RowInclusionView,
    compute_row_inclusions,
)
from tokenizer.aligned_data.loader.vector_batch._inclusion._bfs import (
    _ROOT_EDGE_TYPE,
    _bfs_emit,
    _bfs_full_included,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion


def _variant(vkey, seed, called):
    base = make_simple_variant(vkey, token_seed=seed, n_tokens=6)
    return VariantSpec(
        vkey=base.vkey,
        tokens=base.tokens,
        block_rl=base.block_rl,
        insn_rl=base.insn_rl,
        called=tuple(called),
    )


def _build(tmp_path: Path):
    """A multi-level call graph with diamonds + per-variant fan-out."""
    specs = [
        MatchedFunctionSpec(
            func_name="root",
            variants=(
                _variant(("V", 0), 1, ("a", "b", "c")),
                _variant(("V", 1), 2, ("a", "c")),
                _variant(("V", 2), 3, ("b", "c")),
            ),
            called=("a", "b", "c"),
        ),
        MatchedFunctionSpec(
            func_name="a",
            variants=(
                _variant(("V", 0), 4, ("d", "leaf")),
                _variant(("V", 1), 5, ("leaf",)),
            ),
            called=("d", "leaf"),
        ),
        MatchedFunctionSpec(
            func_name="b",
            variants=(_variant(("V", 0), 6, ("d",)),),
            called=("d",),
        ),
        MatchedFunctionSpec(
            func_name="c",
            variants=(
                _variant(("V", 0), 7, ("d", "leaf")),
                _variant(("V", 1), 8, ("leaf",)),
            ),
            called=("d", "leaf"),
        ),
        MatchedFunctionSpec(
            func_name="d",
            variants=(_variant(("V", 0), 9, ("leaf",)),),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf",
            variants=(_variant(("V", 0), 10, ()),),
            called=(),
        ),
    ]
    build_corpus(tmp_path, "fused", matched=specs)
    starts, lens = read_csv_section_index_arrays(tmp_path / "fused_index.bin")
    blob = np.fromfile(tmp_path / "fused_sections.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lens)
    adj = LiveNodeAdjacency(cols, starts, cols.sec_of_var)
    return cols, starts, adj


def _reference_emit(cols, adj, section_idx, sampled, max_depth, *, reverse=False):
    """Faithful per-level Python subset emission (root + included callees).

    Drives the SAME production ``expand_batch`` + ``OnceOnlyInclusion`` the
    fused kernel reuses. ``reverse=True`` flips each level's intra-level
    sibling order (per row) AFTER inclusion -- the adversarial perturbation
    the order-preserving contract forbids.
    """
    n = int(sampled.size)
    v0 = int(cols.var_offsets[section_idx])
    decider = OnceOnlyInclusion()
    decider.begin_root(max(1, n), section_idx)
    root_nodes = v0 + sampled.astype(np.int64)
    emitted = [[int(root_nodes[i])] for i in range(n)]
    etypes = [[int(_ROOT_EDGE_TYPE)] for i in range(n)]
    parent_row = np.arange(n, dtype=np.int64)
    parent_node = root_nodes
    for _depth in range(1, max_depth + 1):
        if parent_node.size == 0:
            break
        pos, fids, child_nodes, child_types, _m = adj.expand_batch(parent_node)
        rows = parent_row[pos]
        if child_nodes.size == 0:
            break
        result = decider.step_level(rows, fids)
        inc = result.included
        for i in range(n):
            sel = inc & (rows == i)
            nodes_i = child_nodes[sel].tolist()
            types_i = child_types[sel].tolist()
            if reverse:
                nodes_i = nodes_i[::-1]
                types_i = types_i[::-1]
            emitted[i].extend(nodes_i)
            etypes[i].extend(types_i)
        surv = result.survivor_pairs
        parent_row = rows[surv]
        parent_node = child_nodes[surv]
    return (
        [np.asarray(e, dtype=np.int64) for e in emitted],
        [np.asarray(t, dtype=np.uint8) for t in etypes],
    )


def _reference_pool(cols, adj, section_idx, sampled, max_depth):
    """Legacy Python remembered-excluded pool per row (the oracle).

    The pool is the FULL-variant-set-included callee set MINUS this row's
    subset-emitted nodes -- exactly the diff ``_compute.py`` itself computes
    for the (non-fused) Python drive. Drives the SAME production
    ``_bfs_emit`` + ``_bfs_full_included`` + ``OnceOnlyInclusion`` the fused
    kernel ports, with ``unmatched_inline=False`` (the production default the
    fused kernel runs), so the two pools are comparable value-for-value.

    Returns ``(pool_nodes_per_row, pool_types_per_row)`` parallel to
    ``sampled`` -- each pool ascending-unique with the FULL-set edge
    :class:`CallTargetType` gathered per pool node (the very ct_type a re-
    inlined pool node carries through backfill).
    """
    decider = OnceOnlyInclusion()
    emitted_per_row, _etypes, _u = _bfs_emit(
        section_idx=section_idx,
        sampled_variants=sampled,
        cols=cols,
        adjacency=adj,
        decider=decider,
        max_depth=max_depth,
    )
    included_full, full_edge_type = _bfs_full_included(
        section_idx=section_idx,
        cols=cols,
        adjacency=adj,
        decider=decider,
        max_depth=max_depth,
    )
    pool_nodes, pool_types = [], []
    for emitted in emitted_per_row:
        pool = np.setdiff1d(
            included_full, emitted, assume_unique=False
        ).astype(np.int64)
        ptypes = (
            full_edge_type[pool] if pool.size else np.zeros(0, dtype=np.uint8)
        )
        pool_nodes.append(pool)
        pool_types.append(ptypes)
    return pool_nodes, pool_types


def test_fused_excluded_pool_matches_legacy_python(tmp_path):
    """Byte-identity gate: fused kernel excluded-pool VALUES vs the oracle.

    The emitted-node gate above pins the emission; this pins the OTHER fused
    output -- the remembered-excluded backfill pool (nodes AND per-pool-node
    edge types) -- against the legacy Python full-set-minus-subset oracle,
    over the cross-depth diamond fixture. Non-vacuous: at least one row's
    pool is asserted non-empty so the gate has teeth (a kernel that emitted
    an empty pool would still pass a row-by-row equality if every oracle pool
    were empty too).
    """
    cols, starts, adj = _build(tmp_path)
    section_idx = 0
    sampled = np.array([0, 1, 2], dtype=np.int64)
    max_depth = 5

    incs = RowInclusionView(compute_row_inclusions(
        cols,
        starts,
        root_sections=np.full(3, section_idx, dtype=np.int64),
        root_sampled_variants=sampled,
        root_groups=np.zeros(3, dtype=np.int64),
        max_depth=max_depth,
        need_excluded_pool=True,
        adjacency=adj,
    ))
    ref_pool, ref_pool_types = _reference_pool(
        cols, adj, section_idx, sampled, max_depth
    )
    assert any(p.size for p in ref_pool), (
        "every oracle pool is empty on this fixture -- the pool gate is "
        "vacuous; pick a depth/fixture where the subset prunes a callee"
    )
    for i in range(sampled.size):
        np.testing.assert_array_equal(incs[i].excluded_nodes, ref_pool[i])
        np.testing.assert_array_equal(
            incs[i].excluded_edge_types, ref_pool_types[i]
        )


def test_fused_matches_per_level_reference(tmp_path):
    cols, starts, adj = _build(tmp_path)
    section_idx = 0
    sampled = np.array([0, 1, 2], dtype=np.int64)
    max_depth = 5

    incs = RowInclusionView(compute_row_inclusions(
        cols,
        starts,
        root_sections=np.full(3, section_idx, dtype=np.int64),
        root_sampled_variants=sampled,
        root_groups=np.zeros(3, dtype=np.int64),
        max_depth=max_depth,
        need_excluded_pool=True,
        adjacency=adj,
    ))
    ref_nodes, ref_types = _reference_emit(
        cols, adj, section_idx, sampled, max_depth
    )
    for i in range(sampled.size):
        np.testing.assert_array_equal(incs[i].emitted_nodes, ref_nodes[i])
        np.testing.assert_array_equal(incs[i].emitted_edge_types, ref_types[i])
        # The pool is the full-set-included MINUS this row's emitted, so it
        # never overlaps the emitted nodes and is ascending-unique.
        assert not np.intersect1d(
            incs[i].excluded_nodes, incs[i].emitted_nodes
        ).size
        assert np.array_equal(
            incs[i].excluded_nodes, np.unique(incs[i].excluded_nodes)
        )


def test_reversed_intra_level_order_diverges(tmp_path):
    """Teeth: the kernel's intra-level order is load-bearing.

    A reference that REVERSES each level's sibling order yields a different
    emitted sequence than the fused kernel on at least one row -- so the
    order-preserving flatten the kernel ports is not reproducible by an
    arbitrary intra-level order (the byte-identity gate has teeth).
    """
    cols, starts, adj = _build(tmp_path)
    section_idx = 0
    sampled = np.array([0, 1, 2], dtype=np.int64)
    max_depth = 5

    incs = RowInclusionView(compute_row_inclusions(
        cols,
        starts,
        root_sections=np.full(3, section_idx, dtype=np.int64),
        root_sampled_variants=sampled,
        root_groups=np.zeros(3, dtype=np.int64),
        max_depth=max_depth,
        need_excluded_pool=False,
        adjacency=adj,
    ))
    rev_nodes, _rev_types = _reference_emit(
        cols, adj, section_idx, sampled, max_depth, reverse=True
    )
    diverged = any(
        not np.array_equal(incs[i].emitted_nodes, rev_nodes[i])
        for i in range(sampled.size)
    )
    assert diverged, (
        "reversing intra-level sibling order did not change any row -- the "
        "order-preserving contract is untested on this fixture"
    )
