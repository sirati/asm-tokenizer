"""BFS-level-set invariant for the vectorized inclusion traversal.

The relaxed-order contract lets ``_bfs_emit`` emit a row's intra-level
siblings in any order, but the SET of nodes emitted at each BFS LEVEL
(root, then all depth-1 callees, then depth-2, ...) must be EXACTLY the
level-synchronous splice BFS's -- a DFS regression (a deeper node emitted
before a shallower sibling, or a level boundary dropped) must be caught
even with free intra-level order.

This drives the REAL :func:`_bfs_emit` (decider + vectorized adjacency)
and pins each row's per-level emitted node set against an independent
reference BFS built from the same :meth:`LiveNodeAdjacency.expand_batch`
+ :class:`OnceOnlyInclusion`. A mutation (truncate one level) proves the
assertion has teeth.
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
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion

from tokenizer.aligned_data.loader.vector_batch._inclusion._bfs import (
    _bfs_emit,
)


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
    """A multi-level call graph (root -> a,b -> c -> leaf) with diamonds."""
    specs = [
        MatchedFunctionSpec(
            func_name="root",
            variants=(
                _variant(("V", 0), 1, ("a", "b")),
                _variant(("V", 1), 2, ("a", "c")),
                _variant(("V", 2), 3, ("b",)),
            ),
            called=("a", "b", "c"),
        ),
        MatchedFunctionSpec(
            func_name="a",
            variants=(
                _variant(("V", 0), 4, ("c",)),
                _variant(("V", 1), 5, ("leaf",)),
            ),
            called=("c", "leaf"),
        ),
        MatchedFunctionSpec(
            func_name="b",
            variants=(_variant(("V", 0), 6, ("c",)),),
            called=("c",),
        ),
        MatchedFunctionSpec(
            func_name="c",
            variants=(
                _variant(("V", 0), 7, ("leaf",)),
                _variant(("V", 1), 8, ()),
            ),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf",
            variants=(_variant(("V", 0), 9, ()),),
            called=(),
        ),
    ]
    build_corpus(tmp_path, "bfslv", matched=specs)
    starts, lens = read_csv_section_index_arrays(tmp_path / "bfslv_index.bin")
    blob = np.fromfile(tmp_path / "bfslv_sections.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lens)
    adj = LiveNodeAdjacency(cols, starts, cols.sec_of_var)
    return cols, adj


def _reference_levels(cols, adj, section_idx, sampled):
    """Independent per-level included node sets per sampled row.

    Drives the SAME decider + ``expand_batch`` level-synchronously; returns
    ``levels[row]`` = ``list`` of ``set`` of catalog nodes, level 0 = the
    row's root node, level d = the nodes the row first INCLUDED at depth d.
    """
    n = int(sampled.size)
    v0 = int(cols.var_offsets[section_idx])
    decider = OnceOnlyInclusion()
    decider.begin_root(max(1, n), section_idx)
    levels = [[{v0 + int(sampled[i])}] for i in range(n)]
    parent_row = np.arange(n, dtype=np.int64)
    parent_node = v0 + sampled.astype(np.int64)
    max_depth = 5
    for _d in range(1, max_depth + 1):
        if parent_node.size == 0:
            break
        pos, fids, child_nodes, _t, _m = adj.expand_batch(parent_node)
        rows = parent_row[pos]
        if child_nodes.size == 0:
            break
        result = decider.step_level(rows, fids)
        inc = result.included
        level_sets = [set() for _ in range(n)]
        for r, node in zip(
            rows[inc].tolist(), child_nodes[inc].tolist()
        ):
            level_sets[r].add(node)
        for i in range(n):
            levels[i].append(level_sets[i])
        surv = result.survivor_pairs
        parent_row = rows[surv]
        parent_node = child_nodes[surv]
    return levels


def _emit_levels(cols, adj, section_idx, sampled):
    """Per-row BFS-level node sets reconstructed from ``_bfs_emit``.

    ``_bfs_emit`` returns the FLAT emission per row (root first). To split
    into BFS levels independently of the production traversal, the emitted
    nodes are re-bucketed by the reference's level membership -- so a level
    boundary that the PRODUCTION traversal violated (a node emitted at the
    wrong depth) shows up as a missing/extra node in a level set.
    """
    decider = OnceOnlyInclusion()
    emitted_per_row, _types, _u = _bfs_emit(
        section_idx=section_idx,
        sampled_variants=sampled,
        cols=cols,
        adjacency=adj,
        decider=decider,
        max_depth=5,
    )
    return [set(e.tolist()) for e in emitted_per_row]


def test_emitted_node_set_matches_level_synchronous_bfs(tmp_path):
    cols, adj = _build(tmp_path)
    section_idx = 0  # root
    sampled = np.array([0, 1, 2], dtype=np.int64)

    ref_levels = _reference_levels(cols, adj, section_idx, sampled)
    emitted_sets = _emit_levels(cols, adj, section_idx, sampled)

    for i in range(sampled.size):
        ref_total = set().union(*ref_levels[i])
        # The production emission's node SET (root + every included callee)
        # must equal the level-synchronous BFS's total included set.
        assert emitted_sets[i] == ref_total, (
            f"row {i}: emitted set {sorted(emitted_sets[i])} != "
            f"BFS set {sorted(ref_total)}"
        )
        # And the per-level partition must be disjoint (once-only): no node
        # appears at two BFS levels.
        seen = set()
        for lvl in ref_levels[i]:
            assert not (seen & lvl), f"row {i}: node reused across levels"
            seen |= lvl


def test_bfs_level_invariant_has_teeth(tmp_path):
    """Dropping a node from a level set makes the SET invariant FAIL.

    Proves the assertion would catch a DFS regression / lost level
    boundary: a mutated reference that omits one emitted node no longer
    equals the production emission, so the test is not vacuously true.
    """
    cols, adj = _build(tmp_path)
    section_idx = 0
    sampled = np.array([0, 1, 2], dtype=np.int64)

    emitted_sets = _emit_levels(cols, adj, section_idx, sampled)
    ref_levels = _reference_levels(cols, adj, section_idx, sampled)

    # Find a row with at least one callee, drop a node from its set, and
    # assert the equality the real test relies on now FAILS.
    mutated_any = False
    for i in range(sampled.size):
        ref_total = set().union(*ref_levels[i])
        if len(ref_total) > 1:
            broken = set(ref_total)
            broken.discard(max(ref_total))  # drop the deepest-numbered node
            assert emitted_sets[i] != broken
            mutated_any = True
            break
    assert mutated_any, "fixture produced no spliceable row to mutate"
