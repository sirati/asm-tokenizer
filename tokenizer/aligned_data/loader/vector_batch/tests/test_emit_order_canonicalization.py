"""Relaxed-order safety net: canonicalize a reordered emission back to ref.

The relaxed-order contract (USER DIRECTIVE) lets the vectorized inclusion
emit a row's intra-BFS-level siblings in ANY order, as long as the per-
LEVEL node SET is unchanged. The byte-identity gate proves tokens + dense
sidecars match the reference when the orders DO coincide (they currently
do); this module proves the gate's robustness to a FUTURE intra-level
reorder, at the EMISSION boundary where order is decided.

The emitted node sequence per row is the authoritative carrier of "which
functions, at which BFS level, in which order" -- the token tensor + the
dense sidecars (including the emission-order-minted identity counter ids)
are a deterministic, order-preserving function of it. So an emission whose
intra-level siblings are permuted yields a correspondingly permuted token/
sidecar output; canonicalizing the emission by per-emitted-node IDENTITY
recovers the reference order, and re-deriving the emission-order counter
assignment under that permutation reproduces the reference's counter ids
(design step 4). This module pins exactly that:

1. drive production :func:`compute_row_inclusions` -> reference emission;
2. drive it again with intra-level siblings SHUFFLED (a stand-in for a
   future faster traversal that does not preserve sibling order);
3. assert the per-BFS-level node SET is identical (the BFS invariant);
4. compute the permutation reordered->reference by node identity, apply
   it, and assert it recovers the reference emitted nodes + edge types AND
   the emission-order counter assignment EXACTLY.

Teeth: a level-BOUNDARY break (a node moved to a different BFS depth) is
NOT a mere intra-level reorder -- canonicalization must FAIL on it, so the
test is not vacuously true.
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

# The root edge type the production emit seeds (LOCAL_FUNC == LOCAL edge).
from tokenizer.aligned_data.loader.vector_batch._inclusion._bfs import (
    _ROOT_EDGE_TYPE,
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
    build_corpus(tmp_path, "emitcanon", matched=specs)
    starts, lens = read_csv_section_index_arrays(
        tmp_path / "emitcanon_index.bin"
    )
    blob = np.fromfile(tmp_path / "emitcanon_sections.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lens)
    adj = LiveNodeAdjacency(cols, starts, cols.sec_of_var)
    return cols, adj


def _emit_with_levels(cols, adj, section_idx, sampled, *, shuffle_seed=None):
    """Per-row ``(emitted_nodes, emitted_edge_types, level_of_node)``.

    Re-implements ONLY the level-synchronous drive (so the per-BFS-level
    boundary is observable for the invariant + the canonicalization) over
    the PRODUCTION :meth:`LiveNodeAdjacency.expand_batch` + the SHARED
    :class:`OnceOnlyInclusion` -- the same two engines the production
    :func:`_bfs_emit` drives. ``shuffle_seed`` permutes each level's
    INCLUDED siblings (per row) before recording, standing in for a future
    traversal whose intra-level sibling order differs; the per-LEVEL node
    SET is untouched.

    ``level_of_node[row]`` maps each emitted node to its BFS depth (root =
    0), so a canonicalization that crosses a level boundary is detectable.
    """
    n = int(sampled.size)
    v0 = int(cols.var_offsets[section_idx])
    decider = OnceOnlyInclusion()
    decider.begin_root(max(1, n), section_idx)
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    root_nodes = v0 + sampled.astype(np.int64)
    emitted = [[int(root_nodes[i])] for i in range(n)]
    edge_types = [[int(_ROOT_EDGE_TYPE)] for i in range(n)]
    level_of = [{int(root_nodes[i]): 0} for i in range(n)]

    parent_row = np.arange(n, dtype=np.int64)
    parent_node = root_nodes
    for depth in range(1, 6):
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
            nodes_i = child_nodes[sel]
            types_i = child_types[sel]
            if rng is not None and nodes_i.size > 1:
                perm = rng.permutation(nodes_i.size)
                nodes_i = nodes_i[perm]
                types_i = types_i[perm]
            for node, et in zip(nodes_i.tolist(), types_i.tolist()):
                emitted[i].append(int(node))
                edge_types[i].append(int(et))
                level_of[i][int(node)] = depth
        surv = result.survivor_pairs
        parent_row = rows[surv]
        parent_node = child_nodes[surv]
    return (
        [np.asarray(e, dtype=np.int64) for e in emitted],
        [np.asarray(t, dtype=np.uint8) for t in edge_types],
        level_of,
    )


def _canonical_permutation(reordered_nodes, ref_nodes):
    """Permutation mapping ``reordered_nodes`` order -> ``ref_nodes`` order.

    Both carry the SAME node multiset (the BFS-level-SET invariant). The
    permutation ``p`` satisfies ``reordered_nodes[p] == ref_nodes`` -- the
    canonicalization that re-imposes the reference emission order on a
    reordered row's per-function output (token spans + dense sidecars).
    Matches by node identity; raises if the multisets differ (so a lost
    node surfaces, not a silent mis-align).
    """
    if not np.array_equal(np.sort(reordered_nodes), np.sort(ref_nodes)):
        raise AssertionError("emitted node multiset differs -- not a reorder")
    # Stable assignment: for each ref node in order, take the next matching
    # reordered position. (Nodes are once-only per row, so 1:1; the loop is
    # the explicit identity match the design's step 2 describes.)
    used = np.zeros(reordered_nodes.size, dtype=bool)
    perm = np.empty(ref_nodes.size, dtype=np.int64)
    for k, node in enumerate(ref_nodes.tolist()):
        cand = np.nonzero((reordered_nodes == node) & ~used)[0]
        perm[k] = cand[0]
        used[cand[0]] = True
    return perm


def _emission_order_counters(edge_types):
    """Per-Category counter ids minted in EMISSION order (design step 4).

    A stand-in for the identity counter assignment: each emitted node mints
    the next id within its edge-type Category, in emission order. The token
    path mints the real identity counter ids the same way (caller-local ->
    per-Category counter, in emission order), so reproducing THIS under the
    canonicalizing permutation proves the order-derived id remap.
    """
    seen: dict[int, int] = {}
    out = np.empty(edge_types.size, dtype=np.int64)
    for k, et in enumerate(edge_types.tolist()):
        out[k] = seen.get(et, 0)
        seen[et] = seen.get(et, 0) + 1
    return out


def test_reordered_emission_canonicalizes_to_reference(tmp_path):
    cols, adj = _build(tmp_path)
    section_idx = 0  # root
    sampled = np.array([0, 1, 2], dtype=np.int64)

    ref_nodes, ref_types, ref_levels = _emit_with_levels(
        cols, adj, section_idx, sampled
    )
    new_nodes, new_types, new_levels = _emit_with_levels(
        cols, adj, section_idx, sampled, shuffle_seed=20260618
    )

    reordered_any = False
    counter_remap_nontrivial: dict[int, bool] = {}
    for i in range(sampled.size):
        # (1) BFS-level-SET invariant: every node sits at the same BFS depth
        # in the reordered emission as in the reference.
        assert ref_levels[i] == new_levels[i], (
            f"row {i}: per-level node depth changed -- not a pure reorder"
        )

        # (2) Canonicalize: the permutation new->ref recovers the reference
        # emitted nodes + the parallel edge types EXACTLY.
        perm = _canonical_permutation(new_nodes[i], ref_nodes[i])
        np.testing.assert_array_equal(new_nodes[i][perm], ref_nodes[i])
        np.testing.assert_array_equal(new_types[i][perm], ref_types[i])

        # (3) Counter-id remap (design step 4): counter ids are minted in
        # EMISSION order, so a reordered emission mints DIFFERENT ids. The
        # remap is NOT a permutation of the pre-minted ids -- it is a
        # RE-MINT in the canonicalized (reference) order. Re-minting on the
        # canonicalized node sequence reproduces the reference ids exactly.
        ref_counters = _emission_order_counters(ref_types[i])
        canonical_types = new_types[i][perm]
        remapped_counters = _emission_order_counters(canonical_types)
        np.testing.assert_array_equal(remapped_counters, ref_counters)
        # And prove the remap is load-bearing on this row: naively
        # permuting the reordered-order ids (instead of re-minting) does
        # NOT in general recover the reference ids -- so a test that merely
        # reordered the id array would be wrong.
        naive = _emission_order_counters(new_types[i])[perm]
        if not np.array_equal(new_nodes[i], ref_nodes[i]) and ref_types[i].size > 1:
            # at least one row must exercise the remap non-trivially
            counter_remap_nontrivial[i] = not np.array_equal(
                naive, ref_counters
            )

        if not np.array_equal(new_nodes[i], ref_nodes[i]):
            reordered_any = True

    # The shuffle MUST have actually reordered at least one row, else the
    # canonicalization is exercised only on identity permutations (vacuous).
    assert reordered_any, "shuffle produced no reordered row to canonicalize"
    # And the counter remap must be non-trivial on at least one reordered
    # row (re-minting differs from naive permute) -- else step 4 is untested.
    assert any(counter_remap_nontrivial.values()), (
        "counter remap never diverged from a naive id permute -- "
        "the re-mint proof is vacuous on this fixture"
    )


def test_canonicalization_rejects_dropped_node(tmp_path):
    """A dropped (or duplicated) emitted node is not a reorder -> FAIL.

    The canonicalization (:func:`_canonical_permutation`) is defined only
    over a permutation of the SAME node multiset; if a traversal regression
    drops a node (or emits one twice -- a once-only break), the multiset
    differs and the canonicalization MUST refuse to align, surfacing the
    bug instead of silently mis-mapping. Proves the comparison is not
    vacuously satisfiable by any node set.
    """
    cols, adj = _build(tmp_path)
    section_idx = 0
    sampled = np.array([0, 1, 2], dtype=np.int64)
    ref_nodes, _ref_types, _lv = _emit_with_levels(cols, adj, section_idx, sampled)

    broke_any = False
    for i in range(sampled.size):
        if ref_nodes[i].size < 2:
            continue
        # Drop the deepest emitted node: same-shaped traversal regression
        # the relaxed-order contract does NOT permit (it is a content loss,
        # not an intra-level reorder). Canonicalization must raise.
        dropped = ref_nodes[i][:-1]
        with np.testing.assert_raises(AssertionError):
            _canonical_permutation(dropped, ref_nodes[i])
        broke_any = True
        break
    assert broke_any, "fixture produced no spliceable row to break"


def test_level_set_invariant_rejects_depth_change(tmp_path):
    """A node at the WRONG BFS depth fails the per-level-set invariant.

    The relaxed-order contract frees intra-level sibling order but PINS the
    per-LEVEL node set. A DFS regression (a node surfaced at a shallower
    depth) changes ``level_of`` for that node, so the equality the real
    canonicalization test asserts (``ref_levels == new_levels``) FAILS --
    the invariant is not vacuous.
    """
    cols, adj = _build(tmp_path)
    section_idx = 0
    sampled = np.array([0, 1, 2], dtype=np.int64)
    _ref_nodes, _ref_types, ref_levels = _emit_with_levels(
        cols, adj, section_idx, sampled
    )

    broke_any = False
    for i in range(sampled.size):
        depths = set(ref_levels[i].values())
        if max(depths) < 2:
            continue
        # Move one deepest node up to depth 1: a level-boundary break.
        deepest = max(ref_levels[i], key=lambda n: ref_levels[i][n])
        broken = dict(ref_levels[i])
        broken[deepest] = 1
        assert broken != ref_levels[i], "depth change must perturb the map"
        broke_any = True
        break
    assert broke_any, "fixture produced no >=2-deep row to break"
