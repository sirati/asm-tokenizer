"""Equivalence: ``LiveNodeAdjacency.expand_batch`` vs scalar ``__call__``.

Pins the vectorized batched frontier expansion against the per-node
scalar path on a real columnar catalog: for an arbitrary frontier of
parent nodes, the batched ``(parent_pos, child_secs, child_nodes,
child_types, child_matched)`` must, when regrouped per parent, reproduce
EXACTLY the scalar ``adjacency(node)`` 4-tuple (same children, same
ascending-slot order, same secs/types/matched). This is the load-bearing
correctness gate for the BFS vectorization.
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


def _variant(vkey, seed, called):
    base = make_simple_variant(vkey, token_seed=seed, n_tokens=6)
    return VariantSpec(
        vkey=base.vkey,
        tokens=base.tokens,
        block_rl=base.block_rl,
        insn_rl=base.insn_rl,
        called=tuple(called),
    )


def _build_adjacency(tmp_path: Path):
    """A small multi-section call graph with diamonds + per-variant calls."""
    specs = [
        MatchedFunctionSpec(
            func_name="root",
            variants=(
                _variant(("V", 0), 1, ("a", "b")),
                _variant(("V", 1), 2, ("a", "c")),
                _variant(("V", 2), 3, ("b", "c", "a")),
            ),
            called=("a", "b", "c"),
        ),
        MatchedFunctionSpec(
            func_name="a",
            variants=(
                _variant(("V", 0), 4, ("leaf",)),
                _variant(("V", 1), 5, ()),
            ),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="b",
            variants=(_variant(("V", 0), 6, ("a",)),),
            called=("a",),
        ),
        MatchedFunctionSpec(
            func_name="c",
            variants=(
                _variant(("V", 0), 7, ("a", "b")),
                _variant(("V", 1), 8, ("leaf",)),
            ),
            called=("a", "b", "leaf"),
        ),
        MatchedFunctionSpec(
            func_name="leaf",
            variants=(_variant(("V", 0), 9, ()),),
            called=(),
        ),
    ]
    build_corpus(tmp_path, "adjb", matched=specs)
    starts, lens = read_csv_section_index_arrays(tmp_path / "adjb_index.bin")
    blob = np.fromfile(tmp_path / "adjb_sections.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lens)
    adj = LiveNodeAdjacency(cols, starts, cols.sec_of_var)
    return cols, adj


def _scalar_expand(adj, parents):
    """Per-parent scalar 4-tuples, in parent then ascending-slot order."""
    out = []
    for p in parents.tolist():
        out.append(adj(int(p)))
    return out


def test_expand_batch_matches_scalar_full_frontier(tmp_path: Path) -> None:
    cols, adj = _build_adjacency(tmp_path)
    parents = np.arange(int(cols.var_offsets[-1]), dtype=np.int64)

    p_pos, c_secs, c_nodes, c_types, c_matched = adj.expand_batch(parents)
    scalar = _scalar_expand(adj, parents)

    for i, p in enumerate(parents.tolist()):
        mask = p_pos == i
        # The batched flattening keeps parents ascending and a parent's
        # edges in the order it concatenated them (ascending slot, the
        # scalar order), so a stable selection reproduces the scalar tuple.
        b_nodes = c_nodes[mask]
        b_secs = c_secs[mask]
        b_types = c_types[mask]
        b_matched = c_matched[mask]
        s_nodes, s_secs, s_types, s_matched = scalar[i]
        np.testing.assert_array_equal(b_nodes, s_nodes)
        np.testing.assert_array_equal(b_secs, s_secs)
        np.testing.assert_array_equal(b_types, s_types)
        np.testing.assert_array_equal(b_matched, s_matched)


def test_expand_batch_subset_frontier(tmp_path: Path) -> None:
    """A frontier with repeats + arbitrary order still matches per pos."""
    cols, adj = _build_adjacency(tmp_path)
    total = int(cols.var_offsets[-1])
    rng = np.random.default_rng(7)
    parents = rng.integers(0, total, size=11).astype(np.int64)

    p_pos, _secs, c_nodes, c_types, _m = adj.expand_batch(parents)
    for i, p in enumerate(parents.tolist()):
        b_nodes = c_nodes[p_pos == i]
        s_nodes, _ss, s_types, _sm = adj(int(p))
        np.testing.assert_array_equal(b_nodes, s_nodes)
        np.testing.assert_array_equal(c_types[p_pos == i], s_types)


def test_expand_batch_empty_frontier(tmp_path: Path) -> None:
    _cols, adj = _build_adjacency(tmp_path)
    p_pos, secs, nodes, types, matched = adj.expand_batch(
        np.zeros(0, dtype=np.int64)
    )
    assert p_pos.size == nodes.size == secs.size == types.size == 0
    assert matched.size == 0
