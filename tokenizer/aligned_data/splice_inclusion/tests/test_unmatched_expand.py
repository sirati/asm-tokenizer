"""Tests for the unmatched-outline inlining transform.

The transform surfaces matched edges behind unmatched edges, recursing
unmatched->unmatched up to a depth cap with cycle guarding. These tests
exercise a synthetic call graph with a KNOWN unmatched->matched chain and
an unmatched->unmatched->matched chain, asserting the surfaced matched set,
the cap, the cycle guard, and per-row independence. They are
mutation-sensitive: breaking the cap, the cycle guard, the surfacing, or
the row attribution flips at least one assertion.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pytest

from tokenizer.aligned_data.splice_inclusion._unmatched_expand import (
    Edge,
    expand_unmatched_edges,
)


def _graph_resolver(
    graph: Dict[int, List[tuple]],
):
    """Build a ``resolve_children`` callback from a ``key -> children`` map.

    ``graph[key]`` is a list of ``(child_key, is_matched)`` tuples; the
    resolver stamps the PARENT edge's ``mask_row`` onto each child (the
    real loaders do this -- a surfaced callee inherits the row that reached
    its outline). ``payload`` mirrors ``dedup_key`` for assertion clarity.
    """

    def resolve(edge: Edge) -> Sequence[Edge]:
        return [
            Edge(
                mask_row=edge.mask_row,
                dedup_key=child_key,
                is_matched=is_matched,
                payload=child_key,
            )
            for child_key, is_matched in graph.get(edge.dedup_key, [])
        ]

    return resolve


def _matched(out: List[Edge]) -> List[tuple]:
    """``(mask_row, dedup_key)`` of each surfaced edge, in order."""
    return [(e.mask_row, e.dedup_key) for e in out]


def test_direct_matched_pass_through_unchanged():
    """Matched edges feed through in input order; unmatched leaf drops."""
    # 10 matched, 11 unmatched (no children -> leaf outline).
    edges = [
        Edge(mask_row=0, dedup_key=10, is_matched=True, payload=10),
        Edge(mask_row=0, dedup_key=11, is_matched=False, payload=11),
    ]
    out = expand_unmatched_edges(edges, _graph_resolver({}), max_unmatched_depth=3)
    # Matched 10 surfaces; the unmatched leaf 11 surfaces nothing.
    assert _matched(out) == [(0, 10)]


def test_unmatched_to_matched_chain_surfaces_callee():
    """An unmatched edge surfaces the matched callee one hop behind it."""
    # unmatched 20 -> matched 30.
    graph = {20: [(30, True)]}
    edges = [Edge(mask_row=0, dedup_key=20, is_matched=False, payload=20)]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=3)
    assert _matched(out) == [(0, 30)]


def test_unmatched_to_unmatched_to_matched_recurses():
    """Recursion through two unmatched hops surfaces the deep matched node."""
    # unmatched 40 -> unmatched 41 -> matched 42.
    graph = {40: [(41, False)], 41: [(42, True)]}
    edges = [Edge(mask_row=0, dedup_key=40, is_matched=False, payload=40)]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=3)
    assert _matched(out) == [(0, 42)]


def test_depth_cap_stops_recursion():
    """The cap bounds consecutive unmatched->unmatched hops.

    A chain of 4 unmatched hops to a matched leaf is NOT surfaced at cap=3
    (the matched node sits behind the 4th unmatched hop), but IS surfaced
    when the matched node sits within the cap.
    """
    # u50 -> u51 -> u52 -> u53 -> matched 54 (matched is the 5th node;
    # reaching it requires recursing INTO u53, i.e. a 4th unmatched hop).
    deep = {50: [(51, False)], 51: [(52, False)], 52: [(53, False)], 53: [(54, True)]}
    edges = [Edge(mask_row=0, dedup_key=50, is_matched=False, payload=50)]
    out3 = expand_unmatched_edges(edges, _graph_resolver(deep), max_unmatched_depth=3)
    # cap=3: recurse u50(d0)->u51(d1)->u52(d2); at u52 depth==2<3 so we
    # resolve its children (u53) and try to recurse u53 at depth 3, which
    # hits depth>=cap and returns BEFORE resolving u53's children -> 54
    # never surfaced.
    assert _matched(out3) == []
    out4 = expand_unmatched_edges(edges, _graph_resolver(deep), max_unmatched_depth=4)
    assert _matched(out4) == [(0, 54)]


def test_cap_zero_surfaces_nothing_behind_unmatched():
    """cap=0 drops every unmatched edge (no surfacing), matched pass through."""
    graph = {60: [(61, True)]}
    edges = [
        Edge(mask_row=0, dedup_key=60, is_matched=False, payload=60),
        Edge(mask_row=0, dedup_key=70, is_matched=True, payload=70),
    ]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=0)
    assert _matched(out) == [(0, 70)]


def test_unmatched_cycle_is_guarded():
    """An unmatched->unmatched cycle terminates without infinite recursion."""
    # u80 -> u81 -> u80 (cycle) and u81 -> matched 82.
    graph = {80: [(81, False)], 81: [(80, False), (82, True)]}
    edges = [Edge(mask_row=0, dedup_key=80, is_matched=False, payload=80)]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=3)
    # The cycle edge back to 80 is guarded; the matched 82 still surfaces.
    assert _matched(out) == [(0, 82)]


def test_same_matched_surfaced_once_per_row():
    """A matched node reached via two unmatched paths surfaces once per row."""
    # u90 -> matched 99; u91 -> matched 99. Both on row 0.
    graph = {90: [(99, True)], 91: [(99, True)]}
    edges = [
        Edge(mask_row=0, dedup_key=90, is_matched=False, payload=90),
        Edge(mask_row=0, dedup_key=91, is_matched=False, payload=91),
    ]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=3)
    assert _matched(out) == [(0, 99)]


def test_per_row_independence():
    """The same callee on different rows surfaces once PER row."""
    graph = {90: [(99, True)]}
    edges = [
        Edge(mask_row=0, dedup_key=90, is_matched=False, payload=90),
        Edge(mask_row=1, dedup_key=90, is_matched=False, payload=90),
    ]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=3)
    assert _matched(out) == [(0, 99), (1, 99)]


def test_order_is_deterministic_input_order():
    """Direct matched edges keep input order; surfaced follow their unmatched edge."""
    # m1 (matched), then unmatched u2 -> matched m3, then matched m4.
    graph = {2: [(3, True)]}
    edges = [
        Edge(mask_row=0, dedup_key=1, is_matched=True, payload=1),
        Edge(mask_row=0, dedup_key=2, is_matched=False, payload=2),
        Edge(mask_row=0, dedup_key=4, is_matched=True, payload=4),
    ]
    out = expand_unmatched_edges(edges, _graph_resolver(graph), max_unmatched_depth=3)
    assert _matched(out) == [(0, 1), (0, 3), (0, 4)]


def test_negative_cap_rejected():
    with pytest.raises(ValueError):
        expand_unmatched_edges([], _graph_resolver({}), max_unmatched_depth=-1)
