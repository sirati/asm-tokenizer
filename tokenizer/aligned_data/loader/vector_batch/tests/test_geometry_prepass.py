"""Unit tests for the body-free geometry PREPASS (plan C1).

Asserts, on a hand-computable synthetic corpus with a multi-level call
graph + multiple variants (so subsampling, splicing, and a straddler all
fire):

* the per-row INCLUDED set + BFS emission ORDER match the shared length
  twin (:func:`...sorted_index._graph_lengths.compute_node_lengths`) over
  the SAME variant set (the equivalence anchor: ``sum(own over emitted)``
  == the twin's depth-d length);
* the SINGLE straddler index + ``partial_cut_length`` for a hand-computed
  ``L``;
* the dense reservations equal the summed RLG3 totals;
* the remembered-excluded pool is the sampled-vs-full inclusion diff;
* the prepass is BODY-FREE -- it takes no ``_data.bin`` handle and reads
  none (the band derivations come from VocabularyManager, exercised by
  the bulk-geometry tests; the prepass itself sizes from stored RLG3
  counts only).
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.vector_batch import compute_batch_geometry
from tokenizer.aligned_data.loader.vector_batch._inclusion import (
    RowInclusionView,
    compute_row_inclusions,
)
from tokenizer.aligned_data.sorted_index._graph_lengths import (
    compute_node_lengths,
)

from ._synthetic import build_synthetic_corpus


def _geometry(seq_len: int, sampled_variants, *, max_depth: int = 2):
    c = build_synthetic_corpus()
    g = compute_batch_geometry(
        cols=c.cols,
        section_offsets=c.section_offsets,
        geometry=c.geometry,
        variants_u8=c.variants_u8,
        root_sections=np.zeros(len(sampled_variants), dtype=np.int64),
        root_sampled_variants=np.asarray(sampled_variants, dtype=np.int64),
        root_groups=np.zeros(len(sampled_variants), dtype=np.int64),
        seq_len=seq_len,
        max_depth=max_depth,
    )
    return c, g


# ---------------------------------------------------------------------------
# Emission order + included set
# ---------------------------------------------------------------------------


def test_emission_order_and_included_set():
    """row0 (root_v0) splices B then C in BFS order; row1 (root_v1) is
    root-only (its only callee sec1 is all-reached -> excluded)."""
    _c, g = _geometry(seq_len=100, sampled_variants=[0, 1])
    em = g.emission
    assert em.row_offsets.tolist() == [0, 3, 4]
    # row0 = [root_v0(0), B(3), C(4)] in BFS order; row1 = [root_v1(1)].
    assert em.node.tolist() == [0, 3, 4, 1]
    # own = body + 1 (self-token), variant prefix kept OUT of own_length.
    assert em.own_length.tolist() == [6, 10, 7, 8]


def test_emitted_edge_types_root_local_callees_slot_type():
    """The root entry of every row is the LOCAL edge (root self-token
    category is LOCAL_FUNC); the synthetic corpus's call_targets are all
    LOCAL, so every spliced callee carries the LOCAL edge type too."""
    from tokenizer.aligned_data.call_target_type import CallTargetType

    _c, g = _geometry(seq_len=100, sampled_variants=[0, 1])
    em = g.emission
    # edge_type is parallel to node; root + all-LOCAL callees -> all LOCAL.
    assert em.edge_type.dtype == np.uint8
    assert em.edge_type.shape == em.node.shape
    assert em.edge_type.tolist() == [int(CallTargetType.LOCAL)] * em.node.size
    # The first emitted node of each row (the root) is always a LOCAL edge.
    roots = em.row_offsets[:-1]
    assert (em.edge_type[roots] == int(CallTargetType.LOCAL)).all()


def test_inclusion_matches_length_twin():
    """sum(own over emitted) per row equals the body-free length twin's
    depth-2 spliced length for the row's root node -- the equivalence
    anchor proving the included set + own-lengths line up with the index
    build's path."""
    c = build_synthetic_corpus()
    twin = compute_node_lengths(
        c.cols, c.section_offsets, c.body_len, depths=[2]
    )[2]
    own = c.body_len + 1
    incs = RowInclusionView(compute_row_inclusions(
        c.cols,
        c.section_offsets,
        root_sections=np.array([0, 0]),
        root_sampled_variants=np.array([0, 1]),
        root_groups=np.array([0, 0]),
        max_depth=2,
    ))
    # root_v0 node = 0, root_v1 node = 1.
    assert int(own[incs[0].emitted_nodes].sum()) == int(twin[0])
    assert int(own[incs[1].emitted_nodes].sum()) == int(twin[1])


# ---------------------------------------------------------------------------
# Straddler + partial cut
# ---------------------------------------------------------------------------


def test_straddler_and_partial_cut():
    """Hand-computed straddler for L=12. row0 prefix=1, body running ends
    6/16/23; remaining=11 -> first end>11 is 16 (B, local idx 1), prev
    end 6 -> partial cut 5. row1 total 9 <= 12 -> full (no straddler)."""
    _c, g = _geometry(seq_len=12, sampled_variants=[0, 1])
    layout = g.layout
    assert layout.prefix_len.tolist() == [1, 1]
    assert layout.total_length.tolist() == [24, 9]
    assert layout.straddler_local_idx.tolist() == [1, -1]
    assert layout.partial_cut_length.tolist() == [5, 0]


def test_prefix_overflow_cuts_first_body_function():
    """When the variant prefix alone meets/exceeds L the first body
    function is the straddler with a 0-column partial cut."""
    c = build_synthetic_corpus()
    # root_v0 prefix = 1; choose L = 1 so the prefix fills the row and the
    # first body function (root body) cuts to 0 columns.
    g = compute_batch_geometry(
        cols=c.cols,
        section_offsets=c.section_offsets,
        geometry=c.geometry,
        variants_u8=c.variants_u8,
        root_sections=np.array([0]),
        root_sampled_variants=np.array([0]),
        root_groups=np.array([0]),
        seq_len=1,
        max_depth=2,
    )
    # single-variant subset -> FLAG-A: root-only emission.
    assert g.emission.node.tolist() == [0]
    assert g.layout.straddler_local_idx.tolist() == [0]
    assert g.layout.partial_cut_length.tolist() == [0]


def test_full_row_when_total_under_l():
    """A row whose total emitted width is <= L has no straddler."""
    _c, g = _geometry(seq_len=100, sampled_variants=[0, 1])
    assert g.layout.straddler_local_idx.tolist() == [-1, -1]
    assert g.layout.partial_cut_length.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Dense reservations
# ---------------------------------------------------------------------------


def test_dense_reservations_sum_rlg3_totals():
    """Per-row reservation = sum over emitted nodes of the stored RLG3
    id / value TOTALS; offsets are the exclusive prefix sums."""
    c, g = _geometry(seq_len=100, sampled_variants=[0, 1])
    res = g.reservation
    # row0 emitted {0,3,4}: id 2+3+2=7, value 1+1+1=3.
    # row1 emitted {1}:     id 1,        value 0.
    assert res.id_reserved.tolist() == [7, 1]
    assert res.value_reserved.tolist() == [3, 0]
    assert res.id_offsets.tolist() == [0, 7, 8]
    assert res.value_offsets.tolist() == [0, 3, 3]
    # Reservations equal the RLG3 totals gathered over the emission.
    em = g.emission
    assert int(em.id_total.sum()) == int(res.id_reserved.sum())
    assert int(em.value_total.sum()) == int(res.value_reserved.sum())


# ---------------------------------------------------------------------------
# Remembered-excluded pool (sampled-vs-full diff)
# ---------------------------------------------------------------------------


def test_remembered_excluded_pool():
    """row0 (root_v0) emits the full set's callees -> empty pool; row1
    (root_v1) emits nothing the full set included -> pool = {B, C}."""
    _c, g = _geometry(seq_len=100, sampled_variants=[0, 1])
    assert g.excluded_pool_offsets.tolist() == [0, 0, 2]
    assert g.excluded_pool.tolist() == [3, 4]  # B-node, C-node


def test_single_variant_subset_pool_is_full_set():
    """Sampling a single variant (FLAG-A: splices nothing) leaves the
    whole full-set inclusion in the remembered-excluded pool."""
    c = build_synthetic_corpus()
    g = compute_batch_geometry(
        cols=c.cols,
        section_offsets=c.section_offsets,
        geometry=c.geometry,
        variants_u8=c.variants_u8,
        root_sections=np.array([0]),
        root_sampled_variants=np.array([0]),
        root_groups=np.array([0]),
        seq_len=100,
        max_depth=2,
    )
    assert g.emission.node.tolist() == [0]  # root only
    # full-set inclusion of sec0 root = {B(3), C(4)}.
    assert g.excluded_pool.tolist() == [3, 4]


# ---------------------------------------------------------------------------
# Body-free contract
# ---------------------------------------------------------------------------


def test_prepass_takes_no_data_handle():
    """The prepass signature carries NO ``_data.bin`` parameter -- it is
    structurally body-free (positions from stored body_len, reservations
    from stored counts). A regression that started reading bodies would
    need a new parameter here, which this test pins against."""
    import inspect

    sig = inspect.signature(compute_batch_geometry)
    params = set(sig.parameters)
    assert "data_u8" not in params
    assert "data" not in params
    # The handles it DOES take are all body-free sidecars.
    assert {"cols", "section_offsets", "geometry", "variants_u8"} <= params


def test_max_depth_zero_emits_only_roots():
    """depth 0: every row is root-only, reservations are the root's own
    totals, no straddler when L is large."""
    _c, g = _geometry(seq_len=100, sampled_variants=[0, 1], max_depth=0)
    assert g.emission.node.tolist() == [0, 1]
    assert g.reservation.id_reserved.tolist() == [2, 1]
