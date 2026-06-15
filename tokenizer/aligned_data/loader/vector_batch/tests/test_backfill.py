"""Unit tests for the leftover-budget BACKFILL transform (plan TD).

Backfill is a geometry -> geometry transform: it greedily packs each
row's leftover ``[B, L]`` token budget from that row's remembered-
excluded pool. It is validated HERE at the geometry level (no scatter
needed) against its OWN invariants on the hand-computable synthetic
corpus -- NOT against the old decode path.

Synthetic ground truth (sampled variants ``[0, 1]``, max_depth 2):

* body_len per NODE: root_v0=5, root_v1=7, A=3, B=9, C=6
  -> own = body + 1 = 6, 8, 4, 10, 7.
* row0 (root_v0) emits [root_v0(0), B(3), C(4)], prefix 1,
  total_length = 1 + 6 + 10 + 7 = 24, excluded_pool EMPTY.
* row1 (root_v1) emits [root_v1(1)], prefix 1, total_length = 1 + 8 = 9,
  excluded_pool = {B(3) own 10, C(4) own 7}.

So row1 is the backfill subject: its leftover = L - 9 is filled from
{B, C} in POOL ORDER (B first), fit-fully. row0's empty pool makes it a
natural no-op control.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.vector_batch import compute_batch_geometry
from tokenizer.aligned_data.loader.vector_batch._backfill import (
    backfill_geometry,
)

from ._synthetic import build_synthetic_corpus


# --- node-index constants (catalog node indices, var_offsets-major) -----
ROOT_V0 = 0
ROOT_V1 = 1
NODE_A = 2
NODE_B = 3
NODE_C = 4

OWN = {ROOT_V0: 6, ROOT_V1: 8, NODE_A: 4, NODE_B: 10, NODE_C: 7}


def _geometry(seq_len: int, sampled_variants, *, max_depth: int = 2):
    c = build_synthetic_corpus()
    g = compute_batch_geometry(
        cols=c.cols,
        section_offsets=c.section_offsets,
        geometry=c.geometry,
        variants_u8=c.variants_u8,
        root_sections=np.zeros(len(sampled_variants), dtype=np.int64),
        root_sampled_variants=np.asarray(sampled_variants, dtype=np.int64),
        seq_len=seq_len,
        max_depth=max_depth,
    )
    return c, g


def _backfill(c, g):
    return backfill_geometry(
        g,
        body_lengths=c.geometry.body_lengths,
        id_counts=c.geometry.id_counts,
        value_counts=c.geometry.value_counts,
    )


def _row_nodes(g, r):
    lo = int(g.emission.row_offsets[r])
    hi = int(g.emission.row_offsets[r + 1])
    return g.emission.node[lo:hi].tolist()


# ---------------------------------------------------------------------------
# Greedy packing: leftover budget filled as far as the rule allows
# ---------------------------------------------------------------------------


def test_backfill_appends_one_when_only_smaller_fits():
    """L=18: row1 leftover = 18 - 9 = 9. Pool order B(own 10) does NOT
    fit; C(own 7) DOES -> append C only. New total = 16 <= 18."""
    c, g = _geometry(seq_len=18, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert _row_nodes(aug, 1) == [ROOT_V1, NODE_C]
    assert int(aug.layout.total_length[1]) == 16
    # row0's pool is empty -> untouched.
    assert _row_nodes(aug, 0) == _row_nodes(g, 0)


def test_backfill_appends_larger_first_in_pool_order():
    """L=20: leftover 11 admits B(own 10) FIRST (pool order); remaining 1
    rejects C(own 7) -> [root_v1, B]. Proves pool order, not descending
    size, and the fit-fully rule."""
    c, g = _geometry(seq_len=20, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert _row_nodes(aug, 1) == [ROOT_V1, NODE_B]
    assert int(aug.layout.total_length[1]) == 19


def test_backfill_packs_whole_pool_when_budget_large():
    """L=30: leftover 21 admits BOTH B(10) then C(7) -> [root_v1, B, C].
    total = 9 + 10 + 7 = 26 <= 30."""
    c, g = _geometry(seq_len=30, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert _row_nodes(aug, 1) == [ROOT_V1, NODE_B, NODE_C]
    assert int(aug.layout.total_length[1]) == 26


def test_smaller_function_still_tried_after_larger_skipped():
    """A larger pool member that does NOT fit must NOT abort the row: the
    later smaller one is still tried. L=18 (B skipped, C taken) IS that
    case -- assert C survived despite B being earlier + unfit."""
    c, g = _geometry(seq_len=18, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert NODE_B not in _row_nodes(aug, 1)
    assert NODE_C in _row_nodes(aug, 1)


# ---------------------------------------------------------------------------
# Invariant: never overflow L
# ---------------------------------------------------------------------------


def test_backfill_never_pushes_a_fitting_row_over_l():
    """``total_length`` is the PRE-truncation width: a row already over L
    (its body is the straddler's cut) reports total_length > L BEFORE
    backfill -- that is the geometry's own contract, not a backfill bug.
    The backfill INVARIANT is: a row that fit (total_length <= L) STILL
    fits afterwards, and a row already over L gains NOTHING (its negative
    remaining admits no function). Layout stays consistent either way.
    """
    for seq_len in range(1, 40):
        c, g = _geometry(seq_len=seq_len, sampled_variants=[0, 1])
        aug = _backfill(c, g)
        for r in range(aug.n_rows):
            orig_total = int(g.layout.total_length[r])
            aug_total = int(aug.layout.total_length[r])
            if orig_total <= seq_len:
                # A row that fit must STILL fit after backfill.
                assert aug_total <= seq_len, (seq_len, r)
            else:
                # An already-over row gains nothing (no member fits a
                # negative remaining budget); total_length is unchanged.
                assert aug_total == orig_total, (seq_len, r)
            # total_length == prefix + sum(own over the row) -- layout
            # consistency holds regardless.
            lo = int(aug.emission.row_offsets[r])
            hi = int(aug.emission.row_offsets[r + 1])
            own_sum = int(aug.emission.own_length[lo:hi].sum())
            exp = int(aug.layout.prefix_len[r]) + own_sum
            assert aug_total == exp, (seq_len, r)


# ---------------------------------------------------------------------------
# Invariant: no function included twice
# ---------------------------------------------------------------------------


def test_no_function_included_twice():
    """No node appears twice within a row after backfill (original
    emission union backfill stays a set)."""
    for seq_len in range(1, 40):
        c, g = _geometry(seq_len=seq_len, sampled_variants=[0, 1])
        aug = _backfill(c, g)
        for r in range(aug.n_rows):
            nodes = _row_nodes(aug, r)
            assert len(nodes) == len(set(nodes)), (seq_len, r, nodes)


# ---------------------------------------------------------------------------
# Invariant: backfilled functions come ONLY from that row's pool
# ---------------------------------------------------------------------------


def test_backfilled_nodes_are_from_own_pool():
    """Every node added to a row beyond its original emission is a member
    of THAT row's excluded_pool -- never another row's, never fresh."""
    for seq_len in range(1, 40):
        c, g = _geometry(seq_len=seq_len, sampled_variants=[0, 1])
        aug = _backfill(c, g)
        for r in range(aug.n_rows):
            orig = set(_row_nodes(g, r))
            new = [n for n in _row_nodes(aug, r) if n not in orig]
            p_lo = int(g.excluded_pool_offsets[r])
            p_hi = int(g.excluded_pool_offsets[r + 1])
            own_pool = set(g.excluded_pool[p_lo:p_hi].tolist())
            assert set(new) <= own_pool, (seq_len, r, new, own_pool)


def test_pool_carried_through_unchanged():
    """The excluded_pool + offsets are the immutable candidate record;
    backfill carries them through verbatim."""
    c, g = _geometry(seq_len=30, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert aug.excluded_pool.tolist() == g.excluded_pool.tolist()
    assert aug.excluded_pool_offsets.tolist() == (
        g.excluded_pool_offsets.tolist()
    )


# ---------------------------------------------------------------------------
# Invariant: determinism (same input -> same augmented geometry)
# ---------------------------------------------------------------------------


def test_determinism_two_runs_identical():
    """Backfilling the same geometry twice yields byte-identical augmented
    emission / layout / reservation."""
    c, g = _geometry(seq_len=22, sampled_variants=[0, 1])
    a = _backfill(c, g)
    b = _backfill(c, g)
    assert a.emission.node.tolist() == b.emission.node.tolist()
    assert a.emission.row_offsets.tolist() == b.emission.row_offsets.tolist()
    assert a.emission.own_length.tolist() == b.emission.own_length.tolist()
    assert a.layout.total_length.tolist() == b.layout.total_length.tolist()
    assert (
        a.layout.straddler_local_idx.tolist()
        == b.layout.straddler_local_idx.tolist()
    )
    assert a.reservation.id_offsets.tolist() == b.reservation.id_offsets.tolist()
    assert (
        a.reservation.value_offsets.tolist()
        == b.reservation.value_offsets.tolist()
    )


# ---------------------------------------------------------------------------
# No-op: a row already full (or empty pool) is unchanged
# ---------------------------------------------------------------------------


def test_noop_when_no_pool_member_fits():
    """L=15: row1 leftover = 6; neither B(10) nor C(7) fits -> row
    unchanged. row0 (empty pool) also unchanged. Emission is identical."""
    c, g = _geometry(seq_len=15, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert _row_nodes(aug, 1) == _row_nodes(g, 1)
    assert _row_nodes(aug, 0) == _row_nodes(g, 0)
    assert aug.emission.node.tolist() == g.emission.node.tolist()
    assert aug.emission.row_offsets.tolist() == g.emission.row_offsets.tolist()


def test_noop_when_pool_empty():
    """row0's pool is empty for the full batch -> backfill is a no-op for
    it regardless of leftover budget."""
    c, g = _geometry(seq_len=100, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    assert _row_nodes(aug, 0) == _row_nodes(g, 0)


def test_original_geometry_not_mutated():
    """Backfill returns a NEW geometry; the input's emission is untouched
    (callers may keep the OFF-path geometry alongside)."""
    c, g = _geometry(seq_len=30, sampled_variants=[0, 1])
    before = g.emission.node.tolist()
    _ = _backfill(c, g)
    assert g.emission.node.tolist() == before


# ---------------------------------------------------------------------------
# Augmented geometry stays a VALID BatchGeometry
# ---------------------------------------------------------------------------


def test_augmented_csr_and_reservation_consistent():
    """The augmented geometry's CSR offsets, dense reservations, and
    layout are self-consistent (the scatter consumes it unchanged)."""
    c, g = _geometry(seq_len=30, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    em = aug.emission
    # CSR well-formed: monotone, starts 0, ends at n_emitted.
    assert em.row_offsets[0] == 0
    assert int(em.row_offsets[-1]) == int(em.node.size)
    assert bool((np.diff(em.row_offsets) >= 0).all())
    # Reservation = segmented sum of the augmented totals.
    res = aug.reservation
    for r in range(aug.n_rows):
        lo = int(em.row_offsets[r])
        hi = int(em.row_offsets[r + 1])
        assert int(res.id_reserved[r]) == int(em.id_total[lo:hi].sum())
        assert int(res.value_reserved[r]) == int(em.value_total[lo:hi].sum())
    # Offsets are the exclusive prefix sums of the reserved totals.
    assert res.id_offsets.tolist() == (
        [0] + np.cumsum(res.id_reserved).tolist()
    )
    assert res.value_offsets.tolist() == (
        [0] + np.cumsum(res.value_reserved).tolist()
    )


def test_backfilled_geometry_triple_matches_rlg3_axes():
    """An appended node's own_length / id_total / value_total in the
    augmented emission equal the RLG3 axes (own = 1 + body) -- backfill
    sizes from the SAME geometry, not a fresh guess."""
    c, g = _geometry(seq_len=30, sampled_variants=[0, 1])
    aug = _backfill(c, g)
    em = aug.emission
    for flat in range(em.node.size):
        n = int(em.node[flat])
        assert int(em.own_length[flat]) == 1 + int(c.geometry.body_lengths[n])
        assert int(em.id_total[flat]) == int(c.geometry.id_counts[n])
        assert int(em.value_total[flat]) == int(c.geometry.value_counts[n])
