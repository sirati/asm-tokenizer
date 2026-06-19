"""Tests for the per-variant cutoff walk (Stage 2, phase 2b).

Covers the cut-call-target identification + per-call-target surviving
count derivation over a list of predicted full lengths plus a
``context_len`` budget. See ``batch_decode_plan.md`` D8 + Stage 2 step
4 for the algorithm and ``_cutoff_walk.py`` module docstring for the
cut convention at exact boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._cutoff_walk import (
    CutoffResult,
    walk_cutoff,
    walk_cutoff_batched,
)


def _assert_invariants(
    predicted: list[int],
    context_len: int,
    result: CutoffResult,
) -> None:
    """Cross-cutting checks every ``walk_cutoff`` call must satisfy.

    Encapsulates: sum bound, exact-fill-when-overflow, len agreement on
    the two parallel lists, single-cut-entry guarantee, full-included
    vs partial vs dropped per-position consistency with
    ``cut_call_target_index`` + ``partial_cut_length``.
    """

    n = len(predicted)
    assert len(result.surviving_token_counts) == n
    assert len(result.is_cut_flags) == n

    total = sum(result.surviving_token_counts)
    assert total <= context_len
    if sum(predicted) >= context_len:
        assert total == context_len

    # At most one True in is_cut_flags; True position equals cut idx.
    cut_positions = [i for i, f in enumerate(result.is_cut_flags) if f]
    if result.cut_call_target_index < n:
        assert cut_positions == [result.cut_call_target_index]
    else:
        assert cut_positions == []
        assert result.partial_cut_length == 0

    # Per-position consistency with cut_idx + partial.
    for i in range(n):
        if i < result.cut_call_target_index:
            assert result.surviving_token_counts[i] == predicted[i]
            assert result.is_cut_flags[i] is False
        elif i == result.cut_call_target_index:
            assert result.surviving_token_counts[i] == result.partial_cut_length
            assert result.is_cut_flags[i] is True
            # Cut entry's partial must be strictly less than its full
            # length (else it would be "fully included" instead).
            assert 0 <= result.partial_cut_length < predicted[i]
        else:
            assert result.surviving_token_counts[i] == 0
            assert result.is_cut_flags[i] is False


def test_empty_input() -> None:
    """Empty list: sentinel cut_idx = 0 = len, no partial, empty lists."""

    result = walk_cutoff([], context_len=42)

    assert result.cut_call_target_index == 0
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == []
    assert result.is_cut_flags == []


def test_full_fit_under_budget() -> None:
    """Sum < context_len: cut_idx = len, all surviving = predicted, no
    is_cut flags True."""

    predicted = [3, 4, 5]
    context_len = 100

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == len(predicted)
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == predicted
    assert all(f is False for f in result.is_cut_flags)
    _assert_invariants(predicted, context_len, result)


def test_single_call_target_longer_than_context() -> None:
    """One entry whose length exceeds context_len: cut at index 0 with
    partial = context_len; surviving = [context_len]."""

    predicted = [50]
    context_len = 20

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 0
    assert result.partial_cut_length == 20
    assert result.surviving_token_counts == [20]
    assert result.is_cut_flags == [True]
    _assert_invariants(predicted, context_len, result)


def test_three_call_targets_mid_stream_cut() -> None:
    """[10, 10, 10] context=15: cut on index 1 with partial=5, surviving
    = [10, 5, 0], is_cut = [F, T, F]."""

    predicted = [10, 10, 10]
    context_len = 15

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 1
    assert result.partial_cut_length == 5
    assert result.surviving_token_counts == [10, 5, 0]
    assert result.is_cut_flags == [False, True, False]
    _assert_invariants(predicted, context_len, result)


def test_exact_boundary_cut_at_index_one() -> None:
    """[10, 10] context=10: the cumsum hits the budget exactly between
    the two entries; convention puts the cut on index 1 with
    partial = 0 (entry K is "dropped at offset 0"), keeping the cut
    entry's partial strictly less than its full length so the
    fully-included vs cut distinction stays uniform.

    See ``_cutoff_walk.py`` module docstring for the rationale.
    """

    predicted = [10, 10]
    context_len = 10

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 1
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == [10, 0]
    assert result.is_cut_flags == [False, True]
    _assert_invariants(predicted, context_len, result)


def test_exact_boundary_cut_with_zero_length_tail() -> None:
    """[5, 0] context=5: cumsum stays at the budget after the zero-
    length tail; the variant fits entirely (sum == context_len), so
    cut_idx = len with no partial.
    """

    predicted = [5, 0]
    context_len = 5

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 2  # = len(predicted)
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == [5, 0]
    assert result.is_cut_flags == [False, False]
    _assert_invariants(predicted, context_len, result)


def test_all_zero_lengths() -> None:
    """All-zero lengths with context_len > 0: nothing overflows, the
    variant fully fits (with zero tokens), no cut.
    """

    predicted = [0, 0, 0]
    context_len = 5

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == len(predicted)
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == [0, 0, 0]
    assert result.is_cut_flags == [False, False, False]
    _assert_invariants(predicted, context_len, result)


def test_context_len_zero_cuts_first_entry() -> None:
    """context_len = 0 with a non-empty positive-length list: cut on
    index 0 with partial = 0, all subsequent dropped.
    """

    predicted = [3, 4, 5]
    context_len = 0

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 0
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == [0, 0, 0]
    assert result.is_cut_flags == [True, False, False]
    _assert_invariants(predicted, context_len, result)


def test_context_len_zero_with_empty() -> None:
    """context_len = 0 with empty input: sentinel cut_idx = 0 = len."""

    result = walk_cutoff([], context_len=0)

    assert result.cut_call_target_index == 0
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == []
    assert result.is_cut_flags == []


def test_many_call_targets_spot_check_cut_location() -> None:
    """Twelve entries of length 7 with context_len = 50: cumsum
    [7,14,21,28,35,42,49,56,...]; first cumsum > 50 is at index 7
    (cumsum=56), so cut on index 7 with partial = 50 - 49 = 1.
    """

    predicted = [7] * 12
    context_len = 50

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 7
    assert result.partial_cut_length == 1
    assert result.surviving_token_counts == [7] * 7 + [1] + [0] * 4
    assert result.is_cut_flags == [False] * 7 + [True] + [False] * 4
    _assert_invariants(predicted, context_len, result)


def test_zero_length_root_with_positive_callees() -> None:
    """[0, 5, 0, 10] context=7: cumsum [0,5,5,15]; first cumsum > 7 is
    index 3, partial = 7 - 5 = 2; root + zero-length callees fully
    included.
    """

    predicted = [0, 5, 0, 10]
    context_len = 7

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 3
    assert result.partial_cut_length == 2
    assert result.surviving_token_counts == [0, 5, 0, 2]
    assert result.is_cut_flags == [False, False, False, True]
    _assert_invariants(predicted, context_len, result)


def test_exact_boundary_then_more_callees() -> None:
    """[5, 0, 5] context=5: cumsum [5,5,10]; first cumsum > 5 is index
    2 (cumsum=10), partial = 5 - 5 = 0. Cut entry is index 2 with
    partial 0 (per the boundary convention); intervening zero-length
    callee fully included.
    """

    predicted = [5, 0, 5]
    context_len = 5

    result = walk_cutoff(predicted, context_len=context_len)

    assert result.cut_call_target_index == 2
    assert result.partial_cut_length == 0
    assert result.surviving_token_counts == [5, 0, 0]
    assert result.is_cut_flags == [False, False, True]
    _assert_invariants(predicted, context_len, result)


@pytest.mark.parametrize(
    "predicted,context_len",
    [
        ([], 0),
        ([], 100),
        ([10], 0),
        ([10], 5),
        ([10], 10),
        ([10], 20),
        ([5, 5], 0),
        ([5, 5], 5),
        ([5, 5], 9),
        ([5, 5], 10),
        ([5, 5], 11),
        ([3, 4, 5, 6, 7], 12),
        ([1] * 100, 37),
        ([0] * 10, 50),
        ([0, 7, 0, 7], 7),
    ],
)
def test_invariants_hold_across_grid(predicted: list[int], context_len: int) -> None:
    """The per-position consistency + sum bound + exact-fill invariants
    must hold for every (predicted, context_len) pair regardless of
    shape. The grid covers empty inputs, full-fit + cut + exact-boundary
    + zero-length interleavings + large-N.
    """

    result = walk_cutoff(predicted, context_len=context_len)
    _assert_invariants(predicted, context_len, result)


# ---------------------------------------------------------------------------
# Batched walk: a LATER variant must cut with its own per-variant-LOCAL
# cumsum reset (the path production uses exclusively via _length_predict.py).
# ---------------------------------------------------------------------------


def test_batched_later_variant_cuts_with_local_cumsum_reset() -> None:
    """Two variants where the SECOND one cuts, exercising the per-variant
    base subtraction (``- variant_base[ct_variant]``) and the overflow
    counter reset (``- over_base[ct_variant]``) that single-variant tests
    can never reach (with one variant both bases are zero, so the resets
    are no-ops).

    Flat layout (variant-then-DFS order)::

        lengths = [3, 3, 3, 5, 2]
        variant_offsets = [0, 3, 5]
        context_lens = [4, 6]

    Variant 0 (lengths [3,3,3], budget 4):
        cumsum_after = [3, 6, 9]; first > 4 is index 1 (6 > 4).
        partial = budget - cumsum_before = 4 - 3 = 1.
        surviving = [3, 1, 0]; is_cut = [F, T, F]; cut_idx = 1.

    Variant 1 (lengths [5,2], budget 6):
        LOCAL cumsum_after = [5, 7]; first > 6 is index 1 (7 > 6).
        partial = 6 - 5 = 1. surviving = [5, 1]; is_cut = [F, T];
        cut_idx = 1.

    Without the ``variant_base`` reset, variant 1's GLOBAL cumsum_before
    would be [9, 14] -> budget 6 minus that clamps surviving to [0, 0]
    and shifts the overflow to index 0. Without the ``over_base`` reset,
    variant 1's overflow counter is already non-zero (variant 0 cut), so
    ``excl_over == 0`` never holds and variant 1 records NO cut. Either
    mutation changes the asserted values below.
    """

    lengths = np.array([3, 3, 3, 5, 2], dtype=np.int64)
    variant_offsets = np.array([0, 3, 5], dtype=np.int64)
    context_lens = np.array([4, 6], dtype=np.int64)

    columns = walk_cutoff_batched(lengths, variant_offsets, context_lens)

    # Flat per-call-target columns.
    np.testing.assert_array_equal(
        columns.surviving_token_counts,
        np.array([3, 1, 0, 5, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        columns.is_cut_flags,
        np.array([False, True, False, False, True], dtype=np.bool_),
    )

    # Per-variant scalars (LOCAL cut index + partial).
    np.testing.assert_array_equal(
        columns.cut_call_target_index,
        np.array([1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        columns.partial_cut_length,
        np.array([1, 1], dtype=np.int64),
    )

    # Cross-check: each variant's columnar slice equals the scalar walk's
    # per-variant CutoffResult (the documented byte-for-byte equivalence).
    for v, (lo, hi, budget) in enumerate(
        [(0, 3, 4), (3, 5, 6)]
    ):
        scalar = walk_cutoff(
            [int(x) for x in lengths[lo:hi]], context_len=budget
        )
        np.testing.assert_array_equal(
            columns.surviving_token_counts[lo:hi],
            np.array(scalar.surviving_token_counts, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            columns.is_cut_flags[lo:hi],
            np.array(scalar.is_cut_flags, dtype=np.bool_),
        )
        assert int(columns.cut_call_target_index[v]) == (
            scalar.cut_call_target_index
        )
        assert int(columns.partial_cut_length[v]) == scalar.partial_cut_length
