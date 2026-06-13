"""Unit tests for the shared once-only inclusion decider.

Single concern: pin :class:`OnceOnlyInclusion`'s inclusion + survival
contract directly (independent of either consumer) -- once-only dedup,
columnwise-ALL exclusion, FLAG-A (single-variant), FLAG-B (late
convergence), and buffer reuse across many roots through ONE instance.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion


def _step(decider, rows, fids):
    return decider.step_level(
        np.asarray(rows, dtype=np.int64), np.asarray(fids, dtype=np.uint32)
    )


def test_root_seeded_at_column_zero_blocks_self_recursion() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(2, root_function_id=100)
    # Both variants call back to the root function -> already at col 0.
    res = _step(d, [0, 1], [100, 100])
    assert res.included.tolist() == [False, False]
    assert res.survivor_pairs.size == 0


def test_some_not_all_included_all_excluded() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(3, root_function_id=1)
    # fid 200 reached by v0,v1 (not v2) -> some-not-all -> included.
    # fid 300 reached by every variant -> excluded + pruned.
    res = _step(d, [0, 1, 2, 0, 1, 2], [200, 200, 999, 300, 300, 300])
    inc = res.included.tolist()
    # pairs: (v0,200) (v1,200) (v2,999) (v0,300) (v1,300) (v2,300)
    assert inc == [True, True, True, False, False, False]


def test_once_only_dedup_within_level() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(2, root_function_id=1)
    # v0 reaches fid 200 via two call slots in ONE level -> included once.
    res = _step(d, [0, 0, 1], [200, 200, 300])
    assert res.included.tolist() == [True, False, True]


def test_once_only_dedup_across_levels() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(2, root_function_id=1)
    r1 = _step(d, [0, 1], [200, 300])
    assert r1.included.tolist() == [True, True]
    # v0 reaches 200 again at level 2 -> repeat, not included.
    r2 = _step(d, [0, 1], [200, 400])
    assert r2.included.tolist() == [False, True]


def test_flag_a_single_variant_excludes_everything() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(1, root_function_id=1)
    # One row -> columnwise ALL is trivially True for every callee.
    res = _step(d, [0, 0], [200, 300])
    assert res.included.tolist() == [False, False]
    assert res.survivor_pairs.size == 0


def test_flag_b_late_convergence_excludes_late_variants() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(2, root_function_id=1)
    # Level 1: only v0 reaches F (200) -> included, expands.
    r1 = _step(d, [0], [200])
    assert r1.included.tolist() == [True]
    # Level 2: v1 now reaches F too -> column all-True -> v1 does NOT
    # include F (late convergence), F prunes.
    r2 = _step(d, [1], [200])
    assert r2.included.tolist() == [False]
    assert r2.survivor_pairs.size == 0


def test_empty_level_returns_empty() -> None:
    d = OnceOnlyInclusion()
    d.begin_root(2, root_function_id=1)
    res = _step(d, [], [])
    assert res.included.size == 0
    assert res.survivor_pairs.size == 0


def test_buffer_reuse_across_many_roots() -> None:
    """ONE instance drives many roots; begin_root clears in place.

    Each root's result must be independent of prior roots -- a stale
    hashmap entry or un-zeroed mask cell would leak across roots. Drives
    100 roots of growing variant/column counts through one instance and
    re-checks a known-answer root at the end.
    """
    d = OnceOnlyInclusion()
    for k in range(100):
        n_var = 2 + (k % 4)
        d.begin_root(n_var, root_function_id=1000 + k)
        # Every variant calls a distinct fid -> none all-shared.
        rows = list(range(n_var))
        fids = [2000 + k * 10 + v for v in range(n_var)]
        res = _step(d, rows, fids)
        assert res.included.tolist() == [True] * n_var

    # Known-answer root after heavy reuse: 2 variants, shared callee
    # excluded, distinct callee included.
    d.begin_root(2, root_function_id=1)
    res = _step(d, [0, 1, 0], [500, 500, 600])
    assert res.included.tolist() == [False, False, True]


def test_mask_grows_without_losing_marks() -> None:
    """A root wider than the initial mask must still dedup correctly."""
    d = OnceOnlyInclusion(initial_cols=2)
    d.begin_root(5, root_function_id=1)
    # 5 variants, 10 distinct callees -> forces column growth past 2.
    rows = list(range(5)) * 2
    fids = list(range(700, 705)) + list(range(710, 715))
    res = _step(d, rows, fids)
    assert res.included.tolist() == [True] * 10
    # Re-encounter the first five at the next level -> all repeats.
    res2 = _step(d, list(range(5)), list(range(700, 705)))
    assert res2.included.tolist() == [False] * 5
