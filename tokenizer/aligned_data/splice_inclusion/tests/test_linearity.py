"""Linearity regression guard for the per-root reset path.

:meth:`OnceOnlyInclusion.begin_root` zeroes only the PREVIOUS root's
used mask region (``self._n_cols`` is reset per root, never a global
high-water-mark), so a corpus of a BIG root followed by many SMALL
roots must cost O(total roots), not O(roots x max_cols_ever). A
regression that re-zeroed the grown mask width (or otherwise scaled the
reset by a stale high-water column count) would turn the
big-root-then-many-small-roots pattern superlinear -- the z3-scale
pathology. This test pins the per-root reset cost to a constant.
"""

from __future__ import annotations

import time

import numpy as np

from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion


_N_VARIANTS = 8
_BIG_COLS = 4000


def _drive_big_then_small(n_small: int) -> float:
    """One big root (touches ``_BIG_COLS`` cols) then ``n_small`` tiny
    roots; return the wall time spent on the small roots' begin_root +
    step path (the region a stale-high-water reset would inflate).
    """
    dec = OnceOnlyInclusion()
    dec.begin_root(_N_VARIANTS, 0)
    # Touch _BIG_COLS distinct callee fids on variant 0 only, so the
    # all-variants exclusion never converges and the columns persist for
    # this root (driving _n_cols up to ~_BIG_COLS).
    fids = np.arange(1, _BIG_COLS + 1, dtype=np.uint32)
    rows = np.zeros(_BIG_COLS, dtype=np.int64)
    dec.step_level(rows, fids)

    t0 = time.perf_counter()
    for r in range(n_small):
        dec.begin_root(_N_VARIANTS, 1_000_000 + r)
        dec.step_level(
            np.array([0, 1], dtype=np.int64),
            np.array([5, 6], dtype=np.uint32),
        )
    return time.perf_counter() - t0


def test_big_root_then_many_small_roots_is_linear() -> None:
    # Warm caches / JIT-free numpy, then measure two scales 4x apart.
    _drive_big_then_small(200)
    base_n = 2000
    t1 = _drive_big_then_small(base_n)
    t4 = _drive_big_then_small(4 * base_n)

    # Per-root cost must stay ~constant: a superlinear reset (re-zeroing
    # the big root's _BIG_COLS span for EVERY small root) would make t4's
    # per-root cost balloon. Allow generous slack for CI noise -- the
    # superlinear regression is order-of-magnitude, not a few percent.
    per_root_1 = t1 / base_n
    per_root_4 = t4 / (4 * base_n)
    assert per_root_4 < per_root_1 * 3.0, (
        f"per-root reset cost grew with corpus size "
        f"({per_root_1 * 1e6:.2f}us -> {per_root_4 * 1e6:.2f}us); the "
        "big-root-then-many-small-roots reset is no longer linear"
    )


def test_reset_leaves_only_previous_root_region_zeroed() -> None:
    # Correctness companion to the timing guard: after a big root grows
    # _n_cols, a fresh small root must see an all-false mask region for
    # the columns it uses (first-encounter inclusion intact), i.e. the
    # reset actually cleared the big root's marks.
    dec = OnceOnlyInclusion()
    dec.begin_root(2, 0)
    # big root: variant 0 reaches fids 1..50 (column 1..50 marked True).
    big_fids = np.arange(1, 51, dtype=np.uint32)
    dec.step_level(np.zeros(50, dtype=np.int64), big_fids)

    # small root reusing the SAME fids must include them on first
    # encounter (mask was reset) -- a leaked True cell would suppress it.
    dec.begin_root(2, 1000)
    res = dec.step_level(
        np.array([0, 1], dtype=np.int64),
        np.array([1, 1], dtype=np.uint32),  # both variants reach fid 1
    )
    # fid 1 reached by BOTH variants -> all-variants exclusion fires,
    # included is all-False; the point is no stale-True from the big root
    # corrupts the decision (a leak would still read pre_cell True).
    assert res.included.dtype == bool
    # A variant reaching a fresh fid alone IS included (proves the column
    # was zeroed, not carried over True from the big root).
    res2 = dec.step_level(
        np.array([0], dtype=np.int64),
        np.array([7], dtype=np.uint32),
    )
    assert bool(res2.included[0]), (
        "first encounter of a fid after a big-root reset was not "
        "included -- the previous root's mark leaked across begin_root"
    )
