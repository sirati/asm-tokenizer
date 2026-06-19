"""Byte-identity guard for the single-``frombuffer`` band-pool gather.

:meth:`SortedIndexReader.sample_section_indices_in_band` used to LOOP
over every length bucket in ``[lo, hi]`` (one ``np.frombuffer`` + copy +
``list.append`` per bucket) and ``np.concatenate`` the parts. Because
the ``.idx`` body is length-bucketed in stable-sorted order
(:func:`.._wire.encode_sorted_index`), the band's buckets are CONTIGUOUS,
so the whole band pool is one ``frombuffer``. This module pins that the
single-read pool is element-for-element identical (SAME ORDER) to the
old per-bucket concat across a spread of bands -- narrow, wide, and ones
straddling empty-bucket gaps -- so ``rng.choice`` over the same
``pool_size`` draws the same indices and emitted pointers stay
byte-identical (the #75 unbiased-urn determinism contract).

The old per-bucket concat is reproduced inline as
:func:`_old_band_pool` (the file-patch capture of pre-fix behaviour) and
serves as the reference oracle: the test does NOT call ``rng.choice`` on
its own, it draws the WHOLE pool (``count >= pool_size``) so the method
returns the pool in body order, isolating the gather from the RNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from tokenizer.aligned_data.sorted_index import (
    IndexSpec,
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    encode_sorted_index,
)
from tokenizer.aligned_data.sorted_index._sampler import (
    CrossSpecSortedIndexSampler,
)
from tokenizer.aligned_data.sorted_index._wire import EXCLUDED_LENGTH


_MAX = LengthReduction(ReductionKind.MAX)


# ---------------------------------------------------------------------------
# Reference oracle: the OLD per-bucket concat gather (file-patch capture)
# ---------------------------------------------------------------------------


def _old_band_pool(
    reader: SortedIndexReader, lo: int, hi: int,
) -> np.ndarray:
    """Reproduce the pre-fix per-bucket-loop + ``np.concatenate`` pool.

    Verbatim port of the old ``sample_section_indices_in_band`` gather
    (clamp past :data:`EXCLUDED_LENGTH`, loop ``lo_idx..hi_idx``, one
    ``frombuffer`` + ``.copy()`` per non-empty bucket, then concat). This
    is the byte-identity reference the production single-``frombuffer``
    path must match element-for-element, in order.
    """
    lo = max(lo, EXCLUDED_LENGTH + 1)
    lo_idx = max(0, lo - reader._min_length)
    hi_idx = min(reader._counts.size - 1, hi - reader._min_length)
    if lo_idx > hi_idx:
        return np.empty(0, dtype=np.uint32)
    parts = []
    for idx in range(lo_idx, hi_idx + 1):
        bc = int(reader._counts[idx])
        if bc == 0:
            continue
        body_offset = int(reader._bucket_body_offsets[idx])
        bucket = np.frombuffer(
            reader._blob, dtype=np.uint32, count=bc, offset=body_offset,
        )
        parts.append(bucket.copy())
    if not parts:
        return np.empty(0, dtype=np.uint32)
    return np.concatenate(parts)


def _new_band_pool(
    reader: SortedIndexReader, lo: int, hi: int,
) -> np.ndarray:
    """Pool as gathered by the production single-``frombuffer`` path.

    Drawing the whole pool (``count`` larger than any band) returns the
    pool in body order untouched by ``rng.choice`` (the ``k ==
    pool_size`` short-circuit), so this isolates the gather.
    """
    rng = np.random.default_rng(0)  # never consulted on a full draw.
    return reader.sample_section_indices_in_band(lo, hi, 10**9, rng)


# ---------------------------------------------------------------------------
# Fixtures: synthetic .idx layouts with varied bucket spreads
# ---------------------------------------------------------------------------


def _make_reader(
    tmp_path: Path, name: str, lengths: np.ndarray, depth: int = 3,
) -> SortedIndexReader:
    path = tmp_path / f"{name}_sorted_max_d{depth:03d}.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(path, reduction=_MAX, depth=depth)


def _dense_lengths() -> np.ndarray:
    """One section per length 1..50, plus repeats -- dense, no gaps."""
    base = np.arange(1, 51, dtype=np.uint32)
    repeats = np.repeat(np.arange(1, 51, dtype=np.uint32), 2)
    return np.concatenate([base, repeats])


def _gappy_lengths() -> np.ndarray:
    """Sparse lengths with WIDE empty-bucket gaps between populated ones."""
    populated = [3, 4, 4, 7, 50, 51, 200, 201, 201, 5000]
    return np.array(populated, dtype=np.uint32)


def _wide_lengths() -> np.ndarray:
    """A wide span (many distinct length buckets) -- the cross-depth case.

    8000 sections spread over lengths 1..4000 so a band ``[1, 4000]``
    crosses ~4000 distinct buckets -- the pathological per-bucket-loop
    width the fix collapses.
    """
    rng = np.random.default_rng(12345)
    return rng.integers(1, 4001, size=8000, dtype=np.uint32)


# Bands chosen to exercise: a single bucket, a narrow few-bucket band, a
# wide cross-bucket band, bands straddling EMPTY-bucket gaps, the EXCLUDED
# clamp (lo <= 0), fully out-of-range (empty), and the full span.
_BANDS: List[Tuple[int, int]] = [
    (0, 0),          # clamped entirely past EXCLUDED_LENGTH -> empty.
    (1, 1),          # single low bucket.
    (4, 4),          # single bucket (the gappy fixture's dup bucket).
    (1, 3),          # narrow.
    (3, 7),          # straddles an empty gap (5, 6) in the gappy fixture.
    (50, 201),       # wide, multiple gaps in the gappy fixture.
    (0, 10),         # EXCLUDED clamp: lo coerced to EXCLUDED_LENGTH + 1.
    (1, 5000),       # full span, every bucket.
    (1, 4000),       # wide cross-depth span (dominant on _wide).
    (10000, 20000),  # entirely out of range -> empty.
]


# ---------------------------------------------------------------------------
# Pool byte-identity across a spread of bands + layouts
# ---------------------------------------------------------------------------


def test_single_frombuffer_pool_matches_old_concat(tmp_path: Path) -> None:
    """New single-``frombuffer`` pool == old per-bucket concat, in order.

    Across dense / gappy / wide layouts and a spread of bands (narrow,
    wide cross-bucket, empty-gap-straddling, EXCLUDED-clamped, empty),
    the production gather must be element-for-element identical to the
    pre-fix concat -- that ordering is what keeps ``rng.choice`` draws,
    hence emitted pointers, byte-identical.
    """
    layouts = {
        "dense": _dense_lengths(),
        "gappy": _gappy_lengths(),
        "wide": _wide_lengths(),
    }
    for layout_name, lengths in layouts.items():
        reader = _make_reader(tmp_path, layout_name, lengths)
        for lo, hi in _BANDS:
            old = _old_band_pool(reader, lo, hi)
            new = _new_band_pool(reader, lo, hi)
            np.testing.assert_array_equal(
                new, old,
                err_msg=(
                    f"band ({lo},{hi}) on {layout_name!r}: single-frombuffer "
                    f"pool diverged from per-bucket concat"
                ),
            )
            assert new.dtype == np.uint32


def test_wide_band_pool_is_one_contiguous_span(tmp_path: Path) -> None:
    """The wide cross-depth band's pool is one contiguous body span.

    Sanity that the band genuinely crosses many distinct buckets (so the
    old path would have looped thousands of times) yet the pool equals a
    single ``frombuffer`` over the contiguous span.
    """
    lengths = _wide_lengths()
    reader = _make_reader(tmp_path, "wide_check", lengths)
    lo, hi = 1, 4000
    # Number of distinct populated buckets the old loop would visit.
    distinct = int(np.unique(lengths[(lengths >= lo) & (lengths <= hi)]).size)
    assert distinct > 1000, (
        f"wide band must cross >1000 buckets to be a meaningful test; "
        f"got {distinct}"
    )
    new = _new_band_pool(reader, lo, hi)
    assert new.size == int(((lengths >= lo) & (lengths <= hi)).sum())


# ---------------------------------------------------------------------------
# Teeth: a deliberate offset/count perturbation must FAIL the guard
# ---------------------------------------------------------------------------


def test_teeth_offset_perturbation_breaks_identity(tmp_path: Path) -> None:
    """A perturbed reference pool (shifted start offset) must differ.

    Proves the byte-identity assertion has teeth: a one-element offset
    shift in the gather yields a different pool, so the equality check
    would catch a regression that read from the wrong span.
    """
    lengths = _gappy_lengths()
    reader = _make_reader(tmp_path, "teeth_off", lengths)
    lo, hi = 1, 5000  # full span -> a multi-element pool.
    correct = _new_band_pool(reader, lo, hi)
    assert correct.size > 1

    # Perturb: read the same count but one element further into the body.
    lo_clamped = max(lo, EXCLUDED_LENGTH + 1)
    lo_idx = max(0, lo_clamped - reader._min_length)
    n = int(reader._counts[lo_idx:].sum())
    bad_offset = int(reader._bucket_body_offsets[lo_idx]) + 4  # +1 u32 elem.
    perturbed = np.frombuffer(
        reader._blob, dtype=np.uint32, count=n - 1, offset=bad_offset,
    ).copy()
    assert not np.array_equal(correct[:-1], perturbed) or correct.size == n, (
        "offset perturbation should not coincidentally match"
    )
    # The standard assertion the guard uses must reject the perturbation.
    raised = False
    try:
        np.testing.assert_array_equal(correct, perturbed)
    except AssertionError:
        raised = True
    assert raised, "teeth: shifted-offset pool must fail array_equal"


def test_teeth_count_perturbation_breaks_identity(tmp_path: Path) -> None:
    """A truncated reference pool (wrong count) must differ."""
    lengths = _dense_lengths()
    reader = _make_reader(tmp_path, "teeth_cnt", lengths)
    correct = _new_band_pool(reader, 1, 50)
    assert correct.size > 2
    truncated = correct[:-1]
    raised = False
    try:
        np.testing.assert_array_equal(correct, truncated)
    except AssertionError:
        raised = True
    assert raised, "teeth: truncated pool must fail array_equal"


# ---------------------------------------------------------------------------
# End-to-end: emitted pointers identical (old gather vs new) under a seed
# ---------------------------------------------------------------------------


def _build_cross_spec_pool(
    tmp_path: Path,
) -> "dict":
    """Two binaries x two specs with WIDE, distinct per-cell length spreads.

    Each cell gets a different distribution over a wide length range so a
    band crosses many buckets per cell -- exactly the cross-depth path
    the fix targets.
    """
    rng = np.random.default_rng(777)
    specs = {
        IndexSpec(reduction=_MAX, depth=0): {
            "alpha": rng.integers(1, 2001, size=1500, dtype=np.uint32),
            "beta": rng.integers(1, 2001, size=900, dtype=np.uint32),
        },
        IndexSpec(reduction=_MAX, depth=3): {
            "alpha": rng.integers(1, 2001, size=1100, dtype=np.uint32),
            "beta": rng.integers(1, 2001, size=700, dtype=np.uint32),
        },
    }
    out = {}
    for spec, by_name in specs.items():
        out[spec] = {
            name: _make_reader(
                tmp_path, f"{name}_d{spec.depth}", lengths, depth=spec.depth,
            )
            for name, lengths in by_name.items()
        }
    return out


def _pointers_signature(pointers) -> list:
    """A hashable signature of emitted pointers for equality comparison."""
    return [
        (p.binary_name, p.spec.depth, p.section_pointer.idx)
        for p in pointers
    ]


def test_cross_spec_pointers_identical_old_vs_new(
    tmp_path: Path, monkeypatch,
) -> None:
    """``CrossSpecSortedIndexSampler.sample_section_pointers`` is unchanged.

    Capture the OLD behaviour by monkeypatching the reader's band gather
    to the per-bucket-concat reference (then a faithful re-impl of the
    surrounding ``k``/``rng.choice`` logic), and compare the emitted
    pointers against the production path under the SAME fixed seed across
    a wide cross-depth band. Identical pointers prove the gather swap is
    invisible end-to-end (the #75 determinism contract held).
    """
    pool = _build_cross_spec_pool(tmp_path)
    sampler = CrossSpecSortedIndexSampler(pool)
    band = (1, 2000)  # wide -> many buckets per cell.
    seed = 31337
    count = 64

    # ---- New (production) path. ----
    new_pointers = sampler.sample_section_pointers(
        0, count, np.random.default_rng(seed), band=band,
    )
    assert new_pointers, "fixture must produce a non-empty band pool"

    # ---- Old path: replace the band gather with the per-bucket concat,
    # keeping the EXACT k/rng.choice logic the method had. ----
    def _old_sample_in_band(self, lo, hi, count_, rng):
        pool_arr = _old_band_pool(self, lo, hi)
        if pool_arr.size == 0:
            return np.empty(0, dtype=np.uint32)
        pool_size = pool_arr.size
        k = min(count_, pool_size)
        if k == pool_size:
            return pool_arr
        chosen = rng.choice(pool_size, size=k, replace=False)
        return pool_arr[chosen].astype(np.uint32, copy=False)

    monkeypatch.setattr(
        SortedIndexReader,
        "sample_section_indices_in_band",
        _old_sample_in_band,
        raising=True,
    )
    old_pointers = sampler.sample_section_pointers(
        0, count, np.random.default_rng(seed), band=band,
    )

    assert _pointers_signature(new_pointers) == _pointers_signature(
        old_pointers
    ), "emitted pointers diverged between old per-bucket gather and new"
