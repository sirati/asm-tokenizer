"""Equivalence: segmented reduction methods vs their scalar sources.

``reduce_segmented`` / ``reduce_grouped_segmented`` /
``DuplicateHandling.reduce_segmented`` / ``VariantGate.passes_batch``
must agree elementwise with the scalar ``reduce`` / ``reduce_groups``
/ ``reduce_section`` / ``passes`` on randomized segment layouts
(including empty segments, singleton segments, all-duplicate
segments).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.sorted_index._dedup import (
    DEDUP_BY_DATA_POINTER,
    PLAIN,
)
from tokenizer.aligned_data.sorted_index._gating import VariantGate
from tokenizer.aligned_data.sorted_index._types import (
    LengthReduction,
    ReductionKind,
)


REDUCTIONS = [
    LengthReduction(kind=ReductionKind.MAX),
    LengthReduction(kind=ReductionKind.PERCENTILE, percentile=1),
    LengthReduction(kind=ReductionKind.PERCENTILE, percentile=50),
    LengthReduction(kind=ReductionKind.PERCENTILE, percentile=75),
    LengthReduction(kind=ReductionKind.PERCENTILE, percentile=99),
]


def _random_layout(rng: np.random.Generator):
    counts = rng.integers(0, 7, size=40)
    counts[rng.integers(0, 40, size=5)] = 0  # force empty segments
    total = int(counts.sum())
    seg_offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=seg_offsets[1:])
    lengths = rng.integers(0, 5000, size=total).astype(np.int64)
    # Few distinct pointers per section so duplicate-groups are common.
    pointers = rng.integers(0, 3, size=total).astype(np.uint32) * 16 + 32
    return counts, seg_offsets, lengths, pointers


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=lambda r: r.filename_tag())
def test_reduce_segmented_matches_scalar(seed: int, reduction) -> None:
    rng = np.random.default_rng(seed)
    counts, seg_offsets, lengths, _ = _random_layout(rng)
    got = reduction.reduce_segmented(lengths, seg_offsets)
    for s in range(counts.size):
        seg = lengths[seg_offsets[s] : seg_offsets[s + 1]]
        assert got[s] == reduction.reduce(seg), f"segment {s}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("reduction", REDUCTIONS, ids=lambda r: r.filename_tag())
def test_reduce_grouped_segmented_matches_scalar(seed: int, reduction) -> None:
    rng = np.random.default_rng(seed)
    counts, seg_offsets, lengths, pointers = _random_layout(rng)
    got = reduction.reduce_grouped_segmented(lengths, pointers, seg_offsets)
    for s in range(counts.size):
        lo, hi = seg_offsets[s], seg_offsets[s + 1]
        expected = DEDUP_BY_DATA_POINTER.reduce_section(
            reduction, lengths=lengths[lo:hi], data_pointers=pointers[lo:hi]
        )
        assert got[s] == expected, f"segment {s}"


@pytest.mark.parametrize("handling", [PLAIN, DEDUP_BY_DATA_POINTER])
def test_duplicate_handling_segmented_dispatch(handling) -> None:
    rng = np.random.default_rng(9)
    counts, seg_offsets, lengths, pointers = _random_layout(rng)
    reduction = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=75)
    got = handling.reduce_segmented(
        reduction,
        lengths=lengths,
        data_pointers=pointers,
        seg_offsets=seg_offsets,
    )
    for s in range(counts.size):
        lo, hi = seg_offsets[s], seg_offsets[s + 1]
        expected = handling.reduce_section(
            reduction, lengths=lengths[lo:hi], data_pointers=pointers[lo:hi]
        )
        assert got[s] == expected, f"segment {s}"


def test_passes_batch_matches_scalar() -> None:
    gate = VariantGate(min_variants=3, min_variants_unique=2)
    n_total = np.array([0, 2, 3, 5, 3])
    n_unique = np.array([0, 2, 1, 2, 3])
    got = gate.passes_batch(n_total=n_total, n_unique=n_unique)
    expected = [
        gate.passes(n_total=int(t), n_unique=int(u))
        for t, u in zip(n_total, n_unique)
    ]
    assert got.tolist() == expected
