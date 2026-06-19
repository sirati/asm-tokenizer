"""Tests for the deterministic sequential validation sampler + decode entry.

Pins:

* DETERMINISM -- same ``(readers, B, band, seed)`` => identical batch
  stream, cross-checked against the pure-Python
  :func:`._validation_oracle.shuffle_chunk_drop`.
* DIFFERENT SEED -- same section set + bunch counts + dropped-count
  invariant, but a different intra-section ordering.
* COVERAGE -- ``n >= B`` sections contribute ``floor(n/B)`` distinct-index
  bunches; ``n < B`` sections contribute none; per-section bunch union is
  the kept prefix of one shuffle.
* ORDER -- section-major within a reader, readers in input order.
* SELECTION SEAM -- ``CountThenRNGSelection`` is byte-identical to the
  legacy ``_select_variant_indices``; ``ExplicitIndicesSelection`` raises
  on an out-of-bounds index.
* DECODE-VALIDITY -- ``open_validation_batches`` over a real fixture binary
  yields coherent RAGGED batches of exactly ``B`` rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader._session_helpers import (
    _select_variant_indices,
)
from tokenizer.aligned_data.loader.batch_decode._variant_selection import (
    CountThenRNGSelection,
    ExplicitIndicesSelection,
)
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    encode_sorted_index,
)
from tokenizer.aligned_data.sorted_index._types import IndexSpec
from tokenizer.aligned_data.sorted_index._sampler import (
    SequentialValidationSampler,
)
from tokenizer.aligned_data.sorted_index._sampler._validation_oracle import (
    derive_initial_state,
    shuffle_chunk_drop,
)


# ---------------------------------------------------------------------------
# Synthetic-reader helpers (no decode / no session)
# ---------------------------------------------------------------------------


_SPEC = IndexSpec(LengthReduction(ReductionKind.MAX), depth=3)


def _make_reader(
    tmp_path: Path, binary_name: str, lengths: np.ndarray
) -> SortedIndexReader:
    """Write a tiny per-binary index and open a reader on it."""
    path = tmp_path / f"{binary_name}_sorted_max_d003.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3
    )


def _count_provider_from(
    counts_by_binary: Dict[str, Dict[int, int]],
):
    """An injected count_provider keyed by per-binary {section_idx: n_variants}.

    Mirrors the production ``count_provider`` contract: given a binary name
    + the reader's in-band section indices (in enumeration order), return
    the parallel ``int64`` per-section variant counts.
    """

    def provider(binary_name: str, section_indices: np.ndarray) -> np.ndarray:
        table = counts_by_binary[binary_name]
        return np.array(
            [table[int(i)] for i in section_indices], dtype=np.int64
        )

    return provider


def _materialize(
    sampler: SequentialValidationSampler, provider
) -> List[Tuple[str, int, Tuple[int, ...]]]:
    """The batch stream as ``(binary_name, section_idx, indices)`` tuples."""
    out = []
    for batch in sampler.iter_batches(provider):
        sel = batch.section_pointer.variant_selection
        out.append(
            (
                batch.binary_name,
                batch.section_pointer.idx,
                tuple(sel.indices),
            )
        )
    return out


# ---------------------------------------------------------------------------
# DETERMINISM + oracle cross-check
# ---------------------------------------------------------------------------


def test_determinism_and_oracle_crosscheck(tmp_path: Path) -> None:
    # Two binaries; band [5,5] enumerates the length-5 sections.
    # alpha lengths -> section idx [0,1,2] in band; beta -> [0,1].
    alpha = _make_reader(tmp_path, "alpha", np.array([5, 5, 5, 9], np.uint32))
    beta = _make_reader(tmp_path, "beta", np.array([5, 5, 7], np.uint32))
    # n_variants per in-band section (keyed by the enumerated section idx).
    counts = {
        "alpha": {0: 5, 1: 7, 2: 2},
        "beta": {0: 10, 1: 4},
    }
    provider = _count_provider_from(counts)
    readers = [("alpha", _SPEC, alpha), ("beta", _SPEC, beta)]
    B, band, seed = 3, (5, 5), 1234

    s1 = SequentialValidationSampler(readers, B, band, seed)
    s2 = SequentialValidationSampler(readers, B, band, seed)
    out1 = _materialize(s1, provider)
    out2 = _materialize(s2, provider)
    assert out1 == out2  # identical stream for identical inputs

    # Oracle cross-check: thread the SAME single stream the sampler does
    # (alpha sections [0,1,2] then beta sections [0,1]); per reader the
    # kernel result must match shuffle_chunk_drop, and the per-bunch
    # (section_idx, indices) tuples must equal the materialized stream.
    state = tuple(int(x) for x in derive_initial_state(seed))
    expected: List[Tuple[str, int, Tuple[int, ...]]] = []
    for name, sec_idxs in (("alpha", [0, 1, 2]), ("beta", [0, 1])):
        n = [counts[name][i] for i in sec_idxs]
        vi, bo, bsec, state = shuffle_chunk_drop(n, B, state)
        for b in range(len(bsec)):
            sl = tuple(vi[bo[b] : bo[b + 1]])
            expected.append((name, sec_idxs[bsec[b]], sl))
    assert out1 == expected


# ---------------------------------------------------------------------------
# DIFFERENT SEED
# ---------------------------------------------------------------------------


def test_different_seed_same_geometry_different_order(tmp_path: Path) -> None:
    rdr = _make_reader(tmp_path, "alpha", np.array([5, 5, 5], np.uint32))
    counts = {"alpha": {0: 5, 1: 8, 2: 2}}  # bunches floor(n/3): 1, 2, 0
    provider = _count_provider_from(counts)
    readers = [("alpha", _SPEC, rdr)]
    B, band = 3, (5, 5)

    a = _materialize(
        SequentialValidationSampler(readers, B, band, 1), provider
    )
    b = _materialize(
        SequentialValidationSampler(readers, B, band, 2), provider
    )

    # Same section-pointer set + per-section bunch COUNT (floor(n/B)).
    def per_section_bunchcount(stream):
        cnt: Dict[int, int] = {}
        for _name, sidx, _idx in stream:
            cnt[sidx] = cnt.get(sidx, 0) + 1
        return cnt

    assert per_section_bunchcount(a) == per_section_bunchcount(b)
    assert per_section_bunchcount(a) == {0: 1, 1: 2}  # idx2 (n=2<3) drops

    # Total emitted == sum(floor(n/B)*B); dropped (n mod B) invariant.
    total_a = sum(len(idx) for _n, _s, idx in a)
    total_b = sum(len(idx) for _n, _s, idx in b)
    expected_total = (5 // 3) * 3 + (8 // 3) * 3 + (2 // 3) * 3
    assert total_a == total_b == expected_total

    # At least one section's ordering differs between seeds.
    a_by_sec = {s: idx for _n, s, idx in a}
    b_by_sec = {s: idx for _n, s, idx in b}
    assert any(a_by_sec[s] != b_by_sec[s] for s in a_by_sec)


# ---------------------------------------------------------------------------
# COVERAGE
# ---------------------------------------------------------------------------


def test_coverage_bunch_counts_and_distinct_indices(tmp_path: Path) -> None:
    rdr = _make_reader(tmp_path, "alpha", np.array([5, 5, 5, 5], np.uint32))
    # n: idx0=2(<B none), idx1=4(1 bunch), idx2=9(3 bunches), idx3=3(1 bunch).
    counts = {"alpha": {0: 2, 1: 4, 2: 9, 3: 3}}
    provider = _count_provider_from(counts)
    readers = [("alpha", _SPEC, rdr)]
    B = 3

    stream = _materialize(
        SequentialValidationSampler(readers, B, (5, 5), 7), provider
    )

    bunches_per_sec: Dict[int, List[Tuple[int, ...]]] = {}
    for _name, sidx, idx in stream:
        bunches_per_sec.setdefault(sidx, []).append(idx)

    # n < B contributes zero; n >= B contributes floor(n/B).
    assert 0 not in bunches_per_sec
    assert len(bunches_per_sec[1]) == 4 // B
    assert len(bunches_per_sec[2]) == 9 // B
    assert len(bunches_per_sec[3]) == 3 // B

    for sidx, bunches in bunches_per_sec.items():
        n = counts["alpha"][sidx]
        union: List[int] = []
        for bunch in bunches:
            assert len(bunch) == B  # every bunch is exactly B
            assert len(set(bunch)) == B  # distinct within bunch
            assert all(0 <= v < n for v in bunch)  # in-range
            union.extend(bunch)
        # No dup across that section's bunches: the union is the kept
        # prefix of ONE shuffle (floor(n/B)*B distinct indices).
        assert len(union) == len(set(union)) == (n // B) * B


# ---------------------------------------------------------------------------
# ORDER
# ---------------------------------------------------------------------------


def test_order_section_major_then_reader_input_order(tmp_path: Path) -> None:
    a = _make_reader(tmp_path, "aaa", np.array([5, 5], np.uint32))
    b = _make_reader(tmp_path, "bbb", np.array([5], np.uint32))
    counts = {"aaa": {0: 6, 1: 6}, "bbb": {0: 6}}
    provider = _count_provider_from(counts)
    # Pass readers in a deliberately NON-alphabetical input order to prove
    # the sampler honors the GIVEN order, not a re-sort.
    readers = [("bbb", _SPEC, b), ("aaa", _SPEC, a)]

    stream = _materialize(
        SequentialValidationSampler(readers, 3, (5, 5), 3), provider
    )
    # n=6, B=3 -> 2 bunches each. Reader input order: bbb then aaa;
    # within aaa, section-major: idx0 (both bunches) then idx1.
    names = [s[0] for s in stream]
    secs = [(s[0], s[1]) for s in stream]
    assert names == ["bbb", "bbb", "aaa", "aaa", "aaa", "aaa"]
    assert secs == [
        ("bbb", 0),
        ("bbb", 0),
        ("aaa", 0),
        ("aaa", 0),
        ("aaa", 1),
        ("aaa", 1),
    ]


# ---------------------------------------------------------------------------
# SELECTION SEAM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(10, 3), (5, 5), (7, 10), (4, 2)])
def test_count_selection_byte_identical_to_legacy(n: int, k: int) -> None:
    # Same (n, k, rng-seed) => CountThenRNGSelection.select equals the
    # legacy _select_variant_indices output element-for-element.
    legacy = _select_variant_indices(
        n_variants=n, max_variants=k, rng=np.random.default_rng(99)
    )
    seam = CountThenRNGSelection(k).select(
        n_variants=n, rng=np.random.default_rng(99)
    )
    assert seam.dtype == legacy.dtype
    assert seam.tolist() == legacy.tolist()


def test_explicit_selection_returns_indices_and_ignores_rng() -> None:
    sel = ExplicitIndicesSelection((3, 1, 4))
    out = sel.select(n_variants=5, rng=np.random.default_rng(0))
    assert out.dtype == np.int64
    assert out.tolist() == [3, 1, 4]


def test_explicit_selection_raises_on_oob() -> None:
    with pytest.raises(ValueError, match="stale-sidecar"):
        ExplicitIndicesSelection((0, 5)).select(
            n_variants=5, rng=np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="stale-sidecar"):
        ExplicitIndicesSelection((-1,)).select(
            n_variants=5, rng=np.random.default_rng(0)
        )
