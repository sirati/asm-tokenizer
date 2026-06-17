"""Unit tests for :class:`CrossSpecSortedIndexSampler`.

The cross-(binary x spec) sampler widens the per-binary urn into one
cell per ``(binary, spec)`` so a single without-replacement draw is
unbiased across the binary AND the depth axis at once. These tests pin:

* per-cell unbiasedness (empirical per-cell proportions track each
  cell's ``count_in_band`` share, via an inline chi-squared statistic --
  scipy is intentionally absent from the dev shell);
* every cell is reachable;
* determinism under a seeded RNG (incl the ``spec`` tags);
* the spec tag stamped on every emitted pointer matches the cell.

Readers are built from hand-crafted blobs (:func:`encode_sorted_index`).
A sorted-index file's CONTENT is depth-invariant (depth lives only in
the filename), so distinct per-spec length distributions are used to
make the cells distinguishable -- exactly the cell-resolution the urn
must keep unbiased.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
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


_MAX = LengthReduction(ReductionKind.MAX)
_SPEC_D0 = IndexSpec(reduction=_MAX, depth=0)
_SPEC_D3 = IndexSpec(reduction=_MAX, depth=3)


def _make_reader(
    tmp_path: Path, binary_name: str, depth: int, lengths: np.ndarray,
) -> SortedIndexReader:
    """Write a tiny per-binary index at ``depth`` and open a reader on it."""
    path = tmp_path / f"{binary_name}_sorted_max_d{depth:03d}.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(path, reduction=_MAX, depth=depth)


def _two_binary_two_spec_pool(
    tmp_path: Path,
) -> Dict[IndexSpec, Dict[str, SortedIndexReader]]:
    """4 cells with known counts at L=5.

    Cell counts at L=5:
      (alpha, d0)=4, (alpha, d3)=2, (beta, d0)=3, (beta, d3)=1; total=10.
    A length outside L=5 (9) pads each reader so the bucket layout is
    non-trivial.
    """
    return {
        _SPEC_D0: {
            "alpha": _make_reader(
                tmp_path, "alpha", 0,
                np.array([5, 5, 5, 5, 9], dtype=np.uint32),
            ),
            "beta": _make_reader(
                tmp_path, "beta", 0,
                np.array([5, 5, 5, 9], dtype=np.uint32),
            ),
        },
        _SPEC_D3: {
            "alpha": _make_reader(
                tmp_path, "alpha", 3,
                np.array([5, 5, 9], dtype=np.uint32),
            ),
            "beta": _make_reader(
                tmp_path, "beta", 3,
                np.array([5, 9], dtype=np.uint32),
            ),
        },
    }


# ---------------------------------------------------------------------------
# Construction + canonical order
# ---------------------------------------------------------------------------


def test_binary_names_and_specs_are_canonical(tmp_path: Path) -> None:
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    assert sampler.binary_names == ["alpha", "beta"]
    # Specs in sort_key order (d0 before d3).
    assert sampler.specs == [_SPEC_D0, _SPEC_D3]


def test_count_aggregates_over_all_cells(tmp_path: Path) -> None:
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    # 4 + 2 + 3 + 1 = 10.
    assert sampler.count_at(5) == 10
    assert sampler.count_at(99) == 0


# ---------------------------------------------------------------------------
# Shape + spec-tag invariants
# ---------------------------------------------------------------------------


def test_pointers_carry_their_spec_tag(tmp_path: Path) -> None:
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    rng = np.random.default_rng(0)
    # Draw the whole pool so every cell is forced out.
    pointers = sampler.sample_section_pointers(5, 10, rng)
    assert len(pointers) == 10
    for p in pointers:
        assert p.spec in {_SPEC_D0, _SPEC_D3}
        assert p.binary_name in {"alpha", "beta"}
        assert p.section_pointer.arm is SectionKind.MATCHED


def test_every_cell_reachable(tmp_path: Path) -> None:
    """Draining the pool surfaces all four ``(binary, spec)`` cells."""
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    rng = np.random.default_rng(1)
    pointers = sampler.sample_section_pointers(5, 10, rng)
    cells = {(p.binary_name, p.spec) for p in pointers}
    assert cells == {
        ("alpha", _SPEC_D0),
        ("alpha", _SPEC_D3),
        ("beta", _SPEC_D0),
        ("beta", _SPEC_D3),
    }


def test_caps_at_total_when_count_exceeds_pool(tmp_path: Path) -> None:
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    rng = np.random.default_rng(0)
    pointers = sampler.sample_section_pointers(5, 50, rng)
    assert len(pointers) == 10


def test_empty_pool_returns_empty(tmp_path: Path) -> None:
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    rng = np.random.default_rng(0)
    assert sampler.sample_section_pointers(99, 5, rng) == []


def test_deterministic_with_seeded_rng(tmp_path: Path) -> None:
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    a = sampler.sample_section_pointers(5, 7, np.random.default_rng(42))
    b = sampler.sample_section_pointers(5, 7, np.random.default_rng(42))
    assert a == b


# ---------------------------------------------------------------------------
# Per-cell unbiasedness (chi-squared, df=3, alpha=0.05 -> 7.815)
# ---------------------------------------------------------------------------


_CHI2_CRIT_DF3_ALPHA_05 = 7.815


def test_monte_carlo_per_cell_draw_matches_pool_share(tmp_path: Path) -> None:
    """Single-draw trials must respect each ``(binary, spec)`` cell's share.

    Cells at L=5: (alpha,d0)=4, (alpha,d3)=2, (beta,d0)=3, (beta,d3)=1;
    total 10. Expected probabilities (0.4, 0.2, 0.3, 0.1). 8000 trials,
    checked against the chi-squared df=3 alpha=0.05 critical value 7.815.
    """
    sampler = CrossSpecSortedIndexSampler(_two_binary_two_spec_pool(tmp_path))
    cell_order = [
        ("alpha", _SPEC_D0),
        ("alpha", _SPEC_D3),
        ("beta", _SPEC_D0),
        ("beta", _SPEC_D3),
    ]
    cell_to_idx = {cell: i for i, cell in enumerate(cell_order)}
    expected_probs = np.array([0.4, 0.2, 0.3, 0.1])
    n_trials = 8000
    counts = np.zeros(4, dtype=np.int64)
    rng = np.random.default_rng(2025)
    for _ in range(n_trials):
        pointers = sampler.sample_section_pointers(5, 1, rng)
        assert len(pointers) == 1
        p = pointers[0]
        counts[cell_to_idx[(p.binary_name, p.spec)]] += 1
    expected = expected_probs * n_trials
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < _CHI2_CRIT_DF3_ALPHA_05, (
        f"cross-spec unbiased chi-squared = {chi2:.3f}, "
        f"expected < {_CHI2_CRIT_DF3_ALPHA_05}; counts={counts.tolist()}, "
        f"expected={expected.tolist()}"
    )
