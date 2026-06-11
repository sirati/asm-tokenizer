"""Unit tests for sorted_index._sampler -- Reading-A unbiased sampler.

These tests build per-binary :class:`SortedIndexReader` instances from
hand-crafted blobs (via :func:`encode_sorted_index`) and exercise the
sampler at the public surface plus the free function. The Monte-Carlo
test holds Reading-A's contract: per-binary draw frequency at a fixed
target length must match each binary's pool share, verified via a
chi-squared statistic against the standard df=2 / alpha=0.05 critical
value 5.991.

scipy is intentionally not imported (project dev shell does not ship
it); the chi-squared statistic is computed inline from observed +
expected counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    MultiBinarySortedIndexSampler,
    ReductionKind,
    SortedIndexReader,
    encode_sorted_index,
)
from tokenizer.aligned_data.sorted_index._sampler import (
    sample_section_pointers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reader(
    tmp_path: Path, binary_name: str, lengths: np.ndarray,
) -> SortedIndexReader:
    """Write a tiny per-binary index and open a reader on it."""
    path = tmp_path / f"{binary_name}_sorted_max_d003.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )


def _three_binary_pool(tmp_path: Path) -> Dict[str, SortedIndexReader]:
    """3 readers with known counts at target_length=5.

    Counts at L=5 -> {alpha: 5, beta: 3, gamma: 2}; total = 10.
    """
    alpha = _make_reader(
        tmp_path, "alpha", np.array(
            [5, 5, 5, 5, 5, 1, 9], dtype=np.uint32,
        ),
    )
    beta = _make_reader(
        tmp_path, "beta", np.array(
            [5, 5, 5, 8, 4], dtype=np.uint32,
        ),
    )
    gamma = _make_reader(
        tmp_path, "gamma", np.array(
            [5, 5, 12], dtype=np.uint32,
        ),
    )
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


# ---------------------------------------------------------------------------
# Construction + binary_names canonicalisation
# ---------------------------------------------------------------------------


def test_sampler_canonicalises_to_alphabetical(tmp_path: Path) -> None:
    a = _make_reader(tmp_path, "alpha", np.array([1], dtype=np.uint32))
    b = _make_reader(tmp_path, "beta", np.array([1], dtype=np.uint32))
    c = _make_reader(tmp_path, "gamma", np.array([1], dtype=np.uint32))
    # Pass in non-alphabetical insertion order:
    sampler = MultiBinarySortedIndexSampler({"gamma": c, "alpha": a, "beta": b})
    assert sampler.binary_names == ["alpha", "beta", "gamma"]


def test_sampler_count_at_aggregates_pools(tmp_path: Path) -> None:
    sampler = MultiBinarySortedIndexSampler(_three_binary_pool(tmp_path))
    assert sampler.count_at(5) == 10
    assert sampler.count_at(99) == 0


# ---------------------------------------------------------------------------
# sample_section_pointers -- shape + invariants
# ---------------------------------------------------------------------------


def test_sample_section_pointers_returns_typed_pointers(tmp_path: Path) -> None:
    readers = _three_binary_pool(tmp_path)
    rng = np.random.default_rng(0)
    pointers = sample_section_pointers(readers, 5, 4, rng)
    assert len(pointers) == 4
    for p in pointers:
        assert p.binary_name in {"alpha", "beta", "gamma"}
        assert p.section_pointer.arm is SectionKind.MATCHED
        assert isinstance(p.section_pointer.idx, int)


def test_sample_caps_at_total_when_count_exceeds_pool(tmp_path: Path) -> None:
    readers = _three_binary_pool(tmp_path)
    rng = np.random.default_rng(0)
    # total = 10; ask for 50 -> get 10.
    pointers = sample_section_pointers(readers, 5, 50, rng)
    assert len(pointers) == 10


def test_sample_returns_empty_when_pool_is_empty(tmp_path: Path) -> None:
    readers = _three_binary_pool(tmp_path)
    rng = np.random.default_rng(0)
    # target_length=99 has zero entries in every reader.
    assert sample_section_pointers(readers, 99, 4, rng) == []


def test_sample_zero_total_returns_empty(tmp_path: Path) -> None:
    """All readers report count_at == 0 -> []. Distinct from the
    'out-of-range' case above only in that count_at is 0 for at least
    one in-range bucket; here we use a single empty reader."""
    rdr = _make_reader(
        tmp_path, "alpha", np.empty(0, dtype=np.uint32),
    )
    rng = np.random.default_rng(0)
    assert sample_section_pointers({"alpha": rdr}, 0, 4, rng) == []


def test_sample_all_from_one_binary_when_others_empty(tmp_path: Path) -> None:
    """alpha has 3 at L=5; beta and gamma have 0 at L=5."""
    alpha = _make_reader(
        tmp_path, "alpha", np.array([5, 5, 5], dtype=np.uint32),
    )
    beta = _make_reader(
        tmp_path, "beta", np.array([8, 9], dtype=np.uint32),
    )
    gamma = _make_reader(
        tmp_path, "gamma", np.array([1], dtype=np.uint32),
    )
    rng = np.random.default_rng(0)
    pointers = sample_section_pointers(
        {"alpha": alpha, "beta": beta, "gamma": gamma}, 5, 3, rng,
    )
    assert len(pointers) == 3
    assert all(p.binary_name == "alpha" for p in pointers)


def test_sample_indices_unique_within_binary(tmp_path: Path) -> None:
    """Per-binary draws must be without replacement (sample_section_indices
    enforces this); the cross-binary pool inherits that guarantee."""
    readers = _three_binary_pool(tmp_path)
    rng = np.random.default_rng(7)
    pointers = sample_section_pointers(readers, 5, 10, rng)
    per_binary: Dict[str, list] = {}
    for p in pointers:
        per_binary.setdefault(p.binary_name, []).append(p.section_pointer.idx)
    for name, idxs in per_binary.items():
        assert len(idxs) == len(set(idxs)), (
            f"binary {name} has duplicate idxs: {idxs}"
        )


def test_sample_deterministic_with_seeded_rng(tmp_path: Path) -> None:
    readers = _three_binary_pool(tmp_path)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = sample_section_pointers(readers, 5, 6, rng_a)
    b = sample_section_pointers(readers, 5, 6, rng_b)
    assert a == b


def test_sampler_class_delegates_to_free_function(tmp_path: Path) -> None:
    """The class method must produce the same sequence as the free
    function when both are seeded identically (the class is a thin
    delegator)."""
    readers = _three_binary_pool(tmp_path)
    sampler = MultiBinarySortedIndexSampler(readers)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    # Note: the sampler stores readers in alphabetical-canonical order,
    # so the free function called with that same dict ordering must
    # produce an equal output.
    ordered = {name: readers[name] for name in sorted(readers)}
    a = sampler.sample_section_pointers(5, 4, rng_a)
    b = sample_section_pointers(ordered, 5, 4, rng_b)
    assert a == b


# ---------------------------------------------------------------------------
# Reading-A unbiased Monte-Carlo (chi-squared, alpha=0.05, df=2)
# ---------------------------------------------------------------------------


_CHI2_CRIT_DF2_ALPHA_05 = 5.991


def test_monte_carlo_per_binary_draw_matches_pool_share(tmp_path: Path) -> None:
    """5000 single-draw trials must respect Reading-A's per-binary urn
    sizes within the chi-squared df=2 alpha=0.05 critical region.

    Setup:
      - target_length = 5
      - pool: alpha=5, beta=3, gamma=2  (total = 10)
      - expected per-trial probabilities = (0.5, 0.3, 0.2)
      - 5000 single-draw trials -> expected counts (2500, 1500, 1000).
    """
    readers = _three_binary_pool(tmp_path)
    binary_to_idx = {name: i for i, name in enumerate(sorted(readers))}
    expected_probs = np.array([0.5, 0.3, 0.2])
    n_trials = 5000
    counts = np.zeros(3, dtype=np.int64)
    rng = np.random.default_rng(1234)
    for _ in range(n_trials):
        pointers = sample_section_pointers(readers, 5, 1, rng)
        assert len(pointers) == 1
        counts[binary_to_idx[pointers[0].binary_name]] += 1
    expected = expected_probs * n_trials
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < _CHI2_CRIT_DF2_ALPHA_05, (
        f"Reading-A unbiased chi-squared statistic = {chi2:.3f}, "
        f"expected < {_CHI2_CRIT_DF2_ALPHA_05}; counts={counts.tolist()}, "
        f"expected={expected.tolist()}"
    )


def test_monte_carlo_class_path_matches(tmp_path: Path) -> None:
    """Same test as above but through the class API -- ensures the
    canonicalisation step does NOT distort the unbiased property."""
    sampler = MultiBinarySortedIndexSampler(_three_binary_pool(tmp_path))
    binary_to_idx = {name: i for i, name in enumerate(sampler.binary_names)}
    expected_probs = np.array([0.5, 0.3, 0.2])
    n_trials = 5000
    counts = np.zeros(3, dtype=np.int64)
    rng = np.random.default_rng(5678)
    for _ in range(n_trials):
        pointers = sampler.sample_section_pointers(5, 1, rng)
        assert len(pointers) == 1
        counts[binary_to_idx[pointers[0].binary_name]] += 1
    expected = expected_probs * n_trials
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < _CHI2_CRIT_DF2_ALPHA_05, (
        f"class-path chi-squared = {chi2:.3f}, "
        f"expected < {_CHI2_CRIT_DF2_ALPHA_05}; counts={counts.tolist()}"
    )


# ---------------------------------------------------------------------------
# Length-band sampling (sample_section_pointers(band=...))
# ---------------------------------------------------------------------------


_CHI2_CRIT_DF1_ALPHA_05 = 3.841


def _band_two_binary_pool(tmp_path: Path) -> Dict[str, SortedIndexReader]:
    """2 readers with known band-pool sizes for the band ``[3, 4]``.

    Band ``[3, 4]`` pools -> {alpha: 8, beta: 4}; total = 12. Lengths
    outside the band (9 on alpha, 7 on beta) must never be drawn when a
    band is supplied, even though they inflate the exact-bucket
    ``count_at`` totals.
    """
    alpha = _make_reader(
        tmp_path, "alpha", np.array(
            [3, 3, 3, 4, 4, 4, 4, 4, 9], dtype=np.uint32,
        ),
    )
    beta = _make_reader(
        tmp_path, "beta", np.array(
            [3, 3, 4, 4, 7], dtype=np.uint32,
        ),
    )
    return {"alpha": alpha, "beta": beta}


def test_band_pool_aggregates_across_binaries(tmp_path: Path) -> None:
    sampler = MultiBinarySortedIndexSampler(_band_two_binary_pool(tmp_path))
    assert sampler.count_in_band(3, 4) == 12
    # A band entirely outside both readers' ranges is empty.
    assert sampler.count_in_band(100, 200) == 0


def test_band_sample_only_draws_in_band_indices(tmp_path: Path) -> None:
    """Every sampled section index must live in a band ``[3, 4]`` bucket.

    The out-of-band lengths (9 on alpha at section index 8, 7 on beta at
    section index 4) must never appear in the drawn pointers.
    """
    readers = _band_two_binary_pool(tmp_path)
    rng = np.random.default_rng(3)
    # Draw the whole band pool so every in-band index is forced out.
    pointers = sample_section_pointers(
        readers, target_length=999, count=12, rng=rng, band=(3, 4),
    )
    assert len(pointers) == 12
    # alpha's out-of-band section is original index 8; beta's is 4.
    drawn = {(p.binary_name, p.section_pointer.idx) for p in pointers}
    assert ("alpha", 8) not in drawn
    assert ("beta", 4) not in drawn
    # The full in-band pool (alpha idx 0..7, beta idx 0..3) is exactly
    # recovered when count >= pool size.
    expected = {("alpha", i) for i in range(8)} | {
        ("beta", i) for i in range(4)
    }
    assert drawn == expected


def test_band_sample_ignores_target_length(tmp_path: Path) -> None:
    """When a band is given, ``target_length`` is irrelevant.

    The same seeded draw with two wildly different ``target_length``
    values but the same band must produce identical pointer sets.
    """
    readers = _band_two_binary_pool(tmp_path)
    a = sample_section_pointers(
        readers, target_length=0, count=5,
        rng=np.random.default_rng(11), band=(3, 4),
    )
    b = sample_section_pointers(
        readers, target_length=9_999, count=5,
        rng=np.random.default_rng(11), band=(3, 4),
    )
    assert a == b


def test_monte_carlo_band_draw_matches_band_pool_share(
    tmp_path: Path,
) -> None:
    """Per-binary band draw frequency must track each binary's band pool.

    Band ``[3, 4]`` pools: alpha=8, beta=4 (total 12); expected per-draw
    probabilities (2/3, 1/3). 6000 single-draw trials -> expected
    counts (4000, 2000), checked against the chi-squared df=1 alpha=0.05
    critical value 3.841.
    """
    readers = _band_two_binary_pool(tmp_path)
    binary_to_idx = {name: i for i, name in enumerate(sorted(readers))}
    expected_probs = np.array([8.0 / 12.0, 4.0 / 12.0])
    n_trials = 6000
    counts = np.zeros(2, dtype=np.int64)
    rng = np.random.default_rng(2024)
    for _ in range(n_trials):
        pointers = sample_section_pointers(
            readers, target_length=999, count=1, rng=rng, band=(3, 4),
        )
        assert len(pointers) == 1
        counts[binary_to_idx[pointers[0].binary_name]] += 1
    expected = expected_probs * n_trials
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < _CHI2_CRIT_DF1_ALPHA_05, (
        f"band-sampling chi-squared = {chi2:.3f}, "
        f"expected < {_CHI2_CRIT_DF1_ALPHA_05}; counts={counts.tolist()}, "
        f"expected={expected.tolist()}"
    )
