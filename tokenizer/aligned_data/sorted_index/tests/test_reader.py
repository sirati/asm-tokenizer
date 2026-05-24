"""Unit tests for sorted_index._reader (SortedIndexReader + discover_indices).

The reader's correctness is anchored on the wire-format produced by
:func:`encode_sorted_index`. Every test below builds a known-good blob
via the encoder, writes it to a tempfile with a canonical filename,
then exercises the reader's properties + bucket sampling primitives
against an oracle reconstructed from the encoder's input array.

``discover_indices`` is tested with hand-laid filenames against the
canonical grammar plus several non-matching distractors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    discover_indices,
    encode_sorted_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_index(
    dir_path: Path,
    binary_name: str,
    mode_tag: str,
    depth: int,
    lengths: np.ndarray,
) -> Path:
    """Write a canonical sorted-index file and return its path."""
    path = dir_path / f"{binary_name}_sorted_{mode_tag}_d{depth:03d}.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return path


# ---------------------------------------------------------------------------
# SortedIndexReader -- metadata properties
# ---------------------------------------------------------------------------


def test_reader_empty_index(tmp_path: Path) -> None:
    path = _write_index(
        tmp_path, "bin", "max", 3, np.empty(0, dtype=np.uint32),
    )
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    assert rdr.min_length == 0
    assert rdr.max_length == 0
    assert rdr.total_sections() == 0
    assert rdr.count_at(0) == 0
    assert rdr.count_at(5) == 0


def test_reader_min_max_total(tmp_path: Path) -> None:
    lengths = np.array([5, 2, 8, 1, 5, 3], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    assert rdr.min_length == 1
    assert rdr.max_length == 8
    assert rdr.total_sections() == lengths.size


def test_reader_metadata_round_trip(tmp_path: Path) -> None:
    lengths = np.array([5, 2, 8], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "p95", 7, lengths)
    reduction = LengthReduction(ReductionKind.PERCENTILE, 95)
    rdr = SortedIndexReader(path, reduction=reduction, depth=7)
    assert rdr.reduction == reduction
    assert rdr.depth == 7


# ---------------------------------------------------------------------------
# SortedIndexReader -- count_at
# ---------------------------------------------------------------------------


def test_count_at_in_range_and_out_of_range(tmp_path: Path) -> None:
    # min=2, max=10 with a gap at 5..9 (only length 2, 5, 10 populated)
    lengths = np.array([2, 10, 2, 5], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    assert rdr.count_at(2) == 2
    assert rdr.count_at(5) == 1
    assert rdr.count_at(10) == 1
    # Out-of-range below min and above max:
    assert rdr.count_at(1) == 0
    assert rdr.count_at(11) == 0
    # In-range but empty bucket:
    assert rdr.count_at(6) == 0


# ---------------------------------------------------------------------------
# SortedIndexReader -- sample_section_indices
# ---------------------------------------------------------------------------


def test_sample_out_of_range_returns_empty(tmp_path: Path) -> None:
    lengths = np.array([5, 2, 8, 1, 5, 3], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng = np.random.default_rng(0)
    # Below min:
    assert rdr.sample_section_indices(0, 4, rng).size == 0
    # Above max:
    assert rdr.sample_section_indices(99, 4, rng).size == 0


def test_sample_empty_bucket_returns_empty(tmp_path: Path) -> None:
    # Length 6 is inside [2, 10] but has 0 sections in this lengths array.
    lengths = np.array([2, 10, 2, 5], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng = np.random.default_rng(0)
    assert rdr.sample_section_indices(6, 3, rng).size == 0


def test_sample_full_bucket_returned_when_count_at_or_above_bucket_size(
    tmp_path: Path,
) -> None:
    # Lengths: indices 0, 2 have length 7; indices 1, 3 have length 3.
    lengths = np.array([7, 3, 7, 3], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng = np.random.default_rng(0)
    # count == bucket_count
    full = rdr.sample_section_indices(7, 2, rng)
    assert sorted(full.tolist()) == [0, 2]
    # count > bucket_count
    full2 = rdr.sample_section_indices(7, 99, rng)
    assert sorted(full2.tolist()) == [0, 2]
    # Returned array is u32
    assert full.dtype == np.uint32


def test_sample_full_bucket_returns_copy(tmp_path: Path) -> None:
    """Full-bucket path must hand back a fresh array (mutation safe)."""
    lengths = np.array([4, 4, 4], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng = np.random.default_rng(0)
    first = rdr.sample_section_indices(4, 3, rng)
    first[:] = 999
    # Second read must NOT see the mutation.
    second = rdr.sample_section_indices(4, 3, rng)
    assert sorted(second.tolist()) == [0, 1, 2]


def test_sample_partial_deterministic_with_seeded_rng(tmp_path: Path) -> None:
    # Bucket of 5 indices for length 9.
    lengths = np.array([9, 9, 9, 9, 9, 1], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)
    a = rdr.sample_section_indices(9, 3, rng_a)
    b = rdr.sample_section_indices(9, 3, rng_b)
    np.testing.assert_array_equal(a, b)
    # All sampled indices come from the length-9 bucket (original indices 0..4)
    bucket = set(range(5))
    assert set(a.tolist()) <= bucket


def test_sample_partial_without_replacement(tmp_path: Path) -> None:
    lengths = np.array([3, 3, 3, 3, 3], dtype=np.uint32)
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng = np.random.default_rng(0)
    sampled = rdr.sample_section_indices(3, 4, rng)
    assert sampled.size == 4
    assert len(set(sampled.tolist())) == 4   # no duplicates


# ---------------------------------------------------------------------------
# discover_indices
# ---------------------------------------------------------------------------


def test_discover_indices_matches_canonical_grammar(tmp_path: Path) -> None:
    lengths = np.array([3, 5], dtype=np.uint32)
    _write_index(tmp_path, "alpha", "max", 3, lengths)
    _write_index(tmp_path, "alpha", "p95", 3, lengths)
    _write_index(tmp_path, "beta", "max", 3, lengths)
    out = discover_indices(tmp_path, depth=3)
    assert set(out) == {"alpha", "beta"}
    alpha_tags = {r.filename_tag() for r in out["alpha"]}
    assert alpha_tags == {"max", "p95"}
    beta_tags = {r.filename_tag() for r in out["beta"]}
    assert beta_tags == {"max"}


def test_discover_indices_filters_by_depth(tmp_path: Path) -> None:
    lengths = np.array([3, 5], dtype=np.uint32)
    _write_index(tmp_path, "alpha", "max", 3, lengths)
    _write_index(tmp_path, "alpha", "max", 7, lengths)
    out3 = discover_indices(tmp_path, depth=3)
    assert "alpha" in out3 and len(out3["alpha"]) == 1
    out7 = discover_indices(tmp_path, depth=7)
    assert "alpha" in out7 and len(out7["alpha"]) == 1
    out5 = discover_indices(tmp_path, depth=5)
    assert out5 == {}


def test_discover_indices_skips_malformed(tmp_path: Path) -> None:
    lengths = np.array([3, 5], dtype=np.uint32)
    _write_index(tmp_path, "alpha", "max", 3, lengths)
    # Distractors that must NOT be picked up:
    (tmp_path / "alpha_sorted_max_d3.idx").write_bytes(b"")        # depth not zero-padded to 3
    (tmp_path / "alpha_sorted_p9_d003.idx").write_bytes(b"")       # mode digits not zero-padded
    (tmp_path / "alpha_sorted_max_d003.idxx").write_bytes(b"")     # bad suffix
    (tmp_path / "alpha_sorted_med_d003.idx").write_bytes(b"")      # bad mode token
    (tmp_path / "unrelated.idx").write_bytes(b"")                  # not sorted_*
    (tmp_path / "subdir").mkdir()                                  # directory not file
    out = discover_indices(tmp_path, depth=3)
    assert set(out) == {"alpha"}
    assert [r.filename_tag() for r in out["alpha"]] == ["max"]


def test_discover_indices_empty_dir(tmp_path: Path) -> None:
    assert discover_indices(tmp_path, depth=3) == {}


def test_discover_indices_p100_skipped(tmp_path: Path) -> None:
    """``p100`` is canonical-MAX at parse time; filename grammar pins
    two-digit suffixes only, so ``p100`` would not even match the
    grammar. Cross-check that explicitly so the grammar regression is
    visible."""
    (tmp_path / "alpha_sorted_p100_d003.idx").write_bytes(b"")
    out = discover_indices(tmp_path, depth=3)
    assert out == {}


# ---------------------------------------------------------------------------
# Reader byte-fidelity cross-check vs encoder oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths",
    [
        np.array([5, 2, 8, 1, 5, 3], dtype=np.uint32),
        np.array([7, 7, 9, 7, 12, 9], dtype=np.uint32),
        np.array([0, 10, 0, 20, 10, 30], dtype=np.uint32),
    ],
)
def test_reader_bucket_matches_encoder_oracle(
    tmp_path: Path, lengths: np.ndarray,
) -> None:
    """The reader's bucket at length L must equal the encoder's stable
    argsort positions whose lengths equal L."""
    path = _write_index(tmp_path, "bin", "max", 3, lengths)
    rdr = SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
    )
    rng = np.random.default_rng(0)
    unique_lengths = sorted(set(int(x) for x in lengths.tolist()))
    for L in unique_lengths:
        expected = sorted(
            i for i, v in enumerate(lengths.tolist()) if int(v) == L
        )
        # Asking for >= bucket size returns the full bucket in stable
        # argsort order (same as oracle's sorted-by-original-idx because
        # equal-length keys preserve their input order).
        got = rdr.sample_section_indices(L, 999, rng).tolist()
        assert got == expected
