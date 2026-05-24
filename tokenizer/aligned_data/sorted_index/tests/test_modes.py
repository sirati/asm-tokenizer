"""Tests for sorted_index._modes.parse_reduction (plan ALG-2)."""

from __future__ import annotations

import pytest

from tokenizer.aligned_data.sorted_index._modes import parse_reduction
from tokenizer.aligned_data.sorted_index._types import (
    LengthReduction,
    ReductionKind,
)


# ---------------------------------------------------------------------------
# Accepted forms
# ---------------------------------------------------------------------------


def test_parse_max() -> None:
    assert parse_reduction("max") == LengthReduction(ReductionKind.MAX)


def test_parse_p100_canonicalises_to_max() -> None:
    # plan ALG-2: "p100" collapses to MAX at parse time.
    assert parse_reduction("p100") == LengthReduction(ReductionKind.MAX)


@pytest.mark.parametrize("n", [1, 5, 50, 95, 99])
def test_parse_percentile_in_range(n: int) -> None:
    assert parse_reduction(f"p{n}") == LengthReduction(
        ReductionKind.PERCENTILE, n
    )


def test_parse_p01_is_percentile_one() -> None:
    # "p01" is not the canonical filename form (filename_tag emits "p01"
    # for percentile=1) but parse_reduction accepts it as a numeric form
    # because int("01") == 1 and the suffix is all digits.
    assert parse_reduction("p01") == LengthReduction(ReductionKind.PERCENTILE, 1)


# ---------------------------------------------------------------------------
# Rejected forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "p0",     # 0 is out of [1, 100]
        "p101",   # above the accepted ceiling
        "p-1",    # negative; isdigit() is False
        "p1000",  # also out of range
    ],
)
def test_reject_out_of_range_percentile(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_reduction(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "pAB",    # non-digit suffix
        "p1.5",   # fractional percentile
        "p 5",    # whitespace inside suffix
        "p",      # empty suffix
        "px",     # single non-digit
    ],
)
def test_reject_non_digit_percentile_suffix(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_reduction(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "",        # empty
        "P95",     # case-sensitive: capital P is rejected
        "MAX",     # case-sensitive: capital MAX is rejected
        "max1",    # max-like prefix is not "max"
        "max.5",   # max-like prefix is not "max"
        "unknown", # arbitrary string
        " max",    # leading whitespace
        "max ",    # trailing whitespace
    ],
)
def test_reject_unknown_or_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_reduction(bad)
