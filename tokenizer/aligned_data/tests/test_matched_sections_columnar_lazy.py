"""Equivalence: lazy section-bounded columnar vs the eager columnar parse.

Pins :class:`tokenizer.aligned_data.matched_sections_columnar_lazy.
LazyColumnarSections` to :func:`~tokenizer.aligned_data.
matched_sections_columnar.parse_sections_columnar` on the production-writer
fixtures (0-variant sections, odd/even variant counts for the jump-table
pad, MISSING sentinel slots, call-target tables). The lazy catalog is a
drop-in: its section-level skeleton + CSR offsets are eager and equal to
the eager parse; its heavy columns equal the eager ones for every FILLED
section (and a full fill reproduces the eager catalog bit-for-bit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.matched_sections_columnar_lazy import (
    _MISSING_LOG_INTERVAL_S,
    parse_sections_columnar_lazy,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_0_variant_section_fixture,
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)


_SKELETON_FIELDS = (
    "function_name_ptr",
    "is_duplicated",
    "n_call_targets",
    "n_variants",
    "ct_offsets",
    "var_offsets",
)
_HEAVY_FIELDS = (
    "ct_function_name_ptr",
    "ct_function_section_ptr",
    "ct_type",
    "ct_is_matched",
    "var_ref_offset",
    "var_data_offset_shifted",
    "var_n_calls",
    "pce_offsets",
    "pce_called_idx",
    "pce_section_variant_index",
)


def _binary_name(base: Path) -> str:
    (idx,) = (
        p
        for p in base.glob("*_index.bin")
        if not p.name.endswith("_unmatched_index.bin")
    )
    return idx.name[: -len("_index.bin")]


def _load(base: Path):
    name = _binary_name(base)
    starts, lengths = read_csv_section_index_arrays(base / f"{name}_index.bin")
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)
    eager = parse_sections_columnar(blob, starts, lengths)
    lazy = parse_sections_columnar_lazy(blob, starts, lengths)
    return eager, lazy, starts.size


@pytest.mark.parametrize(
    "builder",
    [
        build_0_variant_section_fixture,
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
        build_combined_fixture,
    ],
)
def test_skeleton_eager_pre_fill(builder, tmp_path: Path) -> None:
    """The lazy skeleton + CSR offsets equal the eager parse with no fill."""
    eager, lazy, _ = _load(builder(tmp_path))
    for f in _SKELETON_FIELDS:
        np.testing.assert_array_equal(
            getattr(lazy, f), getattr(eager, f), err_msg=f
        )
    np.testing.assert_array_equal(lazy.sec_of_var, eager.sec_of_var)
    # pce_offsets section boundaries are seeded eagerly (== section bases),
    # so cross-section CSR reads are correct even before any heavy fill.
    np.testing.assert_array_equal(
        lazy.pce_offsets[eager.var_offsets], eager.pce_offsets[eager.var_offsets]
    )


@pytest.mark.parametrize(
    "builder",
    [
        build_0_variant_section_fixture,
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
        build_combined_fixture,
    ],
)
def test_full_fill_byte_identical(builder, tmp_path: Path) -> None:
    """Filling every section reproduces the eager catalog bit-for-bit."""
    eager, lazy, n_sec = _load(builder(tmp_path))
    lazy.ensure_sections(np.arange(n_sec))
    for f in _SKELETON_FIELDS + _HEAVY_FIELDS:
        np.testing.assert_array_equal(
            getattr(lazy, f), getattr(eager, f), err_msg=f
        )
    np.testing.assert_array_equal(lazy.pce_variant(), eager.pce_variant())


@pytest.mark.parametrize(
    "builder",
    [
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
        build_combined_fixture,
    ],
)
def test_partial_fill_matches_eager_on_touched(builder, tmp_path: Path) -> None:
    """A filled section's heavy slices equal the eager ones, idempotently."""
    eager, lazy, n_sec = _load(builder(tmp_path))
    # Fill every other section (and re-call to prove idempotency).
    touched = np.arange(0, n_sec, 2)
    lazy.ensure_sections(touched)
    lazy.ensure_sections(touched)  # idempotent: no double-write
    for s in touched.tolist():
        c0, c1 = int(eager.ct_offsets[s]), int(eager.ct_offsets[s + 1])
        for f in (
            "ct_function_name_ptr",
            "ct_function_section_ptr",
            "ct_type",
            "ct_is_matched",
        ):
            np.testing.assert_array_equal(
                getattr(lazy, f)[c0:c1], getattr(eager, f)[c0:c1], err_msg=f
            )
        v0, v1 = int(eager.var_offsets[s]), int(eager.var_offsets[s + 1])
        for f in ("var_ref_offset", "var_data_offset_shifted", "var_n_calls"):
            np.testing.assert_array_equal(
                getattr(lazy, f)[v0:v1], getattr(eager, f)[v0:v1], err_msg=f
            )
        np.testing.assert_array_equal(
            lazy.pce_offsets[v0 : v1 + 1], eager.pce_offsets[v0 : v1 + 1]
        )
        e0, e1 = int(eager.pce_offsets[v0]), int(eager.pce_offsets[v1])
        for f in ("pce_called_idx", "pce_section_variant_index"):
            np.testing.assert_array_equal(
                getattr(lazy, f)[e0:e1], getattr(eager, f)[e0:e1], err_msg=f
            )


def test_missing_inventory_bounded_to_touched(tmp_path: Path) -> None:
    """The lazy MISSING tally counts only filled sections (0 before fill)."""
    eager, lazy, n_sec = _load(build_missing_variant_index_fixture(tmp_path))
    full = eager.missing_variant_index_count()
    assert full == int(
        (eager.pce_section_variant_index == MISSING_VARIANT_INDEX).sum()
    )
    assert lazy.missing_variant_index_count() == 0  # nothing filled yet
    lazy.ensure_sections(np.arange(n_sec))
    assert lazy.missing_variant_index_count() == full


class _MissingStub:
    """Minimal ``sub`` for :meth:`LazyColumnarSections._tally_missing`.

    The tally reads only ``pce_section_variant_index``; ``n_missing`` slots
    carry the sentinel so each call contributes exactly that many.
    """

    def __init__(self, n_missing: int) -> None:
        self.pce_section_variant_index = np.full(
            n_missing, MISSING_VARIANT_INDEX, dtype=np.uint16
        )


def _missing_log_records(caplog):
    return [r for r in caplog.records if "MISSING_VARIANT_INDEX" in r.message]


def test_missing_log_throttled_to_one_per_interval(tmp_path, caplog) -> None:
    """The MISSING-edge log emits <=1/interval; the tally stays exact."""
    clock = [0.0]
    # Build a lazy catalog with an injected, manually-advanced clock.
    base = build_missing_variant_index_fixture(tmp_path / "throttle")
    name = _binary_name(base)
    starts, lengths = read_csv_section_index_arrays(base / f"{name}_index.bin")
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)
    lazy = parse_sections_columnar_lazy(
        blob, starts, lengths, clock=lambda: clock[0]
    )

    caplog.set_level("ERROR", logger="tokenizer.aligned_data."
                                      "matched_sections_columnar_lazy")

    # (1) First call with new MISSING entries EMITS.
    lazy._tally_missing(_MissingStub(3))
    assert len(_missing_log_records(caplog)) == 1
    assert lazy.missing_variant_index_count() == 3

    # (2) Subsequent calls within the interval ACCUMULATE but do NOT emit.
    clock[0] = _MISSING_LOG_INTERVAL_S - 1.0
    lazy._tally_missing(_MissingStub(5))
    lazy._tally_missing(_MissingStub(7))
    assert len(_missing_log_records(caplog)) == 1  # still just the first
    assert lazy.missing_variant_index_count() == 3 + 5 + 7

    # (3) A call after >= the interval RE-EMITS with the running total, and
    #     reports the entries accumulated since the previous emission.
    clock[0] = _MISSING_LOG_INTERVAL_S
    lazy._tally_missing(_MissingStub(2))
    records = _missing_log_records(caplog)
    assert len(records) == 2
    assert lazy.missing_variant_index_count() == 3 + 5 + 7 + 2
    # running total (17) is in the second line; since-last-emit count is 14.
    assert "17" in records[1].message
    assert records[1].args[0] == 5 + 7 + 2  # since the first emit
    assert records[1].args[1] == 17  # running total


def test_missing_tally_unchanged_by_throttle(tmp_path) -> None:
    """Throttling the LOG does not alter the running-total counting."""
    base = build_missing_variant_index_fixture(tmp_path / "count")
    name = _binary_name(base)
    starts, lengths = read_csv_section_index_arrays(base / f"{name}_index.bin")
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)

    # Fixed clock (never advances -> only the first call ever emits) must
    # still reach the SAME running total as the unthrottled eager catalog.
    eager = parse_sections_columnar(blob, starts, lengths)
    lazy = parse_sections_columnar_lazy(
        blob, starts, lengths, clock=lambda: 0.0
    )
    lazy.ensure_sections(np.arange(starts.size))
    assert (
        lazy.missing_variant_index_count()
        == eager.missing_variant_index_count()
    )
