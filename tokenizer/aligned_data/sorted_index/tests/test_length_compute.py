"""Tests for :mod:`sorted_index._length_compute` (walk-free pipeline).

Covers the columnar pre-pass + the vectorized multi-mode compute:

* :func:`read_section_variant_info` counts match a direct
  :func:`iter_sections_bin` oracle on every Phase 0c fixture.
* :func:`compute_reduced_lengths` produces the expected per-mode
  reductions on the combined fixture: 0-variant stays 0, 1-variant
  collapses to the one length, multi-variant MAX / P50 agree with a
  LEGACY Stage 1+2 walk oracle (the strongest cross-check: the old
  machinery still exists in the loader and must agree).
* The MISSING_VARIANT_INDEX fixture does not crash.
* All (mode, depth) outputs come from ONE graph traversal (asserted
  via :func:`unittest.mock.patch` on ``compute_node_lengths``).
* Deterministic: two invocations produce byte-identical arrays.
* The plan D-2.2 budget guard raises when a spliced length reaches
  ``LARGE_CONTEXT_LEN``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest

from tokenizer.aligned_data.matched_sections_bin import iter_sections_bin
from tokenizer.aligned_data.sorted_index import (
    IndexSpec,
    LengthReduction,
    ReductionKind,
    compute_reduced_lengths,
    read_section_variant_info,
)
from tokenizer.aligned_data.sorted_index._length_compute import (
    LARGE_CONTEXT_LEN,
)

from .fixtures import (
    build_0_variant_section_fixture,
    build_1_variant_section_fixture,
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)


_BINARY_NAME = "sortbin"

_MAX = LengthReduction(kind=ReductionKind.MAX)
_P50 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=50)
_P95 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=95)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oracle_variant_counts(base: Path, num_matched: int) -> np.ndarray:
    """Count matched-section variants via a direct scalar BIN walk."""
    counts = []
    for i, section in enumerate(
        iter_sections_bin(base / f"{_BINARY_NAME}_sections.bin")
    ):
        if i >= num_matched:
            break
        counts.append(len(section.variants))
    return np.array(counts, dtype=np.int64)


def _data_bytes(base: Path) -> np.ndarray:
    return np.memmap(
        str(base / f"{_BINARY_NAME}_data.bin"), dtype=np.uint8, mode="r"
    )


def _compute_on(base: Path, *, reductions, depth: int = 3):
    """Run pre-pass + ``compute_reduced_lengths``; unwrap one depth.

    Returns ``(results, counts)`` where ``results`` maps each
    reduction to its ``u32[num_sections]`` array at ``depth``.
    """
    section_info = read_section_variant_info(base, _BINARY_NAME)
    per_spec = compute_reduced_lengths(
        section_info,
        _data_bytes(base),
        depths=[depth],
        reductions=reductions,
    )
    results = {
        red: per_spec[IndexSpec(reduction=red, depth=depth)]
        for red in reductions
    }
    return results, section_info.counts


# ---------------------------------------------------------------------------
# Pre-pass variant counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,expected_counts",
    [
        (build_0_variant_section_fixture, [0, 1]),
        (build_1_variant_section_fixture, [1, 1]),
        (build_many_variant_section_fixture, [4, 1]),
        # combined fixture: spec order = func_zero(0), solo_a(1),
        # multi_fn(4), caller_fn(1), callee_fn(2).
        (build_combined_fixture, [0, 1, 4, 1, 2]),
        (build_missing_variant_index_fixture, [1, 2]),
    ],
)
def test_prepass_counts_match_oracle(
    tmp_path: Path, builder, expected_counts
) -> None:
    """Pre-pass per-section counts match the scalar BIN-walk oracle."""
    base = builder(tmp_path)
    info = read_section_variant_info(base, _BINARY_NAME)
    assert info.counts.size == len(expected_counts), (
        f"{builder.__name__}: expected {len(expected_counts)} matched "
        f"sections per spec; got {info.counts.size}"
    )
    oracle = _oracle_variant_counts(base, info.counts.size)
    np.testing.assert_array_equal(info.counts, oracle)
    np.testing.assert_array_equal(
        info.counts, np.array(expected_counts, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# compute_reduced_lengths -- combined fixture covers every edge case
# ---------------------------------------------------------------------------


def test_combined_fixture_shape_and_dtype(tmp_path: Path) -> None:
    """Result arrays are ``u32[num_sections]``."""
    base = build_combined_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX, _P50, _P95])
    assert set(results.keys()) == {_MAX, _P50, _P95}
    for red, arr in results.items():
        assert arr.dtype == np.uint32, f"{red}: wrong dtype {arr.dtype}"
        assert arr.shape == (counts.size,), (
            f"{red}: wrong shape {arr.shape} vs {counts.size}"
        )


def test_combined_fixture_zero_variant_stamps_zero(tmp_path: Path) -> None:
    """0-variant sections collapse to 0 under every reduction."""
    base = build_combined_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX, _P50, _P95])
    assert counts[0] == 0
    for red, arr in results.items():
        assert arr[0] == 0, f"{red}: 0-variant section[0] stamped {arr[0]}"


def test_combined_fixture_one_variant_collapses(tmp_path: Path) -> None:
    """1-variant sections agree across MAX / PERCENTILE."""
    base = build_combined_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX, _P50, _P95])
    assert counts[1] == 1
    assert results[_MAX][1] == results[_P50][1]
    assert results[_MAX][1] == results[_P95][1]
    assert results[_MAX][1] > 0, (
        "solo_a is a non-trivial single-variant section; its length "
        "should be strictly positive"
    )


def test_combined_fixture_many_variant_max_vs_p50(tmp_path: Path) -> None:
    """Multi-variant MAX/P50 agree with the LEGACY Stage 1+2 oracle.

    ``multi_fn`` (section[2]) carries four variants. The legacy walk
    machinery still exists in the loader; running it on section[2]
    must produce per-variant ``total_surviving_token_count`` values
    whose max / p50 equal the new pipeline's outputs.
    """
    from tokenizer.aligned_data.loader.batch_decode._length_predict import (
        predict_lengths,
    )
    from tokenizer.aligned_data.loader.batch_decode._section_walk import (
        walk_sections,
    )
    from tokenizer.aligned_data.loader.batch_decode._types import (
        SectionPointerSpec,
        VariantPadding,
    )
    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind

    base = build_combined_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX, _P50])
    assert counts[2] == 4

    dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=None)
    rng = np.random.default_rng(0)
    with dataset.open_session() as session:
        stage1 = walk_sections(
            session,
            [SectionPointerSpec(arm=SectionKind.MATCHED, idx=2)],
            num_variants_per_section=np.iinfo(np.int32).max,
            max_depth=3,
            variant_padding=VariantPadding.RAGGED,
            inlined_equivalent_call_targets_only=False,
            rng=rng,
        )
        stage2 = predict_lengths(stage1, context_len=LARGE_CONTEXT_LEN)
    per_variant = np.array(
        [v.total_surviving_token_count for v in stage2.sections[0].variants],
        dtype=np.uint32,
    )
    assert per_variant.size == 4
    assert int(results[_MAX][2]) == int(per_variant.max())
    assert int(results[_P50][2]) == int(
        np.percentile(per_variant, 50, method="lower")
    )


def test_missing_variant_index_does_not_crash(tmp_path: Path) -> None:
    """The MISSING-slot caller section produces a length without crashing."""
    base = build_missing_variant_index_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX])
    assert counts[0] == 1
    assert results[_MAX][0] > 0
    assert counts[1] == 2
    assert results[_MAX][1] > 0


# ---------------------------------------------------------------------------
# Multi-mode amortisation: ONE graph traversal for all (mode, depth) pairs
# ---------------------------------------------------------------------------


def test_multi_mode_runs_one_graph_traversal(tmp_path: Path) -> None:
    """Three reductions x two depths share one ``compute_node_lengths``."""
    base = build_combined_fixture(tmp_path)
    section_info = read_section_variant_info(base, _BINARY_NAME)

    from tokenizer.aligned_data.sorted_index._graph_lengths import (
        compute_node_lengths as real_compute,
    )

    call_counter = {"n": 0}

    def _counting(*args, **kwargs):
        call_counter["n"] += 1
        return real_compute(*args, **kwargs)

    with patch(
        "tokenizer.aligned_data.sorted_index._length_compute."
        "compute_node_lengths",
        side_effect=_counting,
    ):
        results = compute_reduced_lengths(
            section_info,
            _data_bytes(base),
            depths=[1, 3],
            reductions=[_MAX, _P50, _P95],
        )

    assert call_counter["n"] == 1, (
        f"expected 1 compute_node_lengths call; got {call_counter['n']}"
    )
    assert len(results) == 6  # 3 reductions x 2 depths


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_runs(tmp_path: Path) -> None:
    """Two invocations produce byte-identical result arrays."""
    base = build_combined_fixture(tmp_path)
    first, _ = _compute_on(base, reductions=[_MAX, _P50, _P95])
    second, _ = _compute_on(base, reductions=[_MAX, _P50, _P95])
    assert set(first.keys()) == set(second.keys())
    for red in first:
        np.testing.assert_array_equal(first[red], second[red])
        assert first[red].tobytes() == second[red].tobytes()


# ---------------------------------------------------------------------------
# Empty / invalid-parameter edge cases
# ---------------------------------------------------------------------------


def test_all_zero_variant_stamps_all_zero(tmp_path: Path) -> None:
    """A matched arm whose every section is 0-variant returns all-zeros.

    Built from the 0-variant fixture by slicing the catalog down to
    just the ``func_zero`` section (section[0]).
    """
    base = build_0_variant_section_fixture(tmp_path)
    info = read_section_variant_info(base, _BINARY_NAME)
    sliced = read_section_variant_info(base, _BINARY_NAME)
    # Reparse bounded to ONLY the 0-variant section.
    from tokenizer.aligned_data.matched_sections_columnar import (
        parse_sections_columnar,
    )
    from tokenizer.aligned_data.sorted_index._prepass import (
        SectionVariantInfo,
    )

    blob = np.fromfile(base / f"{_BINARY_NAME}_sections.bin", dtype=np.uint8)
    zero_only = SectionVariantInfo(
        cols=parse_sections_columnar(blob, info.section_offsets[:1]),
        section_offsets=info.section_offsets[:1],
    )
    assert zero_only.counts.tolist() == [0]
    del sliced
    results = compute_reduced_lengths(
        zero_only,
        _data_bytes(base),
        depths=[3],
        reductions=[_MAX, _P50],
    )
    for _spec, arr in results.items():
        np.testing.assert_array_equal(arr, np.zeros(1, dtype=np.uint32))


def test_empty_depths_raises(tmp_path: Path) -> None:
    base = build_combined_fixture(tmp_path)
    section_info = read_section_variant_info(base, _BINARY_NAME)
    with pytest.raises(ValueError, match="depths must be a non-empty"):
        compute_reduced_lengths(
            section_info, _data_bytes(base), depths=[], reductions=[_MAX]
        )


def test_negative_depth_raises(tmp_path: Path) -> None:
    base = build_combined_fixture(tmp_path)
    section_info = read_section_variant_info(base, _BINARY_NAME)
    with pytest.raises(ValueError, match="depths must all be >= 0"):
        compute_reduced_lengths(
            section_info,
            _data_bytes(base),
            depths=[1, -1],
            reductions=[_MAX],
        )


# ---------------------------------------------------------------------------
# D-2.2: budget guard
# ---------------------------------------------------------------------------


def test_budget_guard_raises(tmp_path: Path) -> None:
    """Lengths reaching LARGE_CONTEXT_LEN refuse to under-report.

    Patches the bulk own-length step to claim every record is at the
    budget; the compute MUST raise rather than emit a clipped u32.
    """
    base = build_combined_fixture(tmp_path)
    section_info = read_section_variant_info(base, _BINARY_NAME)

    def _huge(cols, data_u8):
        return np.full(
            cols.var_n_calls.size, LARGE_CONTEXT_LEN, dtype=np.int64
        )

    with patch(
        "tokenizer.aligned_data.sorted_index._graph_lengths._own_lengths",
        side_effect=_huge,
    ):
        with pytest.raises(AssertionError, match="LARGE_CONTEXT_LEN"):
            compute_reduced_lengths(
                section_info,
                _data_bytes(base),
                depths=[3],
                reductions=[_MAX],
            )
