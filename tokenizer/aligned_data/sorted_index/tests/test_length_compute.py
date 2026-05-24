"""Tests for :mod:`sorted_index._length_compute`.

Covers the ALG-7 pre-pass + ALG-1 multi-mode shared walk:

* :func:`_count_variants_per_section` returns the correct per-section
  count on every Phase 0c fixture, oracle-checked against a direct
  :func:`iter_sections_bin` walk.
* :func:`compute_reduced_lengths` produces the expected per-mode
  reductions on the combined fixture: 0-variant stays 0, 1-variant
  collapses to the one length, multi-variant max equals the per-variant
  max, multi-variant p50 equals :func:`numpy.percentile` (method=lower).
* The MISSING_VARIANT_INDEX fixture does not crash; some length is
  produced for the caller section.
* All requested reductions come from a single Stage 1+2 walk per chunk
  (asserted via :func:`unittest.mock.patch` on ``walk_sections``).
* Deterministic: two back-to-back invocations on the same fixture
  produce byte-identical result arrays.
* The plan D-2.2 cut-fired assertion is asserted via a small
  :func:`unittest.mock.patch` of :func:`predict_lengths` that returns
  a synthetic :class:`Stage2Batch` whose variant carries
  ``cut_call_target_index != len(call_targets)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.matched_sections_bin import iter_sections_bin
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    compute_reduced_lengths,
)
from tokenizer.aligned_data.sorted_index._length_compute import (
    LARGE_CONTEXT_LEN,
    _count_variants_per_section,
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
    """Count matched-section variants by re-parsing ``sections.bin`` directly.

    Walks the same iterator the implementation uses but reads the
    length of ``section.variants`` straight out of the parsed
    :class:`Section`. The implementation does the same thing in a
    bounded ndarray; the oracle keeps the comparison value
    independent of the implementation's loop structure.
    """
    counts = []
    for i, section in enumerate(
        iter_sections_bin(base / f"{_BINARY_NAME}_sections.bin")
    ):
        if i >= num_matched:
            break
        counts.append(len(section.variants))
    return np.array(counts, dtype=np.uint32)


def _open(base: Path) -> Tuple[BinaryDataset, int]:
    """Open the dataset and return ``(dataset, num_matched_sections)``."""
    dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=None)
    return dataset, len(dataset.matched_bin_starts)


# ---------------------------------------------------------------------------
# _count_variants_per_section
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
def test_count_variants_matches_oracle(
    tmp_path: Path, builder, expected_counts
) -> None:
    """Per-section variant count matches a direct ``iter_sections_bin`` oracle.

    Also asserts the absolute counts line up with each fixture's
    documented spec order -- the fixture docstrings name the per-
    section variant counts, so this catches both fixture drift and
    implementation drift in one assertion.
    """
    base = builder(tmp_path)
    dataset, num_matched = _open(base)
    assert num_matched == len(expected_counts), (
        f"{builder.__name__}: expected {len(expected_counts)} matched "
        f"sections per spec; matched_index.bin has {num_matched}"
    )
    counts = _count_variants_per_section(base, _BINARY_NAME)
    assert counts.dtype == np.uint32
    oracle = _oracle_variant_counts(base, num_matched)
    np.testing.assert_array_equal(counts, oracle)
    np.testing.assert_array_equal(
        counts, np.array(expected_counts, dtype=np.uint32)
    )


# ---------------------------------------------------------------------------
# compute_reduced_lengths -- combined fixture covers every edge case
# ---------------------------------------------------------------------------


def _compute_on(
    base: Path, *, reductions, depth: int = 3
):
    """Run the full open-session + ``compute_reduced_lengths`` pipeline.

    Returns ``(results, counts)``. ``results`` is the per-mode dict;
    ``counts`` is the upstream variant-count array. Centralised so the
    individual assertions stay focused on the result shape.
    """
    dataset, num_matched = _open(base)
    counts = _count_variants_per_section(base, _BINARY_NAME)
    with dataset.open_session() as session:
        results = compute_reduced_lengths(
            session,
            num_sections=num_matched,
            section_variant_counts=counts,
            depth=depth,
            reductions=reductions,
        )
    return results, counts


def test_combined_fixture_shape_and_dtype(tmp_path: Path) -> None:
    """Result arrays are ``u32[num_sections]`` with no negative entries."""
    base = build_combined_fixture(tmp_path)
    _, num_matched = _open(base)
    results, _ = _compute_on(base, reductions=[_MAX, _P50, _P95])
    assert set(results.keys()) == {_MAX, _P50, _P95}
    for red, arr in results.items():
        assert arr.dtype == np.uint32, f"{red}: wrong dtype {arr.dtype}"
        assert arr.shape == (num_matched,), (
            f"{red}: wrong shape {arr.shape} vs {num_matched}"
        )


def test_combined_fixture_zero_variant_stamps_zero(tmp_path: Path) -> None:
    """0-variant sections collapse to 0 under every reduction.

    ``func_zero`` is section[0] in the combined fixture spec order.
    Every requested mode must stamp 0 there: the result must NOT depend
    on the reduction kind for a 0-variant section.
    """
    base = build_combined_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX, _P50, _P95])
    # Sanity: spec-order index 0 is the 0-variant section.
    assert counts[0] == 0
    for red, arr in results.items():
        assert arr[0] == 0, f"{red}: 0-variant section[0] stamped {arr[0]}"


def test_combined_fixture_one_variant_collapses(tmp_path: Path) -> None:
    """1-variant sections agree across MAX / PERCENTILE -- the per-variant length.

    ``solo_a`` (section[1]) has exactly one variant; every reduction
    must read that variant's ``total_surviving_token_count``.
    """
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
    """Multi-variant MAX/P50 agree with a per-variant oracle.

    ``multi_fn`` (section[2]) carries four variants. Re-run the walk
    via stage 1+2 directly and assert the per-variant
    ``total_surviving_token_count`` vector reduces to the same values
    the implementation returned for MAX (max) and P50
    (``np.percentile(..., 50, method="lower")``).
    """
    base = build_combined_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX, _P50])
    assert counts[2] == 4

    # Oracle: run stage 1+2 on just section[2] under the same params.
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
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind

    dataset, _ = _open(base)
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
    """The MISSING-slot caller section produces a length without crashing.

    The walker must treat a per-call slot whose
    ``section_variant_index == MISSING_VARIANT_INDEX`` as "no inlined
    callee body" rather than indexing into the callee's variant
    array. We do not pin a specific length value -- the caller has
    one variant, so the result is the caller's
    ``total_surviving_token_count`` after the missing slot was
    skipped; pinning the absolute number couples the test to internal
    splice walk semantics. Smoke that the call completes and stamps a
    non-zero length suffices.
    """
    base = build_missing_variant_index_fixture(tmp_path)
    results, counts = _compute_on(base, reductions=[_MAX])
    # caller_fn is section[0] with one variant per fixture docstring.
    assert counts[0] == 1
    assert results[_MAX][0] > 0
    # callee_fn is section[1] with two variants.
    assert counts[1] == 2
    assert results[_MAX][1] > 0


# ---------------------------------------------------------------------------
# Multi-mode shared walk: ONE stage 1+2 pass per chunk
# ---------------------------------------------------------------------------


def test_multi_mode_runs_one_walk_per_chunk(tmp_path: Path) -> None:
    """Three reductions share a single walk per chunk (plan ALG-1 amortisation).

    The combined fixture has 4 populated matched sections, well under
    :data:`CHUNK_SIZE` (64), so exactly one ``walk_sections`` call
    must back the whole compute regardless of how many reductions
    were requested.
    """
    base = build_combined_fixture(tmp_path)
    dataset, num_matched = _open(base)
    counts = _count_variants_per_section(base, _BINARY_NAME)

    real_walk_sections = __import__(
        "tokenizer.aligned_data.sorted_index._length_compute",
        fromlist=["walk_sections"],
    ).walk_sections

    call_counter = {"n": 0}

    def _counting_walk(*args, **kwargs):
        call_counter["n"] += 1
        return real_walk_sections(*args, **kwargs)

    with patch(
        "tokenizer.aligned_data.sorted_index._length_compute.walk_sections",
        side_effect=_counting_walk,
    ):
        with dataset.open_session() as session:
            results = compute_reduced_lengths(
                session,
                num_sections=num_matched,
                section_variant_counts=counts,
                depth=3,
                reductions=[_MAX, _P50, _P95],
            )

    assert call_counter["n"] == 1, (
        f"expected 1 walk_sections call for a single-chunk fixture; "
        f"got {call_counter['n']}"
    )
    # And the three reductions must still produce per-mode arrays.
    assert set(results.keys()) == {_MAX, _P50, _P95}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_runs(tmp_path: Path) -> None:
    """Two back-to-back invocations produce byte-identical result arrays.

    The deterministic RNG seed (:data:`numpy.random.default_rng(0)`)
    has zero effect under :data:`LARGE_VARIANT_CAP`, so the result is
    a pure function of the input bytes. Drift here flags accidental
    non-determinism in the walk.
    """
    base = build_combined_fixture(tmp_path)
    first, _ = _compute_on(base, reductions=[_MAX, _P50, _P95])
    second, _ = _compute_on(base, reductions=[_MAX, _P50, _P95])
    assert set(first.keys()) == set(second.keys())
    for red in first:
        np.testing.assert_array_equal(first[red], second[red])
        assert first[red].tobytes() == second[red].tobytes()


# ---------------------------------------------------------------------------
# Empty / all-zero edge cases
# ---------------------------------------------------------------------------


def test_all_zero_variant_short_circuits(tmp_path: Path) -> None:
    """A binary whose whole matched arm is 0-variant returns all-zero arrays.

    Constructed by hand from
    :func:`build_0_variant_section_fixture` -- only the func_zero
    section. The pre-filter short-circuits before any walk runs;
    every reduction stamps zero.
    """
    base = build_0_variant_section_fixture(tmp_path)
    dataset, num_matched = _open(base)
    counts = _count_variants_per_section(base, _BINARY_NAME)
    # Force the all-zero path by zeroing the count for the companion
    # section (the fixture has func_zero + func_one; we want the
    # pre-filter exercised when populated_idx.size == 0).
    zeroed = np.zeros_like(counts)
    with dataset.open_session() as session:
        results = compute_reduced_lengths(
            session,
            num_sections=num_matched,
            section_variant_counts=zeroed,
            depth=3,
            reductions=[_MAX, _P50],
        )
    for red, arr in results.items():
        np.testing.assert_array_equal(arr, np.zeros(num_matched, dtype=np.uint32))


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    """``section_variant_counts.size != num_sections`` is a hard fail.

    Catches the most likely caller bug (passing the wrong array)
    early with a clear message instead of silently mis-indexing.
    """
    base = build_combined_fixture(tmp_path)
    dataset, num_matched = _open(base)
    with dataset.open_session() as session:
        with pytest.raises(ValueError, match="section_variant_counts.shape"):
            compute_reduced_lengths(
                session,
                num_sections=num_matched,
                section_variant_counts=np.zeros(num_matched + 1, dtype=np.uint32),
                depth=3,
                reductions=[_MAX],
            )


# ---------------------------------------------------------------------------
# D-2.2: cut-fired assertion
# ---------------------------------------------------------------------------


def test_cut_fired_raises(tmp_path: Path) -> None:
    """Stage 2 cut-fired raises with a useful message (plan D-2.2).

    Patches :func:`predict_lengths` to return a synthetic
    :class:`Stage2Batch` whose variant carries
    ``cut_call_target_index != len(call_targets)``. The compute MUST
    refuse to silently under-report and raise
    :class:`AssertionError` instead.
    """
    from dataclasses import replace as _dc_replace

    from tokenizer.aligned_data.loader.batch_decode._length_predict import (
        predict_lengths as real_predict_lengths,
    )

    base = build_combined_fixture(tmp_path)
    dataset, num_matched = _open(base)
    counts = _count_variants_per_section(base, _BINARY_NAME)

    def _cut_predict(stage1, *, context_len):
        real = real_predict_lengths(stage1, context_len=context_len)
        # Mutate the first populated variant of the first section to
        # report a cut firing inside it (cut_call_target_index < n).
        mutated_sections = []
        injected = False
        for st2_sec in real.sections:
            if not injected and st2_sec.variants:
                v0 = st2_sec.variants[0]
                if len(v0.call_targets) >= 1:
                    bad = _dc_replace(v0, cut_call_target_index=0)
                    new_section = _dc_replace(
                        st2_sec, variants=[bad, *st2_sec.variants[1:]]
                    )
                    mutated_sections.append(new_section)
                    injected = True
                    continue
            mutated_sections.append(st2_sec)
        return _dc_replace(real, sections=mutated_sections)

    with patch(
        "tokenizer.aligned_data.sorted_index._length_compute.predict_lengths",
        side_effect=_cut_predict,
    ):
        with dataset.open_session() as session:
            with pytest.raises(AssertionError, match="cut fired"):
                compute_reduced_lengths(
                    session,
                    num_sections=num_matched,
                    section_variant_counts=counts,
                    depth=3,
                    reductions=[_MAX],
                )
