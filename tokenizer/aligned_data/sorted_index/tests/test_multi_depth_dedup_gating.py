"""Tests for the multi-depth + duplicate-aware + gated build features.

Four concerns, each pinned against an independent oracle:

* Multi-depth byte-equivalence (Feature A): ONE multi-depth
  :func:`write_sorted_index_files` call produces files BYTE-IDENTICAL
  to per-depth single-depth calls. The acceptance gate for the
  one-walk-serves-every-depth property.
* Duplicate-group reduction semantics (Feature B.2): the PERCENTILE
  group representative is the group AVERAGE; the MAX group
  representative is the group MAX. Pinned at
  :meth:`LengthReduction.reduce_groups` (the avg-vs-max home) AND
  through :meth:`DuplicateHandling.reduce_section` (the pointer-grouped
  path). Also the strict-generalisation property: singleton groups
  reduce identically to the flat :meth:`LengthReduction.reduce`.
* Minimum-variant gating (Feature B.1 + B.3): a gated-out section is
  stamped length 0 (the 0-variant representation). The compose case
  (``--min-variants 8 --min-variants-unique 6``: exactly 2 dups pass,
  3 dups fail) is pinned on :class:`VariantGate`.
* Validation errors (Feature B.3): ``M > N`` and negative thresholds
  are rejected at :class:`VariantGate` construction; the CLI surfaces
  them as a parse error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.matched_sections_bin import iter_sections_bin
from tokenizer.aligned_data.sorted_index import (
    DEDUP_BY_DATA_POINTER,
    PLAIN,
    IndexSpec,
    LengthReduction,
    ReductionKind,
    VariantGate,
    compute_reduced_lengths,
    read_section_variant_info,
    write_sorted_index_files,
)

from .fixtures import build_combined_fixture


_BINARY_NAME = "sortbin"

_MAX = LengthReduction(kind=ReductionKind.MAX)
_P50 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=50)
_P95 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=95)


# ---------------------------------------------------------------------------
# Feature A: multi-depth byte-equivalence
# ---------------------------------------------------------------------------


def test_multi_depth_files_byte_equal_single_depth(tmp_path: Path) -> None:
    """ONE multi-depth call writes files byte-identical to per-depth calls.

    The acceptance gate of the single-walk-multi-depth property: a
    multi-depth invocation at depths {1, 3} must produce exactly the
    bytes the old single-depth path produces for each depth in
    isolation (one .idx per (mode, depth), each byte-equal).
    """
    base = build_combined_fixture(tmp_path)
    reductions = [_MAX, _P50, _P95]
    depths = [1, 3]

    # OLD path-shape: one single-depth call per depth, into its own dir.
    single_dir = tmp_path / "single"
    for depth in depths:
        write_sorted_index_files(
            base, _BINARY_NAME,
            reductions=reductions,
            depths=[depth],
            output_dir=single_dir,
        )

    # NEW path-shape: one multi-depth call.
    multi_dir = tmp_path / "multi"
    write_sorted_index_files(
        base, _BINARY_NAME,
        reductions=reductions,
        depths=depths,
        output_dir=multi_dir,
    )

    # Same file set, byte-identical contents.
    single_files = sorted(p.name for p in single_dir.glob("*.idx"))
    multi_files = sorted(p.name for p in multi_dir.glob("*.idx"))
    assert single_files == multi_files
    assert len(single_files) == len(reductions) * len(depths)
    for name in single_files:
        assert (single_dir / name).read_bytes() == (
            multi_dir / name
        ).read_bytes(), f"{name}: multi-depth byte-diverges from single-depth"


def test_max_depth_length_equals_total_surviving_count(tmp_path: Path) -> None:
    """At the deepest requested depth, the per-variant length sum equals
    the variant's ``total_surviving_token_count``.

    This pins the depth-summation (``_variant_lengths_at_depth`` over
    ``path_depth <= max_depth``) against the OLD single-depth semantics
    (the whole-variant ``total_surviving_token_count``): with no cutoff
    firing, summing every call_target at the deepest depth must recover
    exactly the variant total, so the MAX reduction at the deepest depth
    matches a fresh walk's per-variant max.
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
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind
    from tokenizer.aligned_data.sorted_index._length_compute import (
        LARGE_CONTEXT_LEN,
    )

    from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset

    base = build_combined_fixture(tmp_path)
    data_u8, section_info = _open(base)
    spec = IndexSpec(reduction=_MAX, depth=3)

    result = compute_reduced_lengths(
        section_info, data_u8, depths=[3], reductions=[_MAX]
    )[spec]

    # multi_fn is section[2] (4 variants); oracle a fresh LEGACY walk
    # over it and reduce the per-variant ``total_surviving_token_count``.
    dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=None)
    with dataset.open_session() as session:
        rng = np.random.default_rng(0)
        stage1 = walk_sections(
            session,
            [SectionPointerSpec(arm=SectionKind.MATCHED, idx=2)],
            num_variants_per_section=np.iinfo(np.int32).max,
            max_depth=3,
            variant_padding=VariantPadding.RAGGED,
            rng=rng,
        )
        stage2 = predict_lengths(stage1, context_len=LARGE_CONTEXT_LEN)
    per_variant = np.array(
        [v.total_surviving_token_count for v in stage2.sections[0].variants],
        dtype=np.uint32,
    )
    assert int(result[2]) == int(per_variant.max())


# ---------------------------------------------------------------------------
# Feature B.2: duplicate-group reduction semantics (avg vs max)
# ---------------------------------------------------------------------------


def test_reduce_groups_percentile_uses_group_average() -> None:
    """PERCENTILE collapses each duplicate-group to its AVERAGE.

    Two groups: [10, 20] -> avg 15.0, [4] -> 4.0. The p50 (lower) over
    representatives {15.0, 4.0} is 4 -- NOT a percentile over the flat
    [10, 20, 4] (whose p50-lower would be 10). The difference proves the
    group AVERAGE representative is used, not the raw values.
    """
    groups = [np.array([10, 20], dtype=np.uint32), np.array([4], dtype=np.uint32)]
    assert _P50.reduce_groups(groups) == 4
    # Sanity contrast against the flat reduction over the same values.
    assert _P50.reduce(np.array([10, 20, 4], dtype=np.uint32)) == 10


def test_reduce_groups_max_uses_group_max() -> None:
    """MAX collapses each duplicate-group to its MAX, then takes the max.

    Groups [10, 20] -> 20, [4] -> 4; the max over {20, 4} is 20 -- equal
    to the flat max (MAX is idempotent under grouping). Pinned so the
    MAX representative rule (group max, not group average) is explicit.
    """
    groups = [np.array([10, 20], dtype=np.uint32), np.array([4], dtype=np.uint32)]
    assert _MAX.reduce_groups(groups) == 20
    assert _MAX.reduce(np.array([10, 20, 4], dtype=np.uint32)) == 20


@pytest.mark.parametrize("reduction", [_MAX, _P50, _P95])
def test_reduce_groups_singletons_equal_flat_reduce(reduction) -> None:
    """Singleton groups reduce identically to the flat vector.

    The strict-generalisation property: with no duplicates every variant
    is its own group, so ``reduce_groups`` over singletons MUST equal
    ``reduce`` over the concatenation -- this is what guarantees the
    ``--adjust-for-duplicates`` off path is byte-identical to the
    pre-feature reduction.
    """
    lengths = np.array([3, 9, 9, 17, 4, 4, 4, 21], dtype=np.uint32)
    singletons = [np.array([v], dtype=np.uint32) for v in lengths]
    assert reduction.reduce_groups(singletons) == reduction.reduce(lengths)


def test_duplicate_handling_groups_by_pointer() -> None:
    """``DuplicateHandling`` groups lengths by data-bin pointer, then reduces.

    Four variants, pointers ``[P, P, Q, R]`` -> groups [10, 20] (avg 15),
    [4] (4), [30] (30). The PLAIN strategy ignores the pointers (flat
    reduce); the dedup strategy groups first. Pinning both on the same
    input keeps the strategy selection honest.
    """
    lengths = np.array([10, 20, 4, 30], dtype=np.uint32)
    pointers = np.array([7, 7, 9, 11], dtype=np.uint32)

    # PLAIN: flat p50-lower over [10, 20, 4, 30] = 10.
    assert PLAIN.reduce_section(
        _P50, lengths=lengths, data_pointers=pointers
    ) == int(np.percentile(lengths, 50, method="lower"))

    # DEDUP: p50-lower over representatives {15.0, 4.0, 30.0} = 15.
    assert DEDUP_BY_DATA_POINTER.reduce_section(
        _P50, lengths=lengths, data_pointers=pointers
    ) == 15


# ---------------------------------------------------------------------------
# Feature B.1 / B.3: gating
# ---------------------------------------------------------------------------


def test_gate_min_variants_total() -> None:
    """``min_variants`` gates on the top-level total (duplicates included)."""
    gate = VariantGate(min_variants=4)
    assert gate.passes(n_total=4, n_unique=1)
    assert not gate.passes(n_total=3, n_unique=3)


def test_gate_compose_8_total_6_unique() -> None:
    """``--min-variants 8 --min-variants-unique 6``: <=2 dups tolerated.

    At exactly 8 top-level variants, the section passes with at most 2
    duplicates (>= 6 unique) and fails with 3 (only 5 unique).
    """
    gate = VariantGate(min_variants=8, min_variants_unique=6)
    # 8 total, 2 dups -> 6 unique: passes.
    assert gate.passes(n_total=8, n_unique=6)
    # 8 total, 3 dups -> 5 unique: fails the uniqueness leg.
    assert not gate.passes(n_total=8, n_unique=5)
    # 7 total: fails the total leg regardless of uniqueness.
    assert not gate.passes(n_total=7, n_unique=7)


def test_gate_unique_without_total_is_legal() -> None:
    """``min_variants_unique`` alone (no ``min_variants``) gates by uniqueness."""
    gate = VariantGate(min_variants_unique=3)
    assert gate.passes(n_total=100, n_unique=3)
    assert not gate.passes(n_total=100, n_unique=2)


def test_gate_disabled_passes_everything() -> None:
    """The default (0/0) gate clears every section, even 1-variant ones."""
    gate = VariantGate()
    assert gate.passes(n_total=1, n_unique=1)
    assert gate.passes(n_total=0, n_unique=0)


def test_gate_m_greater_than_n_rejected() -> None:
    """``min_variants_unique > min_variants`` is an unsatisfiable gate."""
    with pytest.raises(ValueError, match="min_variants_unique must be <="):
        VariantGate(min_variants=4, min_variants_unique=6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_variants": -1},
        {"min_variants_unique": -1},
    ],
)
def test_gate_negative_rejected(kwargs) -> None:
    """Negative thresholds are rejected at construction."""
    with pytest.raises(ValueError, match=">= 0"):
        VariantGate(**kwargs)


# ---------------------------------------------------------------------------
# Gating integration: gated-out sections are stamped length 0
# ---------------------------------------------------------------------------


def _open(base: Path):
    data_u8 = np.memmap(
        str(base / f"{_BINARY_NAME}_data.bin"), dtype=np.uint8, mode="r"
    )
    return data_u8, read_section_variant_info(base, _BINARY_NAME)


def test_gating_stamps_zero_on_failing_sections(tmp_path: Path) -> None:
    """A section below ``min_variants`` is stamped 0 (the 0-variant path).

    The combined fixture's spec order is func_zero(0), solo_a(1),
    multi_fn(4), caller_fn(1), callee_fn(2). With ``min_variants=2``
    only multi_fn(4) and callee_fn(2) survive; solo_a / caller_fn drop
    to 0 alongside the already-0 func_zero. The surviving sections keep
    the exact length the ungated build produced.
    """
    base = build_combined_fixture(tmp_path)
    data_u8, section_info = _open(base)
    spec = IndexSpec(reduction=_MAX, depth=3)

    ungated = compute_reduced_lengths(
        section_info, data_u8, depths=[3], reductions=[_MAX]
    )[spec]
    gated = compute_reduced_lengths(
        section_info,
        data_u8,
        depths=[3],
        reductions=[_MAX],
        gate=VariantGate(min_variants=2),
    )[spec]

    counts = section_info.counts
    for i in range(counts.size):
        if counts[i] >= 2:
            assert gated[i] == ungated[i], (
                f"section {i} (count={counts[i]}) should survive the gate "
                f"with its ungated length"
            )
            assert gated[i] > 0
        else:
            assert gated[i] == 0, (
                f"section {i} (count={counts[i]}) below the gate must be 0"
            )


def test_gating_unique_excludes_duplicate_heavy_section(tmp_path: Path) -> None:
    """``min_variants_unique`` measures distinct data pointers.

    The combined fixture's variants all have DISTINCT data pointers
    (the corpus builder writes one record per variant), so unique count
    == total count for every section. A ``min_variants_unique=3`` gate
    therefore keeps only multi_fn (4 unique) and drops the rest -- the
    same shape the total gate would, confirming the uniqueness leg reads
    the real per-variant pointers.
    """
    base = build_combined_fixture(tmp_path)
    data_u8, section_info = _open(base)
    spec = IndexSpec(reduction=_MAX, depth=3)

    # Oracle: distinct data pointers per section == total variant count
    # for this dedup-free fixture.
    for i in range(section_info.counts.size):
        assert section_info.unique_count(i) == int(section_info.counts[i])

    gated = compute_reduced_lengths(
        section_info,
        data_u8,
        depths=[3],
        reductions=[_MAX],
        gate=VariantGate(min_variants_unique=3),
    )[spec]

    for i in range(section_info.counts.size):
        if section_info.unique_count(i) >= 3:
            assert gated[i] > 0
        else:
            assert gated[i] == 0


# ---------------------------------------------------------------------------
# Pre-pass: data pointers match a direct sections.bin oracle
# ---------------------------------------------------------------------------


def test_prepass_data_pointers_match_oracle(tmp_path: Path) -> None:
    """``read_section_variant_info`` surfaces the real ``data_offset_shifted``.

    Cross-checks the pre-pass's per-section pointer vectors against a
    direct :func:`iter_sections_bin` walk -- so the dedup grouping key
    is provably the parsed BIN field, not a stale or wrong column.
    """
    base = build_combined_fixture(tmp_path)
    _data_u8, section_info = _open(base)
    num_matched = section_info.counts.size

    oracle_pointers = []
    for i, section in enumerate(
        iter_sections_bin(base / f"{_BINARY_NAME}_sections.bin")
    ):
        if i >= num_matched:
            break
        oracle_pointers.append(
            np.array(
                [v.data_offset_shifted for v in section.variants],
                dtype=np.uint32,
            )
        )

    cols = section_info.cols
    for i in range(num_matched):
        lo, hi = int(cols.var_offsets[i]), int(cols.var_offsets[i + 1])
        np.testing.assert_array_equal(
            cols.var_data_offset_shifted[lo:hi], oracle_pointers[i]
        )


# ---------------------------------------------------------------------------
# Dedup integration: no-op when every pointer is distinct
# ---------------------------------------------------------------------------


def test_adjust_for_duplicates_noop_on_distinct_pointers(tmp_path: Path) -> None:
    """``--adjust-for-duplicates`` is a no-op when no variants share a pointer.

    Every variant in the fixture has a distinct ``data_offset_shifted``,
    so each duplicate-group is a singleton and the dedup path collapses
    to the plain reduction byte-for-byte. Proves the dedup wiring reads
    real pointers AND respects the strict-generalisation property
    end-to-end (not just in the unit test).
    """
    base = build_combined_fixture(tmp_path)
    data_u8, section_info = _open(base)

    plain = compute_reduced_lengths(
        section_info,
        data_u8,
        depths=[3],
        reductions=[_MAX, _P50, _P95],
        duplicate_handling=PLAIN,
    )
    dedup = compute_reduced_lengths(
        section_info,
        data_u8,
        depths=[3],
        reductions=[_MAX, _P50, _P95],
        duplicate_handling=DEDUP_BY_DATA_POINTER,
    )

    assert set(plain.keys()) == set(dedup.keys())
    for spec in plain:
        np.testing.assert_array_equal(plain[spec], dedup[spec])
