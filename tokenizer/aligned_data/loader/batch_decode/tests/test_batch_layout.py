"""Tests for :func:`compute_batch_idx_mapping` -- plan ALG-10 across all four
:class:`VariantPadding` policies.

Synthetic ``ResolvedSection`` instances are built directly: the function under
test is pure (no session / no I/O), so the test can populate the dataclass
declared by ``_resolve_pointers`` (1a) without going through a session.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import VariantPadding
from tokenizer.aligned_data.loader.batch_decode._batch_layout import (
    UINT32_MAX,
    compute_batch_idx_mapping,
)
from tokenizer.aligned_data.loader.batch_decode._resolve_pointers import (
    ResolvedSection,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_section_stub(section_offset: int = 0) -> Section:
    """Build a minimal :class:`Section` -- the layout function only reads
    ``ResolvedSection.sampled_variant_indices``; the underlying
    :class:`Section` is opaque to it."""
    return Section(
        function_name_ptr=0,
        section_offset=section_offset,
        call_targets=[],
        variants=[],
    )


def _make_resolved(
    idx: int, sampled: list[int], *, arm: SectionKind = SectionKind.MATCHED
) -> ResolvedSection:
    # The layout function only reads ``sampled_variant_indices`` --
    # ``function_data_per_sampled_variant`` is opaque to it but must be
    # populated parallel-shape for the dataclass contract; ``None``
    # placeholders are fine because this test does not exercise the body
    # consumer.
    sampled_list = list(sampled)
    return ResolvedSection(
        arm=arm,
        idx=idx,
        section=_make_section_stub(section_offset=idx),
        sampled_variant_indices=sampled_list,
        function_data_per_sampled_variant=[None] * len(sampled_list),  # type: ignore[list-item]
    )


def _rng(seed: int = 0xC0FFEE) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Sentinel + dtype contract (plan ALG-10)
# ---------------------------------------------------------------------------


def test_sentinel_value():
    """Padding sentinel is ``UINT32_MAX = np.uint32(0xFFFFFFFF)`` per
    ALG-10."""
    assert UINT32_MAX == np.uint32(0xFFFFFFFF)
    assert UINT32_MAX.dtype == np.uint32


def test_dtype_and_shape_all_policies():
    """Every policy must return ``np.uint32`` with shape ``(batch_size, 2)``."""
    sections = [_make_resolved(0, [0, 1]), _make_resolved(1, [0])]
    for policy in VariantPadding:
        mapping, batch_size = compute_batch_idx_mapping(
            sections,
            num_variants_per_section=2,
            variant_padding=policy,
            rng=_rng(),
        )
        assert mapping.dtype == np.uint32, f"policy={policy}"
        assert mapping.shape == (batch_size, 2), f"policy={policy}"


# ---------------------------------------------------------------------------
# PAD_NULL
# ---------------------------------------------------------------------------


def test_pad_null_short_sections():
    """Short section in slot 1 leaves the final row at the sentinel."""
    sections = [
        _make_resolved(0, [0, 1]),
        _make_resolved(1, [0]),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.PAD_NULL,
        rng=_rng(),
    )
    assert batch_size == 4
    expected = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [UINT32_MAX, UINT32_MAX],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(mapping, expected)


def test_pad_null_exactly_full():
    """Both sections sampled at the budget -- no padding rows."""
    sections = [
        _make_resolved(0, [0, 1]),
        _make_resolved(1, [0, 1]),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.PAD_NULL,
        rng=_rng(),
    )
    assert batch_size == 4
    expected = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint32
    )
    np.testing.assert_array_equal(mapping, expected)
    # No sentinel anywhere.
    assert not (mapping == UINT32_MAX).any()


# ---------------------------------------------------------------------------
# RAGGED
# ---------------------------------------------------------------------------


def test_ragged_short_sections():
    """``batch_size == total_real_variants``; mapping is dense over the
    sampled variants only -- no padding rows."""
    sections = [
        _make_resolved(0, [0, 1]),
        _make_resolved(1, [0]),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.RAGGED,
        rng=_rng(),
    )
    assert batch_size == 3
    expected = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(mapping, expected)
    assert not (mapping == UINT32_MAX).any()


def test_ragged_with_empty_section():
    """A section with zero sampled variants contributes zero rows under
    RAGGED."""
    sections = [
        _make_resolved(0, [0, 1, 2]),
        _make_resolved(1, []),  # empty
        _make_resolved(2, [0]),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.RAGGED,
        rng=_rng(),
    )
    assert batch_size == 4
    expected = np.array(
        [[0, 0], [0, 1], [0, 2], [2, 0]], dtype=np.uint32
    )
    np.testing.assert_array_equal(mapping, expected)


# ---------------------------------------------------------------------------
# RESAMPLE_WITHIN_SECTION
# ---------------------------------------------------------------------------


def test_resample_short_section_single_variant_pool():
    """Section with 1 sampled variant: the deficit slot is resampled from
    its only available slot -- it must be ``(1, 0)``."""
    sections = [
        _make_resolved(0, [0, 1]),
        _make_resolved(1, [0]),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        rng=_rng(),
    )
    assert batch_size == 4
    assert not (mapping == UINT32_MAX).any()
    np.testing.assert_array_equal(
        mapping[:3], np.array([[0, 0], [0, 1], [1, 0]], dtype=np.uint32)
    )
    # The resampled slot must be (1, 0) -- only option in section 1's pool.
    np.testing.assert_array_equal(mapping[3], np.array([1, 0], dtype=np.uint32))


def test_resample_multi_variant_pool_reproducible():
    """With ``>1`` sampled variants in the pool, the same seed must produce
    the same resampled value."""
    sections = [
        _make_resolved(0, [0, 1, 2, 3]),
        _make_resolved(1, [0, 1, 2]),  # short by 2 slots if nv=5
    ]
    nv = 5
    m1, _ = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        rng=_rng(seed=12345),
    )
    m2, _ = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        rng=_rng(seed=12345),
    )
    np.testing.assert_array_equal(m1, m2)
    # Section 0 has 4 sampled, nv=5 -> 1 deficit; section 1 has 3 sampled,
    # nv=5 -> 2 deficits. All deficits must resolve to (s, slot_v) within the
    # section's sampled-slot pool [0, real_count[s]).
    assert not (m1 == UINT32_MAX).any()
    # Section 0 deficit slot (row index 4).
    assert m1[4, 0] == 0
    assert 0 <= m1[4, 1] < 4
    # Section 1 deficit slots (rows 8, 9).
    for row in (8, 9):
        assert m1[row, 0] == 1
        assert 0 <= m1[row, 1] < 3


def test_resample_different_seeds_likely_differ():
    """Sanity: different RNG seeds produce different deficit slots when the
    pool size > 1."""
    sections = [_make_resolved(0, [0, 1, 2, 3, 4, 5, 6, 7])]
    nv = 32
    m1, _ = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        rng=_rng(seed=1),
    )
    m2, _ = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        rng=_rng(seed=2),
    )
    # At least one deficit slot must differ across seeds (24 deficit slots,
    # 8-element pool -> collision probability ~0 in practice).
    assert not np.array_equal(m1, m2)


# ---------------------------------------------------------------------------
# REDISTRIBUTE
# ---------------------------------------------------------------------------


def test_redistribute_donor_fills_short_section():
    """Section 0 (4 sampled) donates 2 of its excess slots to section 1 (0
    sampled). Final mapping is dense; section 1's rows all hold
    ``(0, donor_slot)``."""
    sections = [
        _make_resolved(0, [0, 1, 2, 3]),
        _make_resolved(1, []),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.REDISTRIBUTE,
        rng=_rng(),
    )
    assert batch_size == 4
    assert not (mapping == UINT32_MAX).any()
    # Section 0's own rows = (0, 0) and (0, 1) (first two budget slots).
    np.testing.assert_array_equal(
        mapping[:2], np.array([[0, 0], [0, 1]], dtype=np.uint32)
    )
    # Section 1's rows are filled by donor pairs from section 0's overflow:
    # slot indices 2 and 3 (a permutation of {2, 3}).
    donated_section_ids = mapping[2:4, 0]
    donated_slot_ids = mapping[2:4, 1]
    np.testing.assert_array_equal(
        donated_section_ids, np.array([0, 0], dtype=np.uint32)
    )
    assert set(int(x) for x in donated_slot_ids) == {2, 3}


def test_redistribute_reproducibility():
    """Same RNG seed -> same donor permutation -> same mapping."""
    sections = [
        _make_resolved(0, [0, 1, 2, 3]),
        _make_resolved(1, []),
        _make_resolved(2, [0, 1, 2, 3, 4, 5]),
        _make_resolved(3, [0]),
    ]
    nv = 2
    m1, _ = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.REDISTRIBUTE,
        rng=_rng(seed=7),
    )
    m2, _ = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.REDISTRIBUTE,
        rng=_rng(seed=7),
    )
    np.testing.assert_array_equal(m1, m2)


def test_redistribute_no_donors_falls_back_to_sentinel():
    """When ALL sections are short and no donor pool exists, the layout
    cannot be made dense -- deficit slots stay at the sentinel. The caller
    is responsible for ensuring donor variants exist."""
    sections = [
        _make_resolved(0, [0]),
        _make_resolved(1, [0]),
    ]
    mapping, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=2,
        variant_padding=VariantPadding.REDISTRIBUTE,
        rng=_rng(),
    )
    assert batch_size == 4
    np.testing.assert_array_equal(mapping[0], np.array([0, 0], dtype=np.uint32))
    np.testing.assert_array_equal(mapping[2], np.array([1, 0], dtype=np.uint32))
    # Deficit slots stay sentinel.
    np.testing.assert_array_equal(
        mapping[1], np.array([UINT32_MAX, UINT32_MAX], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        mapping[3], np.array([UINT32_MAX, UINT32_MAX], dtype=np.uint32)
    )


# ---------------------------------------------------------------------------
# batch_size derivation per policy
# ---------------------------------------------------------------------------


def test_batch_size_derivation_per_policy():
    """Cross-policy ``batch_size`` arithmetic per ALG-10."""
    sections = [
        _make_resolved(0, [0, 1, 2]),  # 3 sampled
        _make_resolved(1, [0]),         # 1 sampled
        _make_resolved(2, []),          # 0 sampled
    ]
    nv = 3
    num_sections = 3

    _, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.PAD_NULL,
        rng=_rng(),
    )
    assert batch_size == num_sections * nv

    _, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        rng=_rng(),
    )
    assert batch_size == num_sections * nv

    _, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.RAGGED,
        rng=_rng(),
    )
    assert batch_size == 3 + 1 + 0  # total real variants

    _, batch_size = compute_batch_idx_mapping(
        sections,
        num_variants_per_section=nv,
        variant_padding=VariantPadding.REDISTRIBUTE,
        rng=_rng(),
    )
    assert batch_size == num_sections * nv


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_input_zero_sections():
    """Zero sections -> ``batch_size == 0`` and shape ``(0, 2)`` for every
    policy."""
    for policy in VariantPadding:
        mapping, batch_size = compute_batch_idx_mapping(
            [],
            num_variants_per_section=4,
            variant_padding=policy,
            rng=_rng(),
        )
        assert batch_size == 0, f"policy={policy}"
        assert mapping.shape == (0, 2), f"policy={policy}"
        assert mapping.dtype == np.uint32, f"policy={policy}"


# ---------------------------------------------------------------------------
# Unknown policy
# ---------------------------------------------------------------------------


def test_unknown_policy_raises():
    """Defensive: a non-:class:`VariantPadding` value raises ``ValueError``."""

    class _Fake:
        pass

    with pytest.raises(ValueError):
        compute_batch_idx_mapping(
            [_make_resolved(0, [0])],
            num_variants_per_section=1,
            variant_padding=_Fake(),  # type: ignore[arg-type]
            rng=_rng(),
        )
