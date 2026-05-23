"""Tests for stage-4 :mod:`_variant_padding` sentinel-row helpers.

Single concern: per-row interpretation of the
``batch_idx_to_section_variant`` mapping produced by stage 1. The
``compute_batch_idx_mapping`` layout itself is covered by
``test_batch_layout``; here we only assert the row-level predicates +
the section/variant resolver consume the sentinel correctly.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1Section,
    Stage1Variant,
)
from tokenizer.aligned_data.loader.batch_decode._variant_padding import (
    get_real_row_mask,
    is_padding_row,
    resolve_row_to_variant,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section


# ---------------------------------------------------------------------------
# Builders (mirroring test_types.py -- minimal but typed dummies).
# ---------------------------------------------------------------------------


def _make_section_stub(section_offset: int = 0) -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=section_offset,
        call_targets=[],
        variants=[],
    )


def _make_stage1_variant(variant_idx: int) -> Stage1Variant:
    return Stage1Variant(
        variant_idx=variant_idx,
        variant_ref_offset=variant_idx * 16,
        batch_idx=variant_idx,
        call_targets=[],
    )


def _make_stage1_section(idx: int, num_variants: int) -> Stage1Section:
    return Stage1Section(
        arm=SectionKind.MATCHED,
        idx=idx,
        section=_make_section_stub(section_offset=idx),
        variants=[_make_stage1_variant(v) for v in range(num_variants)],
    )


# ---------------------------------------------------------------------------
# is_padding_row
# ---------------------------------------------------------------------------


def test_is_padding_row_true_for_sentinel():
    """A row whose mapping equals ``(UINT32_MAX, UINT32_MAX)`` is
    classified as padding."""
    mapping = np.array(
        [[0, 0], [UINT32_MAX, UINT32_MAX]],
        dtype=np.uint32,
    )
    assert is_padding_row(mapping, 1) is True


def test_is_padding_row_false_for_real_rows():
    """Rows with any non-sentinel ``section_idx`` are real."""
    mapping = np.array(
        [[0, 0], [0, 1], [3, 2], [UINT32_MAX, UINT32_MAX]],
        dtype=np.uint32,
    )
    assert is_padding_row(mapping, 0) is False
    assert is_padding_row(mapping, 1) is False
    assert is_padding_row(mapping, 2) is False
    # Sanity: the sentinel row at index 3 is still detected.
    assert is_padding_row(mapping, 3) is True


# ---------------------------------------------------------------------------
# get_real_row_mask
# ---------------------------------------------------------------------------


def test_get_real_row_mask_mixed_batch():
    """Mask is ``True`` on real rows, ``False`` on sentinel rows."""
    mapping = np.array(
        [
            [0, 0],
            [UINT32_MAX, UINT32_MAX],
            [1, 0],
            [2, 3],
            [UINT32_MAX, UINT32_MAX],
        ],
        dtype=np.uint32,
    )
    expected = np.array([True, False, True, True, False])
    mask = get_real_row_mask(mapping)
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == bool
    # Count of real rows -- the documented use case for this helper.
    assert int(mask.sum()) == 3


def test_get_real_row_mask_empty_batch():
    """Zero-row mapping yields a zero-length mask, not an error."""
    mapping = np.empty((0, 2), dtype=np.uint32)
    mask = get_real_row_mask(mapping)
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_get_real_row_mask_all_real():
    """A dense (RAGGED / REDISTRIBUTE) mapping yields an all-True mask."""
    mapping = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.uint32)
    mask = get_real_row_mask(mapping)
    np.testing.assert_array_equal(mask, np.array([True, True, True]))


def test_get_real_row_mask_all_padding():
    """If every row is sentinel, every entry is False."""
    mapping = np.full((4, 2), UINT32_MAX, dtype=np.uint32)
    mask = get_real_row_mask(mapping)
    np.testing.assert_array_equal(mask, np.array([False, False, False, False]))


# ---------------------------------------------------------------------------
# resolve_row_to_variant
# ---------------------------------------------------------------------------


def test_resolve_row_to_variant_returns_none_for_padding():
    """Padding rows resolve to ``None`` -- callers short-circuit on this."""
    sections = [
        _make_stage1_section(idx=0, num_variants=2),
        _make_stage1_section(idx=1, num_variants=1),
    ]
    mapping = np.array(
        [[0, 0], [0, 1], [1, 0], [UINT32_MAX, UINT32_MAX]],
        dtype=np.uint32,
    )
    assert resolve_row_to_variant(mapping, sections, row=3) is None


def test_resolve_row_to_variant_returns_real_variant():
    """Real rows resolve to the section's slot-indexed
    :class:`Stage1Variant` -- identity, not a copy."""
    sections = [
        _make_stage1_section(idx=0, num_variants=2),
        _make_stage1_section(idx=1, num_variants=1),
    ]
    mapping = np.array(
        [[0, 0], [0, 1], [1, 0], [UINT32_MAX, UINT32_MAX]],
        dtype=np.uint32,
    )

    variant_0_0 = resolve_row_to_variant(mapping, sections, row=0)
    assert variant_0_0 is sections[0].variants[0]
    assert variant_0_0 is not None
    assert variant_0_0.variant_idx == 0

    variant_0_1 = resolve_row_to_variant(mapping, sections, row=1)
    assert variant_0_1 is sections[0].variants[1]
    assert variant_0_1.variant_idx == 1

    variant_1_0 = resolve_row_to_variant(mapping, sections, row=2)
    assert variant_1_0 is sections[1].variants[0]
    assert variant_1_0.variant_idx == 0


# ---------------------------------------------------------------------------
# Sentinel constant invariant
# ---------------------------------------------------------------------------


def test_sentinel_constant_matches_batch_layout():
    """``_variant_padding`` MUST share the sentinel value with
    ``_batch_layout``; the module re-uses the constant rather than
    defining its own, so identity holds."""
    from tokenizer.aligned_data.loader.batch_decode import _batch_layout
    from tokenizer.aligned_data.loader.batch_decode import _variant_padding

    assert _variant_padding.UINT32_MAX is _batch_layout.UINT32_MAX
    assert _variant_padding.UINT32_MAX == np.uint32(0xFFFFFFFF)
