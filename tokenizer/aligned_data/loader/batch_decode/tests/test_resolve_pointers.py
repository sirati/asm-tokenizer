"""Tests for the section-pointer resolution + RNG variant sampling step.

Single concern: assert that :func:`resolve_section_pointers` faithfully
dispatches per :class:`SectionKind`, preserves input order, and threads
the rng through :func:`_select_variant_indices` correctly.

We use a hand-built fake session (not :class:`BinarySession` itself --
the full session needs a corpus, vocab, and memmaps that are
out-of-scope for this 1a unit test). The fake provides only the two
private load helpers the 1a module touches, returning synthetic
:class:`Section` instances with parameterised variant counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._resolve_pointers import (
    ResolvedSection,
    resolve_section_pointers,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import (
    Section,
    VariantBlock,
)


# ---------------------------------------------------------------------------
# Fake session + Section builders
# ---------------------------------------------------------------------------


def _make_variant(ref_offset: int) -> VariantBlock:
    return VariantBlock(
        variant_ref_offset=ref_offset,
        data_offset_shifted=0,
        per_call_entries=[],
    )


def _make_section(n_variants: int, section_offset: int = 0) -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=section_offset,
        call_targets=[],
        variants=[_make_variant(ref_offset=i + 1) for i in range(n_variants)],
    )


@dataclass
class _FakeSession:
    """Minimal stand-in for :class:`BinarySession`.

    Implements only the two load helpers
    :func:`resolve_section_pointers` calls. Each map keys per-arm
    ``idx`` to the pre-built :class:`Section`. The third tuple element
    is ``None`` for both arms -- the 1a module discards it.
    """

    matched_sections: dict
    unmatched_sections: dict

    def _load_matched_section_and_variants(
        self, idx: int
    ) -> Tuple[Section, int, object]:
        section = self.matched_sections[idx]
        return section, section.section_offset, None

    def _load_unmatched_record_and_section(
        self, idx: int
    ) -> Tuple[Section, int, object]:
        section = self.unmatched_sections[idx]
        return section, section.section_offset, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolution_smoke_matched_and_unmatched():
    """One matched + one unmatched pointer round-trip into two
    :class:`ResolvedSection`s with the right arms + indices."""
    matched_section = _make_section(n_variants=3, section_offset=64)
    unmatched_section = _make_section(n_variants=1, section_offset=128)
    session = _FakeSession(
        matched_sections={5: matched_section},
        unmatched_sections={2: unmatched_section},
    )
    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=5),
        SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=2),
    ]

    result = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=10,
        rng=np.random.default_rng(0),
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, ResolvedSection) for r in result)

    assert result[0].arm is SectionKind.MATCHED
    assert result[0].idx == 5
    assert result[0].section is matched_section

    assert result[1].arm is SectionKind.UNMATCHED
    assert result[1].idx == 2
    assert result[1].section is unmatched_section


def test_rng_reproducibility_same_seed_same_samples():
    """Re-running with a freshly-seeded rng must yield identical
    sampled indices. (``_select_variant_indices`` already has the
    reproducibility contract; this guards against 1a accidentally
    consuming rng state in a way that breaks downstream stability.)"""
    section = _make_section(n_variants=10)
    session = _FakeSession(matched_sections={0: section}, unmatched_sections={})
    pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)]

    result_a = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=3,
        rng=np.random.default_rng(42),
    )
    result_b = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=3,
        rng=np.random.default_rng(42),
    )

    assert result_a[0].sampled_variant_indices == result_b[0].sampled_variant_indices
    # Sanity: the sample is meaningfully a sample (not the trivial
    # ``range(10)`` cover-everything path).
    assert len(result_a[0].sampled_variant_indices) == 3


def test_full_coverage_returns_every_index_in_order():
    """When ``num_variants_per_section >= n_variants`` every variant
    index is returned in encounter order -- no oversample, no drop."""
    section = _make_section(n_variants=4)
    session = _FakeSession(matched_sections={0: section}, unmatched_sections={})
    pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)]

    # Exact match.
    result_exact = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=4,
        rng=np.random.default_rng(7),
    )
    assert result_exact[0].sampled_variant_indices == [0, 1, 2, 3]

    # Over-request -- still no duplication.
    result_over = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=100,
        rng=np.random.default_rng(7),
    )
    assert result_over[0].sampled_variant_indices == [0, 1, 2, 3]


def test_undersample_returns_distinct_indices_of_requested_count():
    """When ``num_variants_per_section < n_variants`` exactly that many
    DISTINCT indices are returned (``_select_variant_indices`` samples
    without replacement)."""
    section = _make_section(n_variants=8)
    session = _FakeSession(matched_sections={0: section}, unmatched_sections={})
    pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)]

    result = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=3,
        rng=np.random.default_rng(12345),
    )
    indices = result[0].sampled_variant_indices
    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert all(0 <= i < 8 for i in indices)
    # ``_select_variant_indices`` sorts the chosen subset for
    # determinism; the 1a output must preserve that order.
    assert indices == sorted(indices)


def test_unmatched_arm_has_at_most_one_variant():
    """Unmatched sections have exactly one variant by the
    matched_sections_bin invariant; sampling -- regardless of the
    ``num_variants_per_section`` request -- returns a 1-element list."""
    unmatched_section = _make_section(n_variants=1, section_offset=32)
    session = _FakeSession(
        matched_sections={},
        unmatched_sections={0: unmatched_section},
    )
    pointers = [SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=0)]

    # Request several variants -- bound clamps to the section's 1.
    result_over = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=5,
        rng=np.random.default_rng(0),
    )
    assert result_over[0].sampled_variant_indices == [0]

    # Request exactly one -- same result.
    result_exact = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=1,
        rng=np.random.default_rng(0),
    )
    assert result_exact[0].sampled_variant_indices == [0]


def test_order_preservation_across_mixed_arms():
    """``result[i]`` must correspond to ``section_pointers[i]`` for
    every ``i``, regardless of arm ordering."""
    sec_m0 = _make_section(n_variants=2, section_offset=8)
    sec_m1 = _make_section(n_variants=5, section_offset=16)
    sec_u0 = _make_section(n_variants=1, section_offset=24)
    sec_u1 = _make_section(n_variants=1, section_offset=40)
    session = _FakeSession(
        matched_sections={3: sec_m0, 7: sec_m1},
        unmatched_sections={1: sec_u0, 9: sec_u1},
    )
    pointers = [
        SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=9),
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=7),
        SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=1),
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=3),
    ]

    result = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=2,
        rng=np.random.default_rng(2),
    )

    assert len(result) == len(pointers)
    expected_arms_idxs = [(p.arm, p.idx) for p in pointers]
    actual_arms_idxs = [(r.arm, r.idx) for r in result]
    assert actual_arms_idxs == expected_arms_idxs

    # Spot-check the section objects line up too.
    assert result[0].section is sec_u1
    assert result[1].section is sec_m1
    assert result[2].section is sec_u0
    assert result[3].section is sec_m0


def test_sampled_indices_are_python_ints():
    """Downstream consumers index plain Python lists
    (``MatchedFunction.variants``) with the sampled values, so the 1a
    module must hand them out as Python ints -- not ``numpy.int64`` --
    so callers do not silently fall through to numpy-typed semantics."""
    section = _make_section(n_variants=6)
    session = _FakeSession(matched_sections={0: section}, unmatched_sections={})
    pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)]

    result = resolve_section_pointers(
        session,  # type: ignore[arg-type]
        pointers,
        num_variants_per_section=2,
        rng=np.random.default_rng(99),
    )
    for v in result[0].sampled_variant_indices:
        assert type(v) is int


def test_resolved_section_is_frozen():
    """The handoff dataclass must be immutable -- consistent with the
    rest of the batch_decode dataclass backbone."""
    section = _make_section(n_variants=1)
    rs = ResolvedSection(
        arm=SectionKind.MATCHED,
        idx=0,
        section=section,
        sampled_variant_indices=[0],
    )
    with pytest.raises(Exception):
        # ``FrozenInstanceError`` is a subclass of ``AttributeError``;
        # accept any failure that prevents mutation.
        rs.idx = 5  # type: ignore[misc]
