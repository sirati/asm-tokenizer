"""Unit tests for :func:`assemble_tokens` -- stage-4 per-row token-tensor
assembly + truncation at ``context_len``.

Single concern: validate that ``assemble_tokens`` correctly walks
``Stage3Batch.stage2.stage1.batch_idx_to_section_variant`` and writes
each variant's call_target ``expanded_token_ids[:partial_cut_length]``
slices into the right row at the running column offset, with the right
truncation + padding semantics.

The function is pure over its inputs (no session, no I/O). We build
synthetic 4-level hierarchies directly via the
:class:`Stage1Batch` / :class:`Stage2Batch` / :class:`Stage3Batch`
constructors -- only the fields ``assemble_tokens`` actually reads need
realistic content; the rest are stubbed with whatever satisfies the
frozen-dataclass interface.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._token_assembly import (
    assemble_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
    Stage2Batch,
    Stage2CallTarget,
    Stage2Section,
    Stage2Variant,
    Stage3Batch,
    Stage3CallTarget,
    Stage3Section,
    Stage3Variant,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Stub builders -- the minimum a Stage{1,2,3}* needs to satisfy the frozen
# dataclass shape while only the fields read by ``assemble_tokens`` carry
# meaningful values.
# ---------------------------------------------------------------------------


class _NullState:
    """Stand-in for :class:`InlineDecodeState`.

    ``assemble_tokens`` never reads ``Stage1CallTarget.state``; we only
    need *something* assignable to the field. The real class is a frozen
    dataclass whose construction requires several aligned numpy arrays;
    a tiny duck-typed stub keeps the per-test setup minimal."""

    pass


def _function_data(name: str = "f") -> FunctionData:
    """Minimal :class:`FunctionData` -- ``assemble_tokens`` doesn't read
    any of these fields, but the frozen dataclass requires the slot."""
    return FunctionData(
        func_name=name,
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _section_stub(section_offset: int = 0) -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=section_offset,
        call_targets=[],
        variants=[],
    )


def _u16(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint16)


def _make_call_target(
    *,
    expanded_token_ids: np.ndarray,
    partial_cut_length: Optional[int] = None,
    is_cut: bool = False,
) -> Stage3CallTarget:
    """Build a Stage3CallTarget whose only used fields are
    ``expanded_token_ids`` + ``partial_cut_length`` + ``predicted_full_length``
    (the latter is read only by the assertion error message).
    """

    if partial_cut_length is None:
        partial_cut_length = int(expanded_token_ids.shape[0])

    stage1_ct = Stage1CallTarget(
        function_data=_function_data(),
        state=_NullState(),  # type: ignore[arg-type]
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=0,
    )
    stage2_ct = Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded_token_ids,
        extra_value_v2_mask=np.zeros(
            expanded_token_ids.shape[0], dtype=bool
        ),
        extra_f128_mask=np.zeros(
            expanded_token_ids.shape[0], dtype=bool
        ),
        predicted_full_length=int(expanded_token_ids.shape[0]),
        surviving_token_count=partial_cut_length,
        surviving_identity_count=0,
        surviving_number_chunk_count=0,
        is_cut=is_cut,
        partial_cut_length=partial_cut_length,
    )
    return Stage3CallTarget(
        stage2=stage2_ct,
        inline_byte_slice=slice(0, 0),
        identity_slice=slice(0, 0),
        number_chunk_slices={},
    )


def _make_batch(
    *,
    variants_per_section: List[List[List[Stage3CallTarget]]],
    mapping: np.ndarray,
    batch_idx_overrides: Optional[
        List[List[Optional[int]]]
    ] = None,
) -> Stage3Batch:
    """Assemble a full 4-level hierarchy from per-section variant
    call_target lists + an explicit ``batch_idx_to_section_variant``
    mapping.

    ``variants_per_section[s][v]`` = list of Stage3CallTargets for
    section ``s``, variant slot ``v``. ``batch_idx_overrides[s][v]``
    (when provided) overrides the otherwise-derived
    ``Stage1Variant.batch_idx`` -- used for the "batch_idx is None"
    padding-out case.
    """

    num_sections = len(variants_per_section)
    batch_size = int(mapping.shape[0])

    # First: derive a default batch_idx per variant by inverting the
    # mapping (each (section, slot) -> the first row where it appears).
    derived_batch_idx: List[List[Optional[int]]] = [
        [None] * len(variants_per_section[s])
        for s in range(num_sections)
    ]
    sentinel = int(UINT32_MAX)
    for row in range(batch_size):
        s = int(mapping[row, 0])
        v = int(mapping[row, 1])
        if s == sentinel or v == sentinel:
            continue
        if derived_batch_idx[s][v] is None:
            derived_batch_idx[s][v] = row

    if batch_idx_overrides is not None:
        for s in range(num_sections):
            for v in range(len(derived_batch_idx[s])):
                derived_batch_idx[s][v] = batch_idx_overrides[s][v]

    # Build stage1 hierarchy.
    stage1_sections: List[Stage1Section] = []
    stage2_sections: List[Stage2Section] = []
    stage3_sections: List[Stage3Section] = []

    for s, variant_list in enumerate(variants_per_section):
        stage1_variants: List[Stage1Variant] = []
        stage2_variants: List[Stage2Variant] = []
        stage3_variants: List[Stage3Variant] = []

        for v, ct_list in enumerate(variant_list):
            stage1_cts = [ct.stage2.stage1 for ct in ct_list]
            stage2_cts = [ct.stage2 for ct in ct_list]
            stage1_variants.append(
                Stage1Variant(
                    variant_idx=v,
                    variant_ref_offset=0,
                    batch_idx=derived_batch_idx[s][v],
                    call_targets=stage1_cts,
                )
            )
            stage2_variants.append(
                Stage2Variant(
                    stage1=stage1_variants[-1],
                    call_targets=stage2_cts,
                    cut_call_target_index=len(ct_list),
                    total_surviving_token_count=sum(
                        ct.stage2.partial_cut_length for ct in ct_list
                    ),
                    total_surviving_identity_count=0,
                    total_surviving_number_chunk_count=0,
                )
            )
            stage3_variants.append(
                Stage3Variant(
                    stage2=stage2_variants[-1],
                    call_targets=ct_list,
                )
            )

        stage1_sections.append(
            Stage1Section(
                arm=SectionKind.MATCHED,
                idx=s,
                section=_section_stub(section_offset=s),
                variants=stage1_variants,
            )
        )
        stage2_sections.append(
            Stage2Section(
                stage1=stage1_sections[-1],
                variants=stage2_variants,
            )
        )
        stage3_sections.append(
            Stage3Section(
                stage2=stage2_sections[-1],
                variants=stage3_variants,
            )
        )

    stage1_batch = Stage1Batch(
        sections=stage1_sections,
        batch_idx_to_section_variant=mapping.astype(np.uint32),
        batch_size=batch_size,
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=stage2_sections,
        identity_row_offsets=np.zeros(batch_size + 1, dtype=np.uint32),
        number_row_offsets=np.zeros(batch_size + 1, dtype=np.uint32),
    )
    return Stage3Batch(
        stage2=stage2_batch,
        sections=stage3_sections,
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=np.zeros(0, dtype=np.uint16),
        numbers_per_TokenType={},
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
    )


# ---------------------------------------------------------------------------
# Dtype + shape invariants
# ---------------------------------------------------------------------------


def test_dtype_and_shape() -> None:
    """Output must be ``u16`` with shape ``(batch_size, context_len)`` --
    plan D5's null-content slot is id 0 and only fits ``u16``."""
    ct = _make_call_target(expanded_token_ids=_u16([100, 200, 300]))
    batch = _make_batch(
        variants_per_section=[[[ct]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=8)

    assert out.dtype == np.uint16
    assert out.shape == (1, 8)


def test_empty_batch_zero_rows() -> None:
    """Zero rows -> zero-shaped output, no errors."""
    batch = _make_batch(
        variants_per_section=[],
        mapping=np.empty((0, 2), dtype=np.uint32),
    )
    out = assemble_tokens(batch, context_len=8)
    assert out.shape == (0, 8)
    assert out.dtype == np.uint16


def test_zero_context_len() -> None:
    """``context_len == 0`` -> zero columns, no writes attempted."""
    ct = _make_call_target(expanded_token_ids=_u16([100, 200, 300]))
    batch = _make_batch(
        variants_per_section=[[[ct]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )
    out = assemble_tokens(batch, context_len=0)
    assert out.shape == (1, 0)


# ---------------------------------------------------------------------------
# 1. Single CT fits under context_len
# ---------------------------------------------------------------------------


def test_single_ct_fits_under_context_len() -> None:
    """One call_target whose ``partial_cut_length`` is below
    ``context_len`` -> row is the verbatim ``expanded_token_ids``
    followed by zeros."""

    expanded = _u16([257, 264, 300, 301])  # 4 tokens
    ct = _make_call_target(expanded_token_ids=expanded)
    batch = _make_batch(
        variants_per_section=[[[ct]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=8)

    expected = np.array(
        [[257, 264, 300, 301, 0, 0, 0, 0]], dtype=np.uint16
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 2. Multi-CT chain -> concatenated correctly
# ---------------------------------------------------------------------------


def test_multi_ct_chain_concatenated() -> None:
    """Root + 2 callees: each call_target's full
    ``expanded_token_ids[:partial_cut_length]`` is concatenated in
    encounter order. The prepend (at position 0 of each
    ``expanded_token_ids``) lands at the running column offset."""

    # Each CT's stream starts with its own "prepend self-token" by
    # convention (plan ALG-9; 2a's output already includes it at
    # position 0). We pick distinct sentinel ids per CT to confirm the
    # write head advances per-CT.
    root = _make_call_target(
        expanded_token_ids=_u16([9, 100, 101])  # prepend=9 (LOCAL_FUNC), then 100, 101
    )
    callee_a = _make_call_target(
        expanded_token_ids=_u16([9, 200, 201])  # prepend=9, then 200, 201
    )
    callee_b = _make_call_target(
        expanded_token_ids=_u16([10, 300])  # prepend=10 (PLT_FUNC), then 300
    )
    batch = _make_batch(
        variants_per_section=[[[root, callee_a, callee_b]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=16)

    expected_row = np.array(
        [9, 100, 101, 9, 200, 201, 10, 300, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(out[0], expected_row)


# ---------------------------------------------------------------------------
# 3. Truncation at context_len -> exactly context_len tokens
# ---------------------------------------------------------------------------


def test_truncation_at_context_len() -> None:
    """When the sum of ``partial_cut_length`` over CTs reaches exactly
    ``context_len``, no padding tail remains and no overflow occurs."""

    # Two CTs each with 4 tokens; context_len = 8 -> exact fit.
    ct_a = _make_call_target(expanded_token_ids=_u16([10, 11, 12, 13]))
    ct_b = _make_call_target(expanded_token_ids=_u16([20, 21, 22, 23]))
    batch = _make_batch(
        variants_per_section=[[[ct_a, ct_b]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=8)

    np.testing.assert_array_equal(
        out[0],
        np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.uint16),
    )


def test_defensive_cap_when_partial_cut_length_overflows() -> None:
    """Defensive cap: if upstream accounting drifted such that the sum
    of ``partial_cut_length`` exceeds ``context_len``, the secondary
    cap inside ``assemble_tokens`` clips the last write to fit. This
    is the belt-and-braces guard documented in the docstring."""

    # CT1: 6 tokens, fully included; CT2: would write 5 more but only
    # 4 columns remain (context_len = 10).
    ct1 = _make_call_target(
        expanded_token_ids=_u16([1, 2, 3, 4, 5, 6])
    )
    ct2 = _make_call_target(
        expanded_token_ids=_u16([10, 20, 30, 40, 50])
    )
    batch = _make_batch(
        variants_per_section=[[[ct1, ct2]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=10)

    # First 6 cols = CT1; next 4 cols = CT2's first 4 tokens; the 5th
    # CT2 token is dropped by the defensive cap.
    np.testing.assert_array_equal(
        out[0],
        np.array(
            [1, 2, 3, 4, 5, 6, 10, 20, 30, 40], dtype=np.uint16
        ),
    )


# ---------------------------------------------------------------------------
# 4. Mid-CT cut -> only partial_cut_length subset present
# ---------------------------------------------------------------------------


def test_mid_ct_cut_drops_subsequent_cts() -> None:
    """Cut in the middle of the 2nd CT -> the cut CT contributes only
    its ``partial_cut_length`` prefix; subsequent (post-cut) CTs have
    ``partial_cut_length == 0`` and contribute nothing."""

    ct_root = _make_call_target(
        expanded_token_ids=_u16([100, 101, 102, 103])  # 4 tokens, all included
    )
    # Cut CT: full length = 5, but cut at 3.
    ct_cut = _make_call_target(
        expanded_token_ids=_u16([200, 201, 202, 203, 204]),
        partial_cut_length=3,
        is_cut=True,
    )
    # Post-cut CT: dropped.
    ct_dropped = _make_call_target(
        expanded_token_ids=_u16([300, 301, 302]),
        partial_cut_length=0,
    )

    batch = _make_batch(
        variants_per_section=[[[ct_root, ct_cut, ct_dropped]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=16)

    expected = np.zeros(16, dtype=np.uint16)
    expected[0:4] = [100, 101, 102, 103]
    expected[4:7] = [200, 201, 202]
    # Cols 7..15 stay at zero (no subsequent CT content; null-content
    # tail).
    np.testing.assert_array_equal(out[0], expected)


# ---------------------------------------------------------------------------
# 5. PAD_NULL padding row (UINT32_MAX sentinel) -> row stays zeros
# ---------------------------------------------------------------------------


def test_pad_null_padding_row_stays_zero() -> None:
    """A row whose mapping entry is ``(UINT32_MAX, UINT32_MAX)`` must
    stay at id 0 -- the null-content padding slot per plan D5."""

    ct = _make_call_target(expanded_token_ids=_u16([100, 101, 102]))
    sentinel = int(UINT32_MAX)
    mapping = np.array(
        [
            [0, 0],
            [sentinel, sentinel],
        ],
        dtype=np.uint32,
    )

    batch = _make_batch(
        variants_per_section=[[[ct]]],
        mapping=mapping,
    )

    out = assemble_tokens(batch, context_len=8)

    # Row 0: real content + zero tail.
    np.testing.assert_array_equal(
        out[0],
        np.array([100, 101, 102, 0, 0, 0, 0, 0], dtype=np.uint16),
    )
    # Row 1: padding -> all zeros.
    np.testing.assert_array_equal(
        out[1], np.zeros(8, dtype=np.uint16)
    )


def test_batch_idx_none_padding_out_stays_zero() -> None:
    """A row whose mapping resolves to a Stage1Variant with
    ``batch_idx is None`` (RAGGED's post-cutoff drop case) must also
    stay at id 0 -- belt-and-braces guard documented in the module
    docstring."""

    ct = _make_call_target(expanded_token_ids=_u16([42, 43, 44]))
    mapping = np.array(
        [
            [0, 0],
            [0, 1],  # second slot is the "padding-out" variant
        ],
        dtype=np.uint32,
    )

    # Two variants on section 0; slot 1's batch_idx is None.
    other_ct = _make_call_target(
        expanded_token_ids=_u16([900, 901, 902])
    )
    batch = _make_batch(
        variants_per_section=[[[ct], [other_ct]]],
        mapping=mapping,
        batch_idx_overrides=[[0, None]],
    )

    out = assemble_tokens(batch, context_len=8)
    np.testing.assert_array_equal(
        out[0],
        np.array([42, 43, 44, 0, 0, 0, 0, 0], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        out[1], np.zeros(8, dtype=np.uint16)
    )


# ---------------------------------------------------------------------------
# 6. RESAMPLE multi-mapped -> both rows have identical content
# ---------------------------------------------------------------------------


def test_resample_multi_mapped_rows_have_identical_content() -> None:
    """RESAMPLE_WITHIN_SECTION can map the same ``(section, slot)`` to
    multiple batch rows. Each row gets the same concatenated content."""

    ct = _make_call_target(expanded_token_ids=_u16([777, 778, 779, 780]))
    # Mapping: rows 0 and 1 both point to (section=0, slot=0); row 2
    # is the sentinel.
    sentinel = int(UINT32_MAX)
    mapping = np.array(
        [
            [0, 0],
            [0, 0],
            [sentinel, sentinel],
        ],
        dtype=np.uint32,
    )
    batch = _make_batch(
        variants_per_section=[[[ct]]],
        mapping=mapping,
    )

    out = assemble_tokens(batch, context_len=8)

    expected_real = np.array(
        [777, 778, 779, 780, 0, 0, 0, 0], dtype=np.uint16
    )
    np.testing.assert_array_equal(out[0], expected_real)
    np.testing.assert_array_equal(out[1], expected_real)
    np.testing.assert_array_equal(out[2], np.zeros(8, dtype=np.uint16))


# ---------------------------------------------------------------------------
# 7. Empty variant (zero CTs) -> row stays all zeros
# ---------------------------------------------------------------------------


def test_empty_variant_zero_call_targets() -> None:
    """A variant with no ``call_targets`` contributes no writes; the
    row stays at zeros (identical to padding-row behavior)."""

    batch = _make_batch(
        variants_per_section=[[[]]],  # one section, one variant, zero CTs
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=8)
    np.testing.assert_array_equal(out[0], np.zeros(8, dtype=np.uint16))


def test_variant_all_zero_partial_cut_length() -> None:
    """A variant whose CTs all have ``partial_cut_length == 0`` (e.g.
    every CT dropped post-cut) contributes nothing -- equivalent to the
    empty-variant case."""

    ct_dropped_a = _make_call_target(
        expanded_token_ids=_u16([100, 101]), partial_cut_length=0
    )
    ct_dropped_b = _make_call_target(
        expanded_token_ids=_u16([200, 201]), partial_cut_length=0
    )
    batch = _make_batch(
        variants_per_section=[[[ct_dropped_a, ct_dropped_b]]],
        mapping=np.array([[0, 0]], dtype=np.uint32),
    )

    out = assemble_tokens(batch, context_len=8)
    np.testing.assert_array_equal(out[0], np.zeros(8, dtype=np.uint16))


# ---------------------------------------------------------------------------
# 8. Multi-row + multi-section sanity
# ---------------------------------------------------------------------------


def test_multi_section_independent_rows() -> None:
    """Two sections each with one variant; rows are written
    independently with no cross-talk."""

    ct_sec0 = _make_call_target(expanded_token_ids=_u16([1, 2, 3]))
    ct_sec1 = _make_call_target(
        expanded_token_ids=_u16([10, 20, 30, 40])
    )
    mapping = np.array(
        [
            [0, 0],
            [1, 0],
        ],
        dtype=np.uint32,
    )
    batch = _make_batch(
        variants_per_section=[[[ct_sec0]], [[ct_sec1]]],
        mapping=mapping,
    )

    out = assemble_tokens(batch, context_len=6)

    np.testing.assert_array_equal(
        out[0],
        np.array([1, 2, 3, 0, 0, 0], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        out[1],
        np.array([10, 20, 30, 40, 0, 0], dtype=np.uint16),
    )


def test_first_row_is_padding_second_is_real() -> None:
    """Robustness: padding-then-real ordering must still leave row 1
    correctly assembled."""

    ct = _make_call_target(expanded_token_ids=_u16([55, 66, 77]))
    sentinel = int(UINT32_MAX)
    mapping = np.array(
        [
            [sentinel, sentinel],
            [0, 0],
        ],
        dtype=np.uint32,
    )
    batch = _make_batch(
        variants_per_section=[[[ct]]],
        mapping=mapping,
    )

    out = assemble_tokens(batch, context_len=5)
    np.testing.assert_array_equal(
        out[0], np.zeros(5, dtype=np.uint16)
    )
    np.testing.assert_array_equal(
        out[1], np.array([55, 66, 77, 0, 0], dtype=np.uint16)
    )
