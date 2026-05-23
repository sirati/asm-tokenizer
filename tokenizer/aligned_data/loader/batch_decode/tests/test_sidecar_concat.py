"""Tests for :func:`assemble_number_sidecars`.

Single concern of this file: pin the per-row numbers sidecar concat
contract in isolation. Stage 1/2/3 builders construct minimum-viable
hierarchies (real upstream stages are stubs at the time this module
ships) so the sidecar logic is exercised on hand-crafted
``Stage3Batch`` instances.

Plan reference: ``batch_decode_plan.md`` Stage 4 step 4; D8 row-offset
sizing rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._sidecar_concat import (
    assemble_number_sidecars,
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
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Shifted-id constants (post-shift NUMBER block -- see _sidecar_concat docs)
# ---------------------------------------------------------------------------


_VC2_SHIFTED = 1
_F16_SHIFTED = 2
_BF16_SHIFTED = 3
_F32_SHIFTED = 4
_F64_SHIFTED = 5
_F80_SHIFTED = 6
_F128_SHIFTED = 7

# Identity-band id used purely to interleave non-number tokens in
# scenarios that want a realistic stream (post-shift identity block is
# 8..15). The sidecar logic only branches on the number band; any
# out-of-band id is treated as "not a number-chunk position".
_LOCAL_FUNC_SHIFTED = 9


# ---------------------------------------------------------------------------
# Minimal builders -- one-knob helpers so individual tests stay focused
# ---------------------------------------------------------------------------


def _u16(seq) -> np.ndarray:
    return np.asarray(seq, dtype=np.uint16)


def _make_function_data() -> FunctionData:
    return FunctionData(
        func_name="dummy_func",
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _make_inline_decode_state() -> InlineDecodeState:
    return InlineDecodeState(
        raw_tokens=np.zeros(0, dtype=np.uint16),
        real_mask=np.zeros(0, dtype=bool),
        number_mask=np.zeros(0, dtype=bool),
        runlen_number=np.zeros(0, dtype=np.uint32),
        runlen_value=np.zeros(0, dtype=np.uint32),
        carries_inline_mask=np.zeros(0, dtype=bool),
        is_negative_per_position=np.zeros(0, dtype=bool),
        digit_cumsum=np.zeros(1, dtype=np.uint32),
    )


def _make_section() -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _make_stage1_call_target(function_name_ptr: int = 0) -> Stage1CallTarget:
    return Stage1CallTarget(
        function_data=_make_function_data(),
        state=_make_inline_decode_state(),
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=function_name_ptr,
    )


def _make_stage2_call_target(
    *,
    expanded_token_ids: np.ndarray,
    partial_cut_length: int | None = None,
    function_name_ptr: int = 0,
) -> Stage2CallTarget:
    if partial_cut_length is None:
        partial_cut_length = int(expanded_token_ids.shape[0])
    surviving = expanded_token_ids[:partial_cut_length]
    number_mask = (surviving >= 1) & (surviving < 8)
    identity_mask = (surviving >= 8) & (surviving < 16)
    return Stage2CallTarget(
        stage1=_make_stage1_call_target(function_name_ptr=function_name_ptr),
        expanded_token_ids=expanded_token_ids,
        extra_value_v2_mask=np.zeros(expanded_token_ids.shape[0], dtype=bool),
        extra_f128_mask=np.zeros(expanded_token_ids.shape[0], dtype=bool),
        predicted_full_length=int(expanded_token_ids.shape[0]),
        surviving_token_count=partial_cut_length,
        surviving_identity_count=int(identity_mask.sum()),
        surviving_number_chunk_count=int(number_mask.sum()),
        is_cut=partial_cut_length < int(expanded_token_ids.shape[0]),
        partial_cut_length=partial_cut_length,
    )


def _make_stage3_call_target(
    *,
    stage2_ct: Stage2CallTarget,
    number_chunk_slices: dict[TokenType, slice],
) -> Stage3CallTarget:
    return Stage3CallTarget(
        stage2=stage2_ct,
        inline_byte_slice=slice(0, 0),
        identity_slice=slice(0, 0),
        number_chunk_slices=number_chunk_slices,
    )


def _wrap_stage3_batch(
    *,
    variants_per_section: list[list[list[Stage3CallTarget]]],
    batch_idx_to_section_variant: np.ndarray,
    number_row_offsets: np.ndarray,
    numbers_per_TokenType: dict[TokenType, tuple[np.ndarray, np.ndarray]],
) -> Stage3Batch:
    """Wrap a 3-level list of Stage3CallTargets into a full Stage3Batch.

    ``variants_per_section[s][v]`` is the list of level-4
    Stage3CallTargets for section ``s``, variant ``v``.
    """
    batch_size = int(batch_idx_to_section_variant.shape[0])

    stage1_sections: list[Stage1Section] = []
    stage2_sections: list[Stage2Section] = []
    stage3_sections: list[Stage3Section] = []

    for section_idx, variants in enumerate(variants_per_section):
        stage1_variants: list[Stage1Variant] = []
        stage2_variants: list[Stage2Variant] = []
        stage3_variants: list[Stage3Variant] = []
        for variant_idx, call_targets in enumerate(variants):
            stage2_cts = [ct.stage2 for ct in call_targets]
            stage1_cts = [s2.stage1 for s2 in stage2_cts]
            stage1_variant = Stage1Variant(
                variant_idx=variant_idx,
                variant_ref_offset=0,
                batch_idx=None,  # not used by sidecar concat
                call_targets=stage1_cts,
            )
            stage2_variant = Stage2Variant(
                stage1=stage1_variant,
                call_targets=stage2_cts,
                cut_call_target_index=len(stage2_cts),
                total_surviving_token_count=sum(
                    s2.surviving_token_count for s2 in stage2_cts
                ),
                total_surviving_identity_count=sum(
                    s2.surviving_identity_count for s2 in stage2_cts
                ),
                total_surviving_number_chunk_count=sum(
                    s2.surviving_number_chunk_count for s2 in stage2_cts
                ),
            )
            stage3_variant = Stage3Variant(
                stage2=stage2_variant,
                call_targets=call_targets,
            )
            stage1_variants.append(stage1_variant)
            stage2_variants.append(stage2_variant)
            stage3_variants.append(stage3_variant)
        stage1_section = Stage1Section(
            arm=SectionKind.MATCHED,
            idx=section_idx,
            section=_make_section(),
            variants=stage1_variants,
        )
        stage2_section = Stage2Section(
            stage1=stage1_section,
            variants=stage2_variants,
        )
        stage3_section = Stage3Section(
            stage2=stage2_section,
            variants=stage3_variants,
        )
        stage1_sections.append(stage1_section)
        stage2_sections.append(stage2_section)
        stage3_sections.append(stage3_section)

    stage1_batch = Stage1Batch(
        sections=stage1_sections,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=stage2_sections,
        identity_row_offsets=np.zeros(batch_size + 1, dtype=np.uint32),
        number_row_offsets=number_row_offsets,
    )
    return Stage3Batch(
        stage2=stage2_batch,
        sections=stage3_sections,
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=np.zeros(0, dtype=np.uint16),
        numbers_per_TokenType=numbers_per_TokenType,
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        f128_is_nan_or_inf=np.zeros(0, dtype=np.bool_),
        f128_visible_chunks=np.zeros(0, dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Test 1: Single-row F32-only.
# ---------------------------------------------------------------------------


def test_single_row_f32_only():
    """Lengths match ``number_row_offsets`` diff; chunks land in order."""
    # Three F32 number positions interleaved with two identity tokens.
    expanded = _u16([_F32_SHIFTED, _LOCAL_FUNC_SHIFTED, _F32_SHIFTED, _F32_SHIFTED])
    stage2_ct = _make_stage2_call_target(expanded_token_ids=expanded)
    f32_sig = np.array([0xAAAA, 0xBBBB, 0xCCCC], dtype=np.uint64)
    f32_sex = np.array([10, 20, 30], dtype=np.uint32)
    stage3_ct = _make_stage3_call_target(
        stage2_ct=stage2_ct,
        number_chunk_slices={TokenType.FLOAT32: slice(0, 3)},
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[stage3_ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 3], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT32: (f32_sig, f32_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    assert sig.dtype == np.uint64
    assert sex.dtype == np.uint32
    assert sig.shape == (3,)
    assert sex.shape == (3,)
    np.testing.assert_array_equal(sig, f32_sig)
    np.testing.assert_array_equal(sex, f32_sex)


# ---------------------------------------------------------------------------
# Test 2: Multi-TokenType mixed row -- chunks emitted in stream order,
# NOT grouped by TokenType.
# ---------------------------------------------------------------------------


def test_multi_token_type_stream_order():
    """Stream-position order is preserved across :class:`TokenType` boundaries."""
    # Stream: F32 F64 F32 F64 F32 -- three F32s + two F64s interleaved.
    expanded = _u16([_F32_SHIFTED, _F64_SHIFTED, _F32_SHIFTED, _F64_SHIFTED, _F32_SHIFTED])
    stage2_ct = _make_stage2_call_target(expanded_token_ids=expanded)
    f32_sig = np.array([100, 200, 300], dtype=np.uint64)
    f32_sex = np.array([1, 2, 3], dtype=np.uint32)
    f64_sig = np.array([1000, 2000], dtype=np.uint64)
    f64_sex = np.array([10, 20], dtype=np.uint32)
    stage3_ct = _make_stage3_call_target(
        stage2_ct=stage2_ct,
        number_chunk_slices={
            TokenType.FLOAT32: slice(0, 3),
            TokenType.FLOAT64: slice(0, 2),
        },
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[stage3_ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 5], dtype=np.uint32),
        numbers_per_TokenType={
            TokenType.FLOAT32: (f32_sig, f32_sex),
            TokenType.FLOAT64: (f64_sig, f64_sex),
        },
    )
    sig, sex = assemble_number_sidecars(stage3)
    # Stream order: F32[0], F64[0], F32[1], F64[1], F32[2]
    np.testing.assert_array_equal(sig, [100, 1000, 200, 2000, 300])
    np.testing.assert_array_equal(sex, [1, 10, 2, 20, 3])


# ---------------------------------------------------------------------------
# Test 3: VC2 multi-chunk source -- 3 entries emitted per chunk.
# ---------------------------------------------------------------------------


def test_vc2_three_chunks_per_source():
    """A VC2 source whose runlength produced K=3 chunks shows as 3 stream
    positions in ``expanded_token_ids`` (per Stage 2 promotion ALG); the
    sidecar emits 3 chunks in stream order."""
    # Pretend the call_target has ONE VC2 source promoted to 3 chunks
    # (positions 0, 1, 2 in expanded_token_ids).
    expanded = _u16([_VC2_SHIFTED, _VC2_SHIFTED, _VC2_SHIFTED])
    stage2_ct = _make_stage2_call_target(expanded_token_ids=expanded)
    vc2_sig = np.array([0xDEAD, 0xBEEF, 0xCAFE], dtype=np.uint64)
    vc2_sex = np.array([0, 64, 128], dtype=np.uint32)  # chunk_index * 64
    stage3_ct = _make_stage3_call_target(
        stage2_ct=stage2_ct,
        number_chunk_slices={TokenType.VALUED_CONST_V2: slice(0, 3)},
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[stage3_ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 3], dtype=np.uint32),
        numbers_per_TokenType={TokenType.VALUED_CONST_V2: (vc2_sig, vc2_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, vc2_sig)
    np.testing.assert_array_equal(sex, vc2_sex)


# ---------------------------------------------------------------------------
# Test 4: F128 finite source -- 2 chunks per source.
# ---------------------------------------------------------------------------


def test_f128_finite_two_chunks_per_source():
    """ALG-2 promotes a finite F128 source to a 2-chunk pair (chunk_index
    in {0, 1}); the sidecar emits both chunks in stream order."""
    expanded = _u16([_F128_SHIFTED, _F128_SHIFTED])
    stage2_ct = _make_stage2_call_target(expanded_token_ids=expanded)
    f128_sig = np.array([0x1111, 0x2222], dtype=np.uint64)
    f128_sex = np.array([0, 64], dtype=np.uint32)
    stage3_ct = _make_stage3_call_target(
        stage2_ct=stage2_ct,
        number_chunk_slices={TokenType.FLOAT128: slice(0, 2)},
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[stage3_ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 2], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT128: (f128_sig, f128_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, f128_sig)
    np.testing.assert_array_equal(sex, f128_sex)


# ---------------------------------------------------------------------------
# Test 5: F128 NaN/Inf source -- 1 chunk per source.
# ---------------------------------------------------------------------------


def test_f128_nan_inf_one_chunk_per_source():
    """ALG-2 leaves NaN/Inf F128 sources as a single chunk -- no
    second-position promotion."""
    expanded = _u16([_F128_SHIFTED])
    stage2_ct = _make_stage2_call_target(expanded_token_ids=expanded)
    f128_sig = np.array([0xFFFF_FFFF_FFFF_FFFF], dtype=np.uint64)
    f128_sex = np.array([0xFFFF], dtype=np.uint32)
    stage3_ct = _make_stage3_call_target(
        stage2_ct=stage2_ct,
        number_chunk_slices={TokenType.FLOAT128: slice(0, 1)},
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[stage3_ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 1], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT128: (f128_sig, f128_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, f128_sig)
    np.testing.assert_array_equal(sex, f128_sex)


# ---------------------------------------------------------------------------
# Test 6: Multi-row batch -- row_offsets correctly delimit each row.
# ---------------------------------------------------------------------------


def test_multi_row_offsets_delimit():
    """Two rows of differing chunk counts; per-row offsets bracket each
    row's contribution exactly."""
    # Row 0: 2 F32 chunks. Row 1: 1 F64 chunk.
    row0_expanded = _u16([_F32_SHIFTED, _F32_SHIFTED])
    row1_expanded = _u16([_F64_SHIFTED])
    row0_ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=row0_expanded),
        number_chunk_slices={TokenType.FLOAT32: slice(0, 2)},
    )
    row1_ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=row1_expanded),
        number_chunk_slices={TokenType.FLOAT64: slice(0, 1)},
    )
    f32_sig = np.array([7, 11], dtype=np.uint64)
    f32_sex = np.array([70, 110], dtype=np.uint32)
    f64_sig = np.array([13], dtype=np.uint64)
    f64_sex = np.array([130], dtype=np.uint32)
    stage3 = _wrap_stage3_batch(
        variants_per_section=[
            [[row0_ct]],  # section 0, variant 0
            [[row1_ct]],  # section 1, variant 0
        ],
        batch_idx_to_section_variant=np.array(
            [[0, 0], [1, 0]], dtype=np.uint32
        ),
        number_row_offsets=np.array([0, 2, 3], dtype=np.uint32),
        numbers_per_TokenType={
            TokenType.FLOAT32: (f32_sig, f32_sex),
            TokenType.FLOAT64: (f64_sig, f64_sex),
        },
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, [7, 11, 13])
    np.testing.assert_array_equal(sex, [70, 110, 130])


# ---------------------------------------------------------------------------
# Test 7: Padding row -- zero contribution.
# ---------------------------------------------------------------------------


def test_padding_row_zero_contribution():
    """A padding row (sentinel ``(UINT32_MAX, UINT32_MAX)``) contributes
    nothing; row_offsets must mark a zero-length segment."""
    real_expanded = _u16([_F32_SHIFTED, _F32_SHIFTED])
    real_ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=real_expanded),
        number_chunk_slices={TokenType.FLOAT32: slice(0, 2)},
    )
    f32_sig = np.array([42, 43], dtype=np.uint64)
    f32_sex = np.array([100, 101], dtype=np.uint32)
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[real_ct]]],
        batch_idx_to_section_variant=np.array(
            [
                [0, 0],
                [UINT32_MAX, UINT32_MAX],
            ],
            dtype=np.uint32,
        ),
        number_row_offsets=np.array([0, 2, 2], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT32: (f32_sig, f32_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    assert sig.shape == (2,)
    assert sex.shape == (2,)
    np.testing.assert_array_equal(sig, f32_sig)
    np.testing.assert_array_equal(sex, f32_sex)


def test_padding_row_with_unexpected_delta_raises():
    """A padding row that's wired with a non-zero ``number_row_offsets``
    delta indicates an upstream sizing bug -- the assembler should
    surface it rather than silently emit garbage."""
    real_expanded = _u16([_F32_SHIFTED])
    real_ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=real_expanded),
        number_chunk_slices={TokenType.FLOAT32: slice(0, 1)},
    )
    f32_sig = np.array([1], dtype=np.uint64)
    f32_sex = np.array([1], dtype=np.uint32)
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[real_ct]]],
        batch_idx_to_section_variant=np.array(
            [
                [0, 0],
                [UINT32_MAX, UINT32_MAX],
            ],
            dtype=np.uint32,
        ),
        # Wrongly claims the padding row owns 1 chunk -- assembler bails.
        number_row_offsets=np.array([0, 1, 2], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT32: (f32_sig, f32_sex)},
    )
    with pytest.raises(AssertionError, match="padding row"):
        assemble_number_sidecars(stage3)


# ---------------------------------------------------------------------------
# Multi-row mapping: one variant referenced by multiple batch rows
# (RESAMPLE / REDISTRIBUTE policy emits identical chunks per row).
# ---------------------------------------------------------------------------


def test_resample_policy_duplicates_chunks_per_referencing_row():
    """RESAMPLE_WITHIN_SECTION can map the same ``(section, variant)`` pair
    to two batch rows -- each row gets the same chunks."""
    expanded = _u16([_F32_SHIFTED, _F32_SHIFTED])
    ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=expanded),
        number_chunk_slices={TokenType.FLOAT32: slice(0, 2)},
    )
    f32_sig = np.array([9, 99], dtype=np.uint64)
    f32_sex = np.array([1, 2], dtype=np.uint32)
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[ct]]],
        batch_idx_to_section_variant=np.array(
            [[0, 0], [0, 0]], dtype=np.uint32
        ),
        number_row_offsets=np.array([0, 2, 4], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT32: (f32_sig, f32_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, [9, 99, 9, 99])
    np.testing.assert_array_equal(sex, [1, 2, 1, 2])


# ---------------------------------------------------------------------------
# Cutoff truncation: partial_cut_length < full length drops trailing chunks.
# ---------------------------------------------------------------------------


def test_partial_cut_truncates_trailing_chunks():
    """``partial_cut_length`` < full length means only the chunks at
    positions ``[0, partial_cut_length)`` survive -- the per-type slice
    is sized accordingly per D8."""
    # Full stream has 3 F32 positions; cut at length 2 drops the last.
    expanded = _u16([_F32_SHIFTED, _F32_SHIFTED, _F32_SHIFTED])
    stage2_ct = _make_stage2_call_target(
        expanded_token_ids=expanded,
        partial_cut_length=2,
    )
    f32_sig = np.array([5, 6], dtype=np.uint64)
    f32_sex = np.array([50, 60], dtype=np.uint32)
    # Stage 3 sized the slice to the 2 surviving chunks.
    ct = _make_stage3_call_target(
        stage2_ct=stage2_ct,
        number_chunk_slices={TokenType.FLOAT32: slice(0, 2)},
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 2], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT32: (f32_sig, f32_sex)},
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, [5, 6])
    np.testing.assert_array_equal(sex, [50, 60])


# ---------------------------------------------------------------------------
# Multi-call_target row: chunks concat across call_targets in encounter order.
# ---------------------------------------------------------------------------


def test_multi_call_target_concat_in_encounter_order():
    """A variant with two call_targets emits chunks in encounter order:
    root's chunks first, then callee_1's chunks."""
    root_expanded = _u16([_F32_SHIFTED])
    callee_expanded = _u16([_F64_SHIFTED, _F64_SHIFTED])
    root_ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=root_expanded),
        number_chunk_slices={TokenType.FLOAT32: slice(0, 1)},
    )
    callee_ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=callee_expanded),
        number_chunk_slices={TokenType.FLOAT64: slice(0, 2)},
    )
    f32_sig = np.array([1], dtype=np.uint64)
    f32_sex = np.array([10], dtype=np.uint32)
    f64_sig = np.array([2, 3], dtype=np.uint64)
    f64_sex = np.array([20, 30], dtype=np.uint32)
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[root_ct, callee_ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 3], dtype=np.uint32),
        numbers_per_TokenType={
            TokenType.FLOAT32: (f32_sig, f32_sex),
            TokenType.FLOAT64: (f64_sig, f64_sex),
        },
    )
    sig, sex = assemble_number_sidecars(stage3)
    np.testing.assert_array_equal(sig, [1, 2, 3])
    np.testing.assert_array_equal(sex, [10, 20, 30])


# ---------------------------------------------------------------------------
# Empty batch (no chunks anywhere).
# ---------------------------------------------------------------------------


def test_empty_batch_returns_empty_arrays():
    """Zero rows worth of content -- sidecars are zero-length, dtypes
    correct."""
    expanded = _u16([_LOCAL_FUNC_SHIFTED, _LOCAL_FUNC_SHIFTED])  # no numbers
    ct = _make_stage3_call_target(
        stage2_ct=_make_stage2_call_target(expanded_token_ids=expanded),
        number_chunk_slices={},
    )
    stage3 = _wrap_stage3_batch(
        variants_per_section=[[[ct]]],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        number_row_offsets=np.array([0, 0], dtype=np.uint32),
        numbers_per_TokenType={},
    )
    sig, sex = assemble_number_sidecars(stage3)
    assert sig.shape == (0,)
    assert sex.shape == (0,)
    assert sig.dtype == np.uint64
    assert sex.dtype == np.uint32
