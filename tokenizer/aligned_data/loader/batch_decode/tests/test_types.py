"""Smoke tests for the staged batch-decode dataclass backbone.

Single concern: assert the structural contract of the four-level
hierarchy from plan section D9 -- frozen enforcement, back-pointer
chain navigation, and the :class:`VariantPadding` enum surface.
Algorithmic content (length prediction, FP normalization, dedup
walk) lives in per-stage test files added by later phases.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
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
# Fixture builders -- minimal-but-typed dummies so the dataclasses can be
# constructed without dragging in a real binary memmap.
# ---------------------------------------------------------------------------


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
    n = 0
    return InlineDecodeState(
        raw_tokens=np.zeros(n, dtype=np.uint16),
        real_mask=np.zeros(n, dtype=bool),
        number_mask=np.zeros(n, dtype=bool),
        runlen_number=np.zeros(n, dtype=np.uint32),
        runlen_value=np.zeros(n, dtype=np.uint32),
        carries_inline_mask=np.zeros(n, dtype=bool),
        is_negative_per_position=np.zeros(n, dtype=bool),
    )


def _make_section() -> Section:
    return Section(
        function_name_ptr=42,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _make_stage1_call_target() -> Stage1CallTarget:
    return Stage1CallTarget(
        function_data=_make_function_data(),
        state=_make_inline_decode_state(),
        call_targets_section=[],
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=42,
    )


def _make_stage1_chain() -> Stage1Batch:
    ct = _make_stage1_call_target()
    variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=[ct],
    )
    section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_make_section(),
        variants=[variant],
    )
    return Stage1Batch(
        sections=[section],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        batch_size=1,
    )


def _make_stage2_chain(stage1_batch: Stage1Batch) -> Stage2Batch:
    stage1_section = stage1_batch.sections[0]
    stage1_variant = stage1_section.variants[0]
    stage1_ct = stage1_variant.call_targets[0]

    stage2_ct = Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=np.zeros(0, dtype=np.uint16),
        extra_value_v2_mask=np.zeros(0, dtype=bool),
        extra_f128_mask=np.zeros(0, dtype=bool),
        predicted_full_length=0,
        surviving_token_count=0,
        surviving_identity_count=0,
        surviving_number_chunk_count=0,
        is_cut=False,
        partial_cut_length=0,
    )
    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=[stage2_ct],
        cut_call_target_index=1,
        total_surviving_token_count=0,
        total_surviving_identity_count=0,
        total_surviving_number_chunk_count=0,
    )
    stage2_section = Stage2Section(
        stage1=stage1_section,
        variants=[stage2_variant],
    )
    return Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section],
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
    )


def _make_stage3_chain(stage2_batch: Stage2Batch) -> Stage3Batch:
    stage2_section = stage2_batch.sections[0]
    stage2_variant = stage2_section.variants[0]
    stage2_ct = stage2_variant.call_targets[0]

    stage3_ct = Stage3CallTarget(
        stage2=stage2_ct,
        inline_byte_slice=slice(0, 0),
        identity_slice=slice(0, 0),
        number_chunk_slices={},
    )
    stage3_variant = Stage3Variant(
        stage2=stage2_variant,
        call_targets=[stage3_ct],
    )
    stage3_section = Stage3Section(
        stage2=stage2_section,
        variants=[stage3_variant],
    )
    return Stage3Batch(
        stage2=stage2_batch,
        sections=[stage3_section],
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=np.zeros(0, dtype=np.uint16),
        numbers_per_TokenType={},
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        f128_is_nan_or_inf=np.zeros(0, dtype=np.bool_),
        f128_visible_chunks=np.zeros(0, dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Frozen enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,field,value",
    [
        (_make_stage1_call_target, "function_name_ptr", 999),
        # SectionPointerSpec
        (lambda: SectionPointerSpec(arm=SectionKind.MATCHED, idx=0), "idx", 5),
    ],
)
def test_frozen_enforcement_simple(factory, field, value):
    instance = factory()
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, value)


def test_frozen_enforcement_full_hierarchy():
    """Each level of each stage must reject mutation."""
    stage1_batch = _make_stage1_chain()
    stage2_batch = _make_stage2_chain(stage1_batch)
    stage3_batch = _make_stage3_chain(stage2_batch)

    instances_to_check = [
        stage1_batch,
        stage1_batch.sections[0],
        stage1_batch.sections[0].variants[0],
        stage1_batch.sections[0].variants[0].call_targets[0],
        stage2_batch,
        stage2_batch.sections[0],
        stage2_batch.sections[0].variants[0],
        stage2_batch.sections[0].variants[0].call_targets[0],
        stage3_batch,
        stage3_batch.sections[0],
        stage3_batch.sections[0].variants[0],
        stage3_batch.sections[0].variants[0].call_targets[0],
    ]
    for instance in instances_to_check:
        with pytest.raises(FrozenInstanceError):
            # ``__dummy_attr__`` is not a real field; any setattr on a
            # frozen dataclass raises ``FrozenInstanceError`` before
            # the field-validity check.
            setattr(instance, "__dummy_attr__", 0)


def test_batch_decode_result_frozen():
    result = BatchDecodeResult(
        tokens=np.zeros((1, 1), dtype=np.uint16),
        identities=np.zeros(0, dtype=np.uint16),
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        numbers_significant=np.zeros(0, dtype=np.uint64),
        numbers_sign_exponent=np.zeros(0, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
        batch_idx_to_section_variant=np.zeros((1, 2), dtype=np.uint32),
        fid_sidecar=None,
        fid_row_offsets=None,
        intermediate=None,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(result, "tokens", np.zeros((2, 2), dtype=np.uint16))


# ---------------------------------------------------------------------------
# Back-pointer chain navigation
# ---------------------------------------------------------------------------


def test_back_pointer_chain_navigation():
    """Each level-N entry navigates to the prior stage's same-level
    entry via a single ``.stageN`` back-pointer. No parallel-list
    co-indexing is required to recover the prior-stage data."""
    stage1_batch = _make_stage1_chain()
    stage2_batch = _make_stage2_chain(stage1_batch)
    stage3_batch = _make_stage3_chain(stage2_batch)

    # Level 1 chain: stage3.stage2.stage1
    assert stage3_batch.stage2 is stage2_batch
    assert stage3_batch.stage2.stage1 is stage1_batch

    # Level 2 chain: stage3_section.stage2.stage1
    stage3_section = stage3_batch.sections[0]
    stage2_section = stage2_batch.sections[0]
    stage1_section = stage1_batch.sections[0]
    assert stage3_section.stage2 is stage2_section
    assert stage3_section.stage2.stage1 is stage1_section

    # Level 3 chain: stage3_variant.stage2.stage1
    stage3_variant = stage3_section.variants[0]
    stage2_variant = stage2_section.variants[0]
    stage1_variant = stage1_section.variants[0]
    assert stage3_variant.stage2 is stage2_variant
    assert stage3_variant.stage2.stage1 is stage1_variant

    # Level 4 chain: stage3_ct.stage2.stage1
    stage3_ct = stage3_variant.call_targets[0]
    stage2_ct = stage2_variant.call_targets[0]
    stage1_ct = stage1_variant.call_targets[0]
    assert stage3_ct.stage2 is stage2_ct
    assert stage3_ct.stage2.stage1 is stage1_ct

    # Plan D9 + D3: root function body lives at call_targets index 0
    # for every variant. Validate the index-0 invariant for the chain
    # we built (single-CT variant -- index 0 is the only entry).
    assert stage1_variant.call_targets[0] is stage1_ct


def test_call_target_uses_existing_function_data_and_state():
    """Level-4 entry must wire to the existing ``FunctionData`` and
    ``InlineDecodeState`` types (no new mirror dataclasses)."""
    ct = _make_stage1_call_target()
    assert isinstance(ct.function_data, FunctionData)
    assert isinstance(ct.state, InlineDecodeState)
    assert isinstance(ct.encounter_category, Category)


# ---------------------------------------------------------------------------
# VariantPadding enum surface
# ---------------------------------------------------------------------------


def test_variant_padding_members():
    """Plan D6 fixes the four-member set + their string values."""
    members = {m.name: m.value for m in VariantPadding}
    assert members == {
        "PAD_NULL": "pad_null",
        "RESAMPLE_WITHIN_SECTION": "resample",
        "RAGGED": "ragged",
        "REDISTRIBUTE": "redistribute",
    }
    assert len(VariantPadding) == 4


def test_section_pointer_spec_shape():
    """Plan section 'Module layout' + the consumer signatures on
    ``BinarySession``: ``(arm: SectionKind, idx: int)``."""
    spec = SectionPointerSpec(arm=SectionKind.MATCHED, idx=7)
    assert spec.arm is SectionKind.MATCHED
    assert spec.idx == 7


# ---------------------------------------------------------------------------
# Per-TokenType keying for stage 3
# ---------------------------------------------------------------------------


def test_stage3_call_target_number_chunk_slices_keyed_by_token_type():
    """``Stage3CallTarget.number_chunk_slices`` is keyed by
    :class:`TokenType` per plan D9."""
    stage3_ct = Stage3CallTarget(
        stage2=_make_stage2_chain(_make_stage1_chain())
        .sections[0]
        .variants[0]
        .call_targets[0],
        inline_byte_slice=slice(0, 0),
        identity_slice=slice(0, 0),
        number_chunk_slices={
            TokenType.FLOAT64: slice(0, 4),
            TokenType.VALUED_CONST_V2: slice(4, 7),
        },
    )
    assert isinstance(stage3_ct.number_chunk_slices, dict)
    for key in stage3_ct.number_chunk_slices:
        assert isinstance(key, TokenType)
