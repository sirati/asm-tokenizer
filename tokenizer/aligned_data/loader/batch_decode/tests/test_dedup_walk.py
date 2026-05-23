"""Unit tests for :func:`apply_per_row_remap`.

Single concern: pin the stage-4 per-row dedup walk behavioural contract
from ``batch_decode_plan.md`` ``## Algorithms`` ALG-3 + ALG-4 + ALG-9.

Tests construct synthetic :class:`Stage3Batch` fixtures directly — the
dedup walk reads only:

* ``stage3_batch.identities_flat_caller_local`` (mutated in place).
* Per Stage3CallTarget: ``identity_slice``, ``stage2.surviving_token_count``,
  ``stage2.partial_cut_length``, ``stage2.expanded_token_ids``.
* Per Stage1CallTarget: ``encounter_category``, ``function_name_ptr``,
  ``call_targets_section``, ``function_data.metadata["category_counts"]``.
* ``Stage1Batch.batch_size``; per Stage1Variant: ``batch_idx``.

Everything else is filled with minimal-but-typed dummies. The
``InlineDecodeState`` is irrelevant here — stage 3 already produced
``identities_flat_caller_local``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pytest

from dedup_hashmap import HashMapU32U16

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk import (
    COUNTER_CATEGORIES,
    FUNCTION_CATEGORIES,
    _CATEGORY_TO_SHIFTED_ID,
    apply_per_row_remap,
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
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
    Section,
)
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _empty_inline_decode_state() -> InlineDecodeState:
    return InlineDecodeState(
        raw_tokens=np.zeros(0, dtype=np.uint16),
        real_mask=np.zeros(0, dtype=bool),
        number_mask=np.zeros(0, dtype=bool),
        runlen_number=np.zeros(0, dtype=np.uint32),
        runlen_value=np.zeros(0, dtype=np.uint32),
        carries_inline_mask=np.zeros(0, dtype=bool),
        is_negative_per_position=np.zeros(0, dtype=bool),
    )


def _function_data(
    *,
    category_counts: Optional[dict[Category, int]] = None,
    func_name: str = "f",
) -> FunctionData:
    metadata: dict = {
        "arch": "x86_64",
        "compiler": "gcc",
        "opt": "O2",
        "category_counts": dict(category_counts or {}),
    }
    return FunctionData(
        func_name=func_name,
        metadata=metadata,
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _make_call_target_row(
    fid: int, ct_type: CallTargetType
) -> CallTarget:
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=0,
        type=ct_type,
        is_matched=False,
    )


def _build_expanded_token_ids(
    in_stream_categories: list[Category],
    *,
    prepend_category: Category,
) -> np.ndarray:
    """Build a synthetic ``expanded_token_ids`` array.

    Slot 0 holds the prepend's token id; slots 1.. hold the in-stream
    identity-band tokens (no number tokens — keeps the test focused on
    identity remap behaviour).
    """
    prepend_id = _CATEGORY_TO_SHIFTED_ID[prepend_category]
    in_stream_ids = [_CATEGORY_TO_SHIFTED_ID[c] for c in in_stream_categories]
    return np.asarray([prepend_id, *in_stream_ids], dtype=np.uint16)


@dataclass
class _CallTargetBuild:
    """Helper for building per-call-target test inputs."""

    fid: int
    encounter_category: Category
    # Caller-local ids of in-stream identity slots, parallel to ``in_stream_categories``.
    in_stream_caller_local_ids: list[int]
    in_stream_categories: list[Category]
    # Section header for THIS function — used by the dedup walk's ALG-3
    # filter. (function_name_ptr, CallTargetType) tuples in encounter
    # order, grouped LOCAL -> PLT -> EXTERN.
    section_call_targets: list[tuple[int, CallTargetType]]
    # Per-COUNTER-Category unique-id count for this function.
    counter_counts: dict[Category, int]


def _make_stage3_variant_from_calls(
    builds: list[_CallTargetBuild],
    *,
    batch_idx: int,
) -> tuple[Stage3Variant, np.ndarray, list[slice]]:
    """Build a Stage3Variant with the per-call-target identities laid out
    contiguously.

    Returns ``(stage3_variant, identities_flat_caller_local, identity_slices)``.
    The flat identities array has 1 prepend slot + len(in_stream_caller_local_ids)
    slots per call_target, contiguous in the order of ``builds``.
    """
    # Pre-size the flat identities array.
    per_call_lengths = [
        1 + len(b.in_stream_caller_local_ids) for b in builds
    ]
    total = sum(per_call_lengths)
    identities_flat = np.zeros(total, dtype=np.uint16)

    # Write the in-stream caller-local ids into their slots; prepend
    # slots remain at 0 (stage 4 writes them per ALG-9).
    identity_slices: list[slice] = []
    offset = 0
    for b, n in zip(builds, per_call_lengths):
        sl = slice(offset, offset + n)
        identity_slices.append(sl)
        identities_flat[sl.start + 1 : sl.stop] = np.asarray(
            b.in_stream_caller_local_ids, dtype=np.uint16
        )
        offset += n

    # Build the 4-level chain per call_target.
    stage3_call_targets: list[Stage3CallTarget] = []
    for b, sl in zip(builds, identity_slices):
        section_calls = [
            _make_call_target_row(fid, ct_type)
            for fid, ct_type in b.section_call_targets
        ]
        stage1_ct = Stage1CallTarget(
            function_data=_function_data(
                category_counts=b.counter_counts,
                func_name=f"f_{b.fid}",
            ),
            state=_empty_inline_decode_state(),
            call_targets_section=section_calls,
            encounter_category=b.encounter_category,
            parent_call_target_index=None,
            function_name_ptr=b.fid,
        )
        expanded = _build_expanded_token_ids(
            b.in_stream_categories,
            prepend_category=b.encounter_category,
        )
        stage2_ct = Stage2CallTarget(
            stage1=stage1_ct,
            expanded_token_ids=expanded,
            extra_value_v2_mask=np.zeros_like(expanded, dtype=bool),
            extra_f128_mask=np.zeros_like(expanded, dtype=bool),
            predicted_full_length=int(expanded.shape[0]),
            surviving_token_count=int(expanded.shape[0]),
            surviving_identity_count=int(expanded.shape[0]),
            surviving_number_chunk_count=0,
            is_cut=False,
            partial_cut_length=int(expanded.shape[0]),
        )
        stage3_ct = Stage3CallTarget(
            stage2=stage2_ct,
            inline_byte_slice=slice(0, 0),
            identity_slice=sl,
            number_chunk_slices={},
        )
        stage3_call_targets.append(stage3_ct)

    # Wire up section/variant/batch.
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=batch_idx,
        call_targets=[ct.stage2.stage1 for ct in stage3_call_targets],
    )
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=Section(
            function_name_ptr=builds[0].fid,
            section_offset=0,
            call_targets=stage3_call_targets[0].stage2.stage1.call_targets_section,
            variants=[],
        ),
        variants=[stage1_variant],
    )

    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=[ct.stage2 for ct in stage3_call_targets],
        cut_call_target_index=len(stage3_call_targets),
        total_surviving_token_count=sum(
            ct.stage2.surviving_token_count for ct in stage3_call_targets
        ),
        total_surviving_identity_count=0,
        total_surviving_number_chunk_count=0,
    )
    stage2_section = Stage2Section(
        stage1=stage1_section,
        variants=[stage2_variant],
    )

    stage3_variant = Stage3Variant(
        stage2=stage2_variant,
        call_targets=stage3_call_targets,
    )

    return stage3_variant, identities_flat, identity_slices


def _wrap_variant_into_batch(
    stage3_variant: Stage3Variant,
    identities_flat: np.ndarray,
    *,
    batch_size: int,
) -> Stage3Batch:
    """Wrap a single Stage3Variant in the surrounding Section/Batch layers."""

    stage1_variant = stage3_variant.stage2.stage1
    section_obj = Section(
        function_name_ptr=stage1_variant.call_targets[0].function_name_ptr,
        section_offset=0,
        call_targets=stage1_variant.call_targets[0].call_targets_section,
        variants=[],
    )
    stage1_section_obj = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=section_obj,
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section_obj],
        batch_idx_to_section_variant=np.asarray(
            [[0, 0]] + [[0xFFFFFFFF, 0xFFFFFFFF]] * (batch_size - 1),
            dtype=np.uint32,
        ),
        batch_size=batch_size,
    )

    stage2_section_obj = Stage2Section(
        stage1=stage1_section_obj,
        variants=[stage3_variant.stage2],
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section_obj],
        identity_row_offsets=np.zeros(batch_size + 1, dtype=np.uint32),
        number_row_offsets=np.zeros(batch_size + 1, dtype=np.uint32),
    )

    stage3_section = Stage3Section(
        stage2=stage2_section_obj,
        variants=[stage3_variant],
    )
    return Stage3Batch(
        stage2=stage2_batch,
        sections=[stage3_section],
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=identities_flat,
        numbers_per_TokenType={},
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        f128_is_nan_or_inf=np.zeros(0, dtype=np.bool_),
    )


def _make_dedup_maps() -> dict[Category, HashMapU32U16]:
    return {cat: HashMapU32U16(capacity=8) for cat in FUNCTION_CATEGORIES}


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_single_call_target_function_dedup_three_distinct_fids():
    """Single root call_target whose body references three distinct
    LOCAL_FUNC FIDs (caller-local ids 0, 1, 2). The dedup walk mints
    counter ids 1, 2, 3 (counter 0 is reserved for the root's
    self-prepend per ALG-3 + D4)."""

    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 1, 2],
        in_stream_categories=[
            Category.LOCAL_FUNC,
            Category.LOCAL_FUNC,
            Category.LOCAL_FUNC,
        ],
        section_call_targets=[
            (200, CallTargetType.LOCAL),  # caller-local 0
            (201, CallTargetType.LOCAL),  # caller-local 1
            (202, CallTargetType.LOCAL),  # caller-local 2
        ],
        counter_counts={},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    remapped, fid_sidecar, fid_row_offsets = apply_per_row_remap(
        stage3_batch, dedup_maps=_make_dedup_maps()
    )

    assert remapped is identities
    # Prepend slot at slot 0 holds the root's self-counter = 0.
    assert int(remapped[slices[0].start]) == 0
    # In-stream slots: caller-local 0/1/2 -> counter 1/2/3.
    assert remapped[slices[0].start + 1 : slices[0].stop].tolist() == [1, 2, 3]
    assert fid_sidecar is None
    assert fid_row_offsets is None


def test_root_self_recursion_dedupes_to_counter_zero():
    """Root calls itself: the in-stream LOCAL_FUNC reference to the
    root's own FID dedupes to counter id 0 (seeded by the root's
    self-prepend per ALG-3 + ALG-9). Two occurrences both get 0."""

    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        # Both in-stream identities point at the root itself
        # (caller-local 0 = first LOCAL row = root).
        in_stream_caller_local_ids=[0, 0],
        in_stream_categories=[
            Category.LOCAL_FUNC,
            Category.LOCAL_FUNC,
        ],
        section_call_targets=[
            (100, CallTargetType.LOCAL),  # self-reference
        ],
        counter_counts={},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    assert int(identities[slices[0].start]) == 0
    assert identities[slices[0].start + 1 : slices[0].stop].tolist() == [0, 0]


def test_multi_call_target_dedup_b_and_c_share_callee_counter():
    """Three call_targets in encounter order: root (A), inlined callee B,
    inlined callee C. Both B and C list a LOCAL_FUNC reference to D.
    D's caller-local id in B's section_call_targets and in C's
    section_call_targets need not match (each function has its own
    caller-local indexing) — but D's variant-global counter id must be
    the SAME in both occurrences (FID-keyed dedup per D4)."""

    # Counter id assignment expected:
    #   LOCAL_FUNC: A=0 (root seed). Root's section_call_targets list
    #   includes B + C, so B gets 1 and C gets 2 (encounter order in
    #   root's call_targets table). Then B's own section_call_targets
    #   has D as caller-local 0 -> fresh counter 3. C's
    #   section_call_targets also has D -> dedupes to counter 3.
    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 1],  # B and C
        in_stream_categories=[Category.LOCAL_FUNC, Category.LOCAL_FUNC],
        section_call_targets=[
            (200, CallTargetType.LOCAL),  # B at caller-local 0
            (201, CallTargetType.LOCAL),  # C at caller-local 1
        ],
        counter_counts={},
    )
    callee_b = _CallTargetBuild(
        fid=200,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0],  # D
        in_stream_categories=[Category.LOCAL_FUNC],
        section_call_targets=[
            (202, CallTargetType.LOCAL),  # D at caller-local 0
        ],
        counter_counts={},
    )
    callee_c = _CallTargetBuild(
        fid=201,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0],  # D (caller-local 0 in C's space)
        in_stream_categories=[Category.LOCAL_FUNC],
        section_call_targets=[
            (202, CallTargetType.LOCAL),  # D at caller-local 0
        ],
        counter_counts={},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root, callee_b, callee_c], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    # Root prepend = counter 0.
    assert int(identities[slices[0].start]) == 0
    # Root's in-stream LOCAL refs to B and C -> counter 1 and 2.
    assert identities[slices[0].start + 1 : slices[0].stop].tolist() == [1, 2]

    # B's prepend = B's counter (= 1).
    assert int(identities[slices[1].start]) == 1
    # B's in-stream LOCAL ref to D -> fresh counter 3.
    assert identities[slices[1].start + 1 : slices[1].stop].tolist() == [3]

    # C's prepend = C's counter (= 2).
    assert int(identities[slices[2].start]) == 2
    # C's in-stream LOCAL ref to D -> dedup to counter 3.
    assert identities[slices[2].start + 1 : slices[2].stop].tolist() == [3]


def test_plt_local_independent_counter_spaces():
    """PLT_FUNC counter 0 and LOCAL_FUNC counter 0 are distinct slots:
    a function appearing as BOTH a LOCAL_FUNC call AND a PLT_FUNC call
    in the same row gets DIFFERENT counter ids — one in each space (per
    D4: per-Category counter spaces are fully INDEPENDENT)."""

    # Root has one LOCAL ref (to FID 200) and one PLT ref (to FID 200
    # too — same FID, different category). LOCAL_FUNC seed reserves
    # counter 0 for the root; in-stream LOCAL ref to 200 mints counter
    # 1. PLT_FUNC starts at 0; in-stream PLT ref to 200 mints counter 0.
    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        # First in-stream slot is LOCAL_FUNC at caller-local 0 (FID 200),
        # second is PLT_FUNC at caller-local 0 (FID 200).
        in_stream_caller_local_ids=[0, 0],
        in_stream_categories=[Category.LOCAL_FUNC, Category.PLT_FUNC],
        section_call_targets=[
            (200, CallTargetType.LOCAL),  # caller-local LOCAL 0
            (200, CallTargetType.PLT),  # caller-local PLT 0
        ],
        counter_counts={},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    # Root prepend = LOCAL_FUNC counter 0.
    assert int(identities[slices[0].start]) == 0
    # LOCAL ref to FID 200 -> counter 1; PLT ref to FID 200 -> counter 0
    # (in PLT_FUNC's independent space).
    assert identities[slices[0].start + 1 : slices[0].stop].tolist() == [1, 0]


def test_counter_category_block_offset_bump():
    """ALG-4 COUNTER renumbering: two LOCAL call_targets each with 3
    BLOCK_V2 locals. First function's in-stream BLOCK ids are 0/1/2 ->
    stay 0/1/2 (offset=0). Second function's BLOCK ids 0/1/2 -> become
    3/4/5 (offset = 3 from first function's category_counts)."""

    # First function: BLOCK in-stream ids [0, 1, 2].
    func1 = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 1, 2],
        in_stream_categories=[Category.BLOCK, Category.BLOCK, Category.BLOCK],
        section_call_targets=[
            (200, CallTargetType.LOCAL),  # callee FID for the recursion-test variant
        ],
        counter_counts={Category.BLOCK: 3},
    )
    # Second function: BLOCK in-stream ids [0, 1, 2].
    func2 = _CallTargetBuild(
        fid=200,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 1, 2],
        in_stream_categories=[Category.BLOCK, Category.BLOCK, Category.BLOCK],
        section_call_targets=[],
        counter_counts={Category.BLOCK: 3},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [func1, func2], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    # First function: no offset, BLOCK ids stay 0/1/2.
    assert identities[slices[0].start + 1 : slices[0].stop].tolist() == [
        0,
        1,
        2,
    ]
    # Second function: offset = 3, BLOCK ids become 3/4/5.
    assert identities[slices[1].start + 1 : slices[1].stop].tolist() == [
        3,
        4,
        5,
    ]


def test_mixed_function_and_counter_categories():
    """Mixed row: in-stream tokens of both FUNCTION (LOCAL_FUNC) and
    COUNTER (BLOCK) categories. The two groups address disjoint token
    ids so the remaps don't interfere."""

    # Two call_targets: root + one callee.
    # Root: in-stream [LOCAL_FUNC@0, BLOCK@0, BLOCK@1]
    # Callee: in-stream [LOCAL_FUNC@0, BLOCK@0]
    #
    # LOCAL_FUNC: root counter 0 (seed). Root's section_call_targets
    # has callee at LOCAL caller-local 0 -> counter 1. So root's
    # in-stream LOCAL ref to caller-local 0 maps to counter 1. Callee's
    # section_call_targets is empty; callee's in-stream LOCAL ref
    # caller-local 0... wait — caller-local 0 here refers to the FIRST
    # LOCAL row of callee's call_targets_section, but callee has none.
    # Adjust: callee has one LOCAL ref to FID 300 (caller-local 0).
    #   -> fresh counter 2.
    #
    # BLOCK: function 1 (root) has BLOCK_count = 2; function 2 has
    # BLOCK_count = 1. Offsets: root=0 -> BLOCK ids 0/1 stay 0/1.
    # Callee offset = 2 -> BLOCK id 0 becomes 2.
    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 0, 1],
        in_stream_categories=[
            Category.LOCAL_FUNC,
            Category.BLOCK,
            Category.BLOCK,
        ],
        section_call_targets=[
            (200, CallTargetType.LOCAL),  # caller-local LOCAL 0 = callee
        ],
        counter_counts={Category.BLOCK: 2},
    )
    callee = _CallTargetBuild(
        fid=200,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 0],
        in_stream_categories=[Category.LOCAL_FUNC, Category.BLOCK],
        section_call_targets=[
            (300, CallTargetType.LOCAL),  # caller-local LOCAL 0 = FID 300
        ],
        counter_counts={Category.BLOCK: 1},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root, callee], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    # Root prepend = counter 0.
    assert int(identities[slices[0].start]) == 0
    # Root in-stream: LOCAL_FUNC@0 -> counter 1; BLOCK@0 -> 0; BLOCK@1 -> 1.
    assert identities[slices[0].start + 1 : slices[0].stop].tolist() == [
        1,
        0,
        1,
    ]
    # Callee prepend = counter 1.
    assert int(identities[slices[1].start]) == 1
    # Callee in-stream: LOCAL_FUNC@0 -> fresh counter 2; BLOCK@0 -> 0+2 = 2.
    assert identities[slices[1].start + 1 : slices[1].stop].tolist() == [2, 2]


def test_dedup_maps_clean_between_rows():
    """Row 2's dedup state is INDEPENDENT of row 1: ``clean()`` resets
    the FUNCTION dedup maps before walking each row."""

    # Build a 2-row Stage3Batch where both rows use the same root FID
    # (but different caller-local stream content). Row 2 should see
    # counter id 0 for ITS root, not the row-1 root's lingering entry.

    # Row 1: root FID 100, in-stream LOCAL ref to FID 100 (self).
    # Both occurrences -> counter 0.
    row1 = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0],
        in_stream_categories=[Category.LOCAL_FUNC],
        section_call_targets=[(100, CallTargetType.LOCAL)],
        counter_counts={},
    )
    row1_variant, identities1, slices1 = _make_stage3_variant_from_calls(
        [row1], batch_idx=0
    )

    # Row 2: root FID 999 (different from row 1).  Its in-stream LOCAL
    # ref to FID 999 (self) should ALSO map to counter 0 — i.e. row 1's
    # LOCAL_FUNC dedup entry must not leak into row 2.
    row2 = _CallTargetBuild(
        fid=999,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0],
        in_stream_categories=[Category.LOCAL_FUNC],
        section_call_targets=[(999, CallTargetType.LOCAL)],
        counter_counts={},
    )
    row2_variant, identities2, slices2 = _make_stage3_variant_from_calls(
        [row2], batch_idx=1
    )

    # Splice into a single 2-row Stage3Batch.
    combined_identities = np.concatenate([identities1, identities2])
    # Translate row2_variant's identity_slice offsets by len(identities1).
    offset = identities1.shape[0]
    new_slices2 = [
        slice(sl.start + offset, sl.stop + offset) for sl in slices2
    ]
    # Rebuild row2's call_targets with shifted slices.
    row2_call_targets = []
    for ct, sl in zip(row2_variant.call_targets, new_slices2):
        row2_call_targets.append(
            replace(ct, identity_slice=sl)
        )
    row2_variant = replace(row2_variant, call_targets=row2_call_targets)
    slices2 = new_slices2

    # Build a Stage3Batch with both rows.
    stage1_section_1 = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=Section(
            function_name_ptr=100,
            section_offset=0,
            call_targets=row1_variant.stage2.stage1.call_targets[0].call_targets_section,
            variants=[],
        ),
        variants=[row1_variant.stage2.stage1],
    )
    stage1_section_2 = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=1,
        section=Section(
            function_name_ptr=999,
            section_offset=0,
            call_targets=row2_variant.stage2.stage1.call_targets[0].call_targets_section,
            variants=[],
        ),
        variants=[row2_variant.stage2.stage1],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section_1, stage1_section_2],
        batch_idx_to_section_variant=np.asarray(
            [[0, 0], [1, 0]], dtype=np.uint32
        ),
        batch_size=2,
    )

    stage2_section_1 = Stage2Section(
        stage1=stage1_section_1,
        variants=[row1_variant.stage2],
    )
    stage2_section_2 = Stage2Section(
        stage1=stage1_section_2,
        variants=[row2_variant.stage2],
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section_1, stage2_section_2],
        identity_row_offsets=np.zeros(3, dtype=np.uint32),
        number_row_offsets=np.zeros(3, dtype=np.uint32),
    )

    stage3_section_1 = Stage3Section(
        stage2=stage2_section_1,
        variants=[row1_variant],
    )
    stage3_section_2 = Stage3Section(
        stage2=stage2_section_2,
        variants=[row2_variant],
    )
    stage3_batch = Stage3Batch(
        stage2=stage2_batch,
        sections=[stage3_section_1, stage3_section_2],
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=combined_identities,
        numbers_per_TokenType={},
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        f128_is_nan_or_inf=np.zeros(0, dtype=np.bool_),
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    # Row 1: prepend = 0, in-stream self-ref = 0.
    assert int(combined_identities[slices1[0].start]) == 0
    assert combined_identities[
        slices1[0].start + 1 : slices1[0].stop
    ].tolist() == [0]
    # Row 2: prepend = 0 (cleaned LOCAL_FUNC map; row-2 root seeded
    # afresh), in-stream self-ref = 0.
    assert int(combined_identities[slices2[0].start]) == 0
    assert combined_identities[
        slices2[0].start + 1 : slices2[0].stop
    ].tolist() == [0]


def test_fid_sidecar_collects_counter_to_fid_mapping():
    """``collect_fid_sidecar=True`` returns the (per-row) counter_id ->
    function_name_ptr inverse mapping per plan D5.

    Layout per row: LOCAL_FUNC inverse (in counter-id order) ++
    PLT_FUNC inverse ++ EXT_FUNC inverse.
    """

    # Two call_targets: root (FID 100, LOCAL) + callee (FID 200, LOCAL).
    # In-stream: root references FID 200 (caller-local LOCAL 0) and
    # FID 300 (caller-local PLT 0).
    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 0],
        in_stream_categories=[Category.LOCAL_FUNC, Category.PLT_FUNC],
        section_call_targets=[
            (200, CallTargetType.LOCAL),  # caller-local LOCAL 0
            (300, CallTargetType.PLT),  # caller-local PLT 0
        ],
        counter_counts={},
    )
    callee = _CallTargetBuild(
        fid=200,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[],
        in_stream_categories=[],
        section_call_targets=[],
        counter_counts={},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root, callee], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    _, fid_sidecar, fid_row_offsets = apply_per_row_remap(
        stage3_batch,
        dedup_maps=_make_dedup_maps(),
        collect_fid_sidecar=True,
    )

    assert fid_sidecar is not None
    assert fid_row_offsets is not None
    # Layout: LOCAL_FUNC counters 0/1 -> FIDs 100 (root seed) + 200
    # (minted by root's ALG-3). PLT_FUNC counter 0 -> FID 300. EXT_FUNC
    # empty.
    assert fid_sidecar.tolist() == [100, 200, 300]
    assert fid_row_offsets.tolist() == [0, 3]


def test_fid_sidecar_default_off_returns_none():
    """When ``collect_fid_sidecar=False`` (default), the sidecar pair is
    ``(None, None)``."""

    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[],
        in_stream_categories=[],
        section_call_targets=[],
        counter_counts={},
    )
    stage3_variant, identities, _ = _make_stage3_variant_from_calls(
        [root], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    _, fid_sidecar, fid_row_offsets = apply_per_row_remap(
        stage3_batch, dedup_maps=_make_dedup_maps()
    )

    assert fid_sidecar is None
    assert fid_row_offsets is None


def test_missing_dedup_map_raises():
    """The caller must supply one dedup_map per FUNCTION Category;
    omitting one is a wiring bug and surfaces as AssertionError."""

    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[],
        in_stream_categories=[],
        section_call_targets=[],
        counter_counts={},
    )
    stage3_variant, identities, _ = _make_stage3_variant_from_calls(
        [root], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    incomplete = {
        Category.LOCAL_FUNC: HashMapU32U16(capacity=8),
        Category.PLT_FUNC: HashMapU32U16(capacity=8),
        # EXT_FUNC missing
    }
    with pytest.raises(AssertionError, match="EXT_FUNC"):
        apply_per_row_remap(stage3_batch, dedup_maps=incomplete)


def test_padding_row_skipped():
    """A Stage3Variant whose ``stage1.batch_idx is None`` is a padding
    row: the dedup walk skips it (no mutation, no sidecar contribution)."""

    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0],
        in_stream_categories=[Category.LOCAL_FUNC],
        section_call_targets=[(200, CallTargetType.LOCAL)],
        counter_counts={},
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root], batch_idx=0
    )
    # Mutate the Stage1Variant to drop batch_idx — this is the padding
    # signal per plan D10. We have to rebuild because the dataclass is
    # frozen.
    stage1_variant = replace(
        stage3_variant.stage2.stage1, batch_idx=None
    )
    new_stage2_variant = replace(
        stage3_variant.stage2, stage1=stage1_variant
    )
    new_stage3_variant = replace(stage3_variant, stage2=new_stage2_variant)

    # Snapshot the original (pre-walk) identities so we can verify the
    # walk left them untouched.
    expected_before = identities.copy()

    # Wrap into a batch with batch_size=1; the single row is padding.
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=Section(
            function_name_ptr=100,
            section_offset=0,
            call_targets=new_stage2_variant.stage1.call_targets[0].call_targets_section,
            variants=[],
        ),
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section],
        batch_idx_to_section_variant=np.asarray(
            [[0xFFFFFFFF, 0xFFFFFFFF]], dtype=np.uint32
        ),
        batch_size=1,
    )
    stage2_section = Stage2Section(
        stage1=stage1_section,
        variants=[new_stage2_variant],
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section],
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
    )
    stage3_section = Stage3Section(
        stage2=stage2_section,
        variants=[new_stage3_variant],
    )
    stage3_batch = Stage3Batch(
        stage2=stage2_batch,
        sections=[stage3_section],
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=identities,
        numbers_per_TokenType={},
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        f128_is_nan_or_inf=np.zeros(0, dtype=np.bool_),
    )

    _, fid_sidecar, fid_row_offsets = apply_per_row_remap(
        stage3_batch,
        dedup_maps=_make_dedup_maps(),
        collect_fid_sidecar=True,
    )

    # Padding row: no mutation.
    np.testing.assert_array_equal(identities, expected_before)
    # Sidecar still produced (shape-correct), but zero-length.
    assert fid_sidecar is not None
    assert fid_sidecar.size == 0
    assert fid_row_offsets is not None
    assert fid_row_offsets.tolist() == [0, 0]


def test_counter_offset_skipped_when_zero():
    """When the running offset is zero (first function in the row), no
    bump is applied — that is a numerical no-op but the implementation
    must not crash or write through invalid views (e.g. for a function
    with no in-stream Counter tokens). This is a smoke for the early-
    return path."""

    root = _CallTargetBuild(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[],
        in_stream_categories=[],
        section_call_targets=[],
        # Declares 5 BLOCK locals even though no in-stream BLOCK tokens —
        # the offset should still advance for the NEXT call_target.
        counter_counts={Category.BLOCK: 5},
    )
    callee = _CallTargetBuild(
        fid=200,
        encounter_category=Category.LOCAL_FUNC,
        in_stream_caller_local_ids=[0, 1],
        in_stream_categories=[Category.BLOCK, Category.BLOCK],
        section_call_targets=[],
        counter_counts={Category.BLOCK: 2},
    )
    # Wire the root's section_call_targets so it lists the callee.
    root = replace(
        root,
        section_call_targets=[(200, CallTargetType.LOCAL)],
    )
    stage3_variant, identities, slices = _make_stage3_variant_from_calls(
        [root, callee], batch_idx=0
    )
    stage3_batch = _wrap_variant_into_batch(
        stage3_variant, identities, batch_size=1
    )

    apply_per_row_remap(stage3_batch, dedup_maps=_make_dedup_maps())

    # Callee's in-stream BLOCK ids [0, 1] -> offset 5 -> [5, 6].
    assert identities[slices[1].start + 1 : slices[1].stop].tolist() == [5, 6]
