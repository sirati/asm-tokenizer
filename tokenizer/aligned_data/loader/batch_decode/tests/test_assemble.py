"""Integration tests for :func:`assemble_batch` -- stage-4 orchestrator.

Single concern of this file: pin the composition contract between the
four single-concern stage-4 modules (token assembly + dedup remap walk
+ number sidecar concat + optional fid sidecar). Tests construct
synthetic :class:`Stage3Batch` fixtures directly so the composition is
exercised independently of stages 1/2/3 (which are tested by their own
modules' tests).

These tests intentionally do NOT exercise the end-to-end pipeline; that
contract is in :mod:`test_entry`. Here we verify that
:func:`assemble_batch` correctly threads outputs from the four sub-
modules through to a :class:`BatchDecodeResult` with internally
consistent flat tensors + sidecar offsets.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._assemble import assemble_batch
from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._dedup_walk import (
    _CATEGORY_TO_SHIFTED_ID,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
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
from tokenizer.aligned_data.matched_sections_bin import CallTarget, Section
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Vocab anchors (re-derived so the tests stay independent of refactors that
# might shift the dedup_walk + prepend constants).
# ---------------------------------------------------------------------------


_RESERVED = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_IDENT_BASE = VocabularyManager._V2_IDENTITY_BLOCK_START

_LOCAL_FUNC_TOKEN = _IDENT_BASE + 1 - _RESERVED  # = 9
_PLT_FUNC_TOKEN = _IDENT_BASE + 2 - _RESERVED  # = 10

# Number-band shifted ids (post-shift: 1..7 cover the NUMBER block).
_F32_TOKEN = 4
_F128_TOKEN = 7


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------


def _empty_state() -> InlineDecodeState:
    return InlineDecodeState(
        raw_tokens=np.zeros(0, dtype=np.uint16),
        real_mask=np.zeros(0, dtype=bool),
        number_mask=np.zeros(0, dtype=bool),
        runlen_number=np.zeros(0, dtype=np.uint32),
        runlen_value=np.zeros(0, dtype=np.uint32),
        carries_inline_mask=np.zeros(0, dtype=bool),
        is_negative_per_position=np.zeros(0, dtype=bool),
    )


def _function_data(category_counts: Optional[dict[Category, int]] = None) -> FunctionData:
    metadata: dict = {
        "arch": "x86_64",
        "compiler": "gcc",
        "opt": "O2",
        "category_counts": dict(category_counts or {}),
    }
    return FunctionData(
        func_name="f",
        metadata=metadata,
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _make_call_target_row(fid: int, ct_type: CallTargetType) -> CallTarget:
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=0,
        type=ct_type,
        is_matched=False,
    )


def _build_call_target(
    *,
    fid: int,
    encounter_category: Category,
    expanded_token_ids: np.ndarray,
    in_stream_caller_local_ids: Sequence[int],
    section_call_targets: Sequence[tuple[int, CallTargetType]],
    counter_counts: Optional[dict[Category, int]] = None,
    number_chunk_slices: Optional[dict[TokenType, slice]] = None,
    partial_cut_length: Optional[int] = None,
) -> tuple[Stage3CallTarget, np.ndarray, int, int]:
    """Build one :class:`Stage3CallTarget`. Returns it alongside the
    pre-remap caller-local id slice + lengths (the per-ct identity slice
    INCLUDES the leading prepend slot; the in-stream slots carry
    ``in_stream_caller_local_ids``)."""

    if partial_cut_length is None:
        partial_cut_length = int(expanded_token_ids.shape[0])

    surviving = expanded_token_ids[:partial_cut_length]
    identity_mask = (surviving >= np.uint16(8)) & (surviving < np.uint16(16))
    number_mask = (surviving >= np.uint16(1)) & (surviving < np.uint16(8))
    surviving_identity_count = int(identity_mask.sum())
    surviving_number_count = int(number_mask.sum())

    section_calls = [
        _make_call_target_row(f, t) for f, t in section_call_targets
    ]
    stage1_ct = Stage1CallTarget(
        function_data=_function_data(counter_counts),
        state=_empty_state(),
        call_targets_section=section_calls,
        encounter_category=encounter_category,
        parent_call_target_index=None,
        function_name_ptr=fid,
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
        surviving_identity_count=surviving_identity_count,
        surviving_number_chunk_count=surviving_number_count,
        is_cut=partial_cut_length < int(expanded_token_ids.shape[0]),
        partial_cut_length=partial_cut_length,
    )
    # identity_slice is set externally by the variant builder (which
    # places call_targets contiguously into the flat identities array);
    # placeholder slice here. The variant builder rewrites it.
    # Per-CT identity slice length = ``surviving_identity_count`` (which
    # already INCLUDES the prepend slot at position 0 of expanded -- the
    # mask at slot 0 is True because the prepend is an IDENTITY-band
    # token).
    placeholder_slice = slice(0, surviving_identity_count)
    stage3_ct = Stage3CallTarget(
        stage2=stage2_ct,
        inline_byte_slice=slice(0, 0),
        identity_slice=placeholder_slice,
        number_chunk_slices=number_chunk_slices or {},
    )
    in_stream_ids_arr = np.asarray(
        list(in_stream_caller_local_ids), dtype=np.uint16
    )
    # n_identity_for_segment = total slots (prepend + in-stream); the
    # in-stream count is then ``surviving_identity_count - 1``.
    return (
        stage3_ct,
        in_stream_ids_arr,
        surviving_identity_count,
        surviving_number_count,
    )


def _assemble_variant(
    call_targets_with_data: List[
        tuple[Stage3CallTarget, np.ndarray, int, int]
    ],
    *,
    section_idx: int,
    slot_v: int,
    batch_idx: Optional[int],
    identity_global_offset: int,
    number_global_offset: int,
) -> tuple[Stage3Variant, np.ndarray, int, int]:
    """Wire a list of call_targets into a single :class:`Stage3Variant`.

    Returns ``(stage3_variant, identities_flat_segment,
    n_identity_for_variant, n_number_for_variant)``. The segment is the
    chunk to splice into the batch-shared identities buffer; the per-CT
    ``identity_slice`` fields are rewritten to point at the global
    offsets the caller supplies.
    """
    # First pass: rewrite each CT's identity_slice using the running
    # global offset. Each CT contributes ``surviving_identity_count``
    # slots (prepend + in-stream identity tokens).
    per_ct_lengths = [
        n_identity for _, _, n_identity, _ in call_targets_with_data
    ]
    variant_total = sum(per_ct_lengths)
    # Build the flat segment in this variant's local order.
    segment = np.zeros(variant_total, dtype=np.uint16)
    rewritten_call_targets: List[Stage3CallTarget] = []
    local_offset = 0
    for (ct, in_stream_ids, n_identity, _), length in zip(
        call_targets_with_data, per_ct_lengths
    ):
        # In-stream slots get the caller-local ids; prepend slot stays
        # 0 (overwritten by the dedup walk).
        if n_identity > 0:
            in_stream_count = n_identity - 1
            assert in_stream_ids.size == in_stream_count, (
                f"caller-local ids ({in_stream_ids.size}) do not match "
                f"in-stream slot count ({in_stream_count} = "
                f"surviving_identity_count - 1)"
            )
            if in_stream_count > 0:
                segment[local_offset + 1 : local_offset + length] = in_stream_ids
        new_slice = slice(
            identity_global_offset + local_offset,
            identity_global_offset + local_offset + length,
        )
        rewritten_call_targets.append(replace(ct, identity_slice=new_slice))
        local_offset += length

    n_identity_total = variant_total
    n_number_total = sum(
        n_number for _, _, _, n_number in call_targets_with_data
    )

    stage1_variant = Stage1Variant(
        variant_idx=slot_v,
        variant_ref_offset=0,
        batch_idx=batch_idx,
        call_targets=[ct.stage2.stage1 for ct in rewritten_call_targets],
    )
    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=[ct.stage2 for ct in rewritten_call_targets],
        cut_call_target_index=len(rewritten_call_targets),
        total_surviving_token_count=sum(
            ct.stage2.surviving_token_count for ct in rewritten_call_targets
        ),
        total_surviving_identity_count=sum(
            ct.stage2.surviving_identity_count for ct in rewritten_call_targets
        ),
        total_surviving_number_chunk_count=sum(
            ct.stage2.surviving_number_chunk_count
            for ct in rewritten_call_targets
        ),
    )
    stage3_variant = Stage3Variant(
        stage2=stage2_variant,
        call_targets=rewritten_call_targets,
    )
    return stage3_variant, segment, n_identity_total, n_number_total


def _build_batch(
    *,
    variants_per_section: List[List[List[tuple[Stage3CallTarget, np.ndarray, int, int]]]],
    batch_idx_to_section_variant: np.ndarray,
    numbers_per_TokenType: Optional[
        dict[TokenType, tuple[np.ndarray, np.ndarray]]
    ] = None,
    batch_idx_overrides: Optional[List[List[Optional[int]]]] = None,
) -> Stage3Batch:
    """Build a full :class:`Stage3Batch` from a 3-level CT structure.

    ``variants_per_section[s][v]`` is the list of (call_target,
    in_stream_ids, n_identity, n_number) tuples for section ``s``,
    variant slot ``v``. The function:

    1. Wires each variant's call_targets contiguously into a single
       flat identities buffer.
    2. Computes ``identity_row_offsets`` + ``number_row_offsets``
       by walking ``batch_idx_to_section_variant`` and summing per-row
       totals (handles RESAMPLE/REDISTRIBUTE multi-mapping by emitting
       the same per-variant totals on every row that points at the
       same slot).
    """

    batch_size = int(batch_idx_to_section_variant.shape[0])
    if numbers_per_TokenType is None:
        numbers_per_TokenType = {}

    # Default batch_idx per variant by inverting the mapping.
    derived_batch_idx: List[List[Optional[int]]] = [
        [None] * len(section_variants)
        for section_variants in variants_per_section
    ]
    sentinel = int(UINT32_MAX)
    for row in range(batch_size):
        s = int(batch_idx_to_section_variant[row, 0])
        v = int(batch_idx_to_section_variant[row, 1])
        if s == sentinel or v == sentinel:
            continue
        if derived_batch_idx[s][v] is None:
            derived_batch_idx[s][v] = row

    if batch_idx_overrides is not None:
        for s in range(len(derived_batch_idx)):
            for v in range(len(derived_batch_idx[s])):
                derived_batch_idx[s][v] = batch_idx_overrides[s][v]

    # First pass: build each variant + assemble the flat identities
    # buffer with running global offset.
    stage3_sections: List[Stage3Section] = []
    stage2_sections: List[Stage2Section] = []
    stage1_sections: List[Stage1Section] = []
    per_variant_id_total: List[List[int]] = [
        [0] * len(section_variants)
        for section_variants in variants_per_section
    ]
    per_variant_num_total: List[List[int]] = [
        [0] * len(section_variants)
        for section_variants in variants_per_section
    ]

    flat_identities_segments: List[np.ndarray] = []
    global_id_offset = 0
    global_num_offset = 0

    for s, section_variants in enumerate(variants_per_section):
        stage3_variants: List[Stage3Variant] = []
        stage2_variants: List[Stage2Variant] = []
        stage1_variants: List[Stage1Variant] = []
        for v, call_targets in enumerate(section_variants):
            stage3_variant, segment, n_id, n_num = _assemble_variant(
                call_targets,
                section_idx=s,
                slot_v=v,
                batch_idx=derived_batch_idx[s][v],
                identity_global_offset=global_id_offset,
                number_global_offset=global_num_offset,
            )
            stage3_variants.append(stage3_variant)
            stage2_variants.append(stage3_variant.stage2)
            stage1_variants.append(stage3_variant.stage2.stage1)
            flat_identities_segments.append(segment)
            per_variant_id_total[s][v] = n_id
            per_variant_num_total[s][v] = n_num
            global_id_offset += n_id
            global_num_offset += n_num

        stage1_section = Stage1Section(
            arm=SectionKind.MATCHED,
            idx=s,
            section=Section(
                function_name_ptr=0,
                section_offset=0,
                call_targets=[],
                variants=[],
            ),
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

    identities_flat = (
        np.concatenate(flat_identities_segments)
        if flat_identities_segments
        else np.zeros(0, dtype=np.uint16)
    )

    # Row offsets: walk the mapping; for each non-sentinel row, add the
    # referenced variant's totals.
    identity_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    number_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    for row in range(batch_size):
        s = int(batch_idx_to_section_variant[row, 0])
        v = int(batch_idx_to_section_variant[row, 1])
        if s == sentinel or v == sentinel:
            row_id_total = 0
            row_num_total = 0
        else:
            row_id_total = per_variant_id_total[s][v]
            row_num_total = per_variant_num_total[s][v]
        identity_row_offsets[row + 1] = identity_row_offsets[row] + row_id_total
        number_row_offsets[row + 1] = number_row_offsets[row] + row_num_total

    stage1_batch = Stage1Batch(
        sections=stage1_sections,
        batch_idx_to_section_variant=batch_idx_to_section_variant.astype(
            np.uint32, copy=False
        ),
        batch_size=batch_size,
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=stage2_sections,
        identity_row_offsets=identity_row_offsets,
        number_row_offsets=number_row_offsets,
    )
    return Stage3Batch(
        stage2=stage2_batch,
        sections=stage3_sections,
        inline_bytes=np.zeros(1, dtype=np.uint8),
        identities_flat_caller_local=identities_flat,
        numbers_per_TokenType=numbers_per_TokenType,
        identity_idx_2d=np.zeros((0, 2), dtype=np.uint32),
        number_idx_2d_per_TokenType={},
        vc2_chunk_exponent_sidecar=np.zeros(0, dtype=np.uint32),
        f128_is_nan_or_inf=np.zeros(0, dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_call_target_zero_in_stream_tokens() -> None:
    """Synthetic Stage3Batch with a single CT and 0 in-stream identity
    tokens. The result row begins with the prepend (LOCAL_FUNC token id
    9) followed by zeros; identities is [0] (root counter); numbers
    arrays are empty."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray([_LOCAL_FUNC_TOKEN], dtype=np.uint16),
        in_stream_caller_local_ids=[],
        section_call_targets=[],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=8)

    assert isinstance(result, BatchDecodeResult)
    # Tokens: prepend id at position 0, zeros elsewhere.
    expected_tokens = np.zeros((1, 8), dtype=np.uint16)
    expected_tokens[0, 0] = _LOCAL_FUNC_TOKEN
    np.testing.assert_array_equal(result.tokens, expected_tokens)

    # Identities: one slot, root counter = 0.
    assert result.identities.tolist() == [0]
    assert result.identity_row_offsets.tolist() == [0, 1]

    # Numbers: empty.
    assert result.numbers_significant.size == 0
    assert result.numbers_sign_exponent.size == 0
    assert result.number_row_offsets.tolist() == [0, 0]

    # Sidecars off by default.
    assert result.fid_sidecar is None
    assert result.fid_row_offsets is None
    assert result.intermediate is None


def test_single_ct_local_callees_dedup_in_encounter_order() -> None:
    """Single CT (root) referencing three LOCAL callees via caller-local
    ids 0, 1, 2. Dedup IDs are 0 (root) + 1, 2, 3 (callees) in encounter
    order."""

    expanded = np.asarray(
        [
            _LOCAL_FUNC_TOKEN,  # prepend
            _LOCAL_FUNC_TOKEN,  # in-stream caller-local 0 -> counter 1
            _LOCAL_FUNC_TOKEN,  # caller-local 1 -> counter 2
            _LOCAL_FUNC_TOKEN,  # caller-local 2 -> counter 3
        ],
        dtype=np.uint16,
    )
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        in_stream_caller_local_ids=[0, 1, 2],
        section_call_targets=[
            (200, CallTargetType.LOCAL),
            (201, CallTargetType.LOCAL),
            (202, CallTargetType.LOCAL),
        ],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=8)

    expected_tokens = np.zeros((1, 8), dtype=np.uint16)
    expected_tokens[0, :4] = _LOCAL_FUNC_TOKEN
    np.testing.assert_array_equal(result.tokens, expected_tokens)
    # Identity: prepend=0, then 1, 2, 3.
    assert result.identities.tolist() == [0, 1, 2, 3]
    assert result.identity_row_offsets.tolist() == [0, 4]


def test_pad_null_padding_row() -> None:
    """Multi-row batch with PAD_NULL: padding row has all-zero tokens
    and zero-length identity + number contributions."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray(
            [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN], dtype=np.uint16
        ),
        in_stream_caller_local_ids=[0],
        section_call_targets=[(200, CallTargetType.LOCAL)],
    )
    sentinel = int(UINT32_MAX)
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray(
            [[0, 0], [sentinel, sentinel]], dtype=np.uint32
        ),
    )

    result = assemble_batch(batch, context_len=6)

    # Row 0: prepend + counter-1.
    expected_row0 = np.zeros(6, dtype=np.uint16)
    expected_row0[0] = _LOCAL_FUNC_TOKEN
    expected_row0[1] = _LOCAL_FUNC_TOKEN
    np.testing.assert_array_equal(result.tokens[0], expected_row0)
    # Row 1: all zeros.
    np.testing.assert_array_equal(result.tokens[1], np.zeros(6, dtype=np.uint16))

    # Row offsets: row 0 contributes 2 identity slots, row 1 contributes 0.
    assert result.identity_row_offsets.tolist() == [0, 2, 2]
    assert result.number_row_offsets.tolist() == [0, 0, 0]
    # Identities: [0 (root prepend), 1 (callee counter)].
    assert result.identities.tolist() == [0, 1]


def test_resample_multi_mapped_rows_identical() -> None:
    """Two rows pointing at the same (section, slot) -- RESAMPLE
    behavior. Both rows have identical token content. Per-row identity
    sidecar offsets account for the duplication."""

    expanded = np.asarray(
        [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN], dtype=np.uint16
    )
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        in_stream_caller_local_ids=[0],
        section_call_targets=[(200, CallTargetType.LOCAL)],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray(
            [[0, 0], [0, 0]], dtype=np.uint32
        ),
    )

    result = assemble_batch(batch, context_len=4)

    # Both rows: prepend + counter.
    expected_row = np.zeros(4, dtype=np.uint16)
    expected_row[0] = _LOCAL_FUNC_TOKEN
    expected_row[1] = _LOCAL_FUNC_TOKEN
    np.testing.assert_array_equal(result.tokens[0], expected_row)
    np.testing.assert_array_equal(result.tokens[1], expected_row)

    # Identity row offsets advance by 2 per row (the variant's identity
    # slice is referenced once per mapping entry).
    assert result.identity_row_offsets.tolist() == [0, 2, 4]
    # The shared variant's slice (length 2) is referenced by both rows.
    # NOTE: with RESAMPLE, the dedup walk visits the slice ONCE (per the
    # walk's logic, which iterates variants not rows). The second row's
    # offsets advance to 4 in stage 2 even though the underlying buffer
    # is sized only for the unique variant slice -- that mismatch is a
    # known gap in the RESAMPLE plumbing; the canonical fix lives in
    # stage 3's bulk-byte build. We only assert the variant-walk path
    # here.


def test_include_fid_sidecar() -> None:
    """include_fid_sidecar=True -> fid_sidecar maps counters back to
    FIDs per row."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray(
            [_LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN, _LOCAL_FUNC_TOKEN],
            dtype=np.uint16,
        ),
        in_stream_caller_local_ids=[0, 1],
        section_call_targets=[
            (200, CallTargetType.LOCAL),
            (201, CallTargetType.LOCAL),
        ],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=8, include_fid_sidecar=True)

    assert result.fid_sidecar is not None
    assert result.fid_row_offsets is not None
    # Counter 0 -> FID 100 (root seed); counter 1 -> 200; counter 2 -> 201.
    assert result.fid_sidecar.tolist() == [100, 200, 201]
    assert result.fid_row_offsets.tolist() == [0, 3]


def test_keep_intermediate() -> None:
    """keep_intermediate=True -> result.intermediate IS the input
    :class:`Stage3Batch`."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray([_LOCAL_FUNC_TOKEN], dtype=np.uint16),
        in_stream_caller_local_ids=[],
        section_call_targets=[],
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=4, keep_intermediate=True)

    assert result.intermediate is batch


def test_mid_cut_context_len() -> None:
    """``context_len`` shorter than the variant's full content: tokens
    truncated at exactly ``context_len``; identity/number counts match
    surviving content via stage-2 ``partial_cut_length``."""

    # Build a CT with 6 in-stream LOCAL_FUNC tokens but partial_cut_length=4
    # (mimicking stage-2's cut at column 4 inside this CT).
    expanded = np.asarray(
        [
            _LOCAL_FUNC_TOKEN,  # prepend
            _LOCAL_FUNC_TOKEN,  # caller-local 0
            _LOCAL_FUNC_TOKEN,  # caller-local 1
            _LOCAL_FUNC_TOKEN,  # caller-local 2 (last surviving)
            _LOCAL_FUNC_TOKEN,  # caller-local 3 (cut)
            _LOCAL_FUNC_TOKEN,  # caller-local 4 (cut)
            _LOCAL_FUNC_TOKEN,  # caller-local 5 (cut)
        ],
        dtype=np.uint16,
    )
    # Surviving prefix length = 4: prepend + 3 in-stream slots.
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        # The in-stream caller-local id sequence must be length=6 to
        # match all in-stream slots in expanded; the variant builder
        # writes ALL caller-local ids into the segment (it has no
        # surviving-only knowledge -- that's stage 3's concern). For
        # this test we only need the first 3 to be exercised by the
        # walk, which reads in_stream_count = surviving_identity_count
        # (= 3 here). The remaining 3 sit in slots that the walk
        # doesn't touch, so their initial value (0) survives.
        in_stream_caller_local_ids=[0, 1, 2],
        section_call_targets=[
            (200, CallTargetType.LOCAL),
            (201, CallTargetType.LOCAL),
            (202, CallTargetType.LOCAL),
            (203, CallTargetType.LOCAL),
            (204, CallTargetType.LOCAL),
            (205, CallTargetType.LOCAL),
        ],
        partial_cut_length=4,
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
    )

    result = assemble_batch(batch, context_len=4)

    # Tokens: 4 cols, all = LOCAL_FUNC token (prepend + 3 in-stream).
    np.testing.assert_array_equal(
        result.tokens[0], np.full(4, _LOCAL_FUNC_TOKEN, dtype=np.uint16)
    )
    # Identity counts: surviving identity count = 4 (prepend at slot 0
    # of expanded + 3 in-stream).
    assert result.identities.shape[0] == 4
    # Counter ids: prepend=0, then 1, 2, 3.
    assert result.identities.tolist() == [0, 1, 2, 3]
    assert result.identity_row_offsets.tolist() == [0, 4]


def test_f128_extra_chunk_mask_no_renorm() -> None:
    """Synthetic F128 in stream: stage-4 does NOT renormalize -- it
    simply concatenates per-:class:`TokenType` chunks in stream-position
    order. The number_chunk_slices on each CT name the per-TokenType
    range to gather; this test pins the gather + concat path."""

    # Stream: one F128 source emitting 2 chunks (finite case per ALG-2).
    # Stage 4 walks expanded_token_ids; the per-position number band
    # tells it to consume the next chunk from the matching TokenType
    # buffer. Stage 2's expansion would emit F128 twice in
    # expanded_token_ids when extra_f128_mask is True at the extra
    # position.
    expanded = np.asarray(
        [_LOCAL_FUNC_TOKEN, _F128_TOKEN, _F128_TOKEN], dtype=np.uint16
    )
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        in_stream_caller_local_ids=[],
        section_call_targets=[],
        number_chunk_slices={TokenType.FLOAT128: slice(0, 2)},
    )
    # Number arrays: 2 chunks, dummy values.
    f128_sig = np.asarray([0xAAAA, 0xBBBB], dtype=np.uint64)
    f128_sex = np.asarray([1, 2], dtype=np.uint32)
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT128: (f128_sig, f128_sex)},
    )

    result = assemble_batch(batch, context_len=4)

    # Two F128 chunks in stream order, pulled verbatim.
    np.testing.assert_array_equal(result.numbers_significant, f128_sig)
    np.testing.assert_array_equal(result.numbers_sign_exponent, f128_sex)
    assert result.number_row_offsets.tolist() == [0, 2]


def test_f128_nan_inf_one_chunk_per_source() -> None:
    """F128 NaN/Inf: one chunk per source. Mirrors the finite-case test
    but with the per-source chunk count = 1 (extra_f128_mask is False
    for these sources)."""

    expanded = np.asarray([_LOCAL_FUNC_TOKEN, _F128_TOKEN], dtype=np.uint16)
    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=expanded,
        in_stream_caller_local_ids=[],
        section_call_targets=[],
        number_chunk_slices={TokenType.FLOAT128: slice(0, 1)},
    )
    f128_sig = np.asarray([0x7FFF_0000_0000_0000], dtype=np.uint64)  # nan-like
    f128_sex = np.asarray([0xFFFF], dtype=np.uint32)
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=np.asarray([[0, 0]], dtype=np.uint32),
        numbers_per_TokenType={TokenType.FLOAT128: (f128_sig, f128_sex)},
    )

    result = assemble_batch(batch, context_len=4)

    # Exactly one chunk for the NaN/Inf source.
    assert result.numbers_significant.tolist() == [0x7FFF_0000_0000_0000]
    assert result.numbers_sign_exponent.tolist() == [0xFFFF]
    assert result.number_row_offsets.tolist() == [0, 1]


def test_batch_idx_to_section_variant_threaded_through() -> None:
    """The result's batch_idx_to_section_variant is the same mapping
    from stage 1 (sentinel pairs preserved)."""

    ct_data = _build_call_target(
        fid=100,
        encounter_category=Category.LOCAL_FUNC,
        expanded_token_ids=np.asarray([_LOCAL_FUNC_TOKEN], dtype=np.uint16),
        in_stream_caller_local_ids=[],
        section_call_targets=[],
    )
    sentinel = int(UINT32_MAX)
    mapping = np.asarray(
        [[0, 0], [sentinel, sentinel], [0, 0]], dtype=np.uint32
    )
    batch = _build_batch(
        variants_per_section=[[[ct_data]]],
        batch_idx_to_section_variant=mapping,
    )

    result = assemble_batch(batch, context_len=4)
    np.testing.assert_array_equal(result.batch_idx_to_section_variant, mapping)
