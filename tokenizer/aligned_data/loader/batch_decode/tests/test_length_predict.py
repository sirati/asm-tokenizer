"""Stage 2d orchestrator integration tests.

Single concern: pin the ``Stage 2 = 2a + 2b + 2c`` composition contract
(:func:`predict_lengths`) on synthetic :class:`Stage1Batch` fixtures. The
underlying algorithm modules (``_expand_tokens``, ``_cutoff_walk``,
``_surviving_counts``) have their own per-concern unit tests; here we
exercise:

* Single CT with no cut -> fully included; counts match the
  surviving-count expectation.
* Multi-CT chain with a mid-cut -> cut CT carries
  ``partial_cut_length < predicted_full_length``; trailing CTs are
  dropped (``surviving_token_count == 0``).
* Empty variant (zero CTs) -> handled gracefully.
* VC2 + F128 mixed in the stream -> ``expanded_token_ids`` matches 2a's
  expected shape; chunk counts respected end-to-end.
* Row offsets shape + monotonicity + final-offset equals total
  surviving counts across the whole batch.
* Multi-row mapping (RESAMPLE / REDISTRIBUTE-style) -> each row gets
  the same per-variant total when distinct batch rows aim at one slot.
* Padding rows (``UINT32_MAX`` sentinel) -> contribute 0 to the row
  cumsum.
* End-to-end pipeline drive: a multi-section, multi-variant,
  multi-CT fixture validates the full Stage2Batch shape + back-pointer
  chain navigation.

All fixtures construct :class:`Stage1Batch` directly -- ``walk_sections``
(stage 1) is still a stub at the time of writing, so a real end-to-end
through it is not available; the orchestrator's invariants are
nonetheless pinned by composing the same shape it would produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    _FLOAT128_VOCAB_ID,
    _LOCAL_FUNC_SHIFTED,
    _PLT_FUNC_SHIFTED,
    _V2_IDENTITY_BLOCK_START,
    _V2_RESERVED_DIGIT_COUNT,
    _VC2_VOCAB_ID,
)
from tokenizer.aligned_data.loader.batch_decode._length_predict import (
    predict_lengths,
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
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Fixture builders -- minimal-but-typed dummies.
#
# We rebuild a clean Stage1Batch from synthetic raw token streams; the
# orchestrator only walks the dataclass hierarchy + calls the three
# sibling modules, so anything those modules don't read is filled with
# zero-shaped arrays / placeholder integers.
# ---------------------------------------------------------------------------


def _u16(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.uint16)


def _empty_function_data() -> FunctionData:
    return FunctionData(
        func_name="dummy",
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _build_state(raw_tokens: np.ndarray) -> InlineDecodeState:
    """Same shape as ``test_expand_tokens``'s ``_build_state``."""
    real_mask = raw_tokens > _V2_RESERVED_DIGIT_COUNT
    number_mask = raw_tokens < _V2_RESERVED_DIGIT_COUNT
    if raw_tokens.shape[0] == 0:
        runlen_number = np.zeros(0, dtype=np.uint16)
        runlen_value = np.zeros(0, dtype=np.uint16)
    else:
        runlen_number = run_lengths(number_mask)
        runlen_value = run_lengths(~real_mask)
    carries_inline_mask = real_mask & (
        raw_tokens < _V2_IDENTITY_BLOCK_START + 8
    )
    is_negative_per_position = np.zeros(raw_tokens.shape[0], dtype=bool)
    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
    )


def _make_call_target(
    raw_tokens: np.ndarray,
    *,
    encounter_category: Category = Category.LOCAL_FUNC,
) -> Stage1CallTarget:
    return Stage1CallTarget(
        function_data=_empty_function_data(),
        state=_build_state(raw_tokens),
        call_targets_section=[],
        encounter_category=encounter_category,
        parent_call_target_index=None,
        function_name_ptr=0,
    )


def _make_section(call_targets_lists: list) -> Section:
    return Section(
        function_name_ptr=42,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _make_variant(
    raw_streams: list[np.ndarray],
    *,
    slot_v: int,
    categories: list[Category] | None = None,
) -> Stage1Variant:
    """Build a Stage1Variant from a list of raw token streams (one per CT).

    The variant's ``batch_idx`` is set to ``slot_v`` for ergonomics --
    note the orchestrator does NOT consult :attr:`Stage1Variant.batch_idx`
    for the row-offset cumsum; it walks
    ``batch_idx_to_section_variant`` instead. The field is set for
    consumers that DO care about it (e.g. tests that prove the
    back-pointer chain is intact)."""
    if categories is None:
        categories = [Category.LOCAL_FUNC] * len(raw_streams)
    cts = [
        _make_call_target(raw, encounter_category=cat)
        for raw, cat in zip(raw_streams, categories)
    ]
    return Stage1Variant(
        variant_idx=slot_v,
        variant_ref_offset=slot_v,
        batch_idx=slot_v,
        call_targets=cts,
    )


def _make_section_obj(
    variant_specs: list[list[np.ndarray]],
    *,
    section_idx: int,
    variant_categories: list[list[Category]] | None = None,
) -> Stage1Section:
    variants = []
    for slot_v, streams in enumerate(variant_specs):
        cats = (
            variant_categories[slot_v]
            if variant_categories is not None
            else None
        )
        variants.append(_make_variant(streams, slot_v=slot_v, categories=cats))
    return Stage1Section(
        arm=SectionKind.MATCHED,
        idx=section_idx,
        section=Section(
            function_name_ptr=section_idx,
            section_offset=section_idx * 100,
            call_targets=[],
            variants=[],
        ),
        variants=variants,
    )


def _make_pad_null_mapping(
    section_variant_counts: list[int],
    *,
    num_variants_per_section: int,
) -> tuple[np.ndarray, int]:
    """Mirror :func:`_batch_layout._layout_pad_null` for tests.

    For each section, the first ``real`` slots hold ``(section, slot)``;
    the trailing ``num_variants_per_section - real`` are
    ``(UINT32_MAX, UINT32_MAX)``.
    """
    num_sections = len(section_variant_counts)
    batch_size = num_sections * num_variants_per_section
    mapping = np.full((batch_size, 2), UINT32_MAX, dtype=np.uint32)
    for s, real in enumerate(section_variant_counts):
        for slot in range(min(real, num_variants_per_section)):
            mapping[s * num_variants_per_section + slot, 0] = np.uint32(s)
            mapping[s * num_variants_per_section + slot, 1] = np.uint32(slot)
    return mapping, batch_size


def _make_batch(
    sections_payload: list[list[list[np.ndarray]]],
    *,
    num_variants_per_section: int,
    variant_categories: list[list[list[Category]]] | None = None,
    mapping_override: np.ndarray | None = None,
) -> Stage1Batch:
    """Assemble a Stage1Batch from raw streams.

    ``sections_payload[section_idx][slot_v][ct_idx]`` -> raw u16 stream
    for that call_target. By default uses PAD_NULL semantics; pass
    ``mapping_override`` to test multi-row mappings / padding rows.
    """
    sections = []
    for s, variant_specs in enumerate(sections_payload):
        cats = (
            variant_categories[s] if variant_categories is not None else None
        )
        sections.append(
            _make_section_obj(
                variant_specs,
                section_idx=s,
                variant_categories=cats,
            )
        )

    if mapping_override is not None:
        mapping = mapping_override
        batch_size = int(mapping.shape[0])
    else:
        real_counts = [len(specs) for specs in sections_payload]
        mapping, batch_size = _make_pad_null_mapping(
            real_counts,
            num_variants_per_section=num_variants_per_section,
        )

    return Stage1Batch(
        sections=sections,
        batch_idx_to_section_variant=mapping,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Test 1: single CT, no cut -> fully included.
# ---------------------------------------------------------------------------


def test_single_ct_no_cut_fully_included() -> None:
    """One variant with one CT carrying a few identity tokens; context_len
    far exceeds the function length so no cut occurs.

    Expectations:
      * ``predicted_full_length`` = 1 prepend + 2 identity carriers = 3.
      * ``surviving_token_count == predicted_full_length``.
      * ``is_cut`` is False; ``partial_cut_length == surviving_token_count``.
      * ``cut_call_target_index == len(call_targets)`` (sentinel past-end).
      * ``surviving_identity_count == 3`` (prepend + 2 identities).
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START  # 264
    LOCAL_FUNC = _V2_IDENTITY_BLOCK_START + 1  # 265
    raw = _u16(BLOCK_V2, LOCAL_FUNC)

    stage1 = _make_batch(
        [[[raw]]],
        num_variants_per_section=1,
    )
    stage2 = predict_lengths(stage1, context_len=100)

    assert isinstance(stage2, Stage2Batch)
    variant = stage2.sections[0].variants[0]
    ct = variant.call_targets[0]

    assert ct.predicted_full_length == 3  # prepend + 2 identities
    assert ct.surviving_token_count == 3
    assert ct.is_cut is False
    assert ct.partial_cut_length == 3
    assert ct.surviving_identity_count == 3
    assert ct.surviving_number_chunk_count == 0
    assert variant.cut_call_target_index == 1  # past-the-end sentinel
    assert variant.total_surviving_token_count == 3
    assert variant.total_surviving_identity_count == 3
    assert variant.total_surviving_number_chunk_count == 0


# ---------------------------------------------------------------------------
# Test 2: multi-CT chain with a mid-stream cut.
# ---------------------------------------------------------------------------


def test_multi_ct_chain_mid_cut() -> None:
    """Three CTs all length 3 (prepend + 2 identities each); context_len=4.

    Cumsum [3,6,9]; first cumsum > 4 is index 1 (cumsum=6), so cut on
    index 1 with partial = 4 - 3 = 1.

    Expectations:
      * CT0 fully included (surviving == 3).
      * CT1 is the cut entry: ``partial_cut_length=1``, ``is_cut=True``,
        ``surviving_token_count=1``.
      * CT2 dropped: surviving == 0, no is_cut.
      * Variant cut idx == 1; total surviving = 3+1+0 = 4 == context_len.
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    LOCAL_FUNC = _V2_IDENTITY_BLOCK_START + 1
    raw = _u16(BLOCK_V2, LOCAL_FUNC)  # each CT yields 3 tokens after expand

    stage1 = _make_batch(
        [[[raw, raw, raw]]],
        num_variants_per_section=1,
    )
    stage2 = predict_lengths(stage1, context_len=4)

    variant = stage2.sections[0].variants[0]
    cts = variant.call_targets

    assert variant.cut_call_target_index == 1

    # CT0: fully included.
    assert cts[0].surviving_token_count == 3
    assert cts[0].is_cut is False
    assert cts[0].partial_cut_length == 3

    # CT1: cut entry.
    assert cts[1].is_cut is True
    assert cts[1].surviving_token_count == 1
    assert cts[1].partial_cut_length == 1
    assert cts[1].partial_cut_length < cts[1].predicted_full_length

    # CT2: dropped.
    assert cts[2].surviving_token_count == 0
    assert cts[2].is_cut is False
    assert cts[2].partial_cut_length == 0

    # Total surviving = exact fill of context_len.
    assert variant.total_surviving_token_count == 4


# ---------------------------------------------------------------------------
# Test 3: empty variant (zero CTs).
# ---------------------------------------------------------------------------


def test_empty_variant_zero_call_targets() -> None:
    """A variant with no call_targets walks cleanly: empty cutoff result
    + zero totals + sentinel cut idx (=0=len)."""

    stage1 = _make_batch(
        [[[]]],  # one section, one variant, zero CTs
        num_variants_per_section=1,
    )
    stage2 = predict_lengths(stage1, context_len=10)

    variant = stage2.sections[0].variants[0]
    assert variant.call_targets == []
    assert variant.cut_call_target_index == 0  # = len(call_targets)
    assert variant.total_surviving_token_count == 0
    assert variant.total_surviving_identity_count == 0
    assert variant.total_surviving_number_chunk_count == 0

    # Row offsets are well-formed for the trivial 1-row batch.
    assert stage2.identity_row_offsets.shape == (2,)
    assert stage2.number_row_offsets.shape == (2,)
    assert stage2.identity_row_offsets[-1] == 0
    assert stage2.number_row_offsets[-1] == 0


# ---------------------------------------------------------------------------
# Test 4: VC2 + F128 mix; expanded length + chunk counts respected.
# ---------------------------------------------------------------------------


def test_vc2_and_f128_mixed_stream() -> None:
    """One CT carrying:
       * VC2 carrier + 9 byte payload (chunk_count = ceil(9/8) = 2)
       * F128 finite carrier + 16-byte payload (chunk_count = 2)
       * BLOCK_V2 trailing identity carrier (no inline payload)

    After expand:
      * 1 prepend (LOCAL_FUNC)
      * 2 VC2 chunks (carrier + 1 promoted)
      * 2 F128 chunks (carrier + 1 promoted)
      * 1 BLOCK_V2 carrier
      = 6 tokens total. 4 number-band + 1 identity-band + 1 prepend-identity.
      Total identity = 2 (prepend + BLOCK_V2); total number_chunk = 4.
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    stream = [
        _VC2_VOCAB_ID,
        # 9 byte payload:
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
        _FLOAT128_VOCAB_ID,
        0x40, 0x00,  # finite high u16
        *([0x00] * 14),
        BLOCK_V2,
    ]
    raw = _u16(*stream)

    stage1 = _make_batch(
        [[[raw]]],
        num_variants_per_section=1,
    )
    stage2 = predict_lengths(stage1, context_len=100)

    ct = stage2.sections[0].variants[0].call_targets[0]
    assert ct.predicted_full_length == 6
    assert ct.surviving_token_count == 6
    assert ct.surviving_identity_count == 2  # prepend (id 9) + BLOCK_V2 (id 8)
    assert ct.surviving_number_chunk_count == 4  # 2 VC2 + 2 F128

    # extra masks shape matches expanded length.
    assert ct.extra_value_v2_mask.shape == (6,)
    assert ct.extra_f128_mask.shape == (6,)
    # Exactly 1 promoted VC2 + 1 promoted F128.
    assert int(ct.extra_value_v2_mask.sum()) == 1
    assert int(ct.extra_f128_mask.sum()) == 1


# ---------------------------------------------------------------------------
# Test 5: row offsets shape + monotonicity + final equals total.
# ---------------------------------------------------------------------------


def test_row_offsets_shape_monotonic_and_total() -> None:
    """Multi-section / multi-variant fixture: build a 2-section x
    2-variants-per-section batch (batch_size=4). Each CT carries a few
    identity tokens + a VC2 number token, with varied lengths per
    variant. Validate the resulting row-offset arrays:

      * shape == (batch_size + 1,)
      * dtype == u32
      * offsets[0] == 0
      * non-decreasing (monotone)
      * final == sum of per-variant totals across the batch
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    LOCAL_FUNC = _V2_IDENTITY_BLOCK_START + 1
    PLT_FUNC = _V2_IDENTITY_BLOCK_START + 2

    # Variant 0,0: one CT with one identity (BLOCK_V2)
    raw_0_0 = _u16(BLOCK_V2)
    # Variant 0,1: one CT with BLOCK_V2 + LOCAL_FUNC + PLT_FUNC carriers
    raw_0_1 = _u16(BLOCK_V2, LOCAL_FUNC, PLT_FUNC)
    # Variant 1,0: empty CT (just the prepend)
    raw_1_0 = _u16()
    # Variant 1,1: VC2 carrier + 1 byte payload (chunk=1) + BLOCK_V2
    raw_1_1 = _u16(_VC2_VOCAB_ID, 0x42, BLOCK_V2)

    stage1 = _make_batch(
        [
            [[raw_0_0], [raw_0_1]],
            [[raw_1_0], [raw_1_1]],
        ],
        num_variants_per_section=2,
    )
    stage2 = predict_lengths(stage1, context_len=100)

    batch_size = stage1.batch_size
    assert batch_size == 4
    assert stage2.identity_row_offsets.shape == (batch_size + 1,)
    assert stage2.number_row_offsets.shape == (batch_size + 1,)
    assert stage2.identity_row_offsets.dtype == np.uint32
    assert stage2.number_row_offsets.dtype == np.uint32

    # offsets[0] == 0, non-decreasing.
    assert int(stage2.identity_row_offsets[0]) == 0
    assert int(stage2.number_row_offsets[0]) == 0
    for i in range(batch_size):
        assert (
            stage2.identity_row_offsets[i + 1]
            >= stage2.identity_row_offsets[i]
        )
        assert (
            stage2.number_row_offsets[i + 1]
            >= stage2.number_row_offsets[i]
        )

    # Final offset == sum of per-variant totals (in mapping order). With
    # PAD_NULL + dense sampling the mapping order matches section/slot
    # nesting.
    total_identity = sum(
        v.total_surviving_identity_count
        for sec in stage2.sections
        for v in sec.variants
    )
    total_number = sum(
        v.total_surviving_number_chunk_count
        for sec in stage2.sections
        for v in sec.variants
    )
    assert int(stage2.identity_row_offsets[-1]) == total_identity
    assert int(stage2.number_row_offsets[-1]) == total_number


# ---------------------------------------------------------------------------
# Test 6: multi-row mapping (RESAMPLE-style) -> each row picks up the
# same per-variant total.
# ---------------------------------------------------------------------------


def test_multi_row_mapping_each_row_contributes_variant_total() -> None:
    """RESAMPLE / REDISTRIBUTE policies can point multiple batch rows
    at the same ``(section_idx, slot_v)`` (no duplicate
    :class:`Stage1Variant` instances; the mapping just repeats).

    Fixture: one section with one variant; the batch has 3 rows all
    aimed at (0, 0). The variant carries 2 identities, so each row
    contributes 2 to the identity cumsum.
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    LOCAL_FUNC = _V2_IDENTITY_BLOCK_START + 1
    raw = _u16(BLOCK_V2, LOCAL_FUNC)  # CT length 3 = 1 prepend + 2 identities

    # 3 rows all mapped to (section=0, slot=0).
    mapping = np.zeros((3, 2), dtype=np.uint32)
    stage1 = _make_batch(
        [[[raw]]],
        num_variants_per_section=1,
        mapping_override=mapping,
    )
    stage2 = predict_lengths(stage1, context_len=100)

    # The single Stage1Variant produced 3 identity tokens (prepend +
    # BLOCK_V2 + LOCAL_FUNC).
    variant = stage2.sections[0].variants[0]
    assert variant.total_surviving_identity_count == 3

    # Each of the 3 rows must contribute 3 to the cumsum.
    expected = np.array([0, 3, 6, 9], dtype=np.uint32)
    np.testing.assert_array_equal(stage2.identity_row_offsets, expected)
    # Numbers all 0 (no number carriers in the stream).
    np.testing.assert_array_equal(
        stage2.number_row_offsets, np.zeros(4, dtype=np.uint32)
    )


# ---------------------------------------------------------------------------
# Test 7: padding rows (UINT32_MAX sentinel) contribute 0.
# ---------------------------------------------------------------------------


def test_padding_rows_contribute_zero() -> None:
    """PAD_NULL fixture: one section with 1 real variant + 2 padding
    rows (num_variants_per_section=3). The two trailing rows hold the
    UINT32_MAX sentinel and must add 0 to the cumsum.
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    raw = _u16(BLOCK_V2)  # 1 prepend + 1 BLOCK_V2 -> 2 identities

    stage1 = _make_batch(
        [[[raw]]],  # one section, one real variant
        num_variants_per_section=3,  # 2 padding rows trailing
    )
    stage2 = predict_lengths(stage1, context_len=100)

    # Row 0: real variant -> 2 identities.
    # Rows 1, 2: padding -> 0 identities each.
    assert stage1.batch_size == 3
    expected = np.array([0, 2, 2, 2], dtype=np.uint32)
    np.testing.assert_array_equal(stage2.identity_row_offsets, expected)
    np.testing.assert_array_equal(
        stage2.number_row_offsets, np.zeros(4, dtype=np.uint32)
    )


def test_padding_row_only_one_axis_sentinel() -> None:
    """Defensive: if only ONE of the two columns equals UINT32_MAX
    (shouldn't happen per the layout contract, but the sentinel check
    is OR'd so it tolerates either-column corruption), the row is
    treated as padding."""

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    raw = _u16(BLOCK_V2)
    # Hand-craft a mapping with a half-sentinel row to ensure the OR
    # semantics behave: row 0 real, row 1 has section=0 but slot=MAX.
    mapping = np.array(
        [
            [0, 0],
            [0, int(UINT32_MAX)],
        ],
        dtype=np.uint32,
    )
    stage1 = _make_batch(
        [[[raw]]],
        num_variants_per_section=1,
        mapping_override=mapping,
    )
    stage2 = predict_lengths(stage1, context_len=100)

    # Row 0 contributes 2; row 1 (half-sentinel) contributes 0.
    expected = np.array([0, 2, 2], dtype=np.uint32)
    np.testing.assert_array_equal(stage2.identity_row_offsets, expected)


# ---------------------------------------------------------------------------
# Test 8: end-to-end full Stage 2 pipeline drive.
#
# walk_sections (stage 1) is still a stub at the time of writing -- a
# real call into it would raise NotImplementedError. The orchestrator's
# end-to-end contract is therefore exercised by driving predict_lengths
# directly on a richly-shaped synthetic Stage1Batch and validating the
# resulting Stage2Batch end-to-end (shape, back-pointer chain,
# aggregation, cumsum consistency).
# ---------------------------------------------------------------------------


def test_end_to_end_full_pipeline_drive() -> None:
    """Multi-section, multi-variant, multi-CT, mixed-content end-to-end.

    Two sections, each with two variants. Each variant has 2 CTs to
    exercise the inlined-callee chain. Streams mix VC2, F128, and
    identity carriers so the surviving-count masks see all three bands.

    Validates:
      * Stage2Batch dataclass populated (back-pointers chain to stage1).
      * Section / variant / call_target counts match stage1.
      * Per-variant totals = sum of per-CT counts (aggregation invariant).
      * Per-row cumsum agrees with the per-variant totals (mapping walk
        invariant).
      * For each surviving CT, ``surviving_token_count <=
        predicted_full_length`` and equals it iff ``not is_cut``.
    """

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    LOCAL_FUNC = _V2_IDENTITY_BLOCK_START + 1

    # CT shapes (raw -> expanded length):
    #   plain identity stream:    [BLOCK_V2]            -> 2 tokens
    #   identity + VC2 source:    [BLOCK_V2, VC2, 0x42] -> 3 tokens
    #                              (VC2 L=1 -> 1 chunk; BLOCK_V2 + VC2)
    #   F128 finite + identity:   [F128, 0x40, 0x00, *14 zeros, LOCAL_FUNC]
    #                              -> 1 + 2 + 1 = 4 tokens
    raw_a = _u16(BLOCK_V2)
    raw_b = _u16(BLOCK_V2, _VC2_VOCAB_ID, 0x42)
    raw_c = _u16(
        _FLOAT128_VOCAB_ID, 0x40, 0x00,
        *([0x00] * 14),
        LOCAL_FUNC,
    )

    # Section 0: variant 0 = [raw_a, raw_b], variant 1 = [raw_c]
    # Section 1: variant 0 = [raw_b, raw_c], variant 1 = [raw_a, raw_a, raw_a]
    sections_payload = [
        [
            [raw_a, raw_b],
            [raw_c],
        ],
        [
            [raw_b, raw_c],
            [raw_a, raw_a, raw_a],
        ],
    ]
    # Mix PLT_FUNC and LOCAL_FUNC categories to exercise the
    # encounter_category dispatch end-to-end.
    variant_categories = [
        [
            [Category.LOCAL_FUNC, Category.PLT_FUNC],
            [Category.LOCAL_FUNC],
        ],
        [
            [Category.PLT_FUNC, Category.LOCAL_FUNC],
            [Category.LOCAL_FUNC, Category.PLT_FUNC, Category.LOCAL_FUNC],
        ],
    ]

    stage1 = _make_batch(
        sections_payload,
        num_variants_per_section=2,
        variant_categories=variant_categories,
    )
    stage2 = predict_lengths(stage1, context_len=200)

    # ----- structural -----
    assert isinstance(stage2, Stage2Batch)
    assert stage2.stage1 is stage1
    assert len(stage2.sections) == len(stage1.sections)
    for s2_sec, s1_sec in zip(stage2.sections, stage1.sections):
        assert isinstance(s2_sec, Stage2Section)
        assert s2_sec.stage1 is s1_sec
        assert len(s2_sec.variants) == len(s1_sec.variants)
        for s2_var, s1_var in zip(s2_sec.variants, s1_sec.variants):
            assert isinstance(s2_var, Stage2Variant)
            assert s2_var.stage1 is s1_var
            assert len(s2_var.call_targets) == len(s1_var.call_targets)
            for s2_ct, s1_ct in zip(s2_var.call_targets, s1_var.call_targets):
                assert isinstance(s2_ct, Stage2CallTarget)
                assert s2_ct.stage1 is s1_ct

    # ----- per-variant aggregation invariant -----
    for s2_sec in stage2.sections:
        for s2_var in s2_sec.variants:
            assert s2_var.total_surviving_token_count == sum(
                ct.surviving_token_count for ct in s2_var.call_targets
            )
            assert s2_var.total_surviving_identity_count == sum(
                ct.surviving_identity_count for ct in s2_var.call_targets
            )
            assert s2_var.total_surviving_number_chunk_count == sum(
                ct.surviving_number_chunk_count
                for ct in s2_var.call_targets
            )

    # ----- per-CT cutoff invariant -----
    for s2_sec in stage2.sections:
        for s2_var in s2_sec.variants:
            for s2_ct in s2_var.call_targets:
                assert s2_ct.surviving_token_count <= s2_ct.predicted_full_length
                if not s2_ct.is_cut:
                    # Fully included OR fully dropped:
                    #   surviving = full (fully included) OR
                    #   surviving = 0   (dropped past cut)
                    assert s2_ct.surviving_token_count in (
                        s2_ct.predicted_full_length,
                        0,
                    )
                else:
                    # Cut entry: 0 <= partial < full.
                    assert 0 <= s2_ct.partial_cut_length
                    assert (
                        s2_ct.partial_cut_length < s2_ct.predicted_full_length
                    )

    # ----- per-row cumsum invariant -----
    # Walk the mapping; sum the per-row counts; final offset must
    # match.
    expected_identity = 0
    expected_number = 0
    mapping = stage1.batch_idx_to_section_variant
    for row in range(stage1.batch_size):
        s = int(mapping[row, 0])
        v = int(mapping[row, 1])
        if s == int(UINT32_MAX) or v == int(UINT32_MAX):
            continue
        var = stage2.sections[s].variants[v]
        expected_identity += var.total_surviving_identity_count
        expected_number += var.total_surviving_number_chunk_count
    assert int(stage2.identity_row_offsets[-1]) == expected_identity
    assert int(stage2.number_row_offsets[-1]) == expected_number

    # context_len=200 is generous; nothing should be cut at all in
    # this fixture.
    for s2_sec in stage2.sections:
        for s2_var in s2_sec.variants:
            assert s2_var.cut_call_target_index == len(s2_var.call_targets)
            for s2_ct in s2_var.call_targets:
                assert s2_ct.is_cut is False
                assert (
                    s2_ct.surviving_token_count
                    == s2_ct.predicted_full_length
                )


# ---------------------------------------------------------------------------
# Test 9: context_len = 0 -> everything dropped, but structure populated.
# ---------------------------------------------------------------------------


def test_context_len_zero_drops_all_but_keeps_structure() -> None:
    """A 0-budget context cuts the very first CT at partial=0 and
    drops the rest; the orchestrator must still populate the full
    Stage2Batch (Stage2CallTarget instances per CT, expanded_token_ids
    intact for downstream stages -- the surviving_token_count is 0
    but ``predicted_full_length`` is preserved)."""

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    raw = _u16(BLOCK_V2)

    stage1 = _make_batch(
        [[[raw, raw]]],  # one variant with 2 CTs
        num_variants_per_section=1,
    )
    stage2 = predict_lengths(stage1, context_len=0)

    variant = stage2.sections[0].variants[0]
    assert variant.cut_call_target_index == 0  # cut at first CT
    assert variant.total_surviving_token_count == 0
    assert variant.total_surviving_identity_count == 0

    # CT0 is the cut entry with partial=0.
    cts = variant.call_targets
    assert cts[0].is_cut is True
    assert cts[0].partial_cut_length == 0
    assert cts[0].surviving_token_count == 0
    # but predicted_full_length is preserved for downstream stages.
    assert cts[0].predicted_full_length == 2

    # CT1 dropped.
    assert cts[1].is_cut is False
    assert cts[1].surviving_token_count == 0
    assert cts[1].predicted_full_length == 2


# ---------------------------------------------------------------------------
# Test 10: empty batch (no sections) -> well-formed empty Stage2Batch.
# ---------------------------------------------------------------------------


def test_empty_batch_no_sections() -> None:
    """An empty :class:`Stage1Batch` (no sections, batch_size=0) yields
    a well-formed empty :class:`Stage2Batch`: empty sections list,
    row offsets shape (1,) with value 0."""

    stage1 = Stage1Batch(
        sections=[],
        batch_idx_to_section_variant=np.empty((0, 2), dtype=np.uint32),
        batch_size=0,
    )
    stage2 = predict_lengths(stage1, context_len=10)

    assert stage2.sections == []
    assert stage2.identity_row_offsets.shape == (1,)
    assert stage2.number_row_offsets.shape == (1,)
    assert int(stage2.identity_row_offsets[0]) == 0
    assert int(stage2.number_row_offsets[0]) == 0


# ---------------------------------------------------------------------------
# Test 11: parametric cumsum monotonicity sanity across varied shapes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "context_len",
    [0, 1, 2, 3, 5, 10, 100],
)
def test_row_offsets_monotonic_invariant_across_context_lens(
    context_len: int,
) -> None:
    """Row offsets must stay monotone non-decreasing regardless of how
    aggressive the cut is."""

    BLOCK_V2 = _V2_IDENTITY_BLOCK_START
    raw = _u16(BLOCK_V2, _VC2_VOCAB_ID, 0x42, BLOCK_V2)
    # 2 sections x 2 variants x 1 CT.
    stage1 = _make_batch(
        [
            [[raw], [raw]],
            [[raw], [raw]],
        ],
        num_variants_per_section=2,
    )
    stage2 = predict_lengths(stage1, context_len=context_len)

    for i in range(stage1.batch_size):
        assert (
            stage2.identity_row_offsets[i + 1]
            >= stage2.identity_row_offsets[i]
        )
        assert (
            stage2.number_row_offsets[i + 1]
            >= stage2.number_row_offsets[i]
        )
