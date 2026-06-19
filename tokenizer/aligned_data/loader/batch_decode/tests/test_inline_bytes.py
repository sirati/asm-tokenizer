"""Unit tests for :func:`build_inline_bytes` (ALG-1).

Single concern: pin the behavioural contract for the stage-3a inline-
byte concatenation: leading-zero pad at index 0, per-call-target slices
abutting in DFS encounter order, and the cut-aware "full payload of
every consumed carrier" buffer-layout rule (per plan D2 + ALG-8 +
test_number_decode.py:622-624 docstring).

For multi-chunk sources at the cut boundary, 3a keeps the FULL ``L``
bytes of the last consumed carrier even when only some of its chunks
survived the visible-stream prefix; 3c (``_number_decode.py``) drops
``idx_2d`` rows for the dropped chunks but its per-chunk offset formula
addresses the canonical full-payload byte layout.

Fixtures construct synthetic :class:`Stage2Batch` objects directly --
``build_inline_bytes`` reads only the surviving-state fields off the
hierarchy, so the rest of the dataclasses are filled with minimal
dummies. The expanded_token_ids + extra_*_mask + partial_cut_length
inputs are derived via :func:`expand_tokens` against synthetic raw
streams; this avoids divergence between the test fixture's notion of
"what the cut means" and the production stage-2 walker's notion.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    _FLOAT128_VOCAB_ID,
    _V2_IDENTITY_BLOCK_START,
    _V2_RESERVED_DIGIT_COUNT,
    _VC2_VOCAB_ID,
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._flat_call_targets import (
    dense_columns_from_stage2,
)
from tokenizer.aligned_data.loader.batch_decode._inline_bytes import (
    build_inline_bytes,
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


def _build_inline_bytes_slices(dense):
    """``build_inline_bytes`` -> ``(inline_bytes, list[slice])`` adapter.

    Stage 3a now returns ``inline_byte_starts`` (the per-call-target
    start-offset CSR). The slice contract these unit tests pin is the old
    abutting slice list: ``slice(starts[i], starts[i + 1])`` with the last
    stop at ``len(inline_bytes)``. Reconstructed here so the byte-layout
    assertions exercise the new CSR start array byte-for-byte.
    """
    inline_bytes, starts = build_inline_bytes(dense)
    stops = list(starts[1:]) + [inline_bytes.shape[0]]
    slices = [slice(int(s), int(e)) for s, e in zip(starts, stops)]
    return inline_bytes, slices


# Identity-band sample carrier id used for "non-multi-chunk identity" fixtures.
# Per the plan vocab table the IDENTITY block starts at id 264; BLOCK_V2 is
# the first slot. Identity carriers carry an inline-byte payload (0..2
# big-endian bytes encoding a caller-local id) but always emit 1 expanded
# slot (K = 1).
_BLOCK_V2_VOCAB_ID = _V2_IDENTITY_BLOCK_START  # 264

# Sample F16/F64 vocab ids -- positions in the NUMBER block after VC2.
# F16 is the 2nd NUMBER carrier; F64 is the 5th.
_F16_VOCAB_ID = _V2_RESERVED_DIGIT_COUNT + 1 + 1   # 258
_F64_VOCAB_ID = _V2_RESERVED_DIGIT_COUNT + 1 + 4   # 261


# ---------------------------------------------------------------------------
# Fixture builders (lifted from test_types.py + test_expand_tokens.py and
# shrunk to the minimum this test file needs).
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
    """Build an :class:`InlineDecodeState` directly from a raw stream.

    Mirrors the test_expand_tokens.py helper so the test fixture is the
    same byte-for-byte as expand_tokens consumes.
    """
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
    digit_cumsum = np.zeros(raw_tokens.shape[0] + 1, dtype=np.uint32)
    if raw_tokens.shape[0] > 0:
        np.cumsum(number_mask.view(np.uint8), out=digit_cumsum[1:])
    return InlineDecodeState(
        raw_tokens=raw_tokens,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
        digit_cumsum=digit_cumsum,
    )


def _make_section() -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _make_stage1_ct(
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


def _make_stage2_ct(
    raw_tokens: np.ndarray,
    *,
    encounter_category: Category = Category.LOCAL_FUNC,
    partial_cut_length: int | None = None,
) -> Stage2CallTarget:
    """Build a stage-2 call_target from a synthetic raw stream.

    Runs :func:`expand_tokens` on the synthetic stream to derive
    ``expanded_token_ids`` + the extra masks; the caller may override
    ``partial_cut_length`` to inject a cut. When ``partial_cut_length``
    is ``None`` the call_target is fully included.
    """
    stage1_ct = _make_stage1_ct(
        raw_tokens, encounter_category=encounter_category
    )
    expanded = expand_tokens(stage1_ct)
    full_length = expanded.predicted_full_length
    if partial_cut_length is None:
        # Fully included.
        surviving = full_length
        is_cut = False
        cut = full_length
    else:
        surviving = partial_cut_length
        is_cut = surviving < full_length
        cut = surviving

    # Surviving identity / number-chunk counts -- not directly used by
    # ALG-1, but the dataclass requires them. Compute via the simple
    # band masks (matches _surviving_counts.count_surviving's formula).
    surv = expanded.expanded_token_ids[:surviving]
    surv_identity = int(((surv >= 8) & (surv < 16)).sum())
    surv_number = int(((surv >= 1) & (surv < 8)).sum())

    return Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded.expanded_token_ids,
        extra_value_v2_mask=expanded.extra_value_v2_mask,
        extra_f128_mask=expanded.extra_f128_mask,
        predicted_full_length=full_length,
        surviving_token_count=surviving,
        surviving_identity_count=surv_identity,
        surviving_number_chunk_count=surv_number,
        is_cut=is_cut,
        partial_cut_length=cut,
    )


def _wrap_stage2_call_targets_as_batch(
    call_targets_per_variant: list[list[Stage2CallTarget]],
) -> Stage2Batch:
    """Wrap a list of per-variant Stage2CallTarget lists as a Stage2Batch.

    Each inner list is one variant's call_target list; we put all variants
    in one section for simplicity. The batch-level offsets are computed
    from the per-variant totals but are not exercised by
    :func:`build_inline_bytes`.
    """
    stage1_sections: list[Stage1Section] = []
    stage2_sections: list[Stage2Section] = []
    batch_size = len(call_targets_per_variant)

    stage1_variants: list[Stage1Variant] = []
    stage2_variants: list[Stage2Variant] = []
    for batch_idx, cts in enumerate(call_targets_per_variant):
        stage1_cts = [ct.stage1 for ct in cts]
        stage1_variant = Stage1Variant(
            variant_idx=batch_idx,
            variant_ref_offset=0,
            batch_idx=batch_idx,
            call_targets=stage1_cts,
            variant_tokens=np.zeros(0, dtype=np.uint16),
        )
        stage1_variants.append(stage1_variant)

        total_tokens = sum(ct.surviving_token_count for ct in cts)
        total_identities = sum(ct.surviving_identity_count for ct in cts)
        total_numbers = sum(ct.surviving_number_chunk_count for ct in cts)
        # Cut call_target index -- index of the first ``is_cut`` entry,
        # else len(cts) when none cut.
        cut_idx = len(cts)
        for i, ct in enumerate(cts):
            if ct.is_cut:
                cut_idx = i
                break
        stage2_variant = Stage2Variant(
            stage1=stage1_variant,
            call_targets=cts,
            cut_call_target_index=cut_idx,
            total_surviving_token_count=total_tokens,
            total_surviving_identity_count=total_identities,
            total_surviving_number_chunk_count=total_numbers,
        )
        stage2_variants.append(stage2_variant)

    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_make_section(),
        variants=stage1_variants,
    )
    stage2_section = Stage2Section(
        stage1=stage1_section,
        variants=stage2_variants,
    )
    stage1_sections.append(stage1_section)
    stage2_sections.append(stage2_section)

    if batch_size > 0:
        batch_idx_map = np.array(
            [[0, i] for i in range(batch_size)], dtype=np.uint32
        )
    else:
        batch_idx_map = np.zeros((0, 2), dtype=np.uint32)
    stage1_batch = Stage1Batch(
        sections=stage1_sections,
        batch_idx_to_section_variant=batch_idx_map,
        batch_size=batch_size,
    )

    # Per-batch sidecar offsets -- not consumed by build_inline_bytes.
    identity_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    number_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    for idx, variant in enumerate(stage2_variants):
        identity_row_offsets[idx + 1] = (
            identity_row_offsets[idx]
            + variant.total_surviving_identity_count
        )
        number_row_offsets[idx + 1] = (
            number_row_offsets[idx]
            + variant.total_surviving_number_chunk_count
        )

    return Stage2Batch(
        stage1=stage1_batch,
        sections=stage2_sections,
        identity_row_offsets=identity_row_offsets,
        number_row_offsets=number_row_offsets,
    )


def _make_batch(
    call_targets_per_variant: list[list[Stage2CallTarget]],
) -> Stage2Batch:
    return _wrap_stage2_call_targets_as_batch(call_targets_per_variant)


# ---------------------------------------------------------------------------
# Test 1 -- Empty batch (zero call_targets).
# ---------------------------------------------------------------------------


def test_empty_batch_yields_only_pad():
    """No call_targets -> just the leading-zero pad, no slices."""
    batch = _make_batch([])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1,)
    assert inline_bytes.dtype == np.uint8
    assert inline_bytes[0] == 0
    assert slices == []


# ---------------------------------------------------------------------------
# Test 2 -- Single call_target, no cuts.
# ---------------------------------------------------------------------------


def test_single_call_target_no_cut_concats_in_raw_order():
    """One call_target with several inline carriers -> bytes in raw order.

    Stream: F16 (2 bytes) + identity carrier (1 byte). Total payload =
    3 bytes. Slice = ``slice(1, 4)``. The leading-zero pad lives at
    ``inline_bytes[0]``.
    """
    # F16 carrier at p=0 with 2 inline-byte payload + identity carrier
    # with 1 inline-byte payload.
    raw = _u16(
        _F16_VOCAB_ID,   # F16 carrier
        0x12,            # F16 byte 0 (MSB)
        0x34,            # F16 byte 1 (LSB)
        _BLOCK_V2_VOCAB_ID,  # identity carrier
        0xAB,            # identity byte 0
    )
    ct = _make_stage2_ct(raw)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))

    assert inline_bytes.shape == (1 + 3,)
    assert inline_bytes[0] == 0  # leading pad
    assert len(slices) == 1
    assert slices[0] == slice(1, 4)
    np.testing.assert_array_equal(inline_bytes[slices[0]], [0x12, 0x34, 0xAB])


# ---------------------------------------------------------------------------
# Test 3 -- Two call_targets, no cuts: slices abut.
# ---------------------------------------------------------------------------


def test_two_call_targets_no_cut_abutting_slices():
    """Second call_target's slice starts at first's stop."""
    raw_a = _u16(_F16_VOCAB_ID, 0x01, 0x02)              # 2 bytes
    raw_b = _u16(_F64_VOCAB_ID, 0x11, 0x22, 0x33, 0x44,
                 0x55, 0x66, 0x77, 0x88)                  # 8 bytes
    ct_a = _make_stage2_ct(raw_a)
    ct_b = _make_stage2_ct(raw_b)
    batch = _make_batch([[ct_a, ct_b]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 2 + 8,)
    assert slices == [slice(1, 3), slice(3, 11)]
    np.testing.assert_array_equal(inline_bytes[slices[0]], [0x01, 0x02])
    np.testing.assert_array_equal(
        inline_bytes[slices[1]],
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    )


# ---------------------------------------------------------------------------
# Test 4 -- Cut call_target drops trailing call_targets to size-zero slices.
# ---------------------------------------------------------------------------


def test_cut_drops_trailing_call_targets_with_zero_slices():
    """A cut in call_target 0 -> call_target 1 gets a zero-length slice.

    Synthetic: ct_a has F16 (2 bytes payload, 1 expanded slot beyond
    the prepend). ct_b has a similar carrier. We cut ct_a at
    ``partial_cut_length=1`` -- only the prepend survives in ct_a, no
    inline bytes. ct_b is fully dropped (zero-length slice).
    """
    raw_a = _u16(_F16_VOCAB_ID, 0x01, 0x02)
    raw_b = _u16(_F16_VOCAB_ID, 0x03, 0x04)
    # Cut ct_a so only the prepend slot survives.
    ct_a = _make_stage2_ct(raw_a, partial_cut_length=1)
    # ct_b is entirely dropped -- mark its full length as 0 surviving.
    ct_b = _make_stage2_ct(raw_b, partial_cut_length=0)
    batch = _make_batch([[ct_a, ct_b]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    # Only the leading-zero pad survives; both call_targets contribute
    # zero bytes.
    assert inline_bytes.shape == (1,)
    assert slices == [slice(1, 1), slice(1, 1)]


# ---------------------------------------------------------------------------
# Test 5 -- Multi-chunk VC2 with mid-cut: FULL L bytes retained.
#
# New contract (post-HD-2 fix): 3a keeps the FULL ``L``-byte payload of
# the last consumed carrier even when some of its chunks were dropped
# past the cut. The byte layout is independent of which chunks survived;
# 3c is the layer that skips emitting ``idx_2d`` rows for dropped chunks
# (via ``K_visible``). This keeps ALG-8's per-chunk offset formula
# ``[p_carrier_byte + L - 8*(c+1), p_carrier_byte + L - 8*c)`` valid
# without an L vs 8*j_last off-by-N correction in 3c.
# ---------------------------------------------------------------------------


def test_vc2_mid_cut_keeps_full_payload():
    """VC2 L=17 (K=3 chunks) cut at j=2 visible chunks -> FULL 17 bytes.

    expanded_token_ids structure: [prepend, VC2_carrier, painted_chunk_1,
    painted_chunk_2]. Cut at partial_cut_length=3 -> visible slots =
    [prepend, carrier, painted_1] (j=2 chunks visible). 3a's buffer
    retains all 17 inline bytes; 3c emits only 2 idx_2d rows.
    """
    # Build a synthetic VC2 source with L=17.
    payload = list(range(1, 18))  # 17 bytes 0x01..0x11; raw values stay < 256
    raw = _u16(_VC2_VOCAB_ID, *payload)
    # Full expansion would be [prepend, VC2_carrier, painted_1, painted_2]
    # (3 chunks emitted -> K=3 -> 3 expanded slots beyond the prepend).
    # Cut at 3 -> 2 chunks visible, but byte buffer still holds all L=17.
    ct = _make_stage2_ct(raw, partial_cut_length=3)
    batch = _make_batch([[ct]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 17,)
    assert slices == [slice(1, 18)]
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


def test_vc2_mid_cut_single_visible_chunk_still_keeps_full_payload():
    """VC2 L=17 cut at j=1 visible chunk -> FULL 17 bytes in buffer.

    Even though only 1 chunk survives the visible-stream prefix, the
    byte layout retains all L=17 bytes so 3c's ALG-8 formula computes
    correct LSB-chunk offsets without a payload-truncation correction.
    """
    payload = list(range(1, 18))
    raw = _u16(_VC2_VOCAB_ID, *payload)
    # Cut at 2 (prepend + carrier slot only) -> j=1 chunk visible.
    ct = _make_stage2_ct(raw, partial_cut_length=2)
    batch = _make_batch([[ct]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 17,)
    assert slices == [slice(1, 18)]
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


# ---------------------------------------------------------------------------
# Test 6 -- F128 finite source fully included contributes 16 bytes.
# ---------------------------------------------------------------------------


def test_f128_finite_full_contributes_16_bytes():
    """F128 finite: K=2 chunks, full inclusion -> 16-byte payload."""
    # Build a finite F128 payload: high u16 != 0x7FFF after masking.
    # Byte 0 = 0x40 -> sign=0, exponent high bits = 0x40 & 0x7F = 0x40.
    # high_u16 = 0x4000_____, masked with 0x7FFF = 0x4000_____, != 0x7FFF.
    payload = [0x40, 0x00] + list(range(2, 16))  # 16 bytes
    raw = _u16(_FLOAT128_VOCAB_ID, *payload)
    ct = _make_stage2_ct(raw)  # fully included
    batch = _make_batch([[ct]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 16,)
    assert slices == [slice(1, 17)]
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


def test_f128_finite_mid_cut_keeps_full_payload():
    """F128 finite cut at j=1 -> buffer still holds all 16 payload bytes.

    Per the post-HD-2 contract the byte layout is independent of which
    F128 chunk survived; 3c reads ALG-7's per-chunk offsets against the
    full 16-byte payload and skips the dropped chunk's idx_2d row.
    """
    payload = [0x40, 0x00] + list(range(2, 16))
    raw = _u16(_FLOAT128_VOCAB_ID, *payload)
    # Full expansion = [prepend, F128_carrier, painted_chunk]. Cut at 2
    # = prepend + carrier slot only -> j=1, but buffer holds all 16.
    ct = _make_stage2_ct(raw, partial_cut_length=2)
    batch = _make_batch([[ct]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 16,)
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


# ---------------------------------------------------------------------------
# Test 7 -- F128 NaN/Inf source: byte payload is STILL 16 bytes
# (chunk-count differs from finite case, but byte payload doesn't).
# ---------------------------------------------------------------------------


def test_f128_nan_inf_contributes_16_bytes():
    """F128 NaN: K=1 chunk; full payload of 16 bytes still surviving."""
    # NaN: high u16 & 0x7FFF == 0x7FFF AND non-zero mantissa.
    # Byte 0 = 0x7F (sign+exp high bits = 0x7F), byte 1 = 0xFF -> high
    # u16 = 0x7FFF; & 0x7FFF == 0x7FFF -> NaN/Inf condition. With
    # non-zero mantissa bytes (e.g. byte 2 = 0x01) it's a NaN.
    payload = [0x7F, 0xFF, 0x01] + [0x00] * 13  # 16 bytes
    raw = _u16(_FLOAT128_VOCAB_ID, *payload)
    ct = _make_stage2_ct(raw)  # fully included
    batch = _make_batch([[ct]])

    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 16,)
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


def test_f128_inf_contributes_16_bytes():
    """F128 +Inf: high u16 & 0x7FFF == 0x7FFF with zero mantissa."""
    payload = [0x7F, 0xFF] + [0x00] * 14
    raw = _u16(_FLOAT128_VOCAB_ID, *payload)
    ct = _make_stage2_ct(raw)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1 + 16,)
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


# ---------------------------------------------------------------------------
# Test 8 -- Narrowing assignment correctness (u16 -> u8 truncation
# preserves the low byte; inline-band ids have high byte == 0 so it's
# lossless).
# ---------------------------------------------------------------------------


def test_narrowing_assignment_preserves_inline_band_values():
    """All inline-band ids fit in u8 -> output equals raw low bytes.

    Build a stream whose inline-digit bytes span the full u8 range
    [0, 256). The output array's values must match the source byte-
    for-byte.
    """
    # F64 with 8 inline-digit bytes covering a wide range of values.
    payload = [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x42, 0xAA]
    raw = _u16(_F64_VOCAB_ID, *payload)
    ct = _make_stage2_ct(raw)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)
    # The dtype is u8 -- the narrowing trick is the documented
    # implementation.
    assert inline_bytes.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 9 -- Slice union covers ``[1, len(inline_bytes))`` exactly (no gaps,
# no overlap).
# ---------------------------------------------------------------------------


def test_slice_union_covers_entire_inline_bytes_post_pad():
    """Concatenating the call_target slices must reproduce inline_bytes[1:]."""
    raw_a = _u16(_F16_VOCAB_ID, 0x01, 0x02)            # 2 bytes
    raw_b = _u16(_F64_VOCAB_ID, *range(0xA0, 0xA8))    # 8 bytes
    raw_c = _u16(_BLOCK_V2_VOCAB_ID, 0xFF)             # 1 byte
    ct_a = _make_stage2_ct(raw_a)
    ct_b = _make_stage2_ct(raw_b)
    ct_c = _make_stage2_ct(raw_c)
    # Multiple variants to also exercise the DFS walk order.
    batch = _make_batch([[ct_a], [ct_b, ct_c]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))

    # Slices abut without gaps.
    assert slices[0].start == 1
    for prev, cur in zip(slices, slices[1:]):
        assert cur.start == prev.stop
    # Last slice's stop == len(inline_bytes).
    assert slices[-1].stop == inline_bytes.shape[0]
    # Concatenation == inline_bytes[1:].
    rebuilt = np.concatenate([inline_bytes[s] for s in slices])
    np.testing.assert_array_equal(rebuilt, inline_bytes[1:])


# ---------------------------------------------------------------------------
# Test 10 -- dtype + leading-pad invariants.
# ---------------------------------------------------------------------------


def test_dtype_is_uint8_and_pad_index_zero():
    raw = _u16(_F16_VOCAB_ID, 0x01, 0x02)
    ct = _make_stage2_ct(raw)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.dtype == np.uint8
    assert inline_bytes[0] == 0
    # Slice never starts at 0 -- the pad lives outside every per-call
    # target range.
    for sl in slices:
        assert sl.start >= 1


# ---------------------------------------------------------------------------
# Additional edge cases (defensive guards we want pinned).
# ---------------------------------------------------------------------------


def test_empty_call_target_contributes_zero_bytes():
    """A call_target with no carriers -> empty slice."""
    raw = _u16()  # empty raw stream
    ct = _make_stage2_ct(raw)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1,)
    assert slices == [slice(1, 1)]


def test_vc2_single_chunk_cut_to_zero_yields_no_bytes():
    """VC2 L<=8 (single chunk) cut at j=0 -> no bytes.

    The carrier itself produces 1 expanded slot; cutting at
    ``partial_cut_length=1`` (only the prepend visible) drops the
    carrier entirely.
    """
    raw = _u16(_VC2_VOCAB_ID, 0x11, 0x22, 0x33)  # L=3, K=1
    ct = _make_stage2_ct(raw, partial_cut_length=1)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    assert inline_bytes.shape == (1,)
    assert slices == [slice(1, 1)]


def test_vc2_single_chunk_full_keeps_all_payload():
    """VC2 L=5 (K=1) fully included -> all 5 payload bytes."""
    payload = [0x10, 0x20, 0x30, 0x40, 0x50]
    raw = _u16(_VC2_VOCAB_ID, *payload)
    ct = _make_stage2_ct(raw)  # fully included
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


def test_vc2_three_chunks_fully_included_keeps_all_bytes():
    """VC2 L=17 fully included -> all 17 bytes in raw stream order."""
    payload = list(range(1, 18))
    raw = _u16(_VC2_VOCAB_ID, *payload)
    ct = _make_stage2_ct(raw)
    batch = _make_batch([[ct]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    np.testing.assert_array_equal(inline_bytes[slices[0]], payload)


def test_cut_inside_second_call_target_keeps_first_fully():
    """A 2-call_target row where the cut lands inside ct_b -> ct_a is
    fully included, ct_b's last consumed carrier contributes its FULL
    ``L``-byte payload (per the post-HD-2 contract).
    """
    raw_a = _u16(_F16_VOCAB_ID, 0x01, 0x02)                # K=1, L=2
    raw_b = _u16(_VC2_VOCAB_ID, *range(1, 18))             # K=3, L=17
    ct_a = _make_stage2_ct(raw_a)
    # ct_b: cut at j=1 visible chunk -> buffer still holds the full
    # L=17-byte payload; 3c emits 1 idx_2d row (the LSB chunk).
    ct_b = _make_stage2_ct(raw_b, partial_cut_length=2)
    batch = _make_batch([[ct_a, ct_b]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    expected_a = [0x01, 0x02]
    expected_b = list(range(1, 18))  # full 17-byte payload of the VC2
    assert inline_bytes.shape == (1 + 2 + 17,)
    np.testing.assert_array_equal(inline_bytes[slices[0]], expected_a)
    np.testing.assert_array_equal(inline_bytes[slices[1]], expected_b)


def test_multi_variant_dfs_order():
    """DFS walk: sections -> variants -> call_targets; slice order matches.

    Build a batch with one section, two variants (each with their own
    call_target). The slice list must enumerate variant_0 first then
    variant_1.
    """
    raw_v0 = _u16(_F16_VOCAB_ID, 0xA1, 0xA2)
    raw_v1 = _u16(_F16_VOCAB_ID, 0xB1, 0xB2)
    ct_v0 = _make_stage2_ct(raw_v0)
    ct_v1 = _make_stage2_ct(raw_v1)
    batch = _make_batch([[ct_v0], [ct_v1]])
    inline_bytes, slices = _build_inline_bytes_slices(dense_columns_from_stage2(batch))
    np.testing.assert_array_equal(inline_bytes[slices[0]], [0xA1, 0xA2])
    np.testing.assert_array_equal(inline_bytes[slices[1]], [0xB1, 0xB2])
