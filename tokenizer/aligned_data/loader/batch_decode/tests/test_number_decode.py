"""Unit tests for stage 3c -- per-:class:`TokenType` number ``idx_2d``.

Single concern: pin the per-:class:`TokenType` byte-layout contract
emitted by
:func:`tokenizer.aligned_data.loader.batch_decode._number_decode.build_number_idx_2d`
per plan ALG-7 + ALG-8.

Tests build synthetic :class:`Stage2Batch` fixtures around real
:class:`InlineDecodeState` -- the function under test reads
``state.raw_tokens``, ``state.real_mask``, ``state.number_mask``,
``state.runlen_number`` plus the level-4 ``expanded_token_ids`` +
``extra_*_mask`` + ``surviving_token_count`` -- enough that we can
exercise every per-:class:`TokenType` branch in isolation. We
deliberately bypass the full stage-2 pipeline (``expand_tokens`` etc.)
and instead build the expanded stream + masks by hand so each test
pins exactly one byte-layout contract.

The "inline_bytes" buffer passed in is a stand-in for 3a's output. We
fabricate it with the leading-zero pad at index 0 and the call_target's
inline-digit bytes laid out contiguously starting at index 1 (the
contiguous prefix matches the natural cumsum layout 3a will emit).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._number_decode import (
    _NUMBER_BLOCK_TOKEN_TYPES,
    build_number_idx_2d,
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
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Vocab constants (kept local so a layout shift surfaces in this file too).
# ---------------------------------------------------------------------------

_VC2_RAW = 257
_F16_RAW = 258
_BF16_RAW = 259
_F32_RAW = 260
_F64_RAW = 261
_F80_RAW = 262
_F128_RAW = 263

_VC2_SHIFTED = 1
_F16_SHIFTED = 2
_BF16_SHIFTED = 3
_F32_SHIFTED = 4
_F64_SHIFTED = 5
_F80_SHIFTED = 6
_F128_SHIFTED = 7

_LOCAL_FUNC_SHIFTED = 9


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _empty_function_data() -> FunctionData:
    return FunctionData(
        func_name="dummy",
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.zeros(0, dtype=np.uint16),
        insn_runlength=np.zeros(0, dtype=np.uint32),
        block_runlength=np.zeros(0, dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _empty_section() -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=[],
    )


def _build_state(raw_tokens: np.ndarray) -> InlineDecodeState:
    """Build an :class:`InlineDecodeState` from ``raw_tokens``.

    Mirrors the production builder's logic on the subset of fields
    ``build_number_idx_2d`` reads. All other fields are filled
    consistently so dataclass invariants hold.
    """
    real_mask = raw_tokens > 256
    number_mask = raw_tokens < 256
    if raw_tokens.shape[0] == 0:
        runlen_number = np.zeros(0, dtype=np.uint32)
        runlen_value = np.zeros(0, dtype=np.uint32)
    else:
        runlen_number = run_lengths(number_mask)
        runlen_value = run_lengths(~real_mask)
    carries_inline_mask = real_mask & (raw_tokens < 272)
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


def _make_call_target(
    raw_tokens: np.ndarray,
    expanded_token_ids: np.ndarray,
    extra_value_v2_mask: np.ndarray,
    extra_f128_mask: np.ndarray,
    *,
    surviving_token_count: int | None = None,
    encounter_category: Category = Category.LOCAL_FUNC,
) -> Stage2CallTarget:
    """Build a Stage2CallTarget around a hand-crafted expanded stream.

    The two-arg shape ``(raw_tokens, expanded_token_ids)`` lets each
    test pin BOTH the raw-stream byte payload + the expanded-stream
    chunk sequence independently -- mid-cut tests need to keep the raw
    stream intact while shrinking ``surviving_token_count``.
    """
    stage1_ct = Stage1CallTarget(
        function_data=_empty_function_data(),
        state=_build_state(raw_tokens),
        call_targets_section=[],
        encounter_category=encounter_category,
        parent_call_target_index=None,
        function_name_ptr=0,
    )
    predicted_full_length = int(expanded_token_ids.shape[0])
    if surviving_token_count is None:
        surviving_token_count = predicted_full_length

    identity_band_mask = (expanded_token_ids[:surviving_token_count] >= 8) & (
        expanded_token_ids[:surviving_token_count] < 16
    )
    number_band_mask = (expanded_token_ids[:surviving_token_count] >= 1) & (
        expanded_token_ids[:surviving_token_count] < 8
    )
    return Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded_token_ids,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
        predicted_full_length=predicted_full_length,
        surviving_token_count=surviving_token_count,
        surviving_identity_count=int(identity_band_mask.sum()),
        surviving_number_chunk_count=int(number_band_mask.sum()),
        is_cut=surviving_token_count < predicted_full_length,
        partial_cut_length=surviving_token_count,
    )


def _wrap_single_call_target(stage2_ct: Stage2CallTarget) -> Stage2Batch:
    """Wrap one Stage2CallTarget into a minimal Stage2Batch.

    The function under test only iterates the 4-level hierarchy +
    reads the level-4 fields, so the level-2/3 wrapping can stay
    trivial.
    """
    stage1_ct = stage2_ct.stage1
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=[stage1_ct],
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_empty_section(),
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        batch_size=1,
    )

    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=[stage2_ct],
        cut_call_target_index=0 if stage2_ct.is_cut else 1,
        total_surviving_token_count=stage2_ct.surviving_token_count,
        total_surviving_identity_count=stage2_ct.surviving_identity_count,
        total_surviving_number_chunk_count=stage2_ct.surviving_number_chunk_count,
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


def _build_inline_bytes_from_raw(
    raw_tokens: np.ndarray,
) -> tuple[np.ndarray, slice]:
    """Build a synthetic 3a-style inline_bytes buffer + a single slice.

    The leading zero pad sits at index 0 (per ALG-1's design); inline-
    digit bytes from raw_tokens (positions where ``raw_tokens < 256``)
    are concatenated immediately after, in stream order. The single
    returned slice covers the post-pad bytes; tests with multiple
    call_targets can build a multi-slice variant directly.
    """
    payload = raw_tokens[raw_tokens < 256].astype(np.uint8)
    inline_bytes = np.empty(1 + payload.shape[0], dtype=np.uint8)
    inline_bytes[0] = 0
    inline_bytes[1:] = payload
    return inline_bytes, slice(1, 1 + payload.shape[0])


# ---------------------------------------------------------------------------
# Per-TokenType: empty + fixed-width.
# ---------------------------------------------------------------------------


def test_empty_batch_emits_all_empty_arrays() -> None:
    """Test 1: an empty batch yields one empty array per NUMBER TokenType
    plus an empty sidecar + empty f128_is_nan_or_inf flag array."""
    stage2_ct = _make_call_target(
        raw_tokens=np.zeros(0, dtype=np.uint16),
        expanded_token_ids=np.array([_LOCAL_FUNC_SHIFTED], dtype=np.uint16),
        extra_value_v2_mask=np.array([False], dtype=bool),
        extra_f128_mask=np.array([False], dtype=bool),
    )
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes = np.array([0], dtype=np.uint8)  # only the pad

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [slice(1, 1)])

    # All TokenType keys are present, all empty.
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        assert idx_2d_per_type[T].shape[0] == 0
        assert chunk_slices_per_type[T] == [slice(0, 0)]
    assert f128_is_nan_or_inf.shape == (0,)
    assert vc2_sidecar.shape == (0,)


def test_f32_single_source_one_row_four_bytes() -> None:
    """Test 2: a single F32 source -> 1 row of 4 contiguous byte offsets.

    raw_tokens layout (single call_target):
        [F32_carrier, b0, b1, b2, b3]
    inline_bytes layout (synthetic): [pad, b0, b1, b2, b3]
                                       0   1   2   3   4
    expected idx_2d[F32] = [[1, 2, 3, 4]]
    """
    raw_tokens = np.array(
        [_F32_RAW, 0xAA, 0xBB, 0xCC, 0xDD], dtype=np.uint16
    )
    # expanded = [prepend, F32_carrier]; the 4 inline-digit bytes get
    # stripped (they live in inline_bytes already).
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _F32_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False], dtype=bool)
    extra_f128 = np.array([False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT32],
        np.array([[1, 2, 3, 4]], dtype=np.uint32),
    )
    assert chunk_slices_per_type[TokenType.FLOAT32] == [slice(0, 1)]
    # All other TokenType arrays empty.
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        if T is TokenType.FLOAT32:
            continue
        assert idx_2d_per_type[T].shape[0] == 0


@pytest.mark.parametrize(
    "carrier_raw,shifted_id,token_type,width",
    [
        (_F16_RAW, _F16_SHIFTED, TokenType.FLOAT16, 2),
        (_BF16_RAW, _BF16_SHIFTED, TokenType.BFLOAT16, 2),
        (_F64_RAW, _F64_SHIFTED, TokenType.FLOAT64, 8),
        (_F80_RAW, _F80_SHIFTED, TokenType.FLOAT80, 10),
    ],
)
def test_fixed_width_fp_single_source(
    carrier_raw: int,
    shifted_id: int,
    token_type: TokenType,
    width: int,
) -> None:
    """Each fixed-width FP type emits 1 row of ``width`` contiguous bytes.

    Covers F16 / BF16 / F64 / F80 (F32 has its own dedicated test).
    """
    raw_payload = np.arange(0xA0, 0xA0 + width, dtype=np.uint16)
    raw_tokens = np.concatenate(
        [np.array([carrier_raw], dtype=np.uint16), raw_payload]
    )
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, shifted_id], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False], dtype=bool)
    extra_f128 = np.array([False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    np.testing.assert_array_equal(
        idx_2d_per_type[token_type],
        np.arange(1, 1 + width, dtype=np.uint32)[np.newaxis, :],
    )
    assert chunk_slices_per_type[token_type] == [slice(0, 1)]
    assert f128_is_nan_or_inf.shape == (0,)
    assert vc2_sidecar.shape == (0,)


# ---------------------------------------------------------------------------
# F128.
# ---------------------------------------------------------------------------


def test_f128_finite_single_source_two_chunks() -> None:
    """Test 3: F128 finite -> 2 rows (LSB limb, then MSB limb).

    Finite signal: ``extra_f128_mask`` is True at the carrier+1 expanded
    slot (set by stage 2's ALG-2 detection).

    raw_tokens layout:
        [F128_carrier, b0, b1, ..., b15]  (b0 = high byte, sign+exponent;
                                            b15 = low byte, LSB)
    inline_bytes: [pad, b0, b1, ..., b15]  with bytes at offsets 1..16.
    Expected:
        idx_2d[F128][0] = [9, 10, 11, 12, 13, 14, 15, 16]   # LSB limb
        idx_2d[F128][1] = [1,  2,  3,  4,  5,  6,  7,  8]   # MSB limb
        f128_is_nan_or_inf == [False]
    """
    # High u16 byte0=0x00 (sign 0, exp high 7 bits = 0) + byte1=0x00
    # -> exponent != 0x7FFF -> finite per ALG-2.
    payload = np.arange(0x10, 0x20, dtype=np.uint16)
    raw_tokens = np.concatenate(
        [np.array([_F128_RAW], dtype=np.uint16), payload]
    )
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _F128_SHIFTED, _F128_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False, False], dtype=bool)
    extra_f128 = np.array([False, False, True], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT128],
        np.array(
            [
                [9, 10, 11, 12, 13, 14, 15, 16],   # LSB chunk
                [1, 2, 3, 4, 5, 6, 7, 8],          # MSB chunk
            ],
            dtype=np.uint32,
        ),
    )
    assert chunk_slices_per_type[TokenType.FLOAT128] == [slice(0, 2)]
    np.testing.assert_array_equal(
        f128_is_nan_or_inf, np.array([False], dtype=np.bool_)
    )
    assert vc2_sidecar.shape == (0,)


def test_f128_finite_mid_cut_still_emits_both_chunks() -> None:
    """F128 finite source whose painted MSB slot is past
    ``partial_cut_length``: 3c MUST emit BOTH chunks anyway.

    3d's per-source layout reads ``actual_exp`` from the MSB limb to
    derive the LSB chunk's exponent base; suppressing the MSB chunk
    here would either break 3d's chunk-count contract (chunks_per_
    source = 2 for finite) OR force 3d to read LSB bytes as if they
    were MSB bytes -- both broken. Stage 4's per-row sidecar walk is
    the layer that drops the trailing invisible MSB chunk via the
    stream-visible count; this layer keeps the full ALG-2 chunk set
    in 3c's output.
    """
    payload = np.arange(0x10, 0x20, dtype=np.uint16)
    raw_tokens = np.concatenate(
        [np.array([_F128_RAW], dtype=np.uint16), payload]
    )
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _F128_SHIFTED, _F128_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False, False], dtype=bool)
    extra_f128 = np.array([False, False, True], dtype=bool)
    # Cut between the carrier (expanded[1]) and the painted MSB
    # (expanded[2]) -- only the carrier survives in the visible stream.
    stage2_ct = _make_call_target(
        raw_tokens,
        expanded,
        extra_vc2,
        extra_f128,
        surviving_token_count=2,
    )
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    # Both chunks present. LSB row first (bytes 9..16), then MSB row
    # (bytes 1..8) -- matches the LSB-first stream emission order.
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT128],
        np.array(
            [
                [9, 10, 11, 12, 13, 14, 15, 16],   # LSB chunk
                [1, 2, 3, 4, 5, 6, 7, 8],          # MSB chunk
            ],
            dtype=np.uint32,
        ),
    )
    assert chunk_slices_per_type[TokenType.FLOAT128] == [slice(0, 2)]
    # Source is finite; the painted slot is the ALG-2 finite signal
    # (read against the FULL mask, not the surviving prefix).
    np.testing.assert_array_equal(
        f128_is_nan_or_inf, np.array([False], dtype=np.bool_)
    )


def test_f128_nan_or_inf_single_source_one_chunk() -> None:
    """Test 4: F128 NaN/Inf -> 1 row (MSB limb only).

    NaN/Inf signal: ``extra_f128_mask`` is False at the carrier+1
    expanded slot (stage 2's ALG-2 detected ``high_u16 & 0x7FFF ==
    0x7FFF`` and skipped painting).
    """
    payload = np.arange(0x30, 0x40, dtype=np.uint16)
    raw_tokens = np.concatenate(
        [np.array([_F128_RAW], dtype=np.uint16), payload]
    )
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _F128_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False], dtype=bool)
    extra_f128 = np.array([False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT128],
        np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint32),
    )
    assert chunk_slices_per_type[TokenType.FLOAT128] == [slice(0, 1)]
    np.testing.assert_array_equal(
        f128_is_nan_or_inf, np.array([True], dtype=np.bool_)
    )
    assert vc2_sidecar.shape == (0,)


# ---------------------------------------------------------------------------
# VC2 byte layout.
# ---------------------------------------------------------------------------


def _vc2_raw_tokens(L: int) -> np.ndarray:
    """Build a single-VC2-source raw_tokens with ``L`` inline-digit bytes.

    Uses byte values ``[0x10, 0x10 + L)`` so the test arithmetic on the
    inline_bytes offsets is easy to track by eye.
    """
    payload = np.arange(0x10, 0x10 + L, dtype=np.uint16)
    return np.concatenate([np.array([_VC2_RAW], dtype=np.uint16), payload])


def test_vc2_L17_three_chunks_with_msb_pad() -> None:
    """Test 5: VC2 L=17 (K=3) -> 3 rows; MSB row = 7 zeros + 1 actual byte.

    Sidecar reports chunk indices [0, 1, 2] in stream-emission order.

    Per ALG-8:
        chunk 0 (LSB): inline_bytes[p_carrier_byte+9 .. p_carrier_byte+17)
        chunk 1:        inline_bytes[p_carrier_byte+1 .. p_carrier_byte+9)
        chunk 2 (MSB): inline_bytes[p_carrier_byte .. p_carrier_byte+1)
    """
    L = 17
    raw_tokens = _vc2_raw_tokens(L)
    # K = ceil(17/8) = 3; the expanded stream has 1 carrier + 2 painted
    # continuation slots, so K-1 = 2 paints. ``extra_value_v2_mask`` is
    # True at carrier+1 and carrier+2 (positions 2 + 3 in expanded).
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _VC2_SHIFTED, _VC2_SHIFTED, _VC2_SHIFTED],
        dtype=np.uint16,
    )
    extra_vc2 = np.array([False, False, True, True], dtype=bool)
    extra_f128 = np.array([False, False, False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)
    # ct_slice.start == 1; bytes occupy [1, 18).
    p = ct_slice.start  # == 1

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    expected_rows = np.array(
        [
            # chunk 0 (LSB): bytes p+9 .. p+17
            [p + 9, p + 10, p + 11, p + 12, p + 13, p + 14, p + 15, p + 16],
            # chunk 1: bytes p+1 .. p+9
            [p + 1, p + 2, p + 3, p + 4, p + 5, p + 6, p + 7, p + 8],
            # chunk 2 (MSB): 7 pad zeros + 1 byte at p
            [0, 0, 0, 0, 0, 0, 0, p],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.VALUED_CONST_V2], expected_rows
    )
    assert chunk_slices_per_type[TokenType.VALUED_CONST_V2] == [slice(0, 3)]
    np.testing.assert_array_equal(
        vc2_sidecar, np.array([0, 1, 2], dtype=np.uint32)
    )


def test_vc2_L8_one_chunk_no_pad() -> None:
    """Test 6: VC2 L=8 (K=1) -> 1 row of 8 contiguous bytes (no pad).

    Carrier at raw position 0; payload bytes at raw positions 1..8;
    inline_bytes layout: [pad, b0, b1, ..., b7] -> offsets 1..8.
    """
    L = 8
    raw_tokens = _vc2_raw_tokens(L)
    # K = 1 -> no paints; expanded has only the carrier.
    expanded = np.array([_LOCAL_FUNC_SHIFTED, _VC2_SHIFTED], dtype=np.uint16)
    extra_vc2 = np.array([False, False], dtype=bool)
    extra_f128 = np.array([False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)
    p = ct_slice.start  # == 1

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    expected_rows = np.array(
        [[p, p + 1, p + 2, p + 3, p + 4, p + 5, p + 6, p + 7]],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.VALUED_CONST_V2], expected_rows
    )
    np.testing.assert_array_equal(
        vc2_sidecar, np.array([0], dtype=np.uint32)
    )


def test_vc2_L0_one_chunk_all_pad() -> None:
    """Test 7: VC2 L=0 (K=1) -> 1 row of 8 zeros (all pad).

    The empty-payload edge case: ``ceil(0/8) = 0``, max-guarded to 1.
    The single chunk has no actual bytes; every column references
    ``inline_bytes[0]`` (the leading-zero pad). Chunk index = 0.
    """
    raw_tokens = np.array(
        [_VC2_RAW, _F32_RAW, 0xAA, 0xBB, 0xCC, 0xDD], dtype=np.uint16
    )
    # Trailing F32 source ensures runlen_number[p_carrier + 1] reads
    # the right value (0 -- the next slot is a real token, not an
    # inline-digit). Without the trailing real token, the VC2 carrier
    # at the last raw position would trip ``_promote_vc2``'s tail
    # assertion. Stage 2's expand_tokens already enforces this.
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _VC2_SHIFTED, _F32_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False, False], dtype=bool)
    extra_f128 = np.array([False, False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    # VC2 chunk: all 8 pad zeros.
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.VALUED_CONST_V2],
        np.zeros((1, 8), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        vc2_sidecar, np.array([0], dtype=np.uint32)
    )
    # F32 came AFTER VC2 in stream; its 4 bytes occupy offsets 1..4 in
    # inline_bytes (VC2 had zero bytes of payload).
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT32],
        np.array([[1, 2, 3, 4]], dtype=np.uint32),
    )


def test_vc2_mid_cut_drops_msb_chunk() -> None:
    """Test 8: VC2 K=3 source with the cut leaving 2 visible -> 2 rows.

    The cut is in the expanded stream at ``partial_cut_length = 3``:
    the carrier (expanded[1]) and the first continuation (expanded[2])
    survive; the MSB continuation (expanded[3]) does NOT. Stream-
    emission order means we keep chunks 0 + 1 (LSB-side) and drop the
    MSB chunk. Sidecar = [0, 1]; the byte rows match the same
    ALG-8 formula as the full case.

    The inline_bytes buffer still contains all L=17 bytes (3a's per-
    source byte count is determined by the carrier's payload length,
    not the visible-chunk count). We just don't emit the MSB row.
    """
    L = 17
    raw_tokens = _vc2_raw_tokens(L)
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _VC2_SHIFTED, _VC2_SHIFTED, _VC2_SHIFTED],
        dtype=np.uint16,
    )
    extra_vc2 = np.array([False, False, True, True], dtype=bool)
    extra_f128 = np.array([False, False, False, False], dtype=bool)
    stage2_ct = _make_call_target(
        raw_tokens,
        expanded,
        extra_vc2,
        extra_f128,
        surviving_token_count=3,  # cut between expanded[2] and expanded[3]
    )
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)
    p = ct_slice.start  # == 1

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    expected_rows = np.array(
        [
            [p + 9, p + 10, p + 11, p + 12, p + 13, p + 14, p + 15, p + 16],
            [p + 1, p + 2, p + 3, p + 4, p + 5, p + 6, p + 7, p + 8],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.VALUED_CONST_V2], expected_rows
    )
    assert chunk_slices_per_type[TokenType.VALUED_CONST_V2] == [slice(0, 2)]
    np.testing.assert_array_equal(
        vc2_sidecar, np.array([0, 1], dtype=np.uint32)
    )


# ---------------------------------------------------------------------------
# Multi-call-target.
# ---------------------------------------------------------------------------


def test_per_call_target_slices_across_two_call_targets() -> None:
    """Test 9: per-call-target slices are correct across multiple call_targets.

    Build a 2-call-target variant where CT0 has an F32 source and CT1
    has a VC2 L=10 source. Verify each TokenType's per-CT slice list
    matches the expected concatenation order.
    """
    # CT0: F32 source.
    raw_ct0 = np.array([_F32_RAW, 0xA0, 0xA1, 0xA2, 0xA3], dtype=np.uint16)
    expanded_ct0 = np.array(
        [_LOCAL_FUNC_SHIFTED, _F32_SHIFTED], dtype=np.uint16
    )
    stage2_ct0 = _make_call_target(
        raw_ct0,
        expanded_ct0,
        extra_value_v2_mask=np.array([False, False], dtype=bool),
        extra_f128_mask=np.array([False, False], dtype=bool),
    )

    # CT1: VC2 L=10 source (K=2 -> 2 chunks). Carrier + 1 painted
    # continuation slot in the expanded stream.
    L = 10
    raw_ct1 = _vc2_raw_tokens(L)
    expanded_ct1 = np.array(
        [_LOCAL_FUNC_SHIFTED, _VC2_SHIFTED, _VC2_SHIFTED], dtype=np.uint16
    )
    stage2_ct1 = _make_call_target(
        raw_ct1,
        expanded_ct1,
        extra_value_v2_mask=np.array([False, False, True], dtype=bool),
        extra_f128_mask=np.array([False, False, False], dtype=bool),
    )

    # Stitch into one variant with two call_targets.
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=[stage2_ct0.stage1, stage2_ct1.stage1],
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )
    stage1_section = Stage1Section(
        arm=SectionKind.MATCHED,
        idx=0,
        section=_empty_section(),
        variants=[stage1_variant],
    )
    stage1_batch = Stage1Batch(
        sections=[stage1_section],
        batch_idx_to_section_variant=np.array([[0, 0]], dtype=np.uint32),
        batch_size=1,
    )
    stage2_variant = Stage2Variant(
        stage1=stage1_variant,
        call_targets=[stage2_ct0, stage2_ct1],
        cut_call_target_index=2,
        total_surviving_token_count=(
            stage2_ct0.surviving_token_count
            + stage2_ct1.surviving_token_count
        ),
        total_surviving_identity_count=0,
        total_surviving_number_chunk_count=3,
    )
    stage2_section = Stage2Section(
        stage1=stage1_section, variants=[stage2_variant]
    )
    stage2_batch = Stage2Batch(
        stage1=stage1_batch,
        sections=[stage2_section],
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
    )

    # Build inline_bytes for both call_targets: pad + CT0's 4 bytes +
    # CT1's 10 bytes.
    payload_ct0 = raw_ct0[raw_ct0 < 256].astype(np.uint8)
    payload_ct1 = raw_ct1[raw_ct1 < 256].astype(np.uint8)
    inline_bytes = np.empty(
        1 + payload_ct0.shape[0] + payload_ct1.shape[0], dtype=np.uint8
    )
    inline_bytes[0] = 0
    inline_bytes[1 : 1 + payload_ct0.shape[0]] = payload_ct0
    inline_bytes[1 + payload_ct0.shape[0] :] = payload_ct1
    ct0_slice = slice(1, 1 + payload_ct0.shape[0])  # [1, 5)
    ct1_slice = slice(
        1 + payload_ct0.shape[0],
        1 + payload_ct0.shape[0] + payload_ct1.shape[0],
    )  # [5, 15)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct0_slice, ct1_slice])

    # CT0's F32 row.
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT32],
        np.array([[1, 2, 3, 4]], dtype=np.uint32),
    )
    # CT1's VC2 rows: L=10, K=2. carrier_byte = 5.
    # chunk 0 (LSB): [5+10-8, 5+10) = [7, 15) -> 8 bytes.
    # chunk 1 (MSB): [5+10-16, 5+10-8) = [-1, 7), clipped to [5, 7)
    #   -> 2 actual bytes (5, 6) + 6 leading pad zeros.
    expected_vc2_rows = np.array(
        [
            [7, 8, 9, 10, 11, 12, 13, 14],
            [0, 0, 0, 0, 0, 0, 5, 6],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.VALUED_CONST_V2], expected_vc2_rows
    )

    # Per-CT slices.
    assert chunk_slices_per_type[TokenType.FLOAT32] == [
        slice(0, 1),  # CT0 contributes 1 F32 row
        slice(1, 1),  # CT1 contributes nothing
    ]
    assert chunk_slices_per_type[TokenType.VALUED_CONST_V2] == [
        slice(0, 0),  # CT0 contributes nothing
        slice(0, 2),  # CT1 contributes 2 VC2 rows
    ]
    np.testing.assert_array_equal(
        vc2_sidecar, np.array([0, 1], dtype=np.uint32)
    )


# ---------------------------------------------------------------------------
# Dtype + prepend handling.
# ---------------------------------------------------------------------------


def test_dtypes_and_sidecar_widths() -> None:
    """Test 10: idx_2d arrays are u32; vc2 sidecar u32; f128 flag bool."""
    raw_tokens = np.array(
        [_F128_RAW] + list(range(0x10, 0x20)), dtype=np.uint16
    )
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, _F128_SHIFTED, _F128_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False, False], dtype=bool)
    extra_f128 = np.array([False, False, True], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    for T, arr in idx_2d_per_type.items():
        assert arr.dtype == np.uint32, f"{T} idx_2d dtype = {arr.dtype}"
    assert vc2_sidecar.dtype == np.uint32
    assert f128_is_nan_or_inf.dtype == np.bool_


def test_prepend_slot_does_not_appear_in_number_idx_2d() -> None:
    """Test 11: no prepend in number idx_2d -- only in-stream number tokens.

    The expanded prepend slot at index 0 holds a LOCAL_FUNC / PLT_FUNC
    identity token (shifted id 9 or 10). It must NEVER contribute a
    number idx_2d row regardless of what follows in the stream.
    """
    # Mixed stream: prepend + identity (LOCAL_FUNC again, no payload) +
    # a single F16 source. Verify only the F16 source emits a row.
    # Use an IDENTITY token at id 264 (BLOCK_V2, raw); shifts to 8.
    raw_tokens = np.array(
        [
            264,        # BLOCK_V2 identity carrier (no payload here)
            _F16_RAW,   # F16 carrier
            0xBE,       # F16 byte 0
            0xEF,       # F16 byte 1
        ],
        dtype=np.uint16,
    )
    expanded = np.array(
        [_LOCAL_FUNC_SHIFTED, 8, _F16_SHIFTED], dtype=np.uint16
    )
    extra_vc2 = np.array([False, False, False], dtype=bool)
    extra_f128 = np.array([False, False, False], dtype=bool)
    stage2_ct = _make_call_target(raw_tokens, expanded, extra_vc2, extra_f128)
    stage2_batch = _wrap_single_call_target(stage2_ct)
    inline_bytes, ct_slice = _build_inline_bytes_from_raw(raw_tokens)

    (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_sidecar,
    ) = build_number_idx_2d(stage2_batch, inline_bytes, [ct_slice])

    # Only F16 emits; everything else empty.
    np.testing.assert_array_equal(
        idx_2d_per_type[TokenType.FLOAT16],
        np.array([[1, 2]], dtype=np.uint32),
    )
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        if T is TokenType.FLOAT16:
            continue
        assert idx_2d_per_type[T].shape[0] == 0
