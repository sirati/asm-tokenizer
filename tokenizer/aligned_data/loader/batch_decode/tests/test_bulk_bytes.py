"""Integration tests for stage 3 -- :func:`build_bulk_bytes`.

Single concern: pin the contract that :func:`build_bulk_bytes` composes
its four sub-modules (3a/3b/3c/3d) into a coherent :class:`Stage3Batch`
where per-call-target slices abut, identity slices reserve the prepend
slot at ``slice.start``, and the per-:class:`TokenType` number arrays
line up 1:1 with stage 4's needs.

These tests build synthetic :class:`Stage2Batch` fixtures around real
:class:`InlineDecodeState` and :func:`expand_tokens` outputs -- the
function under test reads ``state.raw_tokens`` / ``real_mask`` /
``number_mask`` / ``runlen_number`` / ``is_negative_per_position`` plus
the per-call-target ``expanded_token_ids`` + ``extra_*_mask`` +
surviving counts. The fixtures stay minimal (one section, one variant,
N call_targets) but go through :func:`expand_tokens` for realism on the
expanded-stream side and through :func:`predict_lengths` for the
end-to-end byte-equivalence test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._bulk_bytes import (
    _NUMBER_BLOCK_TOKEN_TYPES,
    build_bulk_bytes,
)
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._length_predict import (
    predict_lengths,
)
from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving,
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
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
    build_inline_decode_state,
)
from tokenizer.aligned_data.loader.decoded.custom_float import (
    from_float32,
    from_int,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Vocab anchors (kept local so a layout shift surfaces in this file too).
# ---------------------------------------------------------------------------

_VC2_RAW = 257
_F32_RAW = 260
_F64_RAW = 261
_F128_RAW = 263

_BLOCK_V2_RAW = 264  # IDENTITY block, slot 0
_LOCAL_FUNC_RAW = 265  # IDENTITY block, slot 1
_STRING_PTR_RAW = 268  # IDENTITY block, slot 4

_VALUE_NEGATIVE_RAW = 256

_LOCAL_FUNC_SHIFTED = 9
_BLOCK_V2_SHIFTED = 8


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
    """Build an :class:`InlineDecodeState` from a raw stream.

    For non-empty streams this delegates to
    :func:`build_inline_decode_state` so ``is_negative_per_position`` and
    the other derived fields match the production builder. The
    production builder doesn't handle the empty-stream case
    (``run_lengths`` asserts on ``mask[0]``), so we hand-construct an
    empty state directly for the empty path.
    """
    if raw_tokens.shape[0] == 0:
        return InlineDecodeState(
            raw_tokens=raw_tokens,
            real_mask=np.zeros(0, dtype=bool),
            number_mask=np.zeros(0, dtype=bool),
            runlen_number=np.zeros(0, dtype=np.uint16),
            runlen_value=np.zeros(0, dtype=np.uint16),
            carries_inline_mask=np.zeros(0, dtype=bool),
            is_negative_per_position=np.zeros(0, dtype=bool),
            digit_cumsum=np.zeros(1, dtype=np.uint32),
        )
    return build_inline_decode_state(raw_tokens, format_version=1)


def _make_stage1_ct(
    raw_tokens: np.ndarray,
    *,
    encounter_category: Category = Category.LOCAL_FUNC,
) -> Stage1CallTarget:
    """Build a :class:`Stage1CallTarget` from a raw stream."""
    state = _build_state(raw_tokens)
    return Stage1CallTarget(
        function_data=_empty_function_data(),
        state=state,
        call_targets_section=[],
        encounter_category=encounter_category,
        parent_call_target_index=None,
        function_name_ptr=0,
    )


def _make_stage2_ct(
    stage1_ct: Stage1CallTarget,
    *,
    cut_length: int | None = None,
) -> Stage2CallTarget:
    """Run the real stage-2a expand_tokens + surviving-count to build a
    :class:`Stage2CallTarget`. ``cut_length`` clamps
    ``surviving_token_count`` for testing cut paths.
    """
    expanded = expand_tokens(stage1_ct)
    full_len = expanded.predicted_full_length
    if cut_length is None:
        surviving_token_count = full_len
        is_cut = False
    else:
        assert 0 <= cut_length <= full_len
        surviving_token_count = cut_length
        is_cut = cut_length < full_len
    counts = count_surviving(
        expanded.expanded_token_ids, surviving_token_count
    )
    return Stage2CallTarget(
        stage1=stage1_ct,
        expanded_token_ids=expanded.expanded_token_ids,
        extra_value_v2_mask=expanded.extra_value_v2_mask,
        extra_f128_mask=expanded.extra_f128_mask,
        predicted_full_length=full_len,
        surviving_token_count=surviving_token_count,
        surviving_identity_count=counts.surviving_identity_count,
        surviving_number_chunk_count=counts.surviving_number_chunk_count,
        is_cut=is_cut,
        partial_cut_length=surviving_token_count,
    )


def _wrap_stage2_batch(
    stage2_call_targets: list[Stage2CallTarget],
) -> Stage2Batch:
    """Wrap a flat list of Stage2CallTargets in a single-section,
    single-variant Stage2Batch (DFS order = list order)."""
    stage1_cts = [ct.stage1 for ct in stage2_call_targets]
    stage1_variant = Stage1Variant(
        variant_idx=0,
        variant_ref_offset=0,
        batch_idx=0,
        call_targets=stage1_cts,
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
        call_targets=stage2_call_targets,
        cut_call_target_index=len(stage2_call_targets),
        total_surviving_token_count=sum(
            ct.surviving_token_count for ct in stage2_call_targets
        ),
        total_surviving_identity_count=sum(
            ct.surviving_identity_count for ct in stage2_call_targets
        ),
        total_surviving_number_chunk_count=sum(
            ct.surviving_number_chunk_count for ct in stage2_call_targets
        ),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_call_target_yields_empty_arrays_and_prepend_only_slice() -> None:
    """A call_target with NO number/identity carriers in the body yields:

    * ``inline_bytes == [0]`` (just the leading-zero pad);
    * empty per-:class:`TokenType` number arrays;
    * one identity_slice covering just the prepend (length 1);
    * ``identities_flat_caller_local`` length 1, all zeros.

    The prepend slot is itself an IDENTITY-band token (LOCAL_FUNC at
    expanded position 0) so ``surviving_identity_count == 1`` and the
    identity slice is ``[0, 1)`` -- stage 4 writes the prepend's
    caller-local counter at index 0 per ALG-9.
    """
    # Empty raw stream -> expand_tokens emits a single prepend slot.
    stage1 = _make_stage1_ct(np.zeros(0, dtype=np.uint16))
    stage2_ct = _make_stage2_ct(stage1)
    batch = _wrap_stage2_batch([stage2_ct])

    s3 = build_bulk_bytes(batch)

    assert isinstance(s3, Stage3Batch)
    np.testing.assert_array_equal(
        s3.inline_bytes, np.array([0], dtype=np.uint8)
    )
    # All per-TokenType arrays empty (still present in the dict).
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        sig, sign_exp = s3.numbers_per_TokenType[T]
        assert sig.shape == (0,)
        assert sign_exp.shape == (0,)
        assert s3.number_idx_2d_per_TokenType[T].shape[0] == 0
    # Identity: prepend slot only.
    assert s3.identities_flat_caller_local.shape == (1,)
    assert int(s3.identities_flat_caller_local[0]) == 0
    # Sidecars: empty.
    assert s3.vc2_chunk_exponent_sidecar.shape == (0,)
    # Per-call-target slices.
    ct3 = s3.sections[0].variants[0].call_targets[0]
    assert ct3.inline_byte_slice == slice(1, 1)
    assert ct3.identity_slice == slice(0, 1)
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        assert ct3.number_chunk_slices[T] == slice(0, 0)


def test_single_f32_source_normalises_via_oracle() -> None:
    """A single F32 source: one chunk per source; per-chunk
    ``(significand, sign_exp)`` byte-equivalent to the
    :func:`custom_float.from_float32` oracle.

    Stream: ``[F32_carrier, b0, b1, b2, b3]`` -> expanded:
    ``[prepend, F32]`` (the inline-digit bytes are stripped during the
    shift).
    """
    raw = np.array(
        [_F32_RAW, 0x40, 0x49, 0x0F, 0xDB],  # bit-pattern of pi (~3.14159)
        dtype=np.uint16,
    )
    stage1 = _make_stage1_ct(raw)
    stage2_ct = _make_stage2_ct(stage1)
    batch = _wrap_stage2_batch([stage2_ct])

    s3 = build_bulk_bytes(batch)

    sig, sign_exp = s3.numbers_per_TokenType[TokenType.FLOAT32]
    expected_chunks = from_float32(0x40490FDB)
    assert sig.shape == (1,)
    assert int(sig[0]) == int(expected_chunks[0][0])
    assert int(sign_exp[0]) == int(expected_chunks[0][1])
    # Per-CT slice into the F32 array.
    ct3 = s3.sections[0].variants[0].call_targets[0]
    assert ct3.number_chunk_slices[TokenType.FLOAT32] == slice(0, 1)
    # Other TokenTypes empty.
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        if T is TokenType.FLOAT32:
            continue
        assert s3.numbers_per_TokenType[T][0].shape == (0,)


def test_single_2byte_identity_token_view_cast_lands_at_post_prepend_slot() -> None:
    """A single 2-byte identity payload yields a big-endian u16 at
    ``identities_flat_caller_local[identity_slice.start + 1]``; the
    prepend slot at ``slice.start`` stays at 0.

    Stream: ``[STRING_PTR, b0, b1]`` -> expanded: ``[prepend, STRING_PTR]``.
    Caller-local id = ``(b0 << 8) | b1``.
    """
    b0, b1 = 0x12, 0x34
    raw = np.array([_STRING_PTR_RAW, b0, b1], dtype=np.uint16)
    stage1 = _make_stage1_ct(raw)
    stage2_ct = _make_stage2_ct(stage1)
    batch = _wrap_stage2_batch([stage2_ct])

    s3 = build_bulk_bytes(batch)

    ct3 = s3.sections[0].variants[0].call_targets[0]
    # identity_slice covers [prepend, in_stream_0] -> length 2.
    assert ct3.identity_slice.stop - ct3.identity_slice.start == 2
    # Prepend slot stays 0; in-stream slot holds the big-endian u16.
    assert int(s3.identities_flat_caller_local[ct3.identity_slice.start]) == 0
    assert int(
        s3.identities_flat_caller_local[ct3.identity_slice.start + 1]
    ) == ((b0 << 8) | b1)


def test_mixed_stream_vc2_identity_f64_populates_all_arms() -> None:
    """Mixed stream exercising VC2 multi-chunk + identity + F64 in one
    call_target. F128 is exercised in dedicated tests below.

    Stream layout (raw):
        [VC2, b00..b09,            # VC2 K=2 source (L=10)
         BLOCK_V2, 0xAB, 0xCD,      # identity 2-byte payload
         F64, d0..d7]               # F64 source

    Expected:
      * inline_bytes: pad + 10 VC2 bytes + 2 identity bytes + 8 F64
        bytes = 1 + 20 = 21 bytes.
      * VC2 array: 2 rows (LSB + MSB chunks).
      * F64 array: 1 row.
      * identity_idx_2d: 1 row for the BLOCK_V2 carrier.
      * Per-CT slices abut: every other TokenType stays empty.
    """
    vc2_bytes = list(range(0x10, 0x10 + 10))
    f64_bytes = list(range(0x80, 0x80 + 8))
    raw = np.array(
        [_VC2_RAW] + vc2_bytes
        + [_BLOCK_V2_RAW, 0xAB, 0xCD]
        + [_F64_RAW] + f64_bytes,
        dtype=np.uint16,
    )
    stage1 = _make_stage1_ct(raw)
    stage2_ct = _make_stage2_ct(stage1)
    batch = _wrap_stage2_batch([stage2_ct])

    s3 = build_bulk_bytes(batch)

    # inline_bytes shape.
    assert s3.inline_bytes.shape == (1 + 20,)
    # VC2: 2 chunks, sidecar [0, 1] (chunk indices within source).
    vc2_sig, _ = s3.numbers_per_TokenType[TokenType.VALUED_CONST_V2]
    assert vc2_sig.shape == (2,)
    np.testing.assert_array_equal(
        s3.vc2_chunk_exponent_sidecar,
        np.array([0, 1], dtype=np.uint32),
    )
    # F64: 1 chunk.
    f64_sig, _ = s3.numbers_per_TokenType[TokenType.FLOAT64]
    assert f64_sig.shape == (1,)
    # Identity: 1 in-stream row (the BLOCK_V2 carrier).
    assert s3.identity_idx_2d.shape == (1, 2)
    # In-stream u16 lands at slice.start + 1.
    ct3 = s3.sections[0].variants[0].call_targets[0]
    assert int(
        s3.identities_flat_caller_local[ct3.identity_slice.start + 1]
    ) == ((0xAB << 8) | 0xCD)
    # Per-CT slices.
    assert ct3.number_chunk_slices[TokenType.VALUED_CONST_V2] == slice(0, 2)
    assert ct3.number_chunk_slices[TokenType.FLOAT64] == slice(0, 1)
    for T in (
        TokenType.FLOAT16,
        TokenType.BFLOAT16,
        TokenType.FLOAT32,
        TokenType.FLOAT80,
        TokenType.FLOAT128,
    ):
        assert ct3.number_chunk_slices[T] == slice(0, 0)


def test_multi_call_target_chain_identity_slices_abut() -> None:
    """Two call_targets in a single variant: identity slices abut
    end-to-end across call_targets; per-:class:`TokenType` number
    slices line up across the chain.

    CT_0: [BLOCK_V2, 0x01, 0x02]               (one 2-byte identity)
    CT_1: [BLOCK_V2, 0x03, 0x04, F32, d0..d3]  (identity + F32)
    """
    raw0 = np.array(
        [_BLOCK_V2_RAW, 0x01, 0x02], dtype=np.uint16
    )
    raw1 = np.array(
        [_BLOCK_V2_RAW, 0x03, 0x04, _F32_RAW, 0xAA, 0xBB, 0xCC, 0xDD],
        dtype=np.uint16,
    )
    ct0 = _make_stage2_ct(_make_stage1_ct(raw0))
    ct1 = _make_stage2_ct(_make_stage1_ct(raw1))
    batch = _wrap_stage2_batch([ct0, ct1])

    s3 = build_bulk_bytes(batch)

    ct0_s3 = s3.sections[0].variants[0].call_targets[0]
    ct1_s3 = s3.sections[0].variants[0].call_targets[1]
    # Identity slices abut: ct1.start == ct0.stop.
    assert ct0_s3.identity_slice.start == 0
    assert ct1_s3.identity_slice.start == ct0_s3.identity_slice.stop
    # ct0 surviving_identity_count = 1 prepend + 1 in-stream = 2.
    # ct1 surviving_identity_count = 1 prepend + 1 in-stream = 2.
    assert ct0_s3.identity_slice == slice(0, 2)
    assert ct1_s3.identity_slice == slice(2, 4)
    # Prepends stay 0; in-stream u16s reach the right slots.
    assert int(s3.identities_flat_caller_local[0]) == 0  # ct0 prepend
    assert int(s3.identities_flat_caller_local[1]) == ((0x01 << 8) | 0x02)
    assert int(s3.identities_flat_caller_local[2]) == 0  # ct1 prepend
    assert int(s3.identities_flat_caller_local[3]) == ((0x03 << 8) | 0x04)
    # F32 slice on ct1.
    assert ct0_s3.number_chunk_slices[TokenType.FLOAT32] == slice(0, 0)
    assert ct1_s3.number_chunk_slices[TokenType.FLOAT32] == slice(0, 1)
    # Inline-byte slices abut.
    assert ct0_s3.inline_byte_slice.start == 1
    assert ct1_s3.inline_byte_slice.start == ct0_s3.inline_byte_slice.stop


def test_cut_call_target_drops_post_cut_chunks_and_identities() -> None:
    """A cut call_target's surviving prefix limits the per-type
    contributions: dropped sources contribute 0 chunks; dropped
    identities are absent from ``identity_idx_2d`` and the in-stream
    u16 sub-array.

    Stream: ``[F32, d0..d3, BLOCK_V2, 0xAA, 0xBB, F32, ...]``
    Expanded: ``[prepend, F32, BLOCK_V2, F32]`` (length 4).
    Cut at 2: prepend + first F32 survives -> 1 F32 chunk, 0 in-stream
    identities, slice length 1 (prepend only).
    """
    raw = np.array(
        [_F32_RAW, 0xAA, 0xBB, 0xCC, 0xDD,
         _BLOCK_V2_RAW, 0xEE, 0xFF,
         _F32_RAW, 0x11, 0x22, 0x33, 0x44],
        dtype=np.uint16,
    )
    stage1 = _make_stage1_ct(raw)
    full = _make_stage2_ct(stage1).predicted_full_length
    assert full == 4
    # Cut after the first F32 (slot 1 visible -> length 2).
    stage2_ct = _make_stage2_ct(stage1, cut_length=2)
    batch = _wrap_stage2_batch([stage2_ct])

    s3 = build_bulk_bytes(batch)

    # 1 F32 chunk survives.
    sig, _ = s3.numbers_per_TokenType[TokenType.FLOAT32]
    assert sig.shape == (1,)
    # No in-stream identities surviving (only the prepend at expanded[0]).
    ct3 = s3.sections[0].variants[0].call_targets[0]
    assert ct3.identity_slice == slice(0, 1)
    assert s3.identity_idx_2d.shape == (0, 2)


def test_fully_dropped_trailing_call_target_zero_length_slices() -> None:
    """When the cut leaves a trailing call_target with 0 surviving
    tokens, that CT contributes zero-length slices on every axis."""
    raw = np.array([_F32_RAW, 0xAA, 0xBB, 0xCC, 0xDD], dtype=np.uint16)
    ct0 = _make_stage2_ct(_make_stage1_ct(raw))
    # Trailing CT fully dropped (cut at 0 -> 0 surviving tokens).
    raw1 = np.array([_F32_RAW, 0x11, 0x22, 0x33, 0x44], dtype=np.uint16)
    stage1_1 = _make_stage1_ct(raw1)
    ct1 = _make_stage2_ct(stage1_1, cut_length=0)
    batch = _wrap_stage2_batch([ct0, ct1])

    s3 = build_bulk_bytes(batch)

    ct1_s3 = s3.sections[0].variants[0].call_targets[1]
    # Zero-length slices on every axis.
    assert ct1_s3.inline_byte_slice.stop - ct1_s3.inline_byte_slice.start == 0
    assert ct1_s3.identity_slice.stop - ct1_s3.identity_slice.start == 0
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        sl = ct1_s3.number_chunk_slices[T]
        assert sl.stop - sl.start == 0


def test_f128_nan_emits_single_chunk_finite_emits_two() -> None:
    """F128 NaN/Inf -> 1 chunk per source; F128 finite -> 2 chunks per
    source. Verified via two single-source fixtures in one batch.

    NaN signal: high u16's exponent bits all-ones (binary128 exponent
    mask 0x7FFF). E.g. byte0=0x7F, byte1=0xFF -> exp == 0x7FFF.
    Finite signal: any other high-u16 exponent value.
    """
    nan_bytes = [0x7F, 0xFF] + [0x00] * 14  # exp = 0x7FFF -> NaN/Inf
    finite_bytes = [0x40, 0x00] + list(range(0x10, 0x10 + 14))  # exp != 0x7FFF
    raw_nan = np.array([_F128_RAW] + nan_bytes, dtype=np.uint16)
    raw_fin = np.array([_F128_RAW] + finite_bytes, dtype=np.uint16)
    ct_nan = _make_stage2_ct(_make_stage1_ct(raw_nan))
    ct_fin = _make_stage2_ct(_make_stage1_ct(raw_fin))
    batch = _wrap_stage2_batch([ct_nan, ct_fin])

    s3 = build_bulk_bytes(batch)

    sig, _ = s3.numbers_per_TokenType[TokenType.FLOAT128]
    # 1 chunk (NaN) + 2 chunks (finite) = 3 total.
    assert sig.shape == (3,)
    ct_nan_s3 = s3.sections[0].variants[0].call_targets[0]
    ct_fin_s3 = s3.sections[0].variants[0].call_targets[1]
    assert ct_nan_s3.number_chunk_slices[TokenType.FLOAT128] == slice(0, 1)
    assert ct_fin_s3.number_chunk_slices[TokenType.FLOAT128] == slice(1, 3)


def test_vc2_sign_threaded_per_source_across_chunks() -> None:
    """A negative-signed VC2 multi-chunk source: the
    ``value_negative`` postfix marker (id 256) at p+L+1 sets
    ``is_negative_per_position`` True at the carrier. Both chunks of
    the source get the same sign downstream via
    :func:`_fp_normalize.vc2_per_chunk_sign`.

    Cross-check vs :func:`custom_float.from_int` oracle: a negative
    integer of magnitude that needs 2 VC2 chunks (e.g. ``-(1 << 70)``)
    produces 2 ``(significand, sign_exp)`` chunks whose sign bit reflects
    the source-level negative sign.
    """
    # value = -(1 << 70); magnitude bit_length = 71 bits -> 9 bytes -> K=2
    # (ceil(9/8)). Big-endian byte layout of magnitude in 9 bytes:
    # 0x40 0x00 .. (the high byte is 0x40 = 1 << 6 because bit 70 is set).
    magnitude_bytes_be = (1 << 70).to_bytes(9, "big")
    raw = np.array(
        [_VC2_RAW] + list(magnitude_bytes_be) + [_VALUE_NEGATIVE_RAW],
        dtype=np.uint16,
    )
    stage1 = _make_stage1_ct(raw)
    # Sanity check: the state's is_negative_per_position correctly
    # detected the negative postfix at the VC2 carrier (raw position 0).
    assert bool(stage1.state.is_negative_per_position[0]) is True
    stage2_ct = _make_stage2_ct(stage1)
    batch = _wrap_stage2_batch([stage2_ct])

    s3 = build_bulk_bytes(batch)

    sig, sign_exp = s3.numbers_per_TokenType[TokenType.VALUED_CONST_V2]
    # 2 chunks (K = ceil(9/8) = 2).
    assert sig.shape == (2,)
    # Oracle compare: from_int takes the magnitude + sign separately.
    expected = from_int(1 << 70, sign=-1)
    assert len(expected) == 2
    np.testing.assert_array_equal(
        sig, np.array([int(c[0]) for c in expected], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        sign_exp,
        np.array([int(c[1]) for c in expected], dtype=np.uint32),
    )


def test_signed_carrier_after_surviving_one_call_target_keeps_sign() -> None:
    """A zero-length-body call_target wedged before a signed carrier must
    not corrupt the carrier's sign attribution.

    A kept call_target with ``surviving == 1`` (only the prepend survives)
    has a ZERO-LENGTH body segment. The batched sign collection builds
    per-call_target segment ids over those bodies; a CSR mark-and-cumsum
    expansion silently MERGES the zero-length boundary into the next
    segment, shifting every later carrier onto the WRONG call_target's
    ``real_positions`` -> wrong sign. The empty-safe ``np.repeat`` build
    keeps the zero-length segment, so the trailing negative VC2 carrier
    still reads its own ``value_negative`` marker.

    Layout (DFS order):
      CT_0: positive F32 carrier (real, nonzero-length body).
      CT_1: identity stream cut to surviving==1 (prepend only -> empty body).
      CT_2: negative-signed VC2 multi-chunk source.

    Without the fix CT_2's two chunks come out positive (sign mismatch);
    with the fix they match the ``from_int(magnitude, sign=-1)`` oracle.
    """
    raw0 = np.array(
        [_F32_RAW, 0x40, 0x49, 0x0F, 0xDB], dtype=np.uint16
    )
    ct0 = _make_stage2_ct(_make_stage1_ct(raw0))

    raw1 = np.array([_BLOCK_V2_RAW, 0x01, 0x02], dtype=np.uint16)
    ct1 = _make_stage2_ct(_make_stage1_ct(raw1), cut_length=1)
    # Sanity: CT_1 is kept (surviving==1) with a zero-length body.
    assert ct1.surviving_token_count == 1

    magnitude_bytes_be = (1 << 70).to_bytes(9, "big")
    raw2 = np.array(
        [_VC2_RAW] + list(magnitude_bytes_be) + [_VALUE_NEGATIVE_RAW],
        dtype=np.uint16,
    )
    stage1_2 = _make_stage1_ct(raw2)
    # Sanity: the negative postfix marker registered at the VC2 carrier.
    assert bool(stage1_2.state.is_negative_per_position[0]) is True
    ct2 = _make_stage2_ct(stage1_2)

    batch = _wrap_stage2_batch([ct0, ct1, ct2])
    s3 = build_bulk_bytes(batch)

    sig, sign_exp = s3.numbers_per_TokenType[TokenType.VALUED_CONST_V2]
    assert sig.shape == (2,)
    expected = from_int(1 << 70, sign=-1)
    assert len(expected) == 2
    np.testing.assert_array_equal(
        sig, np.array([int(c[0]) for c in expected], dtype=np.uint64)
    )
    # The sign-bearing field: must match the NEGATIVE oracle. The bug
    # attributed CT_1's (empty) sign band to CT_2, yielding positive chunks.
    np.testing.assert_array_equal(
        sign_exp,
        np.array([int(c[1]) for c in expected], dtype=np.uint32),
    )


def test_end_to_end_predict_lengths_into_build_bulk_bytes() -> None:
    """End-to-end byte-equivalence: build Stage1Batch -> predict_lengths
    -> build_bulk_bytes, verify the per-:class:`TokenType` chunks match
    the per-source Python-loop oracles AND the per-CT slice contracts.

    Stream: a single F32 source (numeric pi 0x40490FDB) so the test pins
    one bit-pattern through the entire pipeline.
    """
    raw = np.array(
        [_F32_RAW, 0x40, 0x49, 0x0F, 0xDB], dtype=np.uint16
    )
    stage1_ct = _make_stage1_ct(raw)
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

    # context_len large enough to fit the full stream.
    stage2 = predict_lengths(stage1_batch, context_len=1024)
    s3 = build_bulk_bytes(stage2)

    # F32 chunk matches oracle.
    sig, sign_exp = s3.numbers_per_TokenType[TokenType.FLOAT32]
    expected = from_float32(0x40490FDB)
    assert sig.shape == (1,)
    assert int(sig[0]) == int(expected[0][0])
    assert int(sign_exp[0]) == int(expected[0][1])

    # Stage3Batch invariants.
    assert s3.stage2 is stage2
    # Inline-byte slice abuts the leading pad.
    ct3 = s3.sections[0].variants[0].call_targets[0]
    assert ct3.inline_byte_slice.start == 1
    assert ct3.inline_byte_slice.stop == 1 + 4
    # Identity slice = just the prepend.
    assert ct3.identity_slice == slice(0, 1)
    # All per-TokenType arrays line up: only F32 non-empty.
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        sig_T, sign_exp_T = s3.numbers_per_TokenType[T]
        expected_n = 1 if T is TokenType.FLOAT32 else 0
        assert sig_T.shape == (expected_n,)
        assert sign_exp_T.shape == (expected_n,)
