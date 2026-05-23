"""Unit tests for :func:`expand_tokens`.

Single concern: pin the stage-2 per-call-target expansion algorithm
behavioural contract from ``batch_decode_plan.md`` ``## Stages --
algorithm sketch`` Stage 2 step 1 + ALG-2. Tests construct synthetic
:class:`Stage1CallTarget` fixtures directly -- expand_tokens reads only
``state.raw_tokens``, ``state.real_mask``, ``state.runlen_number``, and
``encounter_category``, so the rest of the dataclass is filled with
minimal-but-typed dummies.

The tests deliberately do NOT call :func:`build_inline_decode_state` --
that helper has its own test file and including its full state-machine
here would conflate two concerns. Instead each test builds the masks +
runlengths directly from the raw stream so the contract under test is
just expand_tokens itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    _FLOAT128_VOCAB_ID,
    _LOCAL_FUNC_SHIFTED,
    _PLT_FUNC_SHIFTED,
    _V2_IDENTITY_BLOCK_START,
    _V2_NUMBER_BLOCK_START,
    _V2_RESERVED_DIGIT_COUNT,
    _VC2_VOCAB_ID,
    ExpandedTokens,
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1CallTarget,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Fixture builders -- minimal-but-typed dummies (the same shape used by
# ``test_types.py``).
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


def _build_state(raw_tokens: np.ndarray) -> InlineDecodeState:
    """Build an InlineDecodeState with the fields expand_tokens reads.

    expand_tokens uses only ``raw_tokens``, ``real_mask``, and
    ``runlen_number``. The other fields are filled with values
    consistent with the inputs (so the dataclass invariants hold) but
    are NOT exercised by the function under test.
    """
    real_mask = raw_tokens > _V2_RESERVED_DIGIT_COUNT
    number_mask = raw_tokens < _V2_RESERVED_DIGIT_COUNT
    if raw_tokens.shape[0] == 0:
        runlen_number = np.zeros(0, dtype=np.uint16)
        runlen_value = np.zeros(0, dtype=np.uint16)
    else:
        # ``run_lengths`` asserts the first position is False; the test
        # streams below all start with a real-token carrier so this
        # holds. The dataclass requires real arrays; build them
        # honestly.
        runlen_number = run_lengths(number_mask)
        runlen_value = run_lengths(~real_mask)
    carries_inline_mask = real_mask & (raw_tokens < _V2_IDENTITY_BLOCK_START + 8)
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


def _u16(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Python reference implementation for the mixed-stream test.
#
# Deliberately a slow per-position loop. Asserts byte-equivalence with
# the vectorized implementation on a stream that mixes VC2, F128
# (finite + NaN + Inf), identity tokens, and trailing dead inline-digit
# slots.
# ---------------------------------------------------------------------------


def _reference_expand(call_target: Stage1CallTarget) -> ExpandedTokens:
    raw = call_target.state.raw_tokens.copy()
    real_mask = call_target.state.real_mask
    runlen_number = call_target.state.runlen_number
    n = int(raw.shape[0])

    vc2_promoted = np.zeros(n, dtype=bool)
    f128_promoted = np.zeros(n, dtype=bool)

    for p in range(n):
        if not real_mask[p]:
            continue
        token = int(raw[p])
        if token == _VC2_VOCAB_ID:
            if p == n - 1:
                raise AssertionError("VC2 at tail")
            L = int(runlen_number[p + 1])
            chunk_count = max(1, (L + 7) // 8)
            for k in range(1, chunk_count):
                raw[p + k] = _VC2_VOCAB_ID
                vc2_promoted[p + k] = True
        elif token == _FLOAT128_VOCAB_ID:
            if p + 2 >= n:
                raise AssertionError("F128 too close to tail")
            high_byte = int(raw[p + 1])
            low_byte = int(raw[p + 2])
            high_u16 = (high_byte << 8) | low_byte
            is_nan_or_inf = (high_u16 & 0x7FFF) == 0x7FFF
            if not is_nan_or_inf:
                raw[p + 1] = _FLOAT128_VOCAB_ID
                f128_promoted[p + 1] = True

    keep = raw > _V2_RESERVED_DIGIT_COUNT
    expanded_real = (raw[keep] - _V2_RESERVED_DIGIT_COUNT).astype(np.uint16)
    if call_target.encounter_category is Category.LOCAL_FUNC:
        prepend = _LOCAL_FUNC_SHIFTED
    elif call_target.encounter_category is Category.PLT_FUNC:
        prepend = _PLT_FUNC_SHIFTED
    else:
        raise AssertionError("unsupported category")

    expanded = np.concatenate([_u16(prepend), expanded_real])
    extra_vc2 = np.concatenate([np.array([False]), vc2_promoted[keep]])
    extra_f128 = np.concatenate([np.array([False]), f128_promoted[keep]])
    return ExpandedTokens(
        expanded_token_ids=expanded,
        extra_value_v2_mask=extra_vc2,
        extra_f128_mask=extra_f128,
        predicted_full_length=int(expanded.shape[0]),
    )


# ---------------------------------------------------------------------------
# Constant-layout sanity (catches drift from the plan vocab table).
# ---------------------------------------------------------------------------


def test_vocab_constants_match_plan_table():
    """Plan vocab table pins these. The module-level aliases must agree."""
    assert _V2_RESERVED_DIGIT_COUNT == 256
    assert _V2_NUMBER_BLOCK_START == 257
    assert _VC2_VOCAB_ID == 257
    # NUMBER block order: VC2, F16, BF16, F32, F64, F80, F128 (7 entries).
    assert _FLOAT128_VOCAB_ID == 263
    assert _V2_IDENTITY_BLOCK_START == 264
    # IDENTITY block order: BLOCK_V2, LOCAL_FUNC, PLT_FUNC, EXT_FUNC, ...
    assert _LOCAL_FUNC_SHIFTED == 9   # (264 + 1) - 256
    assert _PLT_FUNC_SHIFTED == 10    # (264 + 2) - 256


# ---------------------------------------------------------------------------
# Test 1: empty stream -> only the prepend.
# ---------------------------------------------------------------------------


def test_empty_stream_yields_prepend_only():
    ct = _make_call_target(_u16(), encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)
    np.testing.assert_array_equal(
        result.expanded_token_ids, _u16(_LOCAL_FUNC_SHIFTED)
    )
    assert result.predicted_full_length == 1
    np.testing.assert_array_equal(result.extra_value_v2_mask, np.array([False]))
    np.testing.assert_array_equal(result.extra_f128_mask, np.array([False]))


# ---------------------------------------------------------------------------
# Test 2: VC2 chunk-count formula. Six payload lengths -> expected chunks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload_length,expected_chunk_count",
    [
        (0, 1),
        (1, 1),
        (8, 1),
        (9, 2),
        (16, 2),
        (17, 3),
    ],
)
def test_vc2_chunk_count_formula(payload_length, expected_chunk_count):
    # Build: [VC2_carrier, byte_0, byte_1, ..., byte_{L-1}, BLOCK_V2_sentinel]
    # The trailing identity carrier (BLOCK_V2 = 264, NO payload of its
    # own) serves two purposes:
    #
    # 1. Provides the p+1 slot the algorithm reads for the VC2's
    #    payload-runlength. Without it the L=0 case would be a
    #    VC2-at-tail malformed-stream error, which is a SEPARATE
    #    concern tested by ``test_vc2_at_stream_tail_raises``.
    # 2. Makes the post-strip expanded stream layout explicit:
    #    [prepend, VC2 carrier, (chunk_count-1) promoted slots,
    #     BLOCK_V2 sentinel].
    #
    # The payload bytes are arbitrary < 256; pick 0x42.
    BLOCK_V2_RAW = _V2_IDENTITY_BLOCK_START  # 264
    BLOCK_V2_SHIFTED = BLOCK_V2_RAW - _V2_RESERVED_DIGIT_COUNT  # 8
    stream = [_VC2_VOCAB_ID] + [0x42] * payload_length + [BLOCK_V2_RAW]
    raw = _u16(*stream)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)

    result = expand_tokens(ct)

    # Expected: prepend + chunk_count VC2 tokens (shifted id = 1) +
    # BLOCK_V2 sentinel (shifted id = 8).
    vc2_shifted = _VC2_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT  # 1
    expected = _u16(
        _LOCAL_FUNC_SHIFTED,
        *([vc2_shifted] * expected_chunk_count),
        BLOCK_V2_SHIFTED,
    )
    np.testing.assert_array_equal(result.expanded_token_ids, expected)
    assert result.predicted_full_length == 2 + expected_chunk_count

    # extra_value_v2_mask: True at the PROMOTED VC2 slots only. The
    # carrier slot (index 1) and the sentinel slot (last) are False.
    # Promoted slots span indices 2..(1 + chunk_count - 1).
    expected_extra = np.zeros(2 + expected_chunk_count, dtype=bool)
    if expected_chunk_count > 1:
        expected_extra[2 : 1 + expected_chunk_count] = True
    np.testing.assert_array_equal(result.extra_value_v2_mask, expected_extra)

    # extra_f128_mask all False (no F128 sources in this fixture).
    np.testing.assert_array_equal(
        result.extra_f128_mask, np.zeros(2 + expected_chunk_count, dtype=bool)
    )


# ---------------------------------------------------------------------------
# Test 3-5: F128 finite vs NaN vs Inf.
# ---------------------------------------------------------------------------


def _f128_stream(high_byte: int, low_byte: int) -> np.ndarray:
    """Synthesise an F128 carrier followed by 16 big-endian payload bytes.

    Bytes p+1, p+2 hold ``high_byte``, ``low_byte`` -- ALG-2 reads only
    these two to decide NaN/Inf vs finite. The remaining 14 bytes are
    zero (mantissa-low half); irrelevant to the chunk-count decision
    but kept honest so the codec precondition (16 inline bytes follow
    the carrier) holds.
    """
    stream = [_FLOAT128_VOCAB_ID, high_byte, low_byte] + [0x00] * 14
    return _u16(*stream)


def test_f128_finite_is_two_chunks():
    # Finite: high u16 = 0x4000 (exponent != 0x7FFF). Mantissa is 0
    # below, but exponent is not all-ones so it's a finite value.
    raw = _f128_stream(0x40, 0x00)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)

    # F128 shifted id = 263 - 256 = 7. Finite -> 2 chunks (carrier +
    # painted p+1 slot).
    f128_shifted = _FLOAT128_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT
    expected = _u16(_LOCAL_FUNC_SHIFTED, f128_shifted, f128_shifted)
    np.testing.assert_array_equal(result.expanded_token_ids, expected)
    assert result.predicted_full_length == 3

    # extra_f128_mask True at the promoted slot (index 2; the second
    # F128 chunk). The carrier slot (index 1) is False.
    np.testing.assert_array_equal(
        result.extra_f128_mask, np.array([False, False, True])
    )
    # extra_value_v2_mask all False.
    np.testing.assert_array_equal(
        result.extra_value_v2_mask, np.array([False, False, False])
    )


def test_f128_nan_is_one_chunk():
    # NaN: exponent all-ones (high u16 = 0x7FFF), mantissa != 0.
    # high_byte = 0x7F, low_byte = 0xFF -> high u16 = 0x7FFF after
    # the 0x7FFF mask. Mantissa bits live in subsequent bytes; we put
    # a 1 at the low byte so the value is a quiet NaN, but ALG-2
    # doesn't care about mantissa for chunk-count.
    raw = _f128_stream(0x7F, 0xFF)
    # Override one of the low mantissa bytes so it is unambiguously a
    # NaN (mantissa != 0) -- not strictly needed for the chunk-count
    # test, but documents the test intent.
    raw_list = list(raw)
    raw_list[-1] = 0x01
    raw = _u16(*raw_list)

    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)

    f128_shifted = _FLOAT128_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT
    expected = _u16(_LOCAL_FUNC_SHIFTED, f128_shifted)
    np.testing.assert_array_equal(result.expanded_token_ids, expected)
    assert result.predicted_full_length == 2

    # No promoted F128 slot.
    np.testing.assert_array_equal(
        result.extra_f128_mask, np.array([False, False])
    )


def test_f128_inf_is_one_chunk():
    # +Inf: exponent all-ones, mantissa all-zero. high_byte = 0x7F,
    # low_byte = 0xFF; trailing 14 bytes already 0 via _f128_stream.
    # ALG-2 must treat this exactly like NaN for chunk-count purposes.
    raw = _f128_stream(0x7F, 0xFF)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)

    f128_shifted = _FLOAT128_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT
    expected = _u16(_LOCAL_FUNC_SHIFTED, f128_shifted)
    np.testing.assert_array_equal(result.expanded_token_ids, expected)
    assert result.predicted_full_length == 2


def test_f128_negative_inf_is_one_chunk():
    # -Inf: sign bit set, exponent all-ones. high_byte = 0xFF,
    # low_byte = 0xFF -> high u16 = 0xFFFF; after & 0x7FFF -> 0x7FFF.
    # Sign-bit-set ALSO classifies as NaN/Inf (sign is stripped by
    # the mask).
    raw = _f128_stream(0xFF, 0xFF)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)

    f128_shifted = _FLOAT128_VOCAB_ID - _V2_RESERVED_DIGIT_COUNT
    expected = _u16(_LOCAL_FUNC_SHIFTED, f128_shifted)
    np.testing.assert_array_equal(result.expanded_token_ids, expected)


# ---------------------------------------------------------------------------
# Test 6: stream-tail bounds. VC2 too close to tail must raise.
# ---------------------------------------------------------------------------


def test_vc2_at_stream_tail_raises():
    # VC2 carrier at the LAST position -> no p+1 slot to read.
    raw = _u16(_VC2_VOCAB_ID)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    with pytest.raises(AssertionError, match="VC2 carrier at the last"):
        expand_tokens(ct)


def test_f128_within_2_of_tail_raises():
    # F128 carrier needs bytes at p+1, p+2 -- so it must be at
    # position <= n-3. A carrier at n-2 is malformed.
    raw = _u16(_FLOAT128_VOCAB_ID, 0x00)  # only 1 byte after; need 2.
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    with pytest.raises(AssertionError, match="F128 carrier within 2"):
        expand_tokens(ct)


# ---------------------------------------------------------------------------
# Test 7-9: prepend dispatch on encounter_category.
# ---------------------------------------------------------------------------


def test_prepend_for_local_func():
    ct = _make_call_target(_u16(), encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)
    assert int(result.expanded_token_ids[0]) == 9


def test_prepend_for_plt_func():
    ct = _make_call_target(_u16(), encounter_category=Category.PLT_FUNC)
    result = expand_tokens(ct)
    assert int(result.expanded_token_ids[0]) == 10


def test_ext_func_encounter_category_raises():
    ct = _make_call_target(_u16(), encounter_category=Category.EXT_FUNC)
    with pytest.raises(AssertionError, match="encounter_category="):
        expand_tokens(ct)


def test_non_function_category_raises():
    # Sanity: BLOCK / STRING_PTR / etc are not call_target categories.
    ct = _make_call_target(_u16(), encounter_category=Category.BLOCK)
    with pytest.raises(AssertionError, match="encounter_category="):
        expand_tokens(ct)


# ---------------------------------------------------------------------------
# Test 10: mixed-stream byte-equivalence against the Python reference.
# ---------------------------------------------------------------------------


def test_mixed_stream_matches_python_reference():
    # Layout (raw stream, position : token):
    #   0 : BLOCK_V2 (identity carrier, no inline payload following)
    #   1 : VC2 carrier
    #   2 : payload byte (inline)
    #   3 : payload byte
    #   4 : payload byte
    #   5 : payload byte
    #   6 : payload byte
    #   7 : payload byte
    #   8 : payload byte
    #   9 : payload byte
    #  10 : payload byte    -> VC2 L=9 -> chunk_count=2
    #  11 : LOCAL_FUNC carrier (identity)
    #  12 : 0x42 (inline-digit -- belongs to LOCAL_FUNC's payload, runs into next)
    #  13 : F128 carrier (FINITE: high u16 != 0x7FFF)
    #  14 : 0x40 (high byte of payload)
    #  15 : 0x00 (low byte of payload)
    #  16..29 : zero mantissa bytes (14 bytes, padding the 16-byte payload)
    #  30 : F128 carrier (NaN)
    #  31 : 0x7F
    #  32 : 0xFF
    #  33..46 : zero mantissa bytes
    #  47 : PLT_FUNC carrier (identity, payload follows)
    #  48 : 0x05 (inline byte)
    #
    # Total length: 49.
    BLOCK_V2 = _V2_IDENTITY_BLOCK_START  # 264
    LOCAL_FUNC_RAW = _V2_IDENTITY_BLOCK_START + 1
    PLT_FUNC_RAW = _V2_IDENTITY_BLOCK_START + 2

    stream = [
        BLOCK_V2,
        _VC2_VOCAB_ID,
        # 9-byte payload:
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
        LOCAL_FUNC_RAW,
        0x42,
        _FLOAT128_VOCAB_ID,
        # finite payload (high u16 = 0x4000):
        0x40, 0x00,
        # 14 zero bytes:
        *([0x00] * 14),
        _FLOAT128_VOCAB_ID,
        # NaN payload (high u16 = 0x7FFF + nonzero mantissa):
        0x7F, 0xFF,
        # 13 zero bytes + 1 mantissa byte:
        *([0x00] * 13),
        0x01,
        PLT_FUNC_RAW,
        0x05,
    ]
    raw = _u16(*stream)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)

    actual = expand_tokens(ct)
    reference = _reference_expand(ct)

    np.testing.assert_array_equal(
        actual.expanded_token_ids, reference.expanded_token_ids
    )
    np.testing.assert_array_equal(
        actual.extra_value_v2_mask, reference.extra_value_v2_mask
    )
    np.testing.assert_array_equal(
        actual.extra_f128_mask, reference.extra_f128_mask
    )
    assert actual.predicted_full_length == reference.predicted_full_length

    # Spot-check the surviving token count:
    # - 1 prepend
    # - 1 BLOCK_V2 carrier
    # - 2 VC2 chunks (carrier + 1 promoted; L=9)
    # - 1 LOCAL_FUNC carrier
    # - 2 F128 chunks (finite carrier + 1 promoted)
    # - 1 F128 chunk (NaN; no promotion)
    # - 1 PLT_FUNC carrier
    # = 9 tokens.
    assert actual.predicted_full_length == 9


# ---------------------------------------------------------------------------
# Test 11: predicted_full_length matches expanded_token_ids.shape[0].
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_stream",
    [
        [],
        [_VC2_VOCAB_ID, 0x42],
        [_VC2_VOCAB_ID, *([0x42] * 17)],  # L=17 -> 3 chunks
        [_FLOAT128_VOCAB_ID, 0x40, 0x00, *([0x00] * 14)],
        [_FLOAT128_VOCAB_ID, 0x7F, 0xFF, *([0x00] * 14)],
        [_V2_IDENTITY_BLOCK_START],  # just an identity carrier
    ],
)
def test_predicted_full_length_invariant(raw_stream):
    raw = _u16(*raw_stream) if raw_stream else _u16()
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)
    result = expand_tokens(ct)
    assert result.predicted_full_length == int(
        result.expanded_token_ids.shape[0]
    )
    # And the extra masks share the same shape (invariant the cutoff
    # walk relies on -- it indexes into them with the same prefix).
    assert result.extra_value_v2_mask.shape[0] == result.predicted_full_length
    assert result.extra_f128_mask.shape[0] == result.predicted_full_length


# ---------------------------------------------------------------------------
# Test 12: multi-VC2 byte-equivalence -- many VC2 sources of varying
# chunk counts in a single stream. Pins the np.repeat-driven flat-index
# paint against the per-source scalar reference. The previous
# mixed-stream test only exercises one VC2 source; this one stacks
# several so an off-by-one in the cumulative-sum/repeat indexing
# surfaces immediately.
# ---------------------------------------------------------------------------


def test_multi_vc2_sources_match_python_reference():
    BLOCK_V2 = _V2_IDENTITY_BLOCK_START

    def vc2_with_payload(byte_count: int) -> list[int]:
        # VC2 carrier followed by ``byte_count`` inline-digit bytes
        # (each < 256 so it lives in the inline-digit band and is read
        # by ``runlen_number`` as part of a single contiguous number
        # run). The payload bytes are arbitrary; pick a varying nonzero
        # value so the resulting raw stream is also visually distinct.
        return [_VC2_VOCAB_ID] + [(i & 0x7F) | 0x01 for i in range(byte_count)]

    # Stack a deliberately varied set of payload lengths so the
    # resulting per-source chunk_counts span 1, 2, 3, and 4:
    #   L=0  -> 1 chunk  (no continuation slots painted)
    #   L=1  -> 1 chunk  (no continuation slots painted)
    #   L=8  -> 1 chunk  (no continuation slots painted)
    #   L=9  -> 2 chunks (1 continuation)
    #   L=16 -> 2 chunks (1 continuation)
    #   L=17 -> 3 chunks (2 continuations)
    #   L=24 -> 3 chunks (2 continuations)
    #   L=25 -> 4 chunks (3 continuations)
    #
    # Separate every VC2 group with a BLOCK_V2 identity sentinel so the
    # inline-digit run boundaries are unambiguous (a non-inline-digit
    # token closes the preceding number run -- see ``run_lengths``
    # semantics).
    stream: list[int] = [BLOCK_V2]
    for payload_len in (0, 1, 8, 9, 16, 17, 24, 25):
        stream.extend(vc2_with_payload(payload_len))
        stream.append(BLOCK_V2)

    raw = _u16(*stream)
    ct = _make_call_target(raw, encounter_category=Category.LOCAL_FUNC)

    actual = expand_tokens(ct)
    reference = _reference_expand(ct)

    np.testing.assert_array_equal(
        actual.expanded_token_ids, reference.expanded_token_ids
    )
    np.testing.assert_array_equal(
        actual.extra_value_v2_mask, reference.extra_value_v2_mask
    )
    np.testing.assert_array_equal(
        actual.extra_f128_mask, reference.extra_f128_mask
    )
    assert actual.predicted_full_length == reference.predicted_full_length
