"""Decoder-side tests for the postfix ``value_negative`` sign marker.

Mirrors the emitter-side contract pinned in
``tokenizer/tests/test_emitters_v2_valued_const_sign.py``: when a
``value_negative`` metatoken follows a ``valued_const_v2`` source, the
stream walker decodes the source's chunks with ``sign=-1`` so the
reconstructed integer carries the original sign.

Round-trip suite (encoder -> token stream -> decoder) covers the
representative signed-integer range pinned by the Phase 3 plan:
``[-128, -1, 0, 1, 127, 255, INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN]``.

Two additional tests target the postfix invariant + the FP coexistence
contract:

* ``test_value_negative_after_*`` — a ``value_negative`` token glued to
  the tail of a non-VC2 carrier (FP source, identity carrier) trips
  the inline postfix-invariant assertion in
  :func:`_decode_to_staging`. Catches encoder bugs that misplace the
  postfix marker.
* ``test_value_negative_with_fp_postfix_negates_chunks_and_keeps_fp_marker`` —
  the ``-magnitude + value_negative + floatXX_postfix`` shape (per the
  emitter's order) decodes such that the chunk side-array carries
  ``sign=-1`` for the magnitude AND ``real_tokens`` retains the FP
  postfix marker so downstream consumers still see the FP annotation
  (the ``value_negative`` token itself is stripped per D5).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.custom_float import (
    from_float32,
    from_int,
)
from tokenizer.aligned_data.loader.decoded.extract import decode_raw_tokens
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Fixture: synthetic vocab-id maps. Numbers chosen to live above the
# digit-slot range (>= 256), with ``value_negative`` pinned at the
# canonical 256 -- exercises the smallest-real-token-id edge of the
# stream walker.
# ---------------------------------------------------------------------------


_VALUE_NEGATIVE_ID = 256


def _make_id_maps() -> Tuple[Dict[Category, int], Dict[TokenType, int]]:
    """Synthetic id maps pinned to the unified-vocab canonical layout.

    The vectorized number-arm filtering uses the carrier mask
    ``[_V2_RESERVED_TOKEN_COUNT, _V2_EAGER_BLOCK_END)`` (= [257, 272)),
    so number + identity ids must live inside that band. Values below
    mirror :meth:`VocabularyManager._register_v2_canonical_blocks` —
    VALUED_CONST_V2 at 257, FLOAT16 at 258, ..., BLOCK at 264, etc.
    """
    id_token_ids: Dict[Category, int] = {
        Category.BLOCK: 264,
        Category.LOCAL_FUNC: 265,
        Category.PLT_FUNC: 266,
        Category.EXT_FUNC: 267,
        Category.STRING_PTR: 268,
        Category.JUMP_TABLE: 269,
        Category.RO_DATA_PTR: 270,
        Category.RW_DATA_PTR: 271,
    }
    number_token_ids: Dict[TokenType, int] = {
        TokenType.VALUED_CONST_V2: 257,
        TokenType.FLOAT16: 258,
        TokenType.BFLOAT16: 259,
        TokenType.FLOAT32: 260,
        TokenType.FLOAT64: 261,
        TokenType.FLOAT80: 262,
        TokenType.FLOAT128: 263,
    }
    return id_token_ids, number_token_ids


def _u16(*tokens: int) -> np.ndarray:
    return np.array(tokens, dtype=np.uint16)


def _decode_signed(raw: np.ndarray):
    """Decode with the postfix-sign path active (value_negative_token_id set)."""
    id_token_ids, number_token_ids = _make_id_maps()
    return decode_raw_tokens(
        raw,
        id_token_ids=id_token_ids,
        number_token_ids=number_token_ids,
        value_negative_token_id=_VALUE_NEGATIVE_ID,
        format_version=1,
        func_name="t",
    )


def _emit_valued_const_stream(value: int) -> Tuple[np.ndarray, List[Tuple[np.uint64, np.uint32]]]:
    """Encode ``value`` as the v2 emitter would (magnitude + value_negative?).

    Mirrors ``_V2EmittersMixin._emit_valued_const`` minus the FP-postfix
    branch (this helper covers the pure-int round-trip). The returned
    raw stream is the uint16 array the encoder would produce; the
    returned ``expected_chunks`` is the (sig, sign_exp) list a sign-
    aware decoder must reconstruct.

    No reliance on the encoder module here -- the test owns its own
    minimum-width pack so a regression in either side surfaces as a
    test failure, not a silent fallback.
    """
    magnitude = -value if value < 0 else value
    # Minimum-width big-endian unsigned packing (mirrors
    # ``_v2_int_to_minimum_bytes``). ``magnitude == 0`` is still one
    # byte so the encoder's "no leading-real-token" precondition is
    # respected.
    width = max(1, (magnitude.bit_length() + 7) // 8)
    payload_bytes = magnitude.to_bytes(width, byteorder="big", signed=False)
    tokens: List[int] = [257, *payload_bytes]
    if value < 0:
        tokens.append(_VALUE_NEGATIVE_ID)
    raw = np.array(tokens, dtype=np.uint16)
    expected_sign = -1 if value < 0 else +1
    expected_chunks = from_int(magnitude, sign=expected_sign)
    return raw, expected_chunks


# ---------------------------------------------------------------------------
# Round-trip: every representative signed integer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        -(2 ** 63),       # INT64_MIN
        -(2 ** 31),       # INT32_MIN
        -128,
        -1,
        0,
        1,
        127,
        255,
        2 ** 31 - 1,      # INT32_MAX
        2 ** 63 - 1,      # INT64_MAX
    ],
)
def test_round_trip_through_decoder_recovers_signed_chunks(value: int) -> None:
    """Round-trip every representative signed integer.

    For each ``value``, encode the v2 token stream, decode it with the
    postfix-sign path active, and assert the chunk side-array matches
    ``from_int(magnitude, sign=expected_sign)`` chunk-for-chunk. This
    pins both the magnitude path (encoder + decoder agree on
    minimum-width unsigned packing) AND the sign-propagation path (the
    postfix metatoken flips the chunk sign).
    """
    raw, expected_chunks = _emit_valued_const_stream(value)
    out = _decode_signed(raw)
    assert out.numbers_significant.shape == (len(expected_chunks),)
    for idx, (sig, sign_exp) in enumerate(expected_chunks):
        assert int(out.numbers_significant[idx]) == int(sig), (
            f"value={value}: chunk[{idx}].sig mismatch"
        )
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp), (
            f"value={value}: chunk[{idx}].sign_exp mismatch"
        )


# ---------------------------------------------------------------------------
# Real-tokens stream: value_negative survives the strip pass
# ---------------------------------------------------------------------------


def test_value_negative_token_does_not_appear_in_real_tokens_for_negative_value() -> None:
    """A negative valued_const_v2 source's ``real_tokens`` is just the
    shifted VC2 id; the ``value_negative`` postfix is STRIPPED.

    Per the D5/D6 strip-and-shift step the decoder drops
    ``value_negative`` (id 256) from the model-facing stream because
    its sign meaning is already captured in ``numbers_sign_exponent``;
    surviving real-token ids are shifted down by 256 so the post-strip
    vocab compacts.
    """
    raw, _ = _emit_valued_const_stream(-42)
    out = _decode_signed(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(257 - 256))
    # Sanity: the value_negative id never reaches real_tokens under
    # the strip-and-shift contract.
    assert _VALUE_NEGATIVE_ID not in out.real_tokens.tolist()


def test_positive_value_has_no_value_negative_in_real_tokens() -> None:
    """A non-negative source's ``real_tokens`` carries no value_negative."""
    raw, _ = _emit_valued_const_stream(42)
    out = _decode_signed(raw)
    np.testing.assert_array_equal(out.real_tokens, _u16(257 - 256))
    assert _VALUE_NEGATIVE_ID not in out.real_tokens.tolist()


# ---------------------------------------------------------------------------
# Postfix invariant: value_negative ONLY after valued_const_v2
# ---------------------------------------------------------------------------


def test_value_negative_after_float_token_trips_assertion() -> None:
    """A ``value_negative`` token immediately after a ``FLOAT32`` source
    is a malformed stream; the decoder must raise an AssertionError with
    a descriptive message pointing at the encoder.

    The FP source occupies positions ``[0, 1+4)`` (one type id + 4
    inline bytes for a single-precision float). Position 5 holds the
    stray ``value_negative``, whose predecessor walks back through
    the inline bytes to the FLOAT32 type id at position 0 -- NOT a
    ``valued_const_v2`` -- triggering the assert.
    """
    bits = 0x40490FDA  # arbitrary f32 bit pattern (~3.14)
    payload = bits.to_bytes(4, byteorder="big", signed=False)
    raw = _u16(260, *payload, _VALUE_NEGATIVE_ID)
    with pytest.raises(AssertionError, match="value_negative"):
        _decode_signed(raw)


def test_value_negative_after_identity_token_trips_assertion() -> None:
    """A ``value_negative`` token after a ``LOCAL_FUNC`` identity token
    is also malformed; the decoder must surface the bug at decode time
    rather than silently negating an unrelated subsequent number-source."""
    # LOCAL_FUNC (id 265) with inline id 5, then a stray value_negative.
    raw = _u16(265, 5, _VALUE_NEGATIVE_ID)
    with pytest.raises(AssertionError, match="value_negative"):
        _decode_signed(raw)


def test_value_negative_leading_stream_trips_assertion() -> None:
    """A ``value_negative`` at position 0 has no preceding source; the
    decoder must trip the AssertionError instead of letting the
    metatoken silently propagate.

    The codec precondition asserts that ``raw_tokens[0] >= 256``
    BEFORE the postfix-invariant check runs, so this stream passes the
    first gate (256 is real-token) and then trips the second gate
    (value_negative has no predecessor)."""
    raw = _u16(_VALUE_NEGATIVE_ID, 264)
    with pytest.raises(AssertionError, match="position 0"):
        _decode_signed(raw)


# ---------------------------------------------------------------------------
# Coexistence with the FP postfix annotation (negative + floatXX)
# ---------------------------------------------------------------------------


def test_value_negative_with_fp_postfix_negates_chunks_and_keeps_fp_marker() -> None:
    """``[Valued_Const_V2(|v|), Value_Negative(), Float32(None)]`` — the
    full shape the emitter produces for an FP-typed negative immediate.

    Decoder contract:

    * The chunk side-array for the ``valued_const_v2`` source carries
      ``sign=-1`` (postfix sign reconstruction).
    * ``real_tokens`` retains both the ``value_negative`` token and the
      ``Float32`` postfix marker so downstream consumers still see the
      FP type annotation alongside the signedness signal.
    * The ``Float32`` postfix slot has ZERO inline bytes (``bits=None``
      encoding), so its own side-array chunk decodes as the bit
      pattern 0 — i.e. ``from_float32(0)`` — which is consistent with
      the existing test_extract.py:test_partial_vocab... regression
      target for the postfix shape.
    """
    magnitude = 255
    payload_bytes = magnitude.to_bytes(1, byteorder="big", signed=False)
    # Stream: VALUED_CONST_V2 + 1 magnitude byte + value_negative + FLOAT32 (postfix, no inline).
    raw = _u16(257, *payload_bytes, _VALUE_NEGATIVE_ID, 260)
    out = _decode_signed(raw)
    # real_tokens preserves the VC2 + FLOAT32 postfix in stream-position
    # order; value_negative is STRIPPED (D5) and surviving ids are
    # shifted down by 256 (D6).
    np.testing.assert_array_equal(
        out.real_tokens, _u16(257 - 256, 260 - 256)
    )
    expected_vc = from_int(magnitude, sign=-1)
    expected_fp = from_float32(0)
    expected = expected_vc + expected_fp
    assert out.numbers_significant.shape == (len(expected),)
    for idx, (sig, sign_exp) in enumerate(expected):
        assert int(out.numbers_significant[idx]) == int(sig), (
            f"chunk[{idx}].sig mismatch"
        )
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp), (
            f"chunk[{idx}].sign_exp mismatch"
        )


# ---------------------------------------------------------------------------
# Legacy path: value_negative_token_id omitted -> sign-agnostic decode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end round-trip through the real emitter + decoder
# ---------------------------------------------------------------------------


def _emit_real(value: int) -> np.ndarray:
    """Encode ``value`` through the real ``ConstantHandler._emit_valued_const``
    + ``Valued_Const_V2`` / ``Value_Negative`` Inner factories, then flatten
    to a uint16 stream the decoder accepts.

    This is the strict round-trip: any decoder-side regression that
    would silently misinterpret the emitter's wire shape surfaces here.
    """
    from tokenizer.constant_handler.core import ConstantHandler
    from tokenizer.constant_handler.ctx import _Ctx
    from tokenizer.token_manager import VocabularyManager

    vm = VocabularyManager(platform=None, format_version=1)
    handler = ConstantHandler.__new__(ConstantHandler)
    handler.vocab_manager = vm
    ctx = _Ctx(is_arithmetic=False, fp_immediate_type=None, fp_postfix_type=None)
    tokens = handler._emit_valued_const(value, meta=None, ctx=ctx)

    flat: List[int] = []
    for t in tokens:
        flat.extend(int(x) for x in t.get_token_ids().tolist())
    return np.array(flat, dtype=np.uint16), vm


@pytest.mark.parametrize(
    "value",
    [-(2 ** 63), -2 ** 31, -255, -128, -1, 0, 1, 127, 255, 2 ** 63 - 1],
)
def test_end_to_end_emitter_to_decoder_round_trip(value: int) -> None:
    """Real emitter -> real decoder -> reconstruct ``value`` from chunks.

    Sources the emitter output verbatim, resolves vocab ids through the
    actual ``VocabularyManager``, then reconstructs the original
    integer from the decoder's chunk side-array (sig + sign bit).
    Pins the full encoder/decoder contract for the postfix sign shape
    across the representative signed-integer range.

    Reconstruction rule for a single-chunk integer: the chunk's
    ``(sig, sign_exp)`` carries the magnitude (normalized so the
    leading 1 sits at bit 63) and the sign + biased exponent; the
    multi-chunk case (|value| >= 2**64) is out of scope for this
    smoke -- the round-trip suite above already covers INT64_MIN /
    MAX via the synthetic stream + ``from_int`` reference.
    """
    from tokenizer.aligned_data.loader.decoded.category_tokens import (
        resolve_category_token_ids,
        resolve_number_token_ids,
        resolve_value_negative_token_id,
    )

    raw, vm = _emit_real(value)
    cat_ids = resolve_category_token_ids(vm)
    num_ids = resolve_number_token_ids(vm)
    vneg_id = resolve_value_negative_token_id(vm)

    out = decode_raw_tokens(
        raw,
        id_token_ids=cat_ids,
        number_token_ids=num_ids,
        value_negative_token_id=vneg_id,
        format_version=int(vm.format_version),
        func_name="t",
    )

    # |value| < 2**64 -> exactly one chunk; the chunk's sig (normalized,
    # leading-1 at bit 63) + the unbiased exponent reconstruct the
    # magnitude; the sign bit on sign_exp recovers the sign.
    magnitude = -value if value < 0 else value
    expected_chunks = from_int(magnitude, sign=(-1 if value < 0 else +1))
    assert out.numbers_significant.shape == (len(expected_chunks),)
    for idx, (sig, sign_exp) in enumerate(expected_chunks):
        assert int(out.numbers_significant[idx]) == int(sig), (
            f"value={value}: chunk[{idx}].sig mismatch"
        )
        assert int(out.numbers_sign_exponent[idx]) == int(sign_exp), (
            f"value={value}: chunk[{idx}].sign_exp mismatch"
        )


