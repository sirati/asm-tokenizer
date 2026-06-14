"""Unit tests pinning the v2 valued_const emitter's sign-handling contract.

The v2 emitter (``_V2EmittersMixin._emit_valued_const``) is the sole
owner of sign decomposition for ``valued_const_v2`` tokens. Negative
values become ``[Valued_Const_V2(|v|), Value_Negative()]`` (plus an
optional FP postfix); non-negative values become ``[Valued_Const_V2(v)]``.

These tests construct a minimal handler (no ``TokenResolver`` /
``block_ranges``) because the fallback emitter only
touches ``self.vocab_manager`` and the (inherited) ``_postfix_fp_annotation``
helper — no resolver state, no block ranges. Bypassing
``ConstantHandler.__init__`` via ``__new__`` keeps the fixture surface
narrow to the API actually under test.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from tokenizer.constant_handler.core import ConstantHandler
from tokenizer.constant_handler.ctx import _Ctx
from tokenizer.disasm.types import FpType
from tokenizer.token_manager import VocabularyManager


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def vm() -> VocabularyManager:
    """A fresh v2 VocabularyManager (per-binary CSV; inline-digit wire
    encoding). Equivalent for these tests; v1 unified would behave the
    same because the emitter only touches the Inner-class factories,
    which share their bodies across v1 and v2."""
    return VocabularyManager(platform=None, format_version=2)


@pytest.fixture
def handler(vm: VocabularyManager) -> ConstantHandler:
    """A minimal ``ConstantHandler`` with only ``vocab_manager`` wired.
    ``_emit_valued_const`` + ``_postfix_fp_annotation`` access only that
    attribute, so the rest of the init surface is unused by the
    fallback path."""
    h = ConstantHandler.__new__(ConstantHandler)
    h.vocab_manager = vm
    return h


def _ctx(
    fp_postfix: Optional[FpType] = None,
    fp_postfix_bytes: Optional[bytes] = None,
) -> _Ctx:
    return _Ctx(
        is_arithmetic=False,
        fp_immediate_type=None,
        fp_postfix_type=fp_postfix,
        fp_postfix_bytes=fp_postfix_bytes,
    )


def _ids(tokens) -> List[int]:
    """Flatten a token list into a concatenated id stream."""
    out: List[int] = []
    for t in tokens:
        out.extend(int(x) for x in t.get_token_ids().tolist())
    return out


# --------------------------------------------------------------------------
# Sign-handling contract: postfix value_negative AFTER magnitude bytes
# --------------------------------------------------------------------------


def test_negative_value_emits_magnitude_then_value_negative(handler, vm):
    """``-255`` -> ``[Valued_Const_V2(255), Value_Negative()]`` (no FP)."""
    got = handler._emit_valued_const(-255, meta=None, ctx=_ctx())
    expected = [
        *vm.Valued_Const_V2(255).get_token_ids().tolist(),
        *vm.Value_Negative().get_token_ids().tolist(),
    ]
    assert _ids(got) == expected
    # Postfix marker is the LAST id (255 magnitude byte 0xff, then 256).
    assert _ids(got)[-1] == 256


def test_positive_value_emits_no_value_negative(handler, vm):
    """``255`` -> ``[Valued_Const_V2(255)]``; no postfix marker."""
    got = handler._emit_valued_const(255, meta=None, ctx=_ctx())
    expected = vm.Valued_Const_V2(255).get_token_ids().tolist()
    assert _ids(got) == expected
    assert 256 not in _ids(got)


def test_minus_one_emits_one_magnitude_byte_then_value_negative(handler, vm):
    """``-1`` -> ``[Valued_Const_V2(1), Value_Negative()]``. Smallest
    magnitude (one digit byte) confirms the postfix slots in after a
    one-byte magnitude run."""
    got = handler._emit_valued_const(-1, meta=None, ctx=_ctx())
    expected = [
        *vm.Valued_Const_V2(1).get_token_ids().tolist(),
        *vm.Value_Negative().get_token_ids().tolist(),
    ]
    assert _ids(got) == expected


def test_zero_is_not_negative(handler, vm):
    """``0`` -> ``[Valued_Const_V2(0)]``; zero must NOT carry a postfix
    ``value_negative`` (the discriminant is ``value < 0``, not ``<= 0``)."""
    got = handler._emit_valued_const(0, meta=None, ctx=_ctx())
    expected = vm.Valued_Const_V2(0).get_token_ids().tolist()
    assert _ids(got) == expected
    assert 256 not in _ids(got)


def test_int64_min_packs_as_unsigned_magnitude_then_value_negative(handler, vm):
    """``INT64_MIN`` (-2**63) -> ``[Valued_Const_V2(2**63), Value_Negative()]``.

    Under Python's arbitrary-precision ints, ``-(-2**63) == 2**63`` (no
    overflow), and the magnitude packs minimum-width as 8 bytes
    (``0x80 00 00 00 00 00 00 00``) -- the high bit is set on the first
    byte but the encoding stays unsigned because ``_v2_int_to_minimum_bytes``
    uses ``bit_length()``-derived width, not a signed two's-complement
    rule. This test pins that exact wire shape so future width-fudging
    regressions surface here.
    """
    int64_min = -(2 ** 63)
    got = handler._emit_valued_const(int64_min, meta=None, ctx=_ctx())
    expected_magnitude = vm.Valued_Const_V2(2 ** 63).get_token_ids().tolist()
    expected = [
        *expected_magnitude,
        *vm.Value_Negative().get_token_ids().tolist(),
    ]
    assert _ids(got) == expected
    # 8 magnitude bytes + 1 type id + 1 value_negative = 10 ids.
    assert len(_ids(got)) == 10
    # Magnitude packs as 0x80 00 00 00 00 00 00 00 after the type id.
    assert _ids(got)[1:9] == [0x80, 0, 0, 0, 0, 0, 0, 0]
    # Postfix marker is the last id.
    assert _ids(got)[-1] == 256


# --------------------------------------------------------------------------
# Order with FP postfix: magnitude bytes, then value_negative, then floatXX
# --------------------------------------------------------------------------


def test_negative_with_fp_postfix_orders_value_negative_before_float(handler, vm):
    """``-255`` with ``fp_postfix_type=FLOAT32`` + dereferenced bytes ->
    ``[Valued_Const_V2(255), Value_Negative(), Float32(<bits>)]``.

    Order is load-bearing: the v2 decoder consumes ``value_negative`` as
    a postfix sign marker on the preceding ``valued_const_v2`` token; if
    the FP postfix came first the decoder would attach FP-typedness to
    the magnitude and then see a stray ``value_negative``. Pinning the
    order here catches an accidental swap of the two postfix appends.
    The FP postfix is now a VALUED ``floatXX`` carrying the dereferenced
    image bytes (big-endian).
    """
    fp_bytes = bytes([0x3F, 0x80, 0x00, 0x00])  # IEEE-754 1.0f
    got = handler._emit_valued_const(
        -255, meta=None, ctx=_ctx(fp_postfix=FpType.FLOAT32, fp_postfix_bytes=fp_bytes)
    )
    expected = [
        *vm.Valued_Const_V2(255).get_token_ids().tolist(),
        *vm.Value_Negative().get_token_ids().tolist(),
        *vm.Float32(int.from_bytes(fp_bytes, "big")).get_token_ids().tolist(),
    ]
    assert _ids(got) == expected


def test_positive_with_fp_postfix_emits_only_magnitude_then_float(handler, vm):
    """Positive value with FP postfix + dereferenced bytes: no
    ``value_negative`` between magnitude and the valued ``floatXX``.
    Sanity check that the FP-postfix path is unchanged for non-negative
    values."""
    fp_bytes = bytes([0x3F, 0x80, 0x00, 0x00])  # IEEE-754 1.0f
    got = handler._emit_valued_const(
        255, meta=None, ctx=_ctx(fp_postfix=FpType.FLOAT32, fp_postfix_bytes=fp_bytes)
    )
    expected = [
        *vm.Valued_Const_V2(255).get_token_ids().tolist(),
        *vm.Float32(int.from_bytes(fp_bytes, "big")).get_token_ids().tolist(),
    ]
    assert _ids(got) == expected
    assert 256 not in _ids(got)
