"""RED tests pinning the FP-postfix valued-token contract.

The bare value-less ``floatXX`` postfix form is FORBIDDEN: a ``floatXX``
is a value-token and ALWAYS carries its ``width_bytes`` of inline IEEE
digit bytes. The encoder dereferences the load target to capture the real
constant; when the value is unobtainable it emits the value-less
``float_annotation`` modifier instead.

Covers (encoder side only — the decoder/aligned_data zone is owned by a
separate rewrite and is intentionally NOT exercised here):

  * ro_data_ptr float32 load whose .rodata constant is a known IEEE value
    -> ``[ro_data_ptr(id), float32 + 4 digit bytes == that constant]``,
    NO ``float_annotation``.
  * FP load where ``read_bytes`` returns ``None`` -> tokens end with the
    value-less ``float_annotation`` marker, NO ``floatXX`` inline bytes.
  * Value-less ``floatXX`` is un-constructible: ``Float32(None)`` raises;
    ``_from_token_ids([single_float_type_id])`` raises.
  * ``_register_v2_canonical_blocks`` pins 257..263 to the six floats and
    leaves ``len(id_to_token) == 272``; ``float_annotation`` id ``>= 272``.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from tokenizer.constant_handler.core import ConstantHandler
from tokenizer.constant_handler.ctx import _Ctx
from tokenizer.disasm.types import FpType
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType


# --------------------------------------------------------------------------
# Stubs — a fake MetadataLookup whose read_bytes is wired per-test. No real
# Ghidra/angr required; the contract under test lives entirely in the
# encoder's read-then-emit threading.
# --------------------------------------------------------------------------


class _StubLookup:
    """Minimal ``MetadataLookup`` stub. ``read_bytes`` returns the canned
    payload (or ``None`` for the unobtainable case); ``lookup`` is unused by
    the emitter paths these tests drive directly."""

    def __init__(self, payload: Optional[bytes]):
        self._payload = payload

    def read_bytes(self, addr: int, n: int) -> Optional[bytes]:
        if self._payload is None:
            return None
        assert len(self._payload) == n, (
            f"stub payload width {len(self._payload)} != requested {n}"
        )
        return self._payload

    def lookup(self, addr: int):  # pragma: no cover - not used in these tests
        raise NotImplementedError


@pytest.fixture
def vm() -> VocabularyManager:
    return VocabularyManager(platform=None, format_version=2)


@pytest.fixture
def handler(vm: VocabularyManager) -> ConstantHandler:
    """Minimal handler: the FP-postfix emitter + read helper touch only
    ``vocab_manager`` (and the lookup passed in), so the rest of the init
    surface is unused."""
    h = ConstantHandler.__new__(ConstantHandler)
    h.vocab_manager = vm
    return h


def _ids(tokens) -> List[int]:
    out: List[int] = []
    for t in tokens:
        out.extend(int(x) for x in t.get_token_ids().tolist())
    return out


# IEEE-754 single-precision 3.14f = 0x4048F5C3 (big-endian bytes).
_PI_F32 = bytes([0x40, 0x48, 0xF5, 0xC3])


# --------------------------------------------------------------------------
# Deref succeeds -> valued floatXX, no float_annotation
# --------------------------------------------------------------------------


def test_read_helper_derives_width_and_reads(handler, vm):
    """``read_fp_postfix_bytes`` derives W=4 for FLOAT32 and reads exactly
    those bytes through the lookup."""
    lookup = _StubLookup(_PI_F32)
    got = handler.read_fp_postfix_bytes(lookup, 0x1000, FpType.FLOAT32)
    assert got == _PI_F32


def test_read_helper_returns_none_when_no_postfix(handler, vm):
    """No FP postfix -> no read attempt, ``None`` returned."""
    lookup = _StubLookup(_PI_F32)
    assert handler.read_fp_postfix_bytes(lookup, 0x1000, None) is None


def test_ro_data_ptr_fp_load_emits_valued_float32(handler, vm):
    """A ro_data_ptr whose FP load deref'd to a known IEEE value emits the
    ptr token followed by a VALUED float32 (4 digit bytes == the constant),
    with NO float_annotation."""
    bits = int.from_bytes(_PI_F32, "big")
    expected_float = vm.Float32(bits).get_token_ids().tolist()
    # The valued float32 is exactly [type_id, b0, b1, b2, b3].
    assert len(expected_float) == 5
    assert expected_float[1:] == list(_PI_F32)

    ctx = _Ctx(
        is_arithmetic=False,
        fp_immediate_type=None,
        fp_postfix_type=FpType.FLOAT32,
        fp_postfix_bytes=_PI_F32,
    )
    got = handler._postfix_fp_annotation(ctx)
    assert _ids(got) == expected_float
    # No float_annotation token present.
    fa_id = vm.Float_Annotation().get_token_ids().tolist()[0]
    assert fa_id not in _ids(got)


# --------------------------------------------------------------------------
# Deref unobtainable -> float_annotation, no floatXX
# --------------------------------------------------------------------------


def test_unreadable_fp_load_emits_float_annotation(handler, vm):
    """When the deref read failed (bytes None) the postfix degrades to a
    single value-less float_annotation id — NO floatXX, no inline bytes."""
    ctx = _Ctx(
        is_arithmetic=False,
        fp_immediate_type=None,
        fp_postfix_type=FpType.FLOAT32,
        fp_postfix_bytes=None,
    )
    got = handler._postfix_fp_annotation(ctx)
    ids = _ids(got)
    assert len(ids) == 1
    fa_id = vm.Float_Annotation().get_token_ids().tolist()[0]
    assert ids == [fa_id]
    assert vm.id_to_token_type[ids[0]] == TokenType.FLOAT_ANNOTATION
    # No float32 type id leaked into the stream.
    f32_id = vm.Float32(0).get_token_ids().tolist()[0]
    assert f32_id not in ids


def test_read_none_path_threads_to_annotation(handler, vm):
    """End-to-end of the read helper + emitter: a lookup whose read_bytes
    returns None yields the float_annotation marker, not a floatXX."""
    lookup = _StubLookup(None)
    fp_bytes = handler.read_fp_postfix_bytes(lookup, 0x2000, FpType.FLOAT32)
    assert fp_bytes is None
    ctx = _Ctx(
        is_arithmetic=False,
        fp_immediate_type=None,
        fp_postfix_type=FpType.FLOAT32,
        fp_postfix_bytes=fp_bytes,
    )
    [fa_id] = _ids(handler._postfix_fp_annotation(ctx))
    assert vm.id_to_token_type[fa_id] == TokenType.FLOAT_ANNOTATION


# --------------------------------------------------------------------------
# Value-less floatXX is un-constructible
# --------------------------------------------------------------------------


def test_value_less_float_raises_on_construction(vm):
    """``Float32(None)`` must be impossible — bits is required."""
    with pytest.raises((TypeError, AssertionError)):
        vm.Float32(None)


def test_float_from_single_type_id_raises(vm):
    """``_from_token_ids`` with only the type id (no inline payload) must
    raise — the value-less wire shape is gone."""
    type_id = vm.Float32(0).get_token_ids().tolist()[0]
    with pytest.raises(ValueError):
        vm.Float32._from_token_ids([type_id])


# --------------------------------------------------------------------------
# Canonical-block registration invariant
# --------------------------------------------------------------------------


def test_canonical_blocks_pin_six_floats_and_len_272():
    """257..263 pin the six floats; len(id_to_token) == 272 after eager
    registration; float_annotation lands at >= 272."""
    vm = VocabularyManager(platform=None, format_version=1)
    vm._register_v2_canonical_blocks()

    assert vm.get_token_id("float16") == 258
    assert vm.get_token_id("bfloat16") == 259
    assert vm.get_token_id("float32") == 260
    assert vm.get_token_id("float64") == 261
    assert vm.get_token_id("float80") == 262
    assert vm.get_token_id("float128") == 263
    assert len(vm.id_to_token) == 272

    [fa_id] = vm.Float_Annotation().get_token_ids().tolist()
    assert fa_id >= 272
