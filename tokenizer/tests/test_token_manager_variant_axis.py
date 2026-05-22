"""Tests for the Variant_Axis Inner class + TokenType.VARIANT_AXIS wiring.

Covers:
  * dispatch via `get_token_class_for_type(TokenType.VARIANT_AXIS)`
  * registration through the dispatched class (ID assignment + type lookup)
  * `iter_representative_tokens` yields one wrapper per registered
    Variant_Axis token under format_version=1 (unified vocab)
  * format_version=1 reserved-digit layout matches format_version=2
  * v2 inline-digit Inner classes (e.g. ValuedConstV2, BlockV2) remain
    instantiable on a v1 VM (the unified vocab is a strict superset of
    the per-binary v2 wire encoding)
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType, VariantAxisToken


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
# First vocab id that caller-driven registrations can claim on a v1/v2 VM:
# the digit range occupies 0.._V2_RESERVED_DIGIT_COUNT-1, the
# `value_negative` postfix sign marker is pinned at
# `_V2_VALUE_NEGATIVE_TOKEN_ID` (= _V2_RESERVED_DIGIT_COUNT), so the
# first caller-assignable slot starts one past the marker — i.e.
# `_V2_RESERVED_TOKEN_COUNT`.
_V1_FIRST_CALLER_ID = VocabularyManager._V2_RESERVED_TOKEN_COUNT


def _make_v1_vm() -> VocabularyManager:
    """Construct an empty unified-style v1 VocabularyManager (platform=None)."""
    return VocabularyManager(platform=None, format_version=1)


def _make_v2_vm() -> VocabularyManager:
    """Construct an empty unified-style v2 VocabularyManager (platform=None)."""
    return VocabularyManager(platform=None, format_version=2)


# --------------------------------------------------------------------------
# Dispatch + registration
# --------------------------------------------------------------------------


def test_get_token_class_for_type_returns_variant_axis_class():
    """`get_token_class_for_type(VARIANT_AXIS)` must return the per-VM
    Variant_Axis Inner class (the same class accessible as `vm.Variant_Axis`)."""
    vm = _make_v1_vm()
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)
    assert cls is vm.Variant_Axis
    # The class must declare the right token_type at the class level.
    assert cls.token_type == TokenType.VARIANT_AXIS
    # And it must satisfy the protocol.
    assert issubclass(cls, VariantAxisToken)


def test_variant_axis_registration_assigns_id_and_token_type():
    """Registering a Variant_Axis token (via dispatch) populates the
    vocab dict, exposes the assigned ID, and tags it with VARIANT_AXIS."""
    vm = _make_v1_vm()
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)

    token = cls("arch:x64")

    # ID landed at the first caller-assignable slot (post digit range +
    # past the eagerly-pinned `value_negative` marker at id 256).
    assert token._token_id == _V1_FIRST_CALLER_ID
    # Round-trip via the public id_to_token / token_to_id surface.
    assert vm.get_token_id("arch:x64") == _V1_FIRST_CALLER_ID
    assert vm.get_token_str(_V1_FIRST_CALLER_ID) == "arch:x64"
    # The id_to_token_type cache reflects VARIANT_AXIS.
    assert vm.id_to_token_type[_V1_FIRST_CALLER_ID] == TokenType.VARIANT_AXIS


def test_variant_axis_registration_idempotent_per_string():
    """Registering the same axis string twice returns the same vocab id
    (matches `_private_add_token`'s idempotency contract)."""
    vm = _make_v1_vm()
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)

    t1 = cls("comp:gcc")
    t2 = cls("comp:gcc")

    assert t1._token_id == t2._token_id
    # Baseline = reserved-digit range (256) + value_negative (1) + the
    # single registered variant axis (1).
    assert vm.size == _V1_FIRST_CALLER_ID + 1


def test_variant_axis_distinct_strings_get_distinct_ids():
    """Two different prefixed strings register as two distinct vocab ids,
    contiguous from the first non-reserved slot (no gaps under the v1
    unified vocab reserved layout)."""
    vm = _make_v1_vm()
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)

    arch_tok = cls("arch:x64")
    comp_tok = cls("comp:gcc")
    cver_tok = cls("cver:gcc:13.2.0")
    opt_tok = cls("opt:O2")
    meta_tok = cls("hardening:full")

    ids = [arch_tok._token_id, comp_tok._token_id, cver_tok._token_id,
           opt_tok._token_id, meta_tok._token_id]
    assert ids == [_V1_FIRST_CALLER_ID + i for i in range(5)]
    # All five are tagged VARIANT_AXIS in the type cache.
    for i in range(5):
        assert vm.id_to_token_type[_V1_FIRST_CALLER_ID + i] == TokenType.VARIANT_AXIS


def test_variant_axis_round_trip_from_token_ids():
    """`_from_token_ids([id])` reconstructs an instance carrying the same
    string + id as the original registration."""
    vm = _make_v1_vm()
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)

    original = cls("cver:gcc:13.2.0")
    rebuilt = cls._from_token_ids([original._token_id])

    assert rebuilt.token == "cver:gcc:13.2.0"
    assert rebuilt._token_id == original._token_id
    assert np.array_equal(rebuilt.get_token_ids(), np.array([original._token_id], dtype=np.int_))
    assert rebuilt.token_type == TokenType.VARIANT_AXIS
    assert original == rebuilt


def test_variant_axis_get_token_ids_shape():
    """Wire form is exactly one uint id; the array contract for
    consumers (the unifier, the validator) must be a length-1 ndarray."""
    vm = _make_v1_vm()
    token = vm.Variant_Axis("opt:O2")

    ids = token.get_token_ids()
    assert isinstance(ids, np.ndarray)
    assert ids.shape == (1,)
    assert ids[0] == token._token_id


def test_variant_axis_rejects_whitespace():
    """Spaces in the prefixed string would break vocab.csv round-trip
    (whitespace is the field separator). Mirror PlatformToken's same
    guard."""
    vm = _make_v1_vm()
    with pytest.raises(ValueError):
        vm.Variant_Axis("arch: x64")


def test_variant_axis_from_token_ids_rejects_wrong_arity():
    """Single-id wire form: 0 ids and >1 ids must both raise."""
    vm = _make_v1_vm()
    with pytest.raises(ValueError):
        vm.Variant_Axis._from_token_ids([])
    with pytest.raises(ValueError):
        vm.Variant_Axis._from_token_ids([_V1_FIRST_CALLER_ID, _V1_FIRST_CALLER_ID + 1])


# --------------------------------------------------------------------------
# iter_representative_tokens
# --------------------------------------------------------------------------


def test_iter_representative_tokens_yields_one_wrapper_per_variant_axis():
    """Under v1 (unified vocab), each registered Variant_Axis vocab id must appear as
    one wrapper in `iter_representative_tokens` — matching the
    one-wrapper-per-id contract for opaque-string families."""
    vm = _make_v1_vm()
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)

    strings = ["arch:x64", "comp:gcc", "cver:gcc:13.2.0", "opt:O2",
               "hardening:full", "sanitizer:address"]
    for s in strings:
        cls(s)

    reps = list(vm.iter_representative_tokens())
    # Every yielded representative for VARIANT_AXIS must surface the
    # original string round-tripped — exactly one per registered id.
    variant_reps = [t for t in reps if t.token_type == TokenType.VARIANT_AXIS]
    assert len(variant_reps) == len(strings)
    assert sorted(t.token for t in variant_reps) == sorted(strings)


def test_iter_representative_tokens_skips_reserved_digits_under_v1():
    """The 256 reserved digit slots must NOT appear as representatives
    (their token_type is UNRESOLVED — the dispatch table would fail).
    On a freshly-constructed v1 VM the only registered token is the
    eagerly-pinned `value_negative` marker, so iter_representative_tokens
    yields exactly that one representative."""
    vm = _make_v1_vm()
    reps = list(vm.iter_representative_tokens())
    assert len(reps) == 1
    assert reps[0].token_type == TokenType.VALUE_NEGATIVE


# --------------------------------------------------------------------------
# format_version=1 reserved-digit layout parity with v2
# --------------------------------------------------------------------------


def test_v1_vm_reserves_same_digit_slots_as_v2():
    """v1 unified vocab = v2 wire encoding + variant tokens (additive).
    The reserved-digit prelude (256 `digit_<HH>` placeholders typed
    UNRESOLVED) plus the eagerly-pinned `value_negative` marker must be
    byte-identical between v2 and v1 VMs."""
    v2_vm = _make_v2_vm()
    v1_vm = _make_v1_vm()

    # Baseline size = digit range (256) + the value_negative marker (1).
    assert v2_vm.size == _V1_FIRST_CALLER_ID
    assert v1_vm.size == _V1_FIRST_CALLER_ID
    # Placeholder names match across the reserved digit range.
    assert v2_vm.id_to_token[:_V2_RESERVED_DIGIT_COUNT] == \
        v1_vm.id_to_token[:_V2_RESERVED_DIGIT_COUNT]
    # And the post-digit pinned slot matches: both VMs register
    # `value_negative` at the same id.
    assert v2_vm.id_to_token[_V2_RESERVED_DIGIT_COUNT] == \
        v1_vm.id_to_token[_V2_RESERVED_DIGIT_COUNT] == "value_negative"
    # Token-type tags match (all UNRESOLVED across the digit range in both).
    assert np.array_equal(
        np.asarray(v2_vm.id_to_token_type[:_V2_RESERVED_DIGIT_COUNT]),
        np.asarray(v1_vm.id_to_token_type[:_V2_RESERVED_DIGIT_COUNT]),
    )
    # Both VMs reserve the digit names from the token_to_id lookup so
    # that the digit ids are addressed purely by numeric position.
    for i in range(_V2_RESERVED_DIGIT_COUNT):
        assert v1_vm.get_token_id(f"digit_{i:02X}") == -1


def test_v1_vm_first_caller_token_lands_one_past_value_negative():
    """First caller-driven token on a v1 VM gets id 257 — one past the
    eagerly-pinned `value_negative` marker — so the variant block starts
    at the documented post-marker boundary, same as v2."""
    v1_vm = _make_v1_vm()
    tok = v1_vm.Variant_Axis("arch:x64")
    assert tok._token_id == _V1_FIRST_CALLER_ID


def test_v1_vm_rejects_reserved_digit_name():
    """The `digit_<HH>` collision guard applies under v1 too (same
    reserved-digit-protocol invariant as v2)."""
    v1_vm = _make_v1_vm()
    with pytest.raises(AssertionError):
        v1_vm.Variant_Axis("digit_00")


def test_v1_vm_valued_const_dispatch_uses_v2_form():
    """The format-aware `ValuedConst` / `BlockId` factories must dispatch
    to the v2 inline-digit Inner classes under v1 (the unified vocab
    shares the v2 wire encoding for instruction-stream tokens)."""
    v1_vm = _make_v1_vm()
    vc = v1_vm.ValuedConst(0)
    bk = v1_vm.BlockId(0)
    assert vc.token_type == TokenType.VALUED_CONST_V2
    assert bk.token_type == TokenType.BLOCK_V2


def test_v1_vm_accepts_v2_inner_class_instantiation():
    """v1 unified vocab = v2 wire encoding + variant tokens means v2
    inline-digit category tokens must remain instantiable on a v1 VM
    unchanged (otherwise the unifier couldn't register
    instruction-stream tokens through `iter_representative_tokens`
    against a v1 unified VM)."""
    v1_vm = _make_v1_vm()
    # One representative from each v2 family (identity, valued_const_v2,
    # float, modifier) must construct without raising the
    # `format_version in (1, 2)` assertion.
    v1_vm.Local_Func(0)
    v1_vm.Valued_Const_V2(42)
    v1_vm.Float32(None)
    v1_vm.Thread_Local()


def test_variant_axis_and_v2_token_coexist_with_distinct_ids():
    """Mixing a Variant_Axis registration with v2 inline-digit token
    registrations (which is what the unifier will do in Batch 3) must
    produce distinct vocab ids and correctly-tagged type cache entries."""
    v1_vm = _make_v1_vm()

    # Register two variant axes first (Batch 3's Pass 1 order).
    va1 = v1_vm.Variant_Axis("arch:x64")
    va2 = v1_vm.Variant_Axis("comp:gcc")
    # Then a v2 identity token (Batch 3's Pass 2 representative).
    lf = v1_vm.Local_Func(0)
    # And a v2 modifier.
    tl = v1_vm.Thread_Local()

    ids = {va1._token_id, va2._token_id, lf._type_token_id, tl._type_token_id}
    assert len(ids) == 4
    # All sit above the reserved-digit boundary.
    assert all(i >= _V2_RESERVED_DIGIT_COUNT for i in ids)
    # Type cache distinguishes them.
    assert v1_vm.id_to_token_type[va1._token_id] == TokenType.VARIANT_AXIS
    assert v1_vm.id_to_token_type[va2._token_id] == TokenType.VARIANT_AXIS
    assert v1_vm.id_to_token_type[lf._type_token_id] == TokenType.LOCAL_FUNC
    assert v1_vm.id_to_token_type[tl._type_token_id] == TokenType.THREAD_LOCAL
