"""Tests for the renumbered format_version set-membership in VocabularyManager.

After memoized-booping-wren.md's renumbering, format_version=1 is the
new unified-vocab format that shares the v2 (per-binary CSV)
inline-digit wire encoding. format_version=3 used to be the unified
vocab; it now has no special meaning to the inline-digit encoding —
the digit-reservation branch and the V2 Inner-class dispatch only fire
for `format_version in (1, 2)`. This file pins that renumbering on the
sites that gate the inline-digit encoding.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_V2_RESERVED_TOKEN_COUNT = VocabularyManager._V2_RESERVED_TOKEN_COUNT


# --------------------------------------------------------------------------
# Constructor: reserved-digit pre-population only fires for v1 and v2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fv", [1, 2])
def test_constructor_reserves_digit_slots_for_inline_digit_versions(fv):
    """v1 (unified) and v2 (per-binary CSV) both use the inline-digit
    wire encoding and therefore pre-populate IDs 0..255 with
    `digit_<HH>` placeholders tagged UNRESOLVED. The constructor also
    eagerly registers the `value_negative` postfix sign marker
    immediately after the digit range, pinning it at id 256; the
    detailed invariant for that pinning lives in
    ``test_value_negative_token_id_pinned_at_first_post_digit_slot``.
    """
    vm = VocabularyManager(platform=None, format_version=fv)
    # Reserved digit range fills ids 0..255 unchanged.
    assert all(
        vm.id_to_token[i] == f"digit_{i:02X}"
        for i in range(_V2_RESERVED_DIGIT_COUNT)
    )
    type_cache = np.asarray(vm.id_to_token_type[:_V2_RESERVED_DIGIT_COUNT])
    assert np.all(type_cache == TokenType.UNRESOLVED)
    # Post-digit baseline: the constructor adds exactly one entry — the
    # `value_negative` marker at id `_V2_RESERVED_DIGIT_COUNT`. Total
    # constructor-pinned slot count is `_V2_RESERVED_TOKEN_COUNT` (= 257).
    assert len(vm.id_to_token) == _V2_RESERVED_TOKEN_COUNT
    assert vm.id_to_token[_V2_RESERVED_DIGIT_COUNT] == "value_negative"


def test_constructor_with_format_version_3_does_not_reserve_digits():
    """v3 is no longer a reserved-digit format. With the renumbering,
    only `format_version in (1, 2)` triggers the digit pre-population;
    v3 (and any other value) constructs an empty vocab."""
    vm = VocabularyManager(platform=None, format_version=3)
    assert len(vm.id_to_token) == 0


# --------------------------------------------------------------------------
# Factory dispatch: ValuedConst / BlockId pick V2 wire form only for v1, v2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fv", [1, 2])
def test_valued_const_factory_dispatches_to_v2_form_under_inline_digit(fv):
    """Under v1 and v2 the format-aware factory must produce the
    inline-digit V2 wire form."""
    vm = VocabularyManager(platform=None, format_version=fv)
    vc = vm.ValuedConst(7)
    assert type(vc).__name__ == "ValuedConstV2Inner"
    assert vc.token_type == TokenType.VALUED_CONST_V2


@pytest.mark.parametrize("fv", [1, 2])
def test_block_id_factory_dispatches_to_v2_form_under_inline_digit(fv):
    """Under v1 and v2 the format-aware factory must produce the
    inline-digit V2 wire form."""
    vm = VocabularyManager(platform=None, format_version=fv)
    bk = vm.BlockId(0)
    assert type(bk).__name__ == "BlockV2Inner"
    assert bk.token_type == TokenType.BLOCK_V2


def test_valued_const_factory_falls_back_to_legacy_under_v3():
    """v3 is outside the inline-digit set; the factory must dispatch to
    the legacy `Valued_Const` (non-V2) class."""
    vm = VocabularyManager(platform=None, format_version=3)
    vc = vm.ValuedConst(7)
    assert type(vc).__name__ == "ValuedConstTokenInner"
    assert vc.token_type == TokenType.VALUED_CONST


def test_block_id_factory_falls_back_to_legacy_under_v3():
    """v3 is outside the inline-digit set; the factory must dispatch to
    the legacy `Block` (non-V2) class."""
    vm = VocabularyManager(platform=None, format_version=3)
    bk = vm.BlockId(0)
    assert type(bk).__name__ == "BlockInner"
    assert bk.token_type == TokenType.BLOCK


# --------------------------------------------------------------------------
# Inner-class assertions accept (1, 2); reject everything else
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fv", [1, 2])
def test_v2_inner_class_instantiates_on_inline_digit_vm(fv):
    """The V2 Inner classes (Valued_Const_V2 + Block_V2 + Local_Func +
    Thread_Local) must construct without error on v1 and v2 VMs."""
    vm = VocabularyManager(platform=None, format_version=fv)
    # Pick one representative from each family.
    vm.Valued_Const_V2(42)
    vm.Block_V2(3)
    vm.Local_Func(0)
    vm.Thread_Local()


def test_v2_inner_class_rejects_format_version_outside_inline_digit_set():
    """Instantiating a V2 Inner class on a VM whose format_version is
    outside `(1, 2)` must raise AssertionError pointing at the renumbered
    acceptance set."""
    vm = VocabularyManager(platform=None, format_version=3)
    with pytest.raises(AssertionError) as exc_info:
        vm.Valued_Const_V2(0)
    assert "format_version=1 (unified) or =2 (per-binary CSV)" in str(exc_info.value)
    # Mirror check for an identity-class Inner.
    with pytest.raises(AssertionError):
        vm.Local_Func(0)
    # And for a float-class Inner.
    with pytest.raises(AssertionError):
        vm.Float32(0)
    # And for a modifier-class Inner.
    with pytest.raises(AssertionError):
        vm.Thread_Local()


# --------------------------------------------------------------------------
# Digit-slot collision guard: applies under v1 + v2; not under v3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fv", [1, 2])
def test_private_add_token_rejects_digit_slot_name_under_inline_digit_vm(fv):
    """The `digit_<HH>` collision guard fires when the active format
    pre-populates the reserved slots (v1 + v2)."""
    vm = VocabularyManager(platform=None, format_version=fv)
    cls = vm.get_token_class_for_type(TokenType.VARIANT_AXIS)
    with pytest.raises(AssertionError):
        vm._private_add_token("digit_00", cls)


# --------------------------------------------------------------------------
# value_negative postfix sign marker pinned at id 256 on every v1/v2 VM
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fv", [1, 2])
def test_value_negative_token_id_pinned_at_first_post_digit_slot(fv):
    """The `value_negative` postfix sign marker is registered eagerly in
    the constructor immediately after the reserved-digit pre-population,
    so its vocab id is pinned to `_V2_VALUE_NEGATIVE_TOKEN_ID` (= 256 =
    the first slot after the digit range). The invariant is published as
    the `value_negative_token_id` attribute and exercised through the
    `Value_Negative()` factory."""
    vm = VocabularyManager(platform=None, format_version=fv)
    assert vm.value_negative_token_id == VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID == 256
    assert vm.Value_Negative().get_token_ids().tolist() == [256]
    # The factory returns instances of an Inner class whose vocab type
    # tag is the dedicated VALUE_NEGATIVE; the dispatch table can resolve
    # the class back from the type tag.
    assert vm.Value_Negative().token_type == TokenType.VALUE_NEGATIVE
    assert vm.get_token_class_for_type(TokenType.VALUE_NEGATIVE) is vm.Value_Negative


@pytest.mark.parametrize("fv", [1, 2])
def test_value_negative_factory_is_idempotent(fv):
    """Repeated `Value_Negative()` calls hit the `_private_add_token`
    short-circuit and reuse the pinned id; no second registration, no
    drift of the cached `value_negative_token_id`."""
    vm = VocabularyManager(platform=None, format_version=fv)
    pre_size = vm.size
    [first_id] = vm.Value_Negative().get_token_ids().tolist()
    [second_id] = vm.Value_Negative().get_token_ids().tolist()
    assert first_id == second_id == 256
    assert vm.size == pre_size  # no new vocab entries
    assert vm.value_negative_token_id == 256


def test_value_negative_factory_rejected_on_non_inline_digit_vm():
    """On vocabs outside the inline-digit set (v3 etc.) the constructor
    does not pin the marker, and instantiating the Inner class raises
    the standard v2-format-version assertion."""
    vm = VocabularyManager(platform=None, format_version=3)
    assert vm.value_negative_token_id is None
    with pytest.raises(AssertionError) as exc_info:
        vm.Value_Negative()
    assert "format_version=1 (unified) or =2 (per-binary CSV)" in str(exc_info.value)


@pytest.mark.parametrize("fv", [1, 2])
def test_value_negative_round_trip_via_from_token_ids(fv):
    """`_from_token_ids([256])` reconstructs a Value_Negative instance;
    the reconstructed instance emits the same single-id wire form."""
    vm = VocabularyManager(platform=None, format_version=fv)
    reconstructed = vm.Value_Negative._from_token_ids([256])
    assert reconstructed.get_token_ids().tolist() == [256]
    assert reconstructed.token_type == TokenType.VALUE_NEGATIVE


@pytest.mark.parametrize("fv", [1, 2])
def test_first_caller_registration_lands_after_value_negative(fv):
    """The first caller-driven (non-eager) vocab entry on a v1/v2 VM must
    land at id 257 — strictly after the pinned `value_negative` slot —
    so the marker's id stays stable across vocabs."""
    vm = VocabularyManager(platform=None, format_version=fv)
    # Register one VariantAxis token (an arbitrary opaque-string Inner
    # whose id assignment is auto-incrementing); it must take id 257.
    first = vm.Variant_Axis("arch:x64")
    assert first.get_token_ids().tolist() == [257]
