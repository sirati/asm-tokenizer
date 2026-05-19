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


# --------------------------------------------------------------------------
# Constructor: reserved-digit pre-population only fires for v1 and v2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fv", [1, 2])
def test_constructor_reserves_digit_slots_for_inline_digit_versions(fv):
    """v1 (unified) and v2 (per-binary CSV) both use the inline-digit
    wire encoding and therefore pre-populate IDs 0..255 with
    `digit_<HH>` placeholders tagged UNRESOLVED."""
    vm = VocabularyManager(platform=None, format_version=fv)
    assert len(vm.id_to_token) == _V2_RESERVED_DIGIT_COUNT
    assert all(
        vm.id_to_token[i] == f"digit_{i:02X}"
        for i in range(_V2_RESERVED_DIGIT_COUNT)
    )
    # The token_type cache marks every reserved slot UNRESOLVED.
    type_cache = np.asarray(vm.id_to_token_type[:_V2_RESERVED_DIGIT_COUNT])
    assert np.all(type_cache == TokenType.UNRESOLVED)


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
