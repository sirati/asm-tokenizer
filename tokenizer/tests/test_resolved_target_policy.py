"""Unit tests for the resolved-target keep/drop policy.

Tests cover the layered design:
  1. High-confidence override on ``AddressKind`` (STRING / PLT_FUNCTION /
     LOCAL_FUNCTION / CODE_PTR_TABLE_SLOT) — fires regardless of mem-
     access or mnemonic.
  2. Intra-function-body override — fires when the resolved address is
     inside the current function's body extent.
  3. Fallback on ``instruction_has_mem_access`` — preserves the legacy
     mem-access gate semantics for low-confidence kinds.
  4. Fallback on per-ISA pair-terminal mnemonic allow-list — opens
     ``movt`` / ``movk`` / etc. through without admitting csel-class.
  5. csel-class suppression — low-confidence kind + no mem-access + not
     a pair-terminal mnemonic drops the resolved target.

Mutation-test coverage: replacing the override predicate's body with
``return False`` MUST fail tests in the "override fires" group;
replacing it with ``return True`` MUST fail tests in the "suppression
holds" group.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenizer.disasm.metadata import AddressKind
from tokenizer.disasm.resolved_target_policy import (
    _ADDRESS_PAIR_HIGH_TERMINAL,
    _OVERRIDE_HIGH_CONFIDENCE_KINDS,
    should_honor_resolved_target,
)
from tokenizer.disasm.types import Architecture


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
_FUNC_MIN = 0x1000
_FUNC_MAX = 0x2000
_OUTSIDE_FUNC = 0x9000  # > _FUNC_MAX so intra-function override does NOT fire


def _meta(kind: AddressKind) -> SimpleNamespace:
    """Build a minimal address-metadata stand-in carrying just ``kind``."""
    return SimpleNamespace(kind=kind)


# ---------------------------------------------------------------------------
# Override layer — high-confidence AddressKinds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(_OVERRIDE_HIGH_CONFIDENCE_KINDS, key=int))
def test_override_fires_on_high_confidence_kind_even_without_mem_access(kind):
    """High-confidence kind => override fires regardless of mem-access /
    mnemonic. Mutation-test guard: replacing the override-kind check
    with ``return False`` would fail this test."""
    assert should_honor_resolved_target(
        meta=_meta(kind),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",          # NOT in any allow-list
        has_load_store=False,           # NOT a LOAD/STORE
    ) is True


def test_override_fires_on_string_with_mem_access_false():
    """Explicit STRING-on-csel positive (parity with the integration
    assertion in the plan)."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.STRING),
        resolved_target=0x10C78,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


def test_override_fires_on_plt_function_with_mem_access_false():
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.PLT_FUNCTION),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


def test_override_fires_on_local_function_with_mem_access_false():
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.LOCAL_FUNCTION),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


def test_override_fires_on_code_ptr_table_slot_with_mem_access_false():
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.CODE_PTR_TABLE_SLOT),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


# ---------------------------------------------------------------------------
# Override layer — intra-function body
# ---------------------------------------------------------------------------
def test_override_fires_on_intra_function_block_target():
    """Intra-function block target => override fires even on UNKNOWN
    kind. Mutation-test guard: removing the body-range check would
    fail this."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.UNKNOWN),
        resolved_target=0x1500,          # _FUNC_MIN < x < _FUNC_MAX
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


def test_override_fires_on_intra_function_block_target_at_min():
    """Inclusive lower bound."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.UNKNOWN),
        resolved_target=_FUNC_MIN,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


def test_override_fires_on_intra_function_block_target_at_max():
    """Inclusive upper bound."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.UNKNOWN),
        resolved_target=_FUNC_MAX,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is True


# ---------------------------------------------------------------------------
# Fallback layer — per-ISA pair-terminal mnemonic allow-list
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arch, mnemonic",
    [
        (Architecture.ARM32, "movt"),
        (Architecture.AARCH64, "movk"),
        (Architecture.MIPS, "ori"),
        (Architecture.MIPS, "addi"),
        (Architecture.MIPS, "daddi"),
        (Architecture.RISCV, "addi"),
        (Architecture.PPC, "ori"),
        (Architecture.PPC, "addi"),
    ],
)
def test_fallback_fires_on_pair_terminal_mnemonic_with_ro_data_ptr(arch, mnemonic):
    """Per-ISA pair-terminal mnemonic + low-confidence kind (RODATA) +
    no mem-access => fallback fires. Mutation-test guard: removing
    the per-ISA mnemonic check would fail this."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.RODATA),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=arch,
        base_mnemonic=mnemonic,
        has_load_store=False,
    ) is True


def test_fallback_fires_on_has_load_store_with_ro_data_ptr():
    """Real LOAD/STORE instruction + low-confidence kind (RODATA) =>
    fallback fires (preserves legacy mem-access gate)."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.RODATA),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.ARM32,
        base_mnemonic="ldr",
        has_load_store=True,
    ) is True


# ---------------------------------------------------------------------------
# csel-class suppression — low-confidence + no mem-access + not allow-listed
# ---------------------------------------------------------------------------
def test_suppression_drops_unknown_on_csel():
    """csel-class inheritance + UNKNOWN kind => drop. Mutation-test
    guard: replacing the policy with ``return True`` would fail this."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.UNKNOWN),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is False


def test_suppression_drops_ro_data_ptr_on_csel():
    """csel-class inheritance + RODATA (RO_DATA_PTR analog) => drop."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.RODATA),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is False


def test_suppression_drops_ext_function_synthetic_on_csel():
    """csel-class inheritance + EXT_FUNCTION_SYNTHETIC => drop. The
    EXT_FUNCTION_SYNTHETIC kind is INTENTIONALLY NOT in the override
    set (the user's design decision); a csel inheriting one is csel-
    noise."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.EXT_FUNCTION_SYNTHETIC),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is False


def test_suppression_drops_jump_table_slot_on_csel():
    """JUMP_TABLE_SLOT also not in override set => drop on csel."""
    assert should_honor_resolved_target(
        meta=_meta(AddressKind.JUMP_TABLE_SLOT),
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is False


def test_suppression_drops_none_meta_on_csel():
    """``meta=None`` (defensive shape) + no mem-access + not allow-listed
    => drop. Mirrors the kind-not-in-override path."""
    assert should_honor_resolved_target(
        meta=None,
        resolved_target=_OUTSIDE_FUNC,
        func_min_addr=_FUNC_MIN,
        func_max_addr=_FUNC_MAX,
        arch=Architecture.AARCH64,
        base_mnemonic="csel",
        has_load_store=False,
    ) is False


# ---------------------------------------------------------------------------
# Per-ISA allow-list contents — pin the shipped table
# ---------------------------------------------------------------------------
def test_allow_list_shape_matches_design():
    """The per-ISA pair-terminal allow-list ships exactly the mnemonics
    documented in the design. Pin them so additions/removals require
    an explicit test update."""
    assert _ADDRESS_PAIR_HIGH_TERMINAL == {
        Architecture.ARM32: frozenset({"movt"}),
        Architecture.AARCH64: frozenset({"movk"}),
        Architecture.MIPS: frozenset({"ori", "addi", "daddi"}),
        Architecture.RISCV: frozenset({"addi"}),
        Architecture.PPC: frozenset({"ori", "addi"}),
    }


def test_override_kind_set_matches_design():
    """The override-kind set ships exactly the four address kinds
    documented in the design. Pin them so additions require an
    explicit test update (the user's design decision)."""
    assert _OVERRIDE_HIGH_CONFIDENCE_KINDS == frozenset({
        AddressKind.STRING,
        AddressKind.PLT_FUNCTION,
        AddressKind.LOCAL_FUNCTION,
        AddressKind.CODE_PTR_TABLE_SLOT,
    })
