"""Tests for the Ghidra PPC prefix builder (``_build_prefixes_ppc``).

Concern: the Ghidra disassembly path must emit the SAME typed prefix
views for PowerPC per-instruction modifiers (branch condition,
CR0-update Rc bit) that the angr/Capstone path emits, so the downstream
``arch/ppc/provider`` consumer is provider-agnostic. Ghidra exposes
neither the BO/BI fields nor the Rc bit as a typed operand; both are
folded into the displayed mnemonic (verified empirically against
Ghidra 12.0.4 ``PowerPC:BE:32`` decode), so the builder recovers them
from ``getMnemonicString()`` -- the same idiom the ARM builder uses.

The builder only touches ``getMnemonicString()``, so these tests drive
it with a one-method mock (no JVM), mirroring the mock idiom in
``test_ghidra_switch_table_thunk_skip.py``.

The cross-provider leg asserts equivalence at the RENDERED-TOKEN level
(the ``_PPC_BC_NAMES`` word), not on the raw ``bc`` integer: the angr
path passes Capstone's native ``ppc_bc`` enum value straight through,
whose numeric encoding is Capstone-version-dependent, whereas the Ghidra
path emits the ``_PPC_BC_NAMES`` contract value directly. Equivalence is
meaningful only at the word the consumer table renders.
"""

from __future__ import annotations

from typing import Optional

from tokenizer.arch.ppc.provider import _PPC_BC_NAMES
from tokenizer.disasm.ghidra_provider.prefix_build import _build_prefixes_ppc
from tokenizer.disasm.types import (
    PpcBranchConditionPrefixView,
    PpcUpdateCr0PrefixView,
)

from tokenizer.tests._provider_support import HAS_ANGR

import pytest


class _MockPpcInsn:
    """Minimal Ghidra ``Instruction`` surface for the prefix builder.

    The builder calls only ``getMnemonicString()``; nothing else of the
    real Java instruction is touched.
    """

    def __init__(self, mnemonic: str) -> None:
        self._mnemonic = mnemonic

    def getMnemonicString(self) -> str:  # noqa: N802 (Ghidra Java name)
        return self._mnemonic


def _rendered_bc(views: list) -> Optional[str]:
    """The ``_PPC_BC_NAMES`` word for the (at most one) branch-condition view."""
    for v in views:
        if isinstance(v, PpcBranchConditionPrefixView):
            return _PPC_BC_NAMES.get(v.bc)
    return None


def _has_cr0(views: list) -> bool:
    return any(isinstance(v, PpcUpdateCr0PrefixView) for v in views)


# ---------------------------------------------------------------------------
# CR0-update (Rc bit): trailing ``.`` on the Ghidra mnemonic.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mnemonic", ["add.", "or.", "subf.", "rlwinm.", "neg."])
def test_rc_suffix_emits_cr0_update(mnemonic: str) -> None:
    views = _build_prefixes_ppc(_MockPpcInsn(mnemonic))
    assert _has_cr0(views)


@pytest.mark.parametrize("mnemonic", ["add", "or", "mullw", "cmpw", "lwz", "mr"])
def test_no_rc_suffix_omits_cr0_update(mnemonic: str) -> None:
    views = _build_prefixes_ppc(_MockPpcInsn(mnemonic))
    assert not _has_cr0(views)


# ---------------------------------------------------------------------------
# Branch condition: ``b^CC^...`` folded mnemonic -> rendered cc word.
# Expected words are the Ghidra-rendered conditions (lt/le/eq/ge/gt/ne/so/ns);
# ``un``/``nu`` are Capstone-only aliases Ghidra never renders.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mnemonic,expected_cc",
    [
        ("beq", "eq"), ("bne", "ne"), ("blt", "lt"), ("bge", "ge"),
        ("bgt", "gt"), ("ble", "le"), ("bso", "so"), ("bns", "ns"),
        # link / absolute / register-branch variants carry the same cc.
        ("beql", "eq"), ("beqa", "eq"), ("beqla", "eq"),
        ("beqlr", "eq"), ("beqlrl", "eq"), ("beqctr", "eq"), ("beqctrl", "eq"),
        ("bnelr", "ne"), ("bltctr", "lt"),
    ],
)
def test_conditional_branch_emits_bc(mnemonic: str, expected_cc: str) -> None:
    views = _build_prefixes_ppc(_MockPpcInsn(mnemonic))
    assert _rendered_bc(views) == expected_cc


@pytest.mark.parametrize(
    "mnemonic",
    [
        # Unconditional / CTR / link branches carry no folded condition.
        "b", "ba", "bl", "bla", "blr", "bctr", "bctrl",
        # CTR-decrement family: condition (if any) is a separate operand,
        # NOT folded into the mnemonic -> no branch-condition view.
        "bdnz", "bdnzl", "bdz", "bdnzt", "bdnzf",
        # Non-branch instructions.
        "add", "lwz",
    ],
)
def test_non_conditional_branch_omits_bc(mnemonic: str) -> None:
    views = _build_prefixes_ppc(_MockPpcInsn(mnemonic))
    assert _rendered_bc(views) is None


def test_uppercase_mnemonic_is_normalized() -> None:
    """Ghidra may surface upper-case mnemonics; the builder lowercases."""
    assert _rendered_bc(_build_prefixes_ppc(_MockPpcInsn("BEQ"))) == "eq"
    assert _has_cr0(_build_prefixes_ppc(_MockPpcInsn("ADD.")))


# ---------------------------------------------------------------------------
# Cross-provider equivalence (angr leg skipped when backend unavailable).
#
# Same PPC bytes -> both providers -> equal rendered cc word + equal
# CR0-update presence. Asserts at the consumer-table word, not the raw
# Capstone ``bc`` integer (see module docstring).
# ---------------------------------------------------------------------------

# (bytes, ghidra-mnemonic) for BE:32 PPC; the angr leg decodes the bytes.
_PPC_BYTES_CASES = [
    (bytes([0x41, 0x82, 0x00, 0x08]), "beq"),
    (bytes([0x40, 0x82, 0x00, 0x08]), "bne"),
    (bytes([0x41, 0x80, 0x00, 0x08]), "blt"),
    (bytes([0x40, 0x80, 0x00, 0x08]), "bge"),
    (bytes([0x41, 0x81, 0x00, 0x08]), "bgt"),
    (bytes([0x40, 0x81, 0x00, 0x08]), "ble"),
    (bytes([0x4d, 0x82, 0x00, 0x20]), "beqlr"),
    (bytes([0x7c, 0x64, 0x2a, 0x14]), "add"),
    (bytes([0x7c, 0x64, 0x2a, 0x15]), "add."),
]


@pytest.mark.skipif(not HAS_ANGR, reason="angr disassembler backend unavailable")
@pytest.mark.parametrize("raw,ghidra_mnemonic", _PPC_BYTES_CASES)
def test_ghidra_matches_angr_rendered(raw: bytes, ghidra_mnemonic: str) -> None:
    import capstone
    from capstone import CS_ARCH_PPC, CS_MODE_32, CS_MODE_BIG_ENDIAN, Cs

    from tokenizer.disasm.angr_provider.prefixes import (
        _build_prefixes_ppc as _build_prefixes_ppc_angr,
    )

    md = Cs(CS_ARCH_PPC, CS_MODE_32 | CS_MODE_BIG_ENDIAN)
    md.detail = True
    cs_insn = next(md.disasm(raw, 0x1000))

    angr_views = _build_prefixes_ppc_angr(cs_insn)
    ghidra_views = _build_prefixes_ppc(_MockPpcInsn(ghidra_mnemonic))

    assert _rendered_bc(ghidra_views) == _rendered_bc(angr_views)
    assert _has_cr0(ghidra_views) == _has_cr0(angr_views)
