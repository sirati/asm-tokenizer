"""Keep/drop policy for Ghidra-resolved DATA refs attached to REG operands.

The Ghidra analyzer surfaces value-flow refs on operand destinations whose
register was loaded from a known-pointer slot (PC-relative literal-pool
loads, ARM ``movw``/``movt`` constant-building pairs, MIPS ``lui``/``ori``
high+low halves, RISC-V ``lui``/``addi``, PowerPC ``lis``/``ori``).

Two-layer policy:

1. **High-confidence override.** When Ghidra has classified the resolved
   address as one of the program-structural anchor kinds it maintains
   carefully (``STRING`` / ``PLT_FUNCTION`` / ``LOCAL_FUNCTION`` /
   ``CODE_PTR_TABLE_SLOT``), OR when the resolved address falls inside
   the current function's body extent (intra-function block/jump
   target), the resolved target is ALWAYS honored regardless of the
   instruction's PCode-level mem-access shape. Even csel-class
   inheritance into such a ref is semantically defensible (the
   conditional value flow IS that program-anchor).

2. **Fallback.** For lower-confidence kinds (``RO_DATA_PTR`` /
   ``EXT_FUNCTION_SYNTHETIC`` / ``JUMP_TABLE_SLOT`` / ``UNKNOWN`` /
   not-in-current-function), trust the resolved target only when the
   instruction is a real LOAD/STORE (PCode has-load-store) OR the
   instruction's base mnemonic is the high-half terminal of a known
   per-ISA constant-build pair (e.g. arm32 ``movt``, arm64 ``movk``,
   mips ``ori``/``addi``, riscv ``addi``, ppc ``ori``/``addi``).
   Otherwise the ref is csel-class noise and is dropped.

The override-kind set is INTENTIONALLY narrow. ``RO_DATA_PTR`` and
``EXT_FUNCTION_SYNTHETIC`` are excluded from the override set because
csel-inheritance false-positives are real noise there; they only fire
through the fallback when the instruction itself accesses memory or
matches the pair-terminal mnemonic.

This module is the SINGLE-CONCERN owner of "should we honor the
resolved_target this operand carries"; the per-architecture providers
(``arch/arm32/provider.py`` etc.) consult it via
:func:`should_honor_resolved_target` and treat the returned bool as the
sole admittance criterion.
"""

from __future__ import annotations

from typing import Optional

from tokenizer.disasm.metadata import AddressKind, AddressMetadataView
from tokenizer.disasm.types import Architecture


# High-half terminal mnemonics for canonical address-pair build sequences,
# keyed by architecture. The low-half mnemonic (which Ghidra anchors the
# upstream operand reference to) is listed in the comment for cross-
# reference; only the TERMINAL (high-half) mnemonic appears in the
# allow-list because that is where the COMBINED resolved address is
# attached as a DATA ref by the analyzer.
#
# Verified empirically:
#   - arm32 ``movt`` (paired with upstream ``movw``)
#   - arm64 ``movk`` (paired with upstream ``movz``)
#
# Specified but unverified (no candidate binary available or downstream
# REG-emit consumer not yet wired):
#   - mips32 / mips64 ``ori`` / ``addi`` / (mips64 also ``daddi``)
#       paired with upstream ``lui``
#   - riscv32 / riscv64 ``addi`` paired with upstream ``lui``
#   - ppc ``ori`` / ``addi`` paired with upstream ``lis``
#
# REMOVAL POLICY: entries that cannot be empirically verified within
# Scope C's time budget are kept in the table with a docstring caveat;
# they are inert until the per-architecture provider grows a REG-side
# ``resolved_target`` consumer (currently only ARM has one — see
# ``arch/arm32/provider.py``).
_ADDRESS_PAIR_HIGH_TERMINAL: dict[Architecture, frozenset[str]] = {
    Architecture.ARM32: frozenset({"movt"}),
    Architecture.AARCH64: frozenset({"movk"}),
    Architecture.MIPS: frozenset({"ori", "addi", "daddi"}),
    # NOTE: mips64 collapses onto Architecture.MIPS in the typed enum;
    # ``daddi`` is included in the same set because the MIPS provider
    # handles both 32-bit and 64-bit MIPS through one Architecture
    # value. The unused entries on 32-bit-only binaries are inert
    # (mnemonic never matches).
    Architecture.RISCV: frozenset({"addi"}),
    Architecture.PPC: frozenset({"ori", "addi"}),
}


# Address kinds for which Ghidra's classification is treated as
# high-confidence enough to honor the resolved_target unconditionally,
# bypassing the PCode-level mem-access gate AND the per-ISA pair-
# terminal mnemonic fallback. See module docstring for rationale.
_OVERRIDE_HIGH_CONFIDENCE_KINDS: frozenset[AddressKind] = frozenset({
    AddressKind.STRING,
    AddressKind.PLT_FUNCTION,
    AddressKind.LOCAL_FUNCTION,
    AddressKind.CODE_PTR_TABLE_SLOT,
})


def _in_current_function_body(
    resolved_target: int,
    func_min_addr: int,
    func_max_addr: int,
) -> bool:
    """Return ``True`` when ``resolved_target`` lies inside the current
    function's body extent (``[func_min_addr, func_max_addr]``).

    Intra-function jump/block targets surface a DATA ref on the operand
    whose value-flow led to a basic-block start. The block address is a
    program-structural anchor (the function CFG node identity), so the
    resolved target is honored even when Ghidra has not assigned a
    typed ``AddressKind`` to the block start (the default for most
    intra-function block heads).
    """
    # func_min_addr / func_max_addr are inclusive bounds in the existing
    # callsite convention (see ``arch/operands_base.py`` line 99); preserve
    # the inclusive interpretation here.
    return func_min_addr <= resolved_target <= func_max_addr


def should_honor_resolved_target(
    *,
    meta: Optional[AddressMetadataView],
    resolved_target: int,
    func_min_addr: int,
    func_max_addr: int,
    arch: Architecture,
    base_mnemonic: str,
    has_load_store: bool,
) -> bool:
    """Decide whether Ghidra's resolved DATA ref on a REG operand is honored.

    Returns ``True`` to emit the identity classifier tokens for
    ``resolved_target``; ``False`` to drop it as csel-class inheritance
    noise.

    Caller contract:
      - ``meta`` is the typed metadata view returned by
        ``MetadataLookup.lookup(resolved_target)``. May be ``None`` only
        when the caller declined to look up (defensive; the policy
        treats ``None`` the same as a kind outside the override set).
      - ``resolved_target`` is the address Ghidra resolved this REG
        operand to (already validated as non-None by the caller).
      - ``func_min_addr`` / ``func_max_addr`` are the current
        function's inclusive body bounds (same convention as
        ``arch/operands_base.py`` callsites).
      - ``arch`` is the typed architecture enum.
      - ``base_mnemonic`` is the cc-stripped, alias-canonicalized
        instruction mnemonic (``InstructionView.base_mnemonic``).
      - ``has_load_store`` is the PCode-level signal that this
        instruction performs at least one LOAD or STORE operation
        (``InstructionView.has_load_store``).

    Decision tree (short-circuit evaluation):
      1. High-confidence override on ``meta.kind`` -> True.
      2. Intra-function-body resolved target -> True.
      3. Instruction has memory access -> True (preserved fallback).
      4. Mnemonic in per-ISA pair-terminal allow-list -> True.
      5. Otherwise -> False (csel-suppression).
    """
    # Override layer: program-structural anchor kinds Ghidra maintains
    # carefully. Even when the instruction is a pure-register csel-class
    # operation, the inherited value-flow IS the program anchor.
    if meta is not None and meta.kind in _OVERRIDE_HIGH_CONFIDENCE_KINDS:
        return True

    # Override layer: intra-function block / jump targets. The resolved
    # address is a CFG-node anchor for the current function even when
    # Ghidra has not stamped a typed AddressKind on the block start.
    if _in_current_function_body(resolved_target, func_min_addr, func_max_addr):
        return True

    # Fallback layer: existing memory-access gate. Real LOAD/STORE
    # instructions (PC-relative literal-pool loads on ARM, etc.) get
    # their resolved target honored unconditionally because their
    # operand-level data ref IS the access target.
    if has_load_store:
        return True

    # Fallback layer: per-ISA constant-build pair terminal mnemonic.
    # The high-half terminal (e.g. arm32 ``movt``) is a register-domain
    # instruction with no LOAD/STORE PCode, but Ghidra attaches the
    # COMBINED resolved address as a DATA ref to its destination
    # operand. The allow-list opens this specific pattern through the
    # fallback without admitting the broader csel-class.
    allowed = _ADDRESS_PAIR_HIGH_TERMINAL.get(arch, frozenset())
    if base_mnemonic in allowed:
        return True

    # csel-class inheritance: low-confidence kind, no mem-access, not a
    # pair-terminal -> drop.
    return False
