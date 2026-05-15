"""Typed PCode inspection primitives.

PCode is Ghidra's semantic IR: each instruction becomes a sequence of typed
PCode ops (``LOAD``, ``STORE``, ``INT_ADD``, ``INT_LEFT``, ``COPY``, ...) over
typed varnodes (registers, constants, uniques, memory addresses). This module
exposes the small set of PCode queries needed to derive ARM/AArch64 operand
semantics (shift kind/amount, writeback, pre/post-indexed) WITHOUT touching
the rendered representation strings.

Design notes:
- Varnode equality across PCode op boundaries is by (address-space, offset,
  size); the same register appearing in two PCode op inputs is the SAME
  varnode by this triple, not Python ``is``.
- COPY ops establish reflexive transitive propagation: if a uniq varnode is
  ``COPY``ed from a register, any subsequent use of that uniq is semantically
  equivalent to using the register. Some ISAs (AArch64) chain multiple COPYs
  through intermediate uniqs before consuming the value; we follow the chain
  with a small max-depth bound.
- Constants in PCode are varnodes in the ``const`` address space; their value
  is accessed via ``Varnode.getOffset()``.
"""

from __future__ import annotations

from typing import Any, Optional

from tokenizer.disasm.types import ShiftKind


_PCODE_SHIFT_OPCODE_TO_KIND: dict[int, ShiftKind] = {}
"""Populated lazily on first call (depends on JVM-loaded PcodeOp constants)."""


def _ensure_shift_table() -> dict[int, ShiftKind]:
    global _PCODE_SHIFT_OPCODE_TO_KIND
    if _PCODE_SHIFT_OPCODE_TO_KIND:
        return _PCODE_SHIFT_OPCODE_TO_KIND
    from ghidra.program.model.pcode import PcodeOp

    _PCODE_SHIFT_OPCODE_TO_KIND = {
        int(PcodeOp.INT_LEFT): ShiftKind.LSL,
        int(PcodeOp.INT_RIGHT): ShiftKind.LSR,
        int(PcodeOp.INT_SRIGHT): ShiftKind.ASR,
    }
    return _PCODE_SHIFT_OPCODE_TO_KIND


def _varnode_matches_register(varnode: Any, register: Any) -> bool:
    """True iff ``varnode`` refers to the same register as ``register``.

    Compares by (address-space, offset, size). Ghidra's ``Varnode.isRegister``
    + the register's full address are the canonical typed identity.
    """
    if varnode is None or register is None:
        return False
    try:
        if not varnode.isRegister():
            return False
        addr = varnode.getAddress()
        if addr is None:
            return False
        return (
            addr.getAddressSpace().equals(register.getAddressSpace())
            and int(addr.getOffset()) == int(register.getOffset())
            and int(varnode.getSize()) == int(register.getMinimumByteSize())
        )
    except Exception:
        return False


def _varnode_key(varnode: Any) -> Optional[tuple[str, int, int]]:
    """Triple-key for comparing two varnodes (space-name, offset, size)."""
    if varnode is None:
        return None
    try:
        addr = varnode.getAddress()
        if addr is None:
            return None
        return (
            str(addr.getAddressSpace().getName()),
            int(addr.getOffset()),
            int(varnode.getSize()),
        )
    except Exception:
        return None


def register_is_addressing_mode_written(
    ghidra_insn: Any, register: Any, max_iter: int = 6
) -> bool:
    """True iff the instruction's PCode WRITES ``register`` via addressing-
    mode arithmetic (rich-IR signal for ARM writeback).

    The rule, in two passes over the PCode op sequence:

    1. **Propagation closure** from ``register`` through non-LOAD ops: any
       op whose input set contains a varnode in the current propagated set
       adds its output to the set. This captures both direct self-update
       (``INT_ADD r, const → r``) and the iterative-temp-register pattern
       (arm32 ``STMDB SP!``: ``INT_SUB sp → mult_addr; INT_SUB mult_addr →
       mult_addr; ...; INT_ADD mult_addr → sp`` — sp's final write is by
       an op whose input mult_addr is in propagated-from-sp). LOAD ops are
       excluded from propagation because their output represents the
       loaded VALUE, not the base register's flow.
    2. **Writeback detection**: scan non-LOAD ops for one whose output IS
       ``register`` AND whose input set intersects propagated. The
       LOAD-exclusion handles ``LDR Rn, [Rn, ...]`` where the load
       destination happens to be the base register; the LOAD writes Rn
       because it's the dest, NOT because of writeback semantics.

    This unifies all ARM writeback shapes (direct self-update on mem
    operand pre/post-indexing, iterative-temp pattern on REG_LIST
    stmdb/ldmia), and correctly rejects the false-positive shapes (base
    register coincidentally equals load destination, post-load
    normalization writing a different register).
    """
    if register is None:
        return False
    register_key = _varnode_key_for_register(register)
    if register_key is None:
        return False
    try:
        pcode_ops = list(ghidra_insn.getPcode() or ())
    except Exception:
        return False
    from ghidra.program.model.pcode import PcodeOp

    propagated: set[tuple[str, int, int]] = {register_key}
    for _ in range(max_iter):
        changed = False
        for pop in pcode_ops:
            if pop.getOpcode() == PcodeOp.LOAD:
                continue
            out_key = _varnode_key(pop.getOutput())
            if out_key is None or out_key in propagated:
                continue
            for inp in pop.getInputs():
                if _varnode_key(inp) in propagated:
                    propagated.add(out_key)
                    changed = True
                    break
        if not changed:
            break

    for pop in pcode_ops:
        if pop.getOpcode() == PcodeOp.LOAD:
            continue
        out_key = _varnode_key(pop.getOutput())
        if out_key != register_key:
            continue
        for inp in pop.getInputs():
            if _varnode_key(inp) in propagated:
                return True
    return False


def has_load_store(ghidra_insn: Any) -> bool:
    """True iff the instruction's PCode contains any LOAD or STORE op.

    Used to gate ARM/AArch64 is-memory classification: ARM shifted-register
    operands (``r1, lsl #2`` in arithmetic instructions like ``add``/``sbc``)
    have the same ``OperandType.DYNAMIC`` bit as memory operands but produce
    NO LOAD/STORE in PCode. The instruction-level presence of LOAD/STORE
    is the rich-IR discriminator.
    """
    from ghidra.program.model.pcode import PcodeOp

    try:
        for pop in ghidra_insn.getPcode():
            opc = pop.getOpcode()
            if opc == PcodeOp.LOAD or opc == PcodeOp.STORE:
                return True
    except Exception:
        return False
    return False


def find_shift_on_register(
    ghidra_insn: Any, register: Any, max_copy_depth: int = 6
) -> tuple[ShiftKind, int]:
    """Return the shift modifier applied to ``register`` in this instruction.

    Scans the instruction's PCode for ``INT_LEFT`` / ``INT_RIGHT`` /
    ``INT_SRIGHT`` ops whose first input is ``register`` OR a uniq varnode
    that was COPYed (possibly transitively) from ``register``. The second
    input is the shift amount (a PCode constant).

    AArch64 introduces a COPY chain (``COPY reg -> uniq_a; COPY uniq_a ->
    uniq_b; INT_LEFT uniq_b const -> uniq_c``) before the shift op, so we
    follow COPYs reflexively up to ``max_copy_depth`` hops.

    Returns ``(ShiftKind.NONE, 0)`` if no shift on this register is found.

    NOTE: PCode does NOT distinguish ``ROR`` / ``RRX`` from generic INT_*
    opcodes directly (Ghidra lifts ``ROR`` as a combination of INT_LEFT/
    INT_RIGHT/INT_OR, and ``RRX`` involves the carry flag). If a caller
    needs ROR/RRX, this helper hard-errors via the calling code (the shift
    keyword is detectable in the rendered repr; if it lifts to a non-shift
    PCode pattern the caller should surface that explicitly).
    """
    if register is None:
        return (ShiftKind.NONE, 0)

    shift_table = _ensure_shift_table()
    register_key = _varnode_key_for_register(register)

    pcode_ops = list(ghidra_insn.getPcode() or ())

    # Step 1: reflexive transitive closure over COPYs starting from register.
    # `propagated` is the set of varnode-keys that semantically carry the
    # register's value.
    from ghidra.program.model.pcode import PcodeOp

    propagated: set[tuple[str, int, int]] = set()
    if register_key is not None:
        propagated.add(register_key)
    for _ in range(max_copy_depth):
        changed = False
        for pop in pcode_ops:
            if pop.getOpcode() != PcodeOp.COPY:
                continue
            inputs = pop.getInputs()
            if not inputs:
                continue
            in_key = _varnode_key(inputs[0])
            if in_key is None or in_key not in propagated:
                continue
            out_key = _varnode_key(pop.getOutput())
            if out_key is not None and out_key not in propagated:
                propagated.add(out_key)
                changed = True
        if not changed:
            break

    # Step 2: find a shift PCode op consuming a propagated varnode.
    for pop in pcode_ops:
        opc = int(pop.getOpcode())
        if opc not in shift_table:
            continue
        inputs = pop.getInputs()
        if len(inputs) < 2:
            continue
        first_key = _varnode_key(inputs[0])
        if first_key not in propagated:
            continue
        second = inputs[1]
        try:
            if not second.isConstant():
                # Variable shift amount (rare in mem operands); fall back to 0.
                return (shift_table[opc], 0)
            return (shift_table[opc], int(second.getOffset()))
        except Exception:
            return (shift_table[opc], 0)

    return (ShiftKind.NONE, 0)


def _varnode_key_for_register(register: Any) -> Optional[tuple[str, int, int]]:
    """Project a Ghidra Register into the same triple-key shape as a Varnode."""
    if register is None:
        return None
    try:
        return (
            str(register.getAddressSpace().getName()),
            int(register.getOffset()),
            int(register.getMinimumByteSize()),
        )
    except Exception:
        return None


def classify_memory_addressing(
    ghidra_insn: Any, base_register: Any
) -> tuple[bool, bool, bool]:
    """Return (writeback, pre_indexed, post_indexed) from PCode shape.

    Discrimination, in rich-IR terms:
    - writeback ↔ ``base_register`` is in the instruction's result-objects
      (the instruction modifies the base).
    - When writeback is on, look at the FIRST ``LOAD``/``STORE`` op:
        - if its address varnode IS ``base_register`` directly: pre-indexed
          (the base self-update PCode op runs BEFORE the LOAD/STORE; the
          LOAD/STORE sees the updated base).
        - if its address varnode is a uniq that was ``COPY``ed from
          ``base_register``: post-indexed (the snapshot of the un-updated
          base is what the LOAD/STORE consumes; the base self-update
          happens out-of-band).
    - When writeback is off: plain offset addressing (all three False).
    """
    if base_register is None:
        return (False, False, False)

    writeback = register_is_addressing_mode_written(ghidra_insn, base_register)
    if not writeback:
        return (False, False, False)

    from ghidra.program.model.pcode import PcodeOp

    pcode_ops = list(ghidra_insn.getPcode() or ())

    # Find the first LOAD or STORE op + its address varnode.
    addr_varnode = None
    for pop in pcode_ops:
        opc = pop.getOpcode()
        inputs = pop.getInputs()
        if opc == PcodeOp.LOAD and len(inputs) >= 2:
            addr_varnode = inputs[1]
            break
        if opc == PcodeOp.STORE and len(inputs) >= 2:
            addr_varnode = inputs[1]
            break

    if addr_varnode is None:
        # writeback claimed but no LOAD/STORE — unexpected; treat as plain
        # writeback without pre/post discrimination so caller surfaces it.
        return (writeback, False, False)

    if _varnode_matches_register(addr_varnode, base_register):
        # Pre-indexed: LOAD/STORE reads the updated base directly.
        return (True, True, False)

    # Otherwise: post-indexed iff a COPY of base_register flows to
    # addr_varnode. Build the propagation set and check membership.
    addr_key = _varnode_key(addr_varnode)
    if addr_key is None:
        return (True, False, False)

    register_key = _varnode_key_for_register(base_register)
    propagated: set[tuple[str, int, int]] = set()
    if register_key is not None:
        propagated.add(register_key)
    for _ in range(6):
        changed = False
        for pop in pcode_ops:
            if pop.getOpcode() != PcodeOp.COPY:
                continue
            inputs = pop.getInputs()
            if not inputs:
                continue
            if _varnode_key(inputs[0]) not in propagated:
                continue
            out_key = _varnode_key(pop.getOutput())
            if out_key is not None and out_key not in propagated:
                propagated.add(out_key)
                changed = True
        if not changed:
            break

    if addr_key in propagated:
        return (True, False, True)

    # Writeback claimed but neither pre nor post pattern matched — surface
    # to caller; this means an unexpected PCode shape we should investigate.
    return (True, False, False)
