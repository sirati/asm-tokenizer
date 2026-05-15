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

from tokenizer.disasm.types import Architecture, ShiftKind


# Per-ISA bracket-open characters. The presence of this rich-typed
# ``java.lang.Character`` item in ``getDefaultOperandRepresentationList``
# is the SYNTACTIC discriminator for "operand was written with bracket
# framing in the asm" — orthogonal to the SEMANTIC has_load_store check
# (which says "instruction accesses memory"). Both are needed: x86
# ``rep stosb rdi`` has DYNAMIC + has_load_store but RDI is rendered
# WITHOUT brackets (implicit-memory register), so it's a syntactic REG;
# arm64 ``strh wzr, [...]`` has WZR as DYNAMIC + has_load_store but WZR
# (the zero register, semantically a constant-zero source) is rendered
# WITHOUT brackets, so it's also a syntactic REG. No PCode/OperandType
# bit reliably discriminates these from real bracketed-mem operands,
# but the rich-typed Character marker in the print rendering does.
_BRACKET_OPEN_CHARS: dict[Architecture, frozenset[str]] = {
    Architecture.ARM32: frozenset({"["}),
    Architecture.AARCH64: frozenset({"["}),
    Architecture.X86: frozenset({"["}),
    Architecture.PPC: frozenset({"("}),
    Architecture.MIPS: frozenset({"("}),
    Architecture.RISCV: frozenset({"("}),
}


def operand_is_bracketed(ghidra_insn: Any, op_idx: int, arch: Architecture) -> bool:
    """True iff operand ``op_idx``'s representation list contains the
    per-ISA bracket-open Character marker.

    Uses typed ``java.lang.Character`` ``isinstance`` + ``charValue()``
    against the per-ISA char set; no raw ``str()`` cast. The bracket
    Characters are the only repr-list items we read at this layer; we
    don't need a complete role map for every Character because the
    is-memory classifier only asks one yes/no question (is there a
    bracket?). Hard-error on unknown Characters belongs in a future
    full-role-map module if/when we extend operand-CC / vector-arrangement
    extraction.
    """
    chars = _BRACKET_OPEN_CHARS.get(arch)
    if not chars:
        return False
    try:
        repr_list = ghidra_insn.getDefaultOperandRepresentationList(op_idx) or ()
    except Exception:
        return False
    from java.lang import Character as JavaCharacter

    for item in repr_list:
        if isinstance(item, JavaCharacter):
            try:
                c = chr(item.charValue())
            except Exception:
                continue
            if c in chars:
                return True
    return False


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
    #
    # ARM ``s_flag`` (update-flags) instructions like ``movs``/``adds`` and
    # the SLEIGH spec for any shift-with-flags emit BOTH:
    #   (a) a shift-CARRY-detection op (e.g. ``INT_RIGHT r1, uniq`` where
    #       ``uniq = shift_amount - 1`` — the LSB of the result is the
    #       carry bit out of the value shift).
    #   (b) the VALUE shift op (``INT_LEFT r1, const(amount)``) whose
    #       result feeds the actual arithmetic.
    # The carry op carries a UNIQ second input (the precomputed
    # ``amount - 1``), the value op carries a CONST second input (the
    # actual shift amount). We want the value shift, not the carry. The
    # rich-IR discriminator is exactly that: prefer ops with a constant
    # second input. Fall back to a non-constant-amount match only when
    # NO constant-amount op exists (shift-by-register form, rare).
    fallback: Optional[tuple[ShiftKind, int]] = None
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
            if second.isConstant():
                return (shift_table[opc], int(second.getOffset()))
            if fallback is None:
                fallback = (shift_table[opc], 0)
        except Exception:
            if fallback is None:
                fallback = (shift_table[opc], 0)

    if fallback is not None:
        return fallback
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

    Both pre-indexed (``[r, #imm]!``) and post-indexed (``[r], #imm``)
    forms cause the base register to be self-updated by the addressing-
    mode displacement; the rich-IR signal alone cannot distinguish them
    by self-update alone. The discriminator is which value the LOAD/STORE
    consumes:
    - Pre-indexed: the base self-update PCode op runs BEFORE the
      LOAD/STORE; the LOAD/STORE's address varnode IS ``base_register``
      (the updated value).
    - Post-indexed: the SLEIGH spec emits ``COPY base → uniq`` (snapshot)
      first, then the self-update of ``base``, then ``LOAD/STORE uniq``
      using the snapshot of the un-updated base.

    The TUPLE convention matches the consumer's emission grammar:
    - ``(True,  True,  False)`` -> pre-indexed: render disp inside
      brackets + ``!`` writeback marker.
    - ``(False, False, True)``  -> post-indexed: render brackets without
      disp, then post-index separator + disp tokens.
    - ``(False, False, False)`` -> plain offset: render disp inside
      brackets, no marker.

    ``writeback`` here is ONLY the asm-renderable ``!`` flag (pre-indexed
    only). The semantic "base register is auto-updated" is true for both
    pre and post but the consumer uses two separate tokens for the two
    forms; this function picks the right one.
    """
    if base_register is None:
        return (False, False, False)

    if not register_is_addressing_mode_written(ghidra_insn, base_register):
        return (False, False, False)

    from ghidra.program.model.pcode import PcodeOp

    pcode_ops = list(ghidra_insn.getPcode() or ())

    addr_varnode = None
    for pop in pcode_ops:
        opc = pop.getOpcode()
        inputs = pop.getInputs()
        if opc in (PcodeOp.LOAD, PcodeOp.STORE) and len(inputs) >= 2:
            addr_varnode = inputs[1]
            break

    if addr_varnode is None:
        return (False, False, False)

    if _varnode_matches_register(addr_varnode, base_register):
        return (True, True, False)

    addr_key = _varnode_key(addr_varnode)
    if addr_key is None:
        return (False, False, False)

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
        return (False, False, True)

    return (False, False, False)
