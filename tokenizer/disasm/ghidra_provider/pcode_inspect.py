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

from dataclasses import dataclass
from typing import Any, Optional

from tokenizer.disasm.types import Architecture, ShiftKind
from tokenizer.disasm.ghidra_provider import jvm_types


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
    JavaCharacter = jvm_types.JavaCharacter

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
    PcodeOp = jvm_types.PcodeOp

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
    PcodeOp = jvm_types.PcodeOp

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


# ---------------------------------------------------------------------------
# FLOAT_* p-code opcode classification
# ---------------------------------------------------------------------------
# The set of FP-semantic p-code opcodes (resolved to integer opcode values
# on first use via ``_resolve_float_pcode_sets``), partitioned by which
# side of the op carries integer-typed data:
#
#   - ``_FLOAT_PCODE_OPCODES``: every FLOAT_* opcode (the master scan set).
#   - ``_FLOAT_PCODE_INT_INPUT_OPCODES``: opcodes whose input varnode is
#     integer-typed (``FLOAT_INT2FLOAT``: integer source -> FP destination).
#     The output is still FP.
#   - ``_FLOAT_PCODE_INT_OUTPUT_OPCODES``: opcodes whose output varnode is
#     integer-typed (``FLOAT_TRUNC`` truncate-to-int; the boolean
#     comparison opcodes ``FLOAT_EQUAL`` / ``FLOAT_NOTEQUAL`` /
#     ``FLOAT_LESS`` / ``FLOAT_LESSEQUAL`` / ``FLOAT_NAN`` produce a
#     1-byte bool). The inputs are still FP.
#
# Binding to integer opcode values (rather than ``PcodeOp.getMnemonic``
# strings) is mandatory: Ghidra's mnemonic strings drop the ``FLOAT_``
# prefix for several conversion/rounding opcodes (``INT2FLOAT``,
# ``FLOAT2FLOAT``, ``TRUNC``, ``CEIL``, ``FLOOR``, ``ROUND``) even though
# the class constants are named ``FLOAT_INT2FLOAT`` / ``FLOAT_FLOAT2FLOAT``
# / ``FLOAT_TRUNC`` / etc. The class constants are the only stable handle.
_FLOAT_PCODE_OPCODES: frozenset[int] = frozenset()
_FLOAT_PCODE_INT_INPUT_OPCODES: frozenset[int] = frozenset()
_FLOAT_PCODE_INT_OUTPUT_OPCODES: frozenset[int] = frozenset()


def _resolve_float_pcode_sets() -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """Resolve the FLOAT_* PcodeOp class-constant names to integer opcode
    values; cache on the module globals so the lookup runs once.

    Lazy because importing ``ghidra.program.model.pcode`` at module load
    would require Ghidra to be started -- the import here happens on
    first use, mirroring ``_ensure_shift_table``'s lazy initialisation
    pattern in this module.
    """
    global _FLOAT_PCODE_OPCODES, _FLOAT_PCODE_INT_INPUT_OPCODES, _FLOAT_PCODE_INT_OUTPUT_OPCODES
    if _FLOAT_PCODE_OPCODES:
        return (
            _FLOAT_PCODE_OPCODES,
            _FLOAT_PCODE_INT_INPUT_OPCODES,
            _FLOAT_PCODE_INT_OUTPUT_OPCODES,
        )
    PcodeOp = jvm_types.PcodeOp

    # Master FLOAT_* set: enumerate the class-constant names directly so
    # adding a new FLOAT_* opcode upstream (e.g. FLOAT_HALF support)
    # would surface here without code changes if the constant follows
    # the FLOAT_ prefix convention.
    master = frozenset(
        getattr(PcodeOp, name)
        for name in dir(PcodeOp)
        if name.startswith("FLOAT_") and isinstance(getattr(PcodeOp, name, None), int)
    )
    int_input = frozenset({PcodeOp.FLOAT_INT2FLOAT})
    int_output = frozenset({
        PcodeOp.FLOAT_TRUNC,
        PcodeOp.FLOAT_EQUAL,
        PcodeOp.FLOAT_NOTEQUAL,
        PcodeOp.FLOAT_LESS,
        PcodeOp.FLOAT_LESSEQUAL,
        PcodeOp.FLOAT_NAN,
    })
    _FLOAT_PCODE_OPCODES = master
    _FLOAT_PCODE_INT_INPUT_OPCODES = int_input
    _FLOAT_PCODE_INT_OUTPUT_OPCODES = int_output
    return master, int_input, int_output


# ---------------------------------------------------------------------------
# Per-instruction one-pass PCode summary
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InstructionPcodeSummary:
    """Instruction-level signals extracted from ONE walk of ``getPcode()``.

    Computed once per instruction by the cursor's ``_advance`` (via the
    decode helper) and threaded into every per-operand classification
    call. Replaces the legacy per-operand re-walks: ``has_load_store``,
    the FLOAT_* signature scan (``fp_keys`` / ``fp_width``), the
    LOAD/STORE-routed FP attribution scan (``fp_touches_load_store``)
    and the MEM-access-size scan (``mem_access_size``) each used to
    re-iterate the instruction's PCode independently — on a JPype
    backend every op/varnode accessor is a JVM round-trip, so the
    per-operand multiplication dominated the decode hot path.

    Fields:

    - ``has_load_store``: the instruction's PCode contains a LOAD or
      STORE op. Gates ARM/AArch64 is-memory classification (ARM
      shifted-register operands carry ``OperandType.DYNAMIC`` like
      memory operands but emit no LOAD/STORE) and feeds the downstream
      resolved-target keep/drop policy via
      ``InstructionView.has_load_store``.
    - ``fp_keys``: set of ``(address_string, size)`` tuples covered by
      FP-typed sides of every ``FLOAT_*`` p-code op (skipping the
      int-input side of ``FLOAT_INT2FLOAT`` and the int-output side of
      ``FLOAT_TRUNC`` / comparison opcodes). The stringified address
      key avoids relying on ``Address`` hashability across pyghidra;
      the pair uniquely identifies the register slice or temp the
      FLOAT_* op touches. Empty when no FLOAT_* opcode fires.
    - ``fp_width``: smallest nonzero size among the FP-typed varnodes
      (0 if none) — the operand FP width oracle.
    - ``fp_touches_load_store``: any LOAD output varnode or STORE
      value-input varnode participates in ``fp_keys``. Captures
      register-indirect FP loads (``addsd xmm0, qword ptr [rax]``)
      where SLEIGH emits a separate LOAD feeding the FLOAT_* op. The
      signal is instruction-level (not per-operand) by design: x86
      instructions with FP semantics have at most one memory operand;
      multi-MEM instructions (string ops) do not emit FLOAT_* p-code.
    - ``mem_access_size``: memory-access size in bytes, derived from
      SLEIGH-emitted PCode LOAD output-varnode size / STORE value-input
      size (smallest nonzero; ``default_mem_size`` when none). This is
      the only reliable oracle — the legacy sibling-register-width
      heuristic conflated pointer-width address-computation regs (e.g.
      x64 r14 = 8B) with value regs, breaking 0x66 operand-size-override
      and MOVZX/MOVSX byte/word -> wider dest. The value is keyed per
      instruction, not per operand: x86 mem-operand instructions have
      exactly one LOAD/STORE per operand. ARM / MIPS / PPC / RISC-V
      consumers do not read ``op.size`` on MEM operands so the value is
      harmless on non-x86 ISAs.
    """

    has_load_store: bool
    fp_keys: set
    fp_width: int
    fp_touches_load_store: bool
    mem_access_size: int


def collect_instruction_pcode_summary(
    ghidra_insn: Any, default_mem_size: int = 8
) -> InstructionPcodeSummary:
    """Walk ``insn.getPcode()`` ONCE and extract every instruction-level
    signal the per-operand classifiers consume (see
    :class:`InstructionPcodeSummary`).

    Faithfully merges the legacy ``has_load_store`` scan, the
    ``_collect_fp_pcode_signature`` scan, the LOAD/STORE-routed half of
    ``_operand_touches_fp_pcode`` and the ``_infer_mem_access_size``
    scan into a single pass: per p-code op the opcode is read once;
    inputs/outputs are only touched for FLOAT_* / LOAD / STORE ops
    (exactly the ops the legacy scans touched them for).
    """
    PcodeOp = jvm_types.PcodeOp

    float_ops, int_input_ops, int_output_ops = _resolve_float_pcode_sets()
    load_op = PcodeOp.LOAD
    store_op = PcodeOp.STORE

    has_load_store = False
    fp_keys: set = set()
    fp_sizes: set[int] = set()
    # LOAD output / STORE value-input varnode keys: feed both the FP
    # LOAD/STORE attribution (key-set intersection with ``fp_keys``) and
    # the mem-access-size derivation (their sizes).
    load_store_value_keys: set = set()

    for pop in ghidra_insn.getPcode():
        opcode = pop.getOpcode()
        if opcode == load_op:
            has_load_store = True
            out = pop.getOutput()
            if out is not None:
                load_store_value_keys.add((str(out.getAddress()), int(out.getSize())))
        elif opcode == store_op:
            has_load_store = True
            inputs = pop.getInputs()
            # STORE = (space_id_const, addr_varnode, value_varnode)
            if len(inputs) >= 3:
                v = inputs[2]
                load_store_value_keys.add((str(v.getAddress()), int(v.getSize())))
        elif opcode in float_ops:
            # FP-typed sides per the int-input/int-output classification.
            int_input = opcode in int_input_ops
            int_output = opcode in int_output_ops
            if not int_input:
                for v in pop.getInputs():
                    size = int(v.getSize())
                    if size <= 0:
                        continue
                    fp_keys.add((str(v.getAddress()), size))
                    fp_sizes.add(size)
            out = pop.getOutput()
            if out is not None and not int_output:
                size = int(out.getSize())
                if size > 0:
                    fp_keys.add((str(out.getAddress()), size))
                    fp_sizes.add(size)

    nonzero_mem_sizes = {s for (_addr, s) in load_store_value_keys if s > 0}
    return InstructionPcodeSummary(
        has_load_store=has_load_store,
        fp_keys=fp_keys,
        fp_width=min(fp_sizes) if fp_sizes else 0,
        fp_touches_load_store=not fp_keys.isdisjoint(load_store_value_keys),
        mem_access_size=min(nonzero_mem_sizes) if nonzero_mem_sizes else default_mem_size,
    )


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
    PcodeOp = jvm_types.PcodeOp

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

    PcodeOp = jvm_types.PcodeOp

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
