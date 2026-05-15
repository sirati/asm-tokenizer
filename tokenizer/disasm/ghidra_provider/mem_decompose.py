"""Per-ISA memory-operand decomposition helpers.

Each helper inspects ``ghidra_insn.getOpObjects(op_idx)`` and computes the
decomposed memory-operand fields for a single MEM operand. Helpers are
PURE w.r.t. Ghidra Java state - they read but do not mutate. Each
returns a ``MemoryDecomposition`` dataclass suitable for
``_GhidraMemoryOperandView._populate``.

The x86 path is faithfully ported from the legacy Ghidra-side memory
tokenizer at ``tokenizer/arch/x86/ghidra/operands.py`` (the
``_classify_objects + assign_base_index_scale_disp`` block of
``tokenize_operand_memory_ghidra``). The ARM and base+disp paths
expose the same (base, index, scale, disp, segment) wire-shape that
the typed ``MemoryOperandView`` exposes to consumers, plus the ARM-
specific writeback / pre-indexed / post-indexed flags derived from
PCode rich IR via ``pcode_inspect.classify_memory_addressing`` and
the shifted-index modifier via ``pcode_inspect.find_shift_on_register``.

In addition, this module surfaces ``resolved_target`` -- the address
that Ghidra's analyzer has resolved this memory operand to point at,
when it differs from the operand's literal displacement. This is the
hook that lets the v2 precedence classifier reach the right metadata
for PC-relative loads on ARM (``ldrb r3, [r4, #0]`` where r4 was
loaded from a literal-pool slot resolving to a string in .rodata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tokenizer.disasm.ghidra_provider.mnemonic import _SEGMENT_REGISTERS, _RegisterMap
from tokenizer.disasm.ghidra_provider.pcode_inspect import (
    classify_memory_addressing,
    find_shift_on_register,
)
from tokenizer.disasm.types import ShiftKind


@dataclass
class MemoryDecomposition:
    """Typed payload returned by per-ISA memory-operand decomposers.

    Wire shape consumed by ``_GhidraMemoryOperandView._populate``. The
    eight register/scale/disp/segment fields are populated for every
    ISA; the ARM-specific addressing-mode flags
    (``writeback``/``pre_indexed``/``post_indexed``) are mutually
    exclusive within their cluster and default to False on ISAs whose
    addressing modes do not surface them. ``index_shift_kind`` /
    ``index_shift_amount`` carry the shift modifier on a shifted-index
    addressing mode like ``[base, index, lsl #N]``; the
    ``ShiftKind.NONE`` default applies on every other addressing mode
    and on non-ARM ISAs.

    Pre-indexed (``[base, #imm]!``): base register is updated by the
    offset BEFORE the memory access (``writeback`` is also True).
    Post-indexed (``[base], #imm``): memory access uses the un-updated
    base; ``writeback`` is implicit so the base is updated AFTER the
    access. Plain offset addressing (``[base, #imm]``) leaves all three
    flags False.
    """

    base_name: str = ""
    base_id: int = 0
    index_name: str = ""
    index_id: int = 0
    scale: int = 1
    disp: int = 0
    segment_name: str = ""
    segment_id: int = 0
    writeback: bool = False
    pre_indexed: bool = False
    post_indexed: bool = False
    index_shift_kind: ShiftKind = ShiftKind.NONE
    index_shift_amount: int = 0
    resolved_target: Optional[int] = None


def _compute_resolved_target(
    ghidra_insn: Any, op_idx: int, disp: int
) -> Optional[int]:
    """Return the analyzer-resolved data target for memory operand ``op_idx``.

    Ghidra's CFG analyzer recovers value-flow information on memory
    accesses whose base register was loaded from a known-pointer slot
    (typical for ARM ``ldrb rN, [rM, #0]`` patterns where rM came from
    a literal pool). The resolved target is surfaced on the operand
    via ``getOperandReferences(op_idx)`` as a primary data reference
    whose target address differs from the operand's literal disp.

    When the data reference points to the SAME address as the operand
    disp (the common x86 ``lea rax, [0x...]`` case), there is no
    resolved-target distinction to surface -- the v2 classifier will
    already lookup the disp directly. Return ``None`` in that case so
    consumers know to keep the disp-based lookup path.

    Returns the resolved address as a signed Python int, or ``None``
    when no qualifying ref is present.
    """
    try:
        refs = ghidra_insn.getOperandReferences(op_idx)
    except Exception:
        return None
    for ref in refs or ():
        try:
            ref_type = ref.getReferenceType()
            if not ref_type.isData():
                continue
            # Exclude stack-frame analyzer references: Ghidra surfaces
            # stack-frame slot positions as data refs in a separate
            # ``stack`` address space. The numeric offset is a frame-
            # relative slot identifier, NOT a loadable data address;
            # feeding it through ``lookup()`` produces meaningless
            # metadata. ``Reference.isStackReference()`` is the
            # authoritative discriminator over the also-applicable
            # ``isMemoryReference()`` check.
            if ref.isStackReference():
                continue
            if not ref.isMemoryReference():
                continue
            target = int(ref.getToAddress().getOffset())
        except Exception:
            continue
        # Exclude the same-as-disp case so x86 ``lea`` operands (where
        # disp itself is the absolute address) do not redundantly
        # surface a "resolved" target identical to disp; the disp-based
        # ``lookup()`` already classifies them correctly.
        if target == disp:
            continue
        return target
    return None


def _inspect_arm_index_shift(
    ghidra_insn: Any, op_idx: int
) -> tuple[ShiftKind, int]:
    """Detect the shift modifier on a shifted-index ARM mem operand.

    ARM allows the index register to be shifted as part of the address
    computation, e.g. ``ldrb r5, [r7, r5, lsl #0x1]``. The shift is
    surfaced richly in Ghidra's PCode IR — the index register flows into
    a typed ``INT_LEFT``/``INT_RIGHT``/``INT_SRIGHT`` op whose constant
    second input IS the shift amount.

    The Ghidra SLEIGH operand convention places the index as the SECOND
    Register in ``getOpObjects(op_idx)``; we delegate to
    ``find_shift_on_register`` which walks PCode for a shift op consuming
    (possibly via COPY chains) the index register's varnode.

    Returns ``(kind, amount)``; ``(ShiftKind.NONE, 0)`` when no shift
    modifier is present.
    """
    from ghidra.program.model.lang import Register

    try:
        objects = ghidra_insn.getOpObjects(op_idx)
    except Exception:
        return (ShiftKind.NONE, 0)
    regs = [o for o in (objects or ()) if isinstance(o, Register)]
    if len(regs) < 2:
        return (ShiftKind.NONE, 0)
    index_reg = regs[1]
    return find_shift_on_register(ghidra_insn, index_reg)


def _inspect_arm_mem_addressing(
    ghidra_insn: Any, op_idx: int
) -> tuple[bool, bool, bool]:
    """Classify ARM memory-operand addressing mode from PCode IR.

    Returns ``(writeback, pre_indexed, post_indexed)``:

    - ``[base, #imm]`` (offset-only)        -> (False, False, False)
    - ``[base, #imm]!`` (pre-indexed wb)    -> (True,  True,  False)
    - ``[base], #imm`` (post-indexed)       -> (False, False, True)
    - ``[base, index, lsl #N]`` (shifted-index, offset-only) -> (False, False, False)

    Delegates to ``classify_memory_addressing`` which derives the answer
    from PCode rich IR: ``getResultObjects()`` membership for writeback,
    LOAD/STORE address-varnode shape (base register directly vs uniq
    COPYed from base) for pre vs post.

    The base register is the FIRST Register in ``getOpObjects(op_idx)``
    per the Ghidra SLEIGH operand convention.
    """
    from ghidra.program.model.lang import Register

    try:
        objects = ghidra_insn.getOpObjects(op_idx)
    except Exception:
        return (False, False, False)
    regs = [o for o in (objects or ()) if isinstance(o, Register)]
    if not regs:
        return (False, False, False)
    base_reg = regs[0]
    return classify_memory_addressing(ghidra_insn, base_reg)


def _infer_mem_access_size(ghidra_insn: Any, op_idx: int, default: int = 8) -> int:
    """Return the memory-access size in bytes for operand ``op_idx``.

    Reads SLEIGH-emitted PCode ``LOAD`` output-varnode size and ``STORE``
    value-input size. This is the only reliable oracle - sibling-
    register width conflates address-computation regs (pointer-width,
    e.g. r14 = 8B on x64) with value regs (actual memory-access width),
    breaking 0x66 operand-size-override (``or word ptr [r14+...], ax``
    seen as qword) and MOVZX/MOVSX byte/word -> wider destination.

    ``op_idx`` is currently unused (x86 mem-operand instructions have
    exactly one LOAD/STORE per operand), but kept in the signature to
    future-proof against multi-mem-operand instructions (string ops)
    where the access size could differ per operand.
    """
    from ghidra.program.model.pcode import PcodeOp

    sizes: set[int] = set()
    for pop in ghidra_insn.getPcode():
        opcode = pop.getOpcode()
        if opcode == PcodeOp.LOAD:
            out = pop.getOutput()
            if out is not None:
                sizes.add(int(out.getSize()))
        elif opcode == PcodeOp.STORE:
            inputs = pop.getInputs()
            # STORE: (space_id_const, addr_varnode, value_varnode)
            if len(inputs) >= 3:
                sizes.add(int(inputs[2].getSize()))
    nonzero = {s for s in sizes if s > 0}
    if not nonzero:
        return default
    # x86 mem-operand instructions have exactly one LOAD/STORE per
    # operand. When multiple distinct sizes appear (rare; string ops),
    # the value-side width is the smallest non-zero.
    return min(nonzero)


def _compute_x86_memory_components(
    ghidra_insn: Any,
    op_idx: int,
    reg_map: "_RegisterMap",
) -> MemoryDecomposition:
    """Decompose an x86/x64 MEM operand from raw Ghidra objects.

    Returns a ``MemoryDecomposition`` populating
    ``base_name/base_id/index_name/index_id/scale/disp/segment_name/segment_id``;
    the ARM-specific addressing-mode flags stay at their dataclass
    defaults (all False) because x86 has no pre-/post-indexed modes.
    Empty name + id=0 means the slot is absent.

    Object-count rules from getOpObjects() (faithful port):
        2 general regs   -> first Scalar = scale, remaining Scalars = disp
        0-1 general regs -> all Scalars = disp
        Address objects  -> disp

    The first GP Register in ``getOpObjects()`` is the base; the second is
    the index. This is the Ghidra SLEIGH spec's documented convention.
    Segment-override registers (CS/SS/DS/ES/FS/GS) are partitioned out
    by name into their own slot and do NOT count against the 2-GP-register
    invariant — a segment-prefixed base+index operand legitimately
    carries 3 Registers in ``getOpObjects()`` (e.g.
    ``NOP word ptr CS:[RAX + RAX*0x1]``).
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    objects = ghidra_insn.getOpObjects(op_idx)

    segment_reg_name: str = ""
    segment_reg_id: int = 0
    general_reg_names: list[str] = []
    general_reg_ids: list[int] = []
    scalars: list[int] = []
    signed_scalars: list[int] = []
    disp: int = 0

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            rid = reg_map.get_id(name)
            if name in _SEGMENT_REGISTERS:
                segment_reg_name = name
                segment_reg_id = rid
            else:
                general_reg_names.append(name)
                general_reg_ids.append(rid)
        elif isinstance(obj, Scalar):
            scalars.append(int(obj.getValue()))
            signed_scalars.append(int(obj.getSignedValue()))
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    assert len(general_reg_names) <= 2, (
        f"x86 MEM operand should have at most 2 GP registers (base + "
        f"index), got {len(general_reg_names)}: {general_reg_names!r}. "
        f"Segment-override registers are partitioned out by name; if this "
        f"fires the operand carries an unexpected third GP register."
    )
    assert len(scalars) <= 2, (
        f"x86 MEM operand should have at most 2 Scalar slots (scale + "
        f"disp), got {len(scalars)}: {scalars!r}"
    )

    base_name = general_reg_names[0] if general_reg_names else ""
    base_id = general_reg_ids[0] if general_reg_ids else 0
    index_name = general_reg_names[1] if len(general_reg_names) >= 2 else ""
    index_id = general_reg_ids[1] if len(general_reg_ids) >= 2 else 0
    scale: int = 1

    if len(general_reg_ids) >= 2 and scalars:
        scale = scalars[0]
        if len(scalars) > 1:
            disp = signed_scalars[1]
    elif len(general_reg_ids) <= 1 and scalars:
        disp = signed_scalars[0]

    return MemoryDecomposition(
        base_name=base_name,
        base_id=base_id,
        index_name=index_name,
        index_id=index_id,
        scale=scale,
        disp=disp,
        segment_name=segment_reg_name,
        segment_id=segment_reg_id,
    )


def _compute_arm_memory_components(
    ghidra_insn: Any,
    op_idx: int,
    reg_map: "_RegisterMap",
) -> MemoryDecomposition:
    """Decompose an ARM MEM operand from raw Ghidra objects.

    ARM addressing modes use base + optional index register + optional
    displacement (no scale, no segment). Returns a ``MemoryDecomposition``
    with ``scale=1`` fixed, segment slots absent, and the addressing-mode
    flags (``writeback``/``pre_indexed``/``post_indexed``) derived from
    ``_inspect_arm_mem_addressing`` so the typed view exposes whether
    the base is auto-updated by the displacement.

    First general-purpose Register -> base; second -> index; first
    Scalar/Address -> disp. This is the Ghidra SLEIGH spec's documented
    convention. Operands not conforming (3+ regs => reg-list)
    MUST have been classified upstream as ``OperandKind.REG_LIST`` so
    this function never sees them (stm/ldm/push/pop/vpush/vpop/vstm/vldm
    family); the assert at the end of the object-walk enforces that
    invariant.
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    objects = ghidra_insn.getOpObjects(op_idx)

    general_reg_names: list[str] = []
    general_reg_ids: list[int] = []
    disp: int = 0

    # Detect a shifted-index addressing mode BEFORE we walk the Scalar
    # slots, so the Scalar that lives in ``getOpObjects()`` can be
    # correctly attributed to the index-shift amount instead of being
    # spuriously taken as a displacement. Without this guard,
    # ``ldrb r5, [r7, r5, lsl #0x1]`` reads the ``#0x1`` as ``disp=1``
    # and emits a bogus ``[r7+r5+1]`` rendering.
    index_shift_kind, index_shift_amount = _inspect_arm_index_shift(
        ghidra_insn, op_idx
    )
    has_index_shift = index_shift_kind != ShiftKind.NONE

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            general_reg_names.append(name)
            general_reg_ids.append(reg_map.get_id(name))
        elif isinstance(obj, Scalar):
            # In a shifted-index addressing mode the Scalar is the
            # shift amount, NOT a displacement; ``index_shift_amount``
            # already captured it from the representation list.
            if not has_index_shift:
                disp = int(obj.getSignedValue())
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    assert len(general_reg_names) <= 2, (
        f"ARM MEM operand should have at most 2 GP registers, got "
        f"{len(general_reg_names)}: {general_reg_names!r}. If this fires, "
        f"the operand should have classified as REG_LIST upstream "
        f"(stm/ldm/push/pop/vpush/vpop/vstm/vldm family)."
    )

    base_name = general_reg_names[0] if general_reg_names else ""
    base_id = general_reg_ids[0] if general_reg_ids else 0
    index_name = general_reg_names[1] if len(general_reg_names) >= 2 else ""
    index_id = general_reg_ids[1] if len(general_reg_ids) >= 2 else 0

    writeback, pre_indexed, post_indexed = _inspect_arm_mem_addressing(
        ghidra_insn, op_idx
    )

    return MemoryDecomposition(
        base_name=base_name,
        base_id=base_id,
        index_name=index_name,
        index_id=index_id,
        scale=1,
        disp=disp,
        writeback=writeback,
        pre_indexed=pre_indexed,
        post_indexed=post_indexed,
        index_shift_kind=index_shift_kind,
        index_shift_amount=index_shift_amount,
    )


def _compute_base_disp_memory_components(
    ghidra_insn: Any,
    op_idx: int,
    reg_map: "_RegisterMap",
) -> MemoryDecomposition:
    """Decompose a base+disp MEM operand (MIPS/PPC/RISC-V).

    These ISAs only ever have one base register + one displacement; no
    index, no scale, no segment. Returns a ``MemoryDecomposition`` with
    the index slot absent, scale=1, and segment slots absent.

    The first GP Register in ``getOpObjects()`` is the base. This is the
    Ghidra SLEIGH spec's documented convention. These ISAs have no
    reg-list family, so a base+disp MEM operand here always carries at
    most 1 Register; the assert below enforces the invariant (and would
    catch a Ghidra SLEIGH-spec quirk on a future ISA addition).
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    objects = ghidra_insn.getOpObjects(op_idx)

    base_name: str = ""
    base_id: int = 0
    disp: int = 0
    general_regs: list[str] = []

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            general_regs.append(name)
            if base_name == "":
                base_name = name
                base_id = reg_map.get_id(base_name)
        elif isinstance(obj, Scalar):
            disp = int(obj.getSignedValue())
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    assert len(general_regs) <= 1, (
        f"base+disp MEM operand should have at most 1 GP register, "
        f"got {len(general_regs)}: {general_regs!r}"
    )

    return MemoryDecomposition(
        base_name=base_name,
        base_id=base_id,
        scale=1,
        disp=disp,
    )
