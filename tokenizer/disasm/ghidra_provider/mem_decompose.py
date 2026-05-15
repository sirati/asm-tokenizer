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
specific writeback / pre-indexed / post-indexed flags surfaced from
``getDefaultOperandRepresentationList(op_idx)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenizer.disasm.ghidra_provider.mnemonic import _SEGMENT_REGISTERS, _RegisterMap


@dataclass
class MemoryDecomposition:
    """Typed payload returned by per-ISA memory-operand decomposers.

    Wire shape consumed by ``_GhidraMemoryOperandView._populate``. The
    eight register/scale/disp/segment fields are populated for every
    ISA; the ARM-specific addressing-mode flags
    (``writeback``/``pre_indexed``/``post_indexed``) are mutually
    exclusive within their cluster and default to False on ISAs whose
    addressing modes do not surface them.

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


def _inspect_arm_mem_addressing(
    ghidra_insn: Any, op_idx: int
) -> tuple[bool, bool, bool]:
    """Classify ARM memory-operand addressing mode from its representation list.

    Walks ``getDefaultOperandRepresentationList(op_idx)`` (a Java List of
    ``Register`` / ``Scalar`` / ``Character`` / ``String`` items) and
    returns ``(writeback, pre_indexed, post_indexed)``:

    - ``[base, #imm]`` (offset-only)        -> (False, False, False)
    - ``[base, #imm]!`` (pre-indexed wb)    -> (True,  True,  False)
    - ``[base], #imm`` (post-indexed)       -> (False, False, True)
    - ``[base, index, lsl #N]`` (shifted-index, offset-only) -> (False, False, False)

    The bracket close-position separates "inside" from "outside" the
    address brackets. A Scalar inside the brackets is a pre-indexed
    displacement (or shift amount, but the shift case still leaves
    pre/post both False because there is no separate writeback-style
    update). A Scalar outside is a post-index displacement. The
    explicit ``!`` character anywhere after the close-bracket marks
    writeback.
    """
    from ghidra.program.model.scalar import Scalar

    try:
        items = list(ghidra_insn.getDefaultOperandRepresentationList(op_idx) or ())
    except Exception:
        return (False, False, False)

    bracket_close: int | None = None
    for i, item in enumerate(items):
        if str(item) == "]":
            bracket_close = i
            break
    if bracket_close is None:
        return (False, False, False)

    writeback = any(str(x) == "!" for x in items[bracket_close + 1:])
    scalar_outside = any(isinstance(x, Scalar) for x in items[bracket_close + 1:])
    pre_indexed = writeback
    post_indexed = scalar_outside and not pre_indexed
    return (writeback, pre_indexed, post_indexed)


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
    Operands not conforming (3+ regs => reg-list) MUST have been
    classified upstream as ``OperandKind.REG_LIST`` so this function
    never sees them; the assert at the end of the object-walk enforces
    that invariant.
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
        f"x86 MEM operand should have at most 2 GP registers, got "
        f"{len(general_reg_names)}: {general_reg_names!r}. If this fires, "
        f"the operand should have classified as REG_LIST upstream."
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

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            general_reg_names.append(name)
            general_reg_ids.append(reg_map.get_id(name))
        elif isinstance(obj, Scalar):
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
    Ghidra SLEIGH spec's documented convention. Operands not conforming
    (3+ regs => reg-list) MUST have been classified upstream as
    ``OperandKind.REG_LIST`` so this function never sees them; the
    assert below enforces the invariant.
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
