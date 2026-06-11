"""Lazy decompose-callback + synthetic-spec builders for MEM / REG_LIST operands.

Owns the closure-construction concern: each builder captures the
provider-stable ``reg_map`` (and the per-operand ``ghidra_insn`` /
``op_idx`` / ``arch``) and returns a callable that, when invoked at
lazy-decomposition time, populates a passed-in operand view.

- ``decompose_mem_callback``: per-ISA MEM decomposition closure.
- ``decompose_reg_list_callback``: ARM stm/ldm-family REG_LIST closure.
- ``synthesize_disp_base_mem_spec``: fuse a (disp IMM, base REG) pair
  into one synthetic MEM operand spec (RISC-V c.sdsp SLEIGH-split case).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tokenizer.disasm.ghidra_provider import jvm_types
from tokenizer.disasm.ghidra_provider.mem_decompose import (
    _compute_arm_memory_components,
    _compute_base_disp_memory_components,
    _compute_resolved_target,
    _compute_x86_memory_components,
)
from tokenizer.disasm.ghidra_provider.mnemonic import _RegisterMap
from tokenizer.disasm.ghidra_provider.pcode_inspect import (
    register_is_addressing_mode_written,
)
from tokenizer.disasm.types import (
    Architecture,
    OperandKind,
    ShiftKind as _ShiftKind,
)

if TYPE_CHECKING:
    from tokenizer.disasm.ghidra_views import _GhidraMemoryOperandView


def decompose_mem_callback(
    reg_map: "_RegisterMap",
    ghidra_insn: Any,
    op_idx: int,
    arch: Architecture,
) -> Any:
    """Return a zero-arg callable that decomposes the MEM operand into
    a passed-in ``_GhidraMemoryOperandView``.

    Selects the per-ISA helper. The closure captures ``ghidra_insn``,
    ``op_idx``, and the provider's ``reg_map`` so the operand wrapper
    only needs to invoke the callback at lazy-decomposition time
    (first ``op.mem`` access).
    """
    if arch == Architecture.X86:
        compute = _compute_x86_memory_components
    elif arch in (Architecture.ARM32, Architecture.AARCH64):
        compute = _compute_arm_memory_components
    else:
        compute = _compute_base_disp_memory_components

    def _populate(mem_view) -> None:
        decomp = compute(ghidra_insn, op_idx, reg_map)
        # Resolved-target capture depends on the computed disp so
        # the equal-to-disp filter inside the helper can suppress
        # the trivially-redundant x86-style case where the operand
        # disp IS the absolute address.
        resolved_target = _compute_resolved_target(
            ghidra_insn, op_idx, decomp.disp
        )
        mem_view._populate(
            base_name=decomp.base_name,
            base_id=decomp.base_id,
            index_name=decomp.index_name,
            index_id=decomp.index_id,
            segment_name=decomp.segment_name,
            segment_id=decomp.segment_id,
            scale=decomp.scale,
            disp=decomp.disp,
            writeback=decomp.writeback,
            pre_indexed=decomp.pre_indexed,
            post_indexed=decomp.post_indexed,
            index_shift_kind=decomp.index_shift_kind,
            index_shift_amount=decomp.index_shift_amount,
            resolved_target=resolved_target,
        )

    return _populate


def decompose_reg_list_callback(
    reg_map: "_RegisterMap",
    ghidra_insn: Any,
    op_idx: int,
    arch: Architecture,
) -> Any:
    """Return a zero-arg callable that decomposes a REG_LIST operand
    into a passed-in ``_GhidraRegisterListView``.

    ARM stm/ldm-family operands surface in ``getOpObjects()`` as a
    flat sequence of Register objects. The Ghidra SLEIGH convention
    for these encodings is: the FIRST Register is the writeback
    target (the base register that lives *outside* the braces in
    the asm); the remaining Registers are the list members (the
    registers *inside* the braces).

    Writeback (``!``) detection: the rich-IR signal is that the
    instruction's PCode mutates the base register. The framework-
    level ``getResultObjects()`` enumerates the Registers (and
    Addresses) the instruction writes; the base register being in
    that set is the typed equivalent of Capstone's
    ``cs_insn.writeback`` flag. No string parsing of the rendered
    representation list required.
    """
    Register = jvm_types.Register

    def _populate(reg_list_view) -> None:
        try:
            objects = ghidra_insn.getOpObjects(op_idx)
        except Exception:
            objects = ()

        regs: list[tuple[str, int]] = []
        base_reg_obj: Any = None
        for obj in objects or ():
            if isinstance(obj, Register):
                name = str(obj.getName()).lower()
                regs.append((name, reg_map.get_id(name)))
                if base_reg_obj is None:
                    base_reg_obj = obj

        writeback = register_is_addressing_mode_written(ghidra_insn, base_reg_obj)

        if regs:
            base_name, base_id = regs[0]
            member_specs = regs[1:]
        else:
            # Sentinel-absent: name="" + id=0 matches _GhidraRegisterView
            # 's `_set_absent` shape (sentinels are private to
            # ghidra_views.py; using their values directly keeps the
            # cross-module surface clean).
            base_name, base_id = "", 0
            member_specs = []

        reg_list_view._advance(
            base_name=base_name,
            base_id=base_id,
            writeback=writeback,
            member_specs=member_specs,
        )

    return _populate


def is_sleigh_split_disp_base_pair(disp_spec: dict, base_spec: dict) -> bool:
    """True iff two ADJACENT operand specs are a SLEIGH-split
    disp(base) memory-operand pair: a DYNAMIC-typed IMM (the disp
    scalar) immediately followed by a REG (the base register).

    This is the pair-detection predicate for
    ``synthesize_disp_base_mem_spec`` (RISC-V compressed-instruction
    encodings like ``c.sdsp ra, 0x8(sp)``, which Ghidra's SLEIGH spec
    reports as adjacent flat operands instead of one composite memory
    operand). Adjacency is the CALLER's responsibility — the cursor's
    merge loop only asks about ``raw_specs[i]`` / ``raw_specs[i+1]``.

    Owning the predicate here (next to the fusion builder, behind the
    decode-helper facade) keeps the ``OperandType`` bitmask knowledge
    out of the view layer: the instruction cursor performs the list
    reshaping but never touches a JVM class.
    """
    OperandType = jvm_types.OperandType
    return (
        disp_spec["kind"] == OperandKind.IMM
        and bool(int(disp_spec["type_int"]) & OperandType.DYNAMIC)
        and base_spec["kind"] == OperandKind.REG
    )


def synthesize_disp_base_mem_spec(
    disp_spec: dict,
    base_spec: dict,
) -> dict:
    """Synthesize a MEM operand spec from a (disp IMM, base REG) pair.

    Used when Ghidra's SLEIGH spec splits a disp(base) memory
    operand into two adjacent flat operands - notably the RISC-V
    compressed-instruction encodings (``c.sdsp ra, 0x8(sp)`` is
    reported as 3 operands: ``ra``, ``0x8`` [DYNAMIC scalar],
    ``sp``). Caller pair-detects adjacent IMM-DYNAMIC + REG operands
    and asks us to fuse them into one synthetic MEM operand whose
    decomposition reads the captured base_name + disp directly
    (rather than going back to ``getOpObjects()`` which only sees
    the disjoint Scalar and Register on separate operand indices).

    The values are pre-captured into the closure on this call so
    subsequent calls (e.g. the next instruction) don't rebind the
    closure-bound values mid-iteration.

    ``type_int`` is the bitwise OR of the two halves so consumers
    peeking the raw OperandType bitmask see both the DYNAMIC bit
    (from the disp half) and the REGISTER bit (from the base half).
    """
    base_name = base_spec["reg_name"]
    base_id = base_spec["reg_id"]
    disp = disp_spec["imm"]
    fp_type = disp_spec["fp_type"]
    type_int = int(disp_spec["type_int"]) | int(base_spec["type_int"])

    def _decompose(view: "_GhidraMemoryOperandView") -> None:
        view._populate(
            base_name=base_name,
            base_id=base_id,
            index_name="",
            index_id=0,
            scale=1,
            disp=disp,
            segment_name="",
            segment_id=0,
        )

    # Mirror the default spec shape from ``operand_spec`` so the
    # consumer's ``_GhidraOperandView._advance(**spec)`` accepts
    # every kwarg without surprise.
    spec = dict(
        kind=OperandKind.MEM,
        reg_name="",
        reg_id=0,
        imm=0,
        size=base_spec.get("size", 0),
        fp_type=fp_type,
        type_int=type_int,
        decompose_mem=_decompose,
        shift_kind=_ShiftKind.NONE,
        shift_amount=0,
        crx_reg_name="",
        crx_reg_id=0,
        decompose_reg_list=None,
        resolved_target=None,
    )
    return spec
