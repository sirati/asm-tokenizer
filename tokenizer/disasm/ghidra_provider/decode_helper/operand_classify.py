"""Per-operand classification: build the ``_GhidraOperandView._advance`` kwargs dict.

Owns the operand-kind decision concern. ``operand_spec`` classifies one
Ghidra operand (REG / IMM / MEM / CRX / REG_LIST / OTHER) from the
``OperandType`` bitmask + ``getOpObjects()`` shape + rich-IR PCode
signals, computes per-operand size + FP type, and attaches the
lazy decompose callbacks (built in ``decompose_callbacks``) for the
MEM / REG_LIST kinds.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.ghidra_provider.decode_helper.decompose_callbacks import (
    decompose_mem_callback,
    decompose_reg_list_callback,
)
from tokenizer.disasm.ghidra_provider import jvm_types
from tokenizer.disasm.ghidra_provider.mem_decompose import (
    _compute_resolved_target,
)
from tokenizer.disasm.ghidra_provider.mnemonic import _RegisterMap
from tokenizer.disasm.ghidra_provider.pcode_inspect import (
    InstructionPcodeSummary,
    find_shift_on_register,
    operand_is_bracketed,
)
from tokenizer.disasm.ghidra_provider.prefix_build import _compute_fp_type
from tokenizer.disasm.types import (
    Architecture,
    OperandKind,
    ShiftKind as _ShiftKind,
)


def operand_spec(
    decompose_reg_map: "_RegisterMap",
    ghidra_insn: Any,
    op_idx: int,
    arch: Architecture,
    base_mnemonic: str,
    reg_map: "_RegisterMap",
    *,
    pcode_summary: InstructionPcodeSummary,
) -> dict:
    """Return a kwargs dict for ``_GhidraOperandView._advance``.

    Classifies the operand kind (REG/IMM/MEM/CRX/OTHER) from
    Ghidra's ``OperandType`` bitmask + ``getOpObjects()`` shape,
    computes per-operand size + FP type, and produces the
    decompose-mem callback when the operand is MEM. Non-MEM
    operands carry ``decompose_mem=None`` so the operand wrapper
    skips lazy MEM decomposition.

    ``reg_map`` is passed explicitly (not read from the helper's
    cached map) so the spec composes cleanly with the views'
    constructor wiring; in practice they are the same object.
    ``decompose_reg_map`` is the helper-cached map threaded into the
    lazy decompose callbacks (the original ``self._reg_map``).

    ``pcode_summary`` is the per-instruction one-pass PCode signal
    bundle (LOAD/STORE presence, FLOAT_* signature, mem-access size),
    computed ONCE by the instruction-view cursor in ``_advance`` and
    threaded through here to avoid the per-operand PCode re-walks the
    legacy decode path performed (has_load_store + FP scan + mem-size
    scan each re-iterated ``getPcode()`` per operand).
    """
    Address = jvm_types.Address
    OperandType = jvm_types.OperandType
    Register = jvm_types.Register
    Scalar = jvm_types.Scalar

    try:
        objects = ghidra_insn.getOpObjects(op_idx)
    except Exception:
        objects = ()
    try:
        op_type = ghidra_insn.getOperandType(op_idx)
    except Exception:
        op_type = 0

    # Pre-collect Register objects: used both by the is_memory check
    # below (DYNAMIC-shaped memory operands MUST involve at least one
    # base/index register; the absolute-addressed x86 case below is
    # the explicit exception) and the ARM/AArch64 reg-list classifier
    # below.
    register_objs = [o for o in objects if isinstance(o, Register)]

    # DYNAMIC-shaped memory operands MUST involve at least one base/
    # index register. Without that, Ghidra's DYNAMIC bit on a pure-
    # scalar operand (e.g. RISC-V c.addi's immediate, or c.sdsp's
    # disp scalar that SLEIGH split off from its base register) is
    # misleading and produces a degenerate base-less mem-bracket
    # rendering. The one shape this rule does NOT apply to is the
    # x86 absolute-addressed memory operand (``lea rax, [0x10D7C0]``
    # / ``mov eax, [0x12345678]``): Ghidra surfaces ONLY an Address/
    # Scalar (no Register) on those, with op_type = ADDRESS|SCALAR
    # (no REGISTER, no DYNAMIC, no CODE bits) and brackets in the
    # rendered representation. That shape escapes the register gate
    # via the dedicated ``absolute_addressed_no_register_mem`` branch
    # below; the rest of the is_memory clauses keep the original
    # ``MUST involve at least one base/index register`` invariant.
    #
    # ARM pre-indexed-with-writeback (``stp x29, x30, [sp, #-48]!``):
    # Ghidra reports ``op_type = REGISTER|ADDRESS`` (no DYNAMIC) and
    # surfaces the displacement as a ``Scalar`` inside ``getOpObjects``.
    # The disambiguator vs a bare register operand is the presence of
    # at least one Scalar (a plain REG operand carries no Scalar).
    #
    # The is-memory question has TWO orthogonal axes:
    #
    # 1. SEMANTIC: does this instruction access memory at all? The
    #    rich-IR signal is whether ``getPcode()`` contains a LOAD or
    #    STORE op. We use this for ARM/AArch64 to reject shifted-
    #    register operands like ``sbc r1, r1, r1, lsl #N`` operand
    #    2: they have ``OperandType.DYNAMIC`` (same as memory
    #    operands) but the instruction is pure-arithmetic with no
    #    LOAD/STORE PCode. x86 LEA legitimately is bracketed
    #    syntactically but does not LOAD, so we don't gate x86 on
    #    has_load_store.
    #
    # 2. SYNTACTIC: was this specific operand written with bracket
    #    framing? The rich-typed signal is the presence of the
    #    per-ISA bracket-open ``java.lang.Character`` item in
    #    ``getDefaultOperandRepresentationList``. This rejects
    #    operands like x86 ``rep stosb rdi`` (RDI is rendered
    #    without brackets despite being the implicit memory
    #    pointer) and arm64 ``strh wzr, [...]`` (WZR — the zero
    #    register, semantically a constant source — is rendered
    #    without brackets despite carrying the DYNAMIC bit because
    #    Ghidra surfaces WZR as a runtime-valued operand). No
    #    OperandType bit reliably discriminates these from real
    #    bracketed-mem operands; ``operand_is_bracketed`` reads the
    #    rich-typed Character via isinstance + charValue without
    #    string parsing.
    scalar_in_objects = any(isinstance(o, Scalar) for o in objects or ())

    arm_family = arch in (Architecture.ARM32, Architecture.AARCH64)
    dynamic_admits_memory = (
        bool(op_type & OperandType.DYNAMIC)
        and (not arm_family or pcode_summary.has_load_store)
    )

    # x86 absolute-addressed memory operand: ``lea rax, [0x10D7C0]``
    # / ``mov eax, [0x12345678]``. Ghidra's getOpObjects surfaces only
    # a Scalar/Address (no Register), so ``register_objs`` is empty;
    # the op_type is ADDRESS|SCALAR with no REGISTER/CODE/DYNAMIC.
    # This shape is structurally distinct from every register-bearing
    # mem-operand kind and from every non-mem kind (REGISTER/CODE
    # bits exclude them), so admitting it without the register gate
    # is safe. The CODE-bit exclusion keeps absolute-target branches
    # (``jmp 0x10D7C0`` etc.) routing through their own kind. This
    # branch is the explicit exception to the "MUST involve at least
    # one base/index register" rule documented above.
    #
    # NOTE on the bracket factor: every memory shape below ALSO requires
    # the syntactic bracket marker. The bracket check
    # (``operand_is_bracketed``) reads the operand's representation list
    # — a JVM round-trip per operand — so it is FACTORED OUT of the two
    # original conjunctions and evaluated LAST, only when one of the
    # cheap structural shapes matched. Boolean identity (brackets is a
    # pure predicate):
    #     (br ∧ abs_shape) ∨ (regs ∧ br ∧ rest)
    #   = br ∧ (abs_shape ∨ (regs ∧ rest))
    # REG / IMM operands short-circuit on the structural side and never
    # pay the representation-list fetch.
    absolute_addressed_no_register_mem = (
        bool(op_type & OperandType.ADDRESS)
        and bool(op_type & OperandType.SCALAR)
        and not (op_type & (OperandType.REGISTER | OperandType.CODE | OperandType.DYNAMIC))
    )

    memory_shape = absolute_addressed_no_register_mem or (
        bool(register_objs) and (
            dynamic_admits_memory
            or bool(op_type & OperandType.INDIRECT)
            or (
                bool(op_type & OperandType.ADDRESS)
                and bool(op_type & OperandType.SCALAR)
                and not (op_type & (OperandType.REGISTER | OperandType.CODE))
            )
            or (
                bool(op_type & OperandType.REGISTER)
                and bool(op_type & OperandType.ADDRESS)
                and not (op_type & OperandType.CODE)
                and scalar_in_objects
            )
            or (
                # ARM32 pre-indexed STORE: ``strb r9, [r8, #0x1]!`` op_type
                # is REGISTER-only (no ADDRESS, no DYNAMIC; Ghidra's
                # arm32 SLEIGH spec is asymmetric vs pre-indexed LOAD
                # which DOES carry ADDRESS). The disambiguator vs a
                # plain REGISTER operand is the Scalar in objects (the
                # pre-disp) plus the instruction-level rich-IR signal
                # that it accesses memory (LOAD/STORE in PCode). The
                # trailing ``operand_is_bracketed`` gate keeps this from
                # claiming non-bracketed register operands.
                bool(op_type & OperandType.REGISTER)
                and not (op_type & OperandType.CODE)
                and scalar_in_objects
                and pcode_summary.has_load_store
            )
        )
    )

    is_memory = memory_shape and operand_is_bracketed(ghidra_insn, op_idx, arch)

    fp_type = _compute_fp_type(objects, op_type, arch, base_mnemonic, pcode_summary)

    # Default spec - filled per kind below.
    spec = dict(
        kind=OperandKind.INVALID,
        reg_name="",
        reg_id=0,
        imm=0,
        size=0,
        fp_type=fp_type,
        type_int=int(op_type),
        decompose_mem=None,
        shift_kind=_ShiftKind.NONE,
        shift_amount=0,
        crx_reg_name="",
        crx_reg_id=0,
        decompose_reg_list=None,
        resolved_target=None,
    )

    if not objects:
        return spec

    # Reg-list classification (ARM stm/ldm/push/pop/vpush/vpop/vstm/
    # vldm + AArch64 stp/ldp + VFP vstm/vldm family). Ghidra's
    # SLEIGH spec emits a flat sequence of Register objects for
    # reg-list operands on those ISAs; >=3 Registers on a single
    # operand is a reliable signal there.
    #
    # The "reg-list" concept does NOT apply to x86/MIPS/PPC/RISC-V:
    # x86 in particular legitimately surfaces 3 Registers on a
    # segment-prefixed base+index memory operand (e.g.
    # ``NOP word ptr CS:[RAX + RAX*0x1]`` exposes CS + RAX + RAX
    # in getOpObjects). Gating the classifier on ARM32/AArch64
    # keeps those x86 MEM operands flowing into
    # ``_compute_x86_memory_components`` (which already separates
    # segment from base/index by name).
    if arch in (Architecture.ARM32, Architecture.AARCH64) and len(register_objs) >= 3:
        spec["kind"] = OperandKind.REG_LIST
        spec["decompose_reg_list"] = decompose_reg_list_callback(
            decompose_reg_map, ghidra_insn, op_idx, arch
        )
        return spec

    if is_memory:
        spec["kind"] = OperandKind.MEM
        # Memory access size: derived from SLEIGH-emitted PCode
        # LOAD/STORE varnode sizes (see the rationale on
        # ``InstructionPcodeSummary.mem_access_size``), precomputed
        # once per instruction by the one-pass PCode walk.
        spec["size"] = pcode_summary.mem_access_size
        spec["decompose_mem"] = decompose_mem_callback(
            decompose_reg_map, ghidra_insn, op_idx, arch
        )
        return spec

    first = objects[0]
    if isinstance(first, Register):
        name = str(first.getName()).lower()
        spec["kind"] = OperandKind.REG
        spec["reg_name"] = name
        spec["reg_id"] = reg_map.get_id(name)
        try:
            spec["size"] = int(first.getMinimumByteSize())
        except Exception:
            spec["size"] = 0
        # ARM shifted-register operand (data-processing barrel-shifter
        # form, e.g. ``add r6, r4, r1, lsl #0x2``). On ARM/AArch64 a
        # REG operand whose object list also contains a Scalar is the
        # ``Rn, <shift> #imm`` form; the typed shift kind comes from
        # the rich-IR PCode (``INT_LEFT`` / ``INT_RIGHT`` /
        # ``INT_SRIGHT`` on the operand's register varnode) and the
        # amount from that op's constant second input.
        if (
            arch in (Architecture.ARM32, Architecture.AARCH64)
            and scalar_in_objects
        ):
            kind, amount = find_shift_on_register(ghidra_insn, first)
            spec["shift_kind"] = kind
            spec["shift_amount"] = amount
        # PC-relative literal-pool loads (ARM ``ldr r4, [pc, #0x44]``)
        # surface the analyzer-lifted data-pointer on the destination
        # REG operand, NOT on the MEM operand (the MEM is just the
        # literal-pool slot). Reuse the same helper the MEM path
        # uses; ``disp=0`` here disables the equal-to-disp filter
        # (a REG operand has no disp, so the filter is moot).
        #
        # Capture UNCONDITIONALLY: the keep/drop policy now lives
        # downstream (``tokenizer/disasm/resolved_target_policy.py``)
        # so the decode helper stays a pure data extractor. The
        # legacy gate on ``instruction_has_mem_access`` was a
        # csel-class suppression heuristic; the policy module
        # subsumes it with a layered design that admits high-
        # confidence address-kind matches (STRING / PLT_FUNCTION /
        # LOCAL_FUNCTION / CODE_PTR_TABLE_SLOT) AND intra-function
        # block targets even on pure-register instructions, while
        # preserving the original suppression for low-confidence
        # kinds (RO_DATA_PTR / UNKNOWN). It additionally opens a
        # per-ISA pair-terminal allow-list (e.g. arm32 ``movt``
        # holding the high half of a ``movw``+``movt`` string-
        # pointer build) which the legacy gate dropped because the
        # high-half terminal carries no LOAD/STORE PCode.
        spec["resolved_target"] = _compute_resolved_target(
            ghidra_insn, op_idx, disp=0
        )
        return spec
    if isinstance(first, Scalar):
        spec["kind"] = OperandKind.IMM
        spec["imm"] = int(first.getSignedValue())
        try:
            spec["size"] = int(first.bitLength()) // 8
        except Exception:
            spec["size"] = 0
        return spec
    if isinstance(first, Address):
        spec["kind"] = OperandKind.IMM
        spec["imm"] = int(first.getOffset())
        return spec

    # Unknown op kind - treat as OTHER passthrough so consumers that
    # gate on ``op.kind == OperandKind.OTHER`` can route correctly.
    spec["kind"] = OperandKind.OTHER
    return spec
