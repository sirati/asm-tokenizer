"""Provider-owned decode helper injected into ``_GhidraInstructionView`` wrappers.

Owns ``_GhidraDecodeHelper``: centralizes
- mnemonic split + alias canonicalization
- architecture detection (cached per program)
- per-operand FP-type computation
- per-instruction typed-prefix list build
- per-operand decompose-mem / decompose-reg-list callback construction
- per-operand spec dict (kwargs ready for ``_GhidraOperandView._advance``)
- synthesized disp+base MEM spec for SLEIGH-split pairs (RISC-V c.sdsp).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from tokenizer.disasm.ghidra_provider.mem_decompose import (
    _compute_arm_memory_components,
    _compute_base_disp_memory_components,
    _compute_resolved_target,
    _compute_x86_memory_components,
    _infer_mem_access_size,
)
from tokenizer.disasm.ghidra_provider.mnemonic import (
    _GHIDRA_MNEMONIC_ALIASES,
    _RegisterMap,
    _split_ghidra_mnemonic,
    _strip_arm_cc_suffix,
)
from tokenizer.disasm.ghidra_provider.prefix_build import (
    _compute_fp_type,
    _ghidra_processor_to_architecture,
    _prefix_builder_for_arch,
)
from tokenizer.disasm.types import (
    Architecture,
    ArmConditionCode,
    FpType,
    OperandKind,
)

if TYPE_CHECKING:
    from tokenizer.disasm.ghidra_views import _GhidraMemoryOperandView


# ---------------------------------------------------------------------------
# Decode helper - injected into _GhidraInstructionView wrappers
# ---------------------------------------------------------------------------
class _GhidraDecodeHelper:
    """Provider-owned helper exposing the per-instruction decode surface
    the owned-view wrappers (``ghidra_views.py``) need.

    Construction: one instance per ``GhidraDisassemblyProvider`` (program
    + reg_map are stable for the program's lifetime). The helper is
    passed to each ``_GhidraFunctionView`` -> ``_GhidraBlockView`` ->
    ``_GhidraInstructionView`` constructor so the view chain stays
    self-contained without cross-importing the provider.

    The helper centralizes:
      - mnemonic split + alias canonicalization
      - architecture detection (cached per program)
      - per-operand FP-type computation
      - per-instruction typed-prefix list build
      - per-operand decompose-mem callback construction (lazy: returns a
        zero-arg callable that, when invoked, populates a passed
        ``_GhidraMemoryOperandView``)
      - per-operand spec dict (kwargs ready for
        ``_GhidraOperandView._advance``)
    """

    __slots__ = ("_program", "_reg_map", "_arch")

    def __init__(self, program: Any, reg_map: "_RegisterMap") -> None:
        self._program = program
        self._reg_map = reg_map
        self._arch: Architecture = _ghidra_processor_to_architecture(program)

    @property
    def arch(self) -> Architecture:
        return self._arch

    def split_mnemonic(self, raw: str) -> tuple[str, str | None, int | None]:
        return _split_ghidra_mnemonic(raw)

    def strip_arm_cc(
        self,
        mnemonic: str,
    ) -> tuple[str, ArmConditionCode | None]:
        """Strip the Ghidra ARM/AArch64 cc-suffix from ``mnemonic``."""
        return _strip_arm_cc_suffix(mnemonic)

    def alias_mnemonic(self, base: str) -> str:
        # Ghidra's MIPS SLEIGH spec emits `_sra` / `_li` / ... for the
        # delay-slot variants of `sra` / `li` / .... The underscore is a
        # Ghidra display convention, not a real MIPS-ISA mnemonic
        # distinction (gas/objdump/Capstone all just write `sra`).
        # Delay-slot membership is already encoded positionally by the
        # token sequence, so the duplicate vocab entry is noise.
        if self._arch == Architecture.MIPS and base.startswith("_") and len(base) > 1:
            base = base[1:]
        return _GHIDRA_MNEMONIC_ALIASES.get(base, base)

    def architecture(self, _program: Any) -> Architecture:
        return self._arch

    def compute_fp_type(
        self,
        ghidra_insn: Any,
        operand_index: int,
        arch: Architecture,
        base_mnemonic: str,
    ) -> Optional[FpType]:
        return _compute_fp_type(ghidra_insn, operand_index, arch, base_mnemonic)

    def build_prefixes(self, ghidra_insn: Any, arch: Architecture) -> list[Any]:
        return _prefix_builder_for_arch(arch)(ghidra_insn)

    def _decompose_mem_callback(
        self,
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
        reg_map = self._reg_map
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

    def _decompose_reg_list_callback(
        self,
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

        Writeback (`!`) detection: Ghidra's ``OperandType`` bitmask
        does not carry a documented writeback bit. The flag IS surfaced
        through ``getDefaultOperandRepresentationList(op_idx)``, which
        returns the operand's formatted components as a Java List of
        ``Register`` / ``Scalar`` / ``Character`` / ``String`` items;
        ARM stmdb/ldmia/push/pop with writeback include a
        ``Character('!')`` entry immediately after the base register.
        We scan that list for any ``!`` token to derive ``writeback``.
        This avoids parsing the raw mnemonic string (which is just
        ``stmdb`` either way -- the ``!`` is positional on the operand)
        and avoids PCode self-assignment inspection.
        """
        from ghidra.program.model.lang import Register

        reg_map = self._reg_map

        def _populate(reg_list_view) -> None:
            try:
                objects = ghidra_insn.getOpObjects(op_idx)
            except Exception:
                objects = ()

            try:
                repr_list = ghidra_insn.getDefaultOperandRepresentationList(op_idx)
                writeback = any(str(item) == "!" for item in repr_list or ())
            except Exception:
                writeback = False

            regs: list[tuple[str, int]] = []
            for obj in objects or ():
                if isinstance(obj, Register):
                    name = str(obj.getName()).lower()
                    regs.append((name, reg_map.get_id(name)))

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

    def synthesize_disp_base_mem_spec(
        self,
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
        from tokenizer.disasm.types import ShiftKind as _ShiftKind

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
        )
        return spec

    def operand_spec(
        self,
        ghidra_insn: Any,
        op_idx: int,
        arch: Architecture,
        base_mnemonic: str,
        reg_map: "_RegisterMap",
    ) -> dict:
        """Return a kwargs dict for ``_GhidraOperandView._advance``.

        Classifies the operand kind (REG/IMM/MEM/CRX/OTHER) from
        Ghidra's ``OperandType`` bitmask + ``getOpObjects()`` shape,
        computes per-operand size + FP type, and produces the
        decompose-mem callback when the operand is MEM. Non-MEM
        operands carry ``decompose_mem=None`` so the operand wrapper
        skips lazy MEM decomposition.

        ``reg_map`` is passed explicitly (not read from ``self._reg_map``)
        so the spec composes cleanly with the views' constructor wiring;
        in practice they are the same object.
        """
        from ghidra.program.model.address import Address
        from ghidra.program.model.lang import OperandType, Register
        from ghidra.program.model.scalar import Scalar
        from tokenizer.disasm.types import ShiftKind as _ShiftKind

        try:
            objects = ghidra_insn.getOpObjects(op_idx)
        except Exception:
            objects = ()
        try:
            op_type = ghidra_insn.getOperandType(op_idx)
        except Exception:
            op_type = 0

        # Pre-collect Register objects: used both by the is_memory check
        # below (a memory operand MUST involve at least one base/index
        # register) and the reg-list classifier (>= 3 registers => REG_LIST).
        register_objs = [o for o in objects if isinstance(o, Register)]

        # A memory operand MUST involve at least one base/index register.
        # Without that, Ghidra's DYNAMIC bit on a pure-scalar operand
        # (e.g. RISC-V c.addi's immediate, or c.sdsp's disp scalar that
        # SLEIGH split off from its base register) is misleading and
        # produces a degenerate base-less mem-bracket rendering.
        #
        # ARM pre-indexed-with-writeback (``stp x29, x30, [sp, #-48]!``):
        # Ghidra reports ``op_type = REGISTER|ADDRESS`` (no DYNAMIC) and
        # surfaces the displacement as a ``Scalar`` inside ``getOpObjects``.
        # The disambiguator vs a bare register operand is the presence of
        # at least one Scalar (a plain REG operand carries no Scalar).
        # Without this arm the Scalar + bracket framing + writeback marker
        # are silently dropped.
        scalar_in_objects = any(isinstance(o, Scalar) for o in objects or ())
        is_memory = bool(register_objs) and (
            bool(op_type & OperandType.DYNAMIC)
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
        )

        fp_type = _compute_fp_type(ghidra_insn, op_idx, arch, base_mnemonic)

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
        )

        if not objects:
            return spec

        # Reg-list classification (ARM stm/ldm/push/pop/vpush/vpop/vstm/
        # vldm family). Ghidra's SLEIGH spec emits a flat sequence of
        # Register objects for reg-list operands; standard MEM operands
        # on every supported ISA carry at most 2 Registers (base+index
        # on x86/ARM, base-only on MIPS/PPC/RISC-V). Three or more
        # Registers in a single operand can therefore only be a
        # reg-list; classify accordingly so the MEM-decompose helpers
        # never see them (asserts in those helpers enforce the
        # invariant downstream).
        if len(register_objs) >= 3:
            spec["kind"] = OperandKind.REG_LIST
            spec["decompose_reg_list"] = self._decompose_reg_list_callback(
                ghidra_insn, op_idx, arch
            )
            return spec

        if is_memory:
            spec["kind"] = OperandKind.MEM
            # Memory access size: derived from SLEIGH-emitted PCode
            # LOAD/STORE varnode sizes. This is the only reliable
            # oracle - the legacy sibling-register-width heuristic
            # conflated pointer-width address-computation regs (e.g.
            # x64 r14 = 8B) with value regs, breaking 0x66 operand-
            # size-override and MOVZX/MOVSX byte/word -> wider dest.
            # ARM / MIPS / PPC / RISC-V consumers do not look at
            # ``op.size`` for MEM operands so the value is harmless on
            # non-x86 ISAs.
            spec["size"] = _infer_mem_access_size(ghidra_insn, op_idx)
            spec["decompose_mem"] = self._decompose_mem_callback(ghidra_insn, op_idx, arch)
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
            return spec
        if isinstance(first, Scalar):
            spec["kind"] = OperandKind.IMM
            spec["imm"] = int(first.getValue())
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
