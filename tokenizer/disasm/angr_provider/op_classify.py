"""Capstone operand-classifier helpers + architecture detection.

Owns:
- ``_ARCHINFO_NAME_TO_ARCHITECTURE``: archinfo arch.name -> owned ``Architecture``.
- ``_resolve_architecture``: archinfo ``Arch`` instance -> owned ``Architecture``.
- ``_CAPSTONE_OP_TYPE_*``: Capstone operand-type integer constants.
- ``_capstone_op_type_to_operand_kind``: Capstone op-type int -> ``OperandKind``.
- ``_CAPSTONE_ARM_SHIFT_TO_KIND``: Capstone ARM shift-type id -> ``ShiftKind``.
- ``_stamp_fp_type_default``: attach class-level ``fp_type = None`` to every
  Capstone operand class so ``op.fp_type`` is a typed read on the angr path.

The ``_stamp_fp_type_default()`` import-time side-effect is preserved by
running it at module load below (matching the legacy single-file behaviour).
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.types import Architecture, OperandKind, ShiftKind


# ---------------------------------------------------------------------------
# Capstone-operand uniform ``fp_type`` default
# ---------------------------------------------------------------------------
# The angr path delivers raw Capstone CsOpnd objects (X86Op, ArmOp, ...) to
# consumer code. Capstone never populates an FP-precision signal on these,
# so the angr-side ``op.fp_type`` is uniformly ``None`` (matches the typed
# ``Optional[FpType]`` shape exposed by the Ghidra path's ``OperandView``;
# see ``tokenizer/disasm/types.py``). Stamping the default at module load
# (rather than per-instance per-instruction) keeps the consumer API uniform
# across providers -- ``op.fp_type`` is a direct typed read with no
# ``getattr`` soft-probe -- and avoids touching the Capstone object on the
# hot path. ``angr_limitations.md`` section 1 documents why this field stays
# ``None`` on the angr side.
def _stamp_fp_type_default() -> None:
    """Attach class-level ``fp_type = None`` defaults to every Capstone
    operand class the angr-backed providers deliver to consumers.

    Only the classes we actually traverse are stamped; per-ISA imports are
    wrapped so an ISA whose Capstone bindings are unavailable in the active
    install (e.g. a stripped Capstone build) is silently skipped.
    """
    for module_name, class_name in (
        ("capstone.x86", "X86Op"),
        ("capstone.arm", "ArmOp"),
        ("capstone.arm64", "Arm64Op"),
        ("capstone.mips", "MipsOp"),
        ("capstone.ppc", "PpcOp"),
        ("capstone.riscv", "RiscvOp"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        # Skip if a value is already present (e.g. a future Capstone release
        # exposes the field natively or another module already stamped it).
        if "fp_type" not in cls.__dict__:
            cls.fp_type = None


_stamp_fp_type_default()


# archinfo arch.name -> owned ``Architecture`` enum. Centralised here so each
# wrapper does not need to know about archinfo internals.
_ARCHINFO_NAME_TO_ARCHITECTURE: dict[str, Architecture] = {
    "X86": Architecture.X86,
    "AMD64": Architecture.X86,
    "ARMEL": Architecture.ARM32,
    "ARMHF": Architecture.ARM32,
    "ARMCortexM": Architecture.ARM32,
    "AARCH64": Architecture.AARCH64,
    "MIPS32": Architecture.MIPS,
    "MIPS64": Architecture.MIPS,
    "PPC32": Architecture.PPC,
    "PPC64": Architecture.PPC,
    "RISCV64": Architecture.RISCV,
}


def _resolve_architecture(angr_arch: Any) -> Architecture:
    """Map an archinfo ``Arch`` instance onto our ``Architecture`` enum."""
    name = getattr(angr_arch, "name", None)
    if isinstance(name, str):
        mapped = _ARCHINFO_NAME_TO_ARCHITECTURE.get(name)
        if mapped is not None:
            return mapped
    return Architecture.UNKNOWN


# Capstone operand-type integer (REG=1/IMM=2/MEM=3, PPC CRX=64, ARM extras
# 64..67, ...) -> owned ``OperandKind``. ``OperandKind.OTHER`` covers any
# non-REG/IMM/MEM/CRX value (FP, CIMM, PIMM, SETEND, SYSREG, ...). The raw
# integer is preserved on ``OperandView.type_int`` so consumers that need
# the precise discriminator (e.g. emitting ``op_<n>`` platform tokens for
# ARM extras, see ``tokenizer/arch/arm32/provider.py``) can read it without
# losing the typed kind dispatch.
_CAPSTONE_OP_TYPE_REG = 1
_CAPSTONE_OP_TYPE_IMM = 2
_CAPSTONE_OP_TYPE_MEM = 3
_CAPSTONE_OP_TYPE_PPC_CRX = 64


def _capstone_op_type_to_operand_kind(op_type: int) -> OperandKind:
    if op_type == _CAPSTONE_OP_TYPE_REG:
        return OperandKind.REG
    if op_type == _CAPSTONE_OP_TYPE_IMM:
        return OperandKind.IMM
    if op_type == _CAPSTONE_OP_TYPE_MEM:
        return OperandKind.MEM
    if op_type == _CAPSTONE_OP_TYPE_PPC_CRX:
        return OperandKind.CRX
    if op_type == 0:
        return OperandKind.INVALID
    return OperandKind.OTHER


# Capstone ARM shift type -> owned ``ShiftKind``. Capstone uses the encoding
# ASR=1, LSL=2, LSR=3, ROR=4, RRX=5 (see ``capstone/arm_const.py``). The
# ``_REG`` variants (6..10) carry the same semantic kind with a register-
# valued amount; we collapse them onto the same enum entries (the consumer
# distinguishes register vs immediate amount via the operand structure on
# the parent, not via the shift kind).
_CAPSTONE_ARM_SHIFT_TO_KIND: dict[int, ShiftKind] = {
    0: ShiftKind.NONE,
    1: ShiftKind.ASR,
    2: ShiftKind.LSL,
    3: ShiftKind.LSR,
    4: ShiftKind.ROR,
    5: ShiftKind.RRX,
    6: ShiftKind.ASR,  # ASR_REG
    7: ShiftKind.LSL,  # LSL_REG
    8: ShiftKind.LSR,  # LSR_REG
    9: ShiftKind.ROR,  # ROR_REG
    10: ShiftKind.RRX,  # RRX_REG
}
