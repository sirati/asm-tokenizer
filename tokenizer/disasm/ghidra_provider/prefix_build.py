"""ISA detection + per-operand FP-type + per-ISA prefix builders.

Owns:
- ``_ghidra_processor_to_architecture``: ``program.getLanguage().getProcessor()``
  -> owned ``Architecture``.
- ``_compute_fp_type``: per-operand FP-type computation (Ghidra
  ``OperandType.FLOAT`` -> ``FpType``).
- ``ARM_BF16_MNEMONICS`` / ``X86_BF16_MNEMONICS``: BFloat16 mnemonic
  tables consulted at width=2.
- ``_FP_WIDTH_TO_TYPE``: width-in-bytes -> ``FpType`` dispatch.
- ``_bfloat16_mnemonic_for_arch``: per-ISA BFloat16 mnemonic dispatcher.
- ``_build_prefixes_*``: per-ISA typed-prefix-list builders.
- ``_x86_byte_to_prefix``: lazy x86 prefix-byte -> typed-prefix factory.
- ``_prefix_builder_for_arch``: per-ISA prefix-builder dispatcher.
"""

from __future__ import annotations

from typing import Any, Optional

from tokenizer.disasm.ghidra_provider.mnemonic import (
    _extract_x86_prefixes,
    _strip_arm_cc_suffix,
)
from tokenizer.disasm.types import Architecture, FpType


# BFloat16 mnemonic tables (per-ISA). Width=2 alone cannot distinguish IEEE-754
# Float16 from Google's BFloat16 -- SLEIGH does not tag the bfloat16 type
# distinctly. The reclassification at width=2 consults these per-ISA mnemonic
# sets; ISAs not represented here keep the default Float16 mapping.
ARM_BF16_MNEMONICS: frozenset[str] = frozenset({
    "BFCVT", "BFCVTN", "BFCVTN2", "BFDOT", "BFMMLA",
    "BFMLAL", "BFMLALB", "BFMLALT", "VFMAB", "VFMAT",
})
X86_BF16_MNEMONICS: frozenset[str] = frozenset({
    "VCVTNE2PS2BF16", "VCVTNEPS2BF16", "VDPBF16PS",
})

# width-in-bytes -> FpType dispatch (default mapping; width=2 may be
# reclassified to BFLOAT16 by ``_compute_fp_type``).
_FP_WIDTH_TO_TYPE: dict[int, FpType] = {
    2: FpType.FLOAT16,
    4: FpType.FLOAT32,
    8: FpType.FLOAT64,
    10: FpType.FLOAT80,
    16: FpType.FLOAT128,
}


def _bfloat16_mnemonic_for_arch(arch: Architecture) -> frozenset[str]:
    """Return the BFloat16 mnemonic set for ``arch`` (empty when unsupported).

    Single dispatcher consulted at width=2 by ``_compute_fp_type`` to decide
    whether to reclassify Float16 -> BFloat16 for this instruction. ISAs
    without a curated table fall through with the default Float16 mapping.
    """
    if arch in (Architecture.ARM32, Architecture.AARCH64):
        return ARM_BF16_MNEMONICS
    if arch == Architecture.X86:
        return X86_BF16_MNEMONICS
    return frozenset()


def _ghidra_processor_to_architecture(program: Any) -> Architecture:
    """Map ``program.getLanguage().getProcessor()`` to the owned ``Architecture``.

    Threads the ISA into the FP-type computation that runs per operand.
    Unknown processors map to ``Architecture.UNKNOWN``; the BFloat16
    reclassification then no-ops.
    """
    try:
        processor = str(program.getLanguage().getProcessor()).lower()
    except Exception:
        return Architecture.UNKNOWN
    if processor.startswith("aarch64"):
        return Architecture.AARCH64
    if processor.startswith("arm"):
        return Architecture.ARM32
    if processor in ("x86", "x64") or processor.startswith("x86"):
        return Architecture.X86
    if processor.startswith("mips"):
        return Architecture.MIPS
    if processor.startswith("powerpc") or processor.startswith("ppc"):
        return Architecture.PPC
    if processor.startswith("riscv"):
        return Architecture.RISCV
    return Architecture.UNKNOWN


def _compute_fp_type(
    ghidra_insn: Any,
    operand_index: int,
    arch: Architecture,
    base_mnemonic: str,
) -> Optional[FpType]:
    """Module-level helper backing ``operand_fp_type``.

    Called per operand from the decode path so the resulting ``FpType``
    can be stamped on the owned operand view, keeping the public
    ``GhidraDisassemblyProvider.operand_fp_type`` method as a thin
    wrapper. Returns the matching ``FpType`` when the
    operand is FP-typed (Ghidra ``OperandType.FLOAT`` bitmask) or
    ``None`` otherwise. The width derivation order is:

    1. Inspect each ``getOpObjects(i)`` element. For ``Register`` operands,
       use ``Register.getBitLength() / 8``. For ``Scalar`` operands, use
       ``Scalar.bitLength() / 8``. Take the largest value seen (x87
       ``fld dword ptr [...]`` carries an FP-tagged memory operand whose
       size is the load size).
    2. Map the resulting width-in-bytes through ``_FP_WIDTH_TO_TYPE``.
    3. At width=2, consult ``_bfloat16_mnemonic_for_arch(arch)`` against
       the instruction's ``base_mnemonic`` (uppercase-compared) and
       reclassify Float16 -> BFloat16 on a hit. SLEIGH does not currently
       tag bfloat16 distinctly, so the mnemonic-based reclassification
       is the only signal available.
    4. Widths outside ``_FP_WIDTH_TO_TYPE`` return ``None`` (the
       classifier then routes through step 11 of the precedence list
       rather than emitting a malformed ``floatXX``).
    """
    from ghidra.program.model.lang import OperandType, Register
    from ghidra.program.model.scalar import Scalar

    try:
        op_type = ghidra_insn.getOperandType(operand_index)
    except Exception:
        return None
    if not bool(op_type & OperandType.FLOAT):
        return None

    max_width_bits = 0
    try:
        objects = ghidra_insn.getOpObjects(operand_index)
    except Exception:
        objects = ()
    for obj in objects or ():
        try:
            if isinstance(obj, Register):
                width = int(obj.getBitLength())
            elif isinstance(obj, Scalar):
                width = int(obj.bitLength())
            else:
                continue
        except Exception:
            continue
        if width > max_width_bits:
            max_width_bits = width

    width_bytes = max_width_bits // 8
    fp_type = _FP_WIDTH_TO_TYPE.get(width_bytes)
    if fp_type is None:
        return None

    if fp_type == FpType.FLOAT16:
        bf16_set = _bfloat16_mnemonic_for_arch(arch)
        if bf16_set and base_mnemonic.upper() in bf16_set:
            fp_type = FpType.BFLOAT16

    return fp_type


# ---------------------------------------------------------------------------
# Per-ISA prefix builders
# ---------------------------------------------------------------------------
# Build typed ``InstructionPrefixView`` instances for a Ghidra Instruction.
# x86 reads the prefix-byte set; ARM / PPC / MIPS / RISC-V return empty
# lists for now (their typed-prefix fields stay at defaults, so
# consumer predicates always fall through until those signals become
# available).

_X86_BYTE_TO_PREFIX_BUILDER: dict[int, Any] = {
    # Filled lazily on first use to avoid importing the prefix subclasses
    # at module load time.
}


def _x86_byte_to_prefix(byte: int) -> Any:
    """Return a typed ``InstructionPrefixView`` for an x86 prefix byte.

    Returns ``None`` for bytes outside the recognized prefix set (caller
    skips). Lazy-initializes the byte->builder map to avoid pulling in
    typed prefix classes at module import time.
    """
    if not _X86_BYTE_TO_PREFIX_BUILDER:
        from tokenizer.disasm.ghidra_views import (
            _AddressSizePrefix,
            _LockPrefix,
            _OperandSizePrefix,
            _RepPrefix,
            _SegmentOverridePrefix,
        )
        from tokenizer.disasm.types import X86Segment

        _X86_BYTE_TO_PREFIX_BUILDER.update({
            0xF0: lambda: _LockPrefix(),
            0xF2: lambda: _RepPrefix(repeat_until_zero=False),  # REPNE
            0xF3: lambda: _RepPrefix(repeat_until_zero=True),   # REPE/REP
            0x26: lambda: _SegmentOverridePrefix(X86Segment.ES),
            0x2E: lambda: _SegmentOverridePrefix(X86Segment.CS),
            0x36: lambda: _SegmentOverridePrefix(X86Segment.SS),
            0x3E: lambda: _SegmentOverridePrefix(X86Segment.DS),
            0x64: lambda: _SegmentOverridePrefix(X86Segment.FS),
            0x65: lambda: _SegmentOverridePrefix(X86Segment.GS),
            0x66: lambda: _OperandSizePrefix(),
            0x67: lambda: _AddressSizePrefix(),
        })
    builder = _X86_BYTE_TO_PREFIX_BUILDER.get(byte)
    if builder is None:
        return None
    return builder()


def _build_prefixes_x86(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for an x86 instruction.

    Reads the same legacy prefix-byte set ``_extract_x86_prefixes``
    populates, then translates each byte into a typed
    ``InstructionPrefixView`` instance via ``_x86_byte_to_prefix``.
    Order: the byte-set is sorted so the produced list is stable across
    calls (the per-byte translation is independent of original encoding
    order).
    """
    prefix_bytes = _extract_x86_prefixes(ghidra_insn)
    out: list[Any] = []
    for byte in sorted(prefix_bytes):
        view = _x86_byte_to_prefix(byte)
        if view is not None:
            out.append(view)
    return out


def _build_prefixes_arm(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for an ARM / AArch64 instruction.

    Recovers the condition-code prefix from Ghidra's mnemonic-suffix
    encoding (SLEIGH's ``^COND`` concatenation). The raw mnemonic
    surfaces forms like ``bne`` / ``beq`` / ``streq`` / ``b.eq``;
    ``_strip_arm_cc_suffix`` returns the cc enum when the stem is on the
    SLEIGH-derived allow-list. When present, emit a
    ``ConditionCodePrefixView`` carrying the cc.
    """
    from tokenizer.disasm.ghidra_views import _ConditionCodePrefix

    raw = str(ghidra_insn.getMnemonicString()).lower()
    _stem, cc = _strip_arm_cc_suffix(raw)
    if cc is None:
        return []
    return [_ConditionCodePrefix(cc=cc)]


def _build_prefixes_ppc(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for a PPC instruction.

    Stub for forward-compat; same shape as ``_build_prefixes_arm``.
    The ``bc`` and ``update_cr0`` signals are not extracted by the
    Ghidra path today.
    """
    return []


def _build_prefixes_empty(ghidra_insn: Any) -> list[Any]:
    """No-prefix builder for MIPS/RISC-V."""
    return []


def _prefix_builder_for_arch(arch: Architecture) -> Any:
    """Dispatch the per-ISA prefix builder."""
    if arch == Architecture.X86:
        return _build_prefixes_x86
    if arch in (Architecture.ARM32, Architecture.AARCH64):
        return _build_prefixes_arm
    if arch == Architecture.PPC:
        return _build_prefixes_ppc
    return _build_prefixes_empty
