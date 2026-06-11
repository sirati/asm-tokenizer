"""ISA detection + per-operand FP-type + per-ISA prefix builders.

Owns:
- ``_ghidra_processor_to_architecture``: ``program.getLanguage().getProcessor()``
  -> owned ``Architecture``.
- ``_compute_fp_type``: per-operand FP-type computation. Fast path reads
  ``OperandType.FLOAT`` (set by SLEIGH on ISAs like ARM VFP). Slow path
  attributes FP status from the instruction's one-pass ``FLOAT_*``
  PCode signature (``pcode_inspect.InstructionPcodeSummary``) -- the
  only reliable oracle on x86 SSE, where SLEIGH never sets
  ``OperandType.FLOAT`` on ``MULSD``/``DIVSD``/``ADDSD``/... operands
  but emits ``FLOAT_ADD`` / ``FLOAT_MULT`` / ... p-code ops. The
  FLOAT_* opcode classification tables live in ``pcode_inspect``
  (``_resolve_float_pcode_sets``) beside the one-pass walk consuming
  them.
- ``ARM_BF16_MNEMONICS`` / ``X86_BF16_MNEMONICS``: BFloat16 mnemonic
  tables consulted at width=2.
- ``_FP_WIDTH_TO_TYPE``: width-in-bytes -> ``FpType`` dispatch.
- ``_bfloat16_mnemonic_for_arch``: per-ISA BFloat16 mnemonic dispatcher.
- ``_build_prefixes_*``: per-ISA typed-prefix-list builders.
- ``_x86_byte_to_prefix``: lazy x86 prefix-byte -> typed-prefix factory.
- ``_prefix_builder_for_arch``: per-ISA prefix-builder dispatcher.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from tokenizer.disasm.ghidra_provider import jvm_types
from tokenizer.disasm.ghidra_provider.mnemonic import (
    _extract_x86_prefixes,
    _split_ghidra_mnemonic,
    _strip_arm_cc_suffix,
)
from tokenizer.disasm.types import Architecture, FpType

if TYPE_CHECKING:
    from tokenizer.disasm.ghidra_provider.pcode_inspect import (
        InstructionPcodeSummary,
    )


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


def _apply_bfloat16_reclassification(
    fp_type: FpType,
    arch: Architecture,
    base_mnemonic: str,
) -> FpType:
    """Reclassify a width=2 Float16 to BFloat16 when ``base_mnemonic`` is on
    the per-ISA BFloat16 mnemonic table. Identity for every other ``FpType``.

    Shared by both ``_compute_fp_type`` paths so the BFloat16 signal is
    derived uniformly regardless of whether the FpType came from the
    ``OperandType.FLOAT`` fast path or the FLOAT_* p-code slow path.
    """
    if fp_type != FpType.FLOAT16:
        return fp_type
    bf16_set = _bfloat16_mnemonic_for_arch(arch)
    if bf16_set and base_mnemonic.upper() in bf16_set:
        return FpType.BFLOAT16
    return fp_type


def _fp_type_from_operand_type_bit(
    op_objects: Any,
    op_type: int,
    arch: Architecture,
    base_mnemonic: str,
) -> Optional[FpType]:
    """Fast path: ``OperandType.FLOAT`` bit -> FpType.

    ``op_objects`` / ``op_type`` are the operand's already-fetched
    ``getOpObjects(i)`` array and ``getOperandType(i)`` bitmask — the
    caller fetched them once for classification; re-fetching here would
    repeat the JVM round-trips per operand.

    Width derivation: inspect each ``op_objects`` element. For
    ``Register`` operands, use ``Register.getBitLength() / 8``. For
    ``Scalar`` operands, use ``Scalar.bitLength() / 8``. Take the
    largest value seen (x87 ``fld dword ptr [...]`` carries an FP-tagged
    memory operand whose size is the load size). Maps the resulting
    width-in-bytes through ``_FP_WIDTH_TO_TYPE``; widths outside the
    table return ``None``.

    Preserves fidelity on ISAs where SLEIGH sets ``OperandType.FLOAT``
    (e.g. ARM VFP). On x86 SLEIGH the bit is never set on SSE FP
    mnemonics; the slow path (``_fp_type_from_pcode_scan``) handles
    those.
    """
    OperandType = jvm_types.OperandType
    Register = jvm_types.Register
    Scalar = jvm_types.Scalar

    if not bool(op_type & OperandType.FLOAT):
        return None

    max_width_bits = 0
    for obj in op_objects or ():
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
    return _apply_bfloat16_reclassification(fp_type, arch, base_mnemonic)


def _operand_touches_fp_pcode(
    op_objects: Any,
    pcode_summary: "InstructionPcodeSummary",
) -> bool:
    """Decide whether the operand carries FP data based on the
    instruction's one-pass PCode summary
    (:class:`~tokenizer.disasm.ghidra_provider.pcode_inspect.InstructionPcodeSummary`).

    Attribution rules:

    - REG / CRX operand: a ``Register`` in ``op_objects`` whose
      address matches the address of any FP-tainted varnode.
    - MEM operand (RIP-relative folded into FLOAT_*): a direct
      ``Address`` object in ``op_objects`` whose string-form
      matches an FP-tainted varnode address. x86 SLEIGH folds memory
      loads with constant addresses directly into the FLOAT_* op's
      input varnode (no separate LOAD pcode) -- e.g. ``mulsd xmm0,
      qword ptr [rip+0x1197b]`` emits a single ``FLOAT_MULT`` op whose
      second input varnode is the absolute load address.
    - MEM operand (LOAD/STORE-routed): ``summary.fp_touches_load_store``
      — any ``LOAD`` output varnode or ``STORE`` value-input varnode in
      the instruction's p-code appears in ``fp_keys``. Captures
      register-indirect FP loads (``addsd xmm0, qword ptr [rax]``)
      where SLEIGH does emit a separate LOAD pcode that feeds the
      FLOAT_* op. The signal is instruction-level rather than tied to a
      specific operand because x86 instructions with FP semantics have
      at most one memory operand; multi-MEM instructions (string ops)
      do not emit FLOAT_* p-code.
    - Other kinds: never attributed by the slow path (IMM FP literals
      are exceedingly rare on FP-arithmetic mnemonics and are handled
      by the ``OperandType.FLOAT`` fast path on ISAs that surface them).
    """
    Address = jvm_types.Address
    Register = jvm_types.Register

    # REG / direct-mem-address path: match Register / Address objects in
    # the operand against the FP-tainted varnode keys. Both share the
    # same address-string namespace (Ghidra prints register-space addrs
    # as ``register:0000XXXX`` and ram-space addrs as bare hex).
    fp_keys = pcode_summary.fp_keys
    for obj in op_objects or ():
        try:
            if isinstance(obj, Register):
                obj_addr = str(obj.getAddress())
            elif isinstance(obj, Address):
                obj_addr = str(obj)
            else:
                continue
        except Exception:
            continue
        for key_addr, _size in fp_keys:
            if key_addr == obj_addr:
                return True

    # LOAD/STORE-routed MEM path (precomputed in the one-pass summary).
    return pcode_summary.fp_touches_load_store


def _fp_type_from_pcode_scan(
    op_objects: Any,
    arch: Architecture,
    base_mnemonic: str,
    pcode_summary: "InstructionPcodeSummary",
) -> Optional[FpType]:
    """Slow path: attribute per-operand FP status from the instruction's
    one-pass FLOAT_* PCode signature (``summary.fp_keys`` /
    ``summary.fp_width``, collected by
    ``pcode_inspect.collect_instruction_pcode_summary``).

    Returns ``None`` when:
    - no ``FLOAT_*`` opcode fires (instruction has no FP semantics), or
    - the FP-tainted varnode set carries a width not in
      ``_FP_WIDTH_TO_TYPE``, or
    - the operand neither holds a register matching an FP varnode nor
      participates in a LOAD/STORE that flows into an FP varnode.
    """
    if not pcode_summary.fp_keys:
        return None
    fp_type = _FP_WIDTH_TO_TYPE.get(pcode_summary.fp_width)
    if fp_type is None:
        return None
    if not _operand_touches_fp_pcode(op_objects, pcode_summary):
        return None
    return _apply_bfloat16_reclassification(fp_type, arch, base_mnemonic)


def _compute_fp_type(
    op_objects: Any,
    op_type: int,
    arch: Architecture,
    base_mnemonic: str,
    pcode_summary: "InstructionPcodeSummary",
) -> Optional[FpType]:
    """Module-level helper backing ``operand_fp_type``.

    Called per operand from the decode path so the resulting ``FpType``
    can be stamped on the owned operand view, keeping the public
    ``GhidraDisassemblyProvider.operand_fp_type`` method as a thin
    wrapper. Returns the matching ``FpType`` when the operand is
    FP-typed, ``None`` otherwise.

    Pure reader: consumes the operand's already-fetched
    ``getOpObjects(i)`` / ``getOperandType(i)`` values and the
    instruction-level one-pass PCode summary — no JVM round-trips of
    its own beyond Register/Address matching on FP instructions.

    Two-tier oracle:

    1. ``_fp_type_from_operand_type_bit`` (fast path): reads Ghidra's
       ``OperandType.FLOAT`` bit. Preserves fidelity on ISAs where
       SLEIGH sets the bit (e.g. ARM VFP).
    2. ``_fp_type_from_pcode_scan`` (slow path): when the fast path
       returns ``None``, attributes FP status from the FLOAT_* p-code
       signature -- the only reliable oracle on x86 SLEIGH, where
       ``OperandType.FLOAT`` is never set on SSE FP mnemonics
       (``MULSD`` / ``DIVSD`` / ``ADDSD`` / ``SUBSD`` / ``CVTSI2SD`` /
       ...) but the FP semantics live in ``FLOAT_ADD`` / ``FLOAT_MULT``
       / ``FLOAT_INT2FLOAT`` / ... p-code ops.

    The BFloat16 reclassification at width=2 (consulting
    ``_bfloat16_mnemonic_for_arch(arch)`` against ``base_mnemonic``)
    applies in both paths via ``_apply_bfloat16_reclassification``.
    """
    fast = _fp_type_from_operand_type_bit(
        op_objects, op_type, arch, base_mnemonic
    )
    if fast is not None:
        return fast
    return _fp_type_from_pcode_scan(
        op_objects, arch, base_mnemonic, pcode_summary
    )


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


def _instruction_has_rep_loop(ghidra_insn: Any) -> bool:
    """True iff the instruction's PCode encodes a REP-loop self-branch.

    Ghidra's SLEIGH spec for the x86 string-op family (MOVS, STOS, CMPS,
    SCAS, LODS, INS, OUTS) expands a REP/REPNE-prefixed instance into an
    unrolled PCode loop:

      [0] INT_EQUAL ECX, 0 -> uniq
      [1] CBRANCH <addr_after_this_insn>, uniq   ; jump past if counter exhausted
      [2..n] STORE/LOAD/INT_SUB ECX, ...         ; body + decrement
      [n+1] BRANCH <addr_of_this_insn>            ; loop back

    The smoking gun is the unconditional ``BRANCH`` whose target IS the
    instruction's own address (a self-loop). SSE/SSE2 instructions with
    F2/F3 MANDATORY prefix (ADDSD, MULSD, MOVSS, CVTSI2SD, ...) have NO
    self-loop in their PCode — their F2/F3 byte is an encoding-
    disambiguation prefix, not a repeat semantic.

    This is the rich-IR discriminator for "is this F2/F3 byte a real
    REPNE/REP prefix or an SSE mandatory prefix" — no mnemonic-string
    parsing involved.
    """
    PcodeOp = jvm_types.PcodeOp

    try:
        insn_addr = int(ghidra_insn.getAddress().getOffset())
    except Exception:
        return False
    try:
        pcode_ops = ghidra_insn.getPcode() or ()
    except Exception:
        return False
    for pop in pcode_ops:
        if pop.getOpcode() != PcodeOp.BRANCH:
            continue
        inputs = pop.getInputs()
        if not inputs:
            continue
        try:
            target_addr = inputs[0].getAddress()
            if target_addr is not None and int(target_addr.getOffset()) == insn_addr:
                return True
        except Exception:
            continue
    return False


def _build_prefixes_x86(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for an x86 instruction.

    Reads the same legacy prefix-byte set ``_extract_x86_prefixes``
    populates, then translates each byte into a typed
    ``InstructionPrefixView`` instance via ``_x86_byte_to_prefix``.
    Order: the byte-set is sorted so the produced list is stable across
    calls (the per-byte translation is independent of original encoding
    order).

    Filters out the F2 / F3 bytes when the instruction's PCode does NOT
    show the REP-loop self-branch pattern (see
    ``_instruction_has_rep_loop``). Those F2 / F3 bytes are the SSE/SSE2
    MANDATORY prefix on every non-string-op instruction (ADDSD, MULSD,
    MOVSS, CVTSI2SD, ...) — emitting a spurious ``repne`` token there
    would corrupt FP-heavy code with false-positive repeat semantics.
    """
    prefix_bytes = _extract_x86_prefixes(ghidra_insn)
    if 0xF2 in prefix_bytes or 0xF3 in prefix_bytes:
        # Belt-and-braces: emit REPNE/REP iff EITHER the PCode shows the
        # SLEIGH-unrolled REP-loop self-branch OR the Ghidra mnemonic
        # carries a typed ``.REP``/``.REPE``/``.REPNE`` suffix
        # (``_GHIDRA_SUFFIX_TO_PREFIX``). Both signals say the same thing
        # in practice today, but they survive different categories of
        # future Ghidra-side change — the OR makes the discriminator
        # robust against a SLEIGH-lifting refactor that swaps the
        # unrolled-loop for a CALLOTHER pseudo-op (mnemonic-suffix path
        # still fires) or a mnemonic-rendering change (PCode path still
        # fires).
        if not _instruction_has_rep_loop(ghidra_insn):
            try:
                raw_mnemonic = str(ghidra_insn.getMnemonicString())
            except Exception:
                raw_mnemonic = ""
            _stem, suffix_name, _suffix_byte = _split_ghidra_mnemonic(raw_mnemonic)
            if suffix_name not in ("rep", "repe", "repz", "repne", "repnz"):
                prefix_bytes = {b for b in prefix_bytes if b not in (0xF2, 0xF3)}
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
