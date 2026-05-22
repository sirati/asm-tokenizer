from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import (
    InstructionView,
    OperandView,
    ShiftKind,
    ShiftModifierView,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, RegisterListSymbol, Tokens

# Typed ShiftKind -> ARM asm-form shift word. Mirrors the legacy
# Capstone-int table 1..5 -> {lsl, lsr, asr, ror, rrx}; the typed enum
# drops the magic integers entirely.
_SHIFT_KIND_NAMES: dict[ShiftKind, str] = {
    ShiftKind.LSL: "lsl",
    ShiftKind.LSR: "lsr",
    ShiftKind.ASR: "asr",
    ShiftKind.ROR: "ror",
    ShiftKind.RRX: "rrx",
}

# Re-export the generic immediate tokenizer under the name used by the provider
tokenize_operand_immediate = tokenize_operand_immediate_generic


def _emit_arm_disp_value_tokens(
    value: int,
    has_base: bool,
    op: OperandView,
    lookup,
    text_start: int,
    text_end: int,
    func_min_addr: int,
    func_max_addr: int,
    constant_handler: ConstantHandler,
    vocab_manager: VocabularyManager,
    is_resolved_target: bool = False,
) -> List[Tokens]:
    """Tokenize the displacement *value* of an ARM memory operand.

    Shared between the offset-only / pre-indexed (value INSIDE the
    brackets) and post-indexed (value OUTSIDE the brackets, after an
    ``asm_post_index_separator``) emission paths. The caller is
    responsible for the surrounding bracket framing / sign prefix /
    separator tokens; this function emits only the constant tokens for
    the magnitude (per precedence.md value-classification flow).

    ``is_resolved_target`` signals that ``value`` is a provider-resolved
    data address (``op.mem.resolved_target``) rather than a literal
    operand displacement, in which case the small-disp short-circuit
    is skipped and the metadata-based classification path is always
    taken (otherwise a resolved string at e.g. ``0x42`` would collapse
    to a bare ``valued_const`` and lose the ``string_ptr`` emission).
    """
    tokens: List[Tokens] = []
    # Width-bucket / classifier-lookup decisions use the magnitude; the
    # signed ``value`` is what reaches ``process_constant_v2`` so the
    # v2 emitter can own sign decomposition (postfix ``value_negative``
    # for negatives, per the v2 sign-handling contract).
    abs_value = abs(value)
    # Memory operand → an FP load against a resolved pointer gets a
    # postfix ``floatXX`` (precedence.md "Postfix FP"). Only Ghidra
    # stamps a non-None ``fp_type`` on the operand (see
    # ``angr_limitations.md`` §1); the angr/Capstone path uniformly
    # reports None.
    fp_postfix = op.fp_type
    if abs_value <= 0xFF and not is_resolved_target:
        # Small-disp short-circuit: bypass classifier metadata (small
        # values can't be addresses) but still route through the v2
        # emitter so sign decomposition is owned in one place. With
        # ``meta=None`` the classifier falls through to step 11
        # (``_emit_valued_const``), yielding a bare valued_const_v2 for
        # non-negatives and ``[valued_const_v2, value_negative]`` for
        # negatives.
        disp_tokens = constant_handler.process_constant_v2(
            value,
            meta=None,
            is_arithmetic=False,
            fp_postfix_type=None,
        )
        tokens.extend(disp_tokens)
    else:
        force_opaque = not has_base
        # Address-typed lookup is signed: providers return UNKNOWN for
        # negative inputs, so a negative arithmetic displacement falls
        # through to the v2 valued_const fallback (step 11) and the
        # sign is carried by the postfix ``value_negative`` metatoken.
        # Keying on ``abs_value`` would let a negative disp whose
        # magnitude collides with a real string/data address misclassify
        # as a string_ptr / ro_data_ptr.
        meta = lookup.lookup(value)

        if force_opaque or (abs_value > (1 << 18)) or is_resolved_target:
            disp_tokens = constant_handler.process_constant_v2(
                value,
                meta=meta,
                is_arithmetic=False,
                fp_postfix_type=fp_postfix,
            )
            tokens.extend(disp_tokens)
        else:
            if (text_start <= abs_value < text_end) or (abs_value < func_min_addr or abs_value > func_max_addr):
                disp_tokens = constant_handler.process_constant_v2(
                    value,
                    meta=meta,
                    is_arithmetic=False,
                    fp_postfix_type=fp_postfix,
                )
                tokens.extend(disp_tokens)
            else:
                disp_tokens = constant_handler.process_constant_v2(value, is_arithmetic=True)
                tokens.extend(disp_tokens)
    return tokens


def tokenize_operand_memory(
    insn: InstructionView,
    lookup,
    op: OperandView,
    text_end: int,
    text_start: int,
    func_max_addr: int,
    func_min_addr: int,
    vocab_manager: VocabularyManager,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """
    Tokenize ARM memory operand.

    ARM has base + index + disp (unlike MIPS/PPC/RISC-V which only have base + disp).

    Three addressing modes are surfaced as typed flags on ``op.mem``:

    - Offset-only (``[base, #imm]``): the displacement is rendered INSIDE
      the brackets; no writeback marker is emitted.
    - Pre-indexed with writeback (``[base, #imm]!``): the displacement is
      rendered INSIDE the brackets; the close-bracket is followed by an
      ``asm_writeback_detect`` token signalling the base auto-update.
    - Post-indexed (``[base], #imm``): the displacement is rendered
      OUTSIDE the brackets, after the close-bracket and an
      ``asm_post_index_separator`` token; writeback is implicit so no
      explicit writeback marker is emitted.
    """
    tokens: List[Tokens] = []

    base = op.mem.base
    index = op.mem.index
    disp = op.mem.disp
    post_indexed = op.mem.post_indexed
    writeback = op.mem.writeback
    resolved_target = op.mem.resolved_target

    has_base = not base.is_absent
    has_index = not index.is_absent
    # Resolved-target case (Ghidra-only): substitute the analyzer-
    # resolved data address for the literal disp during classification
    # so precedence step 7 (string_ptr) / step 9 (ro_data_ptr) fire
    # for ARM literal-pool indirections like ``ldrb r3, [r4, #0]``
    # where r4 was loaded from a slot pointing at a string in .rodata.
    has_resolved = resolved_target is not None
    classified_value = resolved_target if has_resolved else disp
    has_disp = classified_value != 0 or has_resolved

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    if has_base:
        tokens.append(vocab_manager.get_registry_token(base.name, base.id))

    if has_index:
        if has_base:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))
        tokens.append(vocab_manager.get_registry_token(index.name, index.id))
        # Shifted-index addressing (``[base, index, lsl #N]``): the
        # shift modifier annotates the index register itself, so it
        # sits BEFORE the close-bracket and AFTER the index-register
        # token. ``index_shift.kind == ShiftKind.NONE`` covers every
        # non-shifted-index addressing mode (no tokens emitted).
        tokens.extend(_emit_shift_modifier_tokens(op.mem.index_shift, vocab_manager))

    # Offset / pre-indexed: disp INSIDE the brackets. Post-indexed: skip
    # the in-bracket disp emission; the disp is rendered after the
    # close-bracket + separator below.
    if has_disp and not post_indexed:
        # Sign is now owned by the v2 valued_const emitter (postfix
        # ``value_negative`` metatoken). The arch operand path no
        # longer pre-flattens negative disps; the structural MEM_PLUS
        # separator is emitted only when there is a preceding operand
        # to separate from (bare-disp ``mem[#imm]`` stays operator-
        # less).
        if has_base or has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))
        tokens.extend(
            _emit_arm_disp_value_tokens(
                classified_value,
                has_base,
                op,
                lookup,
                text_start,
                text_end,
                func_min_addr,
                func_max_addr,
                constant_handler,
                vocab_manager,
                is_resolved_target=has_resolved,
            )
        )

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    if writeback:
        tokens.append(vocab_manager.RegisterList(RegisterListSymbol.WRITEBACK))
    elif post_indexed and has_disp:
        tokens.append(
            vocab_manager.MemoryOperand(MemoryOperandSymbol.POST_INDEX_SEPARATOR)
        )
        # Post-indexed ARM addressing is always ``[base], #imm`` with a
        # base; the sign of the disp is owned by the v2 valued_const
        # emitter (postfix ``value_negative``), no caller-side flip.
        tokens.extend(
            _emit_arm_disp_value_tokens(
                classified_value,
                has_base,
                op,
                lookup,
                text_start,
                text_end,
                func_min_addr,
                func_max_addr,
                constant_handler,
                vocab_manager,
                is_resolved_target=has_resolved,
            )
        )

    return tokens


def tokenize_operand_reg_list(
    op: OperandView,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Tokenize an ARM register-list operand (stm/ldm/push/pop/vpush/vpop/
    vstm/vldm family).

    Asm shape: ``[base [!]] { r0, r1, ... }``. The base register (writeback
    target) is emitted first when present, optionally followed by the
    writeback marker; then the open-brace, the list members in order, and
    the close-brace. Mirrors ``tokenize_operand_memory``'s use of the
    typed ``MemoryOperandSymbol`` enum but uses ``RegisterListSymbol``.
    """
    tokens: List[Tokens] = []
    rl = op.reg_list
    base = rl.base
    if not base.is_absent:
        tokens.append(vocab_manager.get_registry_token(base.name, base.id))
        if rl.writeback:
            tokens.append(vocab_manager.RegisterList(RegisterListSymbol.WRITEBACK))
    tokens.append(vocab_manager.RegisterList(RegisterListSymbol.OPEN_BRACE))
    for member in rl:
        tokens.append(vocab_manager.get_registry_token(member.name, member.id))
    tokens.append(vocab_manager.RegisterList(RegisterListSymbol.CLOSE_BRACE))
    return tokens


def _emit_shift_modifier_tokens(
    shift: ShiftModifierView,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Emit shift-keyword + amount tokens for a ``ShiftModifierView``.

    Shared core used by both ``tokenize_operand_shift`` (operand-level
    shift, e.g. ``mov r0, r1, lsl #2``) and the memory-operand
    ``index_shift`` emission (e.g. ``ldr r0, [r1, r2, lsl #1]``):
    ``ShiftKind.NONE`` yields no tokens; otherwise emit the keyword
    (lsl/lsr/asr/ror/rrx) as an ARITHMETIC platform token, followed by
    a ValuedConst when the amount is non-zero.
    """
    tokens: List[Tokens] = []
    if shift.kind != ShiftKind.NONE:
        shift_name = _SHIFT_KIND_NAMES.get(shift.kind)
        if shift_name is not None:
            tokens.append(vocab_manager.PlatformToken(shift_name, PlatformInstructionTypes.ARITHMETIC))
            if shift.amount != 0:
                tokens.append(vocab_manager.ValuedConst(shift.amount))
    return tokens


def tokenize_operand_shift(
    insn: InstructionView,
    op: OperandView,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Tokenize a shift modifier on an ARM operand (e.g. LSL #2, ASR R3).

    ``op.shift`` is always present on the typed ``OperandView`` -
    ``ShiftKind.NONE`` signals "no shift modifier on this operand",
    replacing the legacy ``hasattr(op, "shift") and op.shift.type != 0``
    probe.
    """
    return _emit_shift_modifier_tokens(op.shift, vocab_manager)
