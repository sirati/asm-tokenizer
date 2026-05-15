from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import InstructionView, OperandView, ShiftKind
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
    disp: int,
    has_base: bool,
    op: OperandView,
    lookup,
    text_start: int,
    text_end: int,
    func_min_addr: int,
    func_max_addr: int,
    constant_handler: ConstantHandler,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Tokenize the displacement *value* of an ARM memory operand.

    Shared between the offset-only / pre-indexed (disp INSIDE the
    brackets) and post-indexed (disp OUTSIDE the brackets, after an
    ``asm_post_index_separator``) emission paths. The caller is
    responsible for the surrounding bracket framing / sign prefix /
    separator tokens; this function emits only the constant tokens for
    the magnitude (per precedence.md value-classification flow).
    """
    tokens: List[Tokens] = []
    abs_disp = abs(disp)
    # Memory operand → an FP load against a resolved pointer gets a
    # postfix ``floatXX`` (precedence.md "Postfix FP"). Only Ghidra
    # stamps a non-None ``fp_type`` on the operand (see
    # ``angr_limitations.md`` §1); the angr/Capstone path uniformly
    # reports None.
    fp_postfix = op.fp_type
    if abs_disp <= 0xFF:
        tokens.append(vocab_manager.ValuedConst(abs_disp))
    else:
        force_opaque = not has_base
        meta = lookup.lookup(abs_disp)

        if force_opaque or (abs_disp > (1 << 18)):
            disp_tokens = constant_handler.process_constant_v2(
                abs_disp,
                meta=meta,
                is_arithmetic=False,
                fp_postfix_type=fp_postfix,
            )
            tokens.extend(disp_tokens)
        else:
            if (text_start <= abs_disp < text_end) or (abs_disp < func_min_addr or abs_disp > func_max_addr):
                disp_tokens = constant_handler.process_constant_v2(
                    abs_disp,
                    meta=meta,
                    is_arithmetic=False,
                    fp_postfix_type=fp_postfix,
                )
                tokens.extend(disp_tokens)
            else:
                disp_tokens = constant_handler.process_constant_v2(abs_disp, is_arithmetic=True)
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

    has_base = not base.is_absent
    has_index = not index.is_absent
    has_disp = disp != 0

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    if has_base:
        tokens.append(vocab_manager.get_registry_token(base.name, base.id))

    if has_index:
        if has_base:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))
        tokens.append(vocab_manager.get_registry_token(index.name, index.id))

    # Offset / pre-indexed: disp INSIDE the brackets. Post-indexed: skip
    # the in-bracket disp emission; the disp is rendered after the
    # close-bracket + separator below.
    if has_disp and not post_indexed:
        if disp < 0:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
        elif has_base or has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))
        tokens.extend(
            _emit_arm_disp_value_tokens(
                disp,
                has_base,
                op,
                lookup,
                text_start,
                text_end,
                func_min_addr,
                func_max_addr,
                constant_handler,
                vocab_manager,
            )
        )

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    if writeback:
        tokens.append(vocab_manager.RegisterList(RegisterListSymbol.WRITEBACK))
    elif post_indexed and has_disp:
        tokens.append(
            vocab_manager.MemoryOperand(MemoryOperandSymbol.POST_INDEX_SEPARATOR)
        )
        if disp < 0:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
        tokens.extend(
            _emit_arm_disp_value_tokens(
                disp,
                has_base,
                op,
                lookup,
                text_start,
                text_end,
                func_min_addr,
                func_max_addr,
                constant_handler,
                vocab_manager,
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
    tokens = []
    shift = op.shift
    if shift.kind != ShiftKind.NONE:
        shift_name = _SHIFT_KIND_NAMES.get(shift.kind)
        if shift_name is not None:
            tokens.append(vocab_manager.PlatformToken(shift_name, PlatformInstructionTypes.ARITHMETIC))
            if shift.amount != 0:
                tokens.append(vocab_manager.ValuedConst(shift.amount))
    return tokens
