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
    """
    tokens = []

    base = op.mem.base
    index = op.mem.index
    disp = op.mem.disp

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

    if has_disp:
        if disp < 0:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
        elif has_base or has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

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

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

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
