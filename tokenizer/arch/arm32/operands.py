from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, Tokens

# ARM32 shift types (from Capstone)
_ARM_SFT_NAMES: dict[int, str] = {
    1: "lsl",
    2: "lsr",
    3: "asr",
    4: "ror",
    5: "rrx",
}

# Re-export the generic immediate tokenizer under the name used by the provider
tokenize_operand_immediate = tokenize_operand_immediate_generic


def tokenize_operand_memory(
    insn,
    lookup,
    op,
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

    has_base = base != 0
    has_index = index != 0
    has_disp = disp != 0

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    if has_base:
        tokens.append(vocab_manager.get_registry_token(insn.reg_name(base), base))

    if has_index:
        if has_base:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))
        tokens.append(vocab_manager.get_registry_token(insn.reg_name(index), index))

    if has_disp:
        if disp < 0:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
        elif has_base or has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

        abs_disp = abs(disp)
        if abs_disp <= 0xFF:
            tokens.append(vocab_manager.Valued_Const(abs_disp))
        else:
            force_opaque = not has_base
            meta, kind = lookup.lookup(abs_disp)

            if force_opaque or (abs_disp > (1 << 18)):
                disp_tokens = constant_handler.process_constant(
                    abs_disp,
                    is_arithmetic=False,
                    meta=meta,
                    library_type=meta.get("library", "unknown") if meta else "unknown",
                    insn_mnemonic=insn.mnemonic,
                )
                tokens.extend(disp_tokens)
            elif meta is not None:
                if (text_start <= abs_disp < text_end) or (abs_disp < func_min_addr or abs_disp > func_max_addr):
                    disp_tokens = constant_handler.process_constant(
                        abs_disp,
                        is_arithmetic=False,
                        meta=meta,
                        library_type=meta.get("library", "unknown"),
                        insn_mnemonic=insn.mnemonic,
                    )
                    tokens.extend(disp_tokens)
                else:
                    disp_tokens = constant_handler.process_constant(
                        abs_disp, is_arithmetic=True, insn_mnemonic=insn.mnemonic
                    )
                    tokens.extend(disp_tokens)
            else:
                disp_tokens = constant_handler.process_constant(
                    abs_disp, is_arithmetic=True, insn_mnemonic=insn.mnemonic
                )
                tokens.extend(disp_tokens)

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    return tokens


def tokenize_operand_shift(
    insn,
    op,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Tokenize a shift modifier on an ARM operand (e.g. LSL #2, ASR R3)."""
    tokens = []
    if hasattr(op, "shift") and op.shift.type != 0:
        shift_name = _ARM_SFT_NAMES.get(op.shift.type)
        if shift_name is not None:
            tokens.append(vocab_manager.PlatformToken(shift_name, PlatformInstructionTypes.ARITHMETIC))
            if op.shift.value != 0:
                tokens.append(vocab_manager.Valued_Const(op.shift.value))
    return tokens
