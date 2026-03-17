"""Shared operand tokenization for architectures with simple base+disp memory operands.

Used by MIPS, PowerPC, RISC-V, and ARM. These architectures all use Capstone's
standard operand types (1=REG, 2=IMM, 3=MEM) with memory operands consisting
of only base register + displacement (no index register or scale factor).
"""

from typing import List

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, Tokens
from tokenizer.utils import num_hex_digits


def tokenize_operand_memory_base_disp(
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
    """Tokenize a memory operand with base+disp structure (no index/scale)."""
    tokens = []

    base = op.mem.base
    disp = op.mem.disp

    has_base = base != 0
    has_disp = disp != 0

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    if has_base:
        tokens.append(vocab_manager.get_registry_token(insn, base))

    if has_disp:
        if disp < 0:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
        elif has_base:
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


def tokenize_operand_immediate_generic(
    addressing_control_flow_instructions: set[str],
    arithmetic_instructions: set[str],
    insn,
    lookup,
    op,
    func_max_addr: int,
    func_min_addr: int,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """Tokenize an immediate operand (shared across simple architectures)."""
    tokens = []

    imm_val = abs(op.imm)
    imm_val_hex_len = num_hex_digits(imm_val)

    if imm_val_hex_len <= 2:
        imm_token = constant_handler.process_constant(imm_val, insn_mnemonic=insn.mnemonic)
        tokens.extend(imm_token)
    elif imm_val_hex_len <= 16:
        base_mnemonic = insn.mnemonic
        if base_mnemonic in arithmetic_instructions:
            imm_token = constant_handler.process_constant(imm_val, is_arithmetic=True, insn_mnemonic=insn.mnemonic)
            tokens.extend(imm_token)
        elif base_mnemonic in addressing_control_flow_instructions:
            meta, kind = lookup.lookup(imm_val)
            if meta is not None:
                if kind == "range":
                    if func_min_addr <= imm_val < func_max_addr:
                        imm_token = constant_handler.process_constant(
                            imm_val, is_arithmetic=True, insn_mnemonic=insn.mnemonic
                        )
                        tokens.extend(imm_token)
                    else:
                        imm_token = constant_handler.process_constant(
                            imm_val,
                            is_arithmetic=False,
                            meta=meta,
                            library_type="function",
                            insn_mnemonic=insn.mnemonic,
                        )
                        tokens.extend(imm_token)
                else:
                    imm_token = constant_handler.process_constant(
                        imm_val, is_arithmetic=False, meta=meta, library_type="unknown", insn_mnemonic=insn.mnemonic
                    )
                    tokens.extend(imm_token)
            else:
                imm_token = constant_handler.process_constant(imm_val, is_arithmetic=True, insn_mnemonic=insn.mnemonic)
                tokens.extend(imm_token)
        else:
            meta, kind = lookup.lookup(imm_val)
            if meta is None:
                meta = {
                    "start_addr": imm_val,
                    "end_addr": imm_val,
                    "name": "unknown",
                    "type": "unknown",
                    "library": "unknown",
                }
            imm_token = constant_handler.process_constant(
                imm_val, is_arithmetic=False, meta=meta, library_type="unknown", insn_mnemonic=insn.mnemonic
            )
            tokens.extend(imm_token)

    return tokens
