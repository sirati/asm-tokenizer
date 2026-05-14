import warnings
from typing import List

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, Tokens
from tokenizer.utils import num_hex_digits

SIZE_MAP: dict[int, str] = {
    1: "byte_ptr",
    2: "word_ptr",
    4: "dword_ptr",
    8: "qword_ptr",
    10: "xword_ptr",  # x87 extended precision (80-bit)
    14: "fpu_env_ptr",  # x87 environment (16-bit mode)
    16: "xmmword_ptr",
    28: "fpu_env_ptr",  # x87 environment (32-bit mode) - ADD THIS
    32: "ymmword_ptr",
    64: "zmmword_ptr",
    94: "fpu_state_ptr",  # x87 state (16-bit mode)
    108: "fpu_state_ptr",  # x87 state (32-bit mode)
}


def tokenize_operand_memory(
    insn,
    lookup,
    op,
    text_end,
    text_start,
    func_max_addr,
    func_min_addr,
    vocab_manager: VocabularyManager,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """
    Tokenize x86/x64 memory operand and return list of tokens.

    Returns:
        List of TokensRepl objects for this memory operand
    """
    tokens = []

    disp = op.mem.disp

    scale = op.mem.scale
    base = op.mem.base
    index = op.mem.index

    has_reg = op.mem.base != 0
    has_index = op.mem.index != 0
    has_disp = op.mem.disp != 0

    if op.size in SIZE_MAP:
        tokens.append(vocab_manager.PlatformToken(SIZE_MAP[op.size], PlatformInstructionTypes.POINTER_LENGTHS))
    else:
        # Find the next bigger size in SIZE_MAP or take the largest available
        next_size = min((s for s in SIZE_MAP if s > op.size), default=max(SIZE_MAP))
        tokens.append(vocab_manager.PlatformToken(SIZE_MAP[next_size], PlatformInstructionTypes.POINTER_LENGTHS))
        warnings.warn(
            f"unexpected memory operand size: {op.size}, using next bigger '{SIZE_MAP[next_size]}' at {next_size}bytes for instruction {insn}"
        )

    if op.mem.segment > 0:
        tokens.append(
            vocab_manager.PlatformToken(
                f"{insn.reg_name(op.mem.segment)}:", PlatformInstructionTypes.MEMORY_ACCESS_MODE
            )
        )

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    # Register the base and index registers
    if has_reg:
        tokens.append(vocab_manager.get_registry_token(insn.reg_name(base), base))

    if has_index:
        if has_reg:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

        tokens.append(vocab_manager.get_registry_token(insn.reg_name(index), index))

    # Process scale as a constant if in expected range
    if scale != 1:
        assert scale > 0
        if has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MULTIPLY))
            valued_const = vocab_manager.Valued_Const_V2 if vocab_manager.format_version == 2 else vocab_manager.Valued_Const
            tokens.append(valued_const(abs(scale)))
        else:
            warnings.warn(f"Scale {scale} used without index register in instruction {insn}")

    if disp < 0:
        tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
    elif has_disp and (has_reg or has_index):
        tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

    # Process displacement
    if not has_disp:
        pass  # noop ignore
    elif (
        disp <= 0xFF
    ):  # if we are in range 00 to 0xFF we always use constant, same if we are negative as its defo not an addr
        valued_const = vocab_manager.Valued_Const_V2 if vocab_manager.format_version == 2 else vocab_manager.Valued_Const
        tokens.append(valued_const(abs(disp)))

    else:
        force_opaque = False

        if has_disp and not has_reg:
            force_opaque = True

        elif disp > (1 << 18):  # = 262,144
            force_opaque = True

        meta, kind = lookup.lookup(disp)

        # Memory operand → if the load is FP-typed Ghidra stamps the width
        # on this _CapOperand at decode time; the v2 emitter appends a
        # postfix ``floatXX`` after the ptr token (precedence.md "Postfix
        # FP"). The angr path leaves this None (``angr_limitations.md`` §1).
        fp_postfix = getattr(op, "fp_width_bytes", None)

        if force_opaque:
            disp_token = constant_handler.process_constant_v2(
                disp,
                meta=meta,
                is_arithmetic=False,
                fp_postfix_width_bytes=fp_postfix,
            )
            tokens.extend(disp_token)
        # For larger displacements, check if pointing to known constant or code or opaque
        meta, kind = lookup.lookup(disp)
        if meta is not None:
            # Check if displacement is in text section or outside function bounds
            if (text_start <= disp < text_end) or (disp < func_min_addr or disp > func_max_addr):
                disp_token = constant_handler.process_constant_v2(
                    disp,
                    meta=meta,
                    is_arithmetic=False,
                    fp_postfix_width_bytes=fp_postfix,
                )
                tokens.extend(disp_token)
            else:
                # Local constant - treat as valued constant literal
                disp_token = constant_handler.process_constant_v2(disp, is_arithmetic=True)
                tokens.extend(disp_token)
        else:
            # No metadata found - treat as valued constant literal
            disp_token = constant_handler.process_constant_v2(disp, is_arithmetic=True)
            tokens.extend(disp_token)

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    return tokens


def tokenize_operand_immediate(
    addressing_control_flow_instructions,
    arithmetic_instructions,
    insn,
    lookup,
    op,
    func_max_addr,
    func_min_addr,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """
    Tokenize x86/x64 immediate operand and return list of tokens.

    Returns:
        List of TokensRepl objects for this immediate operand
    """
    tokens = []

    imm_val = abs(op.imm)
    imm_val_hex_len = num_hex_digits(imm_val)

    # Immediate operand → an FP-tagged immediate (precedence.md step 1) is
    # an inline ``floatXX`` with IEEE bit pattern. Only the Ghidra path
    # stamps ``fp_width_bytes`` on _CapOperand; angr/Capstone CsOpnd has no
    # such attribute (see ``angr_limitations.md`` §1).
    fp_immediate = getattr(op, "fp_width_bytes", None)

    if imm_val_hex_len <= 2:  # Small immediate (0x00 to 0xFF)
        imm_token = constant_handler.process_constant_v2(
            imm_val,
            fp_immediate_width_bytes=fp_immediate,
        )
        tokens.extend(imm_token)
    elif imm_val_hex_len <= (128 / 4):  # Larger immediate (up to 128-bit)
        if insn.mnemonic in arithmetic_instructions:
            # Arithmetic instruction - treat as valued constant literal
            imm_token = constant_handler.process_constant_v2(
                imm_val,
                is_arithmetic=True,
                fp_immediate_width_bytes=fp_immediate,
            )
            tokens.extend(imm_token)
        elif insn.mnemonic in addressing_control_flow_instructions:
            # Addressing/control flow instruction - check for metadata
            meta, kind = lookup.lookup(imm_val)
            # todo we have a major issue here: a lot of targets are NOI in this table, e.g. I got .plt but angr can resolve it cfg.kb.functions.get(call_target_addr)
            if meta is not None:
                if kind == "range":
                    if func_min_addr <= imm_val < func_max_addr:  # Local
                        imm_token = constant_handler.process_constant_v2(
                            imm_val,
                            is_arithmetic=True,
                            fp_immediate_width_bytes=fp_immediate,
                        )
                        tokens.extend(imm_token)
                    else:  # External
                        imm_token = constant_handler.process_constant_v2(
                            imm_val,
                            meta=meta,
                            is_arithmetic=False,
                            fp_immediate_width_bytes=fp_immediate,
                        )
                        tokens.extend(imm_token)
                else:
                    imm_token = constant_handler.process_constant_v2(
                        imm_val,
                        meta=meta,
                        is_arithmetic=False,
                        fp_immediate_width_bytes=fp_immediate,
                    )
                    tokens.extend(imm_token)
            else:
                # No metadata - treat as valued constant literal
                imm_token = constant_handler.process_constant_v2(
                    imm_val,
                    is_arithmetic=True,
                    fp_immediate_width_bytes=fp_immediate,
                )
                tokens.extend(imm_token)
        else:  # Fallback - create opaque constant
            meta, kind = lookup.lookup(imm_val)
            if meta is None:
                # Default/fallback meta if lookup fails
                meta = {
                    "start_addr": imm_val,
                    "end_addr": imm_val,
                    "name": "unknown",
                    "type": "unknown",
                    "library": "unknown",
                }
            imm_token = constant_handler.process_constant_v2(
                imm_val,
                meta=meta,
                is_arithmetic=False,
                fp_immediate_width_bytes=fp_immediate,
            )
            tokens.extend(imm_token)

    return tokens
