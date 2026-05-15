"""Shared operand tokenization for architectures with simple base+disp memory operands.

Used by MIPS, PowerPC, RISC-V, and ARM. These architectures all expose
``OperandKind.MEM`` operands whose memory sub-view consists of base register +
displacement only (no index register or scale factor).
"""

from typing import List

from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.metadata import AddressKind
from tokenizer.disasm.types import InstructionView, OperandView
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, Tokens
from tokenizer.utils import num_hex_digits


def tokenize_operand_memory_base_disp(
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
    """Tokenize a memory operand with base+disp structure (no index/scale)."""
    tokens = []

    base = op.mem.base
    disp = op.mem.disp
    resolved_target = op.mem.resolved_target

    has_base = not base.is_absent
    # Resolved-target case (Ghidra-only): the provider's analyzer has
    # lifted the actual data address the access points at, distinct
    # from the operand's literal disp. The v2 classifier MUST receive
    # the resolved address to reach precedence step 7 (string_ptr) /
    # step 9 (ro_data_ptr); without it the bare-disp lookup either
    # returns UNKNOWN (literal-pool indirection) or misses entirely.
    has_resolved = resolved_target is not None
    has_disp = disp != 0 or has_resolved

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    if has_base:
        tokens.append(vocab_manager.get_registry_token(base.name, base.id))

    if has_disp:
        # For the resolved-target case the displayed value IS the
        # resolved address (the original tiny disp is folded into the
        # provider-side analysis). Sign tracking follows the chosen
        # value.
        classified_value = resolved_target if has_resolved else disp
        if classified_value < 0:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
        elif has_base:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

        abs_value = abs(classified_value)
        # Memory operand → an FP load against the resolved address gets a
        # postfix ``floatXX`` annotation (precedence.md "Postfix FP"). The
        # angr/Capstone path uniformly reports ``op.fp_type is None`` (see
        # ``angr_limitations.md`` §1), so the v2 emitter degrades to a
        # bare ptr token there. Ghidra populates the typed signal at
        # decode time.
        fp_postfix = op.fp_type
        if abs_value <= 0xFF and not has_resolved:
            # Short-circuit only applies to small literal disps; a
            # resolved-target address can never collapse to a bare
            # valued_const without losing the classification.
            tokens.append(vocab_manager.ValuedConst(abs_value))
        else:
            force_opaque = not has_base
            meta = lookup.lookup(abs_value)

            if force_opaque or (abs_value > (1 << 18)) or has_resolved:
                disp_tokens = constant_handler.process_constant_v2(
                    abs_value,
                    meta=meta,
                    is_arithmetic=False,
                    fp_postfix_type=fp_postfix,
                )
                tokens.extend(disp_tokens)
            else:
                if (text_start <= abs_value < text_end) or (abs_value < func_min_addr or abs_value > func_max_addr):
                    disp_tokens = constant_handler.process_constant_v2(
                        abs_value,
                        meta=meta,
                        is_arithmetic=False,
                        fp_postfix_type=fp_postfix,
                    )
                    tokens.extend(disp_tokens)
                else:
                    disp_tokens = constant_handler.process_constant_v2(
                        abs_value,
                        is_arithmetic=True,
                    )
                    tokens.extend(disp_tokens)

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    return tokens


def tokenize_operand_immediate_generic(
    addressing_control_flow_instructions: set[str],
    arithmetic_instructions: set[str],
    insn: InstructionView,
    lookup,
    op: OperandView,
    func_max_addr: int,
    func_min_addr: int,
    constant_handler: ConstantHandler,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Tokenize an immediate operand (shared across simple architectures)."""
    tokens = []

    raw_imm = op.imm
    imm_val = abs(raw_imm)
    imm_val_hex_len = num_hex_digits(imm_val)

    # Negative immediates: valued_const is a typeless byte-stream token
    # (no size, no signedness embedded in the token), so we surface the
    # sign as a leading MEM_MINUS token OUTSIDE any mem[ ]mem context.
    # Float immediates take a separate path (FP-postfix annotation on
    # the constant); only integer-immediate negatives need this branch.
    if raw_imm < 0:
        tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))

    # Immediate operand → an FP-tagged immediate (precedence.md step 1) is
    # an inline ``floatXX`` whose value is the IEEE bit pattern. Only the
    # Ghidra path stamps a non-None ``fp_type`` on the operand at decode
    # time; the angr/Capstone path uniformly reports ``op.fp_type is None``
    # (see ``angr_limitations.md`` §1) so the value flows through as an
    # integer ``valued_const``.
    fp_immediate = op.fp_type

    if imm_val_hex_len <= 2:
        imm_token = constant_handler.process_constant_v2(
            imm_val,
            fp_immediate_type=fp_immediate,
        )
        tokens.extend(imm_token)
    elif imm_val_hex_len <= 16:
        # Preserve the legacy ``base_mnemonic = insn.mnemonic`` behavior:
        # the variable name is misleading but the original consumer indexed
        # ``arithmetic_instructions`` / ``addressing_control_flow_instructions``
        # against the Capstone ``mnemonic`` string verbatim (which on ARM
        # includes the condition-code suffix). The owned ``InstructionView``
        # surfaces the same string as ``insn.mnemonic`` (vs ``base_mnemonic``
        # which strips the cc).
        base_mnemonic = insn.mnemonic
        if base_mnemonic in arithmetic_instructions:
            imm_token = constant_handler.process_constant_v2(
                imm_val,
                is_arithmetic=True,
                fp_immediate_type=fp_immediate,
            )
            tokens.extend(imm_token)
        elif base_mnemonic in addressing_control_flow_instructions:
            meta = lookup.lookup(imm_val)
            # Internal-jump heuristic: a control-flow target inside the
            # current function body (and not at the function entry) is
            # treated as a raw arithmetic value rather than a typed
            # ``block`` token. Function entries (``start_addr == imm_val``)
            # — including recursive calls — fall through to the typed
            # path so the ``local_func`` precedence rule fires.
            if (
                meta.kind == AddressKind.LOCAL_FUNCTION
                and meta.start_addr is not None
                and meta.start_addr != imm_val
                and func_min_addr <= imm_val < func_max_addr
            ):
                imm_token = constant_handler.process_constant_v2(
                    imm_val,
                    is_arithmetic=True,
                    fp_immediate_type=fp_immediate,
                )
                tokens.extend(imm_token)
            else:
                imm_token = constant_handler.process_constant_v2(
                    imm_val,
                    meta=meta,
                    is_arithmetic=False,
                    fp_immediate_type=fp_immediate,
                )
                tokens.extend(imm_token)
        else:
            meta = lookup.lookup(imm_val)
            imm_token = constant_handler.process_constant_v2(
                imm_val,
                meta=meta,
                is_arithmetic=False,
                fp_immediate_type=fp_immediate,
            )
            tokens.extend(imm_token)

    return tokens
