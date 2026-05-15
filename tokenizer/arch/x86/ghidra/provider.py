from pathlib import Path
from typing import List, Literal

from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.arch.x86.operands import (
    emit_x86_prefix_tokens,
    tokenize_operand_immediate,
    tokenize_operand_memory,
)
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import InstructionView, OperandKind
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent.parent / "data_store.json"


class X86GhidraProvider(ArchitectureProvider):
    """Architecture provider for x86/x64 with Ghidra backend.

    Prefix, mnemonic, REG, IMM, and MEM handling are all shared with the
    angr provider via the typed ``InstructionView`` / ``OperandView`` /
    ``PrefixesView`` surface from ``tokenizer/disasm/types.py``. The
    Ghidra-native memory decomposition (formerly in
    ``arch/x86/ghidra/operands.py``) is now performed inside
    ``_GhidraOperandView.mem`` against typed registers, so the consumer
    is provider-neutral.
    """

    def __init__(self, platform: Literal["x86", "x64"]):
        self._platform = platform

    @property
    def platform_str(self) -> str:
        return self._platform

    def load_instruction_sets(self) -> InstructionSets:
        return InstructionSets(_DATA_STORE_PATH)

    def parse_instruction(
        self,
        instr_sets: InstructionSets,
        constant_handler: ConstantHandler,
        func_max_addr: int,
        func_min_addr: int,
        insn: InstructionView,
        lookup,
        text_end: int,
        text_start: int,
        vocab_manager: VocabularyManager,
        insn_tokens: List[Tokens],
    ) -> List[Tokens]:
        # Prefix handling: typed-prefix dispatch (see emit_x86_prefix_tokens)
        insn_tokens.extend(emit_x86_prefix_tokens(insn, vocab_manager))

        # Mnemonic (without any leading prefix word Capstone glued on)
        insn_name = insn.base_mnemonic
        insn_type = instr_sets.get_instruction_type(insn_name)
        insn_tokens.append(vocab_manager.PlatformToken(insn_name, insn_type))

        # Operands
        for op in insn.operands:
            if op.kind == OperandKind.REG:
                reg = op.reg
                insn_tokens.append(vocab_manager.get_registry_token(reg.name, reg.id))
            elif op.kind == OperandKind.IMM:
                immediate_tokens = tokenize_operand_immediate(
                    instr_sets.addressing_control_flow,
                    instr_sets.arithmetic,
                    insn,
                    lookup,
                    op,
                    func_max_addr,
                    func_min_addr,
                    constant_handler,
                )
                insn_tokens.extend(immediate_tokens)
            elif op.kind == OperandKind.MEM:
                memory_tokens = tokenize_operand_memory(
                    insn,
                    lookup,
                    op,
                    text_end,
                    text_start,
                    func_max_addr,
                    func_min_addr,
                    vocab_manager,
                    constant_handler,
                )
                insn_tokens.extend(memory_tokens)
            elif op.kind == OperandKind.INVALID:
                raise Exception(f"Unsupported x86 operand type: {op.type_int}")
            else:
                # OperandKind.CRX / OperandKind.OTHER are never produced
                # by the x86 backend; preserve the legacy guard.
                raise Exception(f"Unsupported x86 operand type: {op.type_int}")

        return insn_tokens
