from pathlib import Path
from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic, tokenize_operand_memory_base_disp
from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import InstructionView, OperandKind
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent / "data_store.json"


class RISCVProvider(ArchitectureProvider):
    """Architecture provider for RISC-V (RV32 and RV64) platforms."""

    def __init__(self, platform: str = "riscv"):
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
        insn_name = insn.base_mnemonic
        insn_type = instr_sets.get_instruction_type(insn_name)
        insn_tokens.append(vocab_manager.PlatformToken(insn_name, insn_type))

        # RISC-V has no per-instruction prefix signal; `insn.prefixes`
        # is the empty list per `_build_prefixes` dispatcher.

        for op in insn.operands:
            if op.kind == OperandKind.REG:
                reg = op.reg
                insn_tokens.append(vocab_manager.get_registry_token(reg.name, reg.id))
            elif op.kind == OperandKind.IMM:
                insn_tokens.extend(
                    tokenize_operand_immediate_generic(
                        instr_sets.addressing_control_flow,
                        instr_sets.arithmetic,
                        insn,
                        lookup,
                        op,
                        func_max_addr,
                        func_min_addr,
                        constant_handler,
                    )
                )
            elif op.kind == OperandKind.MEM:
                insn_tokens.extend(
                    tokenize_operand_memory_base_disp(
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
                )
            elif op.kind == OperandKind.INVALID:
                pass
            else:
                raise ValueError(f"Unsupported RISC-V operand type: {op.type_int}")

        return insn_tokens
