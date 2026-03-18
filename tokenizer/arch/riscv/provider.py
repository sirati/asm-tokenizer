from pathlib import Path
from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic, tokenize_operand_memory_base_disp
from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.constant_handler import ConstantHandler
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent / "data_store.json"

_OP_REG = 1
_OP_IMM = 2
_OP_MEM = 3


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
        insn,
        lookup,
        text_end: int,
        text_start: int,
        vocab_manager: VocabularyManager,
        insn_tokens: List[Tokens],
    ) -> List[Tokens]:
        insn_name = insn.insn.insn_name()
        insn_type = instr_sets.get_instruction_type(insn_name)
        insn_tokens.append(vocab_manager.PlatformToken(insn_name, insn_type))

        if hasattr(insn, "operands"):
            for op in insn.operands:
                if op.type == _OP_REG:
                    insn_tokens.append(vocab_manager.get_registry_token(insn.reg_name(op.reg), op.reg))
                elif op.type == _OP_IMM:
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
                elif op.type == _OP_MEM:
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
                elif op.type == 0:
                    pass
                else:
                    raise ValueError(f"Unsupported RISC-V operand type: {op.type}")
        else:
            raise TypeError(f"INSTRUCTION WITHOUT OPERANDS: {insn}")

        return insn_tokens
