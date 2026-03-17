from pathlib import Path
from typing import List, Literal

from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.arch.x86.operands import tokenize_operand_immediate, tokenize_operand_memory
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent / "data_store.json"

_DEGENERATE_PREFIXES: dict[int, list[str]] = {
    0xF2: ["repne", "repnz"],
    0xF3: ["repe", "repz", "rep"],  # ordering important due to string comparisons
}


class X86Provider(ArchitectureProvider):
    """Architecture provider for x86 and x64 platforms."""

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
        insn,
        lookup,
        text_end: int,
        text_start: int,
        vocab_manager: VocabularyManager,
        insn_tokens: List[Tokens],
    ) -> List[Tokens]:
        # Prefix handling
        for byte in insn.prefix:
            if byte in _DEGENERATE_PREFIXES:
                skip = True
                for prefix_name in _DEGENERATE_PREFIXES[byte]:
                    if insn.mnemonic.startswith(prefix_name):
                        token = vocab_manager.PlatformToken(prefix_name, PlatformInstructionTypes.PREFIXES)
                        insn_tokens.append(token)
                        break
                else:
                    skip = False
                if skip:
                    continue

            if byte in instr_sets.prefixes:
                prefix_name: str = instr_sets.prefixes[byte]
                token = vocab_manager.PlatformToken(prefix_name, PlatformInstructionTypes.PREFIXES)
                insn_tokens.append(token)

        # Mnemonic
        insn_name = insn.insn.insn_name()
        insn_type = instr_sets.get_instruction_type(insn_name)
        token = vocab_manager.PlatformToken(insn_name, insn_type)
        insn_tokens.append(token)

        # Operands
        if hasattr(insn, "operands"):
            for op in insn.operands:
                if op.type == 0 or op.type > 3:
                    raise Exception(f"Unsupported x86 operand type: {op.type}")

                if op.type == 1:  # register
                    token = vocab_manager.get_registry_token(insn, op.reg)
                    insn_tokens.append(token)
                elif op.type == 2:  # immediate
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
                elif op.type == 3:  # memory
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
        else:
            raise TypeError(f"INSTRUCTION WITHOUT OPERANDS: {insn}")

        return insn_tokens
