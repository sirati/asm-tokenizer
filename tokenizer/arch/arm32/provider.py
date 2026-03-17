from pathlib import Path
from typing import List

from tokenizer.arch.arm32.operands import (
    tokenize_operand_immediate,
    tokenize_operand_memory,
    tokenize_operand_shift,
)
from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent / "data_store.json"

# ARM32 Capstone operand types
_ARM_OP_REG = 1
_ARM_OP_IMM = 2
_ARM_OP_MEM = 3

# ARM32 condition codes from Capstone
_ARM_CC_NAMES: dict[int, str] = {
    1: "eq",
    2: "ne",
    3: "hs",
    4: "lo",
    5: "mi",
    6: "pl",
    7: "vs",
    8: "vc",
    9: "hi",
    10: "ls",
    11: "ge",
    12: "lt",
    13: "gt",
    14: "le",
    # 15 = AL (always) — omitted, it's the default
}


class ARM32Provider(ArchitectureProvider):
    """Architecture provider for ARM32 (AArch32 / Thumb) and ARM64 (AArch64) platforms."""

    def __init__(self, platform: str = "arm32"):
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
        # ARM32 has no byte-level prefixes like x86
        # Condition codes are part of the mnemonic in Capstone output

        # Mnemonic
        insn_name = insn.insn.insn_name()
        insn_type = instr_sets.get_instruction_type(insn_name)
        token = vocab_manager.PlatformToken(insn_name, insn_type)
        insn_tokens.append(token)

        # Condition code (if not AL/always)
        if hasattr(insn.insn, "cc") and insn.insn.cc != 0 and insn.insn.cc != 15:
            cc_name = _ARM_CC_NAMES.get(insn.insn.cc)
            if cc_name is not None:
                insn_tokens.append(vocab_manager.PlatformToken(cc_name, PlatformInstructionTypes.CONTROL_FLOW))

        # Update-flags (S suffix)
        if hasattr(insn.insn, "update_flags") and insn.insn.update_flags:
            insn_tokens.append(vocab_manager.PlatformToken("s_flag", PlatformInstructionTypes.PREFIXES))

        # Write-back (! suffix on base register)
        if hasattr(insn.insn, "writeback") and insn.insn.writeback:
            insn_tokens.append(vocab_manager.PlatformToken("writeback", PlatformInstructionTypes.MEMORY_ACCESS_MODE))

        # Operands
        if hasattr(insn, "operands"):
            for op in insn.operands:
                if op.type == _ARM_OP_REG:
                    token = vocab_manager.get_registry_token(insn, op.reg)
                    insn_tokens.append(token)
                    # Shifted register operand
                    shift_tokens = tokenize_operand_shift(insn, op, vocab_manager)
                    insn_tokens.extend(shift_tokens)
                elif op.type == _ARM_OP_IMM:
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
                elif op.type == _ARM_OP_MEM:
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
                elif op.type == 0:
                    pass  # invalid/unused operand slot
                else:
                    # FP, CIMM, PIMM, SETEND, SYSREG — emit as platform tokens
                    insn_tokens.append(vocab_manager.PlatformToken(f"op_{op.type}", PlatformInstructionTypes.SYSTEM))
        else:
            raise TypeError(f"INSTRUCTION WITHOUT OPERANDS: {insn}")

        return insn_tokens
