from pathlib import Path
from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic, tokenize_operand_memory_base_disp
from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import (
    InstructionView,
    OperandKind,
    PpcBranchConditionPrefixView,
    PpcUpdateCr0PrefixView,
)
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent / "data_store.json"

# PowerPC branch condition codes (Capstone ppc_bc enum value -> asm word).
# The typed `PpcBranchConditionPrefixView` carries the same Capstone int
# as ``.bc`` for behavior preservation; we index this table directly.
_PPC_BC_NAMES: dict[int, str] = {
    1: "lt",
    2: "le",
    3: "eq",
    4: "ge",
    5: "gt",
    6: "ne",
    7: "un",
    8: "nu",
    9: "so",
    10: "ns",
}


class PPCProvider(ArchitectureProvider):
    """Architecture provider for PowerPC (32-bit and 64-bit) platforms."""

    def __init__(self, platform: str = "ppc32"):
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

        # PPC per-instruction modifiers (branch-condition `bc`, CR0
        # update Rc bit) flow through `insn.prefixes` as typed instances.
        # The prefix builder emits `PpcBranchConditionPrefix(bc=int)` and
        # `PpcUpdateCr0PrefixView()` (data-less marker); isinstance
        # dispatch replaces the legacy hasattr+getattr probes over
        # `insn.insn.bc` / `insn.insn.update_cr0`.
        for prefix in insn.prefixes:
            if isinstance(prefix, PpcBranchConditionPrefixView):
                bc_name = _PPC_BC_NAMES.get(prefix.bc)
                if bc_name is not None:
                    insn_tokens.append(
                        vocab_manager.PlatformToken(bc_name, PlatformInstructionTypes.CONTROL_FLOW)
                    )
            elif isinstance(prefix, PpcUpdateCr0PrefixView):
                insn_tokens.append(vocab_manager.PlatformToken("rc", PlatformInstructionTypes.PREFIXES))

        for op in insn.operands:
            if op.kind == OperandKind.REG:
                reg = op.reg
                if not reg.is_absent:
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
            elif op.kind == OperandKind.CRX:
                # Condition register field — emit the CR register name.
                crx_reg = op.crx.reg
                if not crx_reg.is_absent:
                    insn_tokens.append(
                        vocab_manager.get_registry_token(crx_reg.name, crx_reg.id)
                    )
            elif op.kind == OperandKind.REG_LIST:
                # No reg-list family instructions exist on PPC; the
                # provider classifier should never produce this kind on
                # PPC. Crash visibly rather than silently dropping.
                raise AssertionError(
                    f"Unexpected REG_LIST operand on PPC "
                    f"(no reg-list family instructions known); operand at "
                    f"insn 0x{insn.address:x}"
                )
            elif op.kind == OperandKind.INVALID:
                pass
            else:
                raise ValueError(f"Unsupported PPC operand type: {op.type_int}")

        return insn_tokens
