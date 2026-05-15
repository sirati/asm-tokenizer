from pathlib import Path
from typing import List

from tokenizer.arch.arm32.operands import (
    tokenize_operand_immediate,
    tokenize_operand_memory,
    tokenize_operand_reg_list,
    tokenize_operand_shift,
)
from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import (
    ArmConditionCode,
    ConditionCodePrefixView,
    InstructionView,
    OperandKind,
    UpdateFlagsPrefixView,
    WritebackPrefixView,
)
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens

_DATA_STORE_PATH = Path(__file__).parent / "data_store.json"

# Typed ``ArmConditionCode`` -> ARM asm-form suffix word. ``CS`` and
# ``CC`` map to the synonymous ``hs`` / ``lo`` spellings the pre-G.3
# consumer emitted via the legacy Capstone-int table; the v1 vocab is
# keyed against those spellings so we preserve them verbatim.
_ARM_CC_NAMES: dict[ArmConditionCode, str] = {
    ArmConditionCode.EQ: "eq",
    ArmConditionCode.NE: "ne",
    ArmConditionCode.CS: "hs",
    ArmConditionCode.CC: "lo",
    ArmConditionCode.MI: "mi",
    ArmConditionCode.PL: "pl",
    ArmConditionCode.VS: "vs",
    ArmConditionCode.VC: "vc",
    ArmConditionCode.HI: "hi",
    ArmConditionCode.LS: "ls",
    ArmConditionCode.GE: "ge",
    ArmConditionCode.LT: "lt",
    ArmConditionCode.GT: "gt",
    ArmConditionCode.LE: "le",
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
        insn: InstructionView,
        lookup,
        text_end: int,
        text_start: int,
        vocab_manager: VocabularyManager,
        insn_tokens: List[Tokens],
    ) -> List[Tokens]:
        # ARM32 has no byte-level prefixes like x86; the typed
        # ``insn.prefixes`` carries the cc / S / ! per-instruction
        # modifiers Capstone surfaces on the cs_insn struct.

        # Mnemonic (the base mnemonic without the cc suffix Capstone
        # glued on, matching the legacy ``insn.insn.insn_name()`` read)
        insn_name = insn.base_mnemonic
        insn_type = instr_sets.get_instruction_type(insn_name)
        insn_tokens.append(vocab_manager.PlatformToken(insn_name, insn_type))

        # Per-instruction modifiers via typed-prefix isinstance dispatch.
        # The prefix builder lives in `tokenizer/disasm/{angr,ghidra}_provider.py`
        # (and is `[]` for AL/always conditions, S=False, !=False). Walking
        # `insn.prefixes` once and dispatching on the typed subclasses
        # replaces the legacy hasattr+getattr probe over `insn.insn.cc /
        # update_flags / writeback`.
        for prefix in insn.prefixes:
            if isinstance(prefix, ConditionCodePrefixView):
                cc_name = _ARM_CC_NAMES.get(prefix.cc)
                if cc_name is not None:
                    insn_tokens.append(
                        vocab_manager.PlatformToken(cc_name, PlatformInstructionTypes.CONTROL_FLOW)
                    )
            elif isinstance(prefix, UpdateFlagsPrefixView):
                insn_tokens.append(vocab_manager.PlatformToken("s_flag", PlatformInstructionTypes.PREFIXES))
            elif isinstance(prefix, WritebackPrefixView):
                insn_tokens.append(
                    vocab_manager.PlatformToken("writeback", PlatformInstructionTypes.MEMORY_ACCESS_MODE)
                )

        # Operands
        for op in insn.operands:
            if op.kind == OperandKind.REG:
                reg = op.reg
                insn_tokens.append(vocab_manager.get_registry_token(reg.name, reg.id))
                # Shifted register operand
                shift_tokens = tokenize_operand_shift(insn, op, vocab_manager)
                insn_tokens.extend(shift_tokens)
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
            elif op.kind == OperandKind.REG_LIST:
                # ARM stm/ldm/push/pop/vpush/vpop/vstm/vldm family — the
                # provider classifies any operand with >= 3 Register objects
                # as REG_LIST so we never lose registers to silent truncation
                # in the MEM-decompose helpers.
                insn_tokens.extend(tokenize_operand_reg_list(op, vocab_manager))
            elif op.kind == OperandKind.INVALID:
                pass  # invalid/unused operand slot
            else:
                # OperandKind.OTHER — ARM FP / CIMM / PIMM / SETEND / SYSREG
                # passthrough; the original integer Capstone op-type is
                # preserved as ``op.type_int`` so the emit shape stays
                # identical to the legacy ``f"op_{op.type}"``.
                insn_tokens.append(
                    vocab_manager.PlatformToken(
                        f"op_{op.type_int}", PlatformInstructionTypes.SYSTEM
                    )
                )

        return insn_tokens
