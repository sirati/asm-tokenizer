from pathlib import Path
from typing import List

from tokenizer.arch.operands_base import tokenize_operand_immediate_generic, tokenize_operand_memory_base_disp
from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.resolved_target_policy import should_honor_resolved_target
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
                # RISC-V ``lui``/``addi`` high+low halves build an
                # absolute address; Ghidra attaches the combined resolved
                # data target as a DATA ref on the ``addi`` terminal's
                # destination REG operand. The keep/drop policy
                # (``resolved_target_policy``) owns the per-ISA pair-
                # terminal allow-list; mirror the arm32 REG-side consumer.
                resolved_target = op.resolved_target
                if resolved_target is not None:
                    meta = lookup.lookup(resolved_target)
                    if should_honor_resolved_target(
                        meta=meta,
                        resolved_target=resolved_target,
                        func_min_addr=func_min_addr,
                        func_max_addr=func_max_addr,
                        arch=reg.arch,
                        base_mnemonic=insn.base_mnemonic,
                        has_load_store=insn.has_load_store,
                    ):
                        insn_tokens.extend(
                            constant_handler.process_constant_v2(
                                resolved_target,
                                meta=meta,
                                is_arithmetic=False,
                            )
                        )
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
                        vocab_manager,
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
            elif op.kind == OperandKind.REG_LIST:
                # No reg-list family instructions exist on RISC-V (the
                # base ISA + standard extensions); the provider
                # classifier should never produce this kind on RISC-V.
                # Crash visibly rather than silently dropping.
                raise AssertionError(
                    f"Unexpected REG_LIST operand on RISC-V "
                    f"(no reg-list family instructions known); operand at "
                    f"insn 0x{insn.address:x}"
                )
            elif op.kind == OperandKind.INVALID:
                pass
            else:
                raise ValueError(f"Unsupported RISC-V operand type: {op.type_int}")

        return insn_tokens
