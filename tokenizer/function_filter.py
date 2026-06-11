"""Trivial-function filter — CURRENTLY UNUSED.

``FunctionFilter.filter_fns`` documents "returns true if function contains
only one jump instruction" but the filter contract was never implemented:
there is no reachable ``return True`` (the candidate paths only log
warnings), so a caller can never actually filter anything. The per-function
call in ``tokenizer/main_loop.py`` was removed because it burned two
``TokenPattern.match`` passes per function for an always-``False`` result.
Kept so the filter can be revived deliberately (with an implemented
contract) later.
"""

import logging

from tokenizer.disasm.types import InsnDebugLabel
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.patterns import (
    Block,
    BlockDef,
    InsnControlFlow,
    InsnNop,
    InsnPointerLengths,
    InsnPrefixes,
    InsnRegistry,
    Maybe,
    MemCloseBracket,
    MemOpenBracket,
    MemPlus,
    Multi,
    OpaqueConst,
    ValuedConst,
)
from tokenizer.patterns.matcher import TokenPattern
from tokenizer.token_manager import VocabularyManager


class FunctionFilter:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        # TokenPattern for a jump-only function:
        # Block_Def, Block(0), [RepeatType.MAYBE, PlatformInstructionTypes.PREFIXES], PlatformInstructionTypes.CONTROL_FLOW,
        # [RepeatType.MAYBE, PlatformInstructionTypes.POINTER_LENGTHS], MemoryOperandSymbol.OPEN_BRACKET,
        # PlatformInstructionTypes.OTHER, MemoryOperandSymbol.CLOSE_BRACKET
        self.jump_only_pattern = TokenPattern(
            BlockDef,
            Block + 0,
            Maybe + InsnPrefixes,
            InsnControlFlow,
            Maybe + InsnPointerLengths,
            MemOpenBracket,
            Maybe + [InsnRegistry, MemPlus],
            (OpaqueConst + 0) | ValuedConst,
            MemCloseBracket,
        )
        self.nop_only_pattern = TokenPattern(
            Maybe + InsnPrefixes,
            Multi + InsnNop,
            Maybe + [Maybe + InsnPointerLengths, MemOpenBracket, OpaqueConst + 0, MemCloseBracket],
        )

    def filter_fns(self, fn_tokens: FunctionTokenList, func_name, vm: VocabularyManager) -> bool:
        (
            """Returns true if function contains only one jump instruction.
        → Remove 'nop' (single and repetitions)
        → Remove "hlt
        """
            """
        if len(block_run_lengths) > 1:
            return False
        """
        )
        if fn_tokens.block_count > 1:
            return False

        if self.jump_only_pattern.match(fn_tokens.iter_raw_tokens(), vm):
            self.logger.warning(
                f"JUMP_ONLY func {func_name}: {fn_tokens.to_asm_original()} / {fn_tokens.to_asm_like()}"
            )

        if self.nop_only_pattern.match(fn_tokens.iter_raw_tokens(), vm):
            self.logger.warning(f"NOP_ONLY func {func_name}: {fn_tokens.to_asm_original()} / {fn_tokens.to_asm_like()}")

        # remove nop only and hlt
        if fn_tokens.block_count == 1:
            arr = fn_tokens.insn_strs[1 : fn_tokens.last_index]

            # Typed padding gate: an entry counts as padding iff it is a
            # bare (operand-less) nop/hlt/ret instruction label. Keyed on
            # the ``InsnDebugLabel`` typed fields rather than the rendered
            # ``"<mnemonic> <op_str>"`` string — the legacy string compare
            # (``insn_str == "nop "`` ⟺ mnemonic "nop" + empty op_str)
            # forced the per-operand text render in production. Non-label
            # entries (block/jump-table rows, unused-slack zeros) are not
            # padding, matching the legacy compare's False on those.
            allowed = ("nop", "hlt", "ret")
            is_padding = [
                isinstance(entry, InsnDebugLabel)
                and entry.operand_count == 0
                and entry.mnemonic in allowed
                for entry in arr
            ]

            if all(is_padding):
                self.logger.warning(
                    f"RETURN_ONLY func {func_name}: {fn_tokens.to_asm_original()} / {fn_tokens.to_asm_like()}"
                )

        # insn_mask = np.array([2, 7])
        # if insn_run_lengths.shape == (2,):
        #     if np.array_equal(insn_mask.flatten(), insn_run_lengths):
        #         jmp_indirect_pattern = re.compile(
        #             r'^jmp\s+'
        #             r'(?:[a-z]{1,8}word\s+)?'  # optional size prefix like dword/qword/xmmword
        #             r'ptr\s*'
        #             r'\[\s*[^]]+\s*\]$',  # everything inside brackets, anything except ]
        #             re.IGNORECASE
        #         )
        #         if str(fn_tokens.insn_strs[1]).split(" ")[0] in instr_sets.addressing_control_flow:
        #             print(f"\nREMOVED {func_name}: {fn_tokens.insn_strs}")
        #             return True

        return False


# --- TESTS ---
