from typing import Any, Optional

import numpy as np

from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm import MetadataLookup
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import BlockToken, TokenResolver, Tokens

VERIFICATION: bool = False


def fill_constant_candidates(
    func_addr: int,
    func: Any,
    instr_sets: InstructionSets,
    constant_dict: dict[str, list[str]],
    lookup: MetadataLookup,
    text_start: int,
    text_end: int,
    resolver: TokenResolver,
    vocab_manager: VocabularyManager,
    arch_provider: ArchitectureProvider,
) -> Optional[
    tuple[
        list[tuple[str, list[list[Tokens]]]],
        list[dict[BlockToken, tuple[str, str]]],
        dict[str, BlockToken],
        ConstantHandler,
        FunctionTokenList,
    ]
]:
    func_min_addr: int = int(func_addr)
    blocks: set = set()

    block_objs = list(func.blocks)
    num_blocks = len(block_objs)
    block_ranges: np.ndarray = np.empty((num_blocks, 2), dtype=np.uint64)

    for i, block in enumerate(block_objs):
        block_ranges[i, 0] = block.addr
        block_ranges[i, 1] = block.addr + block.size

    func_max_addr = int(block_ranges.max())
    constant_handler = ConstantHandler(vocab_manager, resolver, constant_dict, block_ranges)
    temp_bbs: list[tuple[str, list[list[Tokens]]]] = []
    block_list: list[dict[BlockToken, tuple[int, int]]] = []
    block_dict: dict[str, BlockToken] = {}

    if num_blocks == 1 and not block_objs[0].capstone.insns:
        return None

    func_tokens = FunctionTokenList(num_blocks, vocab_manager=vocab_manager)
    ordered_blocks = sorted(block_objs, key=lambda b: b.addr)
    for block in ordered_blocks:
        func_max_addr = max(block.addr, block.addr + block.size)

        block_addr = hex(block.addr)
        block_id = resolver.get_block_id(block_addr)
        block_token = vocab_manager.Block(block_id)
        block_list.append(
            {
                block_token: (
                    block.addr,
                    block.addr + block.size,
                )
            }
        )
        blocks.add(block_addr)

        assert block.capstone.insns is not None, "Block has no instructions, cannot disassemble"

        block_dict[block_addr] = block_token

        block_def = [vocab_manager.Block_Def(), block_token]

        disassembly_list = BlockTokenList(len(block.capstone.insns) + 1, vocab_manager=vocab_manager)
        disassembly_list.append_as_insn(insn_str=f"block {block_addr}", tokens=block_def)

        disassembly_list2 = [block_def]

        for insn in block.capstone.insns:
            insn_tokens = disassembly_list.view(insn_str=f"{insn.mnemonic} {insn.op_str}")

            insn_tokens = arch_provider.parse_instruction(
                instr_sets,
                constant_handler,
                func_max_addr,
                func_min_addr,
                insn,
                lookup,
                text_end,
                text_start,
                vocab_manager,
                insn_tokens,
            )
            disassembly_list.add_insn(insn_tokens)
            if VERIFICATION:
                disassembly_list2.append(list(insn_tokens))

        if VERIFICATION:
            for x, y in zip(
                [token for insn in disassembly_list2 for token in insn],
                disassembly_list.iter_raw_tokens(),
            ):
                if x != y:
                    print(f"Token mismatch: {x} != {y}")
                    raise ValueError("Token mismatch in disassembly list")

        if VERIFICATION:
            temp_bbs.append((block_addr, disassembly_list2))
        func_tokens.add_block(disassembly_list, block_addr)
    return (
        temp_bbs,
        block_list,
        block_dict,
        constant_handler,
        func_tokens,
    )
