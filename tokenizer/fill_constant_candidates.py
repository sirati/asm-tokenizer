from typing import Optional

import angr
import numpy as np

from tokenizer.address_meta_data_lookup import AddressMetaDataLookup
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.instruction_sets import InstructionSets
from tokenizer.op_imm_mem import tokenize_operand_immediate, tokenize_operand_memory
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import BlockToken, TokenResolver, Tokens

VERIFICATION: bool = False

degenerate_prefixes = {
    0xF2: ["repne", "repnz"],
    0xF3: ["repe", "repz", "rep"],
}


def parse_instruction(
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
):
    insn_tokens2 = [] if VERIFICATION else None

    for byte in insn.prefix:
        if byte in degenerate_prefixes:
            skip = True
            for prefix_name in degenerate_prefixes[byte]:
                if insn.mnemonic.startswith(prefix_name):
                    token = vocab_manager.PlatformToken(prefix_name, PlatformInstructionTypes.PREFIXES)
                    insn_tokens.append(token)
                    if VERIFICATION:
                        assert insn_tokens2 is not None
                        insn_tokens2.append(token)
                    break
            else:
                skip = False
            if skip:
                continue

        if byte in instr_sets.prefixes:
            prefix_name: str = instr_sets.prefixes[byte]
            token = vocab_manager.PlatformToken(prefix_name, PlatformInstructionTypes.PREFIXES)
            insn_tokens.append(token)
            if VERIFICATION:
                assert insn_tokens2 is not None
                insn_tokens2.append(token)

    insn_name = insn.insn.insn_name()
    insn_type = instr_sets.get_instruction_type(insn_name)

    token = vocab_manager.PlatformToken(insn_name, insn_type)
    insn_tokens.append(token)
    if VERIFICATION:
        assert insn_tokens2 is not None
        insn_tokens2.append(token)

    if hasattr(insn, "operands"):
        for op in insn.operands:
            if op.type == 0 or op.type > 3:
                raise Exception

            if op.type == 1:
                token = vocab_manager.get_registry_token(insn, op.reg)
                insn_tokens.append(token)
                if VERIFICATION:
                    assert insn_tokens2 is not None
                    insn_tokens2.append(token)
            elif op.type == 2:
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
                if VERIFICATION:
                    assert insn_tokens2 is not None
                    insn_tokens2.extend(immediate_tokens)

            elif op.type == 3:
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
                if VERIFICATION:
                    assert insn_tokens2 is not None
                    insn_tokens2.extend(memory_tokens)

    else:
        print(f"INSTRUCTION WITHOUT OPERANDS: {insn}")
        raise TypeError

    return insn_tokens, insn_tokens2


def fill_constant_candidates(
    func_addr: int,
    func: angr.knowledge_plugins.functions.function.Function,
    instr_sets: InstructionSets,
    constant_dict: dict[str, list[str]],
    lookup: AddressMetaDataLookup,
    text_start: int,
    text_end: int,
    resolver: TokenResolver,
    vocab_manager: VocabularyManager,
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

    num_blocks = len(list(func.blocks))
    block_ranges: np.ndarray = np.empty((num_blocks, 2), dtype=np.uint64)

    for i, block in enumerate(func.blocks):
        block_ranges[i, 0] = block.addr
        block_ranges[i, 1] = block.addr + block.size


    func_max_addr = int(block_ranges.max())
    constant_handler = ConstantHandler(vocab_manager, resolver, constant_dict, block_ranges)
    temp_bbs: list[tuple[str, list[list[Tokens]]]] = []
    block_list: list[dict[BlockToken, tuple[int, int]]] = []
    block_dict: dict[str, BlockToken] = {}

    num_blocks = sum(1 for _ in func.blocks)

    if num_blocks == 1 and not next(func.blocks).capstone.insns:
        return None

    func_tokens = FunctionTokenList(num_blocks, vocab_manager=vocab_manager)
    ordered_blocks = sorted(func.blocks, key=lambda b: b.addr)
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

            (insn_tokens, insn_tokens2) = parse_instruction(
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
                disassembly_list2.append(insn_tokens2)

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
