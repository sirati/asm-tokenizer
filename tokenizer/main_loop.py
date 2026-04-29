import csv
import logging
import time

import numpy as np
from tqdm import tqdm

from dynamic_runner.comm import CommunicationInterface, KeepaliveResponse, PhaseUpdateResponse
from dynamic_batch_tokenizer import TokenizerPhase
from tokenizer.compact_base64_utils import base64_to_ndarray_vec, ndarray_to_base64
from tokenizer.fill_constant_candidates import fill_constant_candidates
from tokenizer.function_data_manager import FunctionData, FunctionDataManager
from tokenizer.function_filter import FunctionFilter
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.opaque_remapping import (
    apply_opaque_mapping,
    apply_opaque_mapping_raw_optimized,
)
from tokenizer.vocab_unifier import save_vocabulary

VERIFICATION: bool = False


def build_vocab_tokenize_and_index(
    func_tokens: FunctionTokenList,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if func_tokens.last_index == 0:
        return (
            np.array([], dtype=np.int_),
            np.array([], dtype=np.int_),
            np.array([], dtype=np.int_),
        )

    (token_ids, _, _, _, insn_idx_run_lengths, _, block_insn_run_lengths, _, _) = func_tokens.get_used_arrays()

    block_insn_split_start_indicies = np.cumsum(np.insert(block_insn_run_lengths[:-1], 0, 0))
    block_idx_run_lengths = np.add.reduceat(insn_idx_run_lengths, block_insn_split_start_indicies)

    return token_ids, block_idx_run_lengths, insn_idx_run_lengths


def main_loop(
    instr_sets,
    provider,
    constant_list,
    func_addr_range,
    func_disas,
    func_disas_token,
    func_name_addr,
    func_names,
    lookup,
    resolver,
    text_end,
    text_start,
    vocab_manager,
    csv_path,
    arch_provider,
    logger: logging.Logger,
    comm: CommunicationInterface,
    **_kwargs,
) -> tuple[FunctionDataManager, int]:
    logger.info("Preparing main loop")

    filter = FunctionFilter(logger)

    total_functions = provider.function_count()
    function_manager = FunctionDataManager(total_functions) if VERIFICATION else FunctionDataManager(0)

    exceptions = []
    filtered_count = 0
    last_keepalive_time = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        print("WRITING OUTPUT")
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "function_name",
                "occurrence",
                "tokens_base64",
                "block_runlength_base64",
                "instruction_runlength_base64",
                "opaque_metadata",
            ]
        )

        occurence = 0
        prev_func_name = ""
        prev_tokens_base64 = ""
        prev_block_base64 = ""
        prev_insn_base64 = ""

        logger.info("Starting main loop")
        comm.send_response(PhaseUpdateResponse(phase_name=TokenizerPhase.TOKENIZATION.value))

        try:
            pbar = tqdm(
                iterable=provider.iter_functions(),
                total=total_functions,
                desc="Retrieving data from alllll functions. Like a big boy.",
            )
            for i, (func_addr, func_name, func) in enumerate(pbar):
                current_time = time.time()
                if (current_time - last_keepalive_time) >= 0.2:
                    comm.send_response(KeepaliveResponse())
                    last_keepalive_time = current_time

                resolver.reset()

                try:
                    (function_analysis) = fill_constant_candidates(
                        func_addr=func_addr,
                        func=func,
                        instr_sets=instr_sets,
                        constant_dict=constant_list,
                        lookup=lookup,
                        text_start=text_start,
                        text_end=text_end,
                        resolver=resolver,
                        vocab_manager=vocab_manager,
                        arch_provider=arch_provider,
                    )
                except Exception as e:
                    logger.warning(f"Error processing {func_name}: {e}. Skipping function.")
                    exceptions.append(e)
                    continue

                if function_analysis is None:
                    continue

                (temp_bbs, block_list, block_dict, constant_handler, func_tokens) = function_analysis

                func_addr_range[func_addr] = sorted(block_list, key=lambda d: list(d.values())[0][0])

                opaque_mapping = constant_handler.create_opaque_mapping()

                if len(opaque_mapping) > 0:
                    func_tokens = apply_opaque_mapping_raw_optimized(
                        func_tokens, opaque_mapping, vocab_manager, constant_handler
                    )
                    if VERIFICATION:
                        temp_bbs = apply_opaque_mapping(temp_bbs, opaque_mapping, constant_handler=None)

                if VERIFICATION:
                    for x, y in zip(
                        [token for (_, block) in temp_bbs for insn in block for token in insn],
                        func_tokens.iter_raw_tokens(),
                    ):
                        if x != y:
                            print(f"Token mismatch: {x} != {y}")
                            raise ValueError("Token mismatch in disassembly list")

                meta_result = constant_handler.get_metadata_list_by_opaque_id()

                tokenized_instructions, block_run_lengths, insn_run_lengths = build_vocab_tokenize_and_index(
                    func_tokens
                )

                if len(tokenized_instructions) == 0:
                    continue

                try:
                    tokens_base64 = ndarray_to_base64(tokenized_instructions)
                    block_base64 = ndarray_to_base64(block_run_lengths)
                    insn_base64 = ndarray_to_base64(insn_run_lengths)
                    if prev_func_name == func_name:
                        occurence += 1
                    else:
                        occurence = 0
                    writer = csv.writer(csvfile)
                    if filter.filter_fns(func_tokens, func_name, vocab_manager):
                        occurence -= 1
                        filtered_count += 1
                        continue

                    if (
                        prev_block_base64 == block_base64
                        and prev_insn_base64 == insn_base64
                        and prev_tokens_base64 == tokens_base64
                    ):
                        occurence -= 1
                        continue

                    row = [
                        func_name,
                        occurence,
                        tokens_base64,
                        block_base64,
                        insn_base64,
                        str(repr(meta_result)),
                    ]

                    writer.writerow(row)
                    prev_func_name = func_name

                    prev_tokens_base64 = tokens_base64
                    prev_block_base64 = block_base64
                    prev_insn_base64 = insn_base64

                    if i & 16383 == 16383:
                        save_vocabulary(vocab_manager, writer)

                    if i & 255 == 255:
                        csvfile.flush()

                    if VERIFICATION:
                        assert np.all(base64_to_ndarray_vec(tokens_base64) == tokenized_instructions), (
                            "Base64 conversion failed for tokens"
                        )
                        assert np.all(base64_to_ndarray_vec(block_base64) == block_run_lengths), (
                            "Base64 conversion failed for block run lengths"
                        )
                        assert np.all(base64_to_ndarray_vec(insn_base64) == insn_run_lengths), (
                            "Base64 conversion failed for instruction run lengths"
                        )
                        function_data = FunctionData(
                            tokens=func_tokens,
                            tokens_base64=tokens_base64,
                            block_runlength_base64=block_base64,
                            instruction_runlength_base64=insn_base64,
                            opaque_metadata=repr(meta_result),
                        )
                        final_func_name = function_manager.add_function_data(
                            func_name, func_addr, temp_bbs, func_tokens, function_data
                        )

                        func_name_addr[final_func_name] = func_addr
                        func_disas[final_func_name] = temp_bbs
                        func_disas_token[final_func_name] = func_tokens
                        func_names.append(final_func_name)
                except Exception as e:
                    logger.warning(
                        f"Error saving {func_name}: {e}.\n"
                        f"Tokenstream: {func_tokens}\n"
                        f"Tokens: {tokenized_instructions}\n"
                        f"Block encoding: {block_run_lengths}\n"
                        f"Instructions: {insn_run_lengths}\n"
                        f"MetaData: {str(meta_result)}"
                    )
                    exceptions.append(e)
                    continue
        except Exception as e:
            print(f"Unrecoverable error in main loop: {e}, writing what we have at least")
            exceptions.append(e)

        comm.send_response(KeepaliveResponse())

        save_vocabulary(vocab_manager, writer)
        csvfile.flush()

    if len(exceptions) > 0:
        all_exection_string = "\n".join([str(e) for e in exceptions])
        raise Exception(f"Errors occurred during disassembly:\n{all_exection_string}") from exceptions[-1]

    if VERIFICATION:
        function_manager.compact_arrays()

    return function_manager, filtered_count
