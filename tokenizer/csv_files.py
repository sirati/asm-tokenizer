import csv
import re
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray, base64_to_ndarray_vec
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.token_manager import VocabularyManager
from tokenizer.utils import register_name_range


# ---------------------------------------------------------------------------
# Wire-format version detection
# ---------------------------------------------------------------------------
#
# v2 tokenizer outputs prefix the CSV with a single-cell prelude row
# ``["version=2"]`` (see ``tokenizer.main_loop``). v1 outputs have no
# prelude and start directly with the header row.
#
# ``open_versioned_csv_reader`` is the single entry point all readers in
# this module use to obtain a ``csv.reader`` that has already consumed
# the prelude (if present) along with the detected version. Per-reader
# logic stays version-agnostic: positional column indices are unchanged
# between v1 and v2 (v2 only renamed the metadata column from
# ``opaque_metadata`` to ``metadata``).


def open_versioned_csv_reader(csvfile: TextIO) -> tuple["csv.reader", int]:
    """Return ``(reader, format_version)`` for an opened tokenizer CSV.

    Peeks the first row to detect the v2 ``version=2`` prelude. If
    present it is consumed and ``format_version == 2``; otherwise the
    row is replayed via a chained iterator and ``format_version == 1``
    so the caller sees the original first row at its first ``next()``.

    The returned object is a normal iterator yielding rows (it is not a
    real ``csv.reader`` instance after replay, but the iteration
    contract is identical, which is all callers use).
    """

    raw_reader = csv.reader(csvfile)
    try:
        first_row = next(raw_reader)
    except StopIteration:
        return raw_reader, 1

    if len(first_row) == 1 and first_row[0] == "version=2":
        # Prelude consumed; remaining rows start with the header.
        return raw_reader, 2

    # v1: replay the first row so callers' first ``next()`` sees it.
    return _chain_first_row(first_row, raw_reader), 1


def _chain_first_row(first_row: list[str], rest: Iterator[list[str]]) -> Iterator[list[str]]:
    yield first_row
    yield from rest


def extract_ldis_blocks_from_file(file_path):
    """
    Reads a structured CSV-like file and extracts disassembly blocks from <LDIS> tags.
    Returns a dict: function_name -> list of disassembled blocks.
    """
    file_path = Path(file_path)
    result = {}

    with file_path.open(encoding="utf-8") as f:
        reader, _format_version = open_versioned_csv_reader(f)
        for row in reader:
            if len(row) < 4:
                continue

            funcname = row[0]
            ldis_field = row[3]

            # Extract content between <LDIS> and </LDIS>
            match = re.search(r"<LDIS>(.*?)</LDIS>", ldis_field, flags=re.DOTALL)
            if match:
                ldis_text = match.group(1).strip()
                # Optional: split blocks if separated by "|"
                blocks = [b.strip() for b in ldis_text.split("|")]
                result[funcname] = blocks
    return result


def parse_init_sections(proj, output_txt="parsed_init_sections.txt", sections_to_parse=None):
    """
    Parse ELF .init/.fini/.init_array/.fini_array sections and write to file.

    Args:
        proj (angr.Project): Loaded angr project.
        output_txt (str): Output file to write parsed content.
        sections_to_parse (list[str], optional): Section names to parse. Defaults to init/fini types.

    Returns:
        list[dict]: list of parsed section entries.
    """
    if sections_to_parse is None:
        sections_to_parse = [".init", ".fini", ".init_array", ".fini_array"]

    entries = []

    with open(output_txt, "w") as f:
        f.write("# Parsed init/fini related sections\n")

        for section in proj.loader.main_object.sections:
            if section.name not in sections_to_parse:
                continue

            try:
                data = proj.loader.memory.load(section.vaddr, section.memsize)
            except Exception as e:
                print(f"Warning: could not read section {section.name}: {e}")
                continue

            if section.name.endswith("_array"):
                word_size = proj.arch.bytes
                for i in range(0, len(data), word_size):
                    chunk = data[i : i + word_size]
                    if len(chunk) != word_size:
                        continue
                    val = int.from_bytes(chunk, byteorder="little")
                    entry = {
                        "section": section.name,
                        "start": hex(section.vaddr + i),
                        "end": hex(section.vaddr + i + word_size),
                        "value": hex(val),
                        "type": "pointer",
                    }
                    entries.append(entry)
                    f.write(f"{entry['section']}, {entry['start']} - {entry['end']}: {entry['value']} (ptr)\n")
            else:
                hex_preview = data[:32].hex()
                entry = {
                    "section": section.name,
                    "start": hex(section.vaddr),
                    "end": hex(section.vaddr + section.memsize),
                    "value": f"hex({hex_preview}...)",
                    "type": "code",
                }
                entries.append(entry)
                f.write(f"{entry['section']}, {entry['start']} - {entry['end']}: {entry['value']} (code)\n")

    print(f"Parsed {len(entries)} entries from init-related sections into {output_txt}")
    return entries


def reverse_tokenization(
    tokenized_instructions: np.ndarray, block_run_lengths: list[int], insn_run_lengths: list[int], vocab: dict[int, str]
) -> list[dict[str, list[str]]]:
    instructions = []
    token_index = 0
    # Step 1: Convert tokens into instructions
    for insn_len in insn_run_lengths:
        insn_tokens = []

        for _ in range(insn_len):
            token_id = int(tokenized_instructions[token_index])
            # """if vocab[token_id] == "VALUED_CONST_34":
            #     print(f"token_id={token_id}, token={vocab[token_id]}")
            #     print(f"Tokenized instructions: {tokenized_instructions}")
            #     return None"""
            insn_tokens.append(vocab[token_id])
            token_index += 1
        instructions.append(insn_tokens)

    # print(instructions)

    # Step 2: Group instructions into blocks
    block_insns = []

    block_index = 0
    j = 0  # index over all instructions
    for block_len in block_run_lengths:
        i = 0  # index that is being reset for each block
        block_instrs = []
        # print(block_len)
        while i < block_len:
            block_instrs.append(" ".join(instructions[j]))
            i += len(instructions[j])
            # print(f"\t{i}")
            j += 1
        if block_index < 16:
            block_insns.append({f"Block_{hex(block_index)[2:].upper()}": block_instrs})
        else:
            block_insns.append({f"{register_name_range(block_index, basename='Block')}": block_instrs})
        block_index += 1

    # print(block_insns)
    return block_insns


def vocab_from_output(output_path: str) -> list[str]:
    with open(output_path, newline="") as csvfile:
        reader, _format_version = open_versioned_csv_reader(csvfile)
        csv_iter = iter(reader)
        vocab: list[str] = []
        for func_name, token in enumerate(next(csv_iter)[6][1:-1].split(",")):
            vocab.append(token)
    return vocab


def token_to_insn(input_path: str, output_path: str, vocab_manager: VocabularyManager):
    with open(input_path, newline="") as csvfile:
        reader, _format_version = open_versioned_csv_reader(csvfile)
        token_list: list[tuple[str, str]] = []
        vocab: dict[int, str] = {}
        csv_iter = iter(reader)
        for func_name, token in enumerate(next(csv_iter)[6][1:-1].split(",")):
            vocab[func_name] = token

        for row in reader:
            function_name = row[0]
            print(f"Function name: {function_name}")

            tokens = base64_to_ndarray_vec(row[2])
            block_runlength = base64_to_ndarray_vec(row[3])
            insn_runlength = base64_to_ndarray_vec(row[4])
            # string_stream = reverse_tokenization(tokens, block_runlength, insn_runlength, vocab)
            # reconst = FunctionTokenList.reconstruct_func_from_raw_bytes(
            #     tokens, block_runlength, insn_runlength, vocab_manager
            # )
            # todo

            token_list.append((function_name, string_stream))

    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for k, v in token_list:
            writer.writerow([k, v])


def token_to_functions(input_path: str):
    with open(input_path, newline="") as csvfile:
        reader, _format_version = open_versioned_csv_reader(csvfile)

        vocab = []
        for token in next(iter(reader))[6][1:-1].split(","):
            vocab.append(token)
        token_man = VocabularyManager.from_vocab(platform="x86", vocab_list=vocab)

        for row in reader:
            function_name = row[0]
            function_duplicate = int(row[1])
            print(f"Function name: {function_name} ({function_duplicate})")

            tokens = base64_to_ndarray_vec(row[2])
            block_runlength = base64_to_ndarray_vec(row[3])
            insn_runlength = base64_to_ndarray_vec(row[4])
            # string_stream = reverse_tokenization(tokens, block_runlength, insn_runlength, vocab)
            reconst = FunctionTokenList.reconstruct_func_from_raw_bytes(
                tokens, block_runlength, insn_runlength, token_man
            )
            yield (function_name, function_duplicate, reconst)


def datastructures_to_insn(
    vocab: dict[int, str],
    token_dict: dict[str, str],
    block_runlength_dict: dict[str, str],
    insn_runlength_dict: dict[str, str],
    duplicate_map: dict[str, str],
):
    reconstructed: dict[str, str] = {}
    vocab = {v: k for k, v in vocab.items()}

    for index in token_dict:
        try:
            # Resolve duplicates (use original name if it's a duplicate)
            original_index = duplicate_map.get(index, index)

            tokens = base64_to_ndarray(token_dict[index])
            block_runlength = base64_to_ndarray(block_runlength_dict[index])
            insn_runlength = base64_to_ndarray(insn_runlength_dict[index])

            string_stream = reverse_tokenization(tokens, block_runlength, insn_runlength, vocab)
            reconstructed[original_index] = string_stream

        except Exception as e:
            print(f"❌ Failed to process index {index}: {e}")

    # Write the result to a CSV
    with open("reconstructed_disassembly_test.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for k, v in reconstructed.items():
            writer.writerow([k, v])


def compare_csv_files(file1: str, file2: str):
    # Increase CSV field size limit
    csv.field_size_limit(10_000_000)

    with open(file1, newline="", encoding="utf-8") as f1, open(file2, newline="", encoding="utf-8") as f2:
        reader1, _v1 = open_versioned_csv_reader(f1)
        reader2, _v2 = open_versioned_csv_reader(f2)

        line_num = 1
        for row1, row2 in zip(reader1, reader2):
            if row1 != row2:
                print(f"Mismatch at line {line_num}:")
                print(f"  {file1}: {row1}")
                print(f"  {file2}: {row2}")
                raise ValueError
            line_num += 1

        for row in reader1:
            print(f"Extra line in {file1} at line {line_num}: {row}")
            line_num += 1

        for row in reader2:
            print(f"Extra line in {file2} at line {line_num}: {row}")
            line_num += 1


def csv_to_dict(filepath):
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader, _format_version = open_versioned_csv_reader(f)
        for row in reader:
            if len(row) != 2:
                continue  # skip malformed lines
            key, value = row[0].strip(), row[1].strip()
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value  # fallback if not int
    return result
