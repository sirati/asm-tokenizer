import csv
import struct

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec


def decode_and_translate_tokens(row, mapping=None):
    tokens = base64_to_ndarray_vec(row["tokens_base64"])
    if mapping is not None:
        tokens = mapping[tokens]
    return tokens.astype(np.uint16)


def decode_runlengths(row):
    block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
    insn_runlength = base64_to_ndarray_vec(row["instruction_runlength_base64"])
    return block_runlength, insn_runlength


def write_function_binary_data(
    file2, tokens, block_runlength, insn_runlength, dedup_cache=None
):
    cache_key = None
    if dedup_cache is not None:
        cache_key = (
            tokens.tobytes(),
            block_runlength.tobytes(),
            insn_runlength.tobytes(),
        )
        if cache_key in dedup_cache:
            return dedup_cache[cache_key]

    data_offset = file2.tell()
    insn_bytes = insn_runlength.astype(np.uint8).tobytes()
    block_enc = (
        0
        if block_runlength.dtype == np.uint8
        else (1 if block_runlength.dtype == np.uint16 else 2)
    )
    block_bytes = block_runlength.astype(
        [np.uint8, np.uint16, np.uint32][block_enc]
    ).tobytes()
    insn_len = len(insn_bytes)
    file2.write(struct.pack("<I", insn_len)[0:3])
    file2.write(struct.pack("B", block_enc))
    file2.write(struct.pack("<H", len(block_bytes)))
    file2.write(insn_bytes)
    file2.write(block_bytes)
    file2.write(tokens.tobytes())
    data_len = file2.tell() - data_offset
    result = (data_offset, data_len)

    if dedup_cache is not None:
        dedup_cache[cache_key] = result

    return result


def write_index_entry(file3, start, length, avg_len):
    file3.write(struct.pack("<I", start))
    file3.write(struct.pack("<I", length)[0:3])
    avg_len_clamped = min(avg_len >> 4, 255)
    file3.write(struct.pack("B", avg_len_clamped))


def format_inlining_dict(inlining_list):
    """Format inlining data as semicolon-separated: idx,hex_offset,hex_length,is_matched;..."""
    if not inlining_list:
        return ""
    parts = []
    for idx, start, length, is_matched in inlining_list:
        hex_start = f"{start:x}"
        hex_length = f"{length:x}"
        parts.append(f"{idx},{hex_start},{hex_length},{is_matched}")
    return ";".join(parts)


def format_compiler_sets(compiler_sets):
    """Format list of compiler set tuples using semicolon separation: arch,compiler,version,opt;..."""
    if not compiler_sets:
        return ""
    parts = []
    for arch, compiler, compilerversion, opt in compiler_sets:
        parts.append(f"{arch},{compiler},{compilerversion},{opt}")
    return ";".join(parts)


def format_unique_called(unique_called):
    """Format list of function names, comma-separated with escaped commas"""
    escaped = [name.replace(",", "\\,") for name in unique_called]
    return ",".join(escaped)


def write_function_section_csv(
    writer,
    arch,
    compiler,
    compilerversion,
    opt,
    inlining_list,
    data_offset,
    data_len,
):
    inlining_str = format_inlining_dict(inlining_list)
    writer.writerow(
        [
            arch,
            compiler,
            compilerversion,
            opt,
            inlining_str,
            f"{data_offset:x}",
            f"{data_len:x}",
        ]
    )


def write_unmatched_section_csv(
    writer,
    func_name,
    platform_tuples,
    called_functions_str,
    inlining_data_str,
    data_offset,
    data_len,
):
    """
    Write unmatched section row with format:
    function_name,"compiler_sets","called_functions","inlining_data",data_offset,data_len
    """
    platform_str = format_compiler_sets(platform_tuples)
    writer.writerow(
        [
            func_name,
            platform_str,
            called_functions_str,
            inlining_data_str,
            f"{data_offset:x}",
            f"{data_len:x}",
        ]
    )


def read_index_file(index_path):
    """Read the index file and yield (start, length, avg_len) for each function."""
    with open(index_path, "rb") as f:
        while True:
            start_bytes = f.read(4)
            if not start_bytes or len(start_bytes) < 4:
                break
            start = int.from_bytes(start_bytes, "little")
            length_bytes = f.read(3)
            if not length_bytes or len(length_bytes) < 3:
                break
            length = int.from_bytes(length_bytes, "little")
            avg_len_byte = f.read(1)
            if not avg_len_byte or len(avg_len_byte) < 1:
                break
            avg_len = int.from_bytes(avg_len_byte, "little")
            yield (start, length, avg_len)


def read_sections_file(sections_path):
    """Read the sections CSV file and yield (func_name, [rows]) for each function section."""
    with open(sections_path, newline="", encoding="ascii") as f:
        reader = csv.reader(f)
        func_name = None
        rows = []
        for row in reader:
            if not row or (len(row) == 1 and row[0]):
                # New section or blank line
                if func_name is not None and rows:
                    yield (func_name, rows)
                func_name = row[0] if row and row[0] else None
                rows = []
            elif func_name:
                rows.append(row)
        if func_name and rows:
            yield (func_name, rows)


def read_data_file(data_path, offset, length):
    """Read the binary data for a function from the data file given offset and length."""
    with open(data_path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
        # Parse header
        insn_len = int.from_bytes(data[0:3], "little")
        block_enc = data[3]
        block_len = int.from_bytes(data[4:6], "little")
        insn_runlength = np.frombuffer(data[6 : 6 + insn_len], dtype=np.uint8)
        block_dtype = [np.uint8, np.uint16, np.uint32][block_enc]
        block_runlength = np.frombuffer(
            data[6 + insn_len : 6 + insn_len + block_len], dtype=block_dtype
        )
        tokens = np.frombuffer(data[6 + insn_len + block_len :], dtype=np.uint16)
        return insn_runlength, block_runlength, tokens


def read_function_data_memmap(data_path, offset, length):
    """
    Read the binary data for a function from the data file using numpy.memmap for random access.
    Returns: insn_runlength, block_runlength, tokens
    """
    data = np.memmap(
        data_path, dtype=np.uint8, mode="r", offset=offset, shape=(length,)
    )
    return parse_function_data_header(data)


def parse_function_data_header(data_bytes):
    """
    Parse the header and return (insn_runlength, block_runlength, tokens) ndarrays.
    data_bytes: bytes or 1D uint8 array
    """
    if isinstance(data_bytes, np.memmap) or isinstance(data_bytes, np.ndarray):
        data_bytes = data_bytes.tobytes()
    insn_len = int.from_bytes(data_bytes[0:3], "little")
    block_enc = data_bytes[3]
    block_len = int.from_bytes(data_bytes[4:6], "little")
    insn_runlength = np.frombuffer(data_bytes[6 : 6 + insn_len], dtype=np.uint8)
    block_dtype = [np.uint8, np.uint16, np.uint32][block_enc]
    block_runlength = np.frombuffer(
        data_bytes[6 + insn_len : 6 + insn_len + block_len], dtype=block_dtype
    )
    tokens = np.frombuffer(data_bytes[6 + insn_len + block_len :], dtype=np.uint16)
    return insn_runlength, block_runlength, tokens
