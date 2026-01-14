import csv
import json
import struct
import numpy as np
from tokenizer.compact_base64_utils import base64_to_ndarray_vec

def decode_and_translate_tokens(row, mapping=None):
    tokens = base64_to_ndarray_vec(row['tokens_base64'])
    if mapping is not None:
        tokens = mapping[tokens]
    return tokens.astype(np.uint16)


def decode_runlengths(row):
    block_runlength = base64_to_ndarray_vec(row['block_runlength_base64'])
    insn_runlength = base64_to_ndarray_vec(row['instruction_runlength_base64'])
    return block_runlength, insn_runlength


def write_function_binary_data(file2, tokens, block_runlength, insn_runlength):
    data_offset = file2.tell()
    insn_bytes = insn_runlength.astype(np.uint8).tobytes()
    block_enc = 0 if block_runlength.dtype == np.uint8 else (1 if block_runlength.dtype == np.uint16 else 2)
    block_bytes = block_runlength.astype([np.uint8, np.uint16, np.uint32][block_enc]).tobytes()
    file2.write(struct.pack('B', len(insn_bytes)))
    file2.write(struct.pack('B', block_enc))
    file2.write(struct.pack('<H', len(block_bytes)))
    file2.write(insn_bytes)
    file2.write(block_bytes)
    file2.write(tokens.tobytes())
    data_len = file2.tell() - data_offset
    return data_offset, data_len


def write_index_entry(file3, start, length, avg_len, always_zero=False):
    file3.write(struct.pack('<I', start))
    file3.write(struct.pack('<I', length)[0:3])
    file3.write(struct.pack('B', 0 if always_zero else (avg_len >> 4)))


def write_function_section_csv(writer, func_name, arch, compiler, compilerversion, opt, called, inlining_map, data_offset, data_len):
    writer.writerow([
        func_name,
        arch, compiler, compilerversion, opt,
        json.dumps(called),
        json.dumps(inlining_map),
        data_offset, data_len
    ])


def read_index_file(index_path):
    """Read the index file and yield (start, length, avg_len) for each function."""
    with open(index_path, 'rb') as f:
        while True:
            start_bytes = f.read(4)
            if not start_bytes or len(start_bytes) < 4:
                break
            start = int.from_bytes(start_bytes, 'little')
            length_bytes = f.read(3)
            if not length_bytes or len(length_bytes) < 3:
                break
            length = int.from_bytes(length_bytes, 'little')
            avg_len_byte = f.read(1)
            if not avg_len_byte or len(avg_len_byte) < 1:
                break
            avg_len = int.from_bytes(avg_len_byte, 'little')
            yield (start, length, avg_len)


def read_sections_file(sections_path):
    """Read the sections CSV file and yield (func_name, [rows]) for each function section."""
    with open(sections_path, newline='', encoding='ascii') as f:
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
    with open(data_path, 'rb') as f:
        f.seek(offset)
        data = f.read(length)
        # Parse header
        insn_len = data[0]
        block_enc = data[1]
        block_len = int.from_bytes(data[2:4], 'little')
        insn_runlength = np.frombuffer(data[4:4+insn_len], dtype=np.uint8)
        block_dtype = [np.uint8, np.uint16, np.uint32][block_enc]
        block_runlength = np.frombuffer(data[4+insn_len:4+insn_len+block_len], dtype=block_dtype)
        tokens = np.frombuffer(data[4+insn_len+block_len:], dtype=np.uint16)
        return insn_runlength, block_runlength, tokens


def read_function_data_memmap(data_path, offset, length):
    """
    Read the binary data for a function from the data file using numpy.memmap for random access.
    Returns: insn_runlength, block_runlength, tokens
    """
    data = np.memmap(data_path, dtype=np.uint8, mode='r', offset=offset, shape=(length,))
    return parse_function_data_header(data)


def parse_function_data_header(data_bytes):
    """
    Parse the header and return (insn_runlength, block_runlength, tokens) ndarrays.
    data_bytes: bytes or 1D uint8 array
    """
    if isinstance(data_bytes, np.memmap) or isinstance(data_bytes, np.ndarray):
        data_bytes = data_bytes.tobytes()
    insn_len = data_bytes[0]
    block_enc = data_bytes[1]
    block_len = int.from_bytes(data_bytes[2:4], 'little')
    insn_runlength = np.frombuffer(data_bytes[4:4+insn_len], dtype=np.uint8)
    block_dtype = [np.uint8, np.uint16, np.uint32][block_enc]
    block_runlength = np.frombuffer(data_bytes[4+insn_len:4+insn_len+block_len], dtype=block_dtype)
    tokens = np.frombuffer(data_bytes[4+insn_len+block_len:], dtype=np.uint16)
    return insn_runlength, block_runlength, tokens
