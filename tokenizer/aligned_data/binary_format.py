import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BinaryHeader:
    insn_len: int
    block_enc: int
    block_len: int


def parse_binary_header(data_bytes) -> BinaryHeader:
    """Parse the binary header from data bytes."""
    if isinstance(data_bytes, (np.memmap, np.ndarray)):
        data_bytes = data_bytes.tobytes()

    insn_len = int.from_bytes(data_bytes[0:3], "little")
    block_enc = data_bytes[3]
    block_len = int.from_bytes(data_bytes[4:6], "little")

    return BinaryHeader(insn_len=insn_len, block_enc=block_enc, block_len=block_len)


def extract_arrays_from_data(data_bytes, header: BinaryHeader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract insn_runlength, block_runlength, tokens arrays from data given header."""
    if isinstance(data_bytes, (np.memmap, np.ndarray)):
        data_bytes = data_bytes.tobytes()

    insn_runlength = np.frombuffer(data_bytes[6 : 6 + header.insn_len], dtype=np.uint8)

    block_dtype = [np.uint8, np.uint16, np.uint32][header.block_enc]
    block_runlength = np.frombuffer(
        data_bytes[6 + header.insn_len : 6 + header.insn_len + header.block_len],
        dtype=block_dtype,
    )

    tokens = np.frombuffer(data_bytes[6 + header.insn_len + header.block_len :], dtype=np.uint16)

    return insn_runlength, block_runlength, tokens


def encode_binary_header(insn_len: int, block_enc: int, block_len: int) -> bytes:
    """Encode binary header to bytes."""
    header = bytearray()
    header.extend(struct.pack("<I", insn_len)[0:3])
    header.extend(struct.pack("B", block_enc))
    header.extend(struct.pack("<H", block_len))
    return bytes(header)


def determine_block_encoding(block_runlength: np.ndarray) -> int:
    """Determine block encoding type from block_runlength array dtype."""
    if block_runlength.dtype == np.uint8:
        return 0
    elif block_runlength.dtype == np.uint16:
        return 1
    else:
        return 2
