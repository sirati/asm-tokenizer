"""Per-record header for ``_data.bin``.

Layout (6 bytes, little-endian)::

    byte 0     u8   packed_byte    bits 0-1: block_enc (0|1|2)
                                   bits 2-3: pad_size  (0..3)
                                   bits 4-7: reserved  (write 0)
    byte 1-3   u24  insn_len       byte count of ``insn_bytes``
    byte 4-5   u16  block_len      byte count of ``block_bytes``

The packed control byte leads so the record header is symmetric with the
``_index.bin`` entry's leading control field. ``pad_size`` records the
number of ``\\x00`` bytes inserted between ``insn_bytes`` and
``block_bytes`` so the record total (header + insn + pad + block +
tokens, plus an optional overlong-length field) is a multiple of 4. The
reader reads the pad size from this header — it never recomputes it.
"""

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Fixed sizes used by the pad-size computation. Kept here so writer and
# reader share a single source of truth.
HEADER_BYTES = 6
OVERLONG_FIELD_BYTES = 3

# Cap on each header field, derived from its on-wire width.
_INSN_LEN_CAP = 1 << 24   # u24
_BLOCK_LEN_CAP = 1 << 16  # u16

# Packed control byte bit layout.
_BLOCK_ENC_MASK = 0b00000011
_PAD_SIZE_SHIFT = 2
_PAD_SIZE_MASK = 0b00000011
_RESERVED_MASK = 0b11110000  # bits 4-7 must be 0


class IndexEntrySkip(Exception):
    """Raised by encoders when a per-section field overflows its cap.

    Callers catch and translate into an ``error.log`` entry plus a
    skipped index entry; the build continues. ``reason`` names the
    overflowing field; ``value`` is the offending integer so the log
    line can record what triggered the skip.
    """

    def __init__(self, reason: str, value: int) -> None:
        super().__init__(f"{reason} (value={value})")
        self.reason = reason
        self.value = value


@dataclass
class BinaryHeader:
    insn_len: int
    block_enc: int
    block_len: int
    pad_size: int


def parse_binary_header(data_bytes) -> BinaryHeader:
    """Parse the 6-byte record header from ``data_bytes``."""
    if isinstance(data_bytes, (np.memmap, np.ndarray)):
        data_bytes = data_bytes.tobytes()

    packed = data_bytes[0]
    if packed & _RESERVED_MASK:
        raise ValueError(
            f"binary header reserved bits set: packed=0x{packed:02x}"
        )
    block_enc = packed & _BLOCK_ENC_MASK
    pad_size = (packed >> _PAD_SIZE_SHIFT) & _PAD_SIZE_MASK
    insn_len = int.from_bytes(data_bytes[1:4], "little")
    block_len = int.from_bytes(data_bytes[4:6], "little")

    return BinaryHeader(
        insn_len=insn_len,
        block_enc=block_enc,
        block_len=block_len,
        pad_size=pad_size,
    )


def extract_arrays_from_data(
    data_bytes,
    header: BinaryHeader,
    is_overlong: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract insn_runlength, block_runlength, tokens arrays from data given header.

    Body layout, starting at ``prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES
    if is_overlong else 0)``::

        [insn_bytes (header.insn_len)]
        [pad        (header.pad_size, all \\x00)]
        [block_bytes(header.block_len)]
        [tokens     (rest, uint16 LE)]

    The pad is skipped purely by arithmetic — the reader never recomputes
    its size and never inspects its bytes (the validator owns that
    invariant). ``is_overlong`` shifts the body start past the 3-byte
    overlong-length field; that field's value is resolved independently
    by the caller (the session layer that decoded the index sentinel),
    so this function never reads it.
    """
    if isinstance(data_bytes, (np.memmap, np.ndarray)):
        data_bytes = data_bytes.tobytes()

    prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES if is_overlong else 0)
    insn_end = prefix + header.insn_len
    block_start = insn_end + header.pad_size
    block_end = block_start + header.block_len

    insn_runlength = np.frombuffer(data_bytes[prefix:insn_end], dtype=np.uint8)

    block_dtype = [np.uint8, np.uint16, np.uint32][header.block_enc]
    block_runlength = np.frombuffer(
        data_bytes[block_start:block_end],
        dtype=block_dtype,
    )

    tokens = np.frombuffer(data_bytes[block_end:], dtype=np.uint16)

    return insn_runlength, block_runlength, tokens


def encode_binary_header(
    insn_len: int,
    block_enc: int,
    block_len: int,
    pad_size: int,
) -> bytes:
    """Encode the 6-byte record header.

    Raises ``ValueError`` for out-of-domain ``block_enc`` or
    ``pad_size`` (programmer errors — the caller is supposed to pick
    these from a constrained set). Raises ``IndexEntrySkip`` when
    ``insn_len`` or ``block_len`` exceeds the on-wire width — those
    caps are corpus-data conditions that the builder catches and
    translates into a logged skip.
    """
    if block_enc not in (0, 1, 2):
        raise ValueError(f"block_enc must be 0, 1, or 2; got {block_enc}")
    if pad_size not in (0, 1, 2, 3):
        raise ValueError(f"pad_size must be in [0,3]; got {pad_size}")
    if insn_len >= _INSN_LEN_CAP:
        raise IndexEntrySkip("insn_len_overflow", insn_len)
    if block_len >= _BLOCK_LEN_CAP:
        raise IndexEntrySkip("block_len_overflow", block_len)

    packed = (block_enc & _BLOCK_ENC_MASK) | (
        (pad_size & _PAD_SIZE_MASK) << _PAD_SIZE_SHIFT
    )
    header = bytearray()
    header.append(packed)
    header.extend(struct.pack("<I", insn_len)[0:3])
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


def compute_pad(
    insn_len: int,
    block_len: int,
    token_count: int,
    is_overlong: bool,
) -> int:
    """Smallest pad in [0,3] so the record total is a multiple of 4."""
    body_prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES if is_overlong else 0)
    unpadded_total = body_prefix + insn_len + block_len + 2 * token_count
    return (-unpadded_total) % 4
