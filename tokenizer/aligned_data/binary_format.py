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


def _byte_at(data, idx: int) -> int:
    """Return one byte of ``data`` as a plain ``int``.

    Works uniformly for ``bytes``/``bytearray``/``memoryview`` (indexing
    returns ``int``) and ``np.ndarray``/``np.memmap`` of dtype ``uint8``
    (indexing returns a 0-d array convertible to ``int``). Touching one
    byte of a memmap pages in only that byte's page, not a copy of the
    record.
    """
    return int(data[idx])


def _slice_to_int(data, start: int, end: int) -> int:
    """Read bytes ``[start:end]`` as a little-endian unsigned int.

    Copies only the ``end - start`` slice (at most 3 bytes for the
    header sub-fields) to a ``bytes`` so ``int.from_bytes`` can consume
    it uniformly across input types. Never touches the full record.
    """
    return int.from_bytes(bytes(data[start:end]), "little")


def parse_binary_header(data) -> BinaryHeader:
    """Parse the 6-byte record header from ``data``.

    ``data`` may be ``bytes``, ``bytearray``, ``memoryview``,
    ``np.ndarray`` (uint8), or ``np.memmap`` (uint8). Only the first 6
    bytes are touched — the full record is never copied, so passing a
    ``np.memmap`` slice does not allocate a record-sized buffer.
    """
    packed = _byte_at(data, 0)
    if packed & _RESERVED_MASK:
        raise ValueError(
            f"binary header reserved bits set: packed=0x{packed:02x}"
        )
    block_enc = packed & _BLOCK_ENC_MASK
    pad_size = (packed >> _PAD_SIZE_SHIFT) & _PAD_SIZE_MASK
    insn_len = _slice_to_int(data, 1, 4)
    block_len = _slice_to_int(data, 4, 6)

    return BinaryHeader(
        insn_len=insn_len,
        block_enc=block_enc,
        block_len=block_len,
        pad_size=pad_size,
    )


_BLOCK_DTYPES: Tuple[type, type, type] = (np.uint8, np.uint16, np.uint32)


def _as_uint8_view(data) -> np.ndarray:
    """Return a 1-D ``uint8`` view over ``data`` without copying contents.

    For ``np.ndarray``/``np.memmap`` of dtype ``uint8`` the original is
    returned as-is (a slice of a memmap stays a memmap-backed view).
    For ``np.ndarray`` of another dtype the bytes are reinterpreted
    with ``.view(np.uint8)`` (zero copy). For ``bytes``/``bytearray``
    /``memoryview`` ``np.frombuffer`` creates a read-only view that
    shares memory with the input buffer — no record-sized allocation.
    """
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            return data.view(np.uint8)
        return data
    return np.frombuffer(data, dtype=np.uint8)


def extract_arrays_from_data(
    data,
    header: BinaryHeader,
    *,
    is_overlong: bool,
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
    invariant). ``is_overlong`` is REQUIRED (keyword-only): callers must
    derive it from the index-entry sentinel (unmatched arm) or the inline
    indexer decode (matched arm) -- a silent default would corrupt
    overlong reads by skipping the 3-byte overlong-length field shift,
    so the API forces an explicit value. The field itself is resolved
    independently by the caller (the session layer that decoded the
    index sentinel), so this function never reads it.

    Zero-copy on memmap input: ``data`` is wrapped in a ``uint8`` view
    (``np.memmap`` slices stay memmap-backed; ``bytes`` are exposed via
    ``np.frombuffer`` without copying contents), then the per-array
    slices are produced by ``arr[i:j].view(target_dtype)``. The returned
    arrays may therefore be views into the original memmap -- the
    enclosing ``BinarySession`` copies them on egress so callers receive
    independent buffers (see ``BinarySession`` class docstring).
    """
    raw = _as_uint8_view(data)

    prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES if is_overlong else 0)
    insn_end = prefix + header.insn_len
    block_start = insn_end + header.pad_size
    block_end = block_start + header.block_len

    insn_runlength = raw[prefix:insn_end]

    block_dtype = _BLOCK_DTYPES[header.block_enc]
    block_slice = raw[block_start:block_end]
    block_runlength = (
        block_slice if block_dtype is np.uint8 else block_slice.view(block_dtype)
    )

    tokens = raw[block_end:].view(np.uint16)

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
