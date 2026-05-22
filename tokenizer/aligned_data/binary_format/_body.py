"""Zero-copy record-body slicing into (insn, block, tokens) ndarrays.

The body layout (after the variable-width header) is::

    [insn_bytes  (header.insn_len)]
    [pre_pad     (derive_pad_placement[0])]
    [block_bytes (header.block_word_count * sizeof(block_enc))]
    [post_pad    (derive_pad_placement[1])]
    [tokens      (2 * header.token_count, uint16 LE)]

``extract_arrays_from_data`` slices via numpy views so passing a
``np.memmap`` does not allocate a record-sized buffer -- the returned
arrays may share memory with the input. The enclosing ``BinarySession``
copies them on egress so external callers receive independent buffers.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from ._header import BLOCK_WORD_SIZE, BinaryHeader
from ._pad import derive_pad_placement

_BLOCK_DTYPES: Tuple[type, type, type] = (np.uint8, np.uint16, np.uint32)


def _as_uint8_view(data) -> np.ndarray:
    """Return a 1-D ``uint8`` view over ``data`` without copying contents.

    ``np.ndarray``/``np.memmap`` of dtype ``uint8`` is returned as-is
    (slices of a memmap stay memmap-backed). Other-dtype ndarrays are
    reinterpreted via ``.view(np.uint8)`` (zero copy). ``bytes`` /
    ``bytearray`` / ``memoryview`` are wrapped with ``np.frombuffer``
    so they share memory with the input buffer.
    """
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            return data.view(np.uint8)
        return data
    return np.frombuffer(data, dtype=np.uint8)


def extract_arrays_from_data(
    data: Union[bytes, bytearray, memoryview, np.ndarray],
    header: BinaryHeader,
    prefix_bytes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice ``data`` into ``(insn_runlength, block_runlength, tokens)``.

    ``prefix_bytes`` is the byte count the header occupied on disk
    (the second element of ``parse_binary_header``'s return tuple).
    Pad placement is re-derived from ``header`` via
    :func:`derive_pad_placement`; the reader never inspects the pad
    bytes themselves (the validator owns that invariant). Returned
    arrays are views into ``data`` so passing a ``np.memmap`` does
    not allocate a record-sized buffer.
    """
    raw = _as_uint8_view(data)

    pre_pad, post_pad = derive_pad_placement(header)
    block_word_size = BLOCK_WORD_SIZE[header.block_enc]
    block_bytes = header.block_word_count * block_word_size

    insn_end = prefix_bytes + header.insn_len
    block_start = insn_end + pre_pad
    block_end = block_start + block_bytes
    tokens_start = block_end + post_pad
    tokens_end = tokens_start + 2 * header.token_count

    insn_runlength = raw[prefix_bytes:insn_end]

    block_slice = raw[block_start:block_end]
    block_dtype = _BLOCK_DTYPES[header.block_enc]
    block_runlength = (
        block_slice if block_dtype is np.uint8 else block_slice.view(block_dtype)
    )

    tokens = raw[tokens_start:tokens_end].view(np.uint16)

    return insn_runlength, block_runlength, tokens
