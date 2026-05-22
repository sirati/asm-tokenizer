"""Pad placement + record total size derived from a parsed header.

The on-disk record body is::

    [insn_bytes  (header.insn_len)]
    [pre_pad     (derive_pad_placement[0])]
    [block_bytes (header.block_word_count * sizeof(block_enc))]
    [post_pad    (derive_pad_placement[1])]
    [tokens      (2 * header.token_count, uint16 LE)]

Records align to ``RECORD_ALIGNMENT`` (= 16) bytes. Pad bytes are
``\\x00`` (validator's invariant). Nothing on disk records the pad
split -- the rule below is the sole source of truth.
"""

from __future__ import annotations

from typing import Tuple

from ._header import (
    BLOCK_WORD_SIZE,
    RECORD_ALIGNMENT,
    BinaryHeader,
    prefix_bytes_for_header,
)


def derive_pad_placement(header: BinaryHeader) -> Tuple[int, int]:
    """Return ``(pre_pad, post_pad)`` for ``header``.

    Prefer placing a few pad bytes between ``insn_bytes`` and
    ``block_bytes`` so the block words land on a multiple of their
    element size; if doing so would *exceed* the total record pad,
    fall back to all-pad-pre-block (the block stays unaligned this
    record).

    Let ``U`` = unpadded total (prefix + insn + block + 2*tokens);
    ``P = (-U) % 16`` = total pad needed to reach the 16-byte record
    boundary; ``block_align = sizeof(block_enc) ∈ {1,2,4}``;
    ``B = (-(prefix + insn)) % block_align`` = pad needed to align the
    block start. If ``B <= P`` we use ``B`` pre-block + ``P-B``
    post-block; otherwise we use ``P`` pre-block + 0 post-block.
    """
    prefix = prefix_bytes_for_header(header)
    block_bytes = header.block_word_count * BLOCK_WORD_SIZE[header.block_enc]
    unpadded_total = prefix + header.insn_len + block_bytes + 2 * header.token_count
    total_pad = (-unpadded_total) % RECORD_ALIGNMENT

    block_align = BLOCK_WORD_SIZE[header.block_enc]
    block_pad = (-(prefix + header.insn_len)) % block_align

    if block_pad <= total_pad:
        return block_pad, total_pad - block_pad
    return total_pad, 0


def record_total_size(header: BinaryHeader) -> int:
    """Total on-disk byte count of the record described by ``header``.

    Always a multiple of :data:`RECORD_ALIGNMENT` (= 16).
    """
    prefix = prefix_bytes_for_header(header)
    block_bytes = header.block_word_count * BLOCK_WORD_SIZE[header.block_enc]
    pre_pad, post_pad = derive_pad_placement(header)
    return (
        prefix + header.insn_len + pre_pad + block_bytes + post_pad
        + 2 * header.token_count
    )
