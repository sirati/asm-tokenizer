"""Vectorized record geometry for arrays of ``_data.bin`` offsets.

Single concern: given the ``_data.bin`` file as a ``uint8`` array and a
batch of record byte offsets, decode every record's header fields with
numpy (no per-record Python) and return the geometry consumers need to
slice token regions in bulk.

This module mirrors :func:`._header.parse_binary_header` +
:func:`._pad.record_total_size` field-for-field; it owns NO layout
decisions of its own. Any change to the on-wire header layout must
land in :mod:`._header` / :mod:`._pad` first and be replicated here --
the cross-equivalence test
(``tests/test_bulk_geometry.py``) pins the two paths together.

Layout recap (see :mod:`._header` for the authoritative docs):

* byte 0 low 2 bits = format tag. ``0`` -> ultrashort, ``1..3`` ->
  normal with ``block_enc = tag - 1``.
* ultrashort: ``insn_len = (b0 >> 2) & 0x3f``, ``block_word_count =
  b1``, ``token_count = b2``, prefix = 7 bytes.
* normal: ``width_tag = (b0 >> 2) & 3``; ``token_count = ((b0 >> 4)
  << (8 * (width_tag + 1))) | LE(bytes 1 .. 1 + width_tag + 1)``;
  ``insn_len`` = u24 LE at cursor; ``block_word_count`` = u16 LE next;
  prefix = ``NORMAL_PREFIX_BYTES[width_tag]``.
* tokens sit at the record TAIL: ``token_start = offset + total -
  2 * token_count`` with ``total = U + ((-U) % RECORD_ALIGNMENT)`` and
  ``U = prefix + insn_len + block_word_count *
  BLOCK_WORD_SIZE[block_enc] + 2 * token_count``.

The widest field byte this decoder touches is index 9 (normal form,
``width_tag == 3``: 1 packed + 4 token-low + 3 insn + 2 block); every
record is at least ``RECORD_ALIGNMENT`` (16) bytes so the 10-byte
gather never reads past the record, hence never past the file.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._header import (
    BLOCK_WORD_SIZE,
    NORMAL_PREFIX_BYTES,
    RECORD_ALIGNMENT,
    ULTRASHORT_PREFIX_BYTES,
)


__all__ = ["bulk_token_spans"]


#: Bytes gathered per record -- enough for every header field this
#: decoder reads (see module docstring for the widest-field argument).
_GATHER_BYTES = 10

#: ``block_enc -> word size`` as an indexable array. Slot 3 is a
#: placeholder hit only by ultrashort rows' masked-out normal lane
#: (``fmt - 1 == -1`` wraps to the last slot); its value never reaches
#: the output because the ultrashort lane wins the ``np.where``.
_BLOCK_WORD_SIZE_LANES = np.array(
    [*BLOCK_WORD_SIZE, BLOCK_WORD_SIZE[-1]], dtype=np.int64
)

#: ``width_tag -> normal-form prefix bytes`` as an indexable array.
_NORMAL_PREFIX_LANES = np.array(NORMAL_PREFIX_BYTES, dtype=np.int64)


def bulk_token_spans(
    data_u8: np.ndarray, offsets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode token-region spans for the records at ``offsets``.

    Parameters
    ----------
    data_u8:
        The full ``_data.bin`` as a 1-D ``uint8`` array (typically a
        read-only memmap). Only header bytes are touched -- at most
        10 bytes per record are paged in.
    offsets:
        Integer array of record byte offsets (any integer dtype; any
        shape is flattened). Offsets must point at record starts --
        garbage offsets produce garbage spans, exactly like the scalar
        parser.

    Returns
    -------
    (token_start, token_count):
        Two ``int64`` arrays parallel to ``offsets``.
        ``data_u8[token_start[i] : token_start[i] + 2 * token_count[i]]``
        is record ``i``'s raw uint16-LE token stream.
    """
    offs = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if offs.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()

    # One fancy-indexed gather of the first _GATHER_BYTES header bytes
    # per record; int64 lanes so the shift/or arithmetic below never
    # overflows or upcasts mid-expression.
    b = data_u8[offs[:, None] + np.arange(_GATHER_BYTES, dtype=np.int64)]
    b = b.astype(np.int64)
    b0 = b[:, 0]
    fmt = b0 & 0b11
    is_ultra = fmt == 0

    # --- ultrashort lane ------------------------------------------------
    ultra_insn = (b0 >> 2) & 0b111111
    ultra_block_bytes = b[:, 1]  # block_enc fixed 0 -> 1 byte/word
    ultra_tokens = b[:, 2]

    # --- normal lane ----------------------------------------------------
    block_enc = fmt - 1  # -1 on ultrashort rows; masked out below
    width_tag = (b0 >> 2) & 0b11
    low_byte_count = width_tag + 1
    token_hi4 = (b0 >> 4) & 0b1111
    # All four candidate low bytes as one LE u32, masked down to the
    # row's actual low-byte width.
    le4 = b[:, 1] | (b[:, 2] << 8) | (b[:, 3] << 16) | (b[:, 4] << 24)
    token_low = le4 & ((np.int64(1) << (8 * low_byte_count)) - 1)
    normal_tokens = (token_hi4 << (8 * low_byte_count)) | token_low

    rows = np.arange(offs.size)
    cursor = 1 + low_byte_count
    normal_insn = (
        b[rows, cursor]
        | (b[rows, cursor + 1] << 8)
        | (b[rows, cursor + 2] << 16)
    )
    normal_block_words = b[rows, cursor + 3] | (b[rows, cursor + 4] << 8)
    normal_block_bytes = (
        normal_block_words * _BLOCK_WORD_SIZE_LANES[block_enc]
    )
    normal_prefix = _NORMAL_PREFIX_LANES[width_tag]

    # --- merge lanes + tail-position arithmetic --------------------------
    token_count = np.where(is_ultra, ultra_tokens, normal_tokens)
    prefix = np.where(
        is_ultra, np.int64(ULTRASHORT_PREFIX_BYTES), normal_prefix
    )
    insn_len = np.where(is_ultra, ultra_insn, normal_insn)
    block_bytes = np.where(is_ultra, ultra_block_bytes, normal_block_bytes)

    unpadded = prefix + insn_len + block_bytes + 2 * token_count
    total = unpadded + ((-unpadded) % RECORD_ALIGNMENT)
    token_start = offs + total - 2 * token_count
    return token_start, token_count
