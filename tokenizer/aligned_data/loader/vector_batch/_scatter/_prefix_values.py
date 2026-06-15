"""Per-row variant-prefix TOKEN IDS from ``_variants.bin`` (shifted).

Single concern: read the actual variant-prefix token IDS of each sampled
row's root variant record -- the row-level identity prefix the scatter
prepends ahead of the root body -- and shift them to model-facing ids
(raw ``- 256``), flattened row-major with a CSR jump table. The WIDTH
twin (:func:`.._prefix.variant_prefix_lengths`) reads only the leading
size header; this reads the PAYLOAD ids the token tensor needs.

WHY ``record[1:]`` shifted by ``- 256`` (matches the scalar assembler):
each ``_variants.bin`` record is ``[n_tokens, *ids]`` little-endian u16
(see :func:`tokenizer.variant_tokens.record.read_record`). The decode
path's ``Stage1Variant.variant_tokens`` is the ``ids`` slice
``tokens[1:]`` (leading size header dropped; see
:func:`...loader.variant_resolver.get_variant_by_ref`), and Stage 4 row
assembly emits ``(variant_tokens - 256)`` once per row before any body
(see ``batch_decode._token_assembly``). variant_tokens are statically
encoded raw vocab ids (>= 257) with no inline-digit followers, so the
``- 256`` shift is the canonical model-facing transform -- no strip /
promotion dynamics apply. This module gathers each row's id slice in one
vectorized pass and subtracts the reserved-digit count; ``_data.bin`` is
never touched.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.token_manager import VocabularyManager


__all__ = ["variant_prefix_values"]

#: ``_variants.bin`` size-header width in bytes (one little-endian u16).
_U16_BYTES = 2
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT


def variant_prefix_values(
    variants_u8: np.ndarray,
    cols: ColumnarSections,
    *,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(prefix_tokens, prefix_offsets)`` -- shifted prefix ids per node.

    Parameters
    ----------
    variants_u8:
        ``_variants.bin`` as a 1-D uint8 array (read-only memmap). NOT
        ``_data.bin``.
    cols:
        The columnar catalog; ``var_ref_offset[node]`` is the byte offset
        of the node's variant record.
    nodes:
        ``int[B]`` flat catalog ROOT node indices (one per batch row --
        only the root carries the row-level prefix), parallel to the
        batch rows.

    Returns
    -------
    tuple
        ``(prefix_tokens, prefix_offsets)`` -- ``prefix_tokens`` is
        ``u16[total_prefix]`` the row-major concatenation of each row's
        ``record[1:] - 256`` ids; ``prefix_offsets`` is ``int64[B + 1]``
        CSR (row ``r``'s prefix is ``prefix_tokens[prefix_offsets[r] :
        prefix_offsets[r + 1]]``). Widths match
        :func:`.._prefix.variant_prefix_lengths`.
    """
    node_idx = np.asarray(nodes, dtype=np.int64).reshape(-1)
    n_rows = node_idx.size
    prefix_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    if n_rows == 0:
        return np.zeros(0, dtype=np.uint16), prefix_offsets

    # Absent / empty ``_variants.bin`` -> no prefix records exist; the
    # session yields empty ``variant_tokens`` (see
    # ``BinarySession._open_variants``), so every row's prefix is empty.
    # Mirror the WIDTH twin (:func:`.._prefix.variant_prefix_lengths`).
    if variants_u8.size == 0:
        return np.zeros(0, dtype=np.uint16), prefix_offsets

    ref_offsets = cols.var_ref_offset[node_idx].astype(np.int64)
    if bool((ref_offsets & 1).any()):
        raise ValueError(
            "variant_prefix_values: var_ref_offset must be u16-aligned "
            "(even); got an odd offset -- corrupt catalog vkey"
        )
    words = variants_u8.view(np.uint16)
    ref_word = ref_offsets >> 1
    if int(ref_word.max()) >= words.size:
        raise ValueError(
            "variant_prefix_values: a variant ref offset points past the "
            "end of _variants.bin; the catalog vkey and the variants "
            "buffer are out of sync"
        )
    n_tokens = words[ref_word].astype(np.int64)
    widths = n_tokens - 1  # the prefix WIDTH (size header dropped)
    if bool((widths < 0).any()):
        raise ValueError(
            "variant_prefix_values: a variant record declares n_tokens=0 "
            "(no room for the dropped size token) -- corrupt record"
        )
    np.cumsum(widths, out=prefix_offsets[1:])
    total = int(prefix_offsets[-1])
    if total == 0:
        return np.zeros(0, dtype=np.uint16), prefix_offsets

    # Gather each row's id slice ``record[1 : 1 + width]`` (words after the
    # size header) in one vectorized pass via the cumulative-offset
    # arange. The end of the gathered region must be in-bounds.
    keep = widths > 0
    base_word = (ref_word + 1)[keep]
    w = widths[keep]
    seg_start = np.concatenate(([0], np.cumsum(w)))
    within = np.arange(total, dtype=np.int64) - np.repeat(seg_start[:-1], w)
    word_idx = np.repeat(base_word, w) + within
    if int(word_idx.max()) >= words.size:
        raise ValueError(
            "variant_prefix_values: a prefix id slice runs past the end "
            "of _variants.bin; the record is truncated or the vkey is "
            "wrong"
        )
    raw_ids = words[word_idx].astype(np.int64)
    shifted = (raw_ids - _V2_RESERVED_DIGIT_COUNT).astype(np.uint16)
    return shifted, prefix_offsets
