"""Body-free per-variant variant-PREFIX length from ``_variants.bin``.

Single concern: read the ``n_axis`` variant-prefix token count of each
sampled ``(section, variant)`` row -- the row-level identity prefix that
is prepended ahead of the root body -- straight from ``_variants.bin``,
with NO ``_data.bin`` touch and NO body decode.

WHY this is body-free + a single u16 read: each variant's record in
``_variants.bin`` is ``[n_tokens, *ids]`` as little-endian u16 (see
:func:`tokenizer.variant_tokens.record.read_record`). The leading u16
``n_tokens`` is the COUNT OF PAYLOAD IDS that follow it (the record array
has length ``1 + n_tokens``; see :func:`...variant_tokens.encoder`). The
decode path's ``variant_tokens`` is the ``ids`` slice ``tokens[1:]`` (the
leading size header dropped; see
:func:`...loader.variant_resolver.get_variant_by_ref`), so it has exactly
``n_tokens`` elements -- the prefix WIDTH is ``n_tokens``. The byte offset
of a variant's record is the catalog's ``var_ref_offset`` (the vkey). We
gather the leading u16 at every sampled variant's ref offset in ONE
vectorized pass -- never reading the payload, never touching ``_data.bin``.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections


__all__ = ["variant_prefix_lengths"]

#: ``_variants.bin`` size-header width in bytes (one little-endian u16).
_U16_BYTES = 2


def variant_prefix_lengths(
    variants_u8: np.ndarray,
    cols: ColumnarSections,
    *,
    nodes: np.ndarray,
) -> np.ndarray:
    """``int64[k]`` variant-prefix (``n_axis``) length per node.

    Parameters
    ----------
    variants_u8:
        The ``_variants.bin`` file as a 1-D uint8 array (typically a
        read-only memmap). Body-free -- this is NOT ``_data.bin``.
    cols:
        The columnar catalog; ``var_ref_offset[node]`` is the byte offset
        of the node's variant record in ``variants_u8``.
    nodes:
        ``int[k]`` flat catalog NODE indices (``var_offsets``-major) to
        read the prefix length for (one per batch ROW -- only the root
        node carries the row-level prefix).

    Returns
    -------
    np.ndarray
        ``int64[k]`` prefix lengths = ``n_tokens`` per node, the column
        count the variant-token prefix occupies ahead of the root body
        (= ``len(get_variant_by_ref(...).variant_tokens)``). Units: token
        columns.
    """
    node_idx = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if node_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    # An ABSENT / empty ``_variants.bin`` means no variant-prefix records
    # exist; the session's variant resolver returns empty ``variant_tokens``
    # in that case (``BinarySession._open_variants`` yields ``None`` for a
    # missing file), so the prefix WIDTH is uniformly 0 -- regardless of the
    # catalog's ``var_ref_offset`` (which references records that were never
    # materialised). Mirror that here rather than reading past the buffer.
    if variants_u8.size == 0:
        return np.zeros(node_idx.size, dtype=np.int64)
    ref_offsets = cols.var_ref_offset[node_idx].astype(np.int64)
    if bool((ref_offsets & 1).any()):
        raise ValueError(
            "variant_prefix_lengths: var_ref_offset must be u16-aligned "
            "(even); got an odd offset -- corrupt catalog vkey"
        )
    max_word = (ref_offsets >> 1).max() if node_idx.size else 0
    words = variants_u8.view(np.uint16)
    if int(max_word) >= words.size:
        raise ValueError(
            "variant_prefix_lengths: a variant ref offset points past the "
            "end of _variants.bin; the catalog vkey and the variants "
            "buffer are out of sync"
        )
    n_tokens = words[ref_offsets >> 1].astype(np.int64)
    return n_tokens
