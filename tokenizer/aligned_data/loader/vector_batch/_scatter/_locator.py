"""Map emission catalog NODES to their ``_data.bin`` token-region spans.

Single concern: turn the geometry prepass's flat ``node[]`` catalog
indices (``var_offsets``-major) into the ``(token_start, token_count)``
byte/word spans of each node's record in ``_data.bin`` -- the input the
batched body load gathers. NO body byte is decoded here; this only
resolves WHERE each node's raw token stream lives.

Locator chain (the SAME one the realized-geometry index build uses, see
:func:`...realized_lengths._compute.realized_geometry_for_offsets`):

* ``record_offset = var_data_offset_shifted[node] << RECORD_OFFSET_SHIFT``
  -- the catalog stores the 16-byte-aligned record offset right-shifted
  by ``RECORD_OFFSET_SHIFT`` (= log2(:data:`RECORD_ALIGNMENT`)); the
  scatter reconstructs the absolute ``_data.bin`` byte offset.
* :func:`...binary_format._bulk_geometry.bulk_token_spans` decodes every
  record header in ONE vectorized pass -> ``(token_start, token_count)``.

Reuse, not re-implementation: the shift constant is derived from the
shared :data:`RECORD_ALIGNMENT`; the span decode is the shared bulk
engine. A header-layout change lands in ``binary_format`` and surfaces
here automatically.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)
from tokenizer.aligned_data.binary_format._header import RECORD_ALIGNMENT
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections


__all__ = ["RECORD_OFFSET_SHIFT", "node_token_spans"]


#: log2(:data:`RECORD_ALIGNMENT`) -- the right-shift the catalog applied
#: to record offsets (``var_data_offset_shifted = record_offset >>
#: RECORD_OFFSET_SHIFT``). Derived from the shared alignment constant so a
#: change to record alignment cannot drift this scatter out of sync.
RECORD_OFFSET_SHIFT: int = int(RECORD_ALIGNMENT).bit_length() - 1
assert (1 << RECORD_OFFSET_SHIFT) == int(RECORD_ALIGNMENT), (
    "RECORD_ALIGNMENT must be a power of two for the catalog's "
    "record_offset >> shift encoding to round-trip"
)


def node_token_spans(
    cols: ColumnarSections,
    data_u8: np.ndarray,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(token_start, token_count)`` for each catalog ``node``.

    Parameters
    ----------
    cols:
        The columnar ``sections.bin`` catalog; ``var_data_offset_shifted
        [node]`` is the node's ``record_offset >> RECORD_OFFSET_SHIFT``.
    data_u8:
        The arm's ``_data.bin`` as a 1-D ``uint8`` array (read-only
        memmap). Only record HEADER bytes are paged in here.
    nodes:
        ``int[k]`` flat catalog node indices (``var_offsets``-major).

    Returns
    -------
    tuple
        ``(token_start, token_count)`` -- ``int64[k]`` byte offset of the
        record's u16 token region + the u16 token count, parallel to
        ``nodes``. The byte offset is even (u16-aligned record tail).
    """
    node_idx = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if node_idx.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    record_offsets = (
        cols.var_data_offset_shifted[node_idx].astype(np.int64)
        << RECORD_OFFSET_SHIFT
    )
    return bulk_token_spans(data_u8, record_offsets)
