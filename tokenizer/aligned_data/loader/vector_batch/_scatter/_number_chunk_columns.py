"""Columnar builder for the number-sidecar global chunk stream.

Single concern (the design-first sentence): *given the vector path's
already-built :class:`DenseColumns` (the per-node surviving expanded
number-band slots), the kernel-built per-:class:`TokenType` per-call_target
chunk-slice ``.start`` boundary arrays, and the emission row CSR (the
variant == row boundaries), produce the per-chunk ``(out_block,
slice_start, ct_ordinal, variant_chunk_offsets)`` the number-sidecar
concat consumes, byte-identical to* :func:`...batch_decode._sidecar_concat.
_build_global_chunk_stream`'s ``sections -> variants -> call_targets``
object-tree walk.

Why this is well-posed (NOT a re-walk): the vector dense adapter lays one
synthetic section per non-padding batch row in row order, one variant per
section, and its call_targets ARE the row's emitted nodes in emission
order (:mod:`._dense_adapter`). So the tree walk's ``ct_ordinal`` IS the
emitted-node index ``e = 0 .. n_nodes - 1``, and its variant boundaries
ARE the emission row CSR (:attr:`BatchRowEmission.row_offsets`). Every
per-chunk field the tree walk reads is a column already in hand here:

* ``out_block`` (``shifted_id - 1``) + the chunk's owning ``ct_ordinal``
  come from the SURVIVING NUMBER-band slots of :class:`DenseColumns`
  (``expanded[node_offsets[e] : node_offsets[e] + surviving[e]]`` ==
  the tree's ``expanded_token_ids[:partial_cut_length]``), in DFS-then-
  stream order (ascending expanded position).
* ``slice_start`` (the per-(call_target, :class:`TokenType`) source base
  into ``numbers_per_TokenType[T]``) comes from the kernel-built
  ``number_chunk_slice_starts_per_type[T][e]`` -- the SAME ``.start``
  the tree reads off ``call_target.number_chunk_slices[T]``. Consuming
  the kernel's boundary arrays (NOT re-deriving them from the expanded
  band-slot count) is REQUIRED: a mid-cut F128 finite source emits two
  ``idx_2d`` rows but only its LSB slot survives the cut, so the
  expanded-slot count under-counts the source's idx_2d rows and the
  next call_target's slice base would shift -- mis-routing significands.

This module re-implements NO rank / gather logic (the per-(ct, block)
rank + the per-type gather into ``numbers_per_TokenType`` stay in the
shared :func:`_build_global_chunk_stream` tail) and NO chunk-slice
arithmetic (it consumes the kernel's boundary arrays) -- it only
RE-EXPRESSES the same per-chunk ``(out_block, slice_start, ct_ordinal)``
triple + the variant CSR in the same DFS-then-stream order, so the
byte-identity gate cannot diverge.

Module boundary: owned at the scatter/expand boundary; the only thing
crossing into the number-sidecar concat is a :class:`NumberChunkColumns`
-- an optional, additive input mirroring the ``dense`` / ``flat`` pattern
the emission kernels + remap already use (the staged ``batch_decode``
path supplies nothing and keeps its tree walk byte-unaffected).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._sidecar_concat import (
    NumberChunkColumns,
    _NUMBER_BLOCK_TOKEN_TYPES,
    _SHIFTED_NUMBER_BAND_HI,
    _SHIFTED_NUMBER_BAND_LO,
)
from tokenizer.aligned_data.loader.batch_decode._dense_columns import (
    DenseColumns,
)

from .._types import BatchGeometry


__all__ = ["build_number_chunk_columns"]


def build_number_chunk_columns(
    geometry: BatchGeometry,
    dense: DenseColumns,
    number_chunk_slice_starts_per_type: dict,
) -> NumberChunkColumns:
    """Build the number-sidecar per-chunk columns from the dense columns.

    Parameters
    ----------
    geometry:
        The body-free prepass result; its ``emission.row_offsets`` is the
        per-row node CSR. The vector path lays one variant per row, so the
        tree walk's per-variant chunk-count CSR IS this row CSR.
    dense:
        The vector path's :class:`DenseColumns`; the source of the
        surviving NUMBER-band expanded slots (``expanded`` clipped to the
        per-node ``surviving_token_count`` prefix) -- the same band-filtered
        ``expanded_token_ids[:partial_cut_length]`` the tree walk reads.
    number_chunk_slice_starts_per_type:
        ``dict[TokenType, int64[n_nodes]]`` -- the kernel-built per-call_target
        chunk-slice ``.start`` into ``numbers_per_TokenType[T]`` (==
        ``call_target.number_chunk_slices[T].start``), one entry per DFS
        call_target.

    Returns
    -------
    NumberChunkColumns
        The per-chunk ``(out_block, slice_start, ct_ordinal)`` triple +
        the per-variant (== per-row) chunk CSR, in DFS-then-stream order,
        byte-identical to the tree walk's loop output.
    """
    row_offsets = np.asarray(geometry.emission.row_offsets, dtype=np.int64)
    n_rows = max(int(row_offsets.shape[0]) - 1, 0)
    n_nodes = int(dense.n_nodes)

    expanded = np.asarray(dense.expanded).reshape(-1)
    node_offsets = np.asarray(dense.node_offsets, dtype=np.int64)
    surviving = np.asarray(dense.surviving_token_count, dtype=np.int64)

    # ----- per-chunk selection: surviving NUMBER-band expanded slots -----
    # A position is a chunk iff it is in the node's SURVIVING prefix
    # (offset_in_node < surviving[node]) AND its expanded id is in the
    # post-shift NUMBER band [LO, HI). Ascending position order groups the
    # chunks by node (DFS) and, within a node, by stream order -- exactly
    # the tree walk's ``expanded_token_ids[:partial_cut_length]`` scan.
    if n_nodes <= 0 or expanded.shape[0] == 0:
        out_block = np.empty(0, dtype=np.int64)
        slice_start = np.empty(0, dtype=np.int64)
        ct_ordinal = np.empty(0, dtype=np.int64)
    else:
        node_len = np.diff(node_offsets)
        node_id = np.repeat(np.arange(n_nodes, dtype=np.int64), node_len)
        offset_in_node = (
            np.arange(expanded.shape[0], dtype=np.int64)
            - node_offsets[node_id]
        )
        within = offset_in_node < surviving[node_id]
        in_band = (expanded >= _SHIFTED_NUMBER_BAND_LO) & (
            expanded < _SHIFTED_NUMBER_BAND_HI
        )
        sel = within & in_band

        out_block = expanded[sel].astype(np.int64) - _SHIFTED_NUMBER_BAND_LO
        ct_ordinal = node_id[sel]

        # Per-chunk source base = the owning call_target's per-block slice
        # ``.start`` (the kernel's boundary array), selected by the chunk's
        # block. Stacked ``[n_blocks, n_nodes]`` so the per-chunk gather is
        # one fancy-index -- the columnar form of the tree's per-call_target
        # ``ct_slice_start[block]`` read.
        starts = _stack_slice_starts(
            number_chunk_slice_starts_per_type, n_nodes
        )
        slice_start = starts[out_block, ct_ordinal]

    variant_chunk_offsets = _variant_chunk_offsets(
        row_offsets, ct_ordinal, n_rows
    )

    return NumberChunkColumns(
        out_block=out_block,
        slice_start=slice_start,
        ct_ordinal=ct_ordinal,
        variant_chunk_offsets=variant_chunk_offsets,
    )


def _stack_slice_starts(
    number_chunk_slice_starts_per_type: dict, n_nodes: int
) -> np.ndarray:
    """Stack the per-:class:`TokenType` ``.start`` columns ``[n_blocks, n_nodes]``.

    Block index ``b`` (0 = VC2, ..., 6 = F128) maps to the canonical
    NUMBER-block :class:`TokenType`; the row is that type's per-call_target
    slice ``.start`` array. Mirrors the tree's per-call_target
    ``ct_slice_start[b] = number_chunk_slices[T].start``.
    """
    return np.stack(
        [
            np.asarray(
                number_chunk_slice_starts_per_type[token_type],
                dtype=np.int64,
            )
            for token_type in _NUMBER_BLOCK_TOKEN_TYPES
        ],
        axis=0,
    ).reshape(len(_NUMBER_BLOCK_TOKEN_TYPES), n_nodes)


def _variant_chunk_offsets(
    row_offsets: np.ndarray, ct_ordinal: np.ndarray, n_rows: int
) -> np.ndarray:
    """Per-variant (== per-row) chunk CSR over the global stream.

    Each chunk's owning node ``ct_ordinal`` falls in exactly one row's
    node run ``[row_offsets[r] : row_offsets[r + 1]]`` (the emission row
    CSR). The tree walk appends one ``variant_chunk_counts`` entry per
    variant in DFS order, which on the vector path is one entry per row in
    row order -- so the per-row chunk count is the histogram of each
    chunk's row, and the CSR is its cumsum.
    """
    variant_chunk_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    if n_rows == 0 or ct_ordinal.shape[0] == 0:
        return variant_chunk_offsets
    # Row of each chunk: searchsorted of its node into the row CSR (the
    # node run boundaries). ``side="right" - 1`` buckets node e into the
    # row whose run contains it.
    chunk_row = (
        np.searchsorted(row_offsets, ct_ordinal, side="right") - 1
    )
    per_row = np.bincount(chunk_row, minlength=n_rows).astype(np.int64)
    np.cumsum(per_row, out=variant_chunk_offsets[1:])
    return variant_chunk_offsets
