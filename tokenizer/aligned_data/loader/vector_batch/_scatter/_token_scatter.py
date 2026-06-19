"""Fully-batched scatter of expanded node bodies into ``tokens[B, L]``.

Single concern: place every emitted node's expanded token stream
(:mod:`._expand`) into its precomputed ``[B, L]`` column window (the
geometry prepass's BFS-order own-length prefix-sum), prepend the per-row
variant-token prefix, and apply the single straddler cut -- with ONE
vectorized scatter, no per-row / per-node Python copy loop.

Column math (all from the body-free geometry, no re-derivation):

* Row ``r``'s body region starts at column ``prefix_len[r]``; columns
  ``[0, prefix_len[r])`` hold the variant prefix.
* Within the body, node at row-local position ``k`` starts at the
  own-length prefix-sum of the row's prior nodes -- exactly the
  ``running_end - own_length`` the layout pass computed. So node ``e``
  (flat emission index) lands at absolute column
  ``prefix_len[row(e)] + body_start(e)``.
* The straddler node keeps only its first ``partial_cut_length[r]``
  columns; nodes AFTER the straddler in the row are dropped entirely
  (their columns fall past ``L``).

The scatter builds, for every SURVIVING expanded slot, a flat
``(dest_row, dest_col, value)`` triple and does a single
``tokens[rows, cols] = values`` assignment. Surviving slots are the
emitted bodies up to and including the straddler's cut prefix, plus the
per-row variant prefix. ``id == 0`` (null-content) is left wherever no
slot writes, matching the zero-allocation.
"""

from __future__ import annotations

import numpy as np
from dedup_hashmap import build_token_scatter_kernel

from ._expand import ExpandedBatch
from ._surviving import row_of_node as _row_of_node
from ._surviving import surviving_token_counts
from .._types import BatchGeometry


__all__ = ["scatter_tokens"]


def scatter_tokens(
    geometry: BatchGeometry,
    expanded: ExpandedBatch,
    variant_prefix_tokens: np.ndarray,
    variant_prefix_offsets: np.ndarray,
) -> np.ndarray:
    """Assemble the ``u16[B, L]`` token tensor, fully vectorized.

    Parameters
    ----------
    geometry:
        The body-free prepass result. Provides the emission CSR + own
        lengths (column windows), the per-row token layout (prefix width
        + straddler cut), and the row count + seq_len.
    expanded:
        The flat per-node expanded token streams (:class:`._expand.
        ExpandedBatch`); ``expanded.node_offsets`` is parallel to
        ``geometry.emission.node`` and each node's length equals its
        ``own_length``.
    variant_prefix_tokens / variant_prefix_offsets:
        The per-row variant-prefix token ids, already SHIFTED to model-
        facing ids (raw ``- 256``), flattened row-major with the CSR
        ``variant_prefix_offsets`` (``int[B + 1]``). Row ``r``'s prefix
        is ``variant_prefix_tokens[variant_prefix_offsets[r] :
        variant_prefix_offsets[r + 1]]``; its width matches
        ``geometry.layout.prefix_len[r]``.

    Returns
    -------
    np.ndarray
        ``u16[B, L]`` -- the model-facing token tensor; ``id == 0`` is
        the null-content pad at every unwritten position.
    """
    n_rows = int(geometry.n_rows)
    seq_len = int(geometry.layout.seq_len)
    if n_rows == 0 or seq_len == 0:
        return np.zeros((n_rows, seq_len), dtype=np.uint16)

    emission = geometry.emission
    layout = geometry.layout
    own = np.asarray(emission.own_length, dtype=np.int64)
    node_off = np.asarray(expanded.node_offsets, dtype=np.int64)
    n_emitted = own.size
    if node_off.size - 1 != n_emitted:
        raise ValueError(
            f"expanded covers {node_off.size - 1} nodes but the emission "
            f"has {n_emitted}"
        )

    # The per-node surviving column count (the straddler cut) and per-node
    # owning row stay python-side -- separate concerns shared with the
    # dense-sidecar pass (:mod:`._surviving`) so the cut can never drift.
    surviving = surviving_token_counts(geometry)
    row_of_node = _row_of_node(geometry)

    # The kernel OWNS the body-start cumsum + index arithmetic + the ordered
    # (prefix-then-body, last-writer-wins) scatter into the dense u16[B, L]
    # buffer; we only extract the geometry arrays + coerce dtypes here.
    flat = build_token_scatter_kernel(
        n_rows,
        seq_len,
        np.ascontiguousarray(emission.row_offsets, dtype=np.int64),
        np.ascontiguousarray(own, dtype=np.int64),
        np.ascontiguousarray(expanded.expanded, dtype=np.uint16),
        np.ascontiguousarray(node_off, dtype=np.int64),
        np.ascontiguousarray(layout.prefix_len, dtype=np.int64),
        np.ascontiguousarray(surviving, dtype=np.int64),
        np.ascontiguousarray(row_of_node, dtype=np.int64),
        np.ascontiguousarray(variant_prefix_tokens, dtype=np.uint16),
        np.ascontiguousarray(variant_prefix_offsets, dtype=np.int64),
    )
    return flat.reshape(n_rows, seq_len)
