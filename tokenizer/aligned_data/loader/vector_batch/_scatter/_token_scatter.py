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

from ._expand import ExpandedBatch
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
    tokens = np.zeros((n_rows, seq_len), dtype=np.uint16)
    if n_rows == 0 or seq_len == 0:
        return tokens

    emission = geometry.emission
    layout = geometry.layout
    roff = np.asarray(emission.row_offsets, dtype=np.int64)
    own = np.asarray(emission.own_length, dtype=np.int64)
    node_off = np.asarray(expanded.node_offsets, dtype=np.int64)
    n_emitted = own.size
    if node_off.size - 1 != n_emitted:
        raise ValueError(
            f"expanded covers {node_off.size - 1} nodes but the emission "
            f"has {n_emitted}"
        )

    pref = np.asarray(layout.prefix_len, dtype=np.int64)
    straddler_local = np.asarray(layout.straddler_local_idx, dtype=np.int64)
    partial_cut = np.asarray(layout.partial_cut_length, dtype=np.int64)

    # --- per-emitted-node geometry ---------------------------------------
    row_of_node = np.repeat(
        np.arange(n_rows, dtype=np.int64), np.diff(roff)
    )
    local_idx = np.arange(n_emitted, dtype=np.int64) - roff[row_of_node]
    # Body-relative start column of each node = own-length prefix-sum of
    # the row's prior nodes (global cumsum minus the row's base cumsum).
    gcum = np.concatenate(([0], np.cumsum(own)))
    body_base = gcum[roff[:-1]]
    body_start = gcum[:-1] - body_base[row_of_node]  # int64[n_emitted]

    # --- per-node surviving column count ---------------------------------
    # Default: the whole own_length survives. The straddler keeps only its
    # partial_cut prefix; nodes after the straddler keep nothing. Rows
    # with no straddler (straddler_local == -1) keep every node whole.
    straddler_of_node = straddler_local[row_of_node]
    cut_of_node = partial_cut[row_of_node]
    has_straddler = straddler_of_node >= 0
    after_straddler = has_straddler & (local_idx > straddler_of_node)
    is_straddler = has_straddler & (local_idx == straddler_of_node)

    surviving = own.copy()
    surviving[is_straddler] = cut_of_node[is_straddler]
    surviving[after_straddler] = 0

    # --- flat (row, col, value) of every surviving body slot -------------
    rows_flat, cols_flat, vals_flat = _flatten_node_writes(
        expanded.expanded,
        node_off,
        surviving,
        row_of_node,
        dest_col_base=pref[row_of_node] + body_start,
    )

    # --- per-row variant prefix slots ------------------------------------
    p_rows, p_cols, p_vals = _flatten_prefix_writes(
        variant_prefix_tokens,
        np.asarray(variant_prefix_offsets, dtype=np.int64),
        seq_len,
    )

    all_rows = np.concatenate([p_rows, rows_flat])
    all_cols = np.concatenate([p_cols, cols_flat])
    all_vals = np.concatenate([p_vals, vals_flat])

    # Defensive: every destination column is < seq_len by construction
    # (the straddler cut + body prefix-sum keep writes inside L, and the
    # prefix write is capped). Guard against any upstream drift rather
    # than silently writing out of bounds.
    in_bounds = (all_cols >= 0) & (all_cols < seq_len)
    if not bool(in_bounds.all()):
        all_rows = all_rows[in_bounds]
        all_cols = all_cols[in_bounds]
        all_vals = all_vals[in_bounds]

    tokens[all_rows, all_cols] = all_vals
    return tokens


def _flatten_node_writes(
    expanded: np.ndarray,
    node_off: np.ndarray,
    surviving: np.ndarray,
    row_of_node: np.ndarray,
    dest_col_base: np.ndarray,
):
    """Flat ``(rows, cols, values)`` for every surviving body slot.

    For node ``e`` keeping ``surviving[e]`` of its ``expanded`` slots:
    the kept values are ``expanded[node_off[e] : node_off[e] +
    surviving[e]]``, destined for row ``row_of_node[e]`` at columns
    ``dest_col_base[e] + (0 .. surviving[e] - 1)``. Built with the
    cumulative-offset arange (no per-node Python loop).
    """
    total = int(surviving.sum())
    if total == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), np.zeros(0, dtype=np.uint16)

    keep = surviving > 0
    src_base = node_off[:-1][keep]
    surv = surviving[keep]
    col_base = dest_col_base[keep]
    rows = row_of_node[keep]

    # Per-kept-slot within-node offset 0,1,...,surv[i]-1 via arange minus
    # the repeated segment start.
    seg_start = np.concatenate(([0], np.cumsum(surv)))
    within = np.arange(total, dtype=np.int64) - np.repeat(
        seg_start[:-1], surv
    )
    src_idx = np.repeat(src_base, surv) + within
    cols = np.repeat(col_base, surv) + within
    rows_flat = np.repeat(rows, surv)
    vals = expanded[src_idx]
    return rows_flat, cols, vals


def _flatten_prefix_writes(
    prefix_tokens: np.ndarray,
    prefix_offsets: np.ndarray,
    seq_len: int,
):
    """Flat ``(rows, cols, values)`` for every variant-prefix slot.

    Row ``r``'s prefix is ``prefix_tokens[prefix_offsets[r] :
    prefix_offsets[r + 1]]`` written at columns ``0 .. width - 1``,
    capped at ``seq_len`` (a degenerate prefix wider than the budget is
    truncated, matching the scalar assembler's ``min(n_axis,
    context_len)``).
    """
    n_rows = prefix_offsets.size - 1
    widths = np.diff(prefix_offsets)
    capped = np.minimum(widths, seq_len)
    total = int(capped.sum())
    if total == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), np.zeros(0, dtype=np.uint16)

    keep = capped > 0
    base = prefix_offsets[:-1][keep]
    cap = capped[keep]
    rows = np.arange(n_rows, dtype=np.int64)[keep]

    seg_start = np.concatenate(([0], np.cumsum(cap)))
    within = np.arange(total, dtype=np.int64) - np.repeat(seg_start[:-1], cap)
    src_idx = np.repeat(base, cap) + within
    cols = within  # prefix always starts at column 0
    rows_flat = np.repeat(rows, cap)
    vals = prefix_tokens[src_idx]
    return rows_flat, cols, vals
