"""Boundary-aware InlineDecodeState field math over the flat raw stream.

Single concern: reproduce the per-node
:class:`...decoded._inline_decode_state.InlineDecodeState` fields (run
lengths, the digit cumsum, the is-negative flag) as boundary-aware
vectorized passes so each per-node SLICE equals the scalar
:func:`build_inline_decode_state` on that node's raw stream. The
raw-stream rewrite that consumes these lives in :mod:`._rewrite`; the
orchestration in :mod:`._expansion`.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "_boundary_run_lengths",
    "_per_node_digit_cumsum",
    "_batched_is_negative",
]


def _boundary_run_lengths(
    mask: np.ndarray, rec_starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Per-node ``run_lengths`` over the concatenated flat ``mask``.

    Reproduces :func:`...decoded.run_lengths.run_lengths` applied to each
    node's slice independently: the run length is carried at each run's
    FIRST position (0 elsewhere), and runs never cross a node boundary.

    A run starts at ``k`` iff ``mask[k]`` and (``k`` is a node start OR
    ``not mask[k-1]``); a run ends at ``k`` iff ``mask[k]`` and
    (``not mask[k+1]`` OR ``k+1`` starts a new node). The length at a
    run-start is ``end - start + 1`` -- exactly the scalar output.
    """
    total = mask.shape[0]
    out = np.zeros(total, dtype=np.uint16)
    if total == 0:
        return out

    is_rec_start = np.zeros(total, dtype=bool)
    # Empty nodes collapse rec_starts onto the next node's start (or
    # ``total``); only non-empty nodes mark a real boundary.
    is_rec_start[rec_starts[counts > 0]] = True

    prev = np.empty(total, dtype=bool)
    prev[0] = False
    prev[1:] = mask[:-1]
    prev[is_rec_start] = False  # a boundary breaks the prior run
    run_start = mask & ~prev

    nxt = np.empty(total, dtype=bool)
    nxt[-1] = False
    nxt[:-1] = mask[1:]
    next_is_rec_start = np.zeros(total, dtype=bool)
    next_is_rec_start[:-1] = is_rec_start[1:]
    run_end = mask & (~nxt | next_is_rec_start)

    start_idx = np.nonzero(run_start)[0]
    end_idx = np.nonzero(run_end)[0]
    out[start_idx] = (end_idx - start_idx + 1).astype(np.uint16)
    return out


def _per_node_digit_cumsum(
    number_mask: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Per-node exclusive-prefix cumsum of ``number_mask``, packed CSR.

    The scalar ``build_inline_decode_state`` builds a length ``N + 1``
    array per node: ``digit_cumsum[0] = 0`` and ``digit_cumsum[k] =
    sum(number_mask[0:k])``. Packed here as one ``uint32`` array of size
    ``total_raw + n_nodes`` where node ``i``'s ``(count_i + 1)``-slot
    block starts at ``rec_starts[i] + i``; the slice
    ``out[rec_starts[i] + i : rec_starts[i + 1] + (i + 1)]`` is node
    ``i``'s ``digit_cumsum``.
    """
    total = number_mask.shape[0]
    out = np.zeros(total + n_nodes, dtype=np.uint32)
    if n_nodes == 0:
        return out

    # Global exclusive prefix (length total + 1): global_excl[k] =
    # sum(number_mask[0:k]).
    global_excl = np.zeros(total + 1, dtype=np.uint32)
    if total > 0:
        np.cumsum(number_mask.view(np.uint8), out=global_excl[1:])

    rec_ends = rec_starts + counts
    node_base = global_excl[rec_starts]  # digits before each node
    node_idx = np.arange(n_nodes, dtype=np.int64)
    dst_block_start = rec_starts + node_idx  # node i's block start

    # Body slots: per-node exclusive prefix at every raw position p is
    # global_excl[p] - node_base[node_of[p]]; dst index = p + node_of[p].
    if total > 0:
        node_of = np.repeat(node_idx, counts)
        raw_pos = np.arange(total, dtype=np.int64)
        out[raw_pos + node_of] = global_excl[:total] - node_base[node_of]
    # Trailing slot of each node (the ``N + 1``-th) = node's digit total.
    out[dst_block_start + counts] = global_excl[rec_ends] - node_base
    return out


def _batched_is_negative(
    *,
    runlen_number: np.ndarray,
    runlen_value: np.ndarray,
    carries_inline_mask: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
    total: int,
) -> np.ndarray:
    """Boundary-aware twin of ``compute_is_negative_per_position``.

    The scalar formula flags a carrier at ``p`` whose ``p+1`` slot is a
    sign marker via ``runlen_value[p+1] != runlen_number[p+1]``, and
    NEVER flags a carrier at a node's LAST position (no ``p+1`` slot).
    Reproduced over the flat stream by excluding each node's last
    position from the carrier candidates -- the run-length arrays are
    already boundary-local, so the ``p+1`` lookup stays inside the node.
    """
    out = np.zeros(total, dtype=bool)
    if total == 0:
        return out
    is_node_last = np.zeros(total, dtype=bool)
    is_node_last[(rec_starts + counts - 1)[counts > 0]] = True
    cand_idx = np.nonzero(carries_inline_mask & ~is_node_last)[0]
    if cand_idx.size:
        out[cand_idx] = (
            runlen_value[cand_idx + 1] != runlen_number[cand_idx + 1]
        )
    return out
