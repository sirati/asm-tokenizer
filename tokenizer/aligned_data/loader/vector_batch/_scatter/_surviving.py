"""Per-emitted-node surviving column (token) count under the straddler cut.

Single concern: compute, for every flat emitted node, how many of its
``own_length`` token columns SURVIVE the per-row ``[B, L]`` straddler cut
-- the byproduct the body read makes available ("adjust during
decoding", plan C1/C2). The token scatter places exactly this many
columns; the dense-sidecar pass classifies exactly this many expanded
positions into the identity / number bands. Both consume the SAME
function so the cut point can never drift between the two arms.

The math is pure geometry (body-free): the straddler keeps only its
``partial_cut_length`` prefix; nodes AFTER the straddler in a row keep
nothing; rows with no straddler keep every node whole. This is the
``surviving`` array the token scatter (:mod:`._token_scatter`) used to
compute inline; it is lifted here so the dense pass reuses it verbatim
instead of recomputing (no duplicated cut logic).
"""

from __future__ import annotations

import numpy as np

from .._types import BatchGeometry


__all__ = ["surviving_token_counts", "row_of_node"]


def row_of_node(geometry: BatchGeometry) -> np.ndarray:
    """``int64[n_emitted]`` -- the batch row owning each flat emitted node.

    Built from the emission CSR ``row_offsets`` via ``np.repeat`` (the
    same construction the token scatter uses). Shared so the dense pass
    and the token scatter agree on the per-node row assignment.
    """
    roff = np.asarray(geometry.emission.row_offsets, dtype=np.int64)
    n_rows = int(geometry.n_rows)
    return np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(roff))


def surviving_token_counts(geometry: BatchGeometry) -> np.ndarray:
    """Per-emitted-node surviving column count after the straddler cut.

    Parameters
    ----------
    geometry:
        The body-free prepass result. Provides the emission CSR + per-
        node ``own_length`` (full column span) and the per-row layout
        (straddler local index + partial-cut length).

    Returns
    -------
    np.ndarray
        ``int64[n_emitted]`` -- node ``e`` contributes ``surviving[e]``
        of its ``own_length`` columns: the whole length for nodes before
        the straddler (and in full rows), ``partial_cut_length`` for the
        straddler, ``0`` for nodes after the straddler. This is exactly
        the ``Stage2CallTarget.partial_cut_length`` of the matching
        decode call_target (= its ``surviving_token_count``).
    """
    emission = geometry.emission
    layout = geometry.layout
    own = np.asarray(emission.own_length, dtype=np.int64)
    roff = np.asarray(emission.row_offsets, dtype=np.int64)
    n_emitted = own.size

    rofn = row_of_node(geometry)
    local_idx = np.arange(n_emitted, dtype=np.int64) - roff[:-1][rofn]

    straddler_local = np.asarray(layout.straddler_local_idx, dtype=np.int64)
    partial_cut = np.asarray(layout.partial_cut_length, dtype=np.int64)
    straddler_of_node = straddler_local[rofn]
    cut_of_node = partial_cut[rofn]
    has_straddler = straddler_of_node >= 0
    after_straddler = has_straddler & (local_idx > straddler_of_node)
    is_straddler = has_straddler & (local_idx == straddler_of_node)

    surviving = own.copy()
    surviving[is_straddler] = cut_of_node[is_straddler]
    surviving[after_straddler] = 0
    return surviving
