"""Body-free per-row dense id / value RESERVATION totals + offsets.

Single concern: from the flat (CSR) per-emitted-function stored TOTAL
id / value counts, compute -- vectorized, no per-row loop -- each row's
reserved dense identity / numeric extent (the UPPER BOUND that sizes the
dense arrays) and the cumulative offsets where TC2 places each row's
block.

WHY upper bound, not the surviving / straddler-partial count: those need
the body (the straddler is cut mid-function; pruned-at-decode survivors
are a body property), and the prepass is body-free. TC2's scatter reads
the body and TIGHTENS the actual offsets within these reservations.
With backfill OFF the actual extent equals the reservation (the
reservation is tight) -- so this is also the byte-identity extent.
"""

from __future__ import annotations

import numpy as np

from ._types import DenseReservation


__all__ = ["compute_dense_reservation"]


def compute_dense_reservation(
    *,
    id_total: np.ndarray,
    value_total: np.ndarray,
    row_offsets: np.ndarray,
) -> DenseReservation:
    """Per-row dense id / value reservation totals + cumulative offsets.

    Parameters
    ----------
    id_total / value_total:
        ``int[n_emitted]`` -- flat per-emitted-function stored TOTAL
        identity / numeric counts (RLG3 ``id_counts`` / ``value_counts``
        of each emitted node), CSR-grouped by ``row_offsets``.
    row_offsets:
        ``int[B + 1]`` -- CSR offsets into the flat count arrays.

    Returns
    -------
    DenseReservation
        Per-row reserved totals + their exclusive-prefix-sum offsets, for
        both the identity and the numeric dense arrays.
    """
    ids = np.asarray(id_total, dtype=np.int64).reshape(-1)
    vals = np.asarray(value_total, dtype=np.int64).reshape(-1)
    roff = np.asarray(row_offsets, dtype=np.int64).reshape(-1)

    id_reserved = _segmented_sum(ids, roff)
    value_reserved = _segmented_sum(vals, roff)
    return DenseReservation(
        id_reserved=id_reserved,
        id_offsets=_exclusive_prefix(id_reserved),
        value_reserved=value_reserved,
        value_offsets=_exclusive_prefix(value_reserved),
    )


def _segmented_sum(flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-segment sum of ``flat`` over the CSR ``offsets`` (one cumsum)."""
    cum = np.concatenate(([0], np.cumsum(flat)))
    return cum[offsets[1:]] - cum[offsets[:-1]]


def _exclusive_prefix(counts: np.ndarray) -> np.ndarray:
    """``int64[len + 1]`` exclusive prefix sum (CSR offsets)."""
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out
