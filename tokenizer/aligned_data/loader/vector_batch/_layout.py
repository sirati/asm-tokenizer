"""Body-free per-row ``[B, L]`` token-column layout + straddler cut.

Single concern: from the flat (CSR) per-row emitted-function own-lengths
+ per-row variant-prefix widths, compute -- FULLY VECTORIZED across the
batch, no per-row Python loop -- each row's total emitted token width,
the SINGLE straddler function crossing the seq_len ``L``, and the
straddler's ``partial_cut_length`` (token columns of it that fit).

WHY a single straddler per row: own-lengths are summed in BFS emission
order; the row's column extent is the strictly-increasing prefix-sum of
those own-lengths (offset by the variant prefix). At most ONE function's
span contains column ``L`` -- the one whose cumulative end first reaches
or passes ``L``. ``searchsorted`` finds it batch-wide in one dispatch.

Body-free: own-lengths come from RLG3 stored body lengths (+ self-token),
prefix widths from ``_variants.bin``; no ``_data.bin`` is read. The
straddler's dense PARTIAL count is NOT computed here (it needs the body
-- TC2's concern); only the token-column cut is.
"""

from __future__ import annotations

import numpy as np

from ._types import BatchTokenLayout


__all__ = ["compute_token_layout"]


def compute_token_layout(
    *,
    own_length: np.ndarray,
    row_offsets: np.ndarray,
    prefix_len: np.ndarray,
    seq_len: int,
) -> BatchTokenLayout:
    """Per-row token-column layout + straddler cut, vectorized.

    Parameters
    ----------
    own_length:
        ``int[n_emitted]`` -- flat per-emitted-function column span
        (``1 + body_len``), CSR-grouped by ``row_offsets`` in BFS order.
    row_offsets:
        ``int[B + 1]`` -- CSR offsets into ``own_length`` (row ``r`` owns
        ``[row_offsets[r] : row_offsets[r + 1]]``).
    prefix_len:
        ``int[B]`` -- the variant-prefix width prepended ahead of each
        row's root body.
    seq_len:
        ``L`` -- the per-row token-column budget rows truncate to
        (``>= 0``).

    Returns
    -------
    BatchTokenLayout
        Per-row prefix width, straddler local index (-1 if full),
        ``partial_cut_length`` (0 if full), and pre-truncation total
        length.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0; got {seq_len}")
    own = np.asarray(own_length, dtype=np.int64).reshape(-1)
    roff = np.asarray(row_offsets, dtype=np.int64).reshape(-1)
    pref = np.asarray(prefix_len, dtype=np.int64).reshape(-1)
    n_rows = roff.size - 1
    if pref.size != n_rows:
        raise ValueError(
            f"prefix_len has {pref.size} entries but row_offsets implies "
            f"{n_rows} rows"
        )

    # Inclusive within-row cumulative END column of each emitted function,
    # measured FROM THE ROW'S BODY START (prefix excluded; added back per
    # row below). The exclusive segment-prefix-sum trick: a global cumsum
    # minus each row's base cumsum yields per-row running ends without a
    # Python loop.
    gcum = np.concatenate(([0], np.cumsum(own)))  # int64[n_emitted + 1]
    # row of each emitted entry (CSR inverse), and the row's body-relative
    # running end at each entry = gcum[i + 1] - gcum[row_offsets[row]].
    row_of_entry = np.repeat(
        np.arange(n_rows, dtype=np.int64), np.diff(roff)
    )
    body_base = gcum[roff[:-1]]  # int64[B] -- row start in the global cumsum
    running_end = gcum[1:] - body_base[row_of_entry]  # int64[n_emitted]

    # Per-row total BODY width (sum of own-lengths) + the prefix.
    body_total = gcum[roff[1:]] - body_base  # int64[B]
    total_length = pref + body_total

    # The straddler: the FIRST emitted function in the row whose
    # prefix-offset running end exceeds ``L``. Equivalent to searching
    # each row's running_end (body-relative) for ``L - prefix_len`` and
    # taking the first end strictly greater than the remaining budget.
    remaining = seq_len - pref  # int64[B] -- body columns that fit
    straddler_local = np.full(n_rows, -1, dtype=np.int64)
    partial_cut = np.zeros(n_rows, dtype=np.int64)

    # A row is FULL iff its whole body fits (body_total <= remaining) AND
    # the prefix itself fits (remaining >= 0). Otherwise it straddles.
    has_emit = np.diff(roff) > 0
    body_fits = body_total <= remaining
    full = has_emit & body_fits & (remaining >= 0)
    straddles = has_emit & ~full

    if bool(straddles.any()):
        s_rows = np.nonzero(straddles)[0]
        # Clamp the per-row remaining budget to [0, body_total) so the
        # searchsorted lands inside the row. A prefix that already
        # overflows (remaining < 0) cuts the very first body function to
        # 0 columns; a body that overflows finds the crossing function.
        rem = np.clip(remaining[s_rows], 0, None)
        lo = roff[s_rows]
        hi = roff[s_rows + 1]
        # First entry whose body-relative running_end is STRICTLY greater
        # than the remaining budget -> the straddler. side="right" finds
        # the first end > rem; entries with end == rem fully fit.
        flat_pos = _segmented_searchsorted(running_end, lo, hi, rem)
        local = flat_pos - lo
        straddler_local[s_rows] = local
        # partial_cut = remaining budget consumed within the straddler =
        # rem - (running_end of the previous entry, body-relative). The
        # previous entry's end is running_end[flat_pos - 1] when the
        # straddler is not the first body function, else 0 (the straddler
        # starts at the row's body origin). ``np.maximum(flat_pos - 1,
        # lo)`` keeps the gather in-bounds; the np.where discards it for
        # the first-function case.
        prev_end = np.where(
            flat_pos > lo, running_end[np.maximum(flat_pos - 1, lo)], 0
        )
        partial_cut[s_rows] = np.clip(
            rem - prev_end, 0, own[flat_pos]
        )

    return BatchTokenLayout(
        seq_len=int(seq_len),
        prefix_len=pref,
        straddler_local_idx=straddler_local,
        partial_cut_length=partial_cut,
        total_length=total_length,
    )


def _segmented_searchsorted(
    values: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """First flat index in ``[lo[i], hi[i])`` whose ``values`` entry is
    strictly greater than ``targets[i]`` (``side="right"``), clamped
    inside the segment.

    ``values`` is strictly increasing within each ``[lo[i], hi[i])``
    segment (own-length running ends are a positive prefix-sum;
    own_length = 1 + body_len >= 1 by the bulk-geometry contract, so no
    in-row ties). One ``np.searchsorted`` PER row would re-introduce a
    Python loop; instead every row's searched entries + target are biased
    by a per-row constant ``r * big`` that exceeds the largest in-row
    value, making the concatenation globally monotone -- a SINGLE
    ``searchsorted`` then answers all rows at once. The straddler exists
    for every input row (the caller passes straddling rows only), so the
    result is always ``< hi[i]``.
    """
    n_rows = lo.size
    big = int(values.max()) + 1 if values.size else 1
    searched_lengths = hi - lo
    # Contiguous per-row gather of the searched segments + their row id,
    # biased into one globally-monotone key.
    gather = _concat_ranges(lo, hi)
    entry_row = np.repeat(np.arange(n_rows, dtype=np.int64), searched_lengths)
    seg_vals = values[gather] + entry_row * big
    seg_targets = targets + np.arange(n_rows, dtype=np.int64) * big
    pos = np.searchsorted(seg_vals, seg_targets, side="right")
    # ``pos`` indexes the contiguous gather; convert to row-local then to
    # the original flat index, clamped inside the segment.
    seg_starts = np.concatenate(([0], np.cumsum(searched_lengths)))
    local = np.minimum(pos - seg_starts[:-1], searched_lengths - 1)
    return lo + local


def _concat_ranges(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Flat concatenation of ``range(lo[i], hi[i])`` for every ``i``.

    Vectorized (no Python per-range loop): the classic cumulative-offset
    ``arange`` construction.
    """
    lengths = hi - lo
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(lo, lengths)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths
    )
    return starts + within
