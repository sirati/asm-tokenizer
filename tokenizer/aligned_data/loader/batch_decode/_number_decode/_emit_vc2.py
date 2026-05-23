"""Per-source row emission for the VC2 (``valued_const_v2``) TokenType.

Single concern: emit ``K_visible`` rows for each VC2 carrier in a
batched-vectorized form per ALG-8. The MSB chunk may have fewer than
8 payload bytes when ``L % 8 != 0``; padding slots reference
``inline_bytes[0]`` (3a's leading-zero pad).

Vectorisation strategy
----------------------

* ``K_visible`` per carrier is computed via a single run-length
  ``cumsum`` trick over ``extra_value_v2_mask[:surviving]``: the count
  of consecutive True positions starting one slot past each carrier
  capped by both the surviving prefix and ``K_full[c] = max(1, ceil(
  L[c] / 8))``.
* Per-chunk byte rows are built via a single meshgrid over all
  carriers' chunks in one shot; the per-chunk MSB-clipping uses
  ``np.where`` to substitute the leading-zero pad reference for the
  short MSB chunk's pad bytes.
"""

from __future__ import annotations

import numpy as np

from tokenizer.tokens import TokenType


__all__ = ["_emit_vc2_sources"]


def _emit_vc2_sources(
    *,
    state_runlen_number: np.ndarray,
    p_carriers: np.ndarray,
    p_carrier_bytes: np.ndarray,
    expanded_positions: np.ndarray,
    extra_value_v2_mask: np.ndarray,
    surviving: int,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    vc2_chunk_indices: list[int],
) -> None:
    """Emit per-chunk byte rows for every VC2 carrier per ALG-8.

    See :mod:`._entry` module docstring (verbatim ALG-8 block) for the
    per-chunk byte-range formula. Per carrier ``c``:

    * ``L[c] = state_runlen_number[p_carrier[c] + 1]``.
    * ``K_full[c] = max(1, ceil(L[c] / 8))``.
    * ``K_visible[c]`` = ``1 +`` count of consecutive
      ``extra_value_v2_mask`` True positions immediately after the
      carrier in expanded space, capped by both the surviving prefix
      and ``K_full[c]``.

    Stream-emission order across carriers is preserved -- each carrier
    contributes ``K_visible[c]`` rows (chunks 0..K_visible[c]-1) and
    the rows are appended in carrier order.

    Short MSB chunks left-pad with ``inline_bytes[0]`` references
    (zeros): for ``L=17`` the MSB chunk yields
    ``[0]*7 + [p_carrier_byte]``.
    """
    n_carriers = int(p_carriers.shape[0])
    if n_carriers == 0:
        return

    # ALG-8: ``L = state.runlen_number[p_carrier + 1]``. The carrier
    # always has a p+1 slot per _promote_vc2's tail assertion.
    L = state_runlen_number[p_carriers + 1].astype(np.int64)
    K_full = np.maximum(1, (L + 7) // 8)

    # K_visible per carrier: count of consecutive True in
    # ``extra_value_v2_mask`` strictly after each carrier's expanded
    # position, capped by ``surviving`` (open bound) and ``K_full[c]``.
    #
    # Build it without a per-position Python loop by mapping each
    # expanded slot to "how many True painted slots remain to its
    # right in the current run of True"; the value at carrier+1 then
    # gives the run length of painted continuations following the
    # carrier.
    extra_int = extra_value_v2_mask[:surviving].astype(np.int64)
    if extra_int.size == 0:
        trailing_true_run = np.empty(0, dtype=np.int64)
    else:
        # ``trailing_true_run[i]`` = number of consecutive True at
        # positions i, i+1, ... in extra_value_v2_mask[:surviving].
        # Computed via reverse-cumsum + segment-reset: reverse the
        # mask, run cumsum, then per-segment subtract via
        # maximum.accumulate of the cum value at each False slot.
        reversed_mask = extra_int[::-1]
        cum = np.cumsum(reversed_mask)
        # Indices where ``reversed_mask`` is False; assign each such
        # index the value of ``cum`` to subtract going forward.
        reset_values = np.where(reversed_mask == 0, cum, 0)
        carried = np.maximum.accumulate(reset_values)
        trailing_true_run = (cum - carried)[::-1].copy()

    K_visible = np.ones(n_carriers, dtype=np.int64)
    # For each carrier, look one slot past the carrier in expanded
    # space; the painted run starting there is ``trailing_true_run``.
    lookahead_positions = expanded_positions + 1
    in_range = lookahead_positions < surviving
    if in_range.any():
        run_lengths = np.zeros(n_carriers, dtype=np.int64)
        run_lengths[in_range] = trailing_true_run[
            lookahead_positions[in_range]
        ]
        K_visible = 1 + np.minimum(run_lengths, K_full - 1)
    # K_visible already capped by K_full via the minimum above.

    # ALG-8 per-chunk byte range: ``[p_carrier_byte + L - 8*(c+1),
    # p_carrier_byte + L - 8*c)`` intersected with the payload region.
    # The intersection only matters for the MSB chunk (``c == K_full -
    # 1``, ``L % 8 != 0``); leading slots reference inline_bytes[0].
    #
    # Flatten every carrier's ``K_visible[c]`` rows into one
    # ``(total_rows, 8)`` block: per output row ``r``, recover
    # ``source_idx`` and ``c_within`` via searchsorted on the
    # per-carrier ``source_starts`` cumulative offset, then evaluate
    # the per-chunk meshgrid in one shot.
    total_rows = int(K_visible.sum())
    source_starts = np.empty(n_carriers + 1, dtype=np.int64)
    source_starts[0] = 0
    np.cumsum(K_visible, out=source_starts[1:])

    row_indices = np.arange(total_rows, dtype=np.int64)
    source_idx_per_row = np.searchsorted(
        source_starts[1:], row_indices, side="right"
    )
    c_within = row_indices - source_starts[source_idx_per_row]

    p_carrier_bytes_i64 = p_carrier_bytes.astype(np.int64, copy=False)
    L_per_row = L[source_idx_per_row]
    p_per_row = p_carrier_bytes_i64[source_idx_per_row]
    unclipped_starts = p_per_row + L_per_row - 8 * (c_within + 1)
    byte_idx = np.arange(8, dtype=np.int64)
    cols = unclipped_starts[:, np.newaxis] + byte_idx[np.newaxis, :]
    rows = np.where(
        cols < p_per_row[:, np.newaxis],
        np.uint32(0),
        cols.astype(np.uint32),
    )

    row_lists_per_type[TokenType.VALUED_CONST_V2].append(rows)
    running_counts[TokenType.VALUED_CONST_V2] += total_rows
    vc2_chunk_indices.extend(c_within.tolist())
