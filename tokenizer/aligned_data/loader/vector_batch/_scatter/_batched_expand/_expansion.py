"""The :class:`BatchedExpansion` record + the one-pass orchestrator.

Single concern: assemble the flat batched expansion -- compute the
boundary-aware InlineDecodeState fields (:mod:`._state_fields`), run the
VC2 / F128 promotion + strip-shift-prepend rewrite (:mod:`._rewrite`),
and pack the result into :class:`BatchedExpansion`. The MATH this
orchestrates is OWNED by the scalar kernels and asserted equivalent by
the cross-check unit test + the corpus byte-identity gate; this module
only sequences the batched twin over the whole flat CSR ``raw`` stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._constants import (
    _FLOAT128_VOCAB_ID,
    _V2_EAGER_BLOCK_END,
    _V2_RESERVED_DIGIT_COUNT,
    _V2_VALUE_NEGATIVE_TOKEN_ID,
    _VC2_VOCAB_ID,
)
from ._rewrite import _promote_batched, _strip_shift_prepend
from ._state_fields import (
    _batched_is_negative,
    _boundary_run_lengths,
    _per_node_digit_cumsum,
)


__all__ = ["BatchedExpansion", "batched_expand"]


@dataclass(frozen=True)
class BatchedExpansion:
    """Flat batched expansion + the per-node STATE fields, all CSR.

    Every array spans the whole batch; per-node slices are taken by
    :mod:`._expand` (CSR ``rec`` for the raw-space arrays, CSR
    ``node_offsets`` for the expanded-space arrays). The state-field
    arrays are exactly the position-by-position
    :class:`...decoded._inline_decode_state.InlineDecodeState` fields,
    boundary-aware so each node slice equals the per-node build.
    """

    # Expanded (model-facing) stream + CSR jump table.
    expanded: np.ndarray  # uint16[total_expanded]
    node_offsets: np.ndarray  # int64[n_nodes + 1]

    # Promotion masks over the EXPANDED stream (slot 0 of each node = the
    # prepended self-token, always False), parallel to ``expanded``.
    extra_value_v2_mask: np.ndarray  # bool[total_expanded]
    extra_f128_mask: np.ndarray  # bool[total_expanded]

    # Per-position InlineDecodeState fields over the flat RAW stream
    # (CSR ``rec``). ``digit_cumsum`` is a per-node exclusive prefix with
    # an extra trailing slot per node -- size ``total_raw + n_nodes``,
    # node ``i``'s block at ``rec_starts[i] + i`` (see :func:`batched_expand`).
    real_mask: np.ndarray  # bool[total_raw]
    number_mask: np.ndarray  # bool[total_raw]
    runlen_number: np.ndarray  # uint16[total_raw]
    runlen_value: np.ndarray  # uint16[total_raw]
    carries_inline_mask: np.ndarray  # bool[total_raw]
    is_negative_per_position: np.ndarray  # bool[total_raw]
    digit_cumsum: np.ndarray  # uint32[total_raw + n_nodes]


def batched_expand(
    raw: np.ndarray,
    record_offsets: np.ndarray,
    self_token_ids: np.ndarray,
) -> BatchedExpansion:
    """Promote + strip + shift + prepend every node body, one batched pass.

    Parameters
    ----------
    raw:
        The flat gathered ``uint16`` body stream (CSR over ``record_offsets``).
    record_offsets:
        ``int64[n_nodes + 1]`` CSR into ``raw`` (node ``i`` owns
        ``raw[record_offsets[i] : record_offsets[i + 1]]``).
    self_token_ids:
        ``uint16[n_nodes]`` the SHIFTED calling-category self-token id for
        each node (the value the scalar ``expand_tokens`` writes at
        ``expanded[0]``); the ``CallTargetType -> Category -> shifted id``
        collapse is the caller's concern (:mod:`._expand`).

    Returns
    -------
    BatchedExpansion
        The flat expanded stream + CSR + promotion masks + the per-node
        InlineDecodeState fields, all batched.
    """
    raw = np.asarray(raw, dtype=np.uint16).reshape(-1)
    rec = np.asarray(record_offsets, dtype=np.int64).reshape(-1)
    n_nodes = rec.size - 1
    total = raw.shape[0]

    rec_starts = rec[:-1]
    rec_ends = rec[1:]
    counts = rec_ends - rec_starts
    node_of = (
        np.repeat(np.arange(n_nodes, dtype=np.int64), counts)
        if total
        else np.zeros(0, dtype=np.int64)
    )

    # --- per-position InlineDecodeState fields (boundary-aware) ----------
    real_mask = raw > _V2_VALUE_NEGATIVE_TOKEN_ID
    number_mask = raw < _V2_RESERVED_DIGIT_COUNT
    value_mask = ~real_mask
    runlen_number = _boundary_run_lengths(number_mask, rec_starts, counts)
    runlen_value = _boundary_run_lengths(value_mask, rec_starts, counts)
    carries_inline_mask = real_mask & (raw < _V2_EAGER_BLOCK_END)
    digit_cumsum = _per_node_digit_cumsum(
        number_mask, rec_starts, counts, n_nodes
    )
    is_negative_per_position = _batched_is_negative(
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        rec_starts=rec_starts,
        counts=counts,
        total=total,
    )

    # --- promotion (paint into a working copy of raw) --------------------
    # The promotion paint is the ONLY mutation of the raw stream, and it
    # only fires when a VC2 / F128 carrier token is present. With no carrier
    # the working stream is never written, so the defensive copy is dead --
    # alias ``raw`` directly (the downstream strip/shift only READS it).
    has_carrier = bool(
        (real_mask & ((raw == _VC2_VOCAB_ID) | (raw == _FLOAT128_VOCAB_ID))).any()
    )
    working = raw.copy() if has_carrier else raw
    extra_vc2_raw, extra_f128_raw = _promote_batched(
        working, real_mask, runlen_number, node_of, rec_starts, counts
    )

    # --- strip + shift + prepend self-token ------------------------------
    (
        expanded,
        node_offsets,
        extra_value_v2_mask,
        extra_f128_mask,
    ) = _strip_shift_prepend(
        working,
        extra_vc2_raw,
        extra_f128_raw,
        node_of,
        rec_starts,
        counts,
        n_nodes,
        self_token_ids,
    )

    return BatchedExpansion(
        expanded=expanded,
        node_offsets=node_offsets,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
        digit_cumsum=digit_cumsum,
    )
