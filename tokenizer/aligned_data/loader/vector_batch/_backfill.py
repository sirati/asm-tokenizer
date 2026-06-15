"""Leftover-budget BACKFILL transform (plan TD) -- geometry -> geometry.

Single concern: GREEDILY pack each batch row's LEFTOVER ``[B, L]`` token
budget with functions drawn from THAT row's remembered-excluded pool,
producing an AUGMENTED :class:`BatchGeometry` the fused scatter consumes
UNCHANGED. This is a NEW opt-in behavior, OFF by default -- it is applied
explicitly BETWEEN the geometry prepass and the scatter; the default path
never calls it.

Boundary contract (the design-first sentence):

  *Given a body-free :class:`BatchGeometry` + the SAME RLG3 geometry axes
  the prepass sized from, return a new :class:`BatchGeometry` whose every
  row has had its leftover token budget filled from its excluded_pool,
  with layout / reservation RECOMPUTED through the shared
  :mod:`._layout` / :mod:`._reserve` passes so the scatter sees a valid,
  consistent geometry -- callers never touch the packing internals.*

WHY the excluded_pool is the (only) candidate source: those nodes are
the legitimate, format-valid callees the SAMPLED-subset BFS pruned that
the FULL-set index had included (we sample << num_variants, so re-running
inclusion over the subset excludes MORE than the index assumed). They are
exactly the functions that "would have been there" -- safe to re-inline.

THE GREEDY RULE (chosen + documented):

  fit-fully, in POOL ORDER. For each row, walk its ``excluded_pool`` in
  its STORED (inclusion-BFS-recorded) order; append a candidate iff it
  FULLY fits the row's CURRENT remaining budget
  (``own_length(candidate) <= L - total_length``) AND it is not already
  in the row's emission AND not already backfilled this row. Stop a row
  when no remaining pool function fits (a later, smaller one is STILL
  tried -- the budget can admit a smaller function after a larger one was
  skipped). The remaining budget ``L - total_length`` is the GATE: a row
  that already fit stays ``<= L`` (so backfill never creates a
  straddler), and a row already OVER L (a truncated row whose body is the
  straddler cut) has a NEGATIVE remaining and so gains nothing -- backfill
  is a strict no-op there.

  Why pool order, not descending-own-length: pool order is the canonical
  deterministic order the BFS recorded (the order the full-set index
  would have included these callees); it carries that semantic and is the
  minimally surprising rule. Descending-own-length packs tighter but
  reorders semantically; correctness + determinism are satisfied by pool
  order alone, so we keep it. The choice is a single documented constant.

BODY-FREE: like the prepass, this transform reads NO ``_data.bin`` -- the
appended functions' own_length / id_total / value_total come from the
SAME RLG3 axes (``body_lengths`` / ``id_counts`` / ``value_counts``),
indexed by catalog node, that :func:`...compute_batch_geometry` gathered.
"""

from __future__ import annotations

import numpy as np

from ._layout import compute_token_layout
from ._reserve import compute_dense_reservation
from ._types import (
    BatchGeometry,
    BatchRowEmission,
    BatchTokenLayout,
)


__all__ = ["backfill_geometry"]


def backfill_geometry(
    geometry: BatchGeometry,
    *,
    body_lengths: np.ndarray,
    id_counts: np.ndarray,
    value_counts: np.ndarray,
) -> BatchGeometry:
    """Greedily fill each row's leftover budget from its excluded_pool.

    Parameters
    ----------
    geometry:
        The body-free prepass result to augment. Left UNMODIFIED (a new
        :class:`BatchGeometry` is returned).
    body_lengths / id_counts / value_counts:
        The RLG3 geometry axes (one entry per catalog NODE, in
        ``var_offsets`` order) -- the SAME axes the prepass sized from.
        A backfilled node ``n`` contributes ``own_length = 1 +
        body_lengths[n]`` (the ``own = body + 1`` self-token convention),
        ``id_total = id_counts[n]``, ``value_total = value_counts[n]``.

    Returns
    -------
    BatchGeometry
        A VALID augmented geometry: each row's emission has its chosen
        backfill functions APPENDED (after the original emitted nodes, in
        pool order), with ``layout`` + ``reservation`` recomputed through
        the shared passes so offsets / CSR / prefix-sum / straddler /
        reservations are all consistent. The
        ``excluded_pool`` / ``excluded_pool_offsets`` are carried through
        UNCHANGED (the pool is the immutable candidate record; which
        members were consumed is recoverable from the emission).
    """
    body = np.asarray(body_lengths, dtype=np.int64).reshape(-1)
    ids = np.asarray(id_counts, dtype=np.int64).reshape(-1)
    vals = np.asarray(value_counts, dtype=np.int64).reshape(-1)

    em = geometry.emission
    layout = geometry.layout
    seq_len = layout.seq_len

    n_rows = geometry.n_rows
    pool = geometry.excluded_pool
    pool_off = geometry.excluded_pool_offsets
    pool_edge_type = geometry.excluded_pool_edge_type

    # Per-row append lists; greedy fit-fully in pool order. A row's loop is
    # over its OWN pool only -- never another row's slice -- so a
    # backfilled node is guaranteed to be in that row's excluded_pool.
    appended_per_row = []
    appended_edge_types_per_row = []
    for r in range(n_rows):
        emit_lo = int(em.row_offsets[r])
        emit_hi = int(em.row_offsets[r + 1])
        already = set(em.node[emit_lo:emit_hi].tolist())

        remaining = seq_len - int(layout.total_length[r])
        chosen: list[int] = []
        chosen_types: list[int] = []
        p_lo = int(pool_off[r])
        p_hi = int(pool_off[r + 1])
        # Walk the row's pool slice with its PARALLEL edge-type slice so a
        # chosen node carries its true parent-slot ct_type (the value it
        # held as a pruned edge), never a default.
        for node, edge_type in zip(
            pool[p_lo:p_hi].tolist(), pool_edge_type[p_lo:p_hi].tolist()
        ):
            if node in already:
                continue
            own = 1 + int(body[node])
            if own <= remaining:
                chosen.append(node)
                chosen_types.append(edge_type)
                already.add(node)
                remaining -= own
        appended_per_row.append(np.array(chosen, dtype=np.int64))
        appended_edge_types_per_row.append(
            np.array(chosen_types, dtype=np.uint8)
        )

    aug_emission = _append_emission(
        em, appended_per_row, appended_edge_types_per_row, body, ids, vals
    )

    # The variant PREFIX is a root-node property; the root is untouched by
    # backfill (only appended-after), so the per-row prefix is unchanged.
    aug_layout = compute_token_layout(
        own_length=aug_emission.own_length,
        row_offsets=aug_emission.row_offsets,
        prefix_len=layout.prefix_len,
        seq_len=seq_len,
    )
    aug_reservation = compute_dense_reservation(
        id_total=aug_emission.id_total,
        value_total=aug_emission.value_total,
        row_offsets=aug_emission.row_offsets,
    )

    return BatchGeometry(
        n_rows=n_rows,
        emission=aug_emission,
        layout=aug_layout,
        reservation=aug_reservation,
        excluded_pool=geometry.excluded_pool,
        excluded_pool_offsets=geometry.excluded_pool_offsets,
        excluded_pool_edge_type=geometry.excluded_pool_edge_type,
    )


def _append_emission(
    em: BatchRowEmission,
    appended_per_row,
    appended_edge_types_per_row,
    body: np.ndarray,
    ids: np.ndarray,
    vals: np.ndarray,
) -> BatchRowEmission:
    """Rebuild the flat CSR emission with per-row backfill nodes appended.

    Each row's original emitted slice is kept IN ORDER, then its chosen
    backfill nodes are concatenated AFTER it (pool order). The appended
    nodes' own_length / id_total / value_total are gathered from the SAME
    RLG3 axes the prepass used (``own = 1 + body_len``), so the augmented
    emission stays self-consistent with the original entries. The
    ``edge_type`` axis is the row's ORIGINAL edge types (verbatim, an EDGE
    property the gather-from-node cannot reconstruct) concatenated with the
    appended nodes' pool edge types (the pruned-edge ct_type, parallel to
    ``appended_per_row``).
    """
    n_rows = len(appended_per_row)
    node_parts: list[np.ndarray] = []
    edge_type_parts: list[np.ndarray] = []
    row_lengths = np.empty(n_rows, dtype=np.int64)
    for r in range(n_rows):
        lo = int(em.row_offsets[r])
        hi = int(em.row_offsets[r + 1])
        orig = em.node[lo:hi]
        extra = appended_per_row[r]
        node_parts.append(orig)
        node_parts.append(extra)
        # edge_type is an EDGE property (not derivable from the node): keep
        # the original slice verbatim, then the appended pool edge types.
        edge_type_parts.append(em.edge_type[lo:hi])
        edge_type_parts.append(appended_edge_types_per_row[r])
        row_lengths[r] = orig.size + extra.size

    node = (
        np.concatenate(node_parts).astype(np.int64)
        if node_parts
        else np.zeros(0, dtype=np.int64)
    )
    edge_type = (
        np.concatenate(edge_type_parts).astype(np.uint8)
        if edge_type_parts
        else np.zeros(0, dtype=np.uint8)
    )
    row_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(row_lengths, out=row_offsets[1:])

    # Re-derive the geometry triple for the FULL (original + appended) node
    # list from the RLG3 axes -- a single gather over the augmented node
    # order, identical to the prepass's ``own = body + 1`` convention.
    own_length = body[node] + 1
    id_total = ids[node]
    value_total = vals[node]

    return BatchRowEmission(
        row_offsets=row_offsets,
        node=node,
        edge_type=edge_type,
        own_length=own_length,
        id_total=id_total,
        value_total=value_total,
    )
