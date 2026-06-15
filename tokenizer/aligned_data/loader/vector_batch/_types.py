"""Typed result of the body-free geometry PREPASS (plan C1).

Single concern: DEFINE the typed `[B, L]` batch-geometry contract the
later fused scatter (TC2) and entry (TC3) consume -- the prepass owns the
construction; this module owns the shape + provenance documentation.

Design contract (carried verbatim from the accepted plan):

* The prepass is BODY-FREE. Token COLUMN positions come from the stored
  realized BODY length (RLG3); dense id / value RESERVATIONS come from
  the stored RLG3 ``id_count`` / ``value_count``. NO ``_data.bin`` byte
  is read. The straddler's PARTIAL dense count is NOT known here -- it is
  a byproduct of TC2's body read ("adjust during decoding").

* POSITIONS ARE ASSIGNED IN OLD BFS EMISSION ORDER. Byte-identity vs the
  post-T0 ``batch_decode`` (backfill OFF) requires each emitted function
  land at the SAME column as the old callee-walk path; the emitted-node
  ORDER in :attr:`BatchRowEmission.node` reproduces that BFS order
  (root, then per level the included callees in parent-then-ascending-
  call_target-slot order), and the column layout is a strict prefix-sum
  of own-lengths over that order.

* The dense offset+len contract ALLOWS gaps, but gaps are a BACKFILL-ON
  phenomenon. With backfill OFF the dense reservation IS the tight
  actual extent (TC2 tightens nothing); the reservations here are the
  alloc upper-bound and the cumulative offsets are where TC2 places +
  then tightens.

Every array is a plain ``np.ndarray`` (no bytes-as-data, no positional
tuples callers index blindly); each field documents its provenance +
units below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "BatchRowEmission",
    "BatchTokenLayout",
    "DenseReservation",
    "BatchGeometry",
]


@dataclass(frozen=True)
class BatchRowEmission:
    """The ordered emitted-function list of EVERY batch row, flattened.

    One entry per (row, emitted function) pair, laid out row-major in
    BFS emission order (``row_offsets`` is the CSR jump table tying each
    flat entry to its row). The straddler -- the single function whose
    own-length crosses the per-row seq_len ``L`` -- and the rows that are
    fully truncated at ``L`` are still PRESENT here up to and including
    the straddler; :class:`BatchTokenLayout` carries the cut.

    Provenance: produced by the shared
    :class:`~tokenizer.aligned_data.splice_inclusion.OnceOnlyInclusion`
    BFS driven over the sampled variant subset (the same decider the
    post-T0 ``batch_decode`` feeds the sampled subset, so the included
    set + order converge by construction).
    """

    row_offsets: np.ndarray
    """``int64[B + 1]`` -- CSR offsets: row ``r``'s emitted functions are
    flat entries ``[row_offsets[r] : row_offsets[r + 1]]``. ``[0] == 0``,
    ``[-1] == n_emitted``."""

    node: np.ndarray
    """``int64[n_emitted]`` -- the flat (section, variant) catalog NODE
    index of each emitted function, IN BFS emission order. The first
    entry of every row is the root node; columns are assigned by the
    own-length prefix-sum over THIS order. Units: catalog node index
    (``var_offsets``-major, the RLG3 axis index)."""

    own_length: np.ndarray
    """``int64[n_emitted]`` -- the emitted function's own column span =
    ``1 self-token + realized body_len`` (RLG3 ``body_lengths[node]``).
    The variant-token row PREFIX is NOT counted here -- it is a per-ROW
    quantity (:attr:`BatchTokenLayout.prefix_len`). Units: token
    columns."""

    id_total: np.ndarray
    """``int64[n_emitted]`` -- the function's TOTAL stored identity-
    carrier count (RLG3 ``id_counts[node]``). The dense-reservation
    upper bound; TC2 finalises the surviving / straddler-partial count
    from the body. Units: dense identity slots."""

    value_total: np.ndarray
    """``int64[n_emitted]`` -- the function's TOTAL stored numeric-chunk
    count (RLG3 ``value_counts[node]``). The dense-reservation upper
    bound (see :attr:`id_total`). Units: dense numeric slots."""


@dataclass(frozen=True)
class BatchTokenLayout:
    """The per-row ``[B, L]`` token COLUMN layout (one entry per row).

    The straddler is the SINGLE emitted function per row whose own-length
    prefix-sum crosses ``L``; rows whose total emitted length is ``<= L``
    have NO straddler (full row).

    Provenance: an own-length prefix-sum over :attr:`BatchRowEmission`
    per row, then a ``searchsorted`` of ``L - prefix_len`` against that
    prefix-sum. Body-free.
    """

    seq_len: int
    """``L`` -- the per-row token-column budget every row is truncated
    to. Units: token columns."""

    prefix_len: np.ndarray
    """``int64[B]`` -- the variant-token PREFIX width prepended ahead of
    the root body in each row (RLG3-independent; read body-free from the
    leading u16 ``n_tokens`` header of the row's sampled variant record
    in ``_variants.bin``, minus the dropped size token). Columns
    ``0 : prefix_len[r]`` hold the prefix; emitted bodies start at
    ``prefix_len[r]``. Units: token columns."""

    straddler_local_idx: np.ndarray
    """``int64[B]`` -- the row-LOCAL index (into the row's
    :attr:`BatchRowEmission` slice) of the straddler function, or ``-1``
    when the row is full (no function crosses ``L``). Units: emitted-
    function ordinal within the row."""

    partial_cut_length: np.ndarray
    """``int64[B]`` -- how many of the straddler's own-length columns FIT
    before ``L`` (``0 <= partial_cut_length < own_length[straddler]``).
    Defined as ``0`` for full rows (no straddler). This is the straddler
    function's body cut at the token level; its dense PARTIAL count is
    NOT derivable here (needs the body) and is TC2's concern. Units:
    token columns."""

    total_length: np.ndarray
    """``int64[B]`` -- the row's total emitted token width BEFORE the
    ``L`` truncation = ``prefix_len + sum(own_length over the row)``.
    Rows with ``total_length <= L`` are full. Units: token columns."""


@dataclass(frozen=True)
class DenseReservation:
    """Per-row dense identity / numeric RESERVATION totals + offsets.

    UPPER-BOUND sizing only: each row reserves the SUM over its emitted
    functions of the stored TOTAL id / value counts (NOT the surviving /
    straddler-partial counts -- those need the body and are finalised by
    TC2's scatter). The cumulative offsets are where TC2 places each
    row's dense block and then tightens.

    Provenance: a segmented sum of :attr:`BatchRowEmission.id_total` /
    ``value_total`` per row. Body-free.
    """

    id_reserved: np.ndarray
    """``int64[B]`` -- reserved identity slots per row (segmented sum of
    ``id_total``). Upper bound. Units: dense identity slots."""

    id_offsets: np.ndarray
    """``int64[B + 1]`` -- exclusive prefix sum of :attr:`id_reserved`;
    row ``r``'s reserved identity block is ``[id_offsets[r] :
    id_offsets[r + 1]]`` in the batch-flat dense identity array TC2
    allocates. Units: dense identity slots."""

    value_reserved: np.ndarray
    """``int64[B]`` -- reserved numeric slots per row (segmented sum of
    ``value_total``). Upper bound. Units: dense numeric slots."""

    value_offsets: np.ndarray
    """``int64[B + 1]`` -- exclusive prefix sum of
    :attr:`value_reserved`; row ``r``'s reserved numeric block is
    ``[value_offsets[r] : value_offsets[r + 1]]``. Units: dense numeric
    slots."""


@dataclass(frozen=True)
class BatchGeometry:
    """The complete typed PREPASS result for one ``[B, L]`` batch.

    Consumed by TC2 (the fused scatter) and TC3 (the entry). It carries
    EVERYTHING the body-free prepass determined: the per-row emitted-
    function list (with column + dense provenance), the token-column
    layout (prefix + straddler cut), the dense reservations, and the
    remembered-excluded backfill pool.

    The ``node`` field of :attr:`emission` is the catalog node index;
    TC2 maps it to a ``_data.bin`` record locator via the catalog's
    ``var_data_offset_shifted`` (the ``record_offset >> 4`` field) +
    ``_index.bin`` -- the prepass deliberately stops at the node index so
    the locator concern stays with the scatter that reads the body.
    """

    n_rows: int
    """``B`` -- the batch row count (one per sampled (section, variant)
    pair)."""

    emission: BatchRowEmission
    """Per-row ordered emitted-function list (BFS order)."""

    layout: BatchTokenLayout
    """Per-row ``[B, L]`` token-column layout."""

    reservation: DenseReservation
    """Per-row dense id / value reservation totals + offsets."""

    excluded_pool: np.ndarray
    """``int64[n_excluded]`` -- the REMEMBERED extra-excluded callee
    NODES per row, flattened row-major (CSR via
    :attr:`excluded_pool_offsets`). These are callees the SAMPLED-subset
    BFS pruned that the FULL-set index would have included -- the
    backfill candidates TD draws from when backfill is ON. With backfill
    OFF this pool is carried but unused. Units: catalog node index."""

    excluded_pool_offsets: np.ndarray
    """``int64[B + 1]`` -- CSR offsets: row ``r``'s remembered-excluded
    pool is ``excluded_pool[excluded_pool_offsets[r] :
    excluded_pool_offsets[r + 1]]``."""
