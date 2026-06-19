"""Geometry-first body-free batch PREPASS (plan C1) -- the orchestrator.

Single concern: compose the body-free sub-passes into the typed
:class:`BatchGeometry` the fused scatter (TC2) + entry (TC3) consume.
Given a sampled batch (root section + sampled variant per row) and the
body-free sidecars, it:

1. drives the SHARED once-only inclusion BFS (:mod:`._inclusion`) ->
   per-row ordered emitted nodes (BFS emission order) + remembered-
   excluded backfill pool;
2. gathers each emitted node's stored geometry from the RLG3 reader
   (:mod:`...realized_lengths`) -> own_length (``1 + body_len``),
   id_total, value_total -- the column + dense PROVENANCE;
3. reads the per-row variant-prefix width body-free from
   ``_variants.bin`` (:mod:`._prefix`);
4. computes the per-row token-column layout + the SINGLE straddler cut
   (:mod:`._layout`), vectorized across the batch;
5. computes the per-row dense id / value RESERVATIONS + offsets
   (:mod:`._reserve`).

BODY-FREE PROOF: the only file handles read are ``sections.bin`` (the
columnar catalog), the RLG3 geometry sidecars (stored lengths/counts),
and ``_variants.bin`` (the prefix size header). ``_data.bin`` is NEVER
opened or addressed here -- the prepass takes no ``_data.bin`` handle at
all (a caller may pass a sentinel that raises on any access; this module
never reaches for one).

REUSE (no re-implementation): inclusion semantics + adjacency come from
:mod:`...splice_inclusion` + :mod:`...sorted_index._graph_lengths`; the
stored geometry triple from the :class:`RealizedGeometryReader`; the
sampler's output (section pointer + sampled variants) is the input
contract -- TC3 owns the sampler call, this prepass owns the geometry.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)
from tokenizer.aligned_data.realized_lengths._geometry_reader import (
    RealizedGeometryReader,
)

from ._inclusion import InclusionCSR, compute_row_inclusions
from ._layout import compute_token_layout
from ._prefix import variant_prefix_lengths
from ._reserve import compute_dense_reservation
from ._types import (
    BatchGeometry,
    BatchRowEmission,
    DenseReservation,
)


__all__ = ["compute_batch_geometry"]


def compute_batch_geometry(
    *,
    cols: ColumnarSections,
    section_offsets: np.ndarray,
    geometry: RealizedGeometryReader,
    variants_u8: np.ndarray,
    root_sections: np.ndarray,
    root_sampled_variants: np.ndarray,
    root_groups: np.ndarray,
    seq_len: int,
    max_depth: int,
    need_excluded_pool: bool = True,
    adjacency: LiveNodeAdjacency | None = None,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
) -> BatchGeometry:
    """The body-free ``[B, L]`` geometry prepass.

    Parameters
    ----------
    cols:
        The columnar ``sections.bin`` catalog (the splice graph + the
        ``var_ref_offset`` vkeys).
    section_offsets:
        ``int[n_sections]`` byte offsets parallel to ``cols`` (the
        adjacency offset->idx key).
    geometry:
        The matched-arm :class:`RealizedGeometryReader` -- its
        ``body_lengths`` / ``id_counts`` / ``value_counts`` axes are
        section-major in ``cols.var_offsets`` order (one entry per
        catalog NODE), the stored geometry the prepass sizes from.
    variants_u8:
        ``_variants.bin`` as a 1-D uint8 array (the prefix size header
        source). NOT ``_data.bin``.
    root_sections / root_sampled_variants:
        ``int[B]`` parallel -- the sampled (root section, root variant)
        of each batch row.
    root_groups:
        ``int[B]`` parallel -- the per-row DECIDER-ROOT group id (rows
        sharing a group are one root's co-sampled variants; see
        :func:`._inclusion.compute_row_inclusions`).
    seq_len:
        ``L`` -- the per-row token-column budget.
    max_depth:
        Splice BFS depth cap (``>= 0``).
    need_excluded_pool:
        Whether the remembered-excluded backfill pool + dense reservation
        (both backfill-only outputs) must be computed. ``False`` (backfill
        off) skips the FULL-variant-set inclusion BFS + the pool/reservation
        flatten -- the dominant prepass cost -- and emits empty,
        correctly-shaped placeholders. The emitted-node geometry is
        unchanged, so the scatter result is byte-identical.
    adjacency:
        A pre-built :class:`LiveNodeAdjacency` over the SAME ``cols`` /
        ``section_offsets``, threaded to :func:`compute_row_inclusions` so
        the cols-invariant adjacency (offset->idx map + per-binary MISSING
        inventory scan) is built once per binary, not per batch. ``None``
        falls back to a fresh per-call build.

    Returns
    -------
    BatchGeometry
        The typed prepass result (see :class:`BatchGeometry`).
    """
    body_axis = geometry.body_lengths
    id_axis = geometry.id_counts
    value_axis = geometry.value_counts
    n_nodes = int(body_axis.size)
    if not (id_axis.size == n_nodes and value_axis.size == n_nodes):
        raise ValueError(
            "RLG3 geometry axes are mis-sized against each other: "
            f"body={body_axis.size} id={id_axis.size} value={value_axis.size}"
        )

    inclusions: InclusionCSR = compute_row_inclusions(
        cols,
        section_offsets,
        root_sections=root_sections,
        root_sampled_variants=root_sampled_variants,
        root_groups=root_groups,
        max_depth=max_depth,
        need_excluded_pool=need_excluded_pool,
        adjacency=adjacency,
        unmatched_inline=unmatched_inline,
        unmatched_inline_depth=unmatched_inline_depth,
    )
    n_rows = int(inclusions.emitted_offsets.size - 1)

    emission, root_nodes = _flatten_emission(
        inclusions, body_axis, id_axis, value_axis
    )

    # Bound the columnar parse to the sections this batch actually emits:
    # every emitted node (roots + included callees, leaves included) has its
    # owning section materialised before the variant-prefix / scatter / dense
    # passes read its heavy ``cols`` columns. The BFS adjacency already filled
    # the parent/frontier sections; this closes the gap for leaf callees that
    # were included at max-depth but never expanded as parents. A no-op on the
    # eager catalog (everything is already resident).
    cols.ensure_sections(cols.sec_of_var[emission.node])

    prefix_len = variant_prefix_lengths(variants_u8, cols, nodes=root_nodes)

    layout = compute_token_layout(
        own_length=emission.own_length,
        row_offsets=emission.row_offsets,
        prefix_len=prefix_len,
        seq_len=seq_len,
    )
    # The dense reservation + the remembered-excluded pool are BOTH
    # backfill-only outputs; with backfill off they are unread, so skip
    # them and emit empty, correctly-shaped placeholders.
    if need_excluded_pool:
        reservation = compute_dense_reservation(
            id_total=emission.id_total,
            value_total=emission.value_total,
            row_offsets=emission.row_offsets,
        )
        excluded_pool = inclusions.pool_nodes
        excluded_offsets = inclusions.pool_offsets
        excluded_edge_type = inclusions.pool_types
    else:
        reservation = _empty_reservation(n_rows)
        excluded_pool = np.zeros(0, dtype=np.int64)
        excluded_offsets = np.zeros(n_rows + 1, dtype=np.int64)
        excluded_edge_type = np.zeros(0, dtype=np.uint8)

    return BatchGeometry(
        n_rows=n_rows,
        emission=emission,
        layout=layout,
        reservation=reservation,
        excluded_pool=excluded_pool,
        excluded_pool_offsets=excluded_offsets,
        excluded_pool_edge_type=excluded_edge_type,
    )


def _flatten_emission(
    inclusions: InclusionCSR,
    body_axis: np.ndarray,
    id_axis: np.ndarray,
    value_axis: np.ndarray,
):
    """Read the flat emission CSR into the typed ``BatchRowEmission``.

    The fused inclusion kernel ALREADY emits the row-major flat CSR (its
    ``emitted_offsets`` are exactly the per-row ``row_offsets``, and
    ``emitted_nodes`` / ``emitted_types`` are the flat node / edge value
    arrays), so this is a pure field read -- no per-row Python object, no
    re-concatenate, no cumsum. ``own_length = 1 + body_len`` composes the
    self-token at the flat gather site -- the SAME ``own = body + 1``
    convention the length twin (``...compute_node_lengths``) uses.

    Returns ``(BatchRowEmission, root_nodes)`` where ``root_nodes`` is the
    per-row first emitted node (the root carrying the variant prefix) --
    gathered at each row's CSR start offset (begin_root seeds the root at
    ``emitted_nodes[emitted_offsets[r]]``, so every row is non-empty).
    """
    row_offsets = inclusions.emitted_offsets
    node = inclusions.emitted_nodes
    edge_type = inclusions.emitted_types

    # The variant-token row PREFIX is a per-row quantity (kept OUT of
    # own_length); own_length is the self-token + the realized body span.
    own_length = body_axis[node].astype(np.int64) + 1
    id_total = id_axis[node].astype(np.int64)
    value_total = value_axis[node].astype(np.int64)

    # Each row's first emitted node is the root (always present; begin_root
    # seeds it at the row's CSR start offset).
    root_nodes = node[row_offsets[:-1]]

    emission = BatchRowEmission(
        row_offsets=row_offsets,
        node=node,
        edge_type=edge_type,
        own_length=own_length,
        id_total=id_total,
        value_total=value_total,
    )
    return emission, root_nodes


def _empty_reservation(n_rows: int) -> DenseReservation:
    """An all-zero, correctly-shaped reservation (backfill-off placeholder).

    The reservation feeds ONLY backfill; with backfill off it is unread,
    so this carries the ``[B]`` / ``[B + 1]`` shapes without paying the
    segmented sums.
    """
    zeros_rows = np.zeros(n_rows, dtype=np.int64)
    zeros_off = np.zeros(n_rows + 1, dtype=np.int64)
    return DenseReservation(
        id_reserved=zeros_rows,
        id_offsets=zeros_off,
        value_reserved=zeros_rows.copy(),
        value_offsets=zeros_off.copy(),
    )
