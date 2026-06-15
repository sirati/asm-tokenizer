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

from typing import List

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.realized_lengths._geometry_reader import (
    RealizedGeometryReader,
)

from ._inclusion import RowInclusion, compute_row_inclusions
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

    inclusions: List[RowInclusion] = compute_row_inclusions(
        cols,
        section_offsets,
        root_sections=root_sections,
        root_sampled_variants=root_sampled_variants,
        root_groups=root_groups,
        max_depth=max_depth,
        need_excluded_pool=need_excluded_pool,
    )
    n_rows = len(inclusions)

    emission, root_nodes = _flatten_emission(
        inclusions, body_axis, id_axis, value_axis
    )

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
        excluded_pool, excluded_offsets, excluded_edge_type = _flatten_excluded(
            inclusions
        )
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
    inclusions: List[RowInclusion],
    body_axis: np.ndarray,
    id_axis: np.ndarray,
    value_axis: np.ndarray,
):
    """Concatenate per-row emitted nodes into the flat CSR emission.

    Returns ``(BatchRowEmission, root_nodes)`` where ``root_nodes`` is
    the per-row first emitted node (the root carrying the variant
    prefix). ``own_length = 1 + body_len`` composes the self-token at the
    flat gather site -- the SAME ``own = body + 1`` convention the
    length twin (``...compute_node_lengths``) uses.
    """
    per_row = [inc.emitted_nodes for inc in inclusions]
    per_row_types = [inc.emitted_edge_types for inc in inclusions]
    lengths = np.array([e.size for e in per_row], dtype=np.int64)
    row_offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=row_offsets[1:])

    if per_row:
        node = np.concatenate(per_row).astype(np.int64)
        edge_type = np.concatenate(per_row_types).astype(np.uint8)
    else:
        node = np.zeros(0, dtype=np.int64)
        edge_type = np.zeros(0, dtype=np.uint8)

    # The variant-token row PREFIX is a per-row quantity (kept OUT of
    # own_length); own_length is the self-token + the realized body span.
    own_length = body_axis[node].astype(np.int64) + 1
    id_total = id_axis[node].astype(np.int64)
    value_total = value_axis[node].astype(np.int64)

    # Each row's first emitted node is the root (always present;
    # begin_root seeds it as emitted_nodes[0]).
    root_nodes = np.array(
        [e[0] for e in per_row] if per_row else [], dtype=np.int64
    )

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


def _flatten_excluded(inclusions: List[RowInclusion]):
    """Concatenate per-row remembered-excluded pools into flat CSR.

    Returns ``(pool, offsets, edge_type)`` -- the flat node pool, its CSR
    offsets, and the PARALLEL per-pool-node edge ct_type (concatenated in
    the SAME row-major order as ``pool``), so backfill can carry each
    re-inlined node's true parent-slot edge type.
    """
    per_row = [inc.excluded_nodes for inc in inclusions]
    per_row_types = [inc.excluded_edge_types for inc in inclusions]
    lengths = np.array([e.size for e in per_row], dtype=np.int64)
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if per_row:
        pool = np.concatenate(per_row).astype(np.int64)
        edge_type = np.concatenate(per_row_types).astype(np.uint8)
    else:
        pool = np.zeros(0, dtype=np.int64)
        edge_type = np.zeros(0, dtype=np.uint8)
    return pool, offsets, edge_type
