"""Columnar builder for the per-row remap kernel's flat int arrays.

Single concern (the design-first sentence): *given the vector path's
already-built :class:`DenseColumns` (surviving / expanded / identity
columns) + the shared :class:`CatalogColumns` (root FID, encounter
Category, COUNTER counts, per-section call_target table) + the emission
row CSR, produce the :class:`FlatRemapInputs` the Rust remap kernel
consumes, byte-identical to :func:`...batch_decode._dedup_walk.
_flat_extract.extract_flat_remap_inputs`' object-tree walk.*

Why this is well-posed (NOT a re-walk): the vector dense adapter lays one
synthetic section per non-padding batch row in row order, one variant per
section, and its call_targets ARE the row's emitted nodes in emission
order (:mod:`._dense_adapter`). So the tree-walk's section -> variant
enumeration (all variants referenced, ``batch_idx`` never ``None``) is the
IDENTITY ``e = 0 .. n_nodes - 1`` emitted-node order, and ``node_row`` is
just the emission row CSR bucket. Every per-node / per-call-target /
per-in-stream field the extractor reads off the tree is a column already
in hand here:

* ``node_skip`` / ``node_prepend_pos`` / the in-stream slot columns come
  from :class:`DenseColumns` (the surviving clip + the per-node
  ``surviving_identity_count`` cumsum + the surviving IDENTITY-band ids
  off the ``expanded`` CSR) -- the SAME columns 3b's ``identity_slice``
  cumsum + ``_surviving_in_stream_token_ids`` read.
* ``node_fid`` / ``node_enc_func_slot`` / the per-section ``ct_*`` columns
  / ``counter_counts`` come from :class:`CatalogColumns` (the columnar
  catalog the adapter already reads).

This module re-implements NO remap rule (the ALG-3/4/9 logic lives only in
the Rust kernel) and NO catalog-walk logic (it consumes the shared
columns) -- it only RE-EXPRESSES the same fields as the kernel's flat int
columns in the kernel's emission order, so the byte-identity gate cannot
diverge.

Module boundary: owned at the scatter/expand boundary; the only thing
crossing into the remap is a :class:`FlatRemapInputs` -- the kernel's
existing flat-int contract, identical to the staged path's.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk import (
    FlatRemapInputs,
)
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    COUNTER_CATEGORIES,
    _FUNCTION_CATEGORY_TO_SLOT,
)
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._flat_extract import (
    _COUNTER_SHIFTED_TO_SLOT,
    _CT_TYPE_TO_FUNC_SLOT,
    _FUNC_SHIFTED_TO_SLOT,
)
from tokenizer.aligned_data.loader.batch_decode._dense_columns import (
    DenseColumns,
)

from .._types import BatchGeometry
from ._catalog_columns import CatalogColumns


__all__ = ["build_flat_remap_inputs"]


# IDENTITY block shifted span [8, 16) -- the same band
# :func:`_surviving_in_stream_token_ids` masks.
_IDENTITY_BAND_LO = np.uint16(8)
_IDENTITY_BAND_HI = np.uint16(16)

# ``CallTargetType`` int value -> FUNCTION slot, as a dense int64 LUT the
# per-node CT CSR kernel indexes by ``ct_type``. Built off the same
# ``_CT_TYPE_TO_FUNC_SLOT`` map the object walk uses, so a layout change
# reshapes both. Length == number of ``CallTargetType`` members.
_CT_TYPE_FUNC_SLOT_LUT = np.asarray(
    [_CT_TYPE_TO_FUNC_SLOT[CallTargetType(t)] for t in range(len(CallTargetType))],
    dtype=np.int64,
)


def build_flat_remap_inputs(
    geometry: BatchGeometry,
    dense: DenseColumns,
    catalog: CatalogColumns,
) -> FlatRemapInputs:
    """Build the kernel's flat int arrays from the vector path's columns.

    Parameters
    ----------
    geometry:
        The body-free prepass result; its ``emission.row_offsets`` is the
        per-row node CSR (``node_row`` + ``n_rows`` + ``row_keys`` source).
    dense:
        The vector path's :class:`DenseColumns` (surviving / expanded /
        identity columns) -- the source of ``node_skip``,
        ``node_prepend_pos`` and the per-in-stream slot columns.
    catalog:
        The shared :class:`CatalogColumns` -- the source of ``node_fid``,
        ``node_enc_func_slot``, the per-section ``ct_*`` columns and
        ``counter_counts``.

    Returns
    -------
    FlatRemapInputs
        The flat int arrays + per-row keys, byte-identical to the tree
        walk's, in the kernel's row/node emission order.
    """
    row_offsets = np.asarray(geometry.emission.row_offsets, dtype=np.int64)
    n_rows = int(geometry.n_rows)
    n_nodes = int(dense.n_nodes)

    # ----- per-node columns (length n_nodes, emission order) -----
    # node_row: the emission row CSR bucket of each node. The adapter lays
    # one variant per row in row order, so the tree-walk's per-variant
    # ``row_idx`` IS this bucket.
    per_row_len = np.diff(row_offsets)
    node_row = np.repeat(
        np.arange(n_rows, dtype=np.int64), per_row_len
    )

    surviving = np.asarray(dense.surviving_token_count, dtype=np.int64)
    node_skip = surviving == 0

    # node_prepend_pos == identity_slice.start == the exclusive prefix sum
    # of the per-node surviving identity count (3b's ``_identity_slices``
    # cumsum). Fully-dropped nodes carry a zero-length slice at the running
    # offset, exactly as the tree walk emits.
    surviving_id = np.asarray(
        dense.surviving_identity_count, dtype=np.int64
    )
    node_prepend_pos = np.zeros(n_nodes, dtype=np.int64)
    np.cumsum(surviving_id[:-1], out=node_prepend_pos[1:])

    # node_fid / node_enc_func_slot from the shared catalog columns.
    section_of_node = np.asarray(catalog.section_of_node, dtype=np.int64)
    section_root_fid = catalog.section_root_fid
    node_fid = (
        section_root_fid[section_of_node]
        if n_nodes
        else np.zeros(0, dtype=np.int64)
    )
    enc_slot_table = np.asarray(
        [
            _FUNCTION_CATEGORY_TO_SLOT[cat]
            for cat in catalog.encounter_category
        ],
        dtype=np.int64,
    )
    node_enc_func_slot = enc_slot_table

    # ----- call-target-section CSR (per node = its section's CT table) -----
    ct_off, ct_fid, ct_func_slot = _build_ct_columns(
        catalog, section_of_node, n_nodes
    )

    # ----- in-stream identity slot CSR (per node) -----
    instream_off, instream_func_slot, instream_counter_slot = (
        _build_instream_columns(dense)
    )

    # ----- per-node COUNTER count rows (projected onto COUNTER order) -----
    counter_counts = _build_counter_counts(catalog, n_nodes)

    row_keys = [(r, 0) for r in range(n_rows)]

    return FlatRemapInputs(
        n_rows=n_rows,
        node_row=node_row,
        node_skip=node_skip,
        node_prepend_pos=node_prepend_pos,
        node_fid=node_fid,
        node_enc_func_slot=node_enc_func_slot,
        ct_off=ct_off,
        ct_fid=ct_fid,
        ct_func_slot=ct_func_slot,
        instream_off=instream_off,
        instream_func_slot=instream_func_slot,
        instream_counter_slot=instream_counter_slot,
        counter_counts=counter_counts,
        row_keys=row_keys,
    )


def _build_ct_columns(
    catalog: CatalogColumns,
    section_of_node: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node call_target-section CSR: ``(ct_off, ct_fid, ct_func_slot)``.

    Each node emits its OWNING section's call_target table in section order
    (the tree walk reads ``stage1.call_targets_section`` -- the same
    node-invariant per-section list). ``ct_off`` is the per-node CSR;
    ``ct_fid`` / ``ct_func_slot`` are the concatenated per-entry columns.

    The per-section ``(fid, func_slot)`` slices are gathered per node by the
    GIL-released :meth:`CatalogColumns.node_ct_csr` kernel directly off the
    columnar ``ct_*`` flats -- no ``CallTarget`` object is materialised. The
    fid is ``ct_function_name_ptr`` widened to int64; the slot is
    ``_CT_TYPE_FUNC_SLOT_LUT[ct_type]``, the same pair the object walk
    extracts from each parsed ``CallTarget``.
    """
    del n_nodes  # node count is carried by section_of_node's length.
    return catalog.node_ct_csr(section_of_node, _CT_TYPE_FUNC_SLOT_LUT)


def _build_instream_columns(
    dense: DenseColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node in-stream identity slot CSR over the surviving prefix.

    Reproduces :func:`_surviving_in_stream_token_ids` +
    :func:`_instream_slots` for every node in one segmented pass: gather
    each node's surviving-prefix IDENTITY-band ids, DROP the first per node
    (the prepend slot at ``expanded`` position 0 is itself an identity
    token), and map each remaining id to a FUNCTION slot (COUNTER ``-1``)
    or a COUNTER slot (FUNCTION ``-1``). ``instream_off`` is the per-node
    CSR; the two slot columns are parallel over the flat in-stream axis.
    """
    expanded = np.asarray(dense.expanded).reshape(-1)
    node_offsets = np.asarray(dense.node_offsets, dtype=np.int64)
    surviving = np.asarray(dense.surviving_token_count, dtype=np.int64)
    n_nodes = node_offsets.shape[0] - 1

    if n_nodes <= 0 or expanded.shape[0] == 0:
        return (
            np.zeros(n_nodes + 1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    # Per-position owning node + offset within its node (the same segmented
    # form ``count_surviving_batched`` uses).
    node_len = np.diff(node_offsets)
    pos = np.arange(expanded.shape[0], dtype=np.int64)
    node_id = np.repeat(np.arange(n_nodes, dtype=np.int64), node_len)
    offset_in_node = pos - node_offsets[node_id]

    # An IDENTITY-band position is in-stream iff it survives AND it is NOT
    # the first IDENTITY-band position of its node (the prepend). The
    # surviving clip is ``offset_in_node < surviving[node_id]``; the first
    # IDENTITY position per node is the prepend (always position 0's self
    # token band-wise), so within each node we drop the FIRST surviving
    # IDENTITY position.
    within = offset_in_node < surviving[node_id]
    is_identity = (expanded >= _IDENTITY_BAND_LO) & (
        expanded < _IDENTITY_BAND_HI
    )
    surviving_identity = within & is_identity

    # The selected positions are in node-major ascending order (``pos`` is
    # ascending and ``node_id`` is ``np.repeat`` of ascending nodes). The
    # FIRST selected position of each node is its prepend slot (the
    # identity token at ``expanded`` position 0) -- :func:
    # `_surviving_in_stream_token_ids` drops it (``identity_token_ids[1:]``).
    # A selected position is first-of-node iff its node differs from the
    # previous selected position's node.
    sel_node = node_id[surviving_identity]
    sel_ids = expanded[surviving_identity]
    if sel_node.shape[0]:
        is_first_of_node = np.ones(sel_node.shape[0], dtype=bool)
        is_first_of_node[1:] = sel_node[1:] != sel_node[:-1]
        in_stream_mask = ~is_first_of_node
    else:
        in_stream_mask = np.zeros(0, dtype=bool)

    # The in-stream token ids (post prepend drop), in node-major order.
    instream_ids = sel_ids[in_stream_mask]
    instream_node = sel_node[in_stream_mask]

    # Per-node in-stream count == surviving identity count - 1 for surviving
    # nodes, 0 for dropped -- the CSR offsets.
    per_node_instream = np.bincount(instream_node, minlength=n_nodes).astype(
        np.int64
    )
    instream_off = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(per_node_instream, out=instream_off[1:])

    # Map each id to a FUNCTION slot (COUNTER -1) or COUNTER slot
    # (FUNCTION -1); the two columns are disjoint by the IDENTITY-band
    # partition.
    func_slot = np.full(instream_ids.shape[0], -1, dtype=np.int64)
    counter_slot = np.full(instream_ids.shape[0], -1, dtype=np.int64)
    for shifted, slot in _FUNC_SHIFTED_TO_SLOT.items():
        func_slot[instream_ids == np.uint16(shifted)] = slot
    for shifted, slot in _COUNTER_SHIFTED_TO_SLOT.items():
        counter_slot[instream_ids == np.uint16(shifted)] = slot

    return instream_off, func_slot, counter_slot


def _build_counter_counts(
    catalog: CatalogColumns, n_nodes: int
) -> np.ndarray:
    """Per-node COUNTER count rows projected onto ``COUNTER_CATEGORIES``.

    The tree walk's ``_category_counts_column`` projects each node's
    ``metadata['category_counts']`` onto :data:`COUNTER_CATEGORIES` order
    (missing categories -> 0). The shared catalog carries one count column
    per COUNTER Category; this gathers them in the canonical order into the
    ``int64[n_nodes, n_counter]`` matrix the kernel reads.
    """
    n_counter = len(COUNTER_CATEGORIES)
    if n_nodes == 0:
        return np.zeros((0, n_counter), dtype=np.int64)
    col_by_cat = dict(
        zip(catalog.counter_count_categories, catalog.counter_count_columns)
    )
    columns = [
        np.asarray(
            col_by_cat.get(cat, np.zeros(n_nodes, dtype=np.int64)),
            dtype=np.int64,
        )
        for cat in COUNTER_CATEGORIES
    ]
    return np.stack(columns, axis=1)
