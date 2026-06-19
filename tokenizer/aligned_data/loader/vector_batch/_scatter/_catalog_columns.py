"""Per-emitted-node CATALOG columns shared by the dense adapter + remap.

Single concern (the design-first sentence): *derive, in emitted-node
order, the per-node CATALOG columns the dense path reads off the columnar
``sections.bin`` catalog (the owning section index, the section root FID,
the encounter :class:`Category`, and the per-node COUNTER counts) plus the
per-section call_target table — the same fields
:mod:`._dense_adapter`'s per-node loop sources today.*

Why this module exists: BOTH the dense adapter (which buries these into a
``Stage2Batch`` tree) and the remap-input builder (:mod:`._remap_inputs`,
which flattens them into the Rust kernel's int arrays) need the IDENTICAL
per-node catalog columns. Computing them ONCE here and handing the same
:class:`CatalogColumns` to both keeps a single source — no second batched
``category_counts_from_runlen`` reduction, no second ``encounter_category``
table, no duplicated catalog-walk logic.

Module boundary: owned at the scatter/expand boundary (it consumes the
geometry emission + :class:`ColumnarSections`). The only things crossing
the boundary are flat numpy columns + the per-section call_target lists;
no consumer sees a ``ColumnarSections`` internal nor re-walks the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CALL_TARGET_TYPE_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.category_counts import (
    category_counts_from_runlen_batched,
)
from tokenizer.aligned_data.matched_sections_bin import CallTarget
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.tokens import Category

from .._types import BatchGeometry
from ._expand import ExpandedBatch


__all__ = ["CatalogColumns", "build_catalog_columns"]


@dataclass(frozen=True)
class CatalogColumns:
    """The per-emitted-node catalog columns + the per-section CT cache.

    Built once in emitted-node order (== the canonical stage-3 DFS
    call_target order, the vector path's BFS emission). Consumed by both
    :mod:`._dense_adapter` (tree build) and :mod:`._remap_inputs` (kernel
    flat-int build).

    Per-node columns (length ``n_nodes``)
    -------------------------------------
    section_of_node:
        ``int64[n_nodes]`` -- the catalog section index owning node ``e``
        (the CSR bucket of ``cols.var_offsets`` the node falls into).
    encounter_category:
        ``list[Category]`` of length ``n_nodes`` -- node ``e``'s encounter
        :class:`Category` (its edge ``CallTargetType`` collapsed through
        :data:`_CALL_TARGET_TYPE_TO_CATEGORY`).
    counter_count_categories / counter_count_columns:
        The per-node COUNTER-Category count columns from one batched
        ``category_counts_from_runlen`` reduction. ``counter_count_columns``
        is a tuple of ``int64[n_nodes]`` arrays parallel to
        ``counter_count_categories`` (the dict key order); column ``k``'s
        entry ``e`` is node ``e``'s count for that Category.

    Per-section data
    ----------------
    section_root_fid:
        ``list[int]`` of length ``n_sections`` -- section ``s``'s root
        ``function_name_ptr`` as a Python int (``cols.function_name_ptr``).
    call_targets_section:
        Returns a section's parsed ``call_targets`` table, cached by
        section index. Node-invariant per section, so a section's nodes
        share ONE frozen read-only list (the adapter's identity-sharing
        contract: the downstream consumers only read it).
    """

    section_of_node: np.ndarray
    encounter_category: List[Category]
    counter_count_categories: tuple[Category, ...]
    counter_count_columns: tuple[np.ndarray, ...]
    section_root_fid: List[int]

    #: Source handles for the lazy per-section ``call_targets`` parse + the
    #: cache it populates. Not part of the per-node column surface.
    _cols: ColumnarSections
    _ct_offsets: np.ndarray
    _ct_section_cache: dict[int, List[CallTarget]] = field(
        default_factory=dict
    )

    def call_targets_section(self, sec: int) -> List[CallTarget]:
        """The (cached) parsed ``call_targets`` table for section ``sec``."""
        ct_section = self._ct_section_cache.get(sec)
        if ct_section is None:
            ct_section = _call_targets_section(
                self._cols, self._ct_offsets, sec
            )
            self._ct_section_cache[sec] = ct_section
        return ct_section


def build_catalog_columns(
    geometry: BatchGeometry,
    expanded: ExpandedBatch,
    *,
    cols: ColumnarSections,
) -> CatalogColumns:
    """Derive the per-emitted-node catalog columns from the catalog.

    Parameters
    ----------
    geometry / expanded / cols:
        See :func:`._dense.build_dense_sidecars`. The emission ``edge_type``
        + ``node`` axes and the columnar catalog are the only sources read.

    Returns
    -------
    CatalogColumns
        The per-node catalog columns + the per-section CT cache, in
        emitted-node order.
    """
    edge_type = np.asarray(geometry.emission.edge_type, dtype=np.uint8)

    section_of_node = _section_of_node(geometry, cols)
    ct_offsets = np.asarray(cols.ct_offsets, dtype=np.int64)

    # Per-node COUNTER counts for EVERY emitted node in one batched
    # reduction (over the flat raw body the per-node ``states`` views
    # slice). ``per_node_counts[category][e]`` is node ``e``'s count.
    per_node_counts = category_counts_from_runlen_batched(
        expanded.raw_flat,
        expanded.runlen_number_flat,
        expanded.raw_record_offsets,
    )
    count_cats = tuple(per_node_counts)
    count_cols = tuple(per_node_counts[cat] for cat in count_cats)

    # Per-node ``encounter_category`` as one precomputed column (a length-3
    # lookup indexed by the already-int ``edge_type``).
    encounter_category = _encounter_category_per_node(edge_type)

    # Per-section root FID as a Python-int column (one ``int()`` per section).
    section_root_fid = np.asarray(cols.function_name_ptr).tolist()

    return CatalogColumns(
        section_of_node=section_of_node,
        encounter_category=encounter_category,
        counter_count_categories=count_cats,
        counter_count_columns=count_cols,
        section_root_fid=section_root_fid,
        _cols=cols,
        _ct_offsets=ct_offsets,
    )


def _encounter_category_per_node(edge_type: np.ndarray) -> List[Category]:
    """Per-node ``encounter_category`` resolved via a length-3 lookup.

    Each node's ``edge_type`` (the int value of its :class:`CallTargetType`)
    maps to a FUNCTION :class:`Category` through
    ``_CALL_TARGET_TYPE_TO_CATEGORY``. Resolving that 3-entry table once and
    indexing it by the (already-int) ``edge_type`` reproduces the per-node
    ``_CALL_TARGET_TYPE_TO_CATEGORY[CallTargetType(int(edge_type[e]))]``
    byte-for-byte, without an enum construction + dict lookup per node.
    """
    table = [
        _CALL_TARGET_TYPE_TO_CATEGORY[CallTargetType(t)]
        for t in range(len(CallTargetType))
    ]
    return [table[t] for t in edge_type.tolist()]


def _section_of_node(
    geometry: BatchGeometry, cols: ColumnarSections
) -> np.ndarray:
    """``int64[n_emitted]`` -- the catalog section index owning each node.

    The emission ``node`` axis is ``var_offsets``-major; the owning
    section is the CSR bucket of ``cols.var_offsets`` the node falls into.
    """
    nodes = np.asarray(geometry.emission.node, dtype=np.int64)
    var_offsets = np.asarray(cols.var_offsets, dtype=np.int64)
    return np.searchsorted(var_offsets, nodes, side="right") - 1


def _call_targets_section(
    cols: ColumnarSections, ct_offsets: np.ndarray, sec: int
) -> List[CallTarget]:
    """The section's parsed ``call_targets`` table from the catalog.

    Rebuilds the ``list[CallTarget]`` the decode walker passes from the
    columnar ``ct_*`` slices -- the encoder's LOCAL->PLT->EXTERN grouping
    (and within-group order) is preserved by the catalog, so the per-
    Category caller-local id density the dedup walk relies on holds.
    """
    lo = int(ct_offsets[sec])
    hi = int(ct_offsets[sec + 1])
    fids = cols.ct_function_name_ptr[lo:hi]
    secptrs = cols.ct_function_section_ptr[lo:hi]
    types = cols.ct_type[lo:hi]
    matched = cols.ct_is_matched[lo:hi]
    return [
        CallTarget(
            function_name_ptr=int(fids[i]),
            function_section_ptr=int(secptrs[i]),
            type=CallTargetType(int(types[i])),
            is_matched=bool(matched[i]),
        )
        for i in range(hi - lo)
    ]
