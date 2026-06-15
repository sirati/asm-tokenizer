"""Live per-node splice adjacency read straight off the catalog memmap.

Single concern: given a (section, variant) NODE, return the splice
children it DIRECTLY calls -- resolved on demand from the columnar
catalog arrays (``pce_*`` per-call entries + ``ct_function_section_ptr``
+ the variant tables), with NO precomputed graph structure. The memmap
catalog IS the adjacency (owner directive); nothing here materialises a
``variants x call_targets`` edge product or a per-slot fallback array.

Per direct call (a per-call entry of the node's variant): the callee
SECTION is the ``ct_function_section_ptr`` target (mapped to a section
index via the offset->idx hashmap); the callee VARIANT is the entry's
own ``J`` if usable, else the ascending-sibling fallback -- mirroring
:func:`...loader.decoded._variant_selection.choose_callee_variant` one
for one (EXTERN / unresolved-pointer / unknown-offset gates;
``MISSING_VARIANT_INDEX`` skipped to the sibling scan). The fallback's
sibling-candidate scan reads the callee SLOT's per-variant entries live
-- a per-section, never graph-wide, lookup.

The once-only mask keys on the callee SECTION (function identity); the
returned child NODE (flat variant index) carries the body length the
BFS sums when the pair is included.

The ``MISSING_VARIANT_INDEX`` population is inventoried ONCE per binary
at ERROR (corpus-scale catalogs carry six-figure slot counts; each one
silently drops a splice edge -- a data-quality defect worth surfacing).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from dedup_hashmap import HashMapU32U32

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections


__all__ = ["LiveNodeAdjacency"]


logger = logging.getLogger(__name__)

_U32_MISS = np.uint32(0xFFFFFFFF)


class LiveNodeAdjacency:
    """Per-node direct-call children, resolved live from the catalog.

    Construction builds only the offset->section-idx hashmap (needed to
    interpret any ``function_section_ptr``) and emits the per-binary
    MISSING inventory once. Children are derived per
    :meth:`__call__` -- no precomputed adjacency.
    """

    def __init__(
        self,
        cols: ColumnarSections,
        section_offsets: np.ndarray,
        sec_of_var: np.ndarray,
    ) -> None:
        self._cols = cols
        self._sec_of_var = sec_of_var
        offs = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
        if offs.size and int(offs.max()) >= 2**32:
            raise ValueError(
                "sections.bin offsets exceed the wire format's u32 "
                "function_section_ptr range; the catalog is corrupt"
            )
        self._offsets = offs
        n_sections = int(cols.n_variants.size)
        self._sec_map = HashMapU32U32(capacity=max(8, n_sections * 2))
        self._sec_map.insert_ndarray(
            offs.astype(np.uint32),
            np.arange(n_sections, dtype=np.uint32),
        )
        # Per-section lazy fallback-J table cache: section idx -> dense
        # ``int64[n_call_targets]`` whose entry at ``called_idx`` is the
        # lowest-sibling-variant usable J for that slot (-1 if none).
        # Built on first fallback for a section (a per-section read, the
        # SAME scope the live double-loop touched -- never graph-wide).
        self._fallback_cache: dict = {}
        self._report_inventory()

    # -- public per-node API ----------------------------------------------

    def __call__(self, node: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(child_nodes, child_secs, child_types)`` for ``node``.

        One entry per DIRECT call the node's variant resolves (skipping
        gated-out / unresolvable slots), in ascending slot order.
        ``child_types`` is the parent slot's :class:`CallTargetType`
        (``ct_type``) per resolved child -- the EDGE attribute the decode
        path turns into the inlined-callee self-token category (LOCAL ->
        LOCAL_FUNC, PLT -> PLT_FUNC; EXTERN is gated out upstream). It is
        an ``uint8`` array parallel to ``child_nodes`` / ``child_secs``.
        """
        cols = self._cols
        sec = int(self._sec_of_var[node])
        ct_lo = int(cols.ct_offsets[sec])
        ct_hi = int(cols.ct_offsets[sec + 1])
        p0 = int(cols.pce_offsets[node])
        p1 = int(cols.pce_offsets[node + 1])
        if p1 == p0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy(), np.zeros(0, dtype=np.uint8)

        # ascending-unique directly-called slots + the node's own J each.
        called = cols.pce_called_idx[p0:p1].astype(np.int64)
        own_J = cols.pce_section_variant_index[p0:p1].astype(np.int64)
        order = np.argsort(called, kind="stable")
        called = called[order]
        own_J = own_J[order]
        uniq_mask = np.ones(called.size, dtype=bool)
        uniq_mask[1:] = called[1:] != called[:-1]
        called = called[uniq_mask]
        own_J = own_J[uniq_mask]

        child_nodes = []
        child_secs = []
        child_types = []
        for ci, J in zip(called.tolist(), own_J.tolist()):
            slot = ct_lo + ci
            if slot >= ct_hi:
                continue
            child = self._resolve_slot(sec, ci, slot, J)
            if child < 0:
                continue
            child_nodes.append(child)
            child_secs.append(int(self._sec_of_var[child]))
            child_types.append(int(cols.ct_type[slot]))
        if not child_nodes:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy(), np.zeros(0, dtype=np.uint8)
        return (
            np.asarray(child_nodes, dtype=np.int64),
            np.asarray(child_secs, dtype=np.uint32),
            np.asarray(child_types, dtype=np.uint8),
        )

    # -- per-slot resolution ----------------------------------------------

    def _resolve_slot(
        self, sec: int, called_idx: int, slot: int, own_J: int
    ) -> int:
        """Flat callee node for one direct call, or ``-1`` to drop.

        Gates + the J fallback chain mirror ``choose_callee_variant``:
        EXTERN / unresolved-pointer / unknown-offset -> drop; own J
        usable -> use it; else ascending-sibling scan for the lowest
        usable J; none -> drop.
        """
        cols = self._cols
        if int(cols.ct_type[slot]) == int(CallTargetType.EXTERN):
            return -1
        ptr = int(cols.ct_function_section_ptr[slot])
        if ptr == 0:
            return -1
        hit = self._sec_map.get(np.uint32(ptr))
        if hit is None:
            return -1
        callee_sec = int(hit)
        if _usable(own_J):
            return int(cols.var_offsets[callee_sec]) + own_J
        # Fallback: lowest sibling variant with a usable J for this slot.
        fb_J = self._fallback_J(sec, called_idx)
        if fb_J < 0:
            return -1
        return int(cols.var_offsets[callee_sec]) + fb_J

    def _fallback_J(self, sec: int, called_idx: int) -> int:
        """Lowest sibling-variant J for ``(sec, called_idx)`` that is
        usable (non-sentinel), or ``-1``.

        O(1) lookup into the section's fallback table (built once per
        section, vectorized, on first fallback -- a per-section read on
        demand, never a graph-wide memoised structure). The table
        preserves the live scan's exact tie-break: ascending sibling
        variant, and within a variant the first usable per-call entry in
        on-disk order, decides each slot's fallback J.
        """
        table = self._fallback_table(sec)
        if called_idx < 0 or called_idx >= table.size:
            return -1
        return int(table[called_idx])

    def _fallback_table(self, sec: int) -> np.ndarray:
        """``int64[n_call_targets(sec)]`` fallback-J per slot for ``sec``.

        Entry at slot ``ci`` is the J of the EARLIEST usable per-call
        entry for ``ci`` across the section's variants in on-disk order
        -- variants laid out ascending, each variant's entries in pce
        order. This is exactly the value the live double-loop returned
        (it stopped at the first ``called_idx``-matching usable entry in
        that same order). Slots with no usable entry hold -1. Cached per
        section so the scan happens once.
        """
        cached = self._fallback_cache.get(sec)
        if cached is not None:
            return cached
        cols = self._cols
        n_cts = int(cols.n_call_targets[sec])
        table = np.full(max(0, n_cts), -1, dtype=np.int64)
        if n_cts > 0:
            # The section's per-call entries, contiguous and already in
            # (ascending variant, pce order) -- the live scan order.
            v0 = int(cols.var_offsets[sec])
            v1 = int(cols.var_offsets[sec + 1])
            e0 = int(cols.pce_offsets[v0])
            e1 = int(cols.pce_offsets[v1])
            called = cols.pce_called_idx[e0:e1].astype(np.int64)
            Js = cols.pce_section_variant_index[e0:e1].astype(np.int64)
            usable = Js != int(MISSING_VARIANT_INDEX)
            in_range = (called >= 0) & (called < n_cts)
            keep = usable & in_range
            # Reverse-then-overwrite so the EARLIEST flat-order usable
            # entry per slot wins the tie (a later assignment to the same
            # slot during the reversed pass is an earlier original entry).
            ci_keep = called[keep][::-1]
            j_keep = Js[keep][::-1]
            table[ci_keep] = j_keep
        self._fallback_cache[sec] = table
        return table

    # -- inventory logging -------------------------------------------------

    def _report_inventory(self) -> None:
        """One per-binary MISSING inventory ERROR line."""
        cols = self._cols
        raw_missing = int(
            (cols.pce_section_variant_index == MISSING_VARIANT_INDEX).sum()
        )
        if raw_missing:
            logger.error(
                "sorted_index: %d per-call entries carry "
                "MISSING_VARIANT_INDEX. Each one silently drops a splice "
                "edge -- the callee's variant set does not cover the "
                "caller's vkey.",
                raw_missing,
            )


def _usable(J: int) -> bool:
    """A J addresses a real callee variant iff it is not the sentinel."""
    return J != MISSING_VARIANT_INDEX
