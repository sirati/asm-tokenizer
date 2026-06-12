"""FUNCTION-Category remap (ALG-3).

Single concern: hole-free dedup keyed on ``function_name_ptr``
applied to ONE call_target's in-stream identity slice for ONE
FUNCTION Category. Reuses a caller-provided
:class:`HashMapU32U16` (cleaned per row by :mod:`._apply`).

Plan reference: ``batch_decode_plan.md`` ``## Algorithms`` ALG-3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dedup_hashmap import HashMapU32U16

from tokenizer.tokens import Category

from ._constants import (
    _CALL_TARGET_TYPE_TO_CATEGORY,
    _CATEGORY_TO_SHIFTED_ID,
    FUNCTION_CATEGORIES,
    NOT_FOUND_U16,
)
from ._helpers import _surviving_in_stream_token_ids
from ._row_state import _RowState


if TYPE_CHECKING:
    from tokenizer.aligned_data.matched_sections_bin import CallTarget

    from .._types import Stage3CallTarget


__all__ = ["_remap_function_category"]


# FID sentinel for a non-call-target FUNCTION identity (no
# call_targets row, so no recoverable ``function_name_ptr``). FIDs are
# 1-indexed line numbers into ``<binary>_function_names.txt``; 0 is the
# natural "no name" sentinel, and the inspector's
# ``line_to_name.get(fid, "?")`` renders it as ``"?"``.
_UNKNOWN_FID: int = 0


def _function_name_ptrs_per_category(
    call_targets_section: "list[CallTarget]",
    category: Category,
) -> np.ndarray:
    """Filter ``call_targets_section`` to one FUNCTION Category.

    Returns a ``u32`` ndarray of ``function_name_ptr`` values for rows
    whose ``CallTargetType`` maps to ``category`` (per
    ``_CALL_TARGET_TYPE_TO_CATEGORY``). Caller-local ids 0..K-1 within
    the category are dense in this filtered order (encoder invariant:
    the call_targets table is grouped LOCAL -> PLT -> EXTERN, and the
    per-Category caller-local id is the row's position WITHIN its
    Category's group).

    The plan would alternatively place this filter as a method on
    :class:`Stage1CallTarget` (``call_targets_section_for_category``).
    Keeping it private here avoids touching ``_types.py`` for a helper
    that has only one caller; if a second caller emerges the helper
    should be lifted to a shared module.
    """
    target_type = next(
        (
            ct_type
            for ct_type, cat in _CALL_TARGET_TYPE_TO_CATEGORY.items()
            if cat is category
        ),
        None,
    )
    if target_type is None:
        raise AssertionError(
            f"_function_name_ptrs_per_category called with non-FUNCTION "
            f"category {category!r}; FUNCTION categories are "
            f"{FUNCTION_CATEGORIES}."
        )
    fids = [
        ct.function_name_ptr
        for ct in call_targets_section
        if ct.type is target_type
    ]
    return np.asarray(fids, dtype=np.uint32)


def _remap_function_category(
    state: _RowState,
    dedup_map: HashMapU32U16,
    category: Category,
    call_target: "Stage3CallTarget",
    identities_flat: np.ndarray,
) -> None:
    """ALG-3: hole-free FUNCTION-category remap for one call_target.

    Reads the call_target's ``call_targets_section`` filtered to
    ``category``; uses the dedup map's batched API to remap existing
    entries; mints fresh dense ids (starting at the per-Category
    ``next_fresh_id``) for misses; writes them back to the dedup map
    so subsequent call_targets in the same row see them.

    Then applies the remap to the call_target's in-stream identity
    slice IN PLACE (skipping the prepend slot at ``identity_slice.start``
    — written separately by :func:`_prepend_slot._write_prepend_slot`
    per ALG-9).
    """
    stage1 = call_target.stage2.stage1
    fn_name_ptrs = _function_name_ptrs_per_category(
        stage1.call_targets_section, category
    )

    # Step 1 (UNCONDITIONAL when K > 0): mint counters for this
    # call_target's K call-target FIDs into the dedup map. This is a
    # load-bearing SIDE EFFECT independent of whether this call_target
    # carries in-stream tokens of ``category``: the dedup map for
    # ``category`` must hold the counter for every call-target FID so
    # that (a) a later inlined callee's prepend self-counter is
    # recoverable via ``dedup_map.lookup(callee.function_name_ptr)``
    # (ALG-9, see :func:`_prepend_slot._write_prepend_slot`), and (b)
    # subsequent call_targets in the same row dedup against it (ALG-3).
    # ``K == 0`` (no call targets) makes ``remap_lookup`` empty and the
    # mint block a no-op; the whole id domain is then handled by the
    # extend step below.
    remap_lookup = dedup_map.lookup_ndarray(fn_name_ptrs)
    mask_remapped = remap_lookup != NOT_FOUND_U16

    # Hole-free mint: fresh dense ids from ``next_fresh_id``.
    n_fresh = int((~mask_remapped).sum())
    if n_fresh > 0:
        next_fresh_id = state.next_fresh_id[category]
        fresh_ids = (
            np.arange(n_fresh, dtype=np.uint16) + np.uint16(next_fresh_id)
        )
        # NumPy does not allow direct assignment of a u16 array into a
        # u16 boolean-indexed slice when the source is the result of an
        # addition with a u16 scalar (the broadcast may upcast on some
        # numpy releases). The explicit ``.astype(np.uint16)`` below is
        # defensive against that drift.
        remap_lookup[~mask_remapped] = fresh_ids.astype(np.uint16)
        dedup_map.insert_ndarray(
            fn_name_ptrs[~mask_remapped], fresh_ids.astype(np.uint16)
        )
        state.next_fresh_id[category] = next_fresh_id + n_fresh

        # Track the inverse mapping for the optional fid sidecar.
        if state.fid_inverse is not None:
            fresh_fids = fn_name_ptrs[~mask_remapped]
            state.fid_inverse[category].extend(int(f) for f in fresh_fids)

    # Step 2 (GATED on in-stream tokens): apply the remap to this
    # call_target's in-stream slice. A call_target that emits no
    # identity token of ``category`` has nothing to rewrite — but the
    # dedup-map population in step 1 still had to run above. Crucially
    # the converse does NOT hold either: a Category with ZERO
    # call_targets rows (``K == 0``) can still carry in-stream
    # identities (purely data-referenced addresses, see the id-domain
    # note below), so the gate is on the in-stream tokens, NOT on ``K``.
    in_stream_sl = slice(
        call_target.identity_slice.start + 1,
        call_target.identity_slice.stop,
    )
    in_stream_token_ids = _surviving_in_stream_token_ids(call_target)
    if in_stream_token_ids.size == 0:
        return
    cat_token_id_shifted = np.uint16(_CATEGORY_TO_SHIFTED_ID[category])
    cat_mask = in_stream_token_ids == cat_token_id_shifted
    if not cat_mask.any():
        return

    target_view = identities_flat[in_stream_sl]
    # boolean fancy index returns a copy of caller-local ids; gather
    # the deduped counter ids via remap_lookup and write back through
    # the view to the underlying ``identities_flat`` array.
    selected = target_view[cat_mask]

    # The caller-local id domain is the encoder's per-Category
    # ``get_identity`` counter (TokenResolver), which numbers EVERY
    # distinct referenced address of the Category in stream-encounter
    # order — call targets AND addresses referenced only as data (a
    # function pointer taken with ``lea`` / stored in a table / a
    # name-collided second address). The call_targets table, by
    # contrast, is name-deduped and call-only, so its ``K`` rows are a
    # SUBSET of the id domain: ids ``>= K`` are legitimate identities
    # that simply have no call-target row (mirrors the inspector's
    # ``kind_to_called_idx[counter] -> None`` model — see
    # ``inspector/_render/_batch_decode_backend/_callee_resolver.py``).
    # The ``remap_lookup`` built above only covers ids ``[0, K)``, so
    # extend it to cover the full id domain before the gather.
    remap_lookup = _extend_remap_for_non_call_target_ids(
        state, category, selected, remap_lookup
    )
    target_view[cat_mask] = remap_lookup[selected]


def _extend_remap_for_non_call_target_ids(
    state: _RowState,
    category: Category,
    selected: np.ndarray,
    remap_lookup: np.ndarray,
) -> np.ndarray:
    """Grow ``remap_lookup`` to cover non-call-target caller-local ids.

    ``remap_lookup`` (length ``K``) maps the call-target-backed
    caller-local ids ``[0, K)`` to their FID-deduped counter ids.
    In-stream ids ``>= K`` are identities for addresses that were
    referenced but never call targets, so they have no FID-backed row.
    They still need a dense per-Category counter (the model-facing
    output is a dense counter, not a FID), so each distinct such id in
    this call_target gets a fresh counter minted from the same
    ``next_fresh_id`` space the FID path uses — keeping the per-row
    counter domain hole-free.

    Their FID is unrecoverable at decode time (the call_targets BIN
    table is name-deduped and call-only; the encoder's full
    identity->FID metadata is not serialised), so the optional fid
    sidecar records the ``UNKNOWN`` sentinel (FID 0); the inspector
    renders these as ``"?"`` via ``line_to_name.get(fid, "?")``. This
    is a defined, non-corrupting path — the counter is correct by
    construction; only the human-readable callee name is unknown.

    Returns ``remap_lookup`` unchanged when no id exceeds ``K`` (the
    common case), else a grown copy indexable by every value in
    ``selected``.
    """
    K = int(remap_lookup.size)
    max_id = int(selected.max())
    if max_id < K:
        return remap_lookup

    # The distinct non-call-target ids actually present in the stream,
    # in ascending id order (deterministic counter assignment).
    non_call_target_ids = np.unique(selected[selected >= K])

    next_fresh_id = state.next_fresh_id[category]
    grown = np.full(max_id + 1, NOT_FOUND_U16, dtype=np.uint16)
    grown[:K] = remap_lookup
    fresh_ids = np.arange(
        non_call_target_ids.size, dtype=np.uint16
    ) + np.uint16(next_fresh_id)
    grown[non_call_target_ids] = fresh_ids.astype(np.uint16)
    state.next_fresh_id[category] = next_fresh_id + int(
        non_call_target_ids.size
    )

    # One UNKNOWN-FID (0) sidecar entry per freshly-minted counter,
    # parallel to the FID path's ``fid_inverse`` extension above.
    if state.fid_inverse is not None:
        state.fid_inverse[category].extend(
            [_UNKNOWN_FID] * int(non_call_target_ids.size)
        )
    return grown
