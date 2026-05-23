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
    K = int(fn_name_ptrs.size)
    if K == 0:
        # No in-stream tokens of this Category can exist when the
        # call_targets table has no entries of this Category (the
        # encoder would not have emitted any caller-local id of this
        # Category).
        return

    # Batched lookup: returns the existing counter id per fn_name_ptr,
    # or ``NOT_FOUND_U16`` for misses.
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

    # The dedup map for ``category`` now holds the counter for every
    # FID in this category's call_targets_section. When we walk an
    # inlined callee in encounter order, the callee's prepend
    # self-counter is recoverable from this same map via
    # ``dedup_map.lookup(callee.function_name_ptr)`` — no separate
    # cache is needed.

    # Apply the remap to this call_target's in-stream identity slice.
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
    target_view[cat_mask] = remap_lookup[selected]
