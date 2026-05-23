"""Stage 4 step 1: per-row identity remap walk.

Single concern: rewrite ``stage3_batch.identities_flat_caller_local`` IN
PLACE from per-function caller-local ids to **variant-global counter
ids per Category**, applying:

* ALG-3 (FUNCTION categories — ``LOCAL_FUNC`` / ``PLT_FUNC`` /
  ``EXT_FUNC``): hole-free dedup keyed on ``function_name_ptr``. One
  ``HashMapU32U16`` per Category, reused across rows via ``clean()``
  per the dedup-walk hot-path discipline (Rust hashmap allocations
  would dominate otherwise).
* ALG-4 (COUNTER categories — ``BLOCK`` / ``STRING_PTR`` /
  ``RO_DATA_PTR`` / ``RW_DATA_PTR`` / ``JUMP_TABLE``): pure offset bump
  per call_target, no dedup lookup.

Prepend slots (the leading ``identity_slice.start`` entry for each
call_target — ALG-9) are written here too, since the self-counter
needed for the prepend is exactly the dedup map's entry for the
call_target's ``function_name_ptr`` in its ``encounter_category``
space, and that value is computed as part of this walk.

This module owns ONLY the per-row remap; the token-tensor assembly +
number-chunk sidecar concatenation + variant-padding policy live in
their own stage-4 modules.

Plan reference: ``batch_decode_plan.md`` ``## Algorithms`` ALG-3 +
ALG-4 + ALG-9 and ``## Stages -- algorithm sketch`` Stage 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from dedup_hashmap import HashMapU32U16

from tokenizer.tokens import Category

from ._constants import (
    COUNTER_CATEGORIES,
    FUNCTION_CATEGORIES,
)
from ._counter_bump import _bump_counter_category
from ._function_remap import _remap_function_category
from ._prepend_slot import _seed_local_func_root, _write_prepend_slot
from ._row_state import _RowState


if TYPE_CHECKING:
    from .._types import Stage3Batch, Stage3Variant


__all__ = ["apply_per_row_remap"]


def apply_per_row_remap(
    stage3_batch: "Stage3Batch",
    *,
    dedup_maps: dict[Category, HashMapU32U16],
    collect_fid_sidecar: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply ALG-3 + ALG-4 + ALG-9 per-row remap IN PLACE.

    Walks every Stage3Variant whose ``stage1.batch_idx is not None``
    (padding rows are skipped — their identity sidecar slice is
    zero-length per the row offsets, so no remap is needed). For each
    real row:

    1. Reset the 3 FUNCTION-category hashmaps via ``clean()``.
    2. Seed LOCAL_FUNC with ``{root.function_name_ptr: 0}``; reset
       per-Category COUNTER offsets to 0.
    3. For each call_target in encounter order whose
       ``surviving_token_count > 0``:
         - Write the prepend slot (ALG-9: self-counter at
           ``identity_slice.start``).
         - Run ALG-3 dedup over every FUNCTION Category present in the
           call_target's ``call_targets_section``; apply remap to the
           in-stream identity slice (skipping the prepend).
         - Run ALG-4 offset bump over every COUNTER Category; apply
           offset to the in-stream identity slice.

    Parameters
    ----------
    stage3_batch
        Owns the batch-shared ``identities_flat_caller_local`` array
        that this function mutates IN PLACE. Returned unchanged in
        the function's first tuple element.
    dedup_maps
        Caller-provided pool: one :class:`HashMapU32U16` per FUNCTION
        Category (LOCAL_FUNC, PLT_FUNC, EXT_FUNC). The caller owns
        these so they can be reused across :func:`batch_decode` calls
        (per the plan's Rust-allocation hot-path discipline).
        ``clean()`` is called per row, NOT per batch — the caller can
        pre-size each map with ``HashMapU32U16(capacity=expected_K)``.
    collect_fid_sidecar
        When True, returns ``fid_sidecar`` + ``fid_row_offsets``
        arrays per plan D5. Layout per row is the (LOCAL_FUNC,
        PLT_FUNC, EXT_FUNC) counter-id-sorted concatenation; the row
        is fully self-describing via ``fid_row_offsets``.

    Returns
    -------
    tuple
        ``(identities_flat_caller_local, fid_sidecar, fid_row_offsets)``.
        The first element is the same array that lives on
        ``stage3_batch`` (mutated in place); returned for caller
        convenience so the parent can pipe it into
        :class:`BatchDecodeResult` without re-reaching. The latter two
        are ``None`` when ``collect_fid_sidecar=False``.
    """
    # Validate the dedup_maps shape: the caller must provide exactly
    # one map per FUNCTION Category. A missing map is a wiring bug;
    # surfacing it here keeps the per-row code free of the check.
    for cat in FUNCTION_CATEGORIES:
        if cat not in dedup_maps:
            raise AssertionError(
                f"apply_per_row_remap: dedup_maps is missing the "
                f"required FUNCTION-Category map for {cat!r}; provide "
                f"one HashMapU32U16 per Category in {FUNCTION_CATEGORIES}."
            )

    identities_flat = stage3_batch.identities_flat_caller_local
    batch_size = stage3_batch.stage2.stage1.batch_size

    # Per-row fid-sidecar accumulators. Each row's segment is appended
    # to ``fid_chunks`` in batch_idx order; ``fid_row_lengths`` records
    # the per-row length so the final ``fid_row_offsets`` can be built
    # via a single cumsum. Padding rows contribute zero-length segments.
    fid_row_lengths: Optional[np.ndarray] = None
    fid_chunks: Optional[list[np.ndarray]] = None
    if collect_fid_sidecar:
        fid_row_lengths = np.zeros(batch_size, dtype=np.uint32)
        fid_chunks = [
            np.zeros(0, dtype=np.uint32) for _ in range(batch_size)
        ]

    for stage3_section in stage3_batch.sections:
        for stage3_variant in stage3_section.variants:
            batch_idx = stage3_variant.stage2.stage1.batch_idx
            if batch_idx is None:
                # Padding row: identity slice for every call_target has
                # length 0 (stage 2's surviving counts dropped it), so
                # there is nothing to remap. Defensive skip in case a
                # future stage carries non-zero slices for padding rows.
                continue

            row_chunks = _walk_one_row(
                stage3_variant,
                dedup_maps,
                identities_flat,
                collect_fid_sidecar=collect_fid_sidecar,
            )
            if collect_fid_sidecar:
                # Per-row sidecar: concatenate LOCAL_FUNC, PLT_FUNC,
                # EXT_FUNC inverse maps in their dense counter-id
                # order (the order ``fid_inverse[cat]`` was built in,
                # which matches counter-id order because fresh ids
                # are minted densely).
                pieces = [row_chunks[cat] for cat in FUNCTION_CATEGORIES]
                row_sidecar = (
                    np.concatenate(pieces).astype(np.uint32)
                    if any(p.size > 0 for p in pieces)
                    else np.zeros(0, dtype=np.uint32)
                )
                assert fid_chunks is not None  # for type-checker
                assert fid_row_lengths is not None
                fid_chunks[batch_idx] = row_sidecar
                fid_row_lengths[batch_idx] = row_sidecar.size

    if collect_fid_sidecar:
        assert fid_chunks is not None
        assert fid_row_lengths is not None
        fid_sidecar = (
            np.concatenate(fid_chunks).astype(np.uint32)
            if any(c.size > 0 for c in fid_chunks)
            else np.zeros(0, dtype=np.uint32)
        )
        fid_row_offsets = np.empty(batch_size + 1, dtype=np.uint32)
        fid_row_offsets[0] = 0
        np.cumsum(fid_row_lengths, out=fid_row_offsets[1:])
        return identities_flat, fid_sidecar, fid_row_offsets

    return identities_flat, None, None


def _walk_one_row(
    stage3_variant: "Stage3Variant",
    dedup_maps: dict[Category, HashMapU32U16],
    identities_flat: np.ndarray,
    *,
    collect_fid_sidecar: bool,
) -> dict[Category, np.ndarray]:
    """Per-row dedup walk.

    Returns a dict from FUNCTION Category to the per-row inverse-map
    array (``u32[counter_count]``: ``counter_id -> function_name_ptr``,
    in dense counter-id order). When ``collect_fid_sidecar=False``,
    each entry is an empty u32 array — the dict shape is preserved so
    the caller's concatenation path stays uniform.
    """
    # Reset the FUNCTION dedup maps for this row. ``clean()`` retains
    # the hashmap's bucket allocation per the plan's hot-path discipline.
    for cat in FUNCTION_CATEGORIES:
        dedup_maps[cat].clean()

    state = _RowState(collect_fid_sidecar=collect_fid_sidecar)

    # Plan ALG-3 + ALG-9: seed LOCAL_FUNC with the root function's FID
    # at counter id 0. The root is ``call_targets[0]`` per plan D9.
    root_call_target = stage3_variant.call_targets[0]
    root_fn_name_ptr = int(
        root_call_target.stage2.stage1.function_name_ptr
    )
    _seed_local_func_root(
        state, dedup_maps[Category.LOCAL_FUNC], root_fn_name_ptr
    )

    for call_target in stage3_variant.call_targets:
        if call_target.stage2.surviving_token_count == 0:
            # Cut by context_len at this call_target's boundary or
            # later — no surviving tokens, no remap work.
            continue

        # Prepend first: the prepend slot's self-counter lives in the
        # dedup map for ``encounter_category``, which was populated by
        # either the row-level seed (for the root) or the parent
        # call_target's ALG-3 walk (for inlined callees). The
        # call_target's own ALG-3 walk happens after, so the prepend
        # write reads a counter the map already holds before the walk
        # mints any new ids for THIS call_target.
        _write_prepend_slot(dedup_maps, call_target, identities_flat)

        for cat in FUNCTION_CATEGORIES:
            _remap_function_category(
                state,
                dedup_maps[cat],
                cat,
                call_target,
                identities_flat,
            )

        # COUNTER Categories: ALG-4 offset bump. Order is independent
        # of FUNCTION dedup since the two groups address disjoint
        # shifted token ids.
        for cat in COUNTER_CATEGORIES:
            _bump_counter_category(state, cat, call_target, identities_flat)

    # Snapshot the per-Category inverse maps for the fid sidecar.
    if state.fid_inverse is not None:
        return {
            cat: np.asarray(state.fid_inverse[cat], dtype=np.uint32)
            for cat in FUNCTION_CATEGORIES
        }
    return {cat: np.zeros(0, dtype=np.uint32) for cat in FUNCTION_CATEGORIES}
