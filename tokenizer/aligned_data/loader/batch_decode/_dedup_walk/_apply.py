"""Stage 4 step 1: per-row identity remap walk — public entry + batch-row loop.

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

Iteration order — variant-keyed walk + row-keyed FID sidecar
------------------------------------------------------------

The walk has TWO concerns with different keys:

1. **Identity-slice writes are variant-keyed.** The
   ``identities_flat_caller_local`` array is sized per-unique-variant
   (one identity slice per :class:`Stage3CallTarget`); multiple batch
   rows referencing the same variant share the SAME slice. The dedup
   walk REWRITES caller-local ids to counter ids in place — applying
   it twice to the same slice would feed already-remapped counters
   back into ``remap_lookup`` and produce out-of-bounds indices. So
   the dedup walk runs ONCE per unique variant (iterates
   ``stage3_batch.sections`` -> ``variants``, skipping variants whose
   ``stage1.batch_idx is None`` — they are not referenced by any
   batch row).

2. **FID sidecar emission is row-keyed.** Each batch row gets its own
   ``counter_id -> function_name_ptr`` slice in the optional
   ``fid_sidecar`` output. Multi-mapped variants (RESAMPLE /
   REDISTRIBUTE) contribute the SAME row sidecar content for every
   batch row that references them — the dedup state is reset per row
   (``clean()`` on each FUNCTION map + fresh :class:`_RowState`), so
   the per-variant counter assignments are deterministic. A
   per-variant FID inverse cache populated in pass 1 is replayed in
   pass 2 (iterates ``batch_idx_to_section_variant`` per row); this
   keeps the sidecar emission O(batch_size + total_unique_variants)
   without re-running the dedup walk per row.

Padding rows (sentinel ``(UINT32_MAX, UINT32_MAX)``) contribute
zero-length sidecar entries in pass 2.

Plan reference: ``batch_decode_plan.md`` ``## Algorithms`` ALG-3 +
ALG-4 + ALG-9 and ``## Stages -- algorithm sketch`` Stage 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from dedup_hashmap import HashMapU32U16

from tokenizer.tokens import Category

from .._row_expand import build_per_row_variant_lookup, concat_per_row
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
) -> tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Apply ALG-3 + ALG-4 + ALG-9 per-row remap IN PLACE.

    Two passes:

    Pass 1 (variant-keyed) — walks every :class:`Stage3Variant` whose
    ``stage1.batch_idx is not None`` (i.e. referenced by at least one
    batch row). For each such variant:

    1. Reset the 3 FUNCTION-category hashmaps via ``clean()``.
    2. Seed LOCAL_FUNC with ``{root.function_name_ptr: 0}``; reset
       per-Category COUNTER offsets to 0.
    3. For each call_target in encounter order whose
       ``surviving_token_count > 0``:
         - Write the prepend slot (ALG-9: self-counter at
           ``identity_slice.start``).
         - Run ALG-3 dedup over every FUNCTION Category present in
           the call_target's ``call_targets_section``; apply remap
           to the in-stream identity slice (skipping the prepend).
         - Run ALG-4 offset bump over every COUNTER Category; apply
           offset to the in-stream identity slice.
    4. Cache the per-variant FID inverse map for pass 2.

    Pass 2 (row-keyed) — walks ``batch_idx_to_section_variant`` and
    emits the per-row FID sidecar contribution by looking up the
    cached per-variant FID inverse from pass 1. Multi-mapped variants
    contribute identical sidecar content for every referencing row;
    padding rows contribute zero-length slices. Pass 2 only runs when
    ``collect_fid_sidecar=True``.

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
        is fully self-describing via ``fid_row_offsets``. Multi-mapped
        rows each get a full copy of the variant's sidecar content.

    Returns
    -------
    tuple
        ``(identities_flat_caller_local, fid_sidecar, fid_row_offsets,
        fid_per_category_counts)``. The first element is the same
        array that lives on ``stage3_batch`` (mutated in place);
        returned for caller convenience so the parent can pipe it into
        :class:`BatchDecodeResult` without re-reaching. The latter
        three are ``None`` when ``collect_fid_sidecar=False``.
        ``fid_per_category_counts`` is ``u32[batch_size, 3]`` whose
        columns follow :data:`FUNCTION_CATEGORIES` order
        (``LOCAL_FUNC, PLT_FUNC, EXT_FUNC``) and whose entries are the
        per-row deduped counter cardinality per Category -- i.e. the
        number of FID sidecar entries the row contributes to each
        Category. Padding rows are ``(0, 0, 0)``.
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
    stage1_batch = stage3_batch.stage2.stage1

    # Per-variant FID inverse cache keyed on (section_idx, slot_idx).
    # Populated in pass 1 ONLY when the sidecar is requested — pass 1
    # otherwise does not need to track the inverse map.
    fid_inverse_per_variant: Optional[
        dict[tuple[int, int], dict[Category, np.ndarray]]
    ] = {} if collect_fid_sidecar else None

    # ----- Pass 1: variant-keyed dedup walk. -----
    for section_idx, stage3_section in enumerate(stage3_batch.sections):
        for slot_idx, stage3_variant in enumerate(stage3_section.variants):
            if stage3_variant.stage2.stage1.batch_idx is None:
                # Variant slot not referenced by any batch row (RAGGED
                # post-cutoff drop, or PAD_NULL / RESAMPLE empty-source
                # fallback). Identity slice for every call_target has
                # length 0 (stage 2's surviving counts dropped it), so
                # there is nothing to remap.
                continue

            row_chunks = _walk_one_variant(
                stage3_variant,
                dedup_maps,
                identities_flat,
                collect_fid_sidecar=collect_fid_sidecar,
            )
            if fid_inverse_per_variant is not None:
                fid_inverse_per_variant[(section_idx, slot_idx)] = row_chunks

    if not collect_fid_sidecar:
        return identities_flat, None, None, None

    # ----- Pass 2: row-keyed FID sidecar emission. -----
    assert fid_inverse_per_variant is not None  # for type-checker

    # Build per-unique-variant flat sidecar arrays in section -> slot
    # order. Variants that pass 1 skipped (``batch_idx is None``) get
    # an empty u32 array -- they cannot be referenced by any batch row,
    # but keeping the slot in the flat index preserves the helper's
    # ``(section_idx, slot_idx) -> flat_variant_idx`` lookup contract.
    #
    # ``per_variant_category_counts`` parallels ``per_variant_sidecar``
    # and records the per-Category deduped counter cardinality for the
    # variant (``len(per_variant[cat])`` in :data:`FUNCTION_CATEGORIES`
    # order). It is the per-row segment length consumers need to slice
    # ``fid_sidecar`` per Category. Reading lengths from the inverse
    # arrays (rather than re-walking ``state.next_fresh_id``) is sound
    # because the dedup walk extends each Category's inverse list by
    # exactly the count of fresh ids minted, so list length equals
    # ``next_fresh_id`` at the end of the walk.
    per_variant_sidecar: list[np.ndarray] = []
    per_variant_category_counts: list[tuple[int, int, int]] = []
    variants_per_section: list[int] = []
    for section_idx, stage3_section in enumerate(stage3_batch.sections):
        variants_per_section.append(len(stage3_section.variants))
        for slot_idx in range(len(stage3_section.variants)):
            per_variant = fid_inverse_per_variant.get(
                (section_idx, slot_idx)
            )
            if per_variant is None:
                per_variant_sidecar.append(np.zeros(0, dtype=np.uint32))
                per_variant_category_counts.append((0, 0, 0))
                continue
            pieces = [per_variant[cat] for cat in FUNCTION_CATEGORIES]
            row_sidecar = (
                np.concatenate(pieces).astype(np.uint32)
                if any(p.size > 0 for p in pieces)
                else np.zeros(0, dtype=np.uint32)
            )
            per_variant_sidecar.append(row_sidecar)
            per_variant_category_counts.append(
                (int(pieces[0].size), int(pieces[1].size), int(pieces[2].size))
            )

    per_row_variant_idx, is_padding = build_per_row_variant_lookup(
        stage1_batch.batch_idx_to_section_variant, variants_per_section
    )
    fid_sidecar, fid_row_offsets = concat_per_row(
        per_variant_sidecar,
        per_row_variant_idx,
        is_padding,
        dtype=np.dtype(np.uint32),
    )
    fid_per_category_counts = _expand_per_category_counts_to_rows(
        per_variant_category_counts,
        per_row_variant_idx,
        is_padding,
    )
    return identities_flat, fid_sidecar, fid_row_offsets, fid_per_category_counts


def _expand_per_category_counts_to_rows(
    per_variant_category_counts: list[tuple[int, int, int]],
    per_row_variant_idx: np.ndarray,
    is_padding: np.ndarray,
) -> np.ndarray:
    """Project per-variant ``(LOCAL, PLT, EXT)`` count tuples onto each
    batch row via the per-row variant lookup.

    Returns ``u32[batch_size, 3]``; padding rows are zero. Multi-mapped
    variants naturally replicate their counts across every referencing
    row (matches the per-row sidecar replication done by
    :func:`concat_per_row`).
    """
    batch_size = int(per_row_variant_idx.shape[0])
    if not per_variant_category_counts:
        return np.zeros((batch_size, 3), dtype=np.uint32)
    per_variant_arr = np.asarray(
        per_variant_category_counts, dtype=np.uint32
    )  # shape (num_unique_variants, 3)
    # ``per_row_variant_idx`` is clamped-safe for padding rows (the
    # caller's contract); the ``np.where`` masks padding rows below.
    per_row = per_variant_arr[per_row_variant_idx]
    return np.where(
        is_padding[:, None],
        np.uint32(0),
        per_row,
    ).astype(np.uint32, copy=False)


def _walk_one_variant(
    stage3_variant: "Stage3Variant",
    dedup_maps: dict[Category, HashMapU32U16],
    identities_flat: np.ndarray,
    *,
    collect_fid_sidecar: bool,
) -> dict[Category, np.ndarray]:
    """Per-variant dedup walk.

    Runs the per-Category dedup + prepend write pipeline over one
    variant's call_targets (in encounter order). Mutates
    ``identities_flat`` in place. Returns a dict from FUNCTION
    Category to the per-variant inverse-map array
    (``u32[counter_count]``: ``counter_id -> function_name_ptr``, in
    dense counter-id order). When ``collect_fid_sidecar=False``, each
    entry is an empty u32 array — the dict shape is preserved so the
    caller's concatenation path stays uniform.

    The dedup state (FUNCTION hashmaps + :class:`_RowState`) is
    reset per call to this function. Under multi-row mapping the
    caller invokes this ONCE per unique variant; the per-row sidecar
    contribution is the same as the per-variant inverse map (the
    dedup walk is deterministic given the variant's call_targets
    encounter order + fresh-state-per-row contract).
    """
    # Reset the FUNCTION dedup maps for this variant. ``clean()``
    # retains the hashmap's bucket allocation per the plan's hot-path
    # discipline.
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
        # either the variant-level seed (for the root) or the parent
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
