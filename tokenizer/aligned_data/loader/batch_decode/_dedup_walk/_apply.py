"""Stage 4 step 1: per-row identity remap walk — public entry + batch-row loop.

Single concern: rewrite ``stage3_batch.identities_flat_caller_local`` IN
PLACE from per-function caller-local ids to **variant-global counter
ids per Category**. The remap LOGIC is a deterministic integer state
machine that lives in the Rust kernel
``dedup_hashmap.apply_remap_walk``; this module ADAPTS the referenced-
variant object-tree into the kernel's flat int arrays
(:func:`._flat_extract.extract_flat_remap_inputs`), drives the kernel,
and emits the row-keyed FID sidecar from the kernel's per-row output.
The kernel applies:

* ALG-3 (FUNCTION categories — ``LOCAL_FUNC`` / ``PLT_FUNC`` /
  ``EXT_FUNC``): hole-free dedup keyed on ``function_name_ptr``, one
  per-row hashmap per FUNCTION Category (owned by the kernel).
* ALG-4 (COUNTER categories — ``BLOCK`` / ``STRING_PTR`` /
  ``RO_DATA_PTR`` / ``RW_DATA_PTR`` / ``JUMP_TABLE``): pure offset bump
  per call_target, no dedup lookup.
* ALG-9 prepend slots (the leading ``identity_slice.start`` entry for
  each call_target): the self-counter is the kernel's dedup-map entry
  for the call_target's ``function_name_ptr`` in its
  ``encounter_category`` space.

This module owns ONLY the per-row remap adapter + sidecar emission; the
token-tensor assembly + number-chunk sidecar concatenation + variant-
padding policy live in their own stage-4 modules.

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

from dedup_hashmap import apply_remap_walk

from tokenizer.tokens import Category

from .._row_expand import build_per_row_variant_lookup, concat_per_row
from ._constants import (
    FUNCTION_CATEGORIES,
    ROOT_FUNC_SLOT,
)
from ._flat_extract import extract_flat_remap_inputs


if TYPE_CHECKING:
    from .._types import Stage3Batch


__all__ = ["apply_per_row_remap"]


def apply_per_row_remap(
    stage3_batch: "Stage3Batch",
    *,
    collect_fid_sidecar: bool = False,
) -> tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Apply ALG-3 + ALG-4 + ALG-9 per-row remap IN PLACE.

    Two passes:

    Pass 1 (variant-keyed) — flattens every referenced
    :class:`Stage3Variant` (``stage1.batch_idx is not None``) into the
    Rust kernel's flat int arrays and runs
    ``dedup_hashmap.apply_remap_walk``. The kernel, per variant in
    encounter order over its call_targets (skipping
    ``surviving_token_count == 0`` call_targets):

    1. Seeds LOCAL_FUNC with ``{root.function_name_ptr: 0}``; resets
       per-Category COUNTER offsets to 0.
    2. Writes the prepend slot (ALG-9: self-counter at
       ``identity_slice.start``).
    3. Runs ALG-3 dedup over every FUNCTION Category present in the
       call_target's ``call_targets_section``; applies the remap to the
       in-stream identity slice (skipping the prepend).
    4. Runs ALG-4 offset bump over every COUNTER Category; applies the
       offset to the in-stream identity slice.

    The kernel returns the per-variant FID inverse map, reshaped here
    into the ``(section_idx, slot_idx) -> {Category: array}`` cache that
    pass 2 consumes.

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
    identities_flat = stage3_batch.identities_flat_caller_local
    stage1_batch = stage3_batch.stage2.stage1

    # ----- Pass 1: variant-keyed dedup walk (Rust kernel). -----
    # The remap LOGIC lives in ``dedup_hashmap.apply_remap_walk``; here we
    # only adapt the object-tree subtree into the kernel's flat int arrays
    # (:func:`extract_flat_remap_inputs`) and reshape the kernel's per-row
    # FID inverse back into the ``(section_idx, slot_idx) -> {Category:
    # array}`` cache that pass 2 consumes. The kernel owns its own per-row
    # FUNCTION-Category hashmaps -- no caller-provided dedup-map pool.
    flat = extract_flat_remap_inputs(stage3_batch)
    per_row_fid_inverse = apply_remap_walk(
        flat.n_rows,
        len(FUNCTION_CATEGORIES),
        ROOT_FUNC_SLOT,
        identities_flat,
        flat.node_row,
        flat.node_skip,
        flat.node_prepend_pos,
        flat.node_fid,
        flat.node_enc_func_slot,
        flat.ct_off,
        flat.ct_fid,
        flat.ct_func_slot,
        flat.instream_off,
        flat.instream_func_slot,
        flat.instream_counter_slot,
        flat.counter_counts,
        collect_fid_sidecar,
    )

    if not collect_fid_sidecar:
        return identities_flat, None, None, None

    # Reshape the per-row kernel output (list[list[u32 ndarray]], outer =
    # row, inner = FUNCTION slot) into the per-variant cache keyed on
    # ``(section_idx, slot_idx)``. The kernel's row order matches
    # ``flat.row_keys`` (both walk referenced variants in section -> slot
    # order); the inner per-slot order matches ``FUNCTION_CATEGORIES``.
    assert per_row_fid_inverse is not None  # collect_fid_sidecar=True
    fid_inverse_per_variant: dict[
        tuple[int, int], dict[Category, np.ndarray]
    ] = {}
    for key, row_inverse in zip(flat.row_keys, per_row_fid_inverse):
        fid_inverse_per_variant[key] = {
            cat: np.asarray(row_inverse[slot], dtype=np.uint32)
            for slot, cat in enumerate(FUNCTION_CATEGORIES)
        }

    # ----- Pass 2: row-keyed FID sidecar emission. -----
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
