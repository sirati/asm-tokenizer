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

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category

if TYPE_CHECKING:
    from tokenizer.aligned_data.matched_sections_bin import CallTarget

    from ._types import Stage3Batch, Stage3CallTarget, Stage3Variant


__all__ = [
    "COUNTER_CATEGORIES",
    "FUNCTION_CATEGORIES",
    "NOT_FOUND_U16",
    "apply_per_row_remap",
]


# ---------------------------------------------------------------------------
# Category partition (plan D4).
#
# The unified vocab's IDENTITY block has 8 Categories. The dedup-walk
# treats them in two disjoint groups:
#
# * FUNCTION categories: identity values dedupe across functions within a
#   row by ``function_name_ptr`` (FID) equality.
# * COUNTER categories: identity values renumber by a running per-row
#   offset; no dedup lookup.
#
# The two tuples below are the single source of truth for the partition;
# every dispatch path in this module routes off them.
# ---------------------------------------------------------------------------
FUNCTION_CATEGORIES: tuple[Category, ...] = (
    Category.LOCAL_FUNC,
    Category.PLT_FUNC,
    Category.EXT_FUNC,
)

COUNTER_CATEGORIES: tuple[Category, ...] = (
    Category.BLOCK,
    Category.STRING_PTR,
    Category.RO_DATA_PTR,
    Category.RW_DATA_PTR,
    Category.JUMP_TABLE,
)


# ---------------------------------------------------------------------------
# Shifted vocab ids for the IDENTITY block (post strip+shift; ``id - 256``).
#
# The unified vocab pins the IDENTITY block at slots 264..271 in the
# pre-shift space (``_V2_IDENTITY_BLOCK_START`` = 264, count = 8). After
# stage 2's strip+shift, the model-facing token ids in
# ``expanded_token_ids`` are these vocab ids minus 256.
#
# The block layout is "user-canonical, then alphabetical" (plan vocab
# table + ALG-6):
#
#   offset 0 -> BLOCK_V2      -> shifted = 8
#   offset 1 -> LOCAL_FUNC    -> shifted = 9
#   offset 2 -> PLT_FUNC      -> shifted = 10
#   offset 3 -> EXT_FUNC      -> shifted = 11
#   offset 4 -> STRING_PTR    -> shifted = 12
#   offset 5 -> JUMP_TABLE    -> shifted = 13
#   offset 6 -> RO_DATA_PTR   -> shifted = 14
#   offset 7 -> RW_DATA_PTR   -> shifted = 15
#
# Resolving these once at import time keeps the per-row walk free of
# attribute lookups. The plan pins the unified vocab and every consumer
# asserts ``format_version=1``.
# ---------------------------------------------------------------------------
_V2_IDENTITY_BLOCK_START = VocabularyManager._V2_IDENTITY_BLOCK_START
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT


def _shifted(category_offset: int) -> int:
    return _V2_IDENTITY_BLOCK_START + category_offset - _V2_RESERVED_DIGIT_COUNT


_CATEGORY_TO_SHIFTED_ID: dict[Category, int] = {
    Category.BLOCK: _shifted(0),
    Category.LOCAL_FUNC: _shifted(1),
    Category.PLT_FUNC: _shifted(2),
    Category.EXT_FUNC: _shifted(3),
    Category.STRING_PTR: _shifted(4),
    Category.JUMP_TABLE: _shifted(5),
    Category.RO_DATA_PTR: _shifted(6),
    Category.RW_DATA_PTR: _shifted(7),
}


# Map ``CallTargetType`` to the FUNCTION Category it produces. The
# call_target table's ``type`` field uses ``CallTargetType`` (LOCAL /
# PLT / EXTERN), while the dedup dispatch is keyed on ``Category``
# (LOCAL_FUNC / PLT_FUNC / EXT_FUNC). This is the single mapping site.
_CALL_TARGET_TYPE_TO_CATEGORY: dict[CallTargetType, Category] = {
    CallTargetType.LOCAL: Category.LOCAL_FUNC,
    CallTargetType.PLT: Category.PLT_FUNC,
    CallTargetType.EXTERN: Category.EXT_FUNC,
}


# Sentinel for ``HashMapU32U16.lookup_ndarray`` misses (plan ALG-3 +
# ``dedup_hashmap/src/lib.rs`` miss-sentinel table for unsigned ints =
# ``<dtype>::MAX``).
NOT_FOUND_U16: np.uint16 = np.uint16(0xFFFF)


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


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


def _per_call_target_counter_count(
    call_target: "Stage3CallTarget",
    category: Category,
) -> int:
    """Per-call-target unique-id count for a COUNTER Category.

    Plan ALG-4 specifies the value as
    ``call_target.stage2.stage1.function_data.metadata["category_counts"][category]``.
    The metadata key is reserved by the plan but populated by the loader
    (a separate concern); this helper centralizes the lookup so the
    main remap walk stays algorithmic.

    Returns 0 if the function has no ids of this Category. Raises
    ``KeyError`` if the metadata field is absent — that is a loader
    contract violation, not a dedup-walk concern, and surfacing it as a
    typed error from one place keeps the diagnostic crisp.
    """
    metadata = call_target.stage2.stage1.function_data.metadata
    category_counts = metadata["category_counts"]
    return int(category_counts.get(category, 0))


def _surviving_in_stream_token_ids(
    call_target: "Stage3CallTarget",
) -> np.ndarray:
    """Token ids parallel to the call_target's in-stream identity slots.

    The call_target's ``identity_slice`` includes the prepend slot at
    its start (ALG-9); the in-stream slots are ``identity_slice``
    minus that first slot. The parallel token ids come from the
    SURVIVING in-stream IDENTITY-band tokens in
    ``expanded_token_ids[:partial_cut_length]`` — the prepend's row in
    that mask is dropped here.

    Returns a u16 ndarray of length
    ``identity_slice.stop - identity_slice.start - 1``.
    """
    stage2 = call_target.stage2
    surviving_expanded = stage2.expanded_token_ids[: stage2.partial_cut_length]
    # IDENTITY block shifted span: [8, 16) per the vocab table.
    identity_band_mask = (surviving_expanded >= np.uint16(8)) & (
        surviving_expanded < np.uint16(16)
    )
    identity_token_ids = surviving_expanded[identity_band_mask]
    # The first entry in ``identity_token_ids`` corresponds to the
    # prepend slot (slot 0 of ``expanded_token_ids`` is the calling-
    # category self-token, an IDENTITY-band id). Drop it so the
    # remaining ids parallel the call_target's in-stream identity
    # slice.
    return identity_token_ids[1:]


# ---------------------------------------------------------------------------
# Per-row state.
# ---------------------------------------------------------------------------


class _RowState:
    """Mutable per-row dedup state.

    Lives for the duration of one row's walk; reset at the top of each
    row. Holds:

    * Per-FUNCTION-Category ``next_fresh_id`` counters.
    * Per-FUNCTION-Category reverse-mapping list (``counter_id ->
      function_name_ptr``) for the optional fid sidecar collection.
    * Per-COUNTER-Category running offsets.

    The 3 ``HashMapU32U16`` instances are NOT owned by ``_RowState`` —
    they are caller-provided (the parent reuses them across rows via
    ``clean()`` per the plan).
    """

    __slots__ = (
        "next_fresh_id",
        "counter_offset",
        "fid_inverse",
    )

    def __init__(self, collect_fid_sidecar: bool) -> None:
        self.next_fresh_id: dict[Category, int] = {
            cat: 0 for cat in FUNCTION_CATEGORIES
        }
        self.counter_offset: dict[Category, int] = {
            cat: 0 for cat in COUNTER_CATEGORIES
        }
        # ``counter_id -> function_name_ptr`` per FUNCTION Category;
        # only populated when the sidecar is requested. Indices align
        # with counter ids because fresh ids are minted densely.
        self.fid_inverse: Optional[dict[Category, list[int]]] = (
            {cat: [] for cat in FUNCTION_CATEGORIES}
            if collect_fid_sidecar
            else None
        )


def _seed_local_func_root(
    state: _RowState,
    local_map: HashMapU32U16,
    root_fn_name_ptr: int,
) -> None:
    """Per ALG-3, ALG-9: LOCAL_FUNC root takes counter id 0.

    The dedup map gets the root FID -> 0 entry; ``next_fresh_id`` for
    LOCAL_FUNC advances to 1 so subsequent fresh LOCAL_FUNC ids start
    at 1. PLT_FUNC and EXT_FUNC remain empty with ``next_fresh_id`` = 0.
    """
    key = np.uint32(root_fn_name_ptr)
    local_map.insert(key, np.uint16(0))
    state.next_fresh_id[Category.LOCAL_FUNC] = 1
    if state.fid_inverse is not None:
        state.fid_inverse[Category.LOCAL_FUNC].append(root_fn_name_ptr)


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
    — written separately by :func:`_write_prepend_slot` per ALG-9).
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


def _bump_counter_category(
    state: _RowState,
    category: Category,
    call_target: "Stage3CallTarget",
    identities_flat: np.ndarray,
) -> None:
    """ALG-4: COUNTER-category offset bump for one call_target.

    Pure offset addition; no dedup lookup. The offset is the running
    total of unique caller-local ids in this Category across all prior
    call_targets in the row.
    """
    offset = state.counter_offset[category]
    per_function_count = _per_call_target_counter_count(call_target, category)

    if per_function_count > 0:
        # Apply offset to in-stream identity positions of this
        # Category. Skip if the offset is zero — the bump would be a
        # no-op.
        if offset > 0:
            in_stream_token_ids = _surviving_in_stream_token_ids(call_target)
            if in_stream_token_ids.size > 0:
                cat_token_id_shifted = np.uint16(
                    _CATEGORY_TO_SHIFTED_ID[category]
                )
                cat_mask = in_stream_token_ids == cat_token_id_shifted
                if cat_mask.any():
                    in_stream_sl = slice(
                        call_target.identity_slice.start + 1,
                        call_target.identity_slice.stop,
                    )
                    target_view = identities_flat[in_stream_sl]
                    target_view[cat_mask] = target_view[cat_mask] + np.uint16(
                        offset
                    )
        state.counter_offset[category] = offset + per_function_count


def _write_prepend_slot(
    dedup_maps: dict[Category, HashMapU32U16],
    call_target: "Stage3CallTarget",
    identities_flat: np.ndarray,
) -> None:
    """ALG-9: write the prepend slot's self-counter.

    The slot at ``identity_slice.start`` carries the call_target's own
    counter in its ``encounter_category`` space. For the root that is
    0 (seeded at row start). For inlined callees the counter was
    minted by the parent's dedup walk when the callee's FID first
    appeared as a call_target row of its category — so the dedup map
    for ``encounter_category`` already holds it.

    The token id at the prepend position (in ``tokens[row, col]``) is
    written by the 4b prepend stage, NOT here. This function owns
    ONLY the counter id at ``identities_flat_caller_local``.
    """
    stage1 = call_target.stage2.stage1
    dedup_map = dedup_maps[stage1.encounter_category]
    self_counter = dedup_map.lookup(np.uint32(stage1.function_name_ptr))
    if self_counter is None:
        # The callee's FID was not in the parent's call_targets_section
        # — that is a stage-1 walker invariant violation (every inlined
        # callee MUST have been an entry in its parent's call_targets
        # table; that is how stage 1 picked the call_target to inline).
        # Surface as a typed AssertionError so the diagnostic points at
        # the upstream concern rather than producing a wrong identity.
        raise AssertionError(
            "callee prepend slot has no minted self-counter for "
            f"(encounter_category={stage1.encounter_category!r}, "
            f"function_name_ptr={stage1.function_name_ptr}); the "
            "stage-1 walker must inline only callees whose FID lives "
            "in the parent's call_targets table for the matching "
            "Category."
        )
    identities_flat[call_target.identity_slice.start] = np.uint16(self_counter)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


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
