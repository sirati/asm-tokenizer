"""Per-arm dispatch of the vectorized batch path (plan C3).

Single concern: given the SHARED arm-agnostic sample (the resolved
sections + the canonical ``batch_idx`` mapping the orchestrator drew),
run the ARM-SCOPED geometry -> scatter -> dense pipeline ONCE PER ARM and
return the per-arm (disjoint-row) full-batch results the merge stitches.

The geometry + scatter + dense passes each read ONE arm's columnar
catalog + RLG3 geometry + ``_data.bin``, so the batch rows are GROUPED by
arm: every row not belonging to the arm under run is masked to the
padding sentinel, the arm's pipeline runs over its rows only, and the
result still carries the full ``[B, *]`` shape so the merge is a row-wise
union. The cross-arm DROP (a root that calls a callee in the OTHER arm)
is automatic -- each arm's ``LiveNodeAdjacency`` ``_sec_map`` holds only
THAT arm's section offsets, so a cross-arm callee misses -> -1 ->
dropped, exactly ``batch_decode``'s arm-keyed behaviour. This module adds
NO cross-arm following / inlining.

The geometry handles are opened body-free via the SAME readers the index
build uses (no bespoke BIN parse). A single-arm
:class:`.session_handles.VectorBatchHandles` is the historical
MATCHED-only contract; a :class:`.session_handles.VectorBatchArmSet`
carries both arms keyed by :class:`SectionKind`.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from ._geometry import compute_batch_geometry
from ._result import VectorBatchResult
from ._scatter import build_dense_sidecars, scatter_batch_tokens
from .session_handles import VectorBatchArmSet, VectorBatchHandles


__all__ = [
    "dispatch_by_arm",
    "dispatch_by_depth_and_arm",
    "empty_result",
]


#: The sentinel a PAD_NULL padding row carries in
#: ``batch_idx_to_section_variant`` (matches ``Stage1Batch``).
_PADDING_SENTINEL = np.iinfo(np.uint32).max


def dispatch_by_depth_and_arm(
    handles,
    *,
    resolved,
    batch_idx_to_section_variant: np.ndarray,
    batch_size: int,
    context_len: int,
    max_depth_per_row: np.ndarray,
    augment_geometry,
    include_fid_sidecar: bool,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
) -> List[VectorBatchResult]:
    """Outer per-DEPTH grouping over :func:`dispatch_by_arm`.

    ``max_depth_per_row`` is the per-row splice depth (``int[batch_size]``,
    padding rows carry any value -- they are sentinel in the mapping and
    never decoded). For each DISTINCT depth ``d`` present among the
    NON-padding rows, the rows whose depth != ``d`` are masked to the
    padding sentinel (:func:`_mask_other_depths`, mirroring
    :func:`_mask_other_arms`), and the existing :func:`dispatch_by_arm`
    runs the arm pipeline at the SCALAR ``max_depth=d`` over that masked
    mapping. The per-(depth, arm) partials are returned flat -- each fills
    only its own (depth-and-arm) rows of the shared ``[B, *]`` layout, so
    the caller's row-wise merge stitches them unambiguously (depth groups
    partition the non-padding rows exactly as arms do).

    BYTE-IDENTITY: when every non-padding row shares ONE depth (the only
    case the scalar-``max_depth`` callers ever produced), the loop runs
    once over an all-real depth mask, so the single ``dispatch_by_arm``
    call is made with the IDENTICAL mapping + scalar depth as before.
    """
    mapping = np.asarray(batch_idx_to_section_variant)
    section_col = mapping[:, 0]
    is_padding = section_col == _PADDING_SENTINEL
    depths = np.asarray(max_depth_per_row).reshape(-1)
    real_depths = depths[~is_padding]
    out: List[VectorBatchResult] = []
    # Ascending distinct depths -> deterministic partial order (the merge
    # is order-agnostic, but a stable order keeps the output reproducible).
    for depth in np.unique(real_depths).tolist():
        depth_masked = _mask_other_depths(
            batch_idx_to_section_variant, depths, int(depth)
        )
        out.extend(
            dispatch_by_arm(
                handles,
                resolved=resolved,
                batch_idx_to_section_variant=depth_masked,
                batch_size=batch_size,
                context_len=context_len,
                max_depth=int(depth),
                augment_geometry=augment_geometry,
                include_fid_sidecar=include_fid_sidecar,
                unmatched_inline=unmatched_inline,
                unmatched_inline_depth=unmatched_inline_depth,
            )
        )
    return out


def _mask_other_depths(
    batch_idx_to_section_variant: np.ndarray,
    max_depth_per_row: np.ndarray,
    depth: int,
) -> np.ndarray:
    """Rewrite every row NOT at ``depth`` to the padding sentinel.

    Mirrors :func:`_mask_other_arms` on the depth axis: a non-padding row
    whose ``max_depth_per_row`` value != ``depth`` is blanked to
    ``(UINT32_MAX, UINT32_MAX)`` so the per-depth arm pipeline sees ONLY
    this depth's rows, yet the result keeps the full ``[B, 2]`` shape so
    the per-depth partials merge row-wise. The original mapping is never
    mutated (a copy is returned).
    """
    mapping = np.asarray(batch_idx_to_section_variant)
    masked = mapping.copy()
    section_col = mapping[:, 0]
    is_padding = section_col == _PADDING_SENTINEL
    depths = np.asarray(max_depth_per_row).reshape(-1)
    keep = (~is_padding) & (depths == depth)
    masked[~keep] = _PADDING_SENTINEL
    return masked


def dispatch_by_arm(
    handles,
    *,
    resolved,
    batch_idx_to_section_variant: np.ndarray,
    batch_size: int,
    context_len: int,
    max_depth: int,
    augment_geometry,
    include_fid_sidecar: bool,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
) -> List[VectorBatchResult]:
    """Per-arm geometry -> scatter -> dense over the shared sample.

    ``handles`` is either a single-arm :class:`.session_handles.
    VectorBatchHandles` (the MATCHED-only contract) or a both-arms
    :class:`.session_handles.VectorBatchArmSet`. The batch rows are
    grouped by arm; each arm's pipeline runs over an arm-masked copy of
    ``batch_idx_to_section_variant`` (other arms' rows blanked to the
    padding sentinel) and yields a full-batch :class:`VectorBatchResult`
    populated for that arm's rows only. Returns one result per arm that
    owns at least one non-padding row (empty list when none do); the
    caller merges them row-wise.
    """
    handles_by_arm = _normalize_handles(handles)
    arm_results: List[VectorBatchResult] = []
    for arm, arm_handles in handles_by_arm.items():
        masked_mapping = _mask_other_arms(
            batch_idx_to_section_variant, resolved, arm
        )
        if not _has_real_rows(masked_mapping):
            continue
        arm_results.append(
            _run_arm_pipeline(
                arm_handles,
                masked_mapping=masked_mapping,
                batch_size=batch_size,
                resolved=resolved,
                context_len=context_len,
                max_depth=max_depth,
                augment_geometry=augment_geometry,
                include_fid_sidecar=include_fid_sidecar,
                unmatched_inline=unmatched_inline,
                unmatched_inline_depth=unmatched_inline_depth,
            )
        )
    return arm_results


def _normalize_handles(handles) -> Dict[SectionKind, VectorBatchHandles]:
    """Map ``handles`` to an arm -> :class:`VectorBatchHandles` dict.

    A bare :class:`VectorBatchHandles` is the historical MATCHED-only
    contract; a :class:`VectorBatchArmSet` carries both arms keyed by
    :class:`SectionKind`. Returning a dict keyed by arm lets the dispatch
    loop treat both shapes uniformly.
    """
    if isinstance(handles, VectorBatchArmSet):
        return dict(handles.by_kind)
    if isinstance(handles, VectorBatchHandles):
        return {SectionKind.MATCHED: handles}
    raise TypeError(
        f"handles must be VectorBatchHandles or VectorBatchArmSet, got "
        f"{type(handles).__name__}"
    )


def _mask_other_arms(
    batch_idx_to_section_variant: np.ndarray,
    resolved,
    arm: SectionKind,
) -> np.ndarray:
    """Rewrite every row NOT belonging to ``arm`` to the padding sentinel.

    A non-padding row ``r`` belongs to whichever arm its resolved root
    carries (``resolved[mapping[r, 0]].arm``). Blanking the other arms'
    rows to ``(UINT32_MAX, UINT32_MAX)`` makes the arm-scoped geometry +
    dense passes see ONLY this arm's rows (the dense builder + the token
    re-expand both key off the non-padding rows), yet the result still
    carries the full ``[B, 2]`` shape so the per-arm results merge
    row-wise. The original mapping is never mutated (a copy is returned).
    """
    mapping = np.asarray(batch_idx_to_section_variant)
    masked = mapping.copy()
    section_col = mapping[:, 0]
    is_padding = section_col == _PADDING_SENTINEL
    keep = np.zeros(mapping.shape[0], dtype=bool)
    real_rows = np.nonzero(~is_padding)[0]
    for r in real_rows.tolist():
        if resolved[int(section_col[r])].arm is arm:
            keep[r] = True
    masked[~keep] = _PADDING_SENTINEL
    return masked


def _has_real_rows(batch_idx_to_section_variant: np.ndarray) -> bool:
    """True iff the mapping carries at least one non-padding row."""
    mapping = np.asarray(batch_idx_to_section_variant)
    if mapping.shape[0] == 0:
        return False
    return bool((mapping[:, 0] != _PADDING_SENTINEL).any())


def _run_arm_pipeline(
    handles: VectorBatchHandles,
    *,
    masked_mapping: np.ndarray,
    batch_size: int,
    resolved,
    context_len: int,
    max_depth: int,
    augment_geometry,
    include_fid_sidecar: bool,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
) -> VectorBatchResult:
    """Geometry -> scatter -> dense for ONE arm's rows, full-batch shaped.

    ``masked_mapping`` is the canonical mapping with every OTHER arm's
    rows rewritten to the padding sentinel, so the geometry + dense
    passes run only over this arm's rows yet scatter back to the true
    batch positions (the re-expand + the dense CSR both key off the
    non-padding rows). The returned result fills only this arm's rows;
    the orchestrator merges the per-arm results row-wise.
    """
    root_sections, root_variants, root_groups = _rows_to_catalog_nodes(
        masked_mapping, resolved, section_offsets=handles.section_offsets
    )
    geometry = compute_batch_geometry(
        cols=handles.cols,
        section_offsets=handles.section_offsets,
        geometry=handles.geometry,
        variants_u8=handles.variants_u8,
        root_sections=root_sections,
        root_sampled_variants=root_variants,
        root_groups=root_groups,
        seq_len=context_len,
        max_depth=max_depth,
        unmatched_inline=unmatched_inline,
        unmatched_inline_depth=unmatched_inline_depth,
        # The remembered-excluded pool + dense reservation feed ONLY
        # backfill; compute them only when the backfill hook is present.
        need_excluded_pool=augment_geometry is not None,
        # The cols-invariant adjacency, built once per binary on the
        # handles -- so the per-binary MISSING inventory scan + offset map
        # are not rebuilt on every batch.
        adjacency=handles.adjacency,
    )
    if augment_geometry is not None:
        geometry = augment_geometry(geometry)

    scattered = scatter_batch_tokens(
        geometry,
        cols=handles.cols,
        data_u8=handles.data_u8,
        variants_u8=handles.variants_u8,
    )
    tokens = _expand_to_batch(
        scattered.tokens,
        masked_mapping,
        batch_size=batch_size,
        context_len=context_len,
    )
    depth_per_row = _depth_per_row_for_partial(
        masked_mapping, batch_size=batch_size, max_depth=max_depth
    )
    dense = build_dense_sidecars(
        geometry,
        scattered.expanded,
        cols=handles.cols,
        batch_idx_to_section_variant=masked_mapping,
        batch_size=batch_size,
        include_fid_sidecar=include_fid_sidecar,
    )
    return VectorBatchResult(
        tokens=tokens,
        batch_idx_to_section_variant=masked_mapping,
        depth_per_row=depth_per_row,
        identities=dense.identities,
        identity_row_offsets=dense.identity_row_offsets,
        numbers_significant=dense.numbers_significant,
        numbers_sign_exponent=dense.numbers_sign_exponent,
        number_row_offsets=dense.number_row_offsets,
        fid_sidecar=dense.fid_sidecar,
        fid_row_offsets=dense.fid_row_offsets,
        fid_per_category_counts=dense.fid_per_category_counts,
    )


def empty_result(
    batch_idx_to_section_variant: np.ndarray,
    *,
    batch_size: int,
    context_len: int,
    include_fid_sidecar: bool,
) -> VectorBatchResult:
    """The all-padding result (no non-padding row in any arm).

    Mirrors ``batch_decode``'s empty-batch shape: an all-zero ``[B, L]``
    token tensor, the canonical mapping, and zero-length dense sidecars
    with ``[B + 1]`` all-zero CSR offsets (+ ``[B, 3]`` zero FID counts
    when the flag is set).
    """
    tokens = np.zeros((batch_size, context_len), dtype=np.uint16)
    zero_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
    if include_fid_sidecar:
        fid_sidecar = np.empty(0, dtype=np.uint16)
        fid_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)
        fid_per_category_counts = np.zeros((batch_size, 3), dtype=np.uint32)
    else:
        fid_sidecar = fid_row_offsets = fid_per_category_counts = None
    return VectorBatchResult(
        tokens=tokens,
        batch_idx_to_section_variant=np.asarray(batch_idx_to_section_variant),
        depth_per_row=np.zeros(batch_size, dtype=np.int64),
        identities=np.empty(0, dtype=np.uint16),
        identity_row_offsets=zero_offsets,
        numbers_significant=np.empty(0, dtype=np.uint64),
        numbers_sign_exponent=np.empty(0, dtype=np.uint32),
        number_row_offsets=zero_offsets.copy(),
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_row_offsets,
        fid_per_category_counts=fid_per_category_counts,
    )


def _rows_to_catalog_nodes(batch_idx_to_section_variant, resolved, *, section_offsets):
    """Per NON-padding batch row, ``(catalog_section_idx, native_variant,
    decider_group)``.

    ``batch_idx_to_section_variant`` column 0 is the position in
    ``resolved``; column 1 is the SLOT into that section's
    ``sampled_variant_indices`` (post-sampling, NOT the native variant).
    The prepass needs the COLUMNAR catalog section index + the NATIVE
    variant index + the DECIDER-ROOT group id.

    The catalog section index is recovered ARM-AGNOSTICALLY: a section's
    BIN byte offset (``rs.section.section_offset``) is its universal key,
    and ``section_offsets`` (parallel to ``cols``, the same array the
    :class:`LiveNodeAdjacency` ``_sec_map`` is built from) maps it to the
    columnar position. ``rs.idx`` is NOT used: it is the per-arm load
    index, which equals the columnar section idx for the matched arm but
    is the per-RECORD idx for the unmatched arm (record idx != section
    idx once a function carries multiple versions). The byte-offset
    lookup is the single source of truth both arms share.

    The decider-root group is the RESOLVED-ENTRY index (mapping column 0)
    -- the originating ``batch_decode`` ``walk_section_callees_pending``
    unit (one resolved section pointer). Rows of the same resolved entry
    are that root's co-sampled variants (one ``begin_root`` mask); two
    rows that collide on the same catalog section but came from DIFFERENT
    resolved entries get DIFFERENT group ids, so the prepass treats each
    as its own root (the #67 fix -- no cross-root mask conflation).
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    offsets = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
    is_padding = mapping[:, 0] == int(_PADDING_SENTINEL)
    real = mapping[~is_padding]
    sec_out = np.empty(real.shape[0], dtype=np.int64)
    var_out = np.empty(real.shape[0], dtype=np.int64)
    grp_out = np.empty(real.shape[0], dtype=np.int64)
    for i, (resolved_pos, slot) in enumerate(real.tolist()):
        rs = resolved[resolved_pos]
        sec_out[i] = _columnar_section_idx(offsets, rs.section.section_offset)
        var_out[i] = int(rs.sampled_variant_indices[slot])
        grp_out[i] = int(resolved_pos)
    return sec_out, var_out, grp_out


def _columnar_section_idx(section_offsets: np.ndarray, section_offset: int) -> int:
    """Position of ``section_offset`` in the arm's ``section_offsets``.

    ``section_offsets`` is ascending (catalog / BIN order), so a binary
    search recovers the columnar index; a miss is a corpus / handle-arm
    mismatch and is raised loudly rather than silently returning a bogus
    neighbour.
    """
    pos = int(np.searchsorted(section_offsets, int(section_offset)))
    if pos >= section_offsets.size or int(section_offsets[pos]) != int(section_offset):
        raise ValueError(
            f"section_offset {int(section_offset)} not in this arm's "
            f"section_offsets (wrong-arm handles?)"
        )
    return pos


def _depth_per_row_for_partial(
    batch_idx_to_section_variant, *, batch_size, max_depth
):
    """The per-row source depth of ONE (depth, arm) partial's rows.

    The partial's ``masked_mapping`` is already masked to this single
    scalar ``max_depth`` (the depth grouping happened in
    :func:`dispatch_by_depth_and_arm`) and to this arm, so every
    non-padding row of it was decoded at ``max_depth``; padding rows
    (sentinel) hold ``0`` (inert -- never decoded). The result is
    ``int64[batch_size]`` populated for this partial's rows only, so the
    orchestrator's disjoint-row merge stitches the per-(depth, arm)
    partials exactly as it stitches the token tensor (each partition fills
    its own rows, all others zero).
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    depth_per_row = np.zeros(batch_size, dtype=np.int64)
    if batch_size == 0:
        return depth_per_row
    is_real = mapping[:, 0] != int(_PADDING_SENTINEL)
    depth_per_row[is_real] = int(max_depth)
    return depth_per_row


def _expand_to_batch(
    row_tokens, batch_idx_to_section_variant, *, batch_size, context_len
):
    """Place the per-non-padding-row token tensor into the full batch.

    The geometry + scatter run over the non-padding rows in batch order;
    this scatters them back to their ``batch_idx`` positions, leaving
    padding rows (sentinel) at the all-zero null-content default. The
    non-padding rows keep their original batch order, so the i-th
    geometry row maps to the i-th non-padding batch row.
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    tokens = np.zeros((batch_size, context_len), dtype=np.uint16)
    if batch_size == 0 or context_len == 0:
        return tokens
    is_real = mapping[:, 0] != int(_PADDING_SENTINEL)
    real_rows = np.nonzero(is_real)[0]
    if real_rows.size != row_tokens.shape[0]:
        raise AssertionError(
            f"geometry produced {row_tokens.shape[0]} rows but the batch "
            f"mapping has {real_rows.size} non-padding rows"
        )
    tokens[real_rows] = row_tokens
    return tokens
