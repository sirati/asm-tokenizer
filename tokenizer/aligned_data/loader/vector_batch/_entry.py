"""Vectorized batch dataloader -- the entry orchestrator (plan C3).

Single concern: thread the vectorized path end to end -- sample the same
``(section, variant)`` rows ``batch_decode`` would, run the body-free
geometry prepass (plan C1), run the fused scatter (plan C2), and return a
typed result whose ``tokens`` + ``batch_idx_to_section_variant`` mirror
``batch_decode``'s. Backfill is a flag, DEFAULT OFF -- the optional
geometry-augmentation hook (TD) is left as a clean seam before the
scatter; this module never implements backfill.

Sampling parity (byte-identity contract): the SAME sampler
``batch_decode`` uses is reused verbatim -- :func:`...batch_decode.
_resolve_pointers.resolve_section_pointers` (RNG-sampled native variant
indices per section) + :func:`...batch_decode._batch_layout.
compute_batch_idx_mapping` (the ALG-10 ``batch_idx`` layout). Driven from
the same ``rng`` the two paths draw IDENTICAL samples, so the geometry +
scatter assemble byte-identically against ``batch_decode`` with backfill
off (the entry harness proves this).

Per-arm dispatch (byte-identity for UNMATCHED roots): the sampler + the
``batch_idx`` layout are ARM-AGNOSTIC -- one shared draw spans both arms.
The geometry + scatter + dense passes, however, are ARM-SCOPED (each
reads one arm's columnar catalog + RLG3 geometry + ``_data.bin``). So the
orchestrator GROUPS the resolved batch rows by arm and runs the
geometry -> scatter -> dense pipeline ONCE PER ARM against that arm's
handles, then merges the per-arm (disjoint-row) full-batch results. The
cross-arm DROP (a root that calls a callee in the OTHER arm) is automatic:
each arm's ``LiveNodeAdjacency`` ``_sec_map`` holds only THAT arm's
section offsets, so a cross-arm callee misses -> -1 -> dropped, exactly
``batch_decode``'s arm-keyed behaviour. This module adds NO cross-arm
following / inlining.

The geometry handles (columnar catalog + RLG3 geometry + ``_variants.bin``
+ ``_data.bin``) are opened body-free via the SAME readers the index
build uses (:func:`...sorted_index._prepass.read_region_section_variant_info`,
:class:`...realized_lengths.RealizedGeometryReader`) -- no bespoke BIN
parse. A single-arm :class:`.session_handles.VectorBatchHandles` is the
historical MATCHED-only contract; a :class:`.session_handles.
VectorBatchArmSet` carries both arms keyed by :class:`SectionKind`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._resolve_pointers import (
    resolve_section_pointers,
)
from tokenizer.aligned_data.loader.batch_decode._batch_layout import (
    compute_batch_idx_mapping,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from ._geometry import compute_batch_geometry
from ._merge import merge_arm_results
from ._scatter import build_dense_sidecars, scatter_batch_tokens
from .session_handles import VectorBatchArmSet, VectorBatchHandles


__all__ = ["VectorBatchResult", "vector_batch_tokens"]


#: The sentinel a PAD_NULL padding row carries in
#: ``batch_idx_to_section_variant`` (matches ``Stage1Batch``).
_PADDING_SENTINEL = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class VectorBatchResult:
    """The vectorized path's full result (backfill OFF).

    Mirrors the ``batch_decode`` ``BatchDecodeResult`` fields the
    byte-identity harness compares: the ``u16[B, L]`` token tensor, the
    ``u32[B, 2]`` ``(section_idx, variant_idx)`` mapping (padding rows
    hold the ``(UINT32_MAX, UINT32_MAX)`` sentinel), AND the DENSE
    sidecars -- the post-remap caller-local->counter identity array (+
    offsets), the ``(significand, sign_exp)`` numeric arrays (+ offsets),
    and the optional per-Category FID sidecars. Every array is
    byte-identical to ``batch_decode`` with backfill off.
    """

    tokens: np.ndarray
    batch_idx_to_section_variant: np.ndarray
    identities: np.ndarray
    identity_row_offsets: np.ndarray
    numbers_significant: np.ndarray
    numbers_sign_exponent: np.ndarray
    number_row_offsets: np.ndarray
    fid_sidecar: Optional[np.ndarray]
    fid_row_offsets: Optional[np.ndarray]
    fid_per_category_counts: Optional[np.ndarray]


def vector_batch_tokens(
    session,
    section_pointers: List[SectionPointerSpec],
    *,
    handles,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    rng: Optional[np.random.Generator] = None,
    augment_geometry=None,
    include_fid_sidecar: bool = False,
) -> VectorBatchResult:
    """Sample -> per-arm (geometry -> scatter -> dense) -> merge.

    Parameters
    ----------
    session:
        The :class:`BinarySession` the sampler resolves pointers +
        variants through (the SAME object ``batch_decode`` samples on).
    section_pointers:
        The section pointers to batch (one ``(arm, idx)`` per section).
        MATCHED and UNMATCHED roots are both supported when ``handles``
        is a both-arms :class:`.session_handles.VectorBatchArmSet`; with
        a single :class:`.session_handles.VectorBatchHandles` only the
        arm that bundle was opened for may be sampled.
    handles:
        Either a single-arm :class:`.session_handles.VectorBatchHandles`
        (treated as the MATCHED arm -- the historical contract) or a
        both-arms :class:`.session_handles.VectorBatchArmSet`. With an
        arm set, matched + unmatched roots are each routed through their
        own arm's columnar catalog + RLG3 geometry + ``_data.bin``, and
        the per-arm row tensors merge back into the one batch.
    num_variants_per_section / context_len / max_depth / variant_padding /
    rng:
        The same knobs ``batch_decode`` takes; ``rng`` defaults to a
        fresh non-reproducible generator (pass an explicit one for
        deterministic / equivalence-tested sampling).
    augment_geometry:
        OPTIONAL backfill seam (default ``None`` = backfill OFF). When
        provided it is a callable ``BatchGeometry -> BatchGeometry``
        applied AFTER the prepass and BEFORE the scatter (TD builds the
        backfill transform separately). This module never implements
        backfill; it only leaves the hook. Applied per arm.
    include_fid_sidecar:
        When True, the dense pass also produces the per-Category FID
        sidecars (``fid_sidecar`` / ``fid_row_offsets`` /
        ``fid_per_category_counts``), matching ``batch_decode``'s
        same-named flag. Default ``False`` (those fields are ``None``).

    Returns
    -------
    VectorBatchResult
        The token tensor + the ``batch_idx_to_section_variant`` mapping
        + the dense identity / numeric sidecars (+ optional FID sidecars).
    """
    if rng is None:
        rng = np.random.default_rng()
    handles_by_arm = _normalize_handles(handles)

    # --- sample EXACTLY as batch_decode does (shared sampler + rng) ------
    # The sampler + batch_idx layout are ARM-AGNOSTIC: one shared draw
    # spans both arms, so the per-arm runs assemble against the SAME
    # canonical mapping batch_decode produced.
    resolved = resolve_section_pointers(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        rng=rng,
    )
    batch_idx_to_section_variant, batch_size = compute_batch_idx_mapping(
        resolved,
        num_variants_per_section=num_variants_per_section,
        variant_padding=variant_padding,
        rng=rng,
    )

    # --- dispatch each non-padding row to its arm's handles --------------
    # Group the batch rows by arm; run the geometry -> scatter -> dense
    # pipeline ONCE PER ARM against that arm's catalog + RLG3 geometry +
    # _data.bin (over an arm-masked mapping so other arms' rows read as
    # padding), then merge the per-arm disjoint-row full-batch results.
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
            )
        )

    if not arm_results:
        return _empty_result(
            batch_idx_to_section_variant,
            batch_size=batch_size,
            context_len=context_len,
            include_fid_sidecar=include_fid_sidecar,
        )
    return merge_arm_results(
        arm_results,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
    )


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
) -> VectorBatchResult:
    """Geometry -> scatter -> dense for ONE arm's rows, full-batch shaped.

    ``masked_mapping`` is the canonical mapping with every OTHER arm's
    rows rewritten to the padding sentinel, so the geometry + dense
    passes run only over this arm's rows yet scatter back to the true
    batch positions (the re-expand + the dense CSR both key off the
    non-padding rows). The returned result fills only this arm's rows;
    the orchestrator merges the per-arm results row-wise.
    """
    root_sections, root_variants = _rows_to_catalog_nodes(
        masked_mapping, resolved
    )
    geometry = compute_batch_geometry(
        cols=handles.cols,
        section_offsets=handles.section_offsets,
        geometry=handles.geometry,
        variants_u8=handles.variants_u8,
        root_sections=root_sections,
        root_sampled_variants=root_variants,
        seq_len=context_len,
        max_depth=max_depth,
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
        identities=dense.identities,
        identity_row_offsets=dense.identity_row_offsets,
        numbers_significant=dense.numbers_significant,
        numbers_sign_exponent=dense.numbers_sign_exponent,
        number_row_offsets=dense.number_row_offsets,
        fid_sidecar=dense.fid_sidecar,
        fid_row_offsets=dense.fid_row_offsets,
        fid_per_category_counts=dense.fid_per_category_counts,
    )


def _empty_result(
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
        identities=np.empty(0, dtype=np.uint16),
        identity_row_offsets=zero_offsets,
        numbers_significant=np.empty(0, dtype=np.uint64),
        numbers_sign_exponent=np.empty(0, dtype=np.uint32),
        number_row_offsets=zero_offsets.copy(),
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_row_offsets,
        fid_per_category_counts=fid_per_category_counts,
    )


def _rows_to_catalog_nodes(batch_idx_to_section_variant, resolved):
    """Per NON-padding batch row, ``(catalog_section_idx, native_variant)``.

    ``batch_idx_to_section_variant`` column 0 is the position in
    ``resolved``; column 1 is the SLOT into that section's
    ``sampled_variant_indices`` (post-sampling, NOT the native variant).
    The prepass needs the catalog section index (per-arm ``idx``) + the
    NATIVE variant index, so we map both through ``resolved``.
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    is_padding = mapping[:, 0] == int(_PADDING_SENTINEL)
    real = mapping[~is_padding]
    sec_out = np.empty(real.shape[0], dtype=np.int64)
    var_out = np.empty(real.shape[0], dtype=np.int64)
    for i, (resolved_pos, slot) in enumerate(real.tolist()):
        rs = resolved[resolved_pos]
        sec_out[i] = int(rs.idx)
        var_out[i] = int(rs.sampled_variant_indices[slot])
    return sec_out, var_out


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
