"""Vectorized batch dataloader -- the entry orchestrator (plan C3).

Single concern: thread the vectorized path end to end -- sample the same
``(section, variant)`` rows ``batch_decode`` would (the shared,
arm-agnostic draw), hand the sample to the per-arm dispatch
(:mod:`._dispatch`), and merge (:mod:`._merge`) the per-arm results into
one typed :class:`._result.VectorBatchResult` whose ``tokens`` +
``batch_idx_to_section_variant`` mirror ``batch_decode``'s. Backfill is a
flag, DEFAULT OFF -- the optional geometry-augmentation hook (TD) is
threaded to the dispatch as a clean seam; this module never implements
backfill.

Sampling parity (byte-identity contract): the SAME sampler
``batch_decode`` uses is reused verbatim -- :func:`...batch_decode.
_resolve_pointers.resolve_section_pointers` (RNG-sampled native variant
indices per section) + :func:`...batch_decode._batch_layout.
compute_batch_idx_mapping` (the ALG-10 ``batch_idx`` layout). Driven from
the same ``rng`` the two paths draw IDENTICAL samples, so the geometry +
scatter assemble byte-identically against ``batch_decode`` with backfill
off (the entry harness proves this).

The sampler + the ``batch_idx`` layout are ARM-AGNOSTIC (one shared draw
spans both arms); the ARM-SCOPED geometry -> scatter -> dense pipeline +
the per-arm grouping live in :mod:`._dispatch`, which this module drives
once with the shared sample. A single-arm :class:`.session_handles.
VectorBatchHandles` is the historical MATCHED-only contract; a
:class:`.session_handles.VectorBatchArmSet` carries both arms keyed by
:class:`SectionKind`.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._batch_layout import (
    compute_batch_idx_mapping,
)
from ._resolve_geometry import resolve_section_geometry
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)

from ._dispatch import dispatch_by_depth_and_arm, empty_result
from ._merge import merge_arm_results
from ._result import VectorBatchResult


__all__ = ["VectorBatchResult", "vector_batch_tokens"]


def vector_batch_tokens(
    session,
    section_pointers: List[SectionPointerSpec],
    *,
    handles,
    num_variants_per_section: int,
    context_len: int,
    max_depth: Union[int, np.ndarray],
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    rng: Optional[np.random.Generator] = None,
    augment_geometry=None,
    include_fid_sidecar: bool = False,
    unmatched_inline: bool = False,
    unmatched_inline_depth: int = 3,
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
    num_variants_per_section / context_len / variant_padding / rng:
        The same knobs ``batch_decode`` takes; ``rng`` defaults to a
        fresh non-reproducible generator (pass an explicit one for
        deterministic / equivalence-tested sampling).
    max_depth:
        Splice depth, as a SCALAR ``int`` (every row decoded at that
        depth -- the historical contract) OR a per-SECTION-POINTER
        ``int`` array of length ``len(section_pointers)`` (each section
        decoded at its OWN depth -- the cross-depth path). Splice depth
        is a property of a section's spec, not of an individual expanded
        variant row, so the array is indexed by section pointer and
        gathered to per-row internally via the batch mapping. The per-row
        depths are grouped by distinct depth and each group runs the arm
        pipeline at its scalar depth, then all (depth x arm) partials
        merge row-wise. With a scalar (or a constant per-pointer array)
        the result is BYTE-IDENTICAL to the single-depth path: one depth
        group, one pass.
    augment_geometry:
        OPTIONAL backfill seam (default ``None`` = backfill OFF). When
        provided it is a callable ``BatchGeometry -> BatchGeometry``
        applied AFTER the prepass and BEFORE the scatter (TD builds the
        backfill transform separately). This module never implements
        backfill; it only threads the hook to the dispatch. Applied per
        arm.
    include_fid_sidecar:
        When True, the dense pass also produces the per-Category FID
        sidecars (``fid_sidecar`` / ``fid_row_offsets`` /
        ``fid_per_category_counts``), matching ``batch_decode``'s
        same-named flag. Default ``False`` (those fields are ``None``).
    unmatched_inline / unmatched_inline_depth:
        Opt-in unmatched-outline inlining (default OFF), mirroring
        ``batch_decode``'s same-named flags so the two loaders stay
        byte-identical with the feature on. When True the inclusion BFS
        surfaces the matched callees behind unmatched (``is_matched=False``)
        outlines, recursing unmatched->unmatched up to
        ``unmatched_inline_depth`` levels, and feeds those to outline
        detection in place of the outline shells. Default ``False``
        reproduces the pre-feature geometry byte-for-byte.

    Returns
    -------
    VectorBatchResult
        The token tensor + the ``batch_idx_to_section_variant`` mapping
        + the dense identity / numeric sidecars (+ optional FID sidecars).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- sample EXACTLY as batch_decode does (shared sampler + rng) ------
    # The sampler + batch_idx layout are ARM-AGNOSTIC: one shared draw
    # spans both arms, so the per-arm runs assemble against the SAME
    # canonical mapping batch_decode produced.
    # The vector_batch dispatch reads only each root's ARM, BIN
    # section_offset, and RNG-sampled variant indices -- never a parsed
    # ``Section`` object. Geometry-only resolve gathers exactly those via
    # parse-free offset lookups + ONE vectorized header read for the
    # per-section variant counts, skipping the ~B full
    # ``parse_section_bin`` walks ``resolve_section_pointers`` pays. The
    # RNG draw stays in lockstep with that resolver (same order, same
    # n_variants), so the sample is byte-identical to ``batch_decode``.
    resolved = resolve_section_geometry(
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

    # --- per-(depth, arm) dispatch over the shared sample, then merge ----
    # Normalise max_depth to a per-ROW array: a scalar broadcasts to a
    # constant array (one depth group -> the single-depth path runs
    # byte-identically); a per-POINTER array (one depth per root section
    # pointer -- the natural cross-depth unit, since depth is a property
    # of the section's spec, NOT of an individual expanded row) is
    # expanded to per-row through the just-computed mapping (column 0 =
    # the resolved/section-pointer index each row came from).
    max_depth_per_row = _normalize_max_depth(
        max_depth,
        batch_size=batch_size,
        num_section_pointers=len(section_pointers),
        batch_idx_to_section_variant=batch_idx_to_section_variant,
    )
    arm_results = dispatch_by_depth_and_arm(
        handles,
        resolved=resolved,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
        context_len=context_len,
        max_depth_per_row=max_depth_per_row,
        augment_geometry=augment_geometry,
        include_fid_sidecar=include_fid_sidecar,
        unmatched_inline=unmatched_inline,
        unmatched_inline_depth=unmatched_inline_depth,
    )
    if not arm_results:
        return empty_result(
            batch_idx_to_section_variant,
            batch_size=batch_size,
            context_len=context_len,
            include_fid_sidecar=include_fid_sidecar,
        )
    return merge_arm_results(
        arm_results,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
    )


_PADDING_SENTINEL = np.iinfo(np.uint32).max


def _normalize_max_depth(
    max_depth: Union[int, np.ndarray],
    *,
    batch_size: int,
    num_section_pointers: int,
    batch_idx_to_section_variant: np.ndarray,
) -> np.ndarray:
    """Resolve ``max_depth`` to an ``int[batch_size]`` per-row array.

    A scalar ``int`` fills every row with that depth (the single-depth
    contract -> one depth group downstream, byte-identical to today). An
    array is the per-SECTION-POINTER depth (length
    ``num_section_pointers`` -- one depth per root section pointer, since
    splice depth is a property of the section's spec, not of the
    individual expanded variant rows): it is GATHERED to per-row through
    ``batch_idx_to_section_variant`` column 0 (the resolved/section-
    pointer index each non-padding row came from). Padding rows (sentinel
    in the mapping) get depth 0 -- they are never decoded, so the value
    is inert; it only keeps the per-row array dense.

    A per-pointer length mismatch is a hard caller error (raised loudly,
    never silently truncated / padded).
    """
    arr = np.asarray(max_depth, dtype=np.int64)
    if arr.ndim == 0:
        return np.full(batch_size, int(arr), dtype=np.int64)
    arr = arr.reshape(-1)
    if arr.shape[0] != num_section_pointers:
        raise ValueError(
            f"per-pointer max_depth has length {arr.shape[0]} but there are "
            f"{num_section_pointers} section pointers"
        )
    mapping = np.asarray(batch_idx_to_section_variant)
    section_col = mapping[:, 0]
    is_padding = section_col == _PADDING_SENTINEL
    per_row = np.zeros(batch_size, dtype=np.int64)
    real = ~is_padding
    per_row[real] = arr[section_col[real].astype(np.int64)]
    return per_row
