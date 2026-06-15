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

from typing import List, Optional

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

from ._dispatch import dispatch_by_arm, empty_result
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
        backfill; it only threads the hook to the dispatch. Applied per
        arm.
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

    # --- sample EXACTLY as batch_decode does (shared sampler + rng) ------
    # The sampler + batch_idx layout are ARM-AGNOSTIC: one shared draw
    # spans both arms, so the per-arm runs assemble against the SAME
    # canonical mapping batch_decode produced.
    resolved = resolve_section_pointers(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        rng=rng,
        # vector_batch gathers bodies via the RLG3 geometry / scatter and
        # never reads ``function_data_per_sampled_variant``; skip the dead
        # per-sampled-variant body parse + category-count.
        load_bodies=False,
    )
    batch_idx_to_section_variant, batch_size = compute_batch_idx_mapping(
        resolved,
        num_variants_per_section=num_variants_per_section,
        variant_padding=variant_padding,
        rng=rng,
    )

    # --- per-arm dispatch over the shared sample, then row-wise merge ----
    arm_results = dispatch_by_arm(
        handles,
        resolved=resolved,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
        context_len=context_len,
        max_depth=max_depth,
        augment_geometry=augment_geometry,
        include_fid_sidecar=include_fid_sidecar,
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
