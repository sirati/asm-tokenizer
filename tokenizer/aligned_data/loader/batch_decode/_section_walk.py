"""Stage 1 outer wiring -- compose the three Phase-1 submodules into a
:class:`Stage1Batch`.

This module owns ONE concern: take the request parameters, dispatch
through :func:`resolve_section_pointers` (1a) +
:func:`compute_batch_idx_mapping` (1c) + :func:`walk_callees` (1b), and
assemble the resulting per-section / per-variant / per-call-target tree.

Boundary contract (the design-first sentence):

  *Given a request's section-pointer list + padding policy + RNG, produce
  the level-1 :class:`Stage1Batch` with every level-2/3/4 child
  populated. ``batch_idx`` on each :class:`Stage1Variant` is the FIRST
  batch row that maps to that ``(section_idx, slot_v)`` in
  ``batch_idx_to_section_variant``, or ``None`` if no batch row maps
  there (e.g. the variant slot is not selected by the policy, or
  RESAMPLE has not chosen it as a leading slot).*

Per the plan, under :attr:`VariantPadding.RESAMPLE_WITHIN_SECTION` and
:attr:`VariantPadding.REDISTRIBUTE` a single ``(section_idx, slot_v)``
may correspond to MULTIPLE batch rows. The plan's
:class:`Stage1Variant.batch_idx` field is a single ``Optional[int]``;
this wiring records the FIRST batch row that maps to that slot, and
downstream stages must walk
:attr:`Stage1Batch.batch_idx_to_section_variant` directly to find the
other rows for the same variant slot. Documented inline at the
:func:`_first_batch_row_for_slot` helper below.

Variant bodies are NOT re-parsed here: the 1a step
(:func:`resolve_section_pointers`) harvests the per-sampled-variant
:class:`FunctionData` from the same per-arm load it already issues for
the parsed :class:`Section`, and threads them through on
:attr:`ResolvedSection.function_data_per_sampled_variant`. This wiring
indexes into that parallel list to pick up the root body for each
sampled slot -- no second per-arm load.

See ``batch_decode_plan.md`` section ``## Stages -- algorithm sketch``
-> ``Stage 1: section walk + raw-data load`` for the full algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)

from ._batch_layout import UINT32_MAX, compute_batch_idx_mapping
from ._callee_walk import (
    PendingCallTarget,
    finalise_pending_call_targets,
    walk_callees_pending,
)
from ._resolve_pointers import ResolvedSection, resolve_section_pointers
from ._types import (
    SectionPointerSpec,
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
    VariantPadding,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..session import BinarySession


__all__ = ["walk_sections"]


def walk_sections(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    max_depth: int,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool,
    rng: np.random.Generator,
) -> Stage1Batch:
    """Compose the Phase-1 submodules into a :class:`Stage1Batch`.

    Steps:

    1. Resolve each :class:`SectionPointerSpec` via
       :func:`resolve_section_pointers`, producing one
       :class:`ResolvedSection` per pointer (with RNG-sampled variant
       indices and the parallel per-variant :class:`FunctionData`).
    2. Compute ``batch_idx_to_section_variant`` via
       :func:`compute_batch_idx_mapping` per the :class:`VariantPadding`
       policy (plan ALG-10).
    3. Per resolved section + per sampled variant slot, read the
       pre-loaded variant body from
       :attr:`ResolvedSection.function_data_per_sampled_variant`, then
       call :func:`walk_callees` to DFS the splice tree.
    4. Wire ``batch_idx`` onto each :class:`Stage1Variant` (FIRST batch
       row matching its ``(section_idx, slot_v)``; ``None`` when no row
       maps to that slot).

    See module docstring for the multi-mapped-slot semantics (RESAMPLE /
    REDISTRIBUTE policies).
    """
    # --- step 1: resolve pointers + sample variants ---------------------
    resolved = resolve_section_pointers(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        rng=rng,
    )

    # --- step 2: compute the layout mapping -----------------------------
    batch_idx_to_section_variant, batch_size = compute_batch_idx_mapping(
        resolved,
        num_variants_per_section=num_variants_per_section,
        variant_padding=variant_padding,
        rng=rng,
    )

    # --- step 3a: drive ONE collector across every (section, variant) ---
    # The pending pattern lets us amortise `run_lengths` over every
    # call_target row in the batch: one pow2-bucketed 2D dispatch per
    # bucket on flush.
    collector = BucketedRunLengthCollector()
    pending_per_variant: List[List[List[PendingCallTarget]]] = []
    for rs in resolved:
        section_pending: List[List[PendingCallTarget]] = []
        for slot_v, variant_idx_in_section in enumerate(
            rs.sampled_variant_indices
        ):
            root_function_data = rs.function_data_per_sampled_variant[slot_v]
            section_pending.append(
                walk_callees_pending(
                    session=session,
                    root_arm=rs.arm,
                    root_section=rs.section,
                    root_variant_idx=variant_idx_in_section,
                    root_function_data=root_function_data,
                    root_function_name_ptr=int(rs.section.function_name_ptr),
                    max_depth=max_depth,
                    inlined_equivalent_call_targets_only=(
                        inlined_equivalent_call_targets_only
                    ),
                    collector=collector,
                )
            )
        pending_per_variant.append(section_pending)

    # --- step 3b: flush the collector + finalise pending rows ----------
    runlen_results = collector.flush()

    # --- step 3c+4: build the frozen Stage1 tree from finalised rows ---
    sections: List[Stage1Section] = []
    for section_idx, (rs, section_pending) in enumerate(
        zip(resolved, pending_per_variant)
    ):
        sections.append(
            _build_stage1_section(
                section_idx=section_idx,
                resolved=rs,
                section_pending=section_pending,
                runlen_results=runlen_results,
                batch_idx_to_section_variant=batch_idx_to_section_variant,
            )
        )

    return Stage1Batch(
        sections=sections,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_stage1_section(
    *,
    section_idx: int,
    resolved: ResolvedSection,
    section_pending: List[List[PendingCallTarget]],
    runlen_results: dict[int, np.ndarray],
    batch_idx_to_section_variant: np.ndarray,
) -> Stage1Section:
    """Build one :class:`Stage1Section` from a resolved pointer + the
    per-variant pending rows collected during the DFS pass.

    Finalisation happens here -- the pending rows for each variant are
    handed to :func:`finalise_pending_call_targets`, which consumes the
    shared collector's flush result to construct the per-row
    :class:`InlineDecodeState` + :class:`Stage1CallTarget` entries.
    """
    variants: List[Stage1Variant] = []
    for slot_v, variant_idx_in_section in enumerate(
        resolved.sampled_variant_indices
    ):
        variants.append(
            _build_stage1_variant(
                section_idx=section_idx,
                slot_v=slot_v,
                variant_idx_in_section=variant_idx_in_section,
                resolved=resolved,
                variant_pending=section_pending[slot_v],
                runlen_results=runlen_results,
                batch_idx_to_section_variant=batch_idx_to_section_variant,
            )
        )

    return Stage1Section(
        arm=resolved.arm,
        idx=resolved.idx,
        section=resolved.section,
        variants=variants,
    )


def _build_stage1_variant(
    *,
    section_idx: int,
    slot_v: int,
    variant_idx_in_section: int,
    resolved: ResolvedSection,
    variant_pending: List[PendingCallTarget],
    runlen_results: dict[int, np.ndarray],
    batch_idx_to_section_variant: np.ndarray,
) -> Stage1Variant:
    """Build one :class:`Stage1Variant`: finalise pending rows into
    :class:`Stage1CallTarget` instances + pick
    :attr:`Stage1Variant.batch_idx`.
    """
    call_targets: List[Stage1CallTarget] = finalise_pending_call_targets(
        variant_pending, runlen_results
    )

    batch_idx = _first_batch_row_for_slot(
        batch_idx_to_section_variant,
        section_idx=section_idx,
        slot_v=slot_v,
    )

    return Stage1Variant(
        variant_idx=int(variant_idx_in_section),
        variant_ref_offset=int(
            resolved.section.variants[variant_idx_in_section].variant_ref_offset
        ),
        batch_idx=batch_idx,
        call_targets=call_targets,
    )


def _first_batch_row_for_slot(
    batch_idx_to_section_variant: np.ndarray,
    *,
    section_idx: int,
    slot_v: int,
) -> Optional[int]:
    """Return the FIRST batch row that maps to ``(section_idx, slot_v)``.

    Under :attr:`VariantPadding.RESAMPLE_WITHIN_SECTION` and
    :attr:`VariantPadding.REDISTRIBUTE` a single ``(section_idx,
    slot_v)`` may correspond to multiple batch rows. Per this module's
    docstring, we record the FIRST matching row and let downstream
    stages walk :attr:`Stage1Batch.batch_idx_to_section_variant`
    directly for the others.

    Padding rows in the mapping carry ``(UINT32_MAX, UINT32_MAX)``; the
    sentinel is structurally != ``(section_idx, slot_v)`` for any real
    pair so the linear search naturally skips them.

    Returns ``None`` when no batch row maps to the slot -- e.g. the
    :attr:`VariantPadding.RAGGED` policy may leave a slot unmapped if
    its section has fewer real variants than another.
    """
    if batch_idx_to_section_variant.shape[0] == 0:
        return None
    section_match = (
        batch_idx_to_section_variant[:, 0] == np.uint32(section_idx)
    )
    slot_match = batch_idx_to_section_variant[:, 1] == np.uint32(slot_v)
    matches = np.flatnonzero(section_match & slot_match)
    if matches.size == 0:
        return None
    return int(matches[0])
