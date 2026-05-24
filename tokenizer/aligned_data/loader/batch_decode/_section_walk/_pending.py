"""Pending Stage-1 batch shape + post-flush finalisation.

Single concern: the data shape that holds a fully-walked Stage-1 batch
BEFORE its shared :class:`BucketedRunLengthCollector` has been flushed,
plus the finalisation step that turns that pending shape into a frozen
:class:`Stage1Batch` once the collector's :meth:`flush` result is
available.

Why this split exists -- the orchestrator pattern:

    collector = BucketedRunLengthCollector()
    pendings = [walk_sections(..., collector=collector) for arg in args]
    runlen_results = collector.flush()        # ONE flush over all walks
    batches = [finalise_pending_stage1(p, runlen_results) for p in pendings]

The Stage 1 walker stages all per-call-target run-length passes on the
caller-owned collector (across as many ``walk_sections`` calls as the
orchestrator chains together). One :meth:`flush` then dispatches one
pow2-bucketed 2D ``run_lengths`` per bucket -- amortising the dispatch
cost across the WHOLE batch load, not just one ``walk_sections`` call.

The finalisation step is pure transformation: no IO, no further DFS, no
re-parsing. The pending shape carries every per-section / per-variant
:class:`PendingCallTarget` row in encounter order; finalisation reads
each row's two collector handles out of ``runlen_results`` and
constructs the frozen :class:`InlineDecodeState` +
:class:`Stage1CallTarget` via :func:`finalise_pending_call_targets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .._callee_walk import PendingCallTarget, finalise_pending_call_targets
from .._resolve_pointers import ResolvedSection
from .._types import (
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
)


__all__ = [
    "PendingStage1Batch",
    "finalise_pending_stage1",
]


@dataclass(frozen=True)
class PendingStage1Batch:
    """Stage-1 walk result BEFORE the shared collector has been flushed.

    Carries the resolved section pointers + per-variant pending call-
    target rows + the precomputed layout mapping. The two collector
    handles on each :class:`PendingCallTarget` are valid only as long
    as the originating collector has not been flushed; once the
    orchestrator flushes the collector, hand the resulting
    ``runlen_results`` dict to :func:`finalise_pending_stage1` together
    with this object to materialise the frozen :class:`Stage1Batch`.

    Lazy view: the resolved + pending lists are kept by reference. No
    copies. Mutation by the caller is undefined behaviour.
    """

    resolved: List[ResolvedSection]
    pending_per_variant: List[List[List[PendingCallTarget]]]
    batch_idx_to_section_variant: np.ndarray
    batch_size: int


def finalise_pending_stage1(
    pending: PendingStage1Batch,
    runlen_results: dict[int, np.ndarray],
) -> Stage1Batch:
    """Materialise a frozen :class:`Stage1Batch` from a pending batch.

    Reads each pending call-target row's run-length handles from
    ``runlen_results`` and builds the :class:`InlineDecodeState` +
    :class:`Stage1CallTarget` via the existing
    :func:`finalise_pending_call_targets` machinery. Pure
    transformation -- no DFS, no IO, no run-length work (that was
    amortized across pow2 buckets during the collector's flush).
    """
    sections: List[Stage1Section] = []
    for section_idx, (rs, section_pending) in enumerate(
        zip(pending.resolved, pending.pending_per_variant)
    ):
        sections.append(
            _build_stage1_section(
                section_idx=section_idx,
                resolved=rs,
                section_pending=section_pending,
                runlen_results=runlen_results,
                batch_idx_to_section_variant=(
                    pending.batch_idx_to_section_variant
                ),
            )
        )

    return Stage1Batch(
        sections=sections,
        batch_idx_to_section_variant=pending.batch_idx_to_section_variant,
        batch_size=pending.batch_size,
    )


# ---------------------------------------------------------------------------
# Per-section / per-variant builders -- the post-flush tree walk
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
    slot_v)`` may correspond to multiple batch rows; per the walker
    module's docstring we record the FIRST matching row and let
    downstream stages walk
    :attr:`Stage1Batch.batch_idx_to_section_variant` directly for the
    others.

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
