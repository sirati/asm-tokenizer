"""Stage 1 outer wiring -- compose the three Phase-1 submodules into a
:class:`Stage1Batch`.

This module owns ONE concern: take the request parameters, dispatch
through :func:`resolve_section_pointers` (1a) +
:func:`compute_batch_idx_mapping` (1c) + :func:`walk_callees_pending`
(1b), and assemble the resulting per-section / per-variant /
per-call-target tree.

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
other rows for the same variant slot. Documented inline at
:func:`_first_batch_row_for_slot` in :mod:`._pending`.

Variant bodies are NOT re-parsed here: the 1a step
(:func:`resolve_section_pointers`) harvests the per-sampled-variant
:class:`FunctionData` from the same per-arm load it already issues for
the parsed :class:`Section`, and threads them through on
:attr:`ResolvedSection.function_data_per_sampled_variant`. This wiring
indexes into that parallel list to pick up the root body for each
sampled slot -- no second per-arm load.

Collector-lifetime contract (the orchestrator-amortisation hook):

The Stage-1 walker stages every per-call-target ``run_lengths`` pass on
a shared :class:`BucketedRunLengthCollector`. Two dispatch shapes on
the single :func:`walk_sections` entry, picked via the ``collector``
kwarg:

* ``collector=None`` (default): the walker allocates a fresh
  collector, runs the walk, flushes once, and returns a finalised
  :class:`Stage1Batch`. Existing callers see the same byte-for-byte
  output as before.
* ``collector`` provided: the walker stages onto the caller-owned
  collector and returns a :class:`PendingStage1Batch`. The caller is
  responsible for flushing the collector + calling
  :func:`finalise_pending_stage1` on each pending batch, in that
  order. This lifts the ``run_lengths`` amortisation from per-walk to
  per-batch-load (one flush spanning every Stage-1 call inside a
  single batch_load).

See ``batch_decode_plan.md`` section ``## Stages -- algorithm sketch``
-> ``Stage 1: section walk + raw-data load`` for the full algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union, overload

import numpy as np

from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion

from .._batch_layout import compute_batch_idx_mapping
from .._callee_walk import (
    CalleeSectionMetaMemo,
    PendingCallTarget,
    walk_section_callees_pending,
)
from .._resolve_pointers import resolve_section_pointers
from .._types import (
    SectionPointerSpec,
    Stage1Batch,
    VariantPadding,
)
from ._pending import PendingStage1Batch, finalise_pending_stage1

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ...session import BinarySession


__all__ = ["walk_sections"]


@overload
def walk_sections(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    max_depth: int,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool = True,
    rng: np.random.Generator,
    collector: None = ...,
) -> Stage1Batch: ...
@overload
def walk_sections(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    max_depth: int,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool = True,
    rng: np.random.Generator,
    collector: BucketedRunLengthCollector,
) -> PendingStage1Batch: ...
def walk_sections(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    max_depth: int,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool = True,
    rng: np.random.Generator,
    collector: Optional[BucketedRunLengthCollector] = None,
) -> Union[Stage1Batch, PendingStage1Batch]:
    """Compose the Phase-1 submodules into a :class:`Stage1Batch` (or
    :class:`PendingStage1Batch` when ``collector`` is provided).

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
       call :func:`walk_callees_pending` to DFS the splice tree (every
       row's two ``run_lengths`` passes get staged on the shared
       collector).
    4. EITHER (``collector=None``) flush a freshly-allocated collector
       and finalise into a :class:`Stage1Batch`; OR (``collector``
       provided) return a :class:`PendingStage1Batch` for the caller
       to finalise after they flush the shared collector.

    See module docstring for the multi-mapped-slot semantics + the
    orchestrator amortisation contract.
    """
    # ONE decision point: own-collector flush-now OR caller-owned defer.
    if collector is None:
        owned = BucketedRunLengthCollector()
        pending = _walk_sections_pending(
            session,
            section_pointers,
            num_variants_per_section=num_variants_per_section,
            max_depth=max_depth,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            rng=rng,
            collector=owned,
        )
        return finalise_pending_stage1(pending, owned.flush())

    return _walk_sections_pending(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        max_depth=max_depth,
        variant_padding=variant_padding,
        inlined_equivalent_call_targets_only=(
            inlined_equivalent_call_targets_only
        ),
        rng=rng,
        collector=collector,
    )


def _walk_sections_pending(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    max_depth: int,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool = True,
    rng: np.random.Generator,
    collector: BucketedRunLengthCollector,
) -> PendingStage1Batch:
    """Internal: do the full Stage-1 walk, staging on the supplied
    collector but NOT flushing it. Returns the
    :class:`PendingStage1Batch` shape the post-flush finaliser
    consumes.

    Pure delegation -- the caller (be it :func:`walk_sections` itself
    on the synchronous path or an outer orchestrator on the deferred
    path) owns the collector's flush.
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

    # --- step 3: level-synchronous BFS per section onto the shared
    # collector. ``inlined_equivalent_call_targets_only`` is absorbed
    # (always-on); the all-variants-equivalence exclusion is the
    # default-and-only behaviour, so the flag no longer threads into the
    # walk (asserted True for the retirement window).
    assert inlined_equivalent_call_targets_only, (
        "inlined_equivalent_call_targets_only is absorbed (always-on)"
    )
    # The once-only inclusion mask is reset per section; ONE decider
    # instance is reused across every section in the batch (buffer reuse
    # -- the mandate's geometric-growth, no-per-section-realloc rule).
    decider = OnceOnlyInclusion()
    # The callee-section parse memo is reset per batch (this walk's
    # lifetime): a callee catalog entry reached from ANY section in the
    # batch parses once. It is dropped when ``_walk_sections_pending``
    # returns -- a within-walk parse memo, not a persistent cache.
    section_meta_memo = CalleeSectionMetaMemo()
    pending_per_variant: List[List[List[PendingCallTarget]]] = []
    for rs in resolved:
        pending_per_variant.append(
            walk_section_callees_pending(
                session=session,
                arm=rs.arm,
                section=rs.section,
                sampled_variant_indices=list(rs.sampled_variant_indices),
                root_function_data_per_sampled=(
                    rs.function_data_per_sampled_variant
                ),
                root_function_name_ptr=int(rs.section.function_name_ptr),
                max_depth=max_depth,
                decider=decider,
                collector=collector,
                section_meta_memo=section_meta_memo,
            )
        )

    return PendingStage1Batch(
        resolved=resolved,
        pending_per_variant=pending_per_variant,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
    )
