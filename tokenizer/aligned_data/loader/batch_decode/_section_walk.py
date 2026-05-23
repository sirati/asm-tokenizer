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

Re-loading the variant body's :class:`FunctionData`: the 1a step
(:func:`resolve_section_pointers`) deliberately DISCARDS the loader's
:class:`FunctionData` because the variant-sampling concern doesn't
need it. This wiring re-issues the per-arm load via
:meth:`BinarySession._load_matched_for_splice` (matched) or
:meth:`BinarySession._load_unmatched_for_splice` (unmatched). The
session caches the underlying memmap segments so the second load is
idempotent and cheap. We chose this path over a 1a API extension to
keep the variant-sampling concern free of per-variant function-data
load coupling -- the 1a module's job is "what variants?" and this
module's job is "load the bodies + walk the trees".

See ``batch_decode_plan.md`` section ``## Stages -- algorithm sketch``
-> ``Stage 1: section walk + raw-data load`` for the full algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from ..metadata_loader import SectionKind
from ._batch_layout import UINT32_MAX, compute_batch_idx_mapping
from ._callee_walk import walk_callees
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
    from ..function_data import FunctionData
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
       indices).
    2. Compute ``batch_idx_to_section_variant`` via
       :func:`compute_batch_idx_mapping` per the :class:`VariantPadding`
       policy (plan ALG-10).
    3. Per resolved section + per sampled variant slot, re-load the
       variant body's :class:`FunctionData` through the per-arm loader,
       then call :func:`walk_callees` to DFS the splice tree.
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

    # --- step 3+4: per resolved section, build Stage1Section ------------
    sections: List[Stage1Section] = []
    for section_idx, rs in enumerate(resolved):
        sections.append(
            _build_stage1_section(
                session=session,
                section_idx=section_idx,
                resolved=rs,
                batch_idx_to_section_variant=batch_idx_to_section_variant,
                max_depth=max_depth,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
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
    session: "BinarySession",
    section_idx: int,
    resolved: ResolvedSection,
    batch_idx_to_section_variant: np.ndarray,
    max_depth: int,
    inlined_equivalent_call_targets_only: bool,
) -> Stage1Section:
    """Build one :class:`Stage1Section` from a resolved pointer.

    Per slot ``v`` in ``resolved.sampled_variant_indices``: re-load the
    body's :class:`FunctionData` through the per-arm loader, walk the
    splice tree via :func:`walk_callees`, and assemble the
    :class:`Stage1Variant`.
    """
    variants: List[Stage1Variant] = []
    for slot_v, variant_idx_in_section in enumerate(
        resolved.sampled_variant_indices
    ):
        variants.append(
            _build_stage1_variant(
                session=session,
                section_idx=section_idx,
                slot_v=slot_v,
                variant_idx_in_section=variant_idx_in_section,
                resolved=resolved,
                batch_idx_to_section_variant=batch_idx_to_section_variant,
                max_depth=max_depth,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
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
    session: "BinarySession",
    section_idx: int,
    slot_v: int,
    variant_idx_in_section: int,
    resolved: ResolvedSection,
    batch_idx_to_section_variant: np.ndarray,
    max_depth: int,
    inlined_equivalent_call_targets_only: bool,
) -> Stage1Variant:
    """Build one :class:`Stage1Variant`: re-load body + DFS callees +
    pick :attr:`Stage1Variant.batch_idx`.
    """
    root_function_data, _root_section, _root_section_offset = _load_variant_body(
        session=session,
        arm=resolved.arm,
        idx=resolved.idx,
        variant_idx_in_section=variant_idx_in_section,
    )

    call_targets: List[Stage1CallTarget] = walk_callees(
        session=session,
        root_arm=resolved.arm,
        root_section=resolved.section,
        root_variant_idx=variant_idx_in_section,
        root_function_data=root_function_data,
        root_function_name_ptr=int(resolved.section.function_name_ptr),
        max_depth=max_depth,
        inlined_equivalent_call_targets_only=(
            inlined_equivalent_call_targets_only
        ),
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


def _load_variant_body(
    *,
    session: "BinarySession",
    arm: SectionKind,
    idx: int,
    variant_idx_in_section: int,
) -> Tuple["FunctionData", object, int]:
    """Per-arm re-load of one variant body.

    Matched: :meth:`BinarySession._load_matched_for_splice(idx,
    variant_index)`. Unmatched:
    :meth:`BinarySession._load_unmatched_for_splice(idx)` (one variant
    per unmatched section by the matched_sections_bin invariant; the
    walker's ``variant_idx_in_section`` is necessarily 0 there).

    Returns the loader's full ``(FunctionData, Section, section_offset)``
    triple; the caller uses only the :class:`FunctionData` -- the
    :class:`Section` was already parsed by 1a and lives on
    :attr:`ResolvedSection.section`.
    """
    if arm is SectionKind.MATCHED:
        return session._load_matched_for_splice(idx, variant_idx_in_section)
    if arm is SectionKind.UNMATCHED:
        return session._load_unmatched_for_splice(idx)
    raise ValueError(f"unknown SectionKind: {arm!r}")


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
