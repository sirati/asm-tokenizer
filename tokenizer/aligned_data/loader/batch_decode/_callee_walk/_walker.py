"""Level-synchronous section-level callee walk.

Single concern: flatten a matched/unmatched section's splice tree into
per-variant encounter-order ``list[PendingCallTarget]`` rows, level by
level, driving the SHARED once-only inclusion decider
(:mod:`...splice_inclusion`) over the section's SAMPLED variant subset.

The owner's algorithm (binding): the inclusion mask is
``[#sampled_variants, #functions]``. Per level, every SAMPLED variant's
resolved callees are marked; a function reached by EVERY sampled variant
is excluded (not emitted, pruned); a variant emits a function's body
only on its FIRST encounter. The root body is seeded at column 0 and
always included once, so self / mutual recursion never re-splices.

Why the mask spans the SAMPLED subset, not the section's full variant
set (the user's Decision 4): the columnwise-ALL exclusion is a property
of the rows that actually emit, so the apples-to-apples baseline for the
subset-based vector_batch path is a mask over exactly those rows. This
DELIBERATELY diverges from the sorted-index graph-lengths build (which
still spans every variant): a callee reached by every SAMPLED variant is
pruned here even when an unsampled variant would not reach it, so the
index length is now a LOOSE UPPER BOUND on the emitted token count, not
an equality (the index/backfill reconciles the slack). FLAG-A edge: a
section with exactly ONE sampled variant splices nothing (columnwise-ALL
over a single row is trivially that row).

Emission order is BFS level order: the root (level 0), then each
level's included callees in parent-then-call_target-slot order. This
CHANGES token order relative to the legacy DFS walk (which emitted a
callee's whole subtree before the next sibling). Stage-1 output shape
is unchanged: per sampled variant a ``list[PendingCallTarget]``
(``Stage1CallTarget`` after finalisation) with ``path_depth`` = the BFS
level.

The per-(parent variant, call_target slot) J-resolution is UNCHANGED;
:mod:`._resolve` owns it, and it still keys on the section's FULL
variant set (vkey matching is a section property, independent of which
rows form the inclusion mask). :mod:`._pending` owns per-row mask
construction + the ``run_lengths`` staging. This module owns only the
level-synchronous traversal + the shared-decider plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion
from tokenizer.tokens import Category

from ._pending import PendingCallTarget, build_pending_call_target
from ._resolve import (
    ResolvedCalleeMeta,
    load_callee_body,
    resolve_callee_metadata,
)

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.session import BinarySession


# Per plan D3 / D4: the in-stream CallTargetType axis collapses to the
# in-vocab Category axis via this fixed table. EXT_FUNC has no entry --
# extern bodies are never inlined; the resolver filters EXTERN rows out.
_CALL_TARGET_TYPE_TO_ENCOUNTER_CATEGORY = {
    CallTargetType.LOCAL: Category.LOCAL_FUNC,
    CallTargetType.PLT: Category.PLT_FUNC,
}


__all__ = ["walk_section_callees_pending", "walk_callees"]


def walk_callees(
    session: "BinarySession",
    *,
    root_arm: SectionKind,
    root_section: Section,
    root_variant_idx: int,
    root_function_data: "FunctionData",
    root_function_name_ptr: int,
    max_depth: int,
    inlined_equivalent_call_targets_only: bool = True,
) -> List[PendingCallTarget]:
    """Single-variant convenience wrapper over the section-level walk.

    Samples exactly ``root_variant_idx`` (the once-only mask spans that
    ONE row), so the section splices NOTHING (FLAG-A: columnwise-ALL over
    a single row is trivially that row) and the returned
    ``list[Stage1CallTarget]`` is root-only. This mirrors the subset
    semantics of the production walk (the mask spans only the sampled
    rows); it is no longer the full-set inclusion oracle.

    The root body is harvested from the session's per-arm load (the same
    load that surfaced the parsed section) so no body is re-parsed.
    ``inlined_equivalent_call_targets_only`` is accepted for source
    compatibility but is now ALWAYS-ON behaviour (the
    all-variants-equivalence exclusion is the default-and-only rule); a
    ``False`` value is asserted against -- the parameter is slated for
    retirement.
    """
    assert inlined_equivalent_call_targets_only, (
        "inlined_equivalent_call_targets_only is absorbed (always-on); "
        "the all-variants-equivalence exclusion is now the default-and-"
        "only behaviour -- pass True or drop the argument"
    )
    bodies = _load_root_bodies(session, root_arm, root_section)
    collector = BucketedRunLengthCollector()
    decider = OnceOnlyInclusion()
    per_variant = walk_section_callees_pending(
        session,
        arm=root_arm,
        section=root_section,
        sampled_variant_indices=[root_variant_idx],
        root_function_data_per_sampled=[bodies[root_variant_idx]],
        root_function_name_ptr=root_function_name_ptr,
        max_depth=max_depth,
        decider=decider,
        collector=collector,
    )
    from ._pending import finalise_pending_call_targets

    return finalise_pending_call_targets(per_variant[0], collector.flush())


def _load_root_bodies(
    session: "BinarySession",
    arm: SectionKind,
    section: Section,
) -> List["FunctionData"]:
    """The per-variant root :class:`FunctionData` for ``section``.

    The caller already holds the parsed ``section``; the per-variant
    bodies are materialised via the section-threaded body loaders
    (:py:meth:`_load_matched_variant_body` /
    :py:meth:`_load_unmatched_variant_body`), so ``_sections.bin`` is NOT
    re-parsed -- mirroring the callee-body load's no-reparse contract.
    Only the ``_idx_for_section_offset`` reverse-lookup (a cheap array
    ``np.where``, not a section parse) recovers the per-arm idx the body
    loaders need.

    The returned list is parallel to ``section.variants`` for BOTH arms,
    so the caller's ``bodies[root_variant_idx]`` selects the sampled
    variant's body. Unmatched sections (one record per variant, laid out
    contiguously from the first-record ``idx``) get one body per slot
    just like matched, so a non-first sampled root variant gets its OWN
    body, not always the first record.
    """
    idx = session._idx_for_section_offset(
        int(section.section_offset), arm.value
    )
    if arm is SectionKind.MATCHED:
        return [
            session._load_matched_variant_body(idx, v, section)
            for v in range(len(section.variants))
        ]
    return [
        session._load_unmatched_variant_body(idx, v, section)
        for v in range(len(section.variants))
    ]


@dataclass
class _RowFrontier:
    """One surviving parent of the level-synchronous BFS.

    ``mask_row`` is the dense SAMPLED-slot index (the
    :class:`OnceOnlyInclusion` row, stable across the whole descent, and
    also the index into the returned per-sampled list); ``section`` /
    ``variant_idx`` are the CURRENT node being expanded (the callee
    reached at the previous level, ``variant_idx`` is the section's
    native variant index that drives the unchanged J-resolution). Every
    frontier row emits -- the mask now spans exactly the sampled rows.
    """

    mask_row: int
    section: Section
    variant_idx: int


def walk_section_callees_pending(
    session: "BinarySession",
    *,
    arm: SectionKind,
    section: Section,
    sampled_variant_indices: List[int],
    root_function_data_per_sampled: List["FunctionData"],
    root_function_name_ptr: int,
    max_depth: int,
    decider: OnceOnlyInclusion,
    collector: BucketedRunLengthCollector,
) -> List[List[PendingCallTarget]]:
    """Level-synchronous BFS over a section's SAMPLED variant subset.

    Returns one ``list[PendingCallTarget]`` per SAMPLED variant (in
    ``sampled_variant_indices`` order): ``[root, callee, ...]`` in BFS
    encounter order, the same per-variant shape the legacy DFS produced.

    The once-only inclusion mask spans exactly the sampled rows: the
    mask row is the dense sampled SLOT (0..#sampled-1), not the section's
    native variant index. The slot doubles as the index into the
    returned per-sampled list, so every mask row emits. The native
    variant index is carried on the frontier separately and still drives
    the unchanged per-edge J-resolution.

    The ``decider`` is reused across sections (the caller resets nothing
    -- :meth:`OnceOnlyInclusion.begin_root` clears it here per section).
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0; got {max_depth}")

    n_sampled = len(sampled_variant_indices)
    decider.begin_root(n_sampled, int(section.section_offset))

    out: List[List[PendingCallTarget]] = []
    for slot, v_idx in enumerate(sampled_variant_indices):
        out.append(
            [
                build_pending_call_target(
                    function_data=root_function_data_per_sampled[slot],
                    raw_tokens=root_function_data_per_sampled[slot].tokens,
                    call_targets_section=list(section.call_targets),
                    encounter_category=Category.LOCAL_FUNC,
                    parent_call_target_index=None,
                    function_name_ptr=root_function_name_ptr,
                    path_depth=0,
                    collector=collector,
                )
            ]
        )

    frontier = [
        _RowFrontier(
            mask_row=slot,
            section=section,
            variant_idx=v_idx,
        )
        for slot, v_idx in enumerate(sampled_variant_indices)
    ]

    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        frontier = _step_level(
            session=session,
            arm=arm,
            frontier=frontier,
            depth=depth,
            decider=decider,
            collector=collector,
            out=out,
        )

    return out


def _step_level(
    *,
    session: "BinarySession",
    arm: SectionKind,
    frontier: List[_RowFrontier],
    depth: int,
    decider: OnceOnlyInclusion,
    collector: BucketedRunLengthCollector,
    out: List[List[PendingCallTarget]],
) -> List[_RowFrontier]:
    """Resolve one level: mark the mask, emit included callees, return
    the next level's frontier (the included pairs, descent per-variant).
    """
    # Resolve each surviving parent's DIRECT calls into flat pair lists.
    # A variant marks/splices ONLY the call_targets it itself called
    # (its ``per_call_entries``), not every slot in the section's union
    # call_target table -- the once-only mask keys on direct calls so the
    # all-sampled-variants-equivalence test ("reached by every sampled
    # variant") is meaningful (resolving every slot via the sibling
    # fallback would make every variant reach every callee and exclude
    # everything).
    rows: List[int] = []
    fids: List[int] = []
    resolved: List[ResolvedCalleeMeta] = []
    called_idxs: List[int] = []
    for fr in frontier:
        if not fr.section.call_targets:
            continue
        sibling_v_idxs = frozenset(range(len(fr.section.variants)))
        for called_idx in _direct_called_idxs(fr):
            ct = fr.section.call_targets[called_idx]
            rc = resolve_callee_metadata(
                session=session,
                arm=arm,
                parent_section=fr.section,
                parent_variant_idx=fr.variant_idx,
                parent_sibling_v_idxs=sibling_v_idxs,
                called_idx=called_idx,
                ct=ct,
            )
            if rc is None:
                continue
            rows.append(fr.mask_row)
            fids.append(rc.section_offset)
            resolved.append(rc)
            called_idxs.append(called_idx)

    if not resolved:
        return []

    result = decider.step_level(
        np.asarray(rows, dtype=np.int64),
        np.asarray(fids, dtype=np.uint32),
    )
    included = result.included

    # Emit included callees; collect the next frontier from every
    # included pair (descent is per-variant). Every mask row is a sampled
    # slot, so ``mask_row`` indexes directly into ``out``. The callee
    # body is read from ``_data.bin`` ONLY here, for survivors -- pruned
    # and multi-parent-deduped edges never pay the body load + egress
    # copy.
    next_frontier: List[_RowFrontier] = []
    for pair_idx in result.survivor_pairs.tolist():
        rc = resolved[pair_idx]
        called_idx = called_idxs[pair_idx]
        mask_row = rows[pair_idx]
        callee_body = load_callee_body(session, arm, rc)
        out[mask_row].append(
            _emit_pending(rc, callee_body, called_idx, depth, collector)
        )
        next_frontier.append(
            _RowFrontier(
                mask_row=mask_row,
                section=rc.section,
                variant_idx=rc.variant_idx,
            )
        )
    # ``survivor_pairs`` == included pairs (every included pair expands);
    # ``included`` and ``survivor_pairs`` agree by construction, so the
    # emit loop above covers exactly the included callees.
    assert int(included.sum()) == result.survivor_pairs.size
    return next_frontier


def _direct_called_idxs(fr: "_RowFrontier") -> List[int]:
    """Ascending-unique ``called_idx`` the frontier variant DIRECTLY
    called (its ``per_call_entries``).

    Ascending order matches the sorted-index build's pce ordering
    (``np.unique`` over per-call entries) so a variant's direct calls are
    walked in a stable, deterministic order -- the BFS emission order is
    then a function of the section bytes alone, not iteration accidents.
    """
    variant = fr.section.variants[fr.variant_idx]
    return sorted({int(ce[0]) for ce in variant.per_call_entries})


def _emit_pending(
    rc: ResolvedCalleeMeta,
    callee_body: "FunctionData",
    called_idx: int,
    depth: int,
    collector: BucketedRunLengthCollector,
) -> PendingCallTarget:
    """Build the :class:`PendingCallTarget` for one included callee."""
    return build_pending_call_target(
        function_data=callee_body,
        raw_tokens=callee_body.tokens,
        call_targets_section=list(rc.section.call_targets),
        encounter_category=_CALL_TARGET_TYPE_TO_ENCOUNTER_CATEGORY[
            rc.call_target_type
        ],
        parent_call_target_index=called_idx,
        function_name_ptr=rc.function_name_ptr,
        path_depth=depth,
        collector=collector,
    )
