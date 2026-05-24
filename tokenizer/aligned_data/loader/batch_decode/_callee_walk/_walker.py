"""DFS recursion + cycle detection for the callee walk.

Single concern: walk a resolved root variant's splice tree in
DFS-encounter order, gating each step on the EXTERN / unresolved /
active-cycle / inlining-filter rules, and emit one
:class:`PendingCallTarget` per included call_target (root first, then
inlined callees).

The per-row mask construction + :class:`InlineDecodeState` building
are NOT this module's concern: :mod:`_pending` owns those. The walker
asks :func:`build_pending_call_target` for each row, which stages the
two ``run_lengths`` passes on the shared
:class:`BucketedRunLengthCollector`.

Two public entry points:

* :func:`walk_callees_pending` is the batched-friendly path: the
  caller passes in a shared collector, gets back pending rows, and is
  responsible for flushing + finalising at the right batch boundary.
* :func:`walk_callees` is the single-variant convenience wrapper: it
  allocates a fresh collector, walks one variant, flushes, and
  finalises -- producing the same :class:`Stage1CallTarget` list the
  pre-bucketed code path produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.decoded._variant_selection import (
    called_by_in_selection,
    choose_callee_variant,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import CallTarget, Section
from tokenizer.tokens import Category

from .._types import Stage1CallTarget
from ._pending import (
    PendingCallTarget,
    build_pending_call_target,
    finalise_pending_call_targets,
)

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.session import BinarySession


# Per plan D3 / D4: the in-stream CallTargetType axis collapses to the
# in-vocab Category axis via this fixed table. EXT_FUNC has no entry --
# extern bodies are never inlined; the walker filters EXTERN rows out
# before reaching this table.
_CALL_TARGET_TYPE_TO_ENCOUNTER_CATEGORY = {
    CallTargetType.LOCAL: Category.LOCAL_FUNC,
    CallTargetType.PLT: Category.PLT_FUNC,
}


__all__ = ["walk_callees", "walk_callees_pending"]


def walk_callees(
    session: "BinarySession",
    *,
    root_arm: SectionKind,
    root_section: Section,
    root_variant_idx: int,
    root_function_data: "FunctionData",
    root_function_name_ptr: int,
    max_depth: int,
    inlined_equivalent_call_targets_only: bool,
) -> List[Stage1CallTarget]:
    """Flatten a variant's splice tree into encounter-order Stage1CallTargets.

    Returns ``[root, callee_1, callee_2, ...]``: index 0 is the root
    body (LOCAL_FUNC, ``parent_call_target_index=None``); indices 1+
    are inlined callees in DFS encounter order. The list shape is
    exactly the level-4 ``call_targets`` field on the owning
    :class:`Stage1Variant`.

    See the module docstring for the algorithm + parameter semantics.

    ``max_depth`` is the recursion-budget cap: ``0`` returns only the
    root (no callees are recursed); ``1`` returns the root + its direct
    callees; ``k`` returns the root + every callee whose DFS depth from
    the root is ``<= k``. ``max_depth < 0`` is rejected -- the caller
    should not be invoking the walker at all in that case.

    This is the single-variant convenience entry point: it allocates a
    fresh :class:`BucketedRunLengthCollector`, runs
    :func:`walk_callees_pending`, flushes the collector, and finalises
    the pending rows into :class:`Stage1CallTarget` instances. The
    batched section-walk path calls :func:`walk_callees_pending`
    directly with a SHARED collector across many variants, then
    flushes + finalises once at the end of the batch.
    """
    collector = BucketedRunLengthCollector()
    pending = walk_callees_pending(
        session=session,
        root_arm=root_arm,
        root_section=root_section,
        root_variant_idx=root_variant_idx,
        root_function_data=root_function_data,
        root_function_name_ptr=root_function_name_ptr,
        max_depth=max_depth,
        inlined_equivalent_call_targets_only=(
            inlined_equivalent_call_targets_only
        ),
        collector=collector,
    )
    results = collector.flush()
    return finalise_pending_call_targets(pending, results)


def walk_callees_pending(
    session: "BinarySession",
    *,
    root_arm: SectionKind,
    root_section: Section,
    root_variant_idx: int,
    root_function_data: "FunctionData",
    root_function_name_ptr: int,
    max_depth: int,
    inlined_equivalent_call_targets_only: bool,
    collector: BucketedRunLengthCollector,
) -> List[PendingCallTarget]:
    """DFS encounter-order :class:`PendingCallTarget` rows.

    Same algorithm as :func:`walk_callees`; differs only in that each
    emitted row carries collector handles instead of a constructed
    :class:`InlineDecodeState`. The caller is responsible for calling
    :meth:`BucketedRunLengthCollector.flush` + passing the result to
    :func:`finalise_pending_call_targets`.

    The collector is passed in so a section-walk batch can SHARE one
    collector across many (section, variant) pairs and pay the
    ``run_lengths`` dispatch cost once per pow2 bucket.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0; got {max_depth}")

    root_pending = build_pending_call_target(
        function_data=root_function_data,
        raw_tokens=root_function_data.full_token_stream(),
        call_targets_section=list(root_section.call_targets),
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=root_function_name_ptr,
        collector=collector,
    )

    out: List[PendingCallTarget] = [root_pending]
    visited: Set[Tuple[SectionKind, int]] = {
        (root_arm, root_section.section_offset)
    }

    _walk_recursive(
        session=session,
        arm=root_arm,
        parent_section=root_section,
        parent_variant_idx=root_variant_idx,
        current_depth=0,
        max_depth=max_depth,
        visited=visited,
        out=out,
        inlining_flag=inlined_equivalent_call_targets_only,
        collector=collector,
    )

    return out


# ---------------------------------------------------------------------------
# Internal recursion -- pure on its inputs modulo the session callbacks.
# ---------------------------------------------------------------------------


def _walk_recursive(
    *,
    session: "BinarySession",
    arm: SectionKind,
    parent_section: Section,
    parent_variant_idx: int,
    current_depth: int,
    max_depth: int,
    visited: Set[Tuple[SectionKind, int]],
    out: List[PendingCallTarget],
    inlining_flag: bool,
    collector: BucketedRunLengthCollector,
) -> None:
    """Depth-capped DFS body. Mutates ``out`` + ``visited`` in place.

    See the module docstring for the cycle / EXTERN / inlining-filter
    semantics. The parent's ``per_call_entries`` (via
    :func:`choose_callee_variant`) drives the callee variant pick at
    each descent step.
    """
    if current_depth >= max_depth:
        return
    if not parent_section.call_targets:
        return

    # Plan stage 1 step 4: the inlining filter (when on) scopes to the
    # parent variant's siblings within the same section. Independently,
    # ``choose_callee_variant`` consumes the same set as its level-2
    # fallback candidate pool. Using every parent variant covers both
    # consumers without threading a narrowed selection -- and matches
    # the plan's explicit scope ("the parent variant's siblings within
    # the same section").
    parent_sibling_v_idxs = frozenset(range(len(parent_section.variants)))

    for called_idx, ct in enumerate(parent_section.call_targets):
        callee_entry = _try_resolve_callee(
            session=session,
            arm=arm,
            parent_section=parent_section,
            parent_variant_idx=parent_variant_idx,
            parent_sibling_v_idxs=parent_sibling_v_idxs,
            called_idx=called_idx,
            ct=ct,
            visited=visited,
            inlining_flag=inlining_flag,
            collector=collector,
        )
        if callee_entry is None:
            continue

        callee_pending, callee_section, callee_variant_idx = callee_entry
        out.append(callee_pending)
        cycle_key = (arm, callee_section.section_offset)
        visited.add(cycle_key)
        try:
            _walk_recursive(
                session=session,
                arm=arm,
                parent_section=callee_section,
                parent_variant_idx=callee_variant_idx,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                visited=visited,
                out=out,
                inlining_flag=inlining_flag,
                collector=collector,
            )
        finally:
            # DAG-active-path semantics (plan D3): a callee reachable
            # via a DIFFERENT branch must be allowed to splice again
            # once we backtrack past it.
            visited.discard(cycle_key)


def _try_resolve_callee(
    *,
    session: "BinarySession",
    arm: SectionKind,
    parent_section: Section,
    parent_variant_idx: int,
    parent_sibling_v_idxs: frozenset,
    called_idx: int,
    ct: CallTarget,
    visited: Set[Tuple[SectionKind, int]],
    inlining_flag: bool,
    collector: BucketedRunLengthCollector,
) -> Optional[Tuple[PendingCallTarget, Section, int]]:
    """Resolve one call_target row to a (PendingCT, callee Section,
    callee variant idx) triple, or ``None`` if the row should be skipped.

    Skip reasons (matched against the existing splice walker's gates):
      - Extern call site (``ct.type is CallTargetType.EXTERN``) -- D3
        prohibits inlining extern bodies.
      - Unresolved pointer (``ct.function_section_ptr == 0``).
      - Active-path cycle (callee key already in ``visited``).
      - Cross-arm or missing section (``_idx_for_section_offset``
        returns ``None``).
      - No usable callee variant (``choose_callee_variant`` returns
        ``None`` -- vkey mismatch at every fallback level).
      - Inlining filter active AND the called_idx was called by EITHER
        no variants OR every variant of the parent section.
    """
    if ct.type is CallTargetType.EXTERN:
        return None
    if ct.function_section_ptr == 0:
        return None

    callee_byte_offset = int(ct.function_section_ptr)
    if (arm, callee_byte_offset) in visited:
        return None

    called_by_set = called_by_in_selection(
        parent_section, parent_sibling_v_idxs, called_idx
    )

    if inlining_flag:
        n_siblings = len(parent_sibling_v_idxs)
        # "Some but not all" -- skip when none called OR all called.
        if not called_by_set or len(called_by_set) == n_siblings:
            return None

    callee_variant_idx = choose_callee_variant(
        parent_section,
        parent_variant_idx,
        called_by_set,
        called_idx,
    )
    if callee_variant_idx is None:
        return None

    callee_idx = session._idx_for_section_offset(callee_byte_offset, arm.value)
    if callee_idx is None:
        return None

    # Per-arm load: matched takes (idx, variant_index); unmatched takes
    # just (idx) because the matched_sections_bin invariant guarantees
    # one variant per unmatched section.
    if arm is SectionKind.MATCHED:
        callee_fd, callee_section, _callee_section_offset = (
            session._load_matched_for_splice(callee_idx, callee_variant_idx)
        )
    else:
        callee_fd, callee_section, _callee_section_offset = (
            session._load_unmatched_for_splice(callee_idx)
        )

    # Variant tokens are prepended once per ROW (only the root carries
    # them); inlined callees feed body-only into the decode state so
    # the row never repeats the variant-axis prefix at each splice.
    callee_pending = build_pending_call_target(
        function_data=callee_fd,
        raw_tokens=callee_fd.tokens,
        call_targets_section=list(callee_section.call_targets),
        encounter_category=_CALL_TARGET_TYPE_TO_ENCOUNTER_CATEGORY[ct.type],
        parent_call_target_index=called_idx,
        function_name_ptr=int(ct.function_name_ptr),
        collector=collector,
    )
    return callee_pending, callee_section, callee_variant_idx
