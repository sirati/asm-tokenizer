"""Stage 1 DFS callee walk + cycle detection.

Single concern of this module: take ONE already-resolved root variant
and flatten its inline-call tree into a DFS encounter-order
``list[Stage1CallTarget]``.

The walk's contract -- one-sentence, per the design-first rule:

  *Given a resolved root function body, produce the level-4 list for
  the owning variant by depth-capped DFS over each call_target row that
  resolves to a non-extern callee in the same arm and isn't already on
  the active recursion path.*

That is the entire concern. Section pointer resolution + RNG variant
sampling at level 2 / 3 is task 1a's module; the
``batch_idx_to_section_variant`` mapping is task 1c's module; the
outer wiring that drives this walker per (section, variant) is task
1d's ``_section_walk.walk_sections``. None of that lives here.

Algorithm (from ``batch_decode_plan.md`` section ``## Stages --
algorithm sketch`` -> ``Stage 1: section walk + raw-data load``):

1. The root function is appended at index 0 with
   ``encounter_category=Category.LOCAL_FUNC`` (root is always a LOCAL
   entity by D3) and ``parent_call_target_index=None``.
2. DFS into the root's ``call_targets`` in encounter order:
   - Skip rows with ``function_section_ptr == 0`` (unresolved -- extern
     or missing callee section).
   - Skip rows whose ``type`` is :attr:`CallTargetType.EXTERN`:
     EXT_FUNC bodies are NOT inlined (plan D3).
   - Skip rows whose callee key ``(arm, section_byte_offset)`` is
     already in the ACTIVE visited set -- this is the cycle guard.
   - Resolve the callee through the session (``_idx_for_section_offset``
     + per-arm load). If the inverse lookup fails (cross-arm pointer,
     missing section) skip the row, matching the existing splice
     walker's ``is_callee_present`` gate.
   - Choose the callee variant index via
     :func:`choose_callee_variant` (data-driven, deterministic).
   - Append a new :class:`Stage1CallTarget` whose
     :attr:`encounter_category` is :attr:`Category.LOCAL_FUNC` for a
     LOCAL call site and :attr:`Category.PLT_FUNC` for a PLT call site
     (plan D3 + D4).
   - Recurse with ``current_depth + 1``; bail when
     ``current_depth >= max_depth``.
3. The visited set is keyed on ``(arm, section.section_offset)``. Popped
   on backtrack so DAG semantics hold: a callee reachable through two
   *different* recursion paths appears TWICE in the output (once per
   path). Only an *active* recursion-path cycle blocks further descent.
4. ``inlined_equivalent_call_targets_only`` filter (plan D5 stage 1
   step 4): when ``True``, skip a call_target whose ``called_idx`` was
   called by EITHER no variants OR every variant of the PARENT's
   section. Only rows where SOME but not ALL variants called the
   target carry "inlining variation" signal worth threading through to
   the model. The "variants" here are the parent variant's siblings
   within the same section -- NOT a globally-narrowed selection (the
   plan deliberately scopes the check to the immediate parent).

Why ``walk_callees`` does NOT take an ``rng``: callee variant choice
is driven by the parent variant's ``per_call_entries`` (data, not a
sampling decision) via :func:`choose_callee_variant`. The only sampling
in the stage-1 pipeline is the top-level per-section variant sampling
in task 1a -- the recursion is purely deterministic given a root
variant.

Why the entry point takes ``root_section`` and ``root_function_data``
already resolved: task 1a (section pointer resolution) is the single
owner of the level-2 section load + level-3 root variant load; this
module is task 1b and consumes those handles. The clean split lets
task 1d's outer walker call task 1a + task 1b in sequence per
``(section_pointer, sampled_variant)`` pair without either module
knowing about the other's internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    build_inline_decode_state,
)
from tokenizer.aligned_data.loader.decoded._variant_selection import (
    called_by_in_selection,
    choose_callee_variant,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import CallTarget, Section
from tokenizer.tokens import Category

from ._types import Stage1CallTarget

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


__all__ = ["walk_callees"]


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
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0; got {max_depth}")

    root_state = build_inline_decode_state(
        root_function_data.full_token_stream(), format_version=1
    )
    root_entry = Stage1CallTarget(
        function_data=root_function_data,
        state=root_state,
        call_targets_section=list(root_section.call_targets),
        encounter_category=Category.LOCAL_FUNC,
        parent_call_target_index=None,
        function_name_ptr=root_function_name_ptr,
    )

    out: List[Stage1CallTarget] = [root_entry]
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
    out: List[Stage1CallTarget],
    inlining_flag: bool,
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
        )
        if callee_entry is None:
            continue

        callee_st1, callee_section, callee_variant_idx = callee_entry
        out.append(callee_st1)
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
) -> Optional[Tuple[Stage1CallTarget, Section, int]]:
    """Resolve one call_target row to a (callee Stage1CT, callee Section,
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
    callee_state = build_inline_decode_state(
        callee_fd.tokens, format_version=1
    )
    callee_st1 = Stage1CallTarget(
        function_data=callee_fd,
        state=callee_state,
        call_targets_section=list(callee_section.call_targets),
        encounter_category=_CALL_TARGET_TYPE_TO_ENCOUNTER_CATEGORY[ct.type],
        parent_call_target_index=called_idx,
        function_name_ptr=int(ct.function_name_ptr),
    )
    return callee_st1, callee_section, callee_variant_idx
