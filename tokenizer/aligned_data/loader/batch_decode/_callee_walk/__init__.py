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

Module layout:

* :mod:`_pending` owns the :class:`PendingCallTarget` dataclass + the
  per-row mask-construction + run-length-handle-staging helper, plus
  :func:`finalise_pending_call_targets` that turns a finished pending
  list into :class:`Stage1CallTarget` rows.
* :mod:`_walker` owns the DFS recursion (the cycle / EXTERN / inlining
  filter logic), the public :func:`walk_callees_pending` entry point
  for the batched section-walk path, and the single-variant
  convenience wrapper :func:`walk_callees`.
"""

from ._pending import (
    PendingCallTarget,
    build_pending_call_target,
    finalise_pending_call_targets,
)
from ._walker import walk_callees, walk_callees_pending


__all__ = [
    "PendingCallTarget",
    "build_pending_call_target",
    "finalise_pending_call_targets",
    "walk_callees",
    "walk_callees_pending",
]
