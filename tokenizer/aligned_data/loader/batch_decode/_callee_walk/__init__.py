"""Stage 1 level-synchronous section-level callee walk.

Single concern of this module: flatten a section's splice tree into
per-variant BFS-encounter-order ``list[Stage1CallTarget]`` rows,
enforcing the owner's once-only + all-variants-equivalence inclusion
semantics over the section's FULL variant set.

The walk's contract -- one-sentence, per the design-first rule:

  *Given a resolved section + its sampled variant root bodies, produce
  one level-4 list per sampled variant by a level-synchronous BFS over
  the section's full variant set, including each function body once per
  variant on first encounter and excluding (+ pruning) any function
  reached by every variant at that level -- the inclusion decision owned
  by the shared :mod:`...splice_inclusion` decider.*

That is the entire concern. Section pointer resolution + RNG variant
sampling is the 1a module (:mod:`.._resolve_pointers`); the
``batch_idx`` mapping is 1c (:mod:`.._batch_layout`); the outer wiring
that drives this walk per section is 1d's
:func:`.._section_walk.walk_sections`. None of that lives here.

Inclusion algorithm (the owner's spec, binding -- supersedes the legacy
plan-D3 active-path DAG semantics; see ``batch_decode_plan.md`` D3 as
amended):

1. The root function is appended at index 0 with
   ``encounter_category=Category.LOCAL_FUNC`` (root is always a LOCAL
   entity) and ``parent_call_target_index=None``. The root's section is
   seeded at the once-only mask's column 0, so any deeper call resolving
   to the root section is already-included (self / mutual recursion
   never re-splices).
2. The mask is ``[#section_variants, #functions]``. Per BFS level, every
   surviving parent's call_targets are resolved per variant (the
   per-edge J-resolution is UNCHANGED -- :mod:`._resolve` owns it), the
   resolved callee SECTION is marked for that variant, and the shared
   decider returns which ``(variant, callee)`` pairs are INCLUDED.
3. **Once-only.** A variant emits a function body only on its FIRST
   encounter (mask cell False->True). Diamonds, branch-shared callees,
   and recursion all dedup -- the function appears once per variant.
4. **All-variants equivalence.** A function reached by EVERY variant at
   a level (columnwise ALL over the variant axis) is NOT emitted and is
   PRUNED (never expanded deeper). This is the default-and-only
   behaviour; the legacy ``inlined_equivalent_call_targets_only`` flag
   is absorbed (always-on) and slated for retirement.

Emission order is BFS level order (root, then each level's included
callees in parent-then-slot order) -- DIFFERENT from the legacy DFS
subtree-first order, but the same per-variant ``Stage1CallTarget`` list
shape (``path_depth`` = BFS level).

Module layout:

* :mod:`._pending` -- the :class:`PendingCallTarget` dataclass + per-row
  mask construction + run-length staging, plus
  :func:`finalise_pending_call_targets`.
* :mod:`._resolve` -- the per-(parent variant, call_target slot)
  J-resolution (UNCHANGED from the legacy walker, minus the now-shared
  visited / inlining gates).
* :mod:`._walker` -- the level-synchronous BFS driving the shared
  inclusion decider, the public
  :func:`walk_section_callees_pending` entry point, and the
  single-variant convenience wrapper :func:`walk_callees`.
"""

from ._pending import (
    PendingCallTarget,
    build_pending_call_target,
    finalise_pending_call_targets,
)
from ._resolve import (
    ResolvedCalleeMeta,
    load_callee_body,
    resolve_callee_metadata,
)
from ._walker import walk_callees, walk_section_callees_pending


__all__ = [
    "PendingCallTarget",
    "ResolvedCalleeMeta",
    "build_pending_call_target",
    "finalise_pending_call_targets",
    "load_callee_body",
    "resolve_callee_metadata",
    "walk_callees",
    "walk_section_callees_pending",
]
