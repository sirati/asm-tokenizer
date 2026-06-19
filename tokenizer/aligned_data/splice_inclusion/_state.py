"""Persistent once-only + all-variants-equivalence inclusion state.

Single concern: decide, per root, which ``(variant, function_id)`` pairs
are INCLUDED at each splice level and which function_ids SURVIVE to the
next level -- enforcing the owner's two rules:

* **once-only per root.** A variant's expansion includes a function's
  body on its FIRST encounter only (the mask cell's False->True
  transition). Self-recursion, mutual recursion, diamonds, and
  root+branch-shared callees all dedup for free: the function_id->column
  hashmap is cumulative across levels AND across parents within a level.
* **all-variants equivalence.** After a level marks every variant's
  resolved callees, a columnwise ALL over the variant axis flags any
  function reached by EVERY variant at that level; such a function is
  NOT included and is PRUNED (never expands deeper). This generalizes
  the legacy ``inlined_equivalent_call_targets_only`` "called by all =>
  skip" filter into the default-and-only behavior.

This module knows NOTHING about tokens, lengths, or bodies. Its inputs
are abstract ``function_id`` integers (the BIN's ``function_name_ptr``
FID space) and a variant axis; its outputs are inclusion / survival
booleans. Both consumers -- the dataloader callee walk and the
sorted-index graph-lengths build -- resolve their own per-call J/variant
choice (unchanged), translate each resolved callee to a ``function_id``,
drive this state level by level, and apply the returned inclusion
decision to their own emission (tokens) or aggregation (lengths).

Buffer-reuse discipline (the owner's efficiency mandate): the hashmap +
mask are allocated once on the :class:`OnceOnlyInclusion` instance and
REUSED across roots. :meth:`begin_root` clears the hashmap and zeroes
only the used mask region; the mask grows geometrically and is never
re-allocated per root.

The boundary sentence (design-first rule):

  *Given a root's variant count + root function_id, then per level the
  flat ``(variant, callee_function_id)`` pairs each variant resolved,
  return a per-pair ``included`` mask + the surviving callee
  function_ids (with the variant that first reached each), enforcing
  once-only-per-root and columnwise-ALL exclusion.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dedup_hashmap import OnceOnlyInclusionKernel


__all__ = ["LevelResult", "OnceOnlyInclusion"]


@dataclass(frozen=True)
class LevelResult:
    """One level's inclusion + survival decision.

    Lazy view over the caller-supplied pair arrays: the boolean arrays
    are fresh per call but index-aligned to the ``variant`` /
    ``callee_function_id`` pair arrays the caller passed to
    :meth:`OnceOnlyInclusion.step_level`.
    """

    included: np.ndarray
    """``bool[n_pairs]`` -- True where this ``(variant, function_id)``
    pair includes the function's body at this level (first encounter for
    the variant AND not excluded by the columnwise-ALL test)."""

    survivor_pairs: np.ndarray
    """``int64[n_survivors]`` -- indices into the pair arrays selecting
    the ``(variant, function_id)`` pairs that expand at the NEXT level.
    Survival is per-variant: a pair survives iff it was included (first
    encounter for THIS variant AND not excluded by the columnwise-ALL
    test). Each surviving pair expands with ITS OWN variant's callee
    choice next level -- the once-only mask is shared across variants,
    but the descent is per-variant (the owner's ``[#variants,
    #calltargets]`` mask stays variant-indexed at every level)."""


class OnceOnlyInclusion:
    """Reusable per-root once-only / equivalence inclusion decider.

    One instance is driven through many roots; :meth:`begin_root` resets
    the per-root state in place (no re-allocation). Within a root,
    :meth:`step_level` is called once per splice level in increasing
    depth order, threading the previous level's survivors as the next
    level's parents (the caller resolves each survivor's callees).

    The whole per-level state machine -- the ``function_id -> dense
    column`` map, the reused ``[n_variants, n_cols]`` inclusion mask, and
    the columnwise-ALL FLAG-A / read-before-write FLAG-B decision -- is the
    GIL-released :class:`~dedup_hashmap.OnceOnlyInclusionKernel`. This class
    is the thin Python facade that owns the kernel and translates the
    ndarray pair arrays at the boundary; the buffer-reuse discipline (mask
    grown geometrically, zeroed not re-allocated per root) lives in the
    kernel's Rust-local state. Both consumers -- the loader inclusion-BFS
    and the sorted-index length build -- drive THIS one class, which holds
    ONE kernel, so their inclusion semantics can never drift.
    """

    def __init__(self, *, initial_cols: int = 64) -> None:
        # The GIL-released Rust decider owns the fid->col map + reused mask
        # as Rust-local state; this facade holds the single instance.
        self._kernel = OnceOnlyInclusionKernel(max(1, initial_cols))

    # -- per-root lifecycle ------------------------------------------------

    def begin_root(self, n_variants: int, root_function_id: int) -> int:
        """Reset for a new root; seed the root body at column 0.

        The root body is always included exactly once as the root, so any
        deeper call resolving to ``root_function_id`` is already-included
        (self / mutual recursion never re-splices). Returns the root's
        column (always 0). Delegates to the kernel, which clears the
        fid->col map and zeroes only the previously-used mask region (the
        mask is grown, never shrunk, to fit ``n_variants`` rows).
        """
        return int(
            self._kernel.begin_root(int(n_variants), np.uint32(root_function_id))
        )

    # -- per-level step ----------------------------------------------------

    def step_level(
        self,
        variant: np.ndarray,
        callee_function_id: np.ndarray,
    ) -> LevelResult:
        """Resolve one level's pairs into inclusion + survival.

        Parameters
        ----------
        variant:
            ``int[n_pairs]`` -- the variant (mask row) each resolved
            call belongs to. Multiple pairs may share a variant (one per
            surviving parent's call_target slot).
        callee_function_id:
            ``int[n_pairs]`` -- the resolved callee FID per pair.

        Both arrays are parallel; ordering is the caller's emission
        order (the returned ``included`` mask is index-aligned to it).

        The per-level decision runs in the GIL-released kernel; this
        method only normalises the input dtypes at the boundary and wraps
        the kernel's ``(included, survivor_pairs)`` pair into a
        :class:`LevelResult`. The kernel preserves the owner's spec
        verbatim: dense-column assignment (ascending-FID new columns),
        the pre-mark snapshot (FLAG-B read-before-write), the
        first-in-level dedup (earliest emission wins), the columnwise-ALL
        exclusion over the touched columns (FLAG-A), and the ascending
        survivor index.
        """
        variant = np.asarray(variant, dtype=np.int64).reshape(-1)
        fids = np.asarray(callee_function_id, dtype=np.uint32).reshape(-1)
        if variant.shape != fids.shape:
            raise ValueError(
                "variant and callee_function_id must be parallel; got "
                f"{variant.shape} vs {fids.shape}"
            )
        included, survivor_pairs = self._kernel.step_level(variant, fids)
        return LevelResult(included=included, survivor_pairs=survivor_pairs)
