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

from dedup_hashmap import HashMapU32U32


__all__ = ["LevelResult", "OnceOnlyInclusion"]


#: ``lookup_ndarray`` miss sentinel (the U32U32 map returns 0xFFFFFFFF
#: for keys it does not contain).
_U32_MISS = np.uint32(0xFFFFFFFF)


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
    """

    def __init__(self, *, initial_cols: int = 64) -> None:
        # function_id -> dense column index, cleared per root.
        self._fid_to_col = HashMapU32U32(capacity=max(8, initial_cols * 2))
        # [n_variants, n_cols] inclusion mask; geometric growth, zeroed
        # (not re-allocated) per root over its used region.
        self._mask = np.zeros((1, max(1, initial_cols)), dtype=bool)
        self._n_variants = 0
        self._n_cols = 0

    # -- per-root lifecycle ------------------------------------------------

    def begin_root(self, n_variants: int, root_function_id: int) -> int:
        """Reset for a new root; seed the root body at column 0.

        ``mask[:, 0] = True`` for every variant: the root body is always
        included exactly once as the root, so any deeper call resolving
        to ``root_function_id`` is already-included (self / mutual
        recursion never re-splices). Returns the root's column (always
        0).

        Clears the hashmap and zeroes only the used mask region; the
        mask is grown (never shrunk) to fit ``n_variants`` rows.
        """
        if n_variants <= 0:
            raise ValueError(f"n_variants must be >= 1; got {n_variants}")
        self._fid_to_col.clean()
        self._ensure_capacity(n_variants, 1)
        # Zero the previously-used region, then seed root col 0.
        self._mask[: self._n_variants, : self._n_cols] = False
        self._n_variants = n_variants
        self._n_cols = 1
        self._mask[:n_variants, 0] = True
        self._fid_to_col.insert(
            np.uint32(root_function_id), np.uint32(0)
        )
        return 0

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

        Algorithm (owner's spec, per level):

        1. Map each callee FID to its dense column (insert-or-get;
           previously-unseen FIDs take fresh ascending columns).
        2. Snapshot each pair's PRE-mark cell (the once-only test reads
           the cell BEFORE this level writes it -- a function reached by
           a variant for the FIRST time at this level has a False
           pre-cell).
        3. Mark ``mask[variant, col] = True`` for every pair.
        4. Columnwise ALL over the variant axis on the columns touched
           THIS level: a column all-True across every variant is
           excluded (not included, pruned).
        5. ``included = pre_cell_false & ~excluded[col]``.
        6. Survivors = function_ids that some variant newly-included and
           that are not excluded; one representative pair each.
        """
        variant = np.asarray(variant, dtype=np.int64).reshape(-1)
        fids = np.asarray(callee_function_id, dtype=np.uint32).reshape(-1)
        if variant.shape != fids.shape:
            raise ValueError(
                "variant and callee_function_id must be parallel; got "
                f"{variant.shape} vs {fids.shape}"
            )
        n_pairs = fids.shape[0]
        if n_pairs == 0:
            empty_b = np.zeros(0, dtype=bool)
            empty_i = np.zeros(0, dtype=np.int64)
            return LevelResult(included=empty_b, survivor_pairs=empty_i)

        cols = self._assign_columns(fids)
        self._ensure_capacity(self._n_variants, self._n_cols)

        # (2) pre-mark snapshot, then (3) mark. ``pre_cell`` reads the
        # mask BEFORE this level writes it. A ``(variant, col)`` pair may
        # repeat WITHIN this level (a variant calls the same function via
        # two call_target slots, or two distinct call slots resolve to
        # the same callee); only the FIRST occurrence in emission order
        # includes the body, the rest are repeats.
        pre_cell = self._mask[variant, cols]
        first_in_level = self._first_occurrence(variant, cols)
        self._mask[variant, cols] = True

        # (4) columnwise ALL over EVERY variant, restricted to the
        # columns this level touched. A column is excluded iff all
        # ``n_variants`` rows are True.
        #
        # FLAG-A (single-decision-site, one-line flip): ``.all(axis=0)``
        # over a single-variant root's ONE row is trivially True for
        # every touched column, so single-variant sections splice
        # nothing. To make single-variant roots splice everything
        # instead, gate this exclusion on ``self._n_variants > 1``.
        touched = np.unique(cols)
        col_all = self._mask[: self._n_variants, touched].all(axis=0)
        excluded_col = np.zeros(self._n_cols, dtype=bool)
        excluded_col[touched] = col_all
        pair_excluded = excluded_col[cols]

        # (5) once-only inclusion: first encounter for this variant
        # (across prior levels AND within this level) AND not excluded by
        # the all-variants test.
        #
        # FLAG-B (single-decision-site): ``~pre_cell`` reads the mask
        # BEFORE this level's marking, so a variant reaching a function
        # LATE (column already True from earlier variants, becoming
        # all-True this level) is excluded here and does NOT include it.
        # To instead let a late variant include a function it reaches
        # before the column converges, the test would read the cell
        # value as of a prior level rather than the live ``pre_cell``.
        included = (~pre_cell) & first_in_level & (~pair_excluded)

        # (6) survivors: descent is per-variant -- every INCLUDED pair
        # expands at the next level with its own variant's callee
        # choice. (An excluded function is pruned for every variant; a
        # repeat-encounter pair already expanded at its first encounter.)
        survivor_pairs = np.nonzero(included)[0]
        return LevelResult(included=included, survivor_pairs=survivor_pairs)

    # -- internals ---------------------------------------------------------

    def _first_occurrence(
        self, variant: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        """``bool[n_pairs]`` -- True at the first ``(variant, col)`` in
        emission order, False on later repeats of the same pair.

        Within one level the same ``(variant, function)`` may appear
        through multiple call_target slots; the once-only rule includes
        the body once, on the earliest pair. Keyed on
        ``variant * n_cols + col`` (both bounded) so the dedup is a
        single stable group scan.
        """
        key = variant.astype(np.int64) * self._n_cols + cols
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        first_sorted = np.ones(sorted_key.size, dtype=bool)
        first_sorted[1:] = sorted_key[1:] != sorted_key[:-1]
        out = np.zeros(key.size, dtype=bool)
        out[order[first_sorted]] = True
        return out

    def _assign_columns(self, fids: np.ndarray) -> np.ndarray:
        """Insert-or-get dense columns for ``fids`` (last-write-safe).

        The native ``insert`` is last-write-wins, so a blind insert would
        clobber an existing column. We look up first, then assign fresh
        ascending columns ONLY to the genuinely-new FIDs (deduped within
        the batch so repeats in one level share a column).
        """
        existing = self._fid_to_col.lookup_ndarray(fids).astype(np.int64)
        miss = existing == int(_U32_MISS)
        if bool(miss.any()):
            new_fids = fids[miss]
            uniq, inverse = np.unique(new_fids, return_inverse=True)
            base = self._n_cols
            new_cols = base + np.arange(uniq.size, dtype=np.int64)
            self._fid_to_col.insert_ndarray(
                uniq, new_cols.astype(np.uint32)
            )
            existing[miss] = new_cols[inverse]
            self._n_cols = base + int(uniq.size)
        return existing

    def _ensure_capacity(self, n_variants: int, n_cols: int) -> None:
        """Grow the mask geometrically to fit ``(n_variants, n_cols)``.

        Preserves the already-marked region; growth doubles the deficient
        axis so amortised reallocation is O(1) per added row/column.
        """
        rows, mcols = self._mask.shape
        if n_variants <= rows and n_cols <= mcols:
            return
        new_rows = rows
        while new_rows < n_variants:
            new_rows *= 2
        new_cols = mcols
        while new_cols < n_cols:
            new_cols *= 2
        grown = np.zeros((new_rows, new_cols), dtype=bool)
        grown[:rows, :mcols] = self._mask
        self._mask = grown
