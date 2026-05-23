"""Per-variant cutoff walk (Stage 2, phase 2b).

Owns ONE concern: given a list of per-call-target predicted full lengths
(root + each inlined callee, in stage-1 DFS encounter order) and a row
``context_len`` budget, identify the cut call-target index and the per-
call-target surviving token counts.

This module is deliberately pure-Python and token-stream agnostic: it
operates on a list of integer lengths plus a budget, and returns the cut
geometry. The token-level masks (identity / number) live in the
surviving-count predictor (2c); the orchestrator (2d) wires this output
together with the expanded-tokens result of 2a to populate the per-
:class:`Stage2CallTarget` ``surviving_token_count`` + ``is_cut`` +
``partial_cut_length`` fields and the :class:`Stage2Variant.
cut_call_target_index` field.

See ``batch_decode_plan.md`` D8 + "Stage 2: length + sidecar-size
prediction + cutoff walk" step 4.

Cut convention (the design point that fixes the exact-boundary case):

The cut call target is ``cut_idx = smallest K such that
cumsum[K] > context_len``. Equivalently: the cut call target is the
first one whose body would be sliced off mid-stream. When the boundary
falls exactly between two call_targets (``cumsum[K] == context_len`` for
some K), ``K+1`` becomes the cut entry with ``partial_cut_length = 0``
(its surviving count is 0; it is "dropped at offset 0"). This keeps the
cut-entry invariant uniform: a cut entry's
``0 <= partial_cut_length < predicted_full_lengths[cut_idx]``; if the
partial equalled the full length we would call the entry "fully
included" instead.

When the full variant fits (sum of lengths <= context_len), there is
no cut: ``cut_call_target_index = len(predicted_full_lengths)`` (sentinel
past-the-end) and ``partial_cut_length = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "CutoffResult",
    "walk_cutoff",
]


@dataclass(frozen=True)
class CutoffResult:
    """2b's output for one Stage1Variant.

    Consumed by 2d (the ``_length_predict.py`` wiring step) to set per-
    :class:`Stage2CallTarget` ``surviving_token_count`` + ``is_cut`` +
    ``partial_cut_length`` fields and the
    :class:`Stage2Variant.cut_call_target_index` field.
    """

    cut_call_target_index: int
    """Index of the first call_target whose body is sliced off mid-
    stream (= first ``K`` with ``cumsum[K] > context_len``). Equals
    ``len(predicted_full_lengths)`` when no cut is needed (the full
    variant fits)."""

    partial_cut_length: int
    """Surviving tokens IN the cut call_target. ``0`` when no cut is
    needed (sentinel ``cut_call_target_index`` points past the end);
    otherwise ``context_len - cumsum_tokens_before_cut`` and satisfies
    ``0 <= partial_cut_length < predicted_full_lengths[cut_call_target_index]``."""

    surviving_token_counts: list[int]
    """Per call_target: ``predicted_full_lengths[i]`` if ``i <
    cut_call_target_index`` (fully included); ``partial_cut_length`` if
    ``i == cut_call_target_index`` (cut); ``0`` otherwise (dropped)."""

    is_cut_flags: list[bool]
    """Per call_target: ``True`` iff this entry is the cut call_target
    (i.e. ``i == cut_call_target_index`` AND a cut occurred)."""


def walk_cutoff(
    predicted_full_lengths: list[int],
    context_len: int,
) -> CutoffResult:
    """Walk per-call-target predicted lengths, find the cut point, and
    return per-call-target surviving counts.

    See ``batch_decode_plan.md`` "## Stages — algorithm sketch" Stage 2
    step 4 + D8. The cut convention is documented at module level.

    Parameters
    ----------
    predicted_full_lengths
        Per-call-target post-promotion token-stream lengths, IN
        ENCOUNTER ORDER (root at index 0, then callees in DFS encounter
        order — same order as ``Stage1Variant.call_targets``).
    context_len
        The per-row cutoff budget in tokens. Must be ``>= 0``.

    Returns
    -------
    CutoffResult
        Per-call-target surviving counts + the cut-call-target index +
        partial-cut length.
    """

    n = len(predicted_full_lengths)
    cumsum_before = 0
    cut_idx = n  # sentinel: no cut needed
    partial_cut_length = 0

    for i, length in enumerate(predicted_full_lengths):
        cumsum_after = cumsum_before + length
        if cumsum_after > context_len:
            cut_idx = i
            partial_cut_length = context_len - cumsum_before
            break
        cumsum_before = cumsum_after

    surviving_token_counts = [
        predicted_full_lengths[i] if i < cut_idx
        else (partial_cut_length if i == cut_idx else 0)
        for i in range(n)
    ]
    is_cut_flags = [i == cut_idx and cut_idx < n for i in range(n)]

    return CutoffResult(
        cut_call_target_index=cut_idx,
        partial_cut_length=partial_cut_length,
        surviving_token_counts=surviving_token_counts,
        is_cut_flags=is_cut_flags,
    )
