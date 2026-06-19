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

import numpy as np


__all__ = [
    "CutoffResult",
    "CutoffColumns",
    "walk_cutoff",
    "walk_cutoff_batched",
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


@dataclass(frozen=True)
class CutoffColumns:
    """Batched columnar twin of :class:`CutoffResult` over MANY variants.

    Single concern: the cut geometry for every call_target in a batch,
    held as flat columns (one entry per call_target, in the flat DFS
    encounter order the caller supplies) plus the per-variant scalars.
    This is the surviving-clip foundation the dense column build (the
    object-tree-elimination plan, step 2) consumes; it is produced by ONE
    numpy-C segmented-cumsum pass, NOT a per-call-target Python loop.

    The columns are byte-for-byte the concatenation of looping
    :func:`walk_cutoff` over each variant's lengths slice (in variant
    order), and the per-variant scalars equal that loop's per-variant
    :class:`CutoffResult` scalars.
    """

    surviving_token_counts: np.ndarray
    """``int64[n_call_targets]`` -- per call_target surviving prefix
    length (full length if before the cut, partial at the cut, ``0``
    after). Equals the concatenated per-variant
    :attr:`CutoffResult.surviving_token_counts`."""

    is_cut_flags: np.ndarray
    """``bool[n_call_targets]`` -- ``True`` exactly at each variant's cut
    call_target (and nowhere when the variant fits). Equals the
    concatenated per-variant :attr:`CutoffResult.is_cut_flags`."""

    cut_call_target_index: np.ndarray
    """``int64[n_variants]`` -- per-variant LOCAL index of the cut
    call_target, or the variant's call_target count (sentinel) when it
    fits. Equals the per-variant
    :attr:`CutoffResult.cut_call_target_index`."""

    partial_cut_length: np.ndarray
    """``int64[n_variants]`` -- per-variant surviving tokens in the cut
    call_target, ``0`` when no cut. Equals the per-variant
    :attr:`CutoffResult.partial_cut_length`."""


def _cutoff_columns(
    lengths: np.ndarray,
    variant_offsets: np.ndarray,
    context_lens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised cut geometry for a flat, variant-segmented length stream.

    Single source of truth for the cut convention documented at module
    level. Both :func:`walk_cutoff` (one variant) and
    :func:`walk_cutoff_batched` (many variants) route through here so the
    cut logic is never duplicated.

    Parameters
    ----------
    lengths:
        ``int64[n_ct]`` -- per-call-target predicted full lengths, flat,
        in variant-then-DFS order.
    variant_offsets:
        ``int64[n_variants + 1]`` -- CSR jump table: variant ``v`` owns
        ``lengths[variant_offsets[v] : variant_offsets[v + 1]]``.
    context_lens:
        ``int64[n_variants]`` -- per-variant budget (the already
        prefix-adjusted ``available_for_call_targets``).

    Returns
    -------
    tuple
        ``(surviving, is_cut, cut_idx_per_variant, partial_per_variant)``
        -- the two ``[n_ct]`` columns plus the two ``[n_variants]``
        per-variant scalar arrays.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    variant_offsets = np.asarray(variant_offsets, dtype=np.int64)
    context_lens = np.asarray(context_lens, dtype=np.int64)

    n_ct = lengths.shape[0]
    n_variants = variant_offsets.shape[0] - 1
    counts = np.diff(variant_offsets)  # per-variant call_target count

    if n_ct == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.bool_),
            counts.astype(np.int64),  # all-sentinel (each = its 0 count)
            np.zeros(n_variants, dtype=np.int64),
        )

    # Owning variant per call_target via repeat-over-segment-LENGTHS (the
    # #92 discipline: never mark-and-cumsum, or consecutive empty variants
    # would merge and shift every later id).
    ct_variant = np.repeat(np.arange(n_variants, dtype=np.int64), counts)

    # Inclusive + exclusive GLOBAL prefix sums; subtract the per-variant
    # base to get the per-variant-LOCAL exclusive prefix the cut math
    # uses (cumsum resets at each variant boundary). The exclusive prefix
    # is gathered at each variant's START offset; we index a copy padded
    # with one trailing entry so an empty trailing variant (start == n_ct)
    # gathers harmlessly -- its base is never read (no call_target maps to
    # it via ``ct_variant``).
    cumsum_after_global = np.cumsum(lengths)
    cumsum_before_global = cumsum_after_global - lengths
    cumsum_before_padded = np.concatenate(
        [cumsum_before_global, cumsum_before_global[-1:]]
    )
    variant_base = cumsum_before_padded[variant_offsets[:-1]]
    cumsum_before = cumsum_before_global - variant_base[ct_variant]
    cumsum_after = cumsum_before + lengths

    budget = context_lens[ct_variant]

    # surviving[i] = clip(budget - cumsum_before, 0, length): the whole
    # length before the cut (budget-cumsum_before >= length), the partial
    # AT the cut (0 <= budget-cumsum_before < length), 0 after (negative
    # clamps to 0). This is exactly the scalar walk's three-way result.
    surviving = np.clip(budget - cumsum_before, 0, lengths)

    # Overflow mask + per-variant "first overflow" selection. The cut is
    # the FIRST call_target whose inclusive cumsum strictly exceeds the
    # budget (strict ``>`` realises the exact-boundary convention: an
    # exact fill, or a zero-length tail after one, does NOT cut).
    over = cumsum_after > budget
    excl_over_global = np.cumsum(over) - over
    excl_over_padded = np.concatenate(
        [excl_over_global, excl_over_global[-1:]]
    )
    over_base = excl_over_padded[variant_offsets[:-1]]
    excl_over = excl_over_global - over_base[ct_variant]
    is_cut = over & (excl_over == 0)

    # Per-variant scalars. Default cut index = call_target count
    # (no-cut sentinel); overwrite at the cut with its LOCAL index.
    cut_idx_per_variant = counts.astype(np.int64).copy()
    partial_per_variant = np.zeros(n_variants, dtype=np.int64)
    cut_positions = np.flatnonzero(is_cut)
    if cut_positions.size:
        cut_variants = ct_variant[cut_positions]
        local_idx = cut_positions - variant_offsets[cut_variants]
        cut_idx_per_variant[cut_variants] = local_idx
        partial_per_variant[cut_variants] = surviving[cut_positions]

    return surviving, is_cut, cut_idx_per_variant, partial_per_variant


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
    lengths = np.asarray(predicted_full_lengths, dtype=np.int64)
    # One-variant framing: a single segment spanning all lengths, with the
    # whole budget. Routes through the shared columnar core so the cut
    # convention is defined in exactly one place.
    surviving, is_cut, cut_idx_per_variant, partial_per_variant = (
        _cutoff_columns(
            lengths,
            variant_offsets=np.array([0, n], dtype=np.int64),
            context_lens=np.array([context_len], dtype=np.int64),
        )
    )

    return CutoffResult(
        cut_call_target_index=int(cut_idx_per_variant[0]),
        partial_cut_length=int(partial_per_variant[0]),
        surviving_token_counts=[int(x) for x in surviving],
        is_cut_flags=[bool(x) for x in is_cut],
    )


def walk_cutoff_batched(
    predicted_full_lengths: np.ndarray,
    variant_offsets: np.ndarray,
    context_lens: np.ndarray,
) -> CutoffColumns:
    """Batched, columnar cutoff walk over MANY variants in one numpy pass.

    Byte-for-byte equivalent to looping :func:`walk_cutoff` over each
    variant's lengths slice (in variant order) and concatenating the
    per-call-target ``surviving_token_counts`` / ``is_cut_flags`` columns
    (with the per-variant scalars collected alongside). No per-call-target
    or per-variant Python loop -- the whole batch is one segmented
    cumsum / clip (the object-tree-elimination plan, KEYSTONE step 1).

    Parameters
    ----------
    predicted_full_lengths:
        ``int64[n_call_targets]`` -- per-call-target predicted full
        lengths, flat, in variant-then-DFS encounter order (variant ``v``
        owns the slice ``variant_offsets[v] : variant_offsets[v + 1]``).
    variant_offsets:
        ``int64[n_variants + 1]`` -- CSR jump table into
        ``predicted_full_lengths``. ``variant_offsets[0] == 0`` and
        ``variant_offsets[-1] == n_call_targets``.
    context_lens:
        ``int64[n_variants]`` -- the per-variant budget each
        :func:`walk_cutoff` call would receive (already adjusted for the
        variant-token prefix, i.e. the ``available_for_call_targets``
        the per-variant orchestrator computes).

    Returns
    -------
    CutoffColumns
        The two flat per-call-target columns plus the per-variant cut
        index + partial-cut-length scalar arrays.
    """
    surviving, is_cut, cut_idx_per_variant, partial_per_variant = (
        _cutoff_columns(
            predicted_full_lengths, variant_offsets, context_lens
        )
    )
    return CutoffColumns(
        surviving_token_counts=surviving,
        is_cut_flags=is_cut,
        cut_call_target_index=cut_idx_per_variant,
        partial_cut_length=partial_per_variant,
    )
