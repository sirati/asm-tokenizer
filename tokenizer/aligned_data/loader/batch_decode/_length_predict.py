"""Stage 2 orchestrator -- length predict + cutoff walk + surviving counts.

Single concern: compose 2a (:mod:`._expand_tokens`), 2b
(:mod:`._cutoff_walk`), and 2c (:mod:`._surviving_counts`) into a
:class:`Stage2Batch`. This module owns nothing algorithmic; it walks the
``Stage1Batch`` 4-level hierarchy in DFS order and threads the per-call-
target / per-variant outputs of the three sibling modules onto the
matching Stage2 dataclasses.

Walk (per :class:`Stage1Section` -> :class:`Stage1Variant` ->
:class:`Stage1CallTarget`, in section-then-variant-then-DFS order):

1. ``expand_tokens(call_target)`` (2a) -> :class:`ExpandedTokens` per
   call_target; the ``predicted_full_length`` feeds 2b.
2. ``walk_cutoff_batched(lengths, variant_offsets, context_lens)`` (2b)
   -> batched :class:`CutoffColumns` (per-variant cut idx + flat
   per-call-target surviving token counts + is_cut flags) in ONE
   segmented-cumsum pass over every variant at once.
3. ``count_surviving(expanded_token_ids, partial_cut_length)`` (2c) ->
   per-call-target identity / number-chunk counts on the surviving
   prefix.
4. Assemble per-call-target :class:`Stage2CallTarget`, per-variant
   :class:`Stage2Variant` with aggregated totals, per-section
   :class:`Stage2Section`.
5. Walk ``stage1.batch_idx_to_section_variant`` row-by-row to build the
   per-row cumsum arrays. Padding rows (``UINT32_MAX`` sentinel)
   contribute 0; multi-mapped variants (RESAMPLE / REDISTRIBUTE may
   point distinct batch rows at the same ``(section_idx, slot_v)``) get
   the same per-variant total for every row that aims at them. The
   final ``identity_row_offsets[i+1] = identity_row_offsets[i] +
   row_sum_at_i`` (and similarly for ``number_row_offsets``); both
   arrays have shape ``(batch_size + 1,)`` per :class:`Stage2Batch`.

See ``batch_decode_plan.md`` ``## Stages -- algorithm sketch`` Stage 2
+ ALG-2 / ALG-10. The cut convention is documented on
:mod:`._cutoff_walk`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from ._cutoff_walk import CutoffColumns, walk_cutoff_batched
from ._expand_tokens import ExpandedTokens, expand_tokens
from ._row_expand import (
    build_per_row_variant_lookup,
    row_offsets_from_per_variant_lengths,
)
from ._surviving_counts import count_surviving
from ._types import (
    Stage1Batch,
    Stage1Variant,
    Stage2Batch,
    Stage2CallTarget,
    Stage2Section,
    Stage2Variant,
)

if TYPE_CHECKING:
    pass


__all__ = ["predict_lengths"]


def _available_for_call_targets(
    stage1_variant: Stage1Variant,
    context_len: int,
) -> int:
    """Per-variant call_target budget after the variant-token prefix.

    The row-level ``variant_tokens`` prefix (one per variant; statically
    encoded; emitted by Stage 4 before any call_target body) consumes its
    length share of the row budget. The cutoff walk sees the reduced
    budget ``context_len - n_variant_tokens`` so per-call-target cutoff
    math stays in per-call-target coordinates -- the prefix never enters
    per-call-target token streams. A prefix that alone exceeds the budget
    clamps to ``0`` (every call_target fully dropped; Stage 4's row writer
    caps the prefix via the ``context_len`` column budget).
    """
    n_variant_tokens = int(stage1_variant.variant_tokens.shape[0])
    available = context_len - n_variant_tokens
    return available if available >= 0 else 0


def _build_variant(
    stage1_variant: Stage1Variant,
    expansions: List[ExpandedTokens],
    cutoff: CutoffColumns,
    variant_idx: int,
    ct_offset: int,
) -> Stage2Variant:
    """Compose 2b + 2c into a :class:`Stage2Variant` for ONE variant.

    ``expansions`` is parallel to ``stage1_variant.call_targets`` (one
    entry per call_target, in stage-1 DFS encounter order). The
    per-call-target surviving counts (full-included, partial, or dropped)
    are read out of the batched :class:`CutoffColumns` columns at this
    variant's flat slice ``ct_offset : ct_offset + len(call_targets)``;
    they feed the surviving-count count. ``variant_idx`` selects the
    per-variant cut-index scalar.
    """

    call_targets: List[Stage2CallTarget] = []
    total_surviving_token_count = 0
    total_surviving_identity_count = 0
    total_surviving_number_chunk_count = 0

    for idx, (stage1_ct, expansion) in enumerate(
        zip(stage1_variant.call_targets, expansions)
    ):
        surviving_token_count = int(
            cutoff.surviving_token_counts[ct_offset + idx]
        )
        is_cut = bool(cutoff.is_cut_flags[ct_offset + idx])
        # partial_cut_length mirrors surviving_token_count per the
        # Stage2CallTarget field doc: "= surviving_token_count" -- and
        # also matches the count_surviving slicing semantic.
        partial_cut_length = surviving_token_count

        survivors = count_surviving(
            expansion.expanded_token_ids,
            partial_cut_length=partial_cut_length,
        )

        call_targets.append(
            Stage2CallTarget(
                stage1=stage1_ct,
                expanded_token_ids=expansion.expanded_token_ids,
                extra_value_v2_mask=expansion.extra_value_v2_mask,
                extra_f128_mask=expansion.extra_f128_mask,
                predicted_full_length=expansion.predicted_full_length,
                surviving_token_count=surviving_token_count,
                surviving_identity_count=survivors.surviving_identity_count,
                surviving_number_chunk_count=(
                    survivors.surviving_number_chunk_count
                ),
                is_cut=is_cut,
                partial_cut_length=partial_cut_length,
            )
        )

        total_surviving_token_count += surviving_token_count
        total_surviving_identity_count += survivors.surviving_identity_count
        total_surviving_number_chunk_count += (
            survivors.surviving_number_chunk_count
        )

    return Stage2Variant(
        stage1=stage1_variant,
        call_targets=call_targets,
        cut_call_target_index=int(
            cutoff.cut_call_target_index[variant_idx]
        ),
        total_surviving_token_count=total_surviving_token_count,
        total_surviving_identity_count=total_surviving_identity_count,
        total_surviving_number_chunk_count=total_surviving_number_chunk_count,
    )


def _build_row_offsets(
    stage1: Stage1Batch,
    stage2_sections: List[Stage2Section],
) -> tuple[np.ndarray, np.ndarray]:
    """Cumsum per-row surviving counts via ``batch_idx_to_section_variant``.

    Walks the canonical ``(section_idx, slot_v)`` mapping in
    :class:`Stage1Batch` -- the multi-row mapping case (RESAMPLE /
    REDISTRIBUTE may point several batch rows at the same
    ``Stage1Variant`` slot) is handled naturally: each row reads the
    same :class:`Stage2Variant` totals. Padding rows
    (``UINT32_MAX`` sentinel) contribute 0.

    Vectorized via :mod:`._row_expand` -- the per-variant scalar totals
    are gathered into flat arrays (one entry per unique variant in
    section -> slot order), then expanded to per-row via the shared
    lookup + cumsum primitive. No Python loop over ``batch_size``.

    Returns ``(identity_row_offsets, number_row_offsets)``; both have
    shape ``(batch_size + 1,)`` and dtype ``u32``. ``offsets[0]`` is
    always 0; ``offsets[i+1] = offsets[i] + row_sum_at_i``.
    """

    # Per-unique-variant flat arrays of surviving counts; section ->
    # slot order matches ``build_per_row_variant_lookup``'s flat index.
    per_variant_identity: List[int] = []
    per_variant_number: List[int] = []
    variants_per_section: List[int] = []
    for stage2_section in stage2_sections:
        variants_per_section.append(len(stage2_section.variants))
        for stage2_variant in stage2_section.variants:
            per_variant_identity.append(
                stage2_variant.total_surviving_identity_count
            )
            per_variant_number.append(
                stage2_variant.total_surviving_number_chunk_count
            )

    per_row_variant_idx, is_padding = build_per_row_variant_lookup(
        stage1.batch_idx_to_section_variant, variants_per_section
    )

    identity_lengths = np.array(per_variant_identity, dtype=np.uint32)
    number_lengths = np.array(per_variant_number, dtype=np.uint32)
    identity_row_offsets = row_offsets_from_per_variant_lengths(
        identity_lengths, per_row_variant_idx, is_padding
    )
    number_row_offsets = row_offsets_from_per_variant_lengths(
        number_lengths, per_row_variant_idx, is_padding
    )
    return identity_row_offsets, number_row_offsets


def predict_lengths(
    stage1: Stage1Batch,
    *,
    context_len: int,
) -> Stage2Batch:
    """Stage 2: ``expanded_token_ids`` + ``extra_*_mask`` construction
    (ALG-2 for F128 NaN/Inf detection) + per-variant cutoff walk +
    surviving-count predictions.

    Produces a :class:`Stage2Batch` with ``identity_row_offsets`` +
    ``number_row_offsets`` cumsumed across the batch.

    Parameters
    ----------
    stage1
        :class:`Stage1Batch` produced by stage 1
        (:mod:`._section_walk`). The 4-level hierarchy is walked in
        DFS order; each call_target is fed to ``expand_tokens``.
    context_len
        Per-row cutoff budget in tokens. Must be ``>= 0``. Reduced by
        each variant's variant-token prefix, then fed to the batched
        cutoff walk (:func:`._cutoff_walk.walk_cutoff_batched`).

    Returns
    -------
    Stage2Batch
        Fully populated 4-level mirror with per-row cumsum offsets.
    """

    # 2a: per-call-target expansion, materialised for the whole batch (we
    # need each ExpandedTokens twice: once for predicted-length in 2b,
    # once for the surviving-count masking in 2c). Collected flat in
    # variant-then-DFS order so the cutoff walk (2b) runs as ONE batched
    # segmented-cumsum pass over every variant at once.
    per_variant: List[tuple[Stage1Variant, List[ExpandedTokens]]] = []
    flat_lengths: List[int] = []
    variant_offsets: List[int] = [0]
    context_lens: List[int] = []
    for stage1_section in stage1.sections:
        for stage1_variant in stage1_section.variants:
            expansions = [
                expand_tokens(ct) for ct in stage1_variant.call_targets
            ]
            per_variant.append((stage1_variant, expansions))
            flat_lengths.extend(ex.predicted_full_length for ex in expansions)
            variant_offsets.append(len(flat_lengths))
            context_lens.append(
                _available_for_call_targets(stage1_variant, context_len)
            )

    cutoff = walk_cutoff_batched(
        np.asarray(flat_lengths, dtype=np.int64),
        np.asarray(variant_offsets, dtype=np.int64),
        np.asarray(context_lens, dtype=np.int64),
    )

    # Assemble the 4-level mirror, reading each variant's cut geometry out
    # of the batched columns at its flat slice.
    variant_iter = iter(per_variant)
    stage2_sections: List[Stage2Section] = []
    variant_idx = 0
    for stage1_section in stage1.sections:
        stage2_variants: List[Stage2Variant] = []
        for _ in stage1_section.variants:
            stage1_variant, expansions = next(variant_iter)
            stage2_variants.append(
                _build_variant(
                    stage1_variant,
                    expansions,
                    cutoff,
                    variant_idx,
                    int(variant_offsets[variant_idx]),
                )
            )
            variant_idx += 1
        stage2_sections.append(
            Stage2Section(stage1=stage1_section, variants=stage2_variants)
        )

    identity_row_offsets, number_row_offsets = _build_row_offsets(
        stage1, stage2_sections
    )

    return Stage2Batch(
        stage1=stage1,
        sections=stage2_sections,
        identity_row_offsets=identity_row_offsets,
        number_row_offsets=number_row_offsets,
    )
