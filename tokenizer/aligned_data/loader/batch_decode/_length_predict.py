"""Stage 2 stub — length predict + cutoff walk + surviving-count predictions.

Operates over ``Stage1Batch`` and produces ``Stage2Batch`` (D9) with the
4-level mirror.

Per level-4 call target (across the whole batch in DFS encounter order):

1. Build ``expanded_token_ids`` by promoting multi-chunk sources in a working
   copy of ``state.raw_tokens`` then stripping + shifting:
     - VC2 sources: positions with ``raw_tokens[p] == VC2_VOCAB_ID & real_mask``;
       per source ``chunk_count = max(1, ceil(state.runlen_number[p+1] / 8))``;
       paint ``working_tokens[p+1 : p+chunk_count] = VC2_VOCAB_ID``.
     - F128 sources: via ALG-2 (15-bit big-endian exponent NaN/Inf detection);
       finite source ``chunk_count = 2`` (paint ``working_tokens[p+1]``);
       NaN/Inf source ``chunk_count = 1`` (no painting).
     - Strip + shift: ``(working_tokens[working_tokens > 256] - 256).astype(u16)``.
     - Prepend self-token: prefix with ``encounter_category_token_id_shifted``.
2. Build ``extra_value_v2_mask`` + ``extra_f128_mask`` over the resulting
   ``expanded_token_ids`` marking the promoted slots.

Per level-3 variant in section-then-variant order:

3. Cumsum ``predicted_full_length`` over the variant's call_targets. The first
   call_target whose cumsum ≥ ``context_len`` is the **cut** entry;
   ``partial_cut_length = context_len − cumsum_before``. Entries after are
   dropped (counts = 0); entries before are fully included.
4. Per surviving call_target compute ``surviving_identity_count`` +
   ``surviving_number_chunk_count`` via masks on
   ``expanded_token_ids[:partial_cut_length]`` (D8 + D2's cut rule).
5. Aggregate per-variant totals; per-batch row totals (indexed by
   ``stage1_variant.batch_idx``) → cumsum into ``identity_row_offsets`` +
   ``number_row_offsets`` at level 1.

Body is intentionally a ``NotImplementedError`` — Phase 2 subagents fill this
in. See ``batch_decode_plan.md`` ``Stage 2: length + sidecar-size prediction +
cutoff walk`` + ALG-2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Stage1Batch, Stage2Batch


def predict_lengths(
    stage1: "Stage1Batch",
    *,
    context_len: int,
) -> "Stage2Batch":
    """Stage 2: ``expanded_token_ids`` + ``extra_*_mask`` construction (ALG-2
    for F128 NaN/Inf detection) + per-variant cutoff walk + surviving-count
    predictions.

    Produces ``Stage2Batch`` with ``identity_row_offsets`` +
    ``number_row_offsets`` cumsumed across the batch.
    """

    raise NotImplementedError(
        "Phase 2 — see batch_decode_plan.md '## Stages — algorithm sketch' "
        "Stage 2 + ALG-2."
    )
