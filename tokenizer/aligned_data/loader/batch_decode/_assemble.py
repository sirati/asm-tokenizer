"""Stage 4 stub — assemble flat output + counter remap + variant padding.

Operates over ``Stage3Batch`` and produces ``BatchDecodeResult`` (D9). The
only stage producing a non-hierarchical output; the hierarchical
``Stage3Batch`` may optionally be carried alongside via ``keep_intermediate=True``.

For each level-3 variant whose ``stage1.batch_idx is not None``:

1. Initialize per-Category dedup state per ALG-3 (LOCAL_FUNC seeded with
   ``{root.function_name_ptr: 0}``; ``next_fresh_id = 1``. PLT_FUNC /
   EXT_FUNC empty; ``next_fresh_id = 0``).
2. For each level-4 call_target in encounter order whose
   ``surviving_token_count > 0``:
     - Write prepend per ALG-9: token id to ``tokens[row, prepend_pos]``;
       counter id (``self_counter`` = the dedup_dict[encounter_category] entry
       for this call_target's ``function_name_ptr``) to
       ``identities_flat_caller_local[identity_slice.start]``.
     - For each FUNCTION Category present in this call_target's
       ``call_targets_section``: run ALG-3 dedup → batch-lookup the existing
       counter ids, mint fresh ids for misses (hole-free per D4), apply to
       the call_target's identity slice (skipping the prepend slot).
     - For each COUNTER Category present: run ALG-4 offset bump (running
       per-Category offset across the row).
3. Write ``call_target.expanded_token_ids[:partial_cut_length]`` into
   ``tokens[row]`` at the running column offset; concatenate root + each
   callee; truncate at exactly ``context_len`` (D2 — mid-multi-chunk allowed).
   Trailing positions stay at id 0 (null-content).
4. Concatenate per-TokenType ``(significand, sign_exp)`` pairs into the global
   ``numbers_significant`` + ``numbers_sign_exponent`` in row-order
   interleaved by ``expanded_token_ids[:partial_cut_length]`` walk; NO
   renormalization at stage 4.
5. If ``include_fid_sidecar=True``: build ``fid_sidecar`` from each
   per-Category dedup map (reverse mapping counter_id → function_name_ptr),
   with per-row offsets.

Variant-padding policy enforcement is upstream at level 1 via stage 1's
``batch_idx_to_section_variant`` — padding rows are skipped at step 1 and
their tokens stay id 0 (null-content); their sidecar offsets stay equal to
the prior offset (zero-length slice).

Body is intentionally a ``NotImplementedError`` — Phase 4 subagents fill this
in. See ``batch_decode_plan.md`` ``Stage 4`` + ALG-3 + ALG-4 + ALG-9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import BatchDecodeResult, Stage3Batch


def assemble_batch(
    stage3: "Stage3Batch",
    *,
    context_len: int,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
) -> "BatchDecodeResult":
    """Stage 4: per-row dedup walk (ALG-3 FUNCTION categories + ALG-4 COUNTER
    categories) + prepend slot writes (ALG-9) + token tensor assembly with
    truncation at ``context_len`` + sidecar concat + ``VariantPadding`` policy
    enforcement.

    Produces ``BatchDecodeResult``.
    """

    raise NotImplementedError(
        "Phase 4 — see batch_decode_plan.md '## Stages — algorithm sketch' "
        "Stage 4 + ALG-3 + ALG-4 + ALG-9."
    )
