"""Stage 4 orchestrator -- compose the four single-concern stage-4 modules
into a :class:`BatchDecodeResult`.

This module owns ONE concern: thread the outputs of the four sibling
stage-4 modules (token assembly, per-row dedup remap walk, number sidecar
concat, fid sidecar collection) into the user-facing
:class:`BatchDecodeResult`. It contains no per-row loops or per-Category
dispatch -- every algorithmic concern lives in its dedicated module.

Boundary contract (the design-first sentence):

  *Given a finalised :class:`Stage3Batch`, produce a
  :class:`BatchDecodeResult` whose ``tokens`` tensor, ``identities``
  array, number sidecars, and optional FID sidecar are all internally
  consistent with the stage-2 row-offset cumsums.*

Composition order (none of these are mutually load-bearing -- the four
write disjoint outputs -- but a specific order is picked for
diagnosability):

1. :func:`assemble_tokens` (4c) -- writes the
   ``u16[batch_size, context_len]`` tensor. Position 0 of every
   call_target's ``expanded_token_ids`` is the prepend self-token (per
   plan ALG-9 + stage 2a's :func:`_expand_tokens.expand_tokens`), so the
   prepend's TOKEN id lands in ``tokens`` here -- no separate prepend
   write needed.
2. :func:`apply_per_row_remap` (4a) -- rewrites
   ``stage3.identities_flat_caller_local`` IN PLACE. The merged dedup +
   prepend-slot walk (see the module docstring on
   :mod:`._dedup_walk`) writes the identity-sidecar prepend slot for
   every call_target as part of the same walk, so the prepend's
   IDENTITY counter lands here too -- again no separate prepend write
   needed. Optionally returns the fid-sidecar pair when
   ``include_fid_sidecar=True``.
3. :func:`assemble_number_sidecars` (4d) -- builds the global
   ``(numbers_significant, numbers_sign_exponent)`` arrays from the
   per-:class:`TokenType` stage-3 chunks. Independent of (1) and (2);
   could equally well run first.

The two prepend writes that the stage-4 plan calls out (``tokens[row,
column]`` + ``identities_flat[identity_slice.start]``) are NOT performed
via :func:`._prepend.write_prepend_slot` -- they are produced as a
SIDE-EFFECT of (1) and (2) respectively. The prepend module remains as a
single-concern helper for callers who do not want to take the dedup walk
(e.g. unit tests pinning the ALG-9 vocab-id contract); the orchestrator
does not need it.

See ``batch_decode_plan.md`` ``## Stages -- algorithm sketch`` Stage 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from dedup_hashmap import HashMapU32U16

from tokenizer.tokens import Category

from ._dedup_walk import FUNCTION_CATEGORIES, apply_per_row_remap
from ._sidecar_concat import assemble_number_sidecars
from ._token_assembly import assemble_tokens
from ._types import BatchDecodeResult

if TYPE_CHECKING:
    from ._types import Stage3Batch


__all__ = ["assemble_batch"]


def _build_dedup_maps(stage3_batch: "Stage3Batch") -> dict[Category, HashMapU32U16]:
    """Allocate the three FUNCTION-Category :class:`HashMapU32U16`
    instances needed by :func:`apply_per_row_remap`.

    Each map is pre-sized to an upper bound on the per-row FID-count it
    will see. The upper bound is the total number of call_targets in any
    one section header (across the whole batch) -- per-row maps see at
    most the call_targets_section of a single root variant plus the
    call_targets_section of every inlined callee. Sizing on the
    sum-across-the-batch is a safe overshoot; the hashmap's ``clean()``
    only resets occupancy, not capacity, so the same allocation backs
    every row.

    Plan reference: ``feedback_no_parallel_indexing`` -- never build a
    cache that memoises parsed state when the bytes are already
    addressable. Here we look at the section headers' call_targets list
    sizes (already in memory on every Stage1Section.section); no parsing.
    """
    # Walk Stage1 sections to estimate capacity. The plan's Rust-
    # allocation hot-path discipline calls for reusing across rows;
    # we allocate once per :func:`assemble_batch` invocation. Pre-sizing
    # avoids rehashing on the first few inserts of each row.
    estimated_cap = 0
    for stage1_section in stage3_batch.stage2.stage1.sections:
        estimated_cap += len(stage1_section.section.call_targets)
    # Floor on a reasonable starting capacity for the typical small case.
    if estimated_cap < 8:
        estimated_cap = 8
    return {cat: HashMapU32U16(capacity=estimated_cap) for cat in FUNCTION_CATEGORIES}


def assemble_batch(
    stage3: "Stage3Batch",
    *,
    context_len: int,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
) -> "BatchDecodeResult":
    """Compose the stage-4 modules into a :class:`BatchDecodeResult`.

    Steps:

    1. :func:`assemble_tokens` -> ``tokens: u16[batch_size, context_len]``.
       The prepend slot's token id is included implicitly via slot 0 of
       each call_target's ``expanded_token_ids`` (stage 2a).
    2. :func:`apply_per_row_remap` -> mutates
       ``stage3.identities_flat_caller_local`` IN PLACE; optionally
       produces ``(fid_sidecar, fid_row_offsets)``. The prepend slot's
       counter id is written as part of the same per-row walk.
    3. :func:`assemble_number_sidecars` ->
       ``(numbers_significant: u64, numbers_sign_exponent: u32)``.
    4. Pack into :class:`BatchDecodeResult` with the stage-2 row-offset
       arrays plumbed through verbatim and ``batch_idx_to_section_variant``
       inherited from stage 1.

    Parameters
    ----------
    stage3:
        Finalised :class:`Stage3Batch` (post bulk-byte build).
        ``identities_flat_caller_local`` is mutated in place by step 2.
    context_len:
        Per-row output budget in tokens (the column count of the output
        tensor). Mirrors :func:`assemble_tokens`'s parameter.
    include_fid_sidecar:
        When True, runs the dedup walk with
        ``collect_fid_sidecar=True`` and packs the resulting
        ``(fid_sidecar, fid_row_offsets)`` into the result.
    keep_intermediate:
        When True, the input :class:`Stage3Batch` is carried on the
        result's :attr:`BatchDecodeResult.intermediate` field. Defaults
        to False -- the level-1 result is the model-facing flat tensor
        layout; the hierarchy is purely a diagnostic.

    Returns
    -------
    BatchDecodeResult
        User-facing flat-tensor result. ``identities`` is the same
        underlying array as ``stage3.identities_flat_caller_local``
        (post-remap); callers that mutate the result after the fact
        are mutating the stage-3 buffer too.
    """

    # Step 1: token tensor.
    tokens = assemble_tokens(stage3, context_len=context_len)

    # Step 2: per-row dedup remap + prepend identity-slot writes.
    # ``apply_per_row_remap`` mutates ``identities_flat_caller_local``
    # in place; the returned ndarray IS that same buffer. We retain
    # the returned reference because ``BatchDecodeResult`` stores it
    # under a different field name (``identities`` vs the stage-3
    # ``identities_flat_caller_local``).
    dedup_maps = _build_dedup_maps(stage3)
    identities, fid_sidecar, fid_row_offsets = apply_per_row_remap(
        stage3,
        dedup_maps=dedup_maps,
        collect_fid_sidecar=include_fid_sidecar,
    )

    # Step 3: number sidecars.
    numbers_significant, numbers_sign_exponent = assemble_number_sidecars(stage3)

    # Step 4: pack.
    stage2 = stage3.stage2
    stage1 = stage2.stage1
    return BatchDecodeResult(
        tokens=tokens,
        identities=identities,
        identity_row_offsets=stage2.identity_row_offsets,
        numbers_significant=numbers_significant,
        numbers_sign_exponent=numbers_sign_exponent,
        number_row_offsets=stage2.number_row_offsets,
        batch_idx_to_section_variant=stage1.batch_idx_to_section_variant,
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_row_offsets,
        intermediate=stage3 if keep_intermediate else None,
    )
