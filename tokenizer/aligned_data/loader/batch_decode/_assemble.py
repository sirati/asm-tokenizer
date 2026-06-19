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
column]`` + ``identities_flat[identity_slice.start]``) are produced as a
SIDE-EFFECT of (1) and (2) respectively. The token id lands via
``expanded_token_ids[0]`` during :func:`assemble_tokens`; the identity
counter is written by :func:`apply_per_row_remap` using the dedup map
that already holds the call_target's self-counter. There is no separate
prepend-write helper -- both writes live with the concern that owns the
target array.

See ``batch_decode_plan.md`` ``## Stages -- algorithm sketch`` Stage 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ._dedup_walk import apply_per_row_remap
from ._runlengths import compute_metatoken_runlengths
from ._sidecar_concat import assemble_number_sidecars
from ._token_assembly import assemble_tokens
from ._types import BatchDecodeResult

if TYPE_CHECKING:
    from ._types import Stage2Batch, Stage3Batch


__all__ = ["assemble_batch"]


def assemble_batch(
    stage3: "Stage3Batch",
    *,
    context_len: int,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
    emit_block_n_insns_runlength: bool = False,
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
    (
        identities,
        fid_sidecar,
        fid_row_offsets,
        fid_per_category_counts,
    ) = apply_per_row_remap(
        stage3,
        collect_fid_sidecar=include_fid_sidecar,
    )

    # Step 3: number sidecars.
    numbers_significant, numbers_sign_exponent = assemble_number_sidecars(stage3)

    # Step 4: optional metatoken-runlength sidecar (per the canonical
    # FTL accountant; see :mod:`._runlengths`).
    stage2 = stage3.stage2
    stage1 = stage2.stage1
    (
        block_runlength,
        block_runlength_row_offsets,
        insn_runlength,
        insn_runlength_row_offsets,
    ) = (
        _assemble_runlength_sidecars(stage2)
        if emit_block_n_insns_runlength
        else (None, None, None, None)
    )

    # Step 5: pack.
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
        fid_per_category_counts=fid_per_category_counts,
        block_runlength=block_runlength,
        block_runlength_row_offsets=block_runlength_row_offsets,
        insn_runlength=insn_runlength,
        insn_runlength_row_offsets=insn_runlength_row_offsets,
        intermediate=stage3 if keep_intermediate else None,
    )


def _truncate_runlengths_to_surviving(
    block_rl: np.ndarray,
    insn_rl: np.ndarray,
    surviving_body_slots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip per-call-target runlengths to the surviving body-slot count.

    ``surviving_body_slots`` is ``surviving_token_count - 1`` (drop the
    self-prepend slot, which is row-assembler-owned and not present in
    the runlengths). Emit blocks until cumulative slot count >=
    ``surviving_body_slots``; the cut block is INCLUDED with its full
    instruction list (the inspector clamps its column range at render
    time). Per-instruction slot counts are NOT trimmed -- a partially-
    cut instruction keeps its full slot count in the sidecar.
    """
    if surviving_body_slots <= 0 or block_rl.size == 0:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
        )
    insn_cumulative = np.cumsum(insn_rl, dtype=np.int64)
    # Determine how many blocks survive: track cumulative slot count
    # via block-end positions in the insn array.
    block_end_insn_idx = np.cumsum(block_rl, dtype=np.int64)
    # Slot count at end of each block:
    # insn_cumulative[block_end_insn_idx[k] - 1] (when block_end_insn_idx[k] > 0)
    surviving_block_count = 0
    for k in range(int(block_rl.size)):
        end_idx = int(block_end_insn_idx[k])
        if end_idx == 0:
            continue
        block_end_slot = int(insn_cumulative[end_idx - 1])
        block_start_slot = (
            int(insn_cumulative[int(block_end_insn_idx[k - 1]) - 1])
            if k > 0 and int(block_end_insn_idx[k - 1]) > 0
            else 0
        )
        surviving_block_count = k + 1
        if block_end_slot >= surviving_body_slots:
            break
        # else: still under cut, continue
    # Insn count: sum of block_rl over surviving blocks.
    if surviving_block_count == 0:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
        )
    surviving_insn_count = int(block_end_insn_idx[surviving_block_count - 1])
    return (
        block_rl[:surviving_block_count].astype(np.uint32, copy=False),
        insn_rl[:surviving_insn_count].astype(np.uint32, copy=False),
    )


def _assemble_runlength_sidecars(
    stage2: "Stage2Batch",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-row per-call_target metatoken runlengths.

    Walks ``stage2.stage1.batch_idx_to_section_variant`` row-by-row and
    for each non-padding row concatenates the per-call_target
    ``(block_runlength, insn_runlength)`` produced by
    :func:`compute_metatoken_runlengths`, clipped to the
    per-call-target ``surviving_token_count`` (stage 2's cutoff walk).
    Padding rows contribute zero blocks and zero instructions; multi-
    mapped variants (RESAMPLE / REDISTRIBUTE) contribute the same per-
    variant content for every referencing row.

    Returns ``(block_runlength, block_runlength_row_offsets,
    insn_runlength, insn_runlength_row_offsets)`` -- each shaped per
    :class:`BatchDecodeResult`'s field docstrings.
    """
    sentinel = int(np.iinfo(np.uint32).max)
    stage1 = stage2.stage1
    batch_idx_map = stage1.batch_idx_to_section_variant
    batch_size = int(stage1.batch_size)

    # Per-unique-variant cache of (block_rl, insn_rl) concatenated
    # across the variant's call_targets in DFS encounter order, with
    # per-CT cutoff clipping applied.
    per_variant_block_rl: list[np.ndarray] = []
    per_variant_insn_rl: list[np.ndarray] = []
    section_variant_offsets: list[int] = [0]
    for s2_section in stage2.sections:
        for s2_variant in s2_section.variants:
            block_pieces: list[np.ndarray] = []
            insn_pieces: list[np.ndarray] = []
            for s2_ct in s2_variant.call_targets:
                # Surviving body slots = surviving_token_count - 1 (drop
                # the row-assembler-owned self-prepend slot).
                surviving = int(s2_ct.surviving_token_count)
                if surviving <= 0:
                    continue
                surviving_body_slots = surviving - 1
                full_br, full_ir = compute_metatoken_runlengths(
                    s2_ct.stage1.function_data
                )
                br, ir = _truncate_runlengths_to_surviving(
                    full_br, full_ir, surviving_body_slots
                )
                block_pieces.append(br)
                insn_pieces.append(ir)
            per_variant_block_rl.append(
                np.concatenate(block_pieces) if block_pieces
                else np.zeros(0, dtype=np.uint32)
            )
            per_variant_insn_rl.append(
                np.concatenate(insn_pieces) if insn_pieces
                else np.zeros(0, dtype=np.uint32)
            )
        section_variant_offsets.append(len(per_variant_block_rl))

    # Walk batch_idx_to_section_variant row-by-row, gathering the per-
    # variant arrays in row order. Pure-Python loop here (batch_size is
    # bounded by the row count, not the much larger per-row content).
    row_block_pieces: list[np.ndarray] = []
    row_insn_pieces: list[np.ndarray] = []
    block_row_offsets = np.empty(batch_size + 1, dtype=np.uint32)
    insn_row_offsets = np.empty(batch_size + 1, dtype=np.uint32)
    block_row_offsets[0] = 0
    insn_row_offsets[0] = 0
    running_blocks = 0
    running_insns = 0
    for row in range(batch_size):
        section_idx = int(batch_idx_map[row, 0])
        slot_idx = int(batch_idx_map[row, 1])
        if section_idx == sentinel or slot_idx == sentinel:
            # Padding row -- contributes zero.
            block_row_offsets[row + 1] = running_blocks
            insn_row_offsets[row + 1] = running_insns
            continue
        flat_variant_idx = section_variant_offsets[section_idx] + slot_idx
        br = per_variant_block_rl[flat_variant_idx]
        ir = per_variant_insn_rl[flat_variant_idx]
        row_block_pieces.append(br)
        row_insn_pieces.append(ir)
        running_blocks += int(br.size)
        running_insns += int(ir.size)
        block_row_offsets[row + 1] = running_blocks
        insn_row_offsets[row + 1] = running_insns

    block_runlength = (
        np.concatenate(row_block_pieces).astype(np.uint32)
        if row_block_pieces
        else np.zeros(0, dtype=np.uint32)
    )
    insn_runlength = (
        np.concatenate(row_insn_pieces).astype(np.uint32)
        if row_insn_pieces
        else np.zeros(0, dtype=np.uint32)
    )
    return block_runlength, block_row_offsets, insn_runlength, insn_row_offsets
