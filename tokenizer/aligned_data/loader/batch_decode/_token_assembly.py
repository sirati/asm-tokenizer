"""Stage 4 sub-concern: per-row token-tensor assembly + truncation.

Single concern: build the model-facing ``u16[batch_size, context_len]``
``tokens`` tensor from a fully prepared :class:`Stage3Batch`. This
module's *only* job is to walk
``stage3.stage2.stage1.batch_idx_to_section_variant`` and, for each
non-padding row, concatenate that variant's call_targets'
``expanded_token_ids[:partial_cut_length]`` slices into the row,
starting at column 0 and stopping at exactly ``context_len`` columns.
Trailing positions are left at id 0 (the null-content padding slot per
plan D5), which the zero-allocation already guarantees.

Why this lives in its own module
--------------------------------
Stage 4's algorithm sketch (``batch_decode_plan.md``  "Stage 4: assemble
flat output + counter remap + variant padding") interleaves four
side-effects per row: token write, identity-sidecar concat, number-chunk
concat, and FID sidecar build. Each of those is an independent concern
crossing a clean module boundary. This module owns concern #1 only;
sibling modules (``_dedup_walk``, ``_sidecar_concat``) own the rest.
Variant-padding sentinel checks are inlined at each consumer site
rather than living in a dedicated helper module. The four are composed
by ``_assemble.py`` -- nothing here knows about the identity arm,
number arm, or sidecar offsets.

Prepend self-token ordering
---------------------------
Stage 2a ("`_expand_tokens.py`") already builds ``expanded_token_ids``
as ``np.concatenate([[calling_category_token_id_shifted],
expanded_real])`` per ``ALG-9``. The self-token sits at
``expanded_token_ids[0]`` of every call_target. This module writes the
full slice ``expanded_token_ids[:partial_cut_length]`` verbatim --
position 0 (the prepend) is copied alongside everything else. The
prepend's IDENTITY-sidecar **counter id** is written by the sibling
dedup walk (see :mod:`._dedup_walk`) into
``stage3.identities_flat_caller_local[identity_slice.start]``, a
*disjoint* destination. The two writes are therefore order-independent:
either may run first.

Variant-padding handling
------------------------
Per plan ALG-10 + Stage 4 step "Variant-padding policy enforcement":
padding rows in ``batch_idx_to_section_variant`` hold the sentinel
``(UINT32_MAX, UINT32_MAX)`` (only ``VariantPadding.PAD_NULL`` produces
these in normal operation; the other three policies yield a dense
mapping). This module skips those rows -- they stay at id 0 (the
null-content slot per plan D5) from the zero-allocation. We also defend
against ``Stage1Variant.batch_idx is None`` (RAGGED's post-cutoff drop
case, per the ``Stage1Variant.batch_idx`` field docstring): we *don't*
write content into those rows either; their row in the tensor stays at
zero. The two checks are belt-and-braces -- the canonical signal is the
mapping sentinel, but a ``None`` ``batch_idx`` on a resolved variant
would indicate the variant is logically "padding out" and content
*should not* be assembled.

Multi-row mapping (RESAMPLE / REDISTRIBUTE)
-------------------------------------------
A single :class:`Stage1Variant` can be referenced by MORE than one batch
row when the layout is :attr:`VariantPadding.RESAMPLE_WITHIN_SECTION`
(deficit slots may resample the same source slot) or
:attr:`VariantPadding.REDISTRIBUTE` (each donor maps to one receiver
row, so duplicates only arise with RESAMPLE; REDISTRIBUTE is 1-to-1).
Each batch row's row in ``tokens`` is filled by re-running the same
concat against the same variant -- no de-duplication, no shared row
views. The two rows end up with identical content (modulo the per-row
mutable identity/COUNTER state owned by sibling modules).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._row_expand import build_per_row_variant_lookup

if TYPE_CHECKING:
    from ._types import Stage3Batch, Stage3Variant


__all__ = ["assemble_tokens"]


def _build_variant_row(
    stage3_variant: "Stage3Variant",
    *,
    context_len: int,
) -> np.ndarray:
    """Build the ``u16[context_len]`` token row for ONE unique variant.

    Concatenates the variant's call_targets' surviving prefixes
    (``expanded_token_ids[:partial_cut_length]``) in encounter order,
    capping at ``context_len`` total columns. Trailing positions stay
    at id 0 (the null-content slot per plan D5) via the zero-allocation.

    Returns a fresh ``u16`` array; callers scatter it into the per-row
    rows of the batch tokens tensor by fancy-indexing.
    """

    row = np.zeros(context_len, dtype=np.uint16)
    if context_len == 0:
        return row

    # ``Stage1Variant.batch_idx is None`` marks a "padding out" slot
    # (RAGGED post-cutoff drop). Such variants must not contribute
    # content even though the mapping entry isn't the sentinel.
    if stage3_variant.stage2.stage1.batch_idx is None:
        return row

    col = 0
    for stage3_ct in stage3_variant.call_targets:
        if col >= context_len:
            break
        stage2_ct = stage3_ct.stage2
        partial_cut_length = stage2_ct.partial_cut_length
        if partial_cut_length <= 0:
            # Dropped (post-cut) or empty call_target -- no write.
            continue

        # Defensive cap: never exceed the remaining column budget.
        # Per the plan's ``partial_cut_length`` accounting (Stage 2
        # step 4), the *sum* of ``partial_cut_length`` across this
        # variant's surviving call_targets is exactly the row's
        # ``surviving_token_count``, which is ``<= context_len``. The
        # cap here is a belt-and-braces guard against any upstream
        # accounting drift; it costs one ``min()`` per call_target.
        remaining = context_len - col
        write_count = (
            partial_cut_length
            if partial_cut_length <= remaining
            else remaining
        )

        src = stage2_ct.expanded_token_ids[:write_count]
        assert src.shape[0] == write_count, (
            f"expanded_token_ids shorter than partial_cut_length: "
            f"got {src.shape[0]}, expected {write_count} "
            f"(predicted_full_length={stage2_ct.predicted_full_length}, "
            f"partial_cut_length={partial_cut_length})"
        )
        row[col : col + write_count] = src
        col += write_count
    return row


def assemble_tokens(
    stage3_batch: "Stage3Batch",
    *,
    context_len: int,
) -> np.ndarray:
    """Build the ``u16[batch_size, context_len]`` token tensor.

    For each row index ``r`` in
    ``stage3_batch.stage2.stage1.batch_idx_to_section_variant``:

    * If the mapping entry is the padding sentinel
      ``(UINT32_MAX, UINT32_MAX)``: the row stays at id 0
      (null-content). The zero-allocation handles this implicitly --
      no explicit branch needed beyond skipping the variant lookup.

    * Otherwise: resolve the :class:`Stage1Variant` at
      ``sections[section_idx].variants[slot_v]`` (where ``slot_v`` is
      the column-1 entry, an INDEX into
      :attr:`Stage1Section.variants`, i.e. the slot index *post* variant
      sampling -- not the original variant index inside the section's
      underlying variants list). For each call_target in
      ``variant.call_targets`` (root first; then callees in stage-1 DFS
      encounter order), copy
      ``call_target.stage2.expanded_token_ids[:call_target.stage2.partial_cut_length]``
      into ``tokens[r]`` starting at the running column offset, advancing
      the column offset by the number of tokens copied. Truncate at
      exactly ``context_len`` columns -- any trailing positions stay at
      id 0 from the zero-allocation.

    Parameters
    ----------
    stage3_batch:
        Fully-prepared :class:`Stage3Batch` with the 4-level hierarchy
        populated. Only ``stage2.stage1.batch_idx_to_section_variant``,
        ``stage2.stage1.batch_size``, and the per-call_target
        ``stage2.expanded_token_ids`` + ``stage2.partial_cut_length``
        are read; identity / number sidecar fields are owned by sibling
        modules.
    context_len:
        Column count of the output tensor. Tokens beyond this column
        are dropped per plan D2 (mid-multi-chunk truncation allowed
        -- the cut is at the *token* level, not the *source* level).
        Stage 2's ``partial_cut_length`` calculation already encodes
        the per-row cut point, so the per-call_target write itself is
        already pre-cut; the ``context_len`` cap here is a defensive
        secondary cap that fires only when the upstream
        ``partial_cut_length`` accounting is internally consistent (the
        cumulative offset never exceeds ``context_len``). The defensive
        cap also handles the degenerate ``batch_size = 0`` /
        ``context_len = 0`` cases without special-casing.

    Returns
    -------
    np.ndarray
        ``u16[batch_size, context_len]`` -- the model-facing token
        tensor. ``id == 0`` is the reserved null-content slot per plan
        D5 (every untouched position).

    Notes
    -----
    * **Dtype + shape contract**: dtype is ``np.uint16`` (plan D5);
      shape is exactly ``(batch_size, context_len)``.
    * **Prepend slot**: 2a already concatenated the encounter-category
      self-token at ``expanded_token_ids[0]``; this module writes it
      alongside every other token. The prepend's
      ``identities_flat_caller_local`` counter is written by a sibling
      (the dedup walk in :mod:`._dedup_walk`).
    * **Null-content tail**: trailing positions are id 0 from the
      zero-allocation -- per plan D5 ``id == 0`` is the null-content
      padding slot.
    * **Empty variants**: a variant with zero ``call_targets`` (or whose
      call_targets all have ``partial_cut_length == 0``) contributes no
      writes; the row stays at zeros, identical to the padding-row
      case.
    """

    stage2_batch = stage3_batch.stage2
    stage1_batch = stage2_batch.stage1
    batch_size = stage1_batch.batch_size

    tokens = np.zeros((batch_size, context_len), dtype=np.uint16)

    # Fast-path: ``context_len == 0`` means no column budget; nothing
    # can ever be written. The zero-allocation already matches the
    # contract -- exit early to avoid spinning the per-variant build.
    # Same logic when there are no rows.
    if batch_size == 0 or context_len == 0:
        return tokens

    # Per-unique-variant row content in section -> slot flat order. The
    # outer loop iterates UNIQUE variants (NOT batch rows): multi-mapped
    # variants are built once and scattered to every referencing row.
    # Variants with ``batch_idx is None`` (RAGGED post-cutoff drop) and
    # variants outside the referenced set produce an all-zero row, which
    # has no effect on the scatter.
    per_variant_rows: list[np.ndarray] = []
    variants_per_section: list[int] = []
    for stage3_section in stage3_batch.sections:
        variants_per_section.append(len(stage3_section.variants))
        for stage3_variant in stage3_section.variants:
            per_variant_rows.append(
                _build_variant_row(stage3_variant, context_len=context_len)
            )

    per_row_variant_idx, is_padding = build_per_row_variant_lookup(
        stage1_batch.batch_idx_to_section_variant, variants_per_section
    )

    if not per_variant_rows:
        # No variants exist in the hierarchy. Every row is padding (or
        # the batch is empty); the zero-allocation already matches.
        return tokens

    per_variant_arr = np.stack(per_variant_rows, axis=0)
    # Scatter per-variant rows to batch rows, skipping padding rows
    # whose tokens row stays at id 0 from the zero-allocation.
    real_mask = ~is_padding
    if real_mask.any():
        tokens[real_mask] = per_variant_arr[per_row_variant_idx[real_mask]]
    return tokens
