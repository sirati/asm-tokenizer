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
sibling modules (``_dedup_walk``, ``_prepend``, ``_sidecar_concat``,
``_variant_padding``) own the rest. The four are composed by
``_assemble.py`` -- nothing here knows about the identity arm, number
arm, or sidecar offsets.

Prepend self-token ordering
---------------------------
Stage 2a ("`_expand_tokens.py`") already builds ``expanded_token_ids``
as ``np.concatenate([[calling_category_token_id_shifted],
expanded_real])`` per ``ALG-9``. The self-token sits at
``expanded_token_ids[0]`` of every call_target. This module writes the
full slice ``expanded_token_ids[:partial_cut_length]`` verbatim --
position 0 (the prepend) is copied alongside everything else. The
prepend's IDENTITY-sidecar **counter id** is written by the sibling
prepend module (``_prepend.py``) into
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

from ._batch_layout import UINT32_MAX

if TYPE_CHECKING:
    from ._types import Stage3Batch


__all__ = ["assemble_tokens"]


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
      (``_prepend.py``).
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
    mapping = stage1_batch.batch_idx_to_section_variant
    sections = stage1_batch.sections  # parallel to stage3_batch.sections

    tokens = np.zeros((batch_size, context_len), dtype=np.uint16)

    # Fast-path: ``context_len == 0`` means no column budget; nothing
    # can ever be written. The zero-allocation already matches the
    # contract -- exit early to avoid spinning the per-row loop. Same
    # logic when there are no rows.
    if batch_size == 0 or context_len == 0:
        return tokens

    sentinel = int(UINT32_MAX)

    for row in range(batch_size):
        section_idx = int(mapping[row, 0])
        slot_idx = int(mapping[row, 1])
        # Padding sentinel -> row stays at id 0 (zero-allocation).
        # Either column being the sentinel is treated as "padding";
        # ALG-10 always sets both columns to the sentinel together, but
        # the OR-check is the defensible reading of "is this a real
        # mapping entry".
        if section_idx == sentinel or slot_idx == sentinel:
            continue

        stage1_section = sections[section_idx]
        stage1_variant = stage1_section.variants[slot_idx]

        # ``Stage1Variant.batch_idx is None`` marks a "padding out" slot
        # per the field's docstring (RAGGED post-cutoff drop). Such
        # variants must not contribute content even though the mapping
        # entry isn't the sentinel.
        if stage1_variant.batch_idx is None:
            continue

        # Resolve the matching stage-3 variant via the parallel
        # ``sections[*].variants[*]`` hierarchy. The 4 stages share
        # identical (section, variant, call_target) indexing per the
        # plan's D9 contract -- ``stage3.sections[section_idx].variants
        # [slot_idx]`` mirrors ``stage1.sections[section_idx].variants
        # [slot_idx]``.
        stage3_variant = stage3_batch.sections[section_idx].variants[slot_idx]

        # Per-row write head; advances by partial_cut_length per
        # call_target. Capped at context_len from the get-go so the
        # secondary defensive cap (slice stop = min(remaining, ...))
        # never accidentally overruns.
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
            # ``surviving_token_count``, which is ``<= context_len``.
            # The cap here is a belt-and-braces guard against any
            # upstream accounting drift; it costs one ``min()`` per
            # call_target.
            remaining = context_len - col
            write_count = (
                partial_cut_length
                if partial_cut_length <= remaining
                else remaining
            )

            # Slice into ``expanded_token_ids`` -- numpy's slice
            # semantics clamp ``stop`` to the array length, so a
            # ``write_count`` that exceeds the array length (impossible
            # under the plan's contract but cheap to defend against)
            # would still copy at most ``len(expanded_token_ids)``
            # tokens. The assignment shape-check fires if the slice
            # and destination disagree -- the assertion below pins the
            # contract explicitly so a mismatch surfaces here rather
            # than as a NumPy traceback.
            src = stage2_ct.expanded_token_ids[:write_count]
            assert src.shape[0] == write_count, (
                f"expanded_token_ids shorter than partial_cut_length: "
                f"got {src.shape[0]}, expected {write_count} "
                f"(predicted_full_length={stage2_ct.predicted_full_length}, "
                f"partial_cut_length={partial_cut_length})"
            )

            tokens[row, col : col + write_count] = src
            col += write_count

    return tokens
