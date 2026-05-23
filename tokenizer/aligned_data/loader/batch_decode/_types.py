"""Frozen dataclass backbone for the staged batch-decode pipeline.

This module owns ONE concern: the immutable handoff shapes that move
data between the four batch-decode stages (load -> length-predict ->
bulk-bytes -> assemble). It contains no algorithmic code; per-stage
modules import these types and populate them.

The plan's design for batch_decode (see ``batch_decode_plan.md``,
section "D9 -- 4-level hierarchical dataclasses") pins a single
shape across every stage:

1. **Request / Batch** (level 1, outermost) -- the whole batch. Owns
   batch-shared arrays (the flat ``u8`` inline buffer, batch-wide
   ``row_offsets``, per-:class:`TokenType` ``(significand, sign_exp)``
   arrays).
2. **Section** (level 2) -- one per section pointer in the request.
   Owns section identity + reference to the parsed
   :class:`Section` BIN object.
3. **Variant** (level 3) -- one per sampled variant (level-2 child).
   Owns variant identity + the ``batch_idx`` this variant occupies in
   the output.
4. **CallTarget** (level 4) -- one per function in the variant's splice
   tree. The root function's body is level-4 entry index 0; same shape
   as inlined callees (no special-case root dataclass). Owns
   per-function state.

Each stage's level-N class wraps the prior stage's same-level class via
a single ``stageN`` back-pointer (no co-indexing of parallel lists).
The lazy-view discipline from
``feedback_lazy_view_no_materialization.md`` applies to per-position
iteration helpers (TBD in later phases); these compositional handoff
types are deliberately frozen because they describe immutable
boundaries between stages.

The plan reserves ``call_targets[0]`` per variant for the root body;
``call_targets[1:]`` are inlined callees in stage-1 DFS encounter order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# Existing types this module references.
# NOTE: the plan section D9 names the parsed call_target row
# ``CallTargetRow`` and the variant header ``VariantHeader``; the
# actual classes in ``matched_sections_bin`` are
# :class:`CallTarget` (parsed call_target row) and
# :class:`VariantBlock` (parsed variant block). We import the actual
# names; ``Stage1CallTarget.call_targets_section`` is a
# ``list[CallTarget]``.
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
    Section,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.tokens import Category, TokenType


__all__ = [
    "BatchDecodeResult",
    "SectionPointerSpec",
    "Stage1Batch",
    "Stage1CallTarget",
    "Stage1Section",
    "Stage1Variant",
    "Stage2Batch",
    "Stage2CallTarget",
    "Stage2Section",
    "Stage2Variant",
    "Stage3Batch",
    "Stage3CallTarget",
    "Stage3Section",
    "Stage3Variant",
    "VariantPadding",
]


# ---------------------------------------------------------------------------
# Runtime enums + spec types (plan section D6 + "Module layout")
# ---------------------------------------------------------------------------


class VariantPadding(Enum):
    """Per-section variant-padding policy for the output batch.

    Determines how the variant-axis of the output tensor is filled when
    sections have FEWER or MORE real variants than the request's
    ``num_variants_per_section``. The exact ``batch_idx``-to-
    ``(section_idx, variant_idx)`` mapping is computed once at stage 1
    (post-sampling, pre-loading) and threaded through every stage; see
    plan ALG-10 for the per-policy details.

    Values:

    * :attr:`PAD_NULL` -- short sections pad with all-null-content rows
      (recommended default).
    * :attr:`RESAMPLE_WITHIN_SECTION` -- oversample within section to
      fill rows.
    * :attr:`RAGGED` -- ``batch_size`` = total real variants; no padding
      rows.
    * :attr:`REDISTRIBUTE` -- take extra samples from sections that had
      MORE variants than requested.
    """

    PAD_NULL = "pad_null"
    RESAMPLE_WITHIN_SECTION = "resample"
    RAGGED = "ragged"
    REDISTRIBUTE = "redistribute"


@dataclass(frozen=True)
class SectionPointerSpec:
    """One section pointer in a :func:`batch_decode` request.

    Identifies a single section in the binary. ``arm`` is
    :attr:`SectionKind.MATCHED` or :attr:`SectionKind.UNMATCHED`;
    ``idx`` is the per-arm function/section index. The pair matches
    the inputs of :meth:`BinarySession._load_matched_for_splice`
    (which takes ``idx`` + a variant index) and
    :meth:`BinarySession._load_unmatched_for_splice` (which takes
    ``idx``) -- i.e. the same ``(arm, idx)`` keying every per-arm
    loader already uses.
    """

    arm: SectionKind
    idx: int  # per-arm function/section idx


# ---------------------------------------------------------------------------
# Stage 1 -- load (plan section D9 "Stage 1 -- load")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage1CallTarget:
    """Level 4. Per-function loaded state.

    One instance per function in a variant's splice tree -- index 0 is
    the variant's root body; indices 1+ are inlined callees in stage-1
    DFS encounter order. The root body is the SAME shape as inlined
    callees; no special-case root dataclass.
    """

    function_data: FunctionData
    """Tokens + insn_runlength + block_runlength + variant_tokens +
    metadata for this function."""

    state: InlineDecodeState
    """Masks + runlengths + carries_inline + is_negative pre-compute
    over ``function_data``'s token stream."""

    call_targets_section: list[CallTarget]
    """THIS function's section's ``call_targets`` table. Used at stage 4
    for FID resolution -- each per-call slot's ``called_idx`` indexes
    into this list to recover the callee's
    ``(function_name_ptr, CallTargetType)``."""

    encounter_category: Category
    """Per plan D3 + D4: ``LOCAL_FUNC`` for the root and for
    LOCAL-inlined callees; ``PLT_FUNC`` for PLT-inlined callees. Drives
    the self-prepend token category at stage 4 (ALG-9). EXT_FUNC is
    out of scope (no body to inline)."""

    parent_call_target_index: Optional[int]
    """Index in the PARENT's ``call_targets_section`` that pointed to
    this call_target. ``None`` for the root."""

    function_name_ptr: int
    """This function's global FID (function-name pointer in the names
    sidecar). Used at stage 4 for FUNCTION-category dedup keying."""


@dataclass(frozen=True)
class Stage1Variant:
    """Level 3. Per-variant load context."""

    variant_idx: int
    """Index within the section's variants list."""

    variant_ref_offset: int
    """vkey for variant-identity lookup (matches the variant's
    ``variant_ref_offset`` on the BIN side)."""

    batch_idx: Optional[int]
    """Row in the output tensor; ``None`` if this variant slot is
    padding-out (e.g. RAGGED policy or post-cutoff drop)."""

    call_targets: list[Stage1CallTarget]
    """``[root, callee_1, callee_2, ...]``; index 0 is the variant body
    per plan D3 + D9."""


@dataclass(frozen=True)
class Stage1Section:
    """Level 2. Per-section-pointer context."""

    arm: SectionKind
    """:attr:`SectionKind.MATCHED` or :attr:`SectionKind.UNMATCHED` --
    mirrors :class:`SectionPointerSpec.arm`."""

    idx: int
    """Per-arm function/section idx -- mirrors
    :class:`SectionPointerSpec.idx`."""

    section: Section
    """The BIN's parsed :class:`Section` (call_targets table + variant
    blocks). Read ``section.section_offset`` for the BIN-side byte
    offset."""

    variants: list[Stage1Variant]
    """One :class:`Stage1Variant` per sampled variant for this
    section pointer."""


@dataclass(frozen=True)
class Stage1Batch:
    """Level 1. The whole batch.

    Per plan D10: ``batch_idx_to_section_variant`` is the canonical
    ``batch_idx -> (section_idx, variant_idx)`` mapping computed once
    at stage 1 (post-sampling, pre-loading) per :class:`VariantPadding`
    policy. Padding rows carry sentinel ``(UINT32_MAX, UINT32_MAX)``.
    """

    sections: list[Stage1Section]
    batch_idx_to_section_variant: np.ndarray
    """``u32[batch_size, 2]``; columns are ``(section_idx,
    variant_idx)``. Padding rows are ``(UINT32_MAX, UINT32_MAX)``."""

    batch_size: int
    """Cached ``len(batch_idx_to_section_variant)``."""


# ---------------------------------------------------------------------------
# Stage 2 -- predict lengths + cutoff (plan section D9 + D8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage2CallTarget:
    """Level 4. Per-function length + cutoff predictions.

    Wraps the level-4 stage-1 entry via the ``stage1`` back-pointer.
    Adds the post-promotion / post-strip / post-shift expanded token
    stream + the side-array population masks + the cutoff-aware
    surviving counts.
    """

    stage1: Stage1CallTarget

    expanded_token_ids: np.ndarray
    """``u16[predicted_full_length]`` -- the function's post-promotion,
    post-strip, post-shift token stream BEFORE stage-4 cutoff."""

    extra_value_v2_mask: np.ndarray
    """``bool[predicted_full_length]`` -- True at positions that are
    "extra" VC2 chunks (chunks beyond the leading VC2 carrier)."""

    extra_f128_mask: np.ndarray
    """``bool[predicted_full_length]`` -- True at positions that are
    the second F128 chunk for a finite F128 source (per ALG-2)."""

    predicted_full_length: int
    """Length of ``expanded_token_ids`` (post-promotion)."""

    surviving_token_count: int
    """Number of tokens that survive the per-row cutoff;
    ``<= predicted_full_length``, equals it when fully included."""

    surviving_identity_count: int
    """Identity-token positions in
    ``expanded_token_ids[:surviving_token_count]``."""

    surviving_number_chunk_count: int
    """Number-token positions in
    ``expanded_token_ids[:surviving_token_count]``."""

    is_cut: bool
    """Whether the row's ``context_len`` cutoff falls inside THIS
    function's body (i.e. this is the cut function for the row)."""

    partial_cut_length: int
    """``= surviving_token_count``; the prefix length included in the
    row when ``is_cut`` is True."""


@dataclass(frozen=True)
class Stage2Variant:
    """Level 3. Per-variant cutoff result."""

    stage1: Stage1Variant

    call_targets: list[Stage2CallTarget]
    """Parallel to ``stage1.call_targets``; same indices."""

    cut_call_target_index: int
    """Index of the level-4 entry whose body was cut;
    ``len(call_targets)`` when no cut was needed (variant fits
    entirely)."""

    total_surviving_token_count: int
    """Sum of ``surviving_token_count`` over ``call_targets``."""

    total_surviving_identity_count: int
    total_surviving_number_chunk_count: int


@dataclass(frozen=True)
class Stage2Section:
    """Level 2. Section wrapper for stage 2."""

    stage1: Stage1Section
    variants: list[Stage2Variant]


@dataclass(frozen=True)
class Stage2Batch:
    """Level 1. The whole batch, post length + cutoff prediction.

    Adds the per-row cumulative-offset arrays that size the stage-3
    flat side arrays.
    """

    stage1: Stage1Batch
    sections: list[Stage2Section]

    identity_row_offsets: np.ndarray
    """``u32[batch_size + 1]`` -- cumsum over batch rows of
    ``Stage2Variant.total_surviving_identity_count``."""

    number_row_offsets: np.ndarray
    """``u32[batch_size + 1]`` -- cumsum over batch rows of
    ``Stage2Variant.total_surviving_number_chunk_count``."""


# ---------------------------------------------------------------------------
# Stage 3 -- bulk byte buffer + FP normalization (plan section D9 + ALG-7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3CallTarget:
    """Level 4. Per-call-target slices into level-1 batch-shared arrays.

    Stage 3 owns the batch-shared bulk arrays at level 1; per-call
    target SLICES live here so consumers (stage 4 + tests) never
    compute offsets externally.
    """

    stage2: Stage2CallTarget

    inline_byte_slice: slice
    """This call_target's range in
    :attr:`Stage3Batch.inline_bytes`."""

    identity_slice: slice
    """This call_target's range in
    :attr:`Stage3Batch.identities_flat_caller_local`. INCLUDES the
    prepend slot at index ``identity_slice.start``; stage 4 writes
    that slot directly per ALG-9."""

    number_chunk_slices: dict[TokenType, slice]
    """Per-:class:`TokenType` ranges in the corresponding entry of
    :attr:`Stage3Batch.numbers_per_TokenType`. Stage 4 pulls the
    right ``(significand, sign_exp)`` chunks for assembling the
    per-row number arrays."""


@dataclass(frozen=True)
class Stage3Variant:
    """Level 3. Variant wrapper for stage 3."""

    stage2: Stage2Variant
    call_targets: list[Stage3CallTarget]
    """Parallel to ``stage2.call_targets``."""


@dataclass(frozen=True)
class Stage3Section:
    """Level 2. Section wrapper for stage 3."""

    stage2: Stage2Section
    variants: list[Stage3Variant]


@dataclass(frozen=True)
class Stage3Batch:
    """Level 1. Owns the batch-shared bulk arrays.

    Stage 3 lifts the surviving inline bytes into a single flat
    ``u8`` buffer (per ALG-1) and produces per-:class:`TokenType`
    normalized ``(significand, sign_exp)`` arrays (per ALG-7). Each
    stage-4 row assembles its slice of the output by indexing into
    these flat arrays via the per-call-target ranges on
    :class:`Stage3CallTarget`.
    """

    stage2: Stage2Batch
    sections: list[Stage3Section]

    inline_bytes: np.ndarray
    """``u8[total_bytes + 1]`` -- index 0 is the leading zero pad
    (ALG-1)."""

    identities_flat_caller_local: np.ndarray
    """``u16[total_surviving_identity_tokens]`` -- pre-remap
    caller-local identities. Stage 4 mutates IN PLACE during the
    per-row dedup walk."""

    numbers_per_TokenType: dict[TokenType, tuple[np.ndarray, np.ndarray]]
    """Per-:class:`TokenType` normalized number arrays (ALG-7 output).

    Keys are :class:`TokenType` members in the number band
    (``VALUED_CONST_V2`` + ``FLOAT16`` / ``BFLOAT16`` / ``FLOAT32`` /
    ``FLOAT64`` / ``FLOAT80`` / ``FLOAT128``). Values are
    ``(significand: u64[n_chunks_of_type],
    sign_exp: u32[n_chunks_of_type])``."""

    identity_idx_2d: np.ndarray
    """``u32[total_surviving_identity_tokens, 2]`` -- intermediate
    diagnostic. Useful for tests + the optional intermediate
    output."""

    number_idx_2d_per_TokenType: dict[TokenType, np.ndarray]
    """Per-:class:`TokenType` ``u32[n_chunks_of_type, 2]`` -- the
    2D indexer arrays used by ALG-7's view-cast decode."""

    vc2_chunk_exponent_sidecar: np.ndarray
    """``u32[total_vc2_chunks]`` -- per-chunk exponent base index
    (chunk_index_within_source) for the VC2 multi-chunk byte
    layout (ALG-8)."""

    f128_is_nan_or_inf: np.ndarray
    """``bool[n_f128_sources]`` -- per-F128-SOURCE NaN/Inf flag from
    ALG-2's high-u16 exponent check. Length = number of F128 sources
    in the batch (NOT chunks); 3d's normalization branches into
    ``_encode_infnan`` for True entries and the finite-source path
    for False entries. Produced by 3c (``_number_decode``)."""

    f128_visible_chunks: np.ndarray
    """``u8[n_f128_sources]`` -- per-F128-SOURCE visible-chunk count
    in ``{1, 2}``. NaN/Inf = 1, finite full = 2, finite mid-cut = 1
    (the painted MSB slot was past ``partial_cut_length``). 3d's
    F128 normalizer reads this to derive ``chunks_per_source`` so the
    row-count assertion stays consistent in the mid-cut case.
    Produced by 3c (``_number_decode``)."""


# ---------------------------------------------------------------------------
# Stage 4 -- final tensor + remap + variant padding (plan section D9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchDecodeResult:
    """Level 1 output. The user-facing flat tensors + sidecar offsets.

    Stage 4 is the only stage that produces a non-hierarchical output:
    the model-facing flat tensors. The hierarchical
    :class:`Stage3Batch` may optionally be carried alongside via
    ``keep_intermediate=True``.
    """

    tokens: np.ndarray
    """``u16[batch_size, context_len]`` -- ``id == 0`` is the
    null-content padding (plan D5)."""

    identities: np.ndarray
    """``u16[total_surviving_identity_tokens]`` -- POST per-Category
    remap. One entry per identity-token position in row-major
    iteration (plan D5)."""

    identity_row_offsets: np.ndarray
    """``u32[batch_size + 1]``."""

    numbers_significant: np.ndarray
    """``u64[total_surviving_number_chunks]`` -- shared across ALL
    number :class:`TokenType` members; chunks in stream-position
    order within each row."""

    numbers_sign_exponent: np.ndarray
    """``u32[total_surviving_number_chunks]`` -- aligned with
    :attr:`numbers_significant`."""

    number_row_offsets: np.ndarray
    """``u32[batch_size + 1]``."""

    batch_idx_to_section_variant: np.ndarray
    """``u32[batch_size, 2]`` -- mirrors the stage-1 mapping; padding
    rows are ``(UINT32_MAX, UINT32_MAX)``."""

    fid_sidecar: Optional[np.ndarray]
    """``u32`` flat array; only when ``include_fid_sidecar=True``.
    Maps ``(LOCAL/PLT/EXT_FUNC, counter_id=K) -> original
    function_name_ptr`` per plan D5."""

    fid_row_offsets: Optional[np.ndarray]
    """``u32[batch_size + 1]``; only when
    ``include_fid_sidecar=True``."""

    intermediate: Optional[Stage3Batch]
    """The full hierarchical :class:`Stage3Batch`; only when
    ``keep_intermediate=True`` (default ``False``)."""
