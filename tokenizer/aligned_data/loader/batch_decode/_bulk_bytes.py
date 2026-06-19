"""Stage 3 entry -- compose 3a/3b/3c/3d into a :class:`Stage3Batch`.

Single concern: orchestrate the four stage-3 sub-modules:

* :mod:`._inline_bytes` (3a) -- flat ``u8`` buffer of surviving inline
  bytes + per-call-target byte slices.
* :mod:`._identity_decode` (3b) -- in-stream identity ``idx_2d`` table +
  per-call-target identity slices (each slice INCLUDES the prepend slot
  at ``slice.start``; stage 4 writes that slot per ALG-9).
* :mod:`._number_decode` (3c) -- per-:class:`TokenType` number
  ``idx_2d`` tables + per-call-target chunk slices + ``f128_is_nan_or_inf``
  sidecar + ``vc2_chunk_exponent_sidecar``.
* :mod:`._fp_normalize` (3d) -- vectorised per-:class:`TokenType` FP /
  VC2 normalisation to the f96 sidecar shape.

Plus the cross-stage glue this module owns and only this module owns:

1. The level-1 ``identities_flat_caller_local`` allocation (u16 with
   prepend slots zeroed; stage 4 fills the prepend at
   ``identity_slice.start``).
2. The view-cast of 3b's ``identity_idx_2d`` into u16 caller-local ids
   and the scatter into the post-prepend sub-slice of each call-target's
   ``identity_slice`` (i.e. ``[start + 1 : stop]``).
3. The per-source sign collection (``is_negative_per_source_per_type``)
   threaded into 3d. For each surviving number-band CARRIER in the
   expanded stream (in stream-source order, post-promotion), read
   ``state.is_negative_per_position`` at the carrier's RAW position and
   append to the per-:class:`TokenType` list. VC2 multi-chunk sources
   share a single source-level sign across all their chunks; 3d expands
   it per chunk via :func:`_fp_normalize.vc2_per_chunk_sign`. IEEE
   bit-pattern types ignore the sign array (sign lives in the bit
   pattern) but the array is still supplied for API uniformity.
4. The :class:`Stage3CallTarget` / :class:`Stage3Variant` /
   :class:`Stage3Section` / :class:`Stage3Batch` assembly.

Plan reference: ``batch_decode_plan.md`` ``## Stages -- algorithm sketch``
Stage 3 + ALG-1 + ALG-5 + ALG-7 + ALG-8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from ._dense_columns import DenseColumns
from ._flat_call_targets import dense_columns_from_stage2
from ._fp_normalize import normalize_per_token_type
from ._identity_decode import build_identity_idx_2d, view_cast_identities
from ._inline_bytes import build_inline_bytes
from ._number_decode import build_number_idx_2d
from ._types import (
    Stage3Batch,
    Stage3CallTarget,
    Stage3Section,
    Stage3Variant,
)

if TYPE_CHECKING:
    from ._types import (
        Stage2Batch,
        Stage2Section,
        Stage2Variant,
    )


__all__ = ["build_bulk_bytes"]


# ---------------------------------------------------------------------------
# Vocab anchors -- the NUMBER band post-shift sits at expanded ids
# ``[1, 8)``; index 0 of that block -> VC2, 1 -> F16, ..., 6 -> F128.
# Kept consistent with :mod:`._number_decode` so a vocab-layout drift
# surfaces in both files.
# ---------------------------------------------------------------------------

_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7

_NUMBER_BAND_LO_SHIFTED = _NUMBER_BLOCK_START - _RESERVED_DIGIT_COUNT  # 1
_NUMBER_BAND_HI_SHIFTED = (
    _NUMBER_BLOCK_START + _NUMBER_BLOCK_COUNT - _RESERVED_DIGIT_COUNT
)  # 8 (exclusive)

# Canonical NUMBER-block ordering (matches :mod:`._number_decode`).
_NUMBER_BLOCK_TOKEN_TYPES: tuple[TokenType, ...] = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)


def build_bulk_bytes(
    stage2: "Stage2Batch",
    dense: "DenseColumns | None" = None,
) -> Stage3Batch:
    """Compose 3a + 3b + 3c + 3d into a :class:`Stage3Batch`.

    Walks the level-4 hierarchy in DFS encounter order. The sub-stage
    functions each consume the same DFS order, so the per-call-target
    slice lists they return zip cleanly with the hierarchy walk used to
    assemble the level-4 :class:`Stage3CallTarget` entries.

    Parameters
    ----------
    stage2
        Output of stage 2 (length-predict + cutoff walk). Provides the
        4-level hierarchy + per-call-target expanded streams + masks +
        surviving counts.
    dense
        The shared :class:`DenseColumns` front-matter, when a caller has
        already built it (the vector dense path builds it ONCE from the
        ``BatchedExpansion``, collapsing the four tree-walks). Omitted by
        the staged ``batch_decode`` path, which builds it here with a
        single DFS walk of ``stage2``.

    Returns
    -------
    Stage3Batch
        Level-1 batch with batch-shared bulk arrays + the 4-level
        Stage3 mirror carrying per-call-target slices into them.
    """
    if dense is None:
        dense = dense_columns_from_stage2(stage2)

    # 1. inline_bytes (3a): per-call-target u8 byte slices into a flat
    #    buffer with a leading zero pad at index 0.
    inline_bytes, inline_byte_slices = build_inline_bytes(dense)

    # 2. identity idx_2d (3b): u32[N_in_stream, 2] of byte offsets into
    #    inline_bytes; per-call-target identity_slices INCLUDE the
    #    prepend slot at slice.start.
    identity_idx_2d, identity_slices = build_identity_idx_2d(
        dense, inline_bytes, inline_byte_slices
    )

    # 3. view-cast 3b's idx_2d to u16 in-stream caller-local ids. NO
    #    prepend slots in here; stage 4 fills those into the level-1
    #    array directly per ALG-9.
    identities_in_stream = view_cast_identities(
        identity_idx_2d, inline_bytes
    )

    # 4. number idx_2d (3c): per-TokenType u32[n_chunks_of_T, payload_T]
    #    arrays + per-call-target chunk slices + sidecars (NaN/Inf
    #    dispatch flag + VC2 chunk-exponent indices).
    (
        idx_2d_per_type,
        number_chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_chunk_exponent_sidecar,
    ) = build_number_idx_2d(dense, inline_bytes, inline_byte_slices)

    # 5. Per-source signs grouped by TokenType, in the same stream-source
    #    order the per-type idx_2d arrays use. Reads
    #    ``state.is_negative_per_position`` at each surviving carrier's
    #    raw position. Sign applies per SOURCE: VC2 multi-chunk sources
    #    have one sign for all their chunks; 3d's ``vc2_per_chunk_sign``
    #    expands per-source to per-chunk via the
    #    ``vc2_chunk_exponent_sidecar``.
    is_negative_per_source_per_type = _collect_is_negative_per_source_per_type(
        dense
    )

    # 6. Vectorised per-TokenType FP normalisation (3d).
    numbers_per_TokenType = normalize_per_token_type(
        idx_2d_per_type=idx_2d_per_type,
        inline_bytes=inline_bytes,
        f128_is_nan_or_inf=f128_is_nan_or_inf,
        vc2_chunk_exponent_sidecar=vc2_chunk_exponent_sidecar,
        is_negative_per_source_per_type=is_negative_per_source_per_type,
    )

    # 7. Allocate the level-1 identities array. Length = sum of each
    #    call_target's ``surviving_identity_count`` (which already
    #    INCLUDES the +1 prepend per Stage 2's count semantics). Prepend
    #    slots stay 0; stage 4 writes them per ALG-9.
    total_surviving_identity_tokens = (
        identity_slices[-1].stop if identity_slices else 0
    )
    identities_flat_caller_local = np.zeros(
        total_surviving_identity_tokens, dtype=np.uint16
    )

    # 8. Scatter in-stream u16 ids into the post-prepend sub-slice of
    #    each call_target's identity_slice. The 3b walk's emission order
    #    is "all in-stream identities of call_target 0, then 1, then 2,
    #    ..." -- a flat list whose per-call-target lengths are
    #    ``surviving_identity_count - 1`` for surviving call_targets and
    #    0 for fully-dropped ones. We re-derive the per-call-target
    #    cumulative offsets from the identity_slices and copy in one
    #    pass per call_target (no per-token loop).
    in_stream_cursor = 0
    for sl in identity_slices:
        slice_len = sl.stop - sl.start
        if slice_len == 0:
            continue
        # In a surviving call_target the slice is [prepend, in_stream_0,
        # in_stream_1, ...]; slot 0 = prepend (stage 4), slots 1..end =
        # in-stream caller-local ids from the view-cast.
        in_stream_len = slice_len - 1
        if in_stream_len > 0:
            identities_flat_caller_local[sl.start + 1 : sl.stop] = (
                identities_in_stream[
                    in_stream_cursor : in_stream_cursor + in_stream_len
                ]
            )
            in_stream_cursor += in_stream_len
    # Sanity: the view-cast emitted exactly as many u16s as we scattered.
    if in_stream_cursor != identities_in_stream.shape[0]:
        raise AssertionError(
            f"identity in-stream count mismatch: scattered "
            f"{in_stream_cursor} u16 ids but the view-cast produced "
            f"{identities_in_stream.shape[0]}"
        )

    # 9 + 10. Assemble the per-call-target / per-variant / per-section
    #         wrappers. DFS encounter order matches every sub-stage's
    #         output, so a single positional cursor lines up the slice
    #         lookups.
    sections = _assemble_hierarchy(
        stage2=stage2,
        inline_byte_slices=inline_byte_slices,
        identity_slices=identity_slices,
        number_chunk_slices_per_type=number_chunk_slices_per_type,
    )

    return Stage3Batch(
        stage2=stage2,
        sections=sections,
        inline_bytes=inline_bytes,
        identities_flat_caller_local=identities_flat_caller_local,
        numbers_per_TokenType=numbers_per_TokenType,
        identity_idx_2d=identity_idx_2d,
        number_idx_2d_per_TokenType=idx_2d_per_type,
        vc2_chunk_exponent_sidecar=vc2_chunk_exponent_sidecar,
        f128_is_nan_or_inf=f128_is_nan_or_inf,
    )


# ---------------------------------------------------------------------------
# Per-source sign collection.
# ---------------------------------------------------------------------------


def _collect_is_negative_per_source_per_type(
    dense: "DenseColumns",
) -> dict[TokenType, np.ndarray]:
    """Group per-source ``is_negative`` flags by :class:`TokenType`.

    Batched (B-S2) form of the per-source sign collection. Instead of one
    compute pass per call_target, the surviving carriers of EVERY
    call_target are identified + signed in a SINGLE vectorised pass over
    the flat ``expanded[1:surviving]`` concatenation, then grouped per
    :class:`TokenType` preserving DFS-then-stream encounter order.

    A multi-chunk source (VC2 K>1, F128 finite) contributes ONE entry
    here (per SOURCE, not per chunk) -- 3d's
    :func:`_fp_normalize.vc2_per_chunk_sign` expands the VC2 per-source
    array to per-chunk via the chunk-exponent sidecar.

    For IEEE bit-pattern types (F16/BF16/F32/F64/F80/F128) the sign
    lives in the gathered byte pattern and 3d doesn't read this array.
    We still populate it for API uniformity (and to make 3d's
    "sign mismatch" diagnostic catch upstream bugs).

    Equivalence to the prior per-call_target walk
    ---------------------------------------------
    Per call_target the scalar walk: (1) restricted to
    ``expanded[1:surviving]`` (slot 0 is the synthetic prepend, never a
    number carrier); (2) marked CARRIERS as NUMBER-band, non-painted
    slots; (3) mapped each carrier to its raw position via
    ``real_positions[cumsum(is_real) - 1]`` (the K-th non-painted slot
    consumes ``real_positions[K-1]``); (4) read
    ``is_negative_per_position[raw_pos]``; (5) appended per type in
    stream order. Each step is reproduced below SEGMENT-WISE: the
    ``cumsum(is_real)`` becomes a per-call_target segmented cumsum, and
    ``real_positions`` is concatenated per call_target with a CSR base so
    a single global gather recovers every carrier's raw position.
    """

    carrier_block_idx, carrier_signs = _batched_carrier_signs(dense)

    out: dict[TokenType, np.ndarray] = {}
    for block_idx, token_type in enumerate(_NUMBER_BLOCK_TOKEN_TYPES):
        type_mask = carrier_block_idx == block_idx
        # Boolean-mask selection preserves the flat (DFS, stream) order,
        # matching the per-type idx_2d row order :mod:`._number_decode`
        # builds.
        out[token_type] = carrier_signs[type_mask].astype(np.bool_, copy=False)
    return out


def _batched_carrier_signs(
    dense: "DenseColumns",
) -> tuple[np.ndarray, np.ndarray]:
    """One-pass carrier ``(block_idx, sign)`` over all kept nodes.

    Returns two parallel ``int64`` / ``bool`` arrays in DFS-then-stream
    encounter order: ``carrier_block_idx`` (0 = VC2, 1 = F16, ..., 6 =
    F128) and ``carrier_signs`` (the per-source negative flag). Painted
    continuation slots and the per-node prepend slot contribute no entries
    (a painted slot borrows its carrier's raw position; the prepend lives
    in the IDENTITY band).
    """
    kept_idx = np.asarray(dense.kept_node_index, dtype=np.int64).tolist()
    n_kept = len(kept_idx)
    if n_kept == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.bool_),
        )

    # --- flat ``expanded[1:surviving]`` concatenation (slot 0 dropped) ---
    expanded_chunks: list[np.ndarray] = []
    painted_chunks: list[np.ndarray] = []
    # --- flat per-node ``real_positions`` + ``is_negative`` ---
    real_pos_chunks: list[np.ndarray] = []
    is_neg_chunks: list[np.ndarray] = []
    exp_seg_len = np.empty(n_kept, dtype=np.int64)
    real_seg_base = np.empty(n_kept, dtype=np.int64)

    real_running = 0
    for i, e in enumerate(kept_idx):
        surviving = int(dense.surviving_token_count[e])
        raw_slice = dense.node_raw_slice(e)
        expanded_slice = dense.node_expanded_slice(e)
        expanded_chunks.append(
            dense.expanded[expanded_slice][1:surviving].astype(
                np.int64, copy=False
            )
        )
        painted_chunks.append(
            dense.extra_value_v2_mask[expanded_slice][1:surviving]
            | dense.extra_f128_mask[expanded_slice][1:surviving]
        )
        exp_seg_len[i] = max(surviving - 1, 0)
        real_positions = np.nonzero(dense.real_mask[raw_slice])[0]
        real_pos_chunks.append(real_positions.astype(np.int64, copy=False))
        is_neg_chunks.append(
            dense.is_negative_per_position[raw_slice][real_positions]
        )
        real_seg_base[i] = real_running
        real_running += int(real_positions.shape[0])

    expanded_flat = np.concatenate(expanded_chunks)
    is_painted_flat = np.concatenate(painted_chunks)
    is_real_flat = ~is_painted_flat
    real_pos_flat = np.concatenate(real_pos_chunks)
    is_neg_at_real_flat = np.concatenate(is_neg_chunks)

    if expanded_flat.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.bool_),
        )

    # CSR over the expanded-prefix axis (segment i = call_target i). A
    # kept call_target with ``surviving == 1`` has a ZERO-LENGTH body
    # segment (its only surviving slot is the prepend, dropped from the
    # ``[1:surviving]`` body axis); ``np.repeat`` over the per-segment
    # lengths yields the per-slot segment id correctly even for those empty
    # segments (the mark-and-cumsum CSR expansion would silently MERGE
    # consecutive zero-length boundaries, shifting all later segment ids).
    exp_seg_offsets = np.zeros(n_kept + 1, dtype=np.int64)
    np.cumsum(exp_seg_len, out=exp_seg_offsets[1:])
    seg_id = np.repeat(np.arange(n_kept, dtype=np.int64), exp_seg_len)

    # SEGMENTED ``cumsum(is_real) - 1`` per call_target. A global cumsum
    # carries the running real-count across segment boundaries; subtract
    # the cumulative value carried in at each segment's exclusive start to
    # reset it to the per-call_target ``cumsum(is_real) - 1`` the scalar
    # walk uses. The per-segment carry-in is the INCLUSIVE global cum at
    # the previous flat element (0 for the first segment), broadcast over
    # the segment via ``seg_id``. ``global_cum_excl`` is the exclusive
    # prefix sum, so its value at each segment's first flat index is
    # exactly that carry-in.
    is_real_i64 = is_real_flat.astype(np.int64)
    global_cum = np.cumsum(is_real_i64)
    global_cum_excl = global_cum - is_real_i64
    # ``seg_carry_in[seg]`` = exclusive global cum at the segment's first
    # element. ``exp_seg_offsets[:-1]`` is the first flat index of each
    # segment; for a zero-length segment that index coincides with the
    # next segment's start (no flat element belongs to it, so it is never
    # read through ``seg_id`` anyway). Trailing zero-length segments would
    # push the index to ``total``; clip to keep the gather in-bounds (the
    # clipped value is unused).
    first_idx = np.minimum(
        exp_seg_offsets[:-1], int(global_cum_excl.shape[0]) - 1
    )
    seg_carry_in = global_cum_excl[first_idx]
    real_idx_inclusive = global_cum - 1 - seg_carry_in[seg_id]

    # NUMBER-band non-painted carriers.
    in_number_band = (expanded_flat >= _NUMBER_BAND_LO_SHIFTED) & (
        expanded_flat < _NUMBER_BAND_HI_SHIFTED
    )
    carrier_mask = in_number_band & is_real_flat
    if not carrier_mask.any():
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.bool_),
        )

    carrier_seg = seg_id[carrier_mask]
    # Global index into the concatenated ``real_pos`` / ``is_neg`` arrays
    # = per-call_target base + per-call_target real index.
    carrier_real_global = (
        real_seg_base[carrier_seg]
        + real_idx_inclusive[carrier_mask]
    )
    carrier_signs = is_neg_at_real_flat[carrier_real_global]
    carrier_block_idx = (
        expanded_flat[carrier_mask] - _NUMBER_BAND_LO_SHIFTED
    )
    return carrier_block_idx, carrier_signs


# ---------------------------------------------------------------------------
# Hierarchy assembly.
# ---------------------------------------------------------------------------


def _assemble_hierarchy(
    *,
    stage2: "Stage2Batch",
    inline_byte_slices: list[slice],
    identity_slices: list[slice],
    number_chunk_slices_per_type: dict[TokenType, list[slice]],
) -> list[Stage3Section]:
    """Build the level-2 / level-3 / level-4 Stage3 wrappers.

    All per-call-target slice lists are in DFS encounter order; we walk
    the stage-2 hierarchy in the same order so a single positional
    cursor lines up each call_target's slice lookups.
    """
    ct_cursor = 0
    sections: list[Stage3Section] = []
    for stage2_section in stage2.sections:
        stage3_variants: list[Stage3Variant] = []
        for stage2_variant in stage2_section.variants:
            stage3_cts: list[Stage3CallTarget] = []
            for stage2_ct in stage2_variant.call_targets:
                per_type_slice: dict[TokenType, slice] = {
                    T: number_chunk_slices_per_type[T][ct_cursor]
                    for T in _NUMBER_BLOCK_TOKEN_TYPES
                }
                stage3_cts.append(
                    Stage3CallTarget(
                        stage2=stage2_ct,
                        inline_byte_slice=inline_byte_slices[ct_cursor],
                        identity_slice=identity_slices[ct_cursor],
                        number_chunk_slices=per_type_slice,
                    )
                )
                ct_cursor += 1
            stage3_variants.append(
                Stage3Variant(
                    stage2=stage2_variant,
                    call_targets=stage3_cts,
                )
            )
        sections.append(
            Stage3Section(
                stage2=stage2_section,
                variants=stage3_variants,
            )
        )
    return sections
