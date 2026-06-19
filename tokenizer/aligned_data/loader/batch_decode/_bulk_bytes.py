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

from dedup_hashmap import build_carrier_signs_kernel

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from ._dense_columns import DenseColumns
from ._flat_call_targets import dense_columns_from_stage2
from ._fp_normalize import normalize_per_token_type
from ._identity_decode import (
    build_identity_idx_2d,
    scatter_in_stream_identities,
    view_cast_identities,
)
from ._inline_bytes import build_inline_bytes
from ._number_decode import build_number_idx_2d
from ._types import (
    Stage3Batch,
    Stage3CallTarget,
    Stage3Section,
    Stage3Variant,
)

if TYPE_CHECKING:
    from ._identity_decode import IdentitySlicesCSR
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
    *,
    build_hierarchy: "bool | None" = None,
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
    build_hierarchy
        Whether to assemble the per-call-target ``Stage3Section`` tree
        (step 9+10). ``None`` (default) derives it from ``dense``: the
        VECTOR dense path supplies ``dense`` AND threads its own columnar
        ``variants_per_section`` / ``numbers`` downstream, so the tree is
        vestigial there and is SKIPPED (``Stage3Batch.sections == ()``);
        the STAGED path leaves ``dense`` ``None`` and builds the tree (its
        ``assemble_batch`` walks ``stage3.sections``). The equivalence
        gates pass ``True`` WITH a ``dense`` to rebuild the full tree as
        their tree-walk oracle.

    Returns
    -------
    Stage3Batch
        Level-1 batch with batch-shared bulk arrays + (when built) the
        4-level Stage3 mirror carrying per-call-target slices into them.
    """
    columnar = dense is not None
    if build_hierarchy is None:
        # Default: build the tree exactly when no columnar ``dense`` was
        # supplied (the STAGED path). The vector dense path supplies
        # ``dense`` -> skips the vestigial tree.
        build_hierarchy = not columnar
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
    #    arrays + per-call-target chunk-slice CSR boundaries + sidecars
    #    (NaN/Inf dispatch flag + VC2 chunk-exponent indices).
    (
        idx_2d_per_type,
        number_chunk_boundaries_per_type,
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

    # 7 + 8. Allocate the level-1 identities array (length = sum of each
    #    call_target's ``surviving_identity_count``, which already INCLUDES
    #    the +1 prepend per Stage 2's count semantics; prepend slots stay 0
    #    for stage 4 per ALG-9) and scatter the in-stream u16 ids into the
    #    POST-PREPEND sub-range ``[start + 1 : stop]`` of every surviving
    #    call_target. The 3b walk's emission order is "all in-stream
    #    identities of call_target 0, then 1, ..." -- the identity CSR
    #    drives the vectorised cumulative-offset scatter in a single
    #    fancy-index assignment (no per-call_target Python loop).
    identities_flat_caller_local = scatter_in_stream_identities(
        identity_slices, identities_in_stream
    )

    # 9 + 10. Assemble the per-call-target / per-variant / per-section
    #         wrappers. DFS encounter order matches every sub-stage's
    #         output, so a single positional cursor lines up the slice
    #         lookups.
    #
    # SKIPPED on the vector dense path (``build_hierarchy`` False): that
    # path's downstream consumers (the per-row remap + the number-chunk
    # stream) read the columnar front-matter (the threaded
    # ``variants_per_section`` / ``numbers``), NOT these ``Stage3Section``
    # wrappers -- so the per-call-target hierarchy build (the ~24.7%
    # GIL-held Stage2/Stage3 tree ctor loop) is vestigial there and is
    # dropped (step-5 object-tree elimination). The STAGED ``batch_decode``
    # path keeps the full assembly (its ``assemble_batch`` walks
    # ``stage3.sections``); the equivalence gates also request it (with a
    # ``dense``) to rebuild the tree-walk oracle.
    sections = (
        _assemble_hierarchy(
            stage2=stage2,
            inline_byte_slices=inline_byte_slices,
            identity_slices=identity_slices,
            number_chunk_boundaries_per_type=number_chunk_boundaries_per_type,
        )
        if build_hierarchy
        else ()
    )

    return Stage3Batch(
        stage2=stage2,
        sections=sections,
        inline_bytes=inline_bytes,
        identities_flat_caller_local=identities_flat_caller_local,
        numbers_per_TokenType=numbers_per_TokenType,
        identity_idx_2d=identity_idx_2d,
        number_idx_2d_per_TokenType=idx_2d_per_type,
        number_chunk_slice_starts_per_type=_chunk_slice_starts(
            number_chunk_boundaries_per_type
        ),
        vc2_chunk_exponent_sidecar=vc2_chunk_exponent_sidecar,
        f128_is_nan_or_inf=f128_is_nan_or_inf,
    )


def _chunk_slice_starts(
    number_chunk_boundaries_per_type: dict[TokenType, np.ndarray],
) -> dict[TokenType, np.ndarray]:
    """Flatten the per-call_target chunk boundaries to their ``.start`` arrays.

    The columnar twin of :attr:`Stage3CallTarget.number_chunk_slices`'
    ``.start`` -- one ``int64[n_total_cts]`` per :class:`TokenType`, in the
    DFS call_target order :func:`_reconstruct_per_ct_boundaries` built. The
    per-call_target slice ``.start`` IS the abutting CSR boundary
    ``bnd[i]``, so the flat starts are the boundary array minus its final
    sentinel (``bnd[:-1]``). Exposed flat on :class:`Stage3Batch` so the
    vector dense path's number-sidecar concat reads the chunk-slice bases
    without re-walking the object tree.
    """
    return {
        token_type: np.asarray(boundaries[:-1], dtype=np.int64)
        for token_type, boundaries in number_chunk_boundaries_per_type.items()
    }


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

    GIL-released (B-S3): the per-kept-node Python gather loop (slicing
    ``expanded[1:surviving]`` + ``np.nonzero(real_mask)`` per node, then
    the segmented ``cumsum(is_real) - 1`` carrier walk) is now a single
    ``py.detach`` Rust kernel reading the flat :class:`DenseColumns`
    columns directly. The kernel reproduces the same arithmetic the numpy
    path did, in the same kept-DFS-then-stream order, byte-identically.
    """
    block_idx, signs = build_carrier_signs_kernel(
        np.ascontiguousarray(dense.expanded, dtype=np.uint16),
        np.ascontiguousarray(dense.extra_value_v2_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.extra_f128_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.node_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.real_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.is_negative_per_position, dtype=np.bool_),
        np.ascontiguousarray(dense.raw_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.surviving_token_count, dtype=np.int64),
        np.ascontiguousarray(dense.kept_node_index, dtype=np.int64),
        int(_NUMBER_BAND_LO_SHIFTED),
        int(_NUMBER_BAND_HI_SHIFTED),
    )
    return block_idx, signs


# ---------------------------------------------------------------------------
# Hierarchy assembly.
# ---------------------------------------------------------------------------


def _assemble_hierarchy(
    *,
    stage2: "Stage2Batch",
    inline_byte_slices: list[slice],
    identity_slices: "IdentitySlicesCSR",
    number_chunk_boundaries_per_type: dict[TokenType, np.ndarray],
) -> list[Stage3Section]:
    """Build the level-2 / level-3 / level-4 Stage3 wrappers.

    All per-call-target geometry is in DFS encounter order; we walk the
    stage-2 hierarchy in the same order so a single positional cursor
    lines up each call_target's lookups. The per-:class:`TokenType`
    chunk ranges arrive as CSR boundary arrays; the per-call_target
    ``slice`` objects :class:`Stage3CallTarget` exposes are materialised
    leaf-locally here (the only consumer that needs ``slice`` objects --
    the staged tree-walk -- and the only path that builds this hierarchy
    at all, ``build_hierarchy``), keeping them off the vector hot path.
    """
    ct_cursor = 0
    sections: list[Stage3Section] = []
    for stage2_section in stage2.sections:
        stage3_variants: list[Stage3Variant] = []
        for stage2_variant in stage2_section.variants:
            stage3_cts: list[Stage3CallTarget] = []
            for stage2_ct in stage2_variant.call_targets:
                per_type_slice: dict[TokenType, slice] = {
                    T: slice(
                        int(number_chunk_boundaries_per_type[T][ct_cursor]),
                        int(number_chunk_boundaries_per_type[T][ct_cursor + 1]),
                    )
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
