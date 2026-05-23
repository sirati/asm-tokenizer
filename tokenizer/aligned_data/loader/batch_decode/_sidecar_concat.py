"""Stage 4 helper -- per-row sidecar concatenation.

Single concern of this module: take a finalised :class:`Stage3Batch` and
produce the flat numbers sidecar pair the model consumer wants:

Numbers sidecar: ``(numbers_significant: u64[total_chunks],
numbers_sign_exponent: u32[total_chunks])`` indexed by
:attr:`Stage2Batch.number_row_offsets`. Chunks are concatenated in
row-major stream order -- the per-:class:`TokenType`
``(significand, sign_exp)`` chunk pairs that stage 3 staged in
``Stage3Batch.numbers_per_TokenType`` are interleaved exactly as the
surviving ``expanded_token_ids[:partial_cut_length]`` stream
discovers them.

What this module is NOT:

- It does not write into the token tensor or the identity sidecar --
  those concerns live in subagents 4a/4c.
- It does not renormalise floats -- the ``(significand, sign_exp)``
  arrays come pre-normalised from stage 3 (ALG-7).

Plan reference: ``batch_decode_plan.md`` Stage 4 step 4; D8 pins the
row-offset sizing contract.

Multi-row mapping note (RESAMPLE_WITHIN_SECTION / REDISTRIBUTE): when
one :class:`Stage1Variant` is referenced by multiple batch rows,
:attr:`Stage1Batch.batch_idx_to_section_variant` carries the same
``(section_idx, variant_idx)`` pair on each row. This module resolves
each batch row's source variant through that mapping, so a single
variant's chunks are emitted once per referencing row -- matching the
sizes already cumsum'd into ``stage2.number_row_offsets``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.tokens import TokenType

from ._row_expand import build_per_row_variant_lookup, concat_per_row

if TYPE_CHECKING:
    from ._types import Stage3Batch, Stage3CallTarget, Stage3Variant


__all__ = [
    "assemble_number_sidecars",
]


# ---------------------------------------------------------------------------
# Vocab constants (post-shift NUMBER block -- plan vocab table + Stage 2 ALG)
# ---------------------------------------------------------------------------


# Shifted ids 1..7 cover the NUMBER block (originals 257..263) in
# source-declaration order: VC2, F16, BF16, F32, F64, F80, F128. The
# original ids are NOT contiguous on :class:`TokenType` (VC2=17, F16=19,
# ...) -- the unified vocab v1 lays them out contiguously starting at 257
# per ``_V2_NUMBER_BLOCK_START`` (see plan "Vocab + wire format
# reference"). After the post-shift (``- 256``) they sit at 1..7.
_SHIFTED_NUMBER_BAND_LO = np.uint16(1)
_SHIFTED_NUMBER_BAND_HI = np.uint16(8)

_SHIFTED_ID_TO_TOKEN_TYPE: dict[int, TokenType] = {
    1: TokenType.VALUED_CONST_V2,
    2: TokenType.FLOAT16,
    3: TokenType.BFLOAT16,
    4: TokenType.FLOAT32,
    5: TokenType.FLOAT64,
    6: TokenType.FLOAT80,
    7: TokenType.FLOAT128,
}


# ---------------------------------------------------------------------------
# Numbers sidecar
# ---------------------------------------------------------------------------


def _emit_call_target_chunks(
    call_target: "Stage3CallTarget",
    numbers_per_TokenType: dict[
        TokenType, tuple[np.ndarray, np.ndarray]
    ],
    out_sig: np.ndarray,
    out_sex: np.ndarray,
    write_offset: int,
) -> int:
    """Write one call_target's surviving chunks into the row's output
    region; return the new ``write_offset``.

    Per ALG (Stage 4 step 4): walk
    ``expanded_token_ids[:partial_cut_length]``; at each number-band
    position pull the next chunk from
    ``numbers_per_TokenType[token_type]`` via the call_target's
    per-:class:`TokenType` slice. Chunks land in the output in stream
    order (not grouped by :class:`TokenType`).

    F128 mid-cut handling: 3c emits ALG-2 chunks INDEPENDENT of the
    cut (2 per finite source, 1 per NaN/Inf) so 3d can read
    ``actual_exp`` from the MSB limb. The stream-visible F128 count
    (``n_for_type``) is smaller than the per-CT FLOAT128 chunk slice
    only when a mid-cut finite source's painted MSB slot is past
    ``partial_cut_length``. That MSB chunk lives at the TAIL of the
    per-CT slice (3c emits chunks in stream emission order; LSB then
    MSB per finite source; the mid-cut source is by construction the
    LAST F128 in the CT), so taking the first ``n_for_type`` chunks
    naturally drops it.
    """

    stage2_ct = call_target.stage2
    partial_cut_length = stage2_ct.partial_cut_length
    if partial_cut_length == 0:
        return write_offset

    surviving = stage2_ct.expanded_token_ids[:partial_cut_length]
    number_mask = (surviving >= _SHIFTED_NUMBER_BAND_LO) & (
        surviving < _SHIFTED_NUMBER_BAND_HI
    )
    n_chunks = int(number_mask.sum())
    if n_chunks == 0:
        return write_offset

    local_token_types = surviving[number_mask]
    # row_segment indices in [0, n_chunks) parallel to the n_chunks
    # number positions (stream order within this call_target). We
    # SCATTER per-T into these positions.
    row_segment_sig = out_sig[write_offset : write_offset + n_chunks]
    row_segment_sex = out_sex[write_offset : write_offset + n_chunks]

    for shifted_id, token_type in _SHIFTED_ID_TO_TOKEN_TYPE.items():
        per_type_slice = call_target.number_chunk_slices.get(token_type)
        if per_type_slice is None:
            continue
        type_mask = local_token_types == np.uint16(shifted_id)
        n_for_type = int(type_mask.sum())
        if n_for_type == 0:
            continue
        slice_len = per_type_slice.stop - per_type_slice.start
        if slice_len < n_for_type:
            raise AssertionError(
                f"Stage3 number_chunk_slices[{token_type!r}] "
                f"(len={slice_len}) is shorter than the {n_for_type} "
                "surviving chunks implied by expanded_token_ids -- "
                "stage 3 sizing bug"
            )
        sig_for_type, sex_for_type = numbers_per_TokenType[token_type]
        take_start = per_type_slice.start
        take_stop = take_start + n_for_type
        row_segment_sig[type_mask] = sig_for_type[take_start:take_stop]
        row_segment_sex[type_mask] = sex_for_type[take_start:take_stop]

    return write_offset + n_chunks


def _emit_variant_chunks(
    stage3_variant: "Stage3Variant",
    numbers_per_TokenType: dict[TokenType, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-variant ``(sig, sex)`` arrays in stream-emission order.

    Single concern: walk one :class:`Stage3Variant`'s call_targets in
    encounter order and emit the surviving chunks into a fresh
    per-variant buffer pair. The buffer size is the variant's
    ``total_surviving_number_chunk_count`` (i.e. the per-variant cumsum
    expected by stage 2's row_offsets sizing). Each call_target's
    contribution lands at the right offset via the same
    :func:`_emit_call_target_chunks` kernel used pre-vectorization --
    the kernel writes per-:class:`TokenType` chunks into a row segment
    scattered by ``type_mask``, which is identical whether the row
    segment is a true per-row slice or a per-variant temporary buffer.
    """

    n_chunks = stage3_variant.stage2.total_surviving_number_chunk_count
    sig = np.empty(n_chunks, dtype=np.uint64)
    sex = np.empty(n_chunks, dtype=np.uint32)
    if n_chunks == 0:
        return sig, sex

    write_offset = 0
    for call_target in stage3_variant.call_targets:
        write_offset = _emit_call_target_chunks(
            call_target,
            numbers_per_TokenType,
            sig,
            sex,
            write_offset,
        )
    if write_offset != n_chunks:
        raise AssertionError(
            f"variant emitted {write_offset} chunks but stage 2 sized "
            f"the per-variant buffer at {n_chunks}; stage 2 / stage 3 "
            "sizing mismatch"
        )
    return sig, sex


def assemble_number_sidecars(
    stage3_batch: "Stage3Batch",
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-:class:`TokenType` ``(significand, sign_exp)``
    chunks into the global row-major number sidecars.

    Vectorized via :mod:`._row_expand`: per-variant ``(sig, sex)``
    buffers are built once per unique variant (no Python loop over
    ``batch_size``), then scattered to batch rows by the shared
    lookup + concat primitive. Padding rows (sentinel
    ``(UINT32_MAX, UINT32_MAX)``) contribute zero chunks; their
    expected ``number_row_offsets`` delta is checked against zero by
    the helper.

    Multi-row mapping: when one :class:`Stage1Variant` is referenced
    by multiple batch rows (RESAMPLE / REDISTRIBUTE), the variant's
    chunks are emitted once per referencing row -- stage 2's
    :attr:`Stage2Batch.number_row_offsets` already accounts for the
    duplication when sizing the output.

    Returns ``(numbers_significant: u64[total_chunks],
    numbers_sign_exponent: u32[total_chunks])``; total length equals
    ``int(stage2_batch.number_row_offsets[-1])``.
    """

    stage2_batch = stage3_batch.stage2
    stage1_batch = stage2_batch.stage1
    number_row_offsets = stage2_batch.number_row_offsets

    numbers_per_TokenType = stage3_batch.numbers_per_TokenType

    # Per-variant ``(sig, sex)`` arrays in flat ``section -> slot`` order.
    per_variant_sig: list[np.ndarray] = []
    per_variant_sex: list[np.ndarray] = []
    variants_per_section: list[int] = []
    for stage3_section in stage3_batch.sections:
        variants_per_section.append(len(stage3_section.variants))
        for stage3_variant in stage3_section.variants:
            sig, sex = _emit_variant_chunks(
                stage3_variant, numbers_per_TokenType
            )
            per_variant_sig.append(sig)
            per_variant_sex.append(sex)

    per_row_variant_idx, is_padding = build_per_row_variant_lookup(
        stage1_batch.batch_idx_to_section_variant, variants_per_section
    )

    sig_flat, _ = concat_per_row(
        per_variant_sig,
        per_row_variant_idx,
        is_padding,
        dtype=np.dtype(np.uint64),
        expected_row_offsets=number_row_offsets,
    )
    sex_flat, _ = concat_per_row(
        per_variant_sex,
        per_row_variant_idx,
        is_padding,
        dtype=np.dtype(np.uint32),
        expected_row_offsets=number_row_offsets,
    )
    return sig_flat, sex_flat
