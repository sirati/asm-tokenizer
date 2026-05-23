"""Stage 3c -- per-:class:`TokenType` number ``idx_2d`` construction.

Single concern: lay out 2D gather-offset arrays into ``inline_bytes``
for the number-arm view-cast step (ALG-7 byte layouts + ALG-8 VC2
multi-chunk packing). The vectorised FP normalisation (denormal +
NaN/Inf branches that turn the u64 bit patterns into the ``(significand,
sign_exp)`` f96 sidecar shape) is 3d's concern.

Byte layouts (ALG-7)
--------------------

* F16 / BF16 -- 1 row of 2 bytes.
* F32        -- 1 row of 4 bytes.
* F64        -- 1 row of 8 bytes.
* F80        -- 1 row of 10 bytes (3d reshapes into 5 big-endian u16
  limbs for the explicit-leading-bit reassembly path).
* F128       -- ``(chunk_count, 8)`` rows. Finite sources (ALG-2)
  emit 2 chunks: LSB limb (bytes 8..15) then MSB limb (bytes 0..7).
  NaN/Inf sources emit 1 chunk = MSB limb (bytes 0..7); 3d uses the
  ``f128_is_nan_or_inf`` sidecar to branch into the
  ``_encode_infnan(sign, mantissa_is_zero)`` path that needs only the
  high limb's sign + exponent + high-mantissa bits. The dataloader's
  ".rodata-robustness" policy (custom_float.py:336) sanctions this
  approximation -- canonical NaN/Inf is fully determined by the high 8
  bytes.
* VC2 -- variable-length payload per ALG-8; ``(K_visible, 8)`` rows.
  ``K_full = max(1, ceil(L / 8))`` and ``K_visible`` is the count of
  chunks whose expanded-stream slot survived the cut. The MSB chunk
  may have fewer than 8 payload bytes when ``L % 8 != 0``; padding
  slots reference ``inline_bytes[0]`` (3a's leading-zero pad).

ALG-8 VC2 byte layout (verbatim from the plan)
----------------------------------------------

::

    # Per VC2 source at carrier position p_carrier (raw_tokens position):
    #   p_carrier_byte: source's first inline-byte offset in inline_bytes
    #                   (after the leading pad).
    #   L            : inline_len = state.runlen_number[p_carrier + 1].
    #   K            : max(1, ceil(L / 8))  # chunk count.
    #
    # Per chunk c (0 = LSB chunk, K-1 = MSB chunk):
    #   Payload bytes: [p_carrier_byte + L - 8*(c+1), p_carrier_byte + L - 8*c)
    #     intersected with [p_carrier_byte, p_carrier_byte + L).

Mid-cut sources: chunks emit in LSB-first stream-emission order. A cut
inside a multi-chunk source drops the trailing (MSB-end) chunks. The
carrier slot is chunk 0; continuation slots are chunks 1, 2, ...

Iteration order: sections -> variants -> call_targets in DFS encounter
order (the same linearisation stage 1 + 3a use). Within each
call_target's surviving prefix we walk the expanded-stream positions
and emit per-source rows in stream order. The per-call-target
``inline_byte_slice`` produced by 3a anchors the offsets we emit.

ALG-8 reads ``L = state.runlen_number[p_carrier + 1]``, so we still
need the raw-stream carrier position. The expanded stream alone
doesn't tell us ``L`` (only ``K``); two distinct ``L`` values can map
to the same ``K``. We rebuild the expanded->raw position map locally
per call_target.

Plan refs: ``batch_decode_plan.md`` ALG-7 + ALG-8 + Stage 3 step 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

if TYPE_CHECKING:
    from ._types import Stage2Batch


__all__ = ["build_number_idx_2d"]


# ---------------------------------------------------------------------------
# Band constants -- derived from the :class:`VocabularyManager` source of
# truth. NUMBER band post-shift = ``[1, 8)``; id 1 -> VC2, 2 -> F16,
# 3 -> BF16, 4 -> F32, 5 -> F64, 6 -> F80, 7 -> F128.
# ---------------------------------------------------------------------------

_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7

_NUMBER_BAND_LO_SHIFTED = _NUMBER_BLOCK_START - _RESERVED_DIGIT_COUNT  # 1
_NUMBER_BAND_HI_SHIFTED = (
    _NUMBER_BLOCK_START + _NUMBER_BLOCK_COUNT - _RESERVED_DIGIT_COUNT
)  # 8 (exclusive)


# Canonical NUMBER block ordering (plan vocab table + token_manager.py
# class docstring lines 94-97): VC2, F16, BF16, F32, F64, F80, F128.
# Indexed by ``shifted_id - 1`` (so shifted id 1 -> VC2 at index 0).
_NUMBER_BLOCK_TOKEN_TYPES: tuple[TokenType, ...] = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)
assert len(_NUMBER_BLOCK_TOKEN_TYPES) == _NUMBER_BLOCK_COUNT, (
    "_NUMBER_BLOCK_TOKEN_TYPES length must match VocabularyManager"
    "._V2_NUMBER_BLOCK_COUNT; a vocab-layout change touched one without "
    "the other."
)


# Per-TokenType byte width of a single source's full payload (matches
# the ALG-7 widths). For VC2 the payload is variable-length so this
# constant doesn't apply.
_FIXED_PAYLOAD_BYTES: dict[TokenType, int] = {
    TokenType.FLOAT16: 2,
    TokenType.BFLOAT16: 2,
    TokenType.FLOAT32: 4,
    TokenType.FLOAT64: 8,
    TokenType.FLOAT80: 10,
    TokenType.FLOAT128: 16,
}

# Per-TokenType row width (columns in idx_2d). F16/BF16/F32/F64/F80 emit
# 1 row covering the full payload; F128 emits ``chunk_count`` rows of
# 8 bytes each; VC2 emits ``K_visible`` rows of 8 bytes each.
_FIXED_ROW_WIDTH: dict[TokenType, int] = {
    TokenType.FLOAT16: 2,
    TokenType.BFLOAT16: 2,
    TokenType.FLOAT32: 4,
    TokenType.FLOAT64: 8,
    TokenType.FLOAT80: 10,
    TokenType.FLOAT128: 8,
    TokenType.VALUED_CONST_V2: 8,
}


def build_number_idx_2d(
    stage2_batch: "Stage2Batch",
    inline_bytes: np.ndarray,
    inline_byte_slices: list[slice],
) -> tuple[
    dict[TokenType, np.ndarray],
    dict[TokenType, list[slice]],
    np.ndarray,
    np.ndarray,
]:
    """Build per-:class:`TokenType` ``idx_2d`` arrays per ALG-7 + ALG-8.

    Walks every call_target in DFS encounter order, iterates surviving
    expanded-stream positions, and emits ``u32`` gather-offset rows per
    surviving number-source carrier into the per-:class:`TokenType`
    ``idx_2d`` array.

    Parameters
    ----------
    stage2_batch
        Per-call-target reads pull ``stage1.state`` (raw_tokens, masks,
        runlen_number) -- the byte-width / payload-length data lives
        in the ORIGINAL pre-promotion stream, while expansion identifies
        which positions are chunk-carriers.
    inline_bytes
        3a's flat ``u8`` buffer (index 0 = leading-zero pad). Not
        mutated here -- the rows we emit are gather offsets into it.
    inline_byte_slices
        Parallel to the DFS-flat call_target enumeration: entry ``i``
        is the range in ``inline_bytes`` owned by call_target ``i``.

    Returns
    -------
    idx_2d_per_type
        ``dict[TokenType, np.ndarray]`` with every NUMBER-block
        TokenType always present (empty types get zero-row arrays of
        the canonical row width).
    number_chunk_slices_per_type
        ``dict[TokenType, list[slice]]``; entry ``[T][i]`` is the
        slice into ``idx_2d_per_type[T]`` owned by call_target ``i``.
    f128_is_nan_or_inf
        ``bool[n_f128_sources]``. One entry per F128 SOURCE (not
        chunk); NaN/Inf sources contribute 1 row, finite contribute 2.
    vc2_chunk_exponent_sidecar
        ``u32[total_vc2_chunks]``. Per-chunk index within source
        (``0 = LSB``, ``K-1 = MSB``); stage 4 multiplies by 64 for
        ``exponent_base``.
    """

    # Output accumulators. Every NUMBER-block TokenType key is always
    # present so callers can index without a KeyError guard.
    row_lists_per_type: dict[TokenType, list[np.ndarray]] = {
        T: [] for T in _NUMBER_BLOCK_TOKEN_TYPES
    }
    chunk_slices_per_type: dict[TokenType, list[slice]] = {
        T: [] for T in _NUMBER_BLOCK_TOKEN_TYPES
    }
    running_counts: dict[TokenType, int] = {
        T: 0 for T in _NUMBER_BLOCK_TOKEN_TYPES
    }
    f128_nan_or_inf_flags: list[bool] = []
    vc2_chunk_indices: list[int] = []

    # DFS encounter order matches 3a's ``inline_byte_slices`` so a
    # single positional index lines up byte-slice + row-slice handoffs.
    ct_dfs_idx = 0
    for section in stage2_batch.sections:
        for variant in section.variants:
            for ct in variant.call_targets:
                # Per-TokenType row counts BEFORE processing this ct, to
                # produce the per-ct slice into the final concatenation.
                pre_counts = {
                    T: running_counts[T] for T in _NUMBER_BLOCK_TOKEN_TYPES
                }

                if ct.surviving_token_count > 0:
                    _emit_call_target_rows(
                        ct=ct,
                        inline_byte_slice=inline_byte_slices[ct_dfs_idx],
                        row_lists_per_type=row_lists_per_type,
                        running_counts=running_counts,
                        f128_nan_or_inf_flags=f128_nan_or_inf_flags,
                        vc2_chunk_indices=vc2_chunk_indices,
                    )

                for T in _NUMBER_BLOCK_TOKEN_TYPES:
                    chunk_slices_per_type[T].append(
                        slice(pre_counts[T], running_counts[T])
                    )

                ct_dfs_idx += 1

    idx_2d_per_type: dict[TokenType, np.ndarray] = {}
    for T in _NUMBER_BLOCK_TOKEN_TYPES:
        if row_lists_per_type[T]:
            idx_2d_per_type[T] = np.concatenate(
                row_lists_per_type[T], axis=0
            )
        else:
            idx_2d_per_type[T] = np.empty(
                (0, _FIXED_ROW_WIDTH[T]), dtype=np.uint32
            )

    f128_is_nan_or_inf = np.asarray(f128_nan_or_inf_flags, dtype=np.bool_)
    vc2_chunk_exponent_sidecar = np.asarray(
        vc2_chunk_indices, dtype=np.uint32
    )

    return (
        idx_2d_per_type,
        chunk_slices_per_type,
        f128_is_nan_or_inf,
        vc2_chunk_exponent_sidecar,
    )


# ---------------------------------------------------------------------------
# Per-call-target emission helper.
# ---------------------------------------------------------------------------


def _emit_call_target_rows(
    *,
    ct,  # Stage2CallTarget -- forward-ref to avoid a runtime import cycle.
    inline_byte_slice: slice,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    f128_nan_or_inf_flags: list[bool],
    vc2_chunk_indices: list[int],
) -> None:
    """Walk one call_target's surviving expanded stream and emit rows.

    Per-source emission is dispatched by :class:`TokenType` derived
    from the carrier's shifted vocab id. The helper mutates the
    accumulators in place.

    Per-source row construction uses ``np.arange`` over byte offsets;
    the only Python-level loop is the per-source dispatch. Per
    call_target the source count is small (single-digit to low
    hundreds), so the hot path is 3d's cross-source vectorisation, NOT
    this layout step.
    """

    state = ct.stage1.state
    raw_tokens = state.raw_tokens
    number_mask = state.number_mask
    runlen_number = state.runlen_number

    # Rebuild the expanded->raw position map. ``state.raw_tokens`` /
    # ``state.real_mask`` reflect the pre-promotion stream; the painted
    # continuation slots are NOT in real_mask but ARE in the kept set
    # from ``expand_tokens``'s strip step. The map walks real_positions
    # while injecting painted slots after each carrier when the
    # extra_*_mask is True.
    keep_raw_positions = _expanded_to_raw_position_map(
        state=state,
        extra_value_v2_mask=ct.extra_value_v2_mask,
        extra_f128_mask=ct.extra_f128_mask,
    )

    # Cumulative count of inline-digit positions up to (but not
    # including) each raw-stream position. Used to compute
    # ``p_carrier_byte`` for each surviving carrier without a per-source
    # search.  Length ``n_raw + 1`` so ``inline_cumsum[p] = number of
    # inline-digit bytes at raw positions [0, p)``.
    inline_cumsum = _build_inline_cumsum(number_mask)

    inline_slice_start = int(inline_byte_slice.start)

    expanded_token_ids = ct.expanded_token_ids
    extra_value_v2_mask = ct.extra_value_v2_mask
    extra_f128_mask = ct.extra_f128_mask
    surviving = int(ct.surviving_token_count)

    # The prepend slot (expanded[0]) holds an IDENTITY-band id; the
    # NUMBER-band predicate below filters it out naturally.
    expanded_idx = 0
    while expanded_idx < surviving:
        tok_id = int(expanded_token_ids[expanded_idx])
        is_number_carrier = (
            _NUMBER_BAND_LO_SHIFTED <= tok_id < _NUMBER_BAND_HI_SHIFTED
            and not bool(extra_value_v2_mask[expanded_idx])
            and not bool(extra_f128_mask[expanded_idx])
        )
        if not is_number_carrier:
            expanded_idx += 1
            continue

        token_type = _NUMBER_BLOCK_TOKEN_TYPES[
            tok_id - _NUMBER_BAND_LO_SHIFTED
        ]
        # expanded[0] = prepend (no raw counterpart); subtract 1 for
        # the raw-position lookup.
        p_carrier = int(keep_raw_positions[expanded_idx - 1])
        # First inline-digit byte of this source = cumulative
        # inline-digit count in raw[0..p_carrier], shifted by the
        # call_target's slice start.
        p_carrier_byte = inline_slice_start + int(inline_cumsum[p_carrier + 1])

        if token_type is TokenType.VALUED_CONST_V2:
            chunks_consumed = _emit_vc2_source(
                state_runlen_number=runlen_number,
                p_carrier=p_carrier,
                p_carrier_byte=p_carrier_byte,
                expanded_idx=expanded_idx,
                expanded_token_ids=expanded_token_ids,
                extra_value_v2_mask=extra_value_v2_mask,
                surviving=surviving,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
                vc2_chunk_indices=vc2_chunk_indices,
            )
        elif token_type is TokenType.FLOAT128:
            chunks_consumed = _emit_f128_source(
                p_carrier_byte=p_carrier_byte,
                expanded_idx=expanded_idx,
                extra_f128_mask=extra_f128_mask,
                surviving=surviving,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
                f128_nan_or_inf_flags=f128_nan_or_inf_flags,
            )
        else:
            # Fixed-width FP types (F16 / BF16 / F32 / F64 / F80).
            _emit_fixed_fp_source(
                p_carrier_byte=p_carrier_byte,
                token_type=token_type,
                row_lists_per_type=row_lists_per_type,
                running_counts=running_counts,
            )
            chunks_consumed = 1

        expanded_idx += chunks_consumed


# ---------------------------------------------------------------------------
# Per-source emission helpers.
# ---------------------------------------------------------------------------


def _emit_fixed_fp_source(
    *,
    p_carrier_byte: int,
    token_type: TokenType,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
) -> None:
    """Emit 1 row for a fixed-width FP source (F16/BF16/F32/F64/F80).

    Row is the contiguous byte range ``[p_carrier_byte, +width)``;
    3d's view-cast turns that into the big-endian ``u16`` / ``u32`` /
    ``u64`` / 5x``u16``-limb bit pattern per ALG-7.
    """
    width = _FIXED_PAYLOAD_BYTES[token_type]
    row = np.arange(
        p_carrier_byte, p_carrier_byte + width, dtype=np.uint32
    )[np.newaxis, :]
    row_lists_per_type[token_type].append(row)
    running_counts[token_type] += 1


def _emit_f128_source(
    *,
    p_carrier_byte: int,
    expanded_idx: int,
    extra_f128_mask: np.ndarray,
    surviving: int,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    f128_nan_or_inf_flags: list[bool],
) -> int:
    """Emit 1 or 2 rows for one F128 source.

    Finite source (ALG-2 painted continuation): 2 chunks -- LSB limb
    (bytes 8..15) then MSB limb (bytes 0..7). NaN/Inf source (no
    continuation): 1 chunk = MSB limb (bytes 0..7); 3d branches on
    ``f128_is_nan_or_inf`` to call ``_encode_infnan``.

    Mid-cut: if a finite source's chunk-1 slot is past the cut, emit
    only chunk 0 (LSB). The ``f128_is_nan_or_inf`` flag stays False
    (the source's nature is from ALG-2, not from how many chunks
    survived).

    Returns the number of expanded positions consumed (1 or 2).
    """
    # ALG-2's painted bit at ``expanded_idx + 1`` is the authoritative
    # finite/NaN-Inf signal -- read against the FULL mask, NOT clipped
    # to ``surviving``, so a mid-cut finite source still reports
    # is_finite=True. ``has_continuation`` separately gates whether
    # we ALSO emit chunk 1.
    has_continuation = (
        expanded_idx + 1 < surviving
        and bool(extra_f128_mask[expanded_idx + 1])
    )
    is_finite_source = (
        expanded_idx + 1 < extra_f128_mask.shape[0]
        and bool(extra_f128_mask[expanded_idx + 1])
    )

    if is_finite_source:
        f128_nan_or_inf_flags.append(False)
        row_lsb = np.arange(
            p_carrier_byte + 8, p_carrier_byte + 16, dtype=np.uint32
        )[np.newaxis, :]
        row_lists_per_type[TokenType.FLOAT128].append(row_lsb)
        running_counts[TokenType.FLOAT128] += 1

        if has_continuation:
            row_msb = np.arange(
                p_carrier_byte, p_carrier_byte + 8, dtype=np.uint32
            )[np.newaxis, :]
            row_lists_per_type[TokenType.FLOAT128].append(row_msb)
            running_counts[TokenType.FLOAT128] += 1
            return 2
        return 1

    # NaN/Inf source: 1 row (MSB limb, bytes 0..7).
    f128_nan_or_inf_flags.append(True)
    row_msb = np.arange(
        p_carrier_byte, p_carrier_byte + 8, dtype=np.uint32
    )[np.newaxis, :]
    row_lists_per_type[TokenType.FLOAT128].append(row_msb)
    running_counts[TokenType.FLOAT128] += 1
    return 1


def _emit_vc2_source(
    *,
    state_runlen_number: np.ndarray,
    p_carrier: int,
    p_carrier_byte: int,
    expanded_idx: int,
    expanded_token_ids: np.ndarray,
    extra_value_v2_mask: np.ndarray,
    surviving: int,
    row_lists_per_type: dict[TokenType, list[np.ndarray]],
    running_counts: dict[TokenType, int],
    vc2_chunk_indices: list[int],
) -> int:
    """Emit ``K_visible`` rows for one VC2 source per ALG-8.

    See module docstring (verbatim ALG-8 block) for the per-chunk
    byte-range formula. ``K_visible`` = 1 carrier + count of consecutive
    ``extra_value_v2_mask`` True positions immediately following, capped
    by both the surviving prefix and ``K_full``.

    Short MSB chunks left-pad with ``inline_bytes[0]`` references
    (zeros): for ``L=17`` the MSB chunk yields ``[0]*7 + [p_carrier_byte]``.

    Returns the number of expanded positions consumed (``K_visible``).
    """
    # ALG-8: ``L = state.runlen_number[p_carrier + 1]``. The carrier
    # always has a p+1 slot per _promote_vc2's tail assertion.
    L = int(state_runlen_number[p_carrier + 1])
    K_full = max(1, (L + 7) // 8)

    K_visible = 1
    while (
        K_visible < K_full
        and expanded_idx + K_visible < surviving
        and bool(extra_value_v2_mask[expanded_idx + K_visible])
    ):
        K_visible += 1

    # ALG-8 per-chunk byte range: ``[p_carrier_byte + L - 8*(c+1),
    # p_carrier_byte + L - 8*c)`` intersected with the payload region.
    # The intersection only matters for the MSB chunk (``c == K_full -
    # 1``, ``L % 8 != 0``); leading slots reference inline_bytes[0].
    for c in range(K_visible):
        unclipped_start = p_carrier_byte + L - 8 * (c + 1)
        unclipped_end = p_carrier_byte + L - 8 * c
        clipped_start = max(unclipped_start, p_carrier_byte)
        clipped_end = unclipped_end  # always <= p_carrier_byte + L
        n_actual_bytes = clipped_end - clipped_start
        n_pad_bytes = 8 - n_actual_bytes

        row = np.empty(8, dtype=np.uint32)
        if n_pad_bytes > 0:
            row[:n_pad_bytes] = 0
        if n_actual_bytes > 0:
            row[n_pad_bytes:] = np.arange(
                clipped_start, clipped_end, dtype=np.uint32
            )

        row_lists_per_type[TokenType.VALUED_CONST_V2].append(
            row[np.newaxis, :]
        )
        running_counts[TokenType.VALUED_CONST_V2] += 1
        vc2_chunk_indices.append(c)

    return K_visible


# ---------------------------------------------------------------------------
# Cross-call-target helpers.
# ---------------------------------------------------------------------------


def _expanded_to_raw_position_map(
    *,
    state,
    extra_value_v2_mask: np.ndarray,
    extra_f128_mask: np.ndarray,
) -> np.ndarray:
    """Recover the raw-stream position for each expanded[1:] slot.

    Painted VC2 / F128 continuation slots in 2a are contiguous in
    raw-space immediately after their carrier. We walk
    ``state.real_mask``'s nonzero positions for carriers (and other
    non-promoted real tokens); when the extra_*_mask flags a painted
    continuation, the painted slot's raw position is the prior slot's
    raw position + 1.

    Returns ``u32[predicted_full_length - 1]`` (i.e. one entry per
    expanded[1:] slot; expanded[0] = synthetic prepend has no raw
    counterpart).
    """
    real_positions = np.nonzero(state.real_mask)[0].astype(np.uint32)
    n_expanded_real = int(extra_value_v2_mask.shape[0]) - 1  # subtract prepend

    if n_expanded_real == 0:
        return np.empty(0, dtype=np.uint32)

    out = np.empty(n_expanded_real, dtype=np.uint32)

    # ``real_idx`` cursors over real_positions (raw-stream carriers +
    # other non-painted real tokens). When an extra_*_mask is True at
    # the current expanded slot, the painted continuation's raw
    # position = prior expanded slot's raw position + 1 (painted slots
    # are contiguous after their carrier).
    real_idx = 0
    for expanded_real_idx in range(n_expanded_real):
        is_extra = bool(
            extra_value_v2_mask[expanded_real_idx + 1]
            | extra_f128_mask[expanded_real_idx + 1]
        )
        if is_extra:
            out[expanded_real_idx] = out[expanded_real_idx - 1] + 1
        else:
            out[expanded_real_idx] = real_positions[real_idx]
            real_idx += 1

    return out


def _build_inline_cumsum(number_mask: np.ndarray) -> np.ndarray:
    """Cumulative inline-digit count: ``cumsum[p] = #digits in raw[0..p)``.

    Length ``len(number_mask) + 1``. The caller reads ``cumsum[p + 1]``
    to get the count of inline-digit bytes preceding raw position
    ``p + 1`` -- which is exactly the offset of the inline-digit byte
    at ``p + 1`` within the call_target's slice of ``inline_bytes``
    (3a guarantees no per-byte cut inside the slice; chunk-granularity
    cuts are handled by the caller).
    """
    n = int(number_mask.shape[0])
    cumsum = np.empty(n + 1, dtype=np.uint32)
    cumsum[0] = 0
    if n > 0:
        np.cumsum(number_mask.astype(np.uint32), out=cumsum[1:])
    return cumsum
