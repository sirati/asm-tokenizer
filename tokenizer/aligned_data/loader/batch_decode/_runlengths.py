"""Per-function metatoken runlength helper for ``batch_decode``'s
optional ``emit_block_n_insns_runlength`` sidecar.

Single concern: take ONE :class:`FunctionData` and produce
``(block_runlength, insn_runlength)`` per the post-decode slot layout,
using the SAME helper :class:`FunctionTokenList` consumes when
reconstructing block / instruction structure from raw runlengths
(:func:`CA_BArle_to_CBrle`). The only addition on top of FTL's accounting
is the per-metatoken slot-count scaling for multi-chunk sources
(F128 finite -> 2 slots, VC2 K-chunk -> K slots), which mirrors stage 2's
``_expand_tokens`` promotion logic.

This module is the canonical computation: stage 4's per-row concatenator
calls :func:`compute_metatoken_runlengths` once per call_target, and the
public :func:`manual_calc_block_n_insn_runlengths` re-exports the same
function under the user-asked name so cross-check tests + future
analyzers can verify against ``batch_decode``'s emitted output.

Outputs are PURELY function-local: the row-level variant_tokens prefix
and the per-call-target self-prepend slot are NOT included here -- they
are emission-layer concerns owned by the row assembler (see
:mod:`._assemble` / :mod:`._token_assembly`). The function-body
accountant is symmetric across root + inlined callees.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.token_manager import VocabularyManager
from tokenizer.utils import CA_BArle_to_CBrle


__all__ = [
    "compute_metatoken_runlengths",
    "manual_calc_block_n_insn_runlengths",
]


# Vocab anchors (mirrors :mod:`._expand_tokens` -- same source of truth).
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7
_VC2_VOCAB_ID = _V2_NUMBER_BLOCK_START
_FLOAT128_VOCAB_ID = _V2_NUMBER_BLOCK_START + _V2_NUMBER_BLOCK_COUNT - 1


def _per_metatoken_slot_counts(
    raw_tokens: np.ndarray,
    boundary_positions: np.ndarray,
    metatoken_run_length: np.ndarray,
) -> np.ndarray:
    """Per-metatoken post-decode slot count (1 default, K for VC2,
    1-or-2 for F128).

    Walks the per-metatoken first-id stream and applies the same
    promotion rules :func:`._expand_tokens.expand_tokens` uses:

    * VC2 (``raw_tokens[p] == _VC2_VOCAB_ID``): chunk_count =
      ``max(1, ceil(payload_runlen / 8))`` where ``payload_runlen`` is
      the count of inline-digit followers (= ``metatoken_run_length[i] - 1``).
    * F128 (``raw_tokens[p] == _FLOAT128_VOCAB_ID``): chunk_count = 1
      if the high u16 of the payload is all-ones (NaN/Inf, sign bit
      stripped via ``& 0x7FFF``), else 2.
    * Everything else: 1 slot.

    Tail-bound checks for F128 (need raw_tokens[p+1] AND raw_tokens[p+2])
    mirror :func:`._expand_tokens._promote_f128` -- a malformed payload
    surfaces as an :class:`AssertionError` rather than silent OOB reads.
    """
    n_metatokens = int(boundary_positions.size)
    slot_counts = np.ones(n_metatokens, dtype=np.uint32)
    if n_metatokens == 0:
        return slot_counts

    first_ids = raw_tokens[boundary_positions]

    # VC2 promotion: chunk_count = max(1, ceil(payload_runlen / 8))
    vc2_mask = first_ids == _VC2_VOCAB_ID
    if bool(vc2_mask.any()):
        # payload_runlen = metatoken_run_length[i] - 1 (drop the carrier slot).
        payload_lengths = metatoken_run_length[vc2_mask].astype(np.int64) - np.int64(1)
        # max(0, ...) guards the (theoretical) zero-payload carrier.
        payload_lengths = np.maximum(np.int64(0), payload_lengths)
        chunk_counts = np.maximum(np.int64(1), (payload_lengths + 7) // 8)
        slot_counts[vc2_mask] = chunk_counts.astype(np.uint32)

    # F128 promotion: 1 chunk (NaN/Inf) or 2 chunks (finite). Mirrors
    # ALG-2 high-u16 sign-bit-stripped all-ones detection.
    f128_mask = first_ids == _FLOAT128_VOCAB_ID
    if bool(f128_mask.any()):
        f128_positions = boundary_positions[f128_mask]
        n = int(raw_tokens.shape[0])
        # Bounds: ALG-2 reads p+1 + p+2. A carrier in the last 2 positions
        # of the stream is malformed.
        if f128_positions.size > 0 and int(f128_positions[-1]) >= n - 2:
            raise AssertionError(
                "F128 carrier within 2 positions of the raw-stream tail -- "
                "malformed v2 stream (ALG-2 needs the high u16 of the "
                "binary128 payload at p+1, p+2)."
            )
        high_bytes = raw_tokens[f128_positions + 1].astype(np.uint16) << np.uint16(8)
        low_bytes = raw_tokens[f128_positions + 2].astype(np.uint16)
        high_u16 = high_bytes | low_bytes
        is_nan_or_inf = (high_u16 & np.uint16(0x7FFF)) == np.uint16(0x7FFF)
        # Finite -> 2 slots; NaN/Inf -> 1 slot (default).
        f128_slot_counts = np.where(
            is_nan_or_inf, np.uint32(1), np.uint32(2)
        ).astype(np.uint32)
        slot_counts[f128_mask] = f128_slot_counts

    return slot_counts


def compute_metatoken_runlengths(
    function_data: FunctionData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(block_runlength, insn_runlength)`` per the post-decode
    slot layout for ONE function body.

    ``block_runlength`` is ``u32[block_count]`` -- number of instructions
    in each block. ``insn_runlength`` is ``u32[insn_count]`` -- post-
    promotion slot count for each instruction (sum of the instruction's
    metatokens' slot counts, with F128 finite -> 2, F128 NaN/Inf -> 1,
    VC2 K-chunk -> K, everything else -> 1).

    Algorithm mirrors :meth:`FunctionTokenList.reconstruct_func_from_raw_bytes`:

    1. v2 boundary detection: metatoken boundaries are positions where
       ``raw_tokens >= 256`` (inline-digit ids 0..255 always continue the
       previous metatoken).
    2. Per-metatoken runlength: gaps between consecutive boundaries
       (last gap runs to the stream end).
    3. ``insn_metatoken_run_lengths = CA_BArle_to_CBrle(insn_idx_runlength,
       metatoken_run_length)`` -- the same helper FTL uses to fold the
       raw-token-per-instruction count + raw-token-per-metatoken count
       into metatokens-per-instruction.
    4. ``block_insn_run_lengths = CA_BArle_to_CBrle(block_idx_runlength,
       insn_idx_runlength)`` -- folds raw-tokens-per-block + raw-tokens-
       per-insn into instructions-per-block.
    5. Per-metatoken slot count (post-promotion) via
       :func:`_per_metatoken_slot_counts`.
    6. Per-instruction slot count = sum of slot counts over the
       instruction's metatokens (vectorized via cumulative reduce).

    The function-local accountant produces NO row-level prefix and NO
    self-prepend slot; those are emission-layer concerns owned by the
    row assembler.

    Parameters
    ----------
    function_data:
        :class:`FunctionData` with raw ``tokens``, ``insn_runlength`` (per-
        instruction RAW-token count), and ``block_runlength`` (per-block
        RAW-token count) fields. ``variant_tokens`` is not consumed here.

    Returns
    -------
    (block_runlength, insn_runlength):
        ``u32[block_count]``, ``u32[insn_count]``. Both empty for an
        empty function.
    """
    raw_tokens = function_data.tokens
    insn_idx_runlength = function_data.insn_runlength
    block_idx_runlength = function_data.block_runlength

    if raw_tokens.size == 0:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
        )

    # v2 metatoken boundaries: ids >= 256 are real carriers; ids 0..255
    # are inline-digit followers. See FTL.reconstruct_func_from_raw_bytes.
    boundary_positions = np.flatnonzero(raw_tokens >= _V2_RESERVED_DIGIT_COUNT).astype(
        np.int64
    )
    n_metatokens = int(boundary_positions.size)

    metatoken_run_length = np.empty(n_metatokens, dtype=np.int64)
    if n_metatokens > 0:
        metatoken_run_length[:-1] = (
            boundary_positions[1:] - boundary_positions[:-1]
        )
        metatoken_run_length[-1] = int(raw_tokens.size) - int(
            boundary_positions[-1]
        )

    # Fold raw-runlengths into metatoken-level + instruction-level
    # counts via FTL's helper (the canonical source of truth).
    insn_metatoken_run_lengths = CA_BArle_to_CBrle(
        insn_idx_runlength.astype(np.int64), metatoken_run_length
    )
    block_insn_run_lengths = CA_BArle_to_CBrle(
        block_idx_runlength.astype(np.int64),
        insn_idx_runlength.astype(np.int64),
    )

    # Per-metatoken post-promotion slot count.
    per_metatoken_slot_count = _per_metatoken_slot_counts(
        raw_tokens, boundary_positions, metatoken_run_length
    )

    # Per-instruction slot count = sum of slot counts over the
    # instruction's metatokens. Vectorized via cumulative bucketization:
    # split per_metatoken_slot_count at insn_metatoken_run_lengths
    # boundaries and sum each chunk.
    insn_slot_count = _segment_sum(
        per_metatoken_slot_count, insn_metatoken_run_lengths
    )

    return (
        block_insn_run_lengths.astype(np.uint32),
        insn_slot_count.astype(np.uint32),
    )


def _segment_sum(
    values: np.ndarray, segment_lengths: np.ndarray
) -> np.ndarray:
    """Sum ``values`` in contiguous chunks of size ``segment_lengths[i]``.

    Returns ``u32[len(segment_lengths)]`` where output[i] is the sum of
    ``values[start_i : start_i + segment_lengths[i]]``. Empty segments
    (``segment_lengths[i] == 0``) produce 0.

    Vectorized via ``np.cumsum`` + segment-end indexing. No Python loop
    over instructions.
    """
    n_segments = int(segment_lengths.size)
    if n_segments == 0:
        return np.zeros(0, dtype=np.uint32)

    seg_lens = segment_lengths.astype(np.int64)
    # Segment-end positions (exclusive) in ``values``: cumsum of lengths.
    seg_ends = np.cumsum(seg_lens, dtype=np.int64)
    # Running sum of values; prepend 0 for the empty-prefix base.
    running = np.empty(values.size + 1, dtype=np.uint64)
    running[0] = 0
    if values.size > 0:
        np.cumsum(values.astype(np.uint64), out=running[1:])
    # Segment sum = running[seg_ends] - running[seg_starts]; segment
    # starts = seg_ends shifted right by one with a leading 0.
    seg_starts = np.empty(n_segments, dtype=np.int64)
    seg_starts[0] = 0
    seg_starts[1:] = seg_ends[:-1]
    return (running[seg_ends] - running[seg_starts]).astype(np.uint32)


# Public alias: the user-asked name for the canonical computation that
# batch_decode invokes when ``emit_block_n_insns_runlength=True``.
# Cross-check tests + future analyzers verify against this entry point.
manual_calc_block_n_insn_runlengths = compute_metatoken_runlengths
