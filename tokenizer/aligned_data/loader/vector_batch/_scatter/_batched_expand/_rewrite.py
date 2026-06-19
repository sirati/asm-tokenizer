"""VC2 / F128 promotion + strip-shift-prepend over the flat raw stream.

Single concern: the raw-stream MUTATION half of the batched expand --
paint the VC2 / F128 continuation slots (:func:`_promote_batched`,
mirroring :func:`...batch_decode._expand_tokens._promote_vc2` /
``_promote_f128``) then strip ``<= 256`` slots, shift surviving ids down
by 256, and prepend each node's self-token (:func:`_strip_shift_prepend`,
mirroring :func:`expand_tokens` steps 3 + 4). The malformed-stream tail
guards raise the same shapes :func:`expand_tokens` rejects. The
boundary-aware state fields it reads come from :mod:`._state_fields`; the
orchestration from :mod:`._expansion`.
"""

from __future__ import annotations

import numpy as np
from dedup_hashmap import build_strip_shift_prepend_kernel

from ._constants import (
    _FLOAT128_VOCAB_ID,
    _V2_RESERVED_DIGIT_COUNT,
    _VC2_VOCAB_ID,
)


__all__ = ["_promote_batched", "_strip_shift_prepend"]


def _promote_batched(
    working: np.ndarray,
    real_mask: np.ndarray,
    runlen_number: np.ndarray,
    node_of: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
):
    """Paint VC2 + F128 continuation slots over the flat working stream.

    Returns ``(extra_vc2_raw, extra_f128_raw)`` boolean masks over the
    raw-stream index space, True at painted continuation slots. Mirrors
    :func:`...batch_decode._expand_tokens._promote_vc2` / ``_promote_f128``
    per node, batched: per-source chunk counts + flat painted indices,
    with the SAME malformed-stream tail guards (now node-local). Painted
    indices never cross a node boundary -- the bounds check uses the
    per-node tail.
    """
    total = working.shape[0]
    extra_vc2_raw = np.zeros(total, dtype=bool)
    extra_f128_raw = np.zeros(total, dtype=bool)
    if total == 0:
        return extra_vc2_raw, extra_f128_raw

    # --- VC2 promotion ---------------------------------------------------
    vc2_pos = np.nonzero(real_mask & (working == _VC2_VOCAB_ID))[0]
    if vc2_pos.size:
        node = node_of[vc2_pos]
        local = vc2_pos - rec_starts[node]
        if bool((local >= counts[node] - 1).any()):
            raise AssertionError(
                "VC2 carrier at the last raw-stream position -- malformed "
                "v2 stream (carrier needs a p+1 slot for the payload "
                "inline-digit run)."
            )
        payload_len = runlen_number[vc2_pos + 1].astype(np.int64)
        chunk_counts = np.maximum(np.int64(1), (payload_len + 7) // 8)
        node_tail = rec_starts[node] + counts[node]
        ends = vc2_pos.astype(np.int64) + chunk_counts
        if bool((ends > node_tail).any()):
            bad = int(np.nonzero(ends > node_tail)[0][0])
            p = int(vc2_pos[bad])
            raise AssertionError(
                f"VC2 carrier at position {p} declares "
                f"{int(chunk_counts[bad])} chunks but only "
                f"{int(node_tail[bad]) - p} raw-stream slots remain -- "
                "malformed v2 stream."
            )
        paint_lens = chunk_counts - np.int64(1)  # >= 0
        total_paint = int(paint_lens.sum())
        if total_paint > 0:
            base = np.repeat(vc2_pos.astype(np.int64) + 1, paint_lens)
            cum = np.cumsum(paint_lens)
            within = np.arange(total_paint, dtype=np.int64) - np.repeat(
                cum - paint_lens, paint_lens
            )
            flat = base + within
            working[flat] = _VC2_VOCAB_ID
            extra_vc2_raw[flat] = True

    # --- F128 promotion --------------------------------------------------
    f128_pos = np.nonzero(real_mask & (working == _FLOAT128_VOCAB_ID))[0]
    if f128_pos.size:
        node = node_of[f128_pos]
        local = f128_pos - rec_starts[node]
        if bool((local >= counts[node] - 2).any()):
            raise AssertionError(
                "F128 carrier within 2 positions of the raw-stream tail -- "
                "malformed v2 stream (ALG-2 needs the high u16 of the "
                "binary128 payload at p+1, p+2)."
            )
        high = working[f128_pos + 1].astype(np.uint16) << np.uint16(8)
        low = working[f128_pos + 2].astype(np.uint16)
        is_nan_or_inf = ((high | low) & np.uint16(0x7FFF)) == np.uint16(0x7FFF)
        finite = f128_pos[~is_nan_or_inf]
        if finite.size:
            targets = finite + 1
            working[targets] = _FLOAT128_VOCAB_ID
            extra_f128_raw[targets] = True

    return extra_vc2_raw, extra_f128_raw


def _strip_shift_prepend(
    working: np.ndarray,
    extra_vc2_raw: np.ndarray,
    extra_f128_raw: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
    self_token_ids: np.ndarray,
):
    """Strip ``<= 256`` slots, shift surviving ids, prepend self-tokens.

    Mirrors :func:`...batch_decode._expand_tokens.expand_tokens` steps
    3 + 4 batched: the keep predicate ``working > 256`` drops the
    inline-digit band + sign marker; surviving ids shift down by 256;
    each node's surviving body is prefixed with its shifted self-token.

    Returns ``(expanded, node_offsets, extra_value_v2_mask,
    extra_f128_mask)`` over the EXPANDED stream (slot 0 per node = the
    prepended self-token, both masks False there).

    The keep-filter + shift + scatter is performed by the GIL-released
    :func:`build_strip_shift_prepend_kernel` over the per-node CSR windows
    (``rec_starts`` + ``counts``); the kernel derives each survivor's
    intra-node rank from its window scan, so ``node_of`` is no longer read
    here (the per-node window IS the segmentation). The #92 per-node-length
    discipline is preserved: each node's body length is the count of
    survivors in its own raw window, so consecutive empty bodies never
    merge.
    """
    expanded, node_offsets, extra_value_v2_mask, extra_f128_mask = (
        build_strip_shift_prepend_kernel(
            np.ascontiguousarray(working, dtype=np.uint16),
            np.ascontiguousarray(extra_vc2_raw, dtype=bool),
            np.ascontiguousarray(extra_f128_raw, dtype=bool),
            np.ascontiguousarray(rec_starts, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64),
            np.ascontiguousarray(self_token_ids, dtype=np.uint16),
            int(_V2_RESERVED_DIGIT_COUNT),
        )
    )
    return expanded, node_offsets, extra_value_v2_mask, extra_f128_mask
