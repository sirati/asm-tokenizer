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
from dedup_hashmap import (
    build_promote_batched_kernel,
    build_strip_shift_prepend_kernel,
)

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

    Returns ``(working_painted, extra_vc2_raw, extra_f128_raw)``: a FRESH
    painted copy of ``working`` (the input is not mutated) plus the two
    boolean masks over the raw-stream index space, True at painted
    continuation slots. Mirrors
    :func:`...batch_decode._expand_tokens._promote_vc2` / ``_promote_f128``
    per node, batched, with the SAME malformed-stream tail guards (now
    node-local). Painted indices never cross a node boundary -- the bounds
    check uses the per-node tail.

    The per-carrier ceil-div chunk count + node-local bounds guards +
    segment paint and the F128 NaN/Inf finite filter are performed by the
    GIL-released :func:`build_promote_batched_kernel`; the malformed-stream
    guards surface as :class:`AssertionError` (the same contract the scalar
    twin raises).
    """
    working_painted, extra_vc2_raw, extra_f128_raw = (
        build_promote_batched_kernel(
            np.ascontiguousarray(working, dtype=np.uint16),
            np.ascontiguousarray(real_mask, dtype=bool),
            np.ascontiguousarray(runlen_number, dtype=np.uint16),
            np.ascontiguousarray(node_of, dtype=np.int64),
            np.ascontiguousarray(rec_starts, dtype=np.int64),
            np.ascontiguousarray(counts, dtype=np.int64),
            int(_VC2_VOCAB_ID),
            int(_FLOAT128_VOCAB_ID),
        )
    )
    return working_painted, extra_vc2_raw, extra_f128_raw


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
