"""Batched promotion + strip + expand twin over the flat gathered bodies.

Single concern: run the per-node ``build_inline_decode_state`` +
``expand_tokens`` MATH (VC2 / F128 promotion, strip + shift, the
prepended calling-category self-token) as a few vectorized numpy passes
over the WHOLE flat CSR ``raw`` body stream at once -- the batched twin
of the per-node scalar drive in :mod:`._expand`. This is to the expansion
what :func:`...batch_decode._bulk_expand_lengths.bulk_contributing_geometry`
is to the contributing-length scan: one boundary-aware pass instead of N
Python calls.

REUSE, NOT RE-IMPLEMENTATION (byte-identity contract): every rule this
module vectorizes is OWNED elsewhere and asserted equivalent by the
cross-check unit test + the corpus byte-identity gate --

* the per-stream masks + run-lengths + ``digit_cumsum`` +
  ``is_negative_per_position`` are the
  :class:`...decoded._inline_decode_state.InlineDecodeState` fields; this
  module reproduces them boundary-aware so each per-node SLICE equals
  :func:`build_inline_decode_state` on that node's raw stream.
* VC2 / F128 promotion + strip + shift + self-token prepend are owned by
  :func:`...batch_decode._expand_tokens.expand_tokens`; the per-source
  chunk-count + ALG-2 NaN/Inf detection + the ``> 256`` keep predicate +
  the ``- 256`` shift are reproduced here verbatim over the flat stream.

The malformed-stream guards (VC2 carrier at a node tail; F128 carrier
within 2 of a node tail) mirror the scalar asserts -- the same shapes
:func:`expand_tokens` rejects raise here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.token_manager import VocabularyManager


__all__ = ["BatchedExpansion", "batched_expand"]


# ---------------------------------------------------------------------------
# Unified-vocab layout constants -- the SAME source-of-truth the scalar
# kernels resolve (see ``_expand_tokens`` / ``_inline_decode_state`` /
# ``_bulk_expand_lengths``); a canonical-layout shift surfaces here as a
# constant-import update + a test cascade.
# ---------------------------------------------------------------------------
_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_V2_VALUE_NEGATIVE_TOKEN_ID = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID  # 256
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7
_V2_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272

_VC2_VOCAB_ID = _V2_NUMBER_BLOCK_START
_FLOAT128_VOCAB_ID = _V2_NUMBER_BLOCK_START + _V2_NUMBER_BLOCK_COUNT - 1


@dataclass(frozen=True)
class BatchedExpansion:
    """Flat batched expansion + the per-node STATE fields, all CSR.

    Every array spans the whole batch; per-node slices are taken by
    :mod:`._expand` (CSR ``rec`` for the raw-space arrays, CSR
    ``node_offsets`` for the expanded-space arrays). The state-field
    arrays are exactly the position-by-position
    :class:`...decoded._inline_decode_state.InlineDecodeState` fields,
    boundary-aware so each node slice equals the per-node build.
    """

    # Expanded (model-facing) stream + CSR jump table.
    expanded: np.ndarray  # uint16[total_expanded]
    node_offsets: np.ndarray  # int64[n_nodes + 1]

    # Promotion masks over the EXPANDED stream (slot 0 of each node = the
    # prepended self-token, always False), parallel to ``expanded``.
    extra_value_v2_mask: np.ndarray  # bool[total_expanded]
    extra_f128_mask: np.ndarray  # bool[total_expanded]

    # Per-position InlineDecodeState fields over the flat RAW stream
    # (CSR ``rec``). ``digit_cumsum`` is a per-node exclusive prefix with
    # an extra trailing slot per node -- size ``total_raw + n_nodes``,
    # node ``i``'s block at ``rec_starts[i] + i`` (see :func:`batched_expand`).
    real_mask: np.ndarray  # bool[total_raw]
    number_mask: np.ndarray  # bool[total_raw]
    runlen_number: np.ndarray  # uint16[total_raw]
    runlen_value: np.ndarray  # uint16[total_raw]
    carries_inline_mask: np.ndarray  # bool[total_raw]
    is_negative_per_position: np.ndarray  # bool[total_raw]
    digit_cumsum: np.ndarray  # uint32[total_raw + n_nodes]


def _boundary_run_lengths(
    mask: np.ndarray, rec_starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Per-node ``run_lengths`` over the concatenated flat ``mask``.

    Reproduces :func:`...decoded.run_lengths.run_lengths` applied to each
    node's slice independently: the run length is carried at each run's
    FIRST position (0 elsewhere), and runs never cross a node boundary.

    A run starts at ``k`` iff ``mask[k]`` and (``k`` is a node start OR
    ``not mask[k-1]``); a run ends at ``k`` iff ``mask[k]`` and
    (``not mask[k+1]`` OR ``k+1`` starts a new node). The length at a
    run-start is ``end - start + 1`` -- exactly the scalar output.
    """
    total = mask.shape[0]
    out = np.zeros(total, dtype=np.uint16)
    if total == 0:
        return out

    is_rec_start = np.zeros(total, dtype=bool)
    # Empty nodes collapse rec_starts onto the next node's start (or
    # ``total``); only non-empty nodes mark a real boundary.
    is_rec_start[rec_starts[counts > 0]] = True

    prev = np.empty(total, dtype=bool)
    prev[0] = False
    prev[1:] = mask[:-1]
    prev[is_rec_start] = False  # a boundary breaks the prior run
    run_start = mask & ~prev

    nxt = np.empty(total, dtype=bool)
    nxt[-1] = False
    nxt[:-1] = mask[1:]
    next_is_rec_start = np.zeros(total, dtype=bool)
    next_is_rec_start[:-1] = is_rec_start[1:]
    run_end = mask & (~nxt | next_is_rec_start)

    start_idx = np.nonzero(run_start)[0]
    end_idx = np.nonzero(run_end)[0]
    out[start_idx] = (end_idx - start_idx + 1).astype(np.uint16)
    return out


def _per_node_digit_cumsum(
    number_mask: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Per-node exclusive-prefix cumsum of ``number_mask``, packed CSR.

    The scalar ``build_inline_decode_state`` builds a length ``N + 1``
    array per node: ``digit_cumsum[0] = 0`` and ``digit_cumsum[k] =
    sum(number_mask[0:k])``. Packed here as one ``uint32`` array of size
    ``total_raw + n_nodes`` where node ``i``'s ``(count_i + 1)``-slot
    block starts at ``rec_starts[i] + i``; the slice
    ``out[rec_starts[i] + i : rec_starts[i + 1] + (i + 1)]`` is node
    ``i``'s ``digit_cumsum``.
    """
    total = number_mask.shape[0]
    out = np.zeros(total + n_nodes, dtype=np.uint32)
    if n_nodes == 0:
        return out

    # Global exclusive prefix (length total + 1): global_excl[k] =
    # sum(number_mask[0:k]).
    global_excl = np.zeros(total + 1, dtype=np.uint32)
    if total > 0:
        np.cumsum(number_mask.view(np.uint8), out=global_excl[1:])

    rec_ends = rec_starts + counts
    node_base = global_excl[rec_starts]  # digits before each node
    node_idx = np.arange(n_nodes, dtype=np.int64)
    dst_block_start = rec_starts + node_idx  # node i's block start

    # Body slots: per-node exclusive prefix at every raw position p is
    # global_excl[p] - node_base[node_of[p]]; dst index = p + node_of[p].
    if total > 0:
        node_of = np.repeat(node_idx, counts)
        raw_pos = np.arange(total, dtype=np.int64)
        out[raw_pos + node_of] = global_excl[:total] - node_base[node_of]
    # Trailing slot of each node (the ``N + 1``-th) = node's digit total.
    out[dst_block_start + counts] = global_excl[rec_ends] - node_base
    return out


def _batched_is_negative(
    *,
    runlen_number: np.ndarray,
    runlen_value: np.ndarray,
    carries_inline_mask: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
    total: int,
) -> np.ndarray:
    """Boundary-aware twin of ``compute_is_negative_per_position``.

    The scalar formula flags a carrier at ``p`` whose ``p+1`` slot is a
    sign marker via ``runlen_value[p+1] != runlen_number[p+1]``, and
    NEVER flags a carrier at a node's LAST position (no ``p+1`` slot).
    Reproduced over the flat stream by excluding each node's last
    position from the carrier candidates -- the run-length arrays are
    already boundary-local, so the ``p+1`` lookup stays inside the node.
    """
    out = np.zeros(total, dtype=bool)
    if total == 0:
        return out
    is_node_last = np.zeros(total, dtype=bool)
    is_node_last[(rec_starts + counts - 1)[counts > 0]] = True
    cand_idx = np.nonzero(carries_inline_mask & ~is_node_last)[0]
    if cand_idx.size:
        out[cand_idx] = (
            runlen_value[cand_idx + 1] != runlen_number[cand_idx + 1]
        )
    return out


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
    node_of: np.ndarray,
    rec_starts: np.ndarray,
    counts: np.ndarray,
    n_nodes: int,
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
    """
    total = working.shape[0]
    node_offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    if n_nodes == 0:
        return (
            np.zeros(0, dtype=np.uint16),
            node_offsets,
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
        )

    keep = working > _V2_RESERVED_DIGIT_COUNT  # bool[total]

    # Per-node surviving body length via cumsum-diff at the node bounds.
    surv_cum = np.zeros(total + 1, dtype=np.int64)
    if total > 0:
        np.cumsum(keep.view(np.uint8), out=surv_cum[1:])
    rec_ends = rec_starts + counts
    body_len = surv_cum[rec_ends] - surv_cum[rec_starts]  # int64[n_nodes]
    own_length = body_len + 1  # + the prepended self-token

    np.cumsum(own_length, out=node_offsets[1:])
    total_expanded = int(node_offsets[-1])

    expanded = np.empty(total_expanded, dtype=np.uint16)
    extra_value_v2_mask = np.zeros(total_expanded, dtype=bool)
    extra_f128_mask = np.zeros(total_expanded, dtype=bool)

    # Self-token at each node's slot 0 (both masks already False there).
    expanded[node_offsets[:-1]] = self_token_ids.astype(np.uint16, copy=False)

    # Surviving body ids land at slots [node_offset + 1 + body_rank]; the
    # rank within the node's surviving body is the local survivor cumsum.
    if total > 0:
        kept_idx = np.nonzero(keep)[0]
        node_k = node_of[kept_idx]
        body_rank = surv_cum[kept_idx] - surv_cum[rec_starts[node_k]]
        dst = node_offsets[node_k] + 1 + body_rank
        expanded[dst] = (
            working[kept_idx] - _V2_RESERVED_DIGIT_COUNT
        ).astype(np.uint16)
        extra_value_v2_mask[dst] = extra_vc2_raw[kept_idx]
        extra_f128_mask[dst] = extra_f128_raw[kept_idx]

    return expanded, node_offsets, extra_value_v2_mask, extra_f128_mask


def batched_expand(
    raw: np.ndarray,
    record_offsets: np.ndarray,
    self_token_ids: np.ndarray,
) -> BatchedExpansion:
    """Promote + strip + shift + prepend every node body, one batched pass.

    Parameters
    ----------
    raw:
        The flat gathered ``uint16`` body stream (CSR over ``record_offsets``).
    record_offsets:
        ``int64[n_nodes + 1]`` CSR into ``raw`` (node ``i`` owns
        ``raw[record_offsets[i] : record_offsets[i + 1]]``).
    self_token_ids:
        ``uint16[n_nodes]`` the SHIFTED calling-category self-token id for
        each node (the value the scalar ``expand_tokens`` writes at
        ``expanded[0]``); the ``CallTargetType -> Category -> shifted id``
        collapse is the caller's concern (:mod:`._expand`).

    Returns
    -------
    BatchedExpansion
        The flat expanded stream + CSR + promotion masks + the per-node
        InlineDecodeState fields, all batched.
    """
    raw = np.asarray(raw, dtype=np.uint16).reshape(-1)
    rec = np.asarray(record_offsets, dtype=np.int64).reshape(-1)
    n_nodes = rec.size - 1
    total = raw.shape[0]

    rec_starts = rec[:-1]
    rec_ends = rec[1:]
    counts = rec_ends - rec_starts
    node_of = (
        np.repeat(np.arange(n_nodes, dtype=np.int64), counts)
        if total
        else np.zeros(0, dtype=np.int64)
    )

    # --- per-position InlineDecodeState fields (boundary-aware) ----------
    real_mask = raw > _V2_VALUE_NEGATIVE_TOKEN_ID
    number_mask = raw < _V2_RESERVED_DIGIT_COUNT
    value_mask = ~real_mask
    runlen_number = _boundary_run_lengths(number_mask, rec_starts, counts)
    runlen_value = _boundary_run_lengths(value_mask, rec_starts, counts)
    carries_inline_mask = real_mask & (raw < _V2_EAGER_BLOCK_END)
    digit_cumsum = _per_node_digit_cumsum(
        number_mask, rec_starts, counts, n_nodes
    )
    is_negative_per_position = _batched_is_negative(
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        rec_starts=rec_starts,
        counts=counts,
        total=total,
    )

    # --- promotion (paint into a working copy of raw) --------------------
    working = raw.copy()
    extra_vc2_raw, extra_f128_raw = _promote_batched(
        working, real_mask, runlen_number, node_of, rec_starts, counts
    )

    # --- strip + shift + prepend self-token ------------------------------
    (
        expanded,
        node_offsets,
        extra_value_v2_mask,
        extra_f128_mask,
    ) = _strip_shift_prepend(
        working,
        extra_vc2_raw,
        extra_f128_raw,
        node_of,
        rec_starts,
        counts,
        n_nodes,
        self_token_ids,
    )

    return BatchedExpansion(
        expanded=expanded,
        node_offsets=node_offsets,
        extra_value_v2_mask=extra_value_v2_mask,
        extra_f128_mask=extra_f128_mask,
        real_mask=real_mask,
        number_mask=number_mask,
        runlen_number=runlen_number,
        runlen_value=runlen_value,
        carries_inline_mask=carries_inline_mask,
        is_negative_per_position=is_negative_per_position,
        digit_cumsum=digit_cumsum,
    )
