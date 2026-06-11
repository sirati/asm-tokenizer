"""Bulk contributing-length scan over raw ``_data.bin`` token regions.

Single concern: for a batch of records (token-region spans on the
``_data.bin`` uint8 array), compute each record's contributing BODY
length -- the number of positions the record's body occupies in the
post-promotion, post-strip expanded stream of :func:`._expand_tokens.
expand_tokens`. This is the bulk twin of the scalar expansion; the
expansion SEMANTICS are owned by :mod:`._expand_tokens` and any rule
change must land there first -- the cross-equivalence test
(``tests/test_bulk_expand_lengths.py``) pins the two paths to each
other.

Definition (mirrors ``expand_tokens`` step by step):

* every raw token ``> 256`` survives the strip (1 position each);
* every VC2 carrier (raw id 257) paints ``max(1, ceil(L / 8)) - 1``
  continuation slots, where ``L`` is the inline-digit run length
  (consecutive ``raw < 256``) starting exactly at the carrier's ``p+1``;
* every FINITE F128 carrier (raw id 263 whose payload high u16 masked
  with ``0x7fff`` differs from ``0x7fff``) paints exactly 1 slot;
* inline digits (``< 256``) and the sign marker (``== 256``) contribute
  nothing.

The PREPENDED self-token (always exactly 1 per spliced call target) is
deliberately NOT included: it is the callee walk's concern, not the
record's. A call target's ``predicted_full_length`` equals
``1 + bulk_contributing_body_lengths(...)[record]``.

Malformed-stream guards replicate the scalar asserts: a VC2 carrier at
a record's last position and an F128 carrier within 2 positions of a
record's tail both raise :class:`AssertionError`.

Memory: the scan walks records in chunks of ~``_CHUNK_TOKENS`` gathered
tokens, so peak working-set stays a few tens of MB regardless of corpus
size.
"""

from __future__ import annotations

import numpy as np

from tokenizer.token_manager import VocabularyManager


__all__ = ["bulk_contributing_body_lengths"]


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7

_VC2_VOCAB_ID = _V2_NUMBER_BLOCK_START
_FLOAT128_VOCAB_ID = _V2_NUMBER_BLOCK_START + _V2_NUMBER_BLOCK_COUNT - 1

#: Gathered-token budget per chunk (u16 tokens). 2**22 tokens ~= 8 MB
#: for the gathered stream plus a handful of same-shaped masks --
#: comfortably inside a worker's working set while keeping the numpy
#: dispatch count negligible.
_CHUNK_TOKENS = 1 << 22


def bulk_contributing_body_lengths(
    data_u8: np.ndarray,
    token_starts: np.ndarray,
    token_counts: np.ndarray,
) -> np.ndarray:
    """Contributing body length per record, vectorized + chunked.

    Parameters
    ----------
    data_u8:
        The ``_data.bin`` file as a 1-D uint8 array (typically a
        read-only memmap).
    token_starts / token_counts:
        Parallel integer arrays from
        :func:`~tokenizer.aligned_data.binary_format._bulk_geometry.
        bulk_token_spans` -- byte offset and u16 count of each record's
        raw token stream.

    Returns
    -------
    np.ndarray
        ``int64`` array parallel to the inputs; entry ``i`` is record
        ``i``'s contributing body length (see module docstring).

    Raises
    ------
    AssertionError
        On the same malformed-stream shapes the scalar expansion
        rejects (VC2 carrier at the record tail; F128 carrier within 2
        positions of the record tail).
    """
    starts = np.asarray(token_starts, dtype=np.int64).reshape(-1)
    counts = np.asarray(token_counts, dtype=np.int64).reshape(-1)
    if starts.shape != counts.shape:
        raise ValueError(
            f"token_starts and token_counts must be parallel; got "
            f"{starts.shape} vs {counts.shape}"
        )
    n_records = starts.size
    out = np.zeros(n_records, dtype=np.int64)
    if n_records == 0:
        return out

    # Chunk over records so that each chunk gathers at most
    # ~_CHUNK_TOKENS tokens. Records are never split across chunks --
    # the run-length and tail logic is per-record local.
    chunk_first = 0
    while chunk_first < n_records:
        chunk_last = chunk_first
        budget = 0
        while chunk_last < n_records:
            budget += int(counts[chunk_last])
            chunk_last += 1
            if budget >= _CHUNK_TOKENS:
                break
        sl = slice(chunk_first, chunk_last)
        out[sl] = _scan_chunk(data_u8, starts[sl], counts[sl])
        chunk_first = chunk_last
    return out


def _scan_chunk(
    data_u8: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Scan one chunk of records gathered into a flat token stream."""
    total = int(counts.sum())
    n_records = starts.size
    if total == 0:
        return np.zeros(n_records, dtype=np.int64)

    # record_of[k] / local byte index for the k-th gathered token.
    rec_ends = np.cumsum(counts)
    rec_starts = rec_ends - counts
    record_of = np.repeat(np.arange(n_records, dtype=np.int64), counts)
    within = np.arange(total, dtype=np.int64) - rec_starts[record_of]
    byte_idx = starts[record_of] + 2 * within

    # Raw u16 stream (byte-pair gather: no alignment assumption on the
    # token region's absolute offset).
    raw = data_u8[byte_idx].astype(np.int64) | (
        data_u8[byte_idx + 1].astype(np.int64) << 8
    )

    # --- survivors of the strip ----------------------------------------
    survives = raw > _V2_RESERVED_DIGIT_COUNT
    kept_per_record = np.bincount(
        record_of, weights=survives, minlength=n_records
    ).astype(np.int64)

    # --- digit-run lengths starting at each position --------------------
    # is_digit runs must not leak across record boundaries: a run starts
    # at k iff is_digit[k] and (k is a record start or not is_digit[k-1]).
    is_digit = raw < _V2_RESERVED_DIGIT_COUNT
    is_rec_start = np.zeros(total, dtype=bool)
    # Zero-token records collapse to rec_starts == rec_ends; such a
    # start can equal ``total`` (or alias the NEXT record's start), so
    # only non-empty records mark a boundary.
    is_rec_start[rec_starts[counts > 0]] = True
    prev_digit = np.empty(total, dtype=bool)
    prev_digit[0] = False
    prev_digit[1:] = is_digit[:-1]
    prev_digit[is_rec_start] = False
    run_start = is_digit & ~prev_digit
    # Run length per run via start/end positions; runs also end at
    # record boundaries.
    next_digit = np.empty(total, dtype=bool)
    next_digit[-1] = False
    next_digit[:-1] = is_digit[1:]
    # Position k is a run END iff digit and (next is not a digit or next
    # position starts a new record).
    next_is_rec_start = np.zeros(total, dtype=bool)
    next_is_rec_start[:-1] = is_rec_start[1:]
    run_end = is_digit & (~next_digit | next_is_rec_start)
    run_start_idx = np.nonzero(run_start)[0]
    run_end_idx = np.nonzero(run_end)[0]
    # run_len_at[k] = run length if a digit run starts at k, else 0.
    run_len_at = np.zeros(total, dtype=np.int64)
    run_len_at[run_start_idx] = run_end_idx - run_start_idx + 1

    # --- VC2 promotion ---------------------------------------------------
    vc2_pos = np.nonzero(raw == _VC2_VOCAB_ID)[0]
    vc2_extra = np.zeros(n_records, dtype=np.int64)
    if vc2_pos.size:
        # Tail guard: carrier needs a p+1 slot INSIDE its own record.
        local = vc2_pos - rec_starts[record_of[vc2_pos]]
        at_tail = local >= counts[record_of[vc2_pos]] - 1
        if bool(at_tail.any()):
            raise AssertionError(
                "VC2 carrier at the last raw-stream position -- malformed "
                "v2 stream (carrier needs a p+1 slot for the payload "
                "inline-digit run)."
            )
        payload_len = run_len_at[vc2_pos + 1]
        chunks = np.maximum(np.int64(1), (payload_len + 7) // 8)
        np.add.at(vc2_extra, record_of[vc2_pos], chunks - 1)

    # --- F128 promotion --------------------------------------------------
    f128_pos = np.nonzero(raw == _FLOAT128_VOCAB_ID)[0]
    f128_extra = np.zeros(n_records, dtype=np.int64)
    if f128_pos.size:
        local = f128_pos - rec_starts[record_of[f128_pos]]
        near_tail = local >= counts[record_of[f128_pos]] - 2
        if bool(near_tail.any()):
            raise AssertionError(
                "F128 carrier within 2 positions of the raw-stream tail -- "
                "malformed v2 stream (ALG-2 needs the high u16 of the "
                "binary128 payload at p+1, p+2)."
            )
        high_u16 = (raw[f128_pos + 1] << 8) | raw[f128_pos + 2]
        finite = (high_u16 & 0x7FFF) != 0x7FFF
        np.add.at(f128_extra, record_of[f128_pos[finite]], 1)

    return kept_per_record + vc2_extra + f128_extra
