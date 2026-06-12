"""Dedup-aware realized-length compute for one arm.

Single concern: turn a flat per-variant record-offset column + the
arm's ``_data.bin`` bytes into a flat per-variant ``u32`` realized-length
column, computing each DISTINCT record offset's length exactly once.

Realized length per record is the contributing BODY length computed by
:func:`...loader.batch_decode._bulk_expand_lengths.
bulk_contributing_body_lengths` over the record spans from
:func:`...binary_format._bulk_geometry.bulk_token_spans` -- the ONLY
length engine (survivor tokens, VC2 multi-chunk extras, finite-F128
extras; the self/identity token is deliberately excluded, exactly as the
bulk engine defines it). This module never reimplements that arithmetic;
it only handles the dedup + chunking around it.

Boundary contract (the design-first sentence):

  *Given the arm's data bytes + the per-variant record offsets, return
  the per-variant realized lengths -- deduping repeated offsets through
  the build-time Rust hashmap so each unique record is measured once,
  in bounded chunks so peak memory stays flat on multi-GB corpora.*

Why the Rust hashmap (and not ``np.unique``): many variants share one
identical tokenization -> one shared record (write-time dedup in
memmap_builder). The hashmap maps record offset (u64) -> realized length
(u32) so a chunk's already-measured offsets are answered by lookup; only
the chunk's NEWLY-seen unique offsets reach the bulk length engine. The
map is build-time only -- discarded once the column is filled; readers
never touch it. One map per arm (matched / unmatched have separate data
files, so their offsets must never be conflated -- that conflation would
be a correctness bug, which is why the map is constructed per call).
"""

from __future__ import annotations

import numpy as np

from dedup_hashmap import HashMapU64U32

from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)
from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
    bulk_contributing_body_lengths,
)

from ._format import MAX_REALIZED_LENGTH


__all__ = ["realized_lengths_for_offsets"]


#: Record offsets resolved per chunk. Each chunk does one hashmap lookup
#: + (for the chunk's misses) one bulk geometry/length pass + one insert,
#: so the working set is a few same-shaped int64 columns of this size
#: plus the bulk engine's own bounded scratch -- flat regardless of how
#: many variants the arm carries.
_CHUNK_OFFSETS = 1 << 20

#: ``HashMapU64U32`` lookup sentinel for an absent key.
_U32_MISS = np.uint32(0xFFFFFFFF)


def realized_lengths_for_offsets(
    data_u8: np.ndarray, record_offsets: np.ndarray
) -> np.ndarray:
    """Per-variant realized length, deduped over distinct record offsets.

    Parameters
    ----------
    data_u8:
        The arm's ``_data.bin`` as a 1-D ``uint8`` array (typically a
        read-only ``np.memmap``). Only record headers + token regions
        are paged in, in bounded chunks.
    record_offsets:
        Integer array of per-variant real ``_data.bin`` byte offsets
        (``data_offset_shifted << 4``); any integer dtype, flattened.

    Returns
    -------
    np.ndarray
        ``u32`` array parallel to ``record_offsets``; entry ``i`` is the
        realized body length of the record at ``record_offsets[i]``.

    Raises
    ------
    OverflowError
        If any realized length exceeds the ``u32`` range. The pass
        hard-errors (never clamps) so an out-of-range length is surfaced
        rather than silently corrupted.
    """
    offsets = np.asarray(record_offsets, dtype=np.int64).reshape(-1)
    n = offsets.size
    out = np.empty(n, dtype=np.uint32)
    if n == 0:
        return out

    keys_all = offsets.astype(np.uint64)
    # Capacity sized to the worst case (every offset distinct); the map
    # is discarded at function exit so the transient over-allocation is
    # bounded by the arm's variant count, not the corpus.
    table = HashMapU64U32(capacity=int(n * 2) + 1)

    for lo in range(0, n, _CHUNK_OFFSETS):
        hi = min(lo + _CHUNK_OFFSETS, n)
        keys = keys_all[lo:hi]
        found = table.lookup_ndarray(keys)
        miss = found == _U32_MISS
        if bool(miss.any()):
            miss_keys = keys[miss]
            uniq = np.unique(miss_keys)
            lengths = _measure(data_u8, uniq)
            table.insert_ndarray(uniq, lengths)
            # Re-resolve the whole chunk now that every miss is inserted.
            found = table.lookup_ndarray(keys)
        out[lo:hi] = found

    return out


def _measure(data_u8: np.ndarray, unique_offsets: np.ndarray) -> np.ndarray:
    """Realized body length per UNIQUE record offset, via the bulk engine.

    Returns a ``u32`` array parallel to ``unique_offsets``. Raises
    :class:`OverflowError` if any length exceeds the ``u32`` range
    (hard-error, never clamp -- the sidecar dtype is ``u32`` and a
    silently-truncated length would be a correctness bug).
    """
    starts, counts = bulk_token_spans(
        data_u8, unique_offsets.astype(np.int64)
    )
    body = bulk_contributing_body_lengths(data_u8, starts, counts)
    # Raise BEFORE the uint32 cast (so the OverflowError surfaces the true
    # value, never a wrapped one). ``0xFFFFFFFF`` is the reserved hashmap
    # miss sentinel, so the largest storable length is ``MAX_REALIZED_LENGTH``
    # (== 0xFFFFFFFE); any length >= the sentinel hard-errors, never clamps.
    if body.size and int(body.max()) >= int(_U32_MISS):
        worst = int(body.max())
        raise OverflowError(
            f"realized length {worst} exceeds the u32 sidecar range "
            f"({MAX_REALIZED_LENGTH}); the lengths sidecar cannot store it "
            f"(0xFFFFFFFF is reserved as the hashmap miss sentinel)"
        )
    return body.astype(np.uint32)
