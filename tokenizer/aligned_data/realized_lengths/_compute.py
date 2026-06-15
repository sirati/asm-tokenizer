"""Dedup-aware realized-GEOMETRY compute for one arm.

Single concern: turn a flat per-variant record-offset column + the
arm's ``_data.bin`` bytes into the flat per-variant ``u32`` geometry
triple ``(body_len, id_count, value_count)`` -- measuring each DISTINCT
record offset exactly once.

Realized geometry per record is the contributing triple computed by
:func:`...loader.batch_decode._bulk_expand_lengths.
bulk_contributing_geometry` over the record spans from
:func:`...binary_format._bulk_geometry.bulk_token_spans` -- the ONLY
geometry engine (body length, identity-carrier count, numeric-chunk
count; the self/identity token is deliberately excluded, exactly as the
bulk engine defines it). This module never reimplements that arithmetic;
it only handles the dedup + chunking around it.

Boundary contract (the design-first sentence):

  *Given the arm's data bytes + the per-variant record offsets, return
  the per-variant geometry triple -- deduping repeated offsets through
  the build-time Rust hashmap so each unique record is measured once, in
  bounded chunks so peak memory stays flat on multi-GB corpora.*

Why the Rust hashmap (and not ``np.unique``): many variants share one
identical tokenization -> one shared record (write-time dedup in
memmap_builder). One scalar ``u32`` value per offset is no longer enough
to carry the whole triple, so the hashmap maps record offset (u64) -> a
``u32`` ROW-INDEX into the growing unique-record accumulator columns; a
chunk's already-measured offsets are answered by lookup + gather, and
only the chunk's NEWLY-seen unique offsets reach the bulk geometry
engine (one scan -> all three axes). The map is build-time only --
discarded once the columns are filled; readers never touch it. One map
per arm (matched / unmatched have separate data files, so their offsets
must never be conflated -- that conflation would be a correctness bug,
which is why the map is constructed per call).

The realized-LENGTH compute (:func:`realized_lengths_for_offsets`) is a
thin wrapper that returns only the ``body_len`` axis of the SAME single
dedup loop, so the #17 index path's output stays byte-identical and
there is exactly one dedup loop, not two.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dedup_hashmap import HashMapU64U32

from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)
from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
    bulk_contributing_geometry,
)

from ._geometry_format import MAX_REALIZED_VALUE, N_GEOMETRY_AXES


__all__ = [
    "RealizedGeometry",
    "realized_geometry_for_offsets",
    "realized_lengths_for_offsets",
]


#: Record offsets resolved per chunk. Each chunk does one hashmap lookup
#: + (for the chunk's misses) one bulk geometry pass + one insert, so the
#: working set is a few same-shaped int64 columns of this size plus the
#: bulk engine's own bounded scratch -- flat regardless of how many
#: variants the arm carries.
_CHUNK_OFFSETS = 1 << 20

#: ``HashMapU64U32`` lookup sentinel for an absent key. Reserved as BOTH
#: the row-index miss marker here AND (by the same reservation) the
#: largest illegal geometry value -- see :data:`MAX_REALIZED_VALUE`.
_U32_MISS = np.uint32(0xFFFFFFFF)


class RealizedGeometry(NamedTuple):
    """Per-variant geometry triple, deduped over distinct record offsets.

    Three parallel ``u32`` arrays, each entry ``i`` describing the record
    at ``record_offsets[i]`` (see :class:`...loader.batch_decode.
    _bulk_expand_lengths.ContributingGeometry` for the per-axis
    semantics).
    """

    body_len: np.ndarray
    id_count: np.ndarray
    value_count: np.ndarray


def realized_geometry_for_offsets(
    data_u8: np.ndarray, record_offsets: np.ndarray
) -> RealizedGeometry:
    """Per-variant geometry triple, deduped over distinct record offsets.

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
    RealizedGeometry
        Three ``u32`` arrays parallel to ``record_offsets``.

    Raises
    ------
    OverflowError
        If any geometry axis exceeds the ``u32`` range, OR if the number
        of distinct records exceeds the ``u32`` row-index range. The pass
        hard-errors (never clamps) so an out-of-range value is surfaced
        rather than silently corrupted.
    """
    offsets = np.asarray(record_offsets, dtype=np.int64).reshape(-1)
    n = offsets.size
    out = tuple(np.empty(n, dtype=np.uint32) for _ in range(N_GEOMETRY_AXES))
    if n == 0:
        return RealizedGeometry(*out)

    keys_all = offsets.astype(np.uint64)
    # Capacity sized to the worst case (every offset distinct); the map
    # is discarded at function exit so the transient over-allocation is
    # bounded by the arm's variant count, not the corpus.
    table = HashMapU64U32(capacity=int(n * 2) + 1)
    # Growing unique-record accumulator columns -- one row per distinct
    # offset, three axes. The hashmap stores the ROW-INDEX of each offset
    # so a triple is gathered by lookup, never re-measured.
    # Per-axis list of measured blocks (one per miss-chunk) + the running
    # total ROW count across all blocks. The row count -- NOT the block
    # count -- is the base row index a new block's first measured record
    # maps to, so a record's row-index stays stable across many chunks.
    columns = [[] for _ in range(N_GEOMETRY_AXES)]
    n_rows = 0

    for lo in range(0, n, _CHUNK_OFFSETS):
        hi = min(lo + _CHUNK_OFFSETS, n)
        keys = keys_all[lo:hi]
        rows = table.lookup_ndarray(keys)
        miss = rows == _U32_MISS
        if bool(miss.any()):
            uniq = np.unique(keys[miss])
            triple = _measure(data_u8, uniq)
            _check_row_capacity(n_rows + uniq.size)
            new_rows = (n_rows + np.arange(uniq.size)).astype(np.uint32)
            for axis_idx in range(N_GEOMETRY_AXES):
                columns[axis_idx].append(triple[axis_idx])
            n_rows += int(uniq.size)
            table.insert_ndarray(uniq, new_rows)
            # Re-resolve the whole chunk now that every miss is inserted.
            rows = table.lookup_ndarray(keys)
        # Gather each axis from its accumulator by the resolved row index.
        for axis_idx in range(N_GEOMETRY_AXES):
            packed = (
                np.concatenate(columns[axis_idx])
                if columns[axis_idx]
                else np.zeros(0, dtype=np.uint32)
            )
            out[axis_idx][lo:hi] = packed[rows]

    return RealizedGeometry(*out)


def realized_lengths_for_offsets(
    data_u8: np.ndarray, record_offsets: np.ndarray
) -> np.ndarray:
    """Per-variant realized BODY length, deduped over distinct offsets.

    Thin wrapper over :func:`realized_geometry_for_offsets` returning
    only the ``body_len`` axis: the historical #17 index path, kept
    byte-identical to the geometry scan's body output (one shared dedup
    loop, never a second). See that function for the full contract.
    """
    return realized_geometry_for_offsets(data_u8, record_offsets).body_len


def _measure(data_u8: np.ndarray, unique_offsets: np.ndarray) -> RealizedGeometry:
    """Geometry triple per UNIQUE record offset, via the bulk engine.

    Returns three ``u32`` arrays parallel to ``unique_offsets``. Raises
    :class:`OverflowError` if ANY axis exceeds the ``u32`` range
    (hard-error, never clamp -- the sidecar dtype is ``u32`` and a
    silently-truncated value would be a correctness bug).
    """
    starts, counts = bulk_token_spans(
        data_u8, unique_offsets.astype(np.int64)
    )
    geometry = bulk_contributing_geometry(data_u8, starts, counts)
    # Raise BEFORE the uint32 cast (so the OverflowError surfaces the true
    # value, never a wrapped one). ``0xFFFFFFFF`` is the reserved hashmap
    # miss sentinel, so the largest storable value is ``MAX_REALIZED_VALUE``
    # (== 0xFFFFFFFE); any value >= the sentinel hard-errors, never clamps.
    axes = (geometry.body_len, geometry.id_count, geometry.value_chunk_count)
    for axis in axes:
        if axis.size and int(axis.max()) >= int(_U32_MISS):
            worst = int(axis.max())
            raise OverflowError(
                f"realized geometry value {worst} exceeds the u32 sidecar "
                f"range ({MAX_REALIZED_VALUE}); the geometry sidecar cannot "
                f"store it (0xFFFFFFFF is reserved as the hashmap miss "
                f"sentinel)"
            )
    return RealizedGeometry(*(axis.astype(np.uint32) for axis in axes))


def _check_row_capacity(n_unique: int) -> None:
    """Hard-error if the distinct-record count overflows the u32 row index.

    The dedup hashmap stores a ``u32`` row-index per distinct offset and
    reserves ``0xFFFFFFFF`` as the lookup-miss sentinel, so at most
    ``MAX_REALIZED_VALUE`` distinct records can be addressed. Any real
    single binary is far below this; the guard exists so a pathological
    corpus surfaces an explicit error rather than aliasing a real row to
    the miss sentinel.
    """
    if n_unique > MAX_REALIZED_VALUE:
        raise OverflowError(
            f"distinct-record count {n_unique} exceeds the u32 dedup "
            f"row-index range ({MAX_REALIZED_VALUE}); 0xFFFFFFFF is reserved "
            f"as the hashmap miss sentinel"
        )
