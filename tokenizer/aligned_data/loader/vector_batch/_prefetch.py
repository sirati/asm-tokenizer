"""Page-prefetch ``_data.bin`` byte ranges via ``MADV_WILLNEED``.

Single concern: turn a set of ``(start, length)`` byte ranges over a
read-only memmap into the smallest set of page-aligned, EOF-clamped,
COALESCED ``mmap.madvise(MADV_WILLNEED, ...)`` syscalls.

This is a PURE ADVISORY HINT. ``MADV_WILLNEED`` only asks the kernel to
begin reading the named pages ahead of the fault that would otherwise
page them in synchronously; it cannot change a single output byte, and a
failed hint silently degrades to the unhinted (still-correct) read path.
Consequently this module never raises into its caller and knows NOTHING
about the decode -- it is given raw ranges and emits raw syscalls.

The vectorized pipeline:

1. Drop empty / past-EOF ranges (``length <= 0`` or ``start >= size``).
2. Page-align each ``start`` DOWN to a :data:`mmap.PAGESIZE` boundary,
   growing ``length`` by the alignment delta so the original range stays
   covered (``mmap.madvise`` rejects an unaligned ``start`` with EINVAL).
3. Clamp each range's end to the mmap size (an over-long length past EOF
   is otherwise clamped internally, but we clamp explicitly so a fully
   out-of-range tail drops out).
4. Coalesce touching/overlapping ranges into runs -- adjacent records
   share pages, so this collapses the per-record hints into a handful of
   syscalls.
"""

from __future__ import annotations

import mmap

import numpy as np


__all__ = ["prefetch_willneed"]


_PAGE = mmap.PAGESIZE


def prefetch_willneed(
    mm: mmap.mmap, starts: np.ndarray, lengths: np.ndarray
) -> None:
    """Issue ``MADV_WILLNEED`` over the coalesced, aligned ``ranges``.

    Parameters
    ----------
    mm:
        The python mmap backing a read-only ``np.memmap`` (``arr._mmap``).
    starts:
        ``int64[k]`` absolute byte offsets of the ranges to prefetch.
    lengths:
        ``int64[k]`` byte lengths, parallel to ``starts``.

    Notes
    -----
    Advisory only: never raises into the caller (a hint cannot affect the
    decoded output). Empty input is a no-op.
    """
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return

    size = int(mm.size())
    if size <= 0:
        return

    # (1) Drop empty / past-EOF ranges.
    keep = (lengths > 0) & (starts < size)
    starts = starts[keep]
    lengths = lengths[keep]
    if starts.size == 0:
        return

    # (2) Page-align start DOWN; grow length so the original range stays
    # covered (madvise rejects an unaligned start with EINVAL).
    aligned = starts & ~np.int64(_PAGE - 1)
    lengths = lengths + (starts - aligned)

    # (3) Clamp end to mmap size; drop ranges that begin past EOF.
    ends = np.minimum(aligned + lengths, np.int64(size))
    keep = ends > aligned
    aligned = aligned[keep]
    ends = ends[keep]
    if aligned.size == 0:
        return

    # (4) Coalesce: sort by aligned start, merge ranges whose intervals
    # touch or overlap into single runs -- fully vectorized. A new run
    # starts wherever a range's start exceeds the running max-end of all
    # earlier ranges; ``reduceat`` then takes each run's first start and
    # its max end. Only the (few) merged runs reach Python for the
    # madvise calls -- adjacent records coalesce heavily, so this is a
    # handful of syscalls regardless of batch size.
    order = np.argsort(aligned, kind="stable")
    aligned = aligned[order]
    ends = ends[order]

    # Running max-end of everything strictly before each range.
    prev_max_end = np.empty_like(ends)
    prev_max_end[0] = aligned[0]  # first range always opens a run
    np.maximum.accumulate(ends[:-1], out=prev_max_end[1:])
    run_open = aligned > prev_max_end  # True => this range begins a new run
    run_open[0] = True
    run_first = np.flatnonzero(run_open)  # index of each run's first range
    run_starts = aligned[run_first]
    run_ends = np.maximum.reduceat(ends, run_first)

    for start, end in zip(run_starts.tolist(), run_ends.tolist()):
        _advise(mm, start, end - start)


def _advise(mm: mmap.mmap, start: int, length: int) -> None:
    """Best-effort ``MADV_WILLNEED`` over one aligned, clamped span.

    The alignment + clamp upstream guarantee a valid call, but a hint is
    advisory: any unexpected ``OSError`` is swallowed rather than aborting
    the batch (it cannot change the decoded bytes).
    """
    if length <= 0:
        return
    try:
        mm.madvise(mmap.MADV_WILLNEED, start, length)
    except OSError:
        pass
