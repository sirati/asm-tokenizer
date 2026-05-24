"""Power-of-two bucketed batch dispatch for :func:`run_lengths`.

Single concern: collect a variable-length stream of 1D bool masks, sort
each by smallest-pow2-ceiling length, and dispatch ONE 2D
:func:`run_lengths` per bucket on :meth:`flush`. Amortizes the Python
call overhead + numpy-broadcasting setup over many masks at the price
of zero-padding each mask up to its bucket's pow2 length.

This module knows numpy + bool masks. It knows NOTHING about
:class:`InlineDecodeState`, batch_decode, call_targets, sections, or
variants -- those concerns own a collector and decide what to do with
its handles, but never reach inside.

Why pow2 buckets, not exact-length buckets?
* Exact-length buckets collapse when masks are mostly the same length
  (e.g. one-section batches), but real workloads see a wide spread of
  function-body lengths. Pow2 buckets cap the number of dispatches at
  ``log2(max_length)`` regardless of input distribution, which is the
  cheapest worst case.
* Zero-padding is correct: :func:`run_lengths` tolerates trailing
  ``False`` values. The result's per-position run lengths in the
  ``[:L]`` prefix are byte-identical to the unpadded call (since a
  trailing-False tail neither opens a new run nor extends an existing
  one; runs are anchored at their start position).

Bucket sizing detail: the smallest pow2 ``>= L`` is the bucket key. A
mask of length exactly ``2**k`` lands in bucket ``2**k`` with no
padding overhead; a mask of length ``2**k + 1`` lands in bucket
``2**(k+1)`` (worst-case ~2x padding).

Lifetime contract: :meth:`flush` returns numpy view slices into the
per-bucket output arrays. The collector releases its own references
during :meth:`flush` so the caller becomes the unique owner of the
underlying arrays. The slices are zero-copy (``result_2d[i, :L]``);
they remain valid as long as the caller keeps the returned dict alive.
A subsequent :meth:`add` allocates fresh bucket buffers and does not
disturb the previous flush's outputs.
"""

from __future__ import annotations

import numpy as np

from .run_lengths import run_lengths


__all__ = ["BucketedRunLengthCollector"]


def _smallest_pow2_geq(n: int) -> int:
    """Return the smallest power of two ``>= n``.

    Used as the bucket key. ``n == 1`` -> ``1``; ``n == 2`` -> ``2``;
    ``n == 3`` -> ``4``; ``n == 1024`` -> ``1024``; ``n == 1025`` ->
    ``2048``. Rejects ``n < 1`` -- a zero-length mask cannot satisfy
    :func:`run_lengths`'s ``mask[0] == False`` precondition anyway, so
    the collector's :meth:`add` rejects zero-length inputs upstream.
    """
    if n < 1:
        raise ValueError(f"bucket size must be >= 1; got {n}")
    # Python 3.14: int.bit_length() is the canonical "ceil log2" path.
    # ``(n - 1).bit_length()`` is 0 when n == 1 (bucket 1), else
    # ``ceil(log2(n))``. Shift left to materialize the bucket key.
    return 1 << (n - 1).bit_length()


class BucketedRunLengthCollector:
    """Stage 1D bool masks into pow2 buckets; flush dispatches one
    :func:`run_lengths` per bucket.

    Usage:

        collector = BucketedRunLengthCollector()
        h_a = collector.add(mask_a)
        h_b = collector.add(mask_b)
        results = collector.flush()  # {h_a: uint16[len(mask_a)], ...}

    Each handle is an opaque ``int`` (the position the mask was added
    at). Treat it as a token; do not depend on its numeric meaning.
    """

    __slots__ = ("_buckets", "_lengths", "_n_added", "_flushed_once")

    def __init__(self) -> None:
        # Per-bucket-size: list of (handle, mask) tuples. Lazy view +
        # object reuse (no wrapper class per add); the tuple is the
        # smallest python container that carries both pieces.
        self._buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        # Per-handle: the original mask length (for the post-flush slice).
        # A flat list indexed by handle; cheaper than a dict for a
        # contiguous-handle workload.
        self._lengths: list[int] = []
        self._n_added: int = 0
        self._flushed_once: bool = False

    def add(self, mask: np.ndarray) -> int:
        """Stage a 1D bool mask; return an opaque handle.

        Preconditions (matching :func:`run_lengths`'s contract):

        * ``mask.ndim == 1``.
        * ``mask.dtype == np.bool_``.
        * ``mask.shape[0] >= 1`` (zero-length rejected, mirroring
          :func:`run_lengths`).
        * ``mask[0] == False`` (the first-position-False invariant
          that makes the run-length encoding well-defined).

        The mask is NOT copied; the collector keeps a reference until
        :meth:`flush` runs. Callers should not mutate the mask between
        ``add`` and ``flush``.
        """
        if mask.ndim != 1:
            raise ValueError(
                f"BucketedRunLengthCollector.add: expected 1D mask; "
                f"got ndim={mask.ndim}"
            )
        if mask.dtype != np.bool_:
            raise ValueError(
                f"BucketedRunLengthCollector.add: expected bool dtype; "
                f"got dtype={mask.dtype}"
            )
        if mask.shape[0] == 0:
            raise ValueError(
                "BucketedRunLengthCollector.add: zero-length mask "
                "rejected (run_lengths requires mask[0] == False)."
            )
        if bool(mask[0]):
            raise ValueError(
                "BucketedRunLengthCollector.add: mask[0] must be False "
                "(run_lengths precondition)."
            )

        length = int(mask.shape[0])
        bucket = _smallest_pow2_geq(length)
        handle = self._n_added
        self._n_added += 1
        self._lengths.append(length)
        self._buckets.setdefault(bucket, []).append((handle, mask))
        return handle

    def flush(self) -> dict[int, np.ndarray]:
        """Dispatch one 2D :func:`run_lengths` per bucket; return
        ``{handle: uint16[original_length]}``.

        After the call, the collector's internal state is cleared so a
        second :meth:`flush` returns ``{}`` and subsequent :meth:`add`
        calls start a fresh staging round.

        Per the module docstring: returned arrays are zero-copy views
        into the freshly-allocated per-bucket output. The collector
        drops its reference here, so the caller owns the lifetime.
        """
        out: dict[int, np.ndarray] = {}
        for bucket_size, entries in self._buckets.items():
            n_in_bucket = len(entries)
            # Allocate a fresh (n_in_bucket, bucket_size) padded buffer.
            # ``np.zeros(..., bool)`` gives False everywhere; we write
            # the real mask into ``[i, :L_i]`` so the trailing slots
            # remain False -- exactly the postpend-False shape that
            # ``run_lengths`` tolerates.
            buf = np.zeros((n_in_bucket, bucket_size), dtype=bool)
            for row_idx, (_handle, mask) in enumerate(entries):
                buf[row_idx, : mask.shape[0]] = mask
            # One 2D dispatch -- the per-row work happens in numpy.
            runlen_2d = run_lengths(buf)
            for row_idx, (handle, _mask) in enumerate(entries):
                length = self._lengths[handle]
                # Zero-copy view into runlen_2d; the caller's reference
                # to ``out`` keeps the underlying array alive.
                out[handle] = runlen_2d[row_idx, :length]

        # Clear state. The original masks (now no longer needed) drop
        # their refcount here; the runlen_2d arrays survive via the
        # views in ``out``.
        self._buckets.clear()
        self._lengths.clear()
        self._n_added = 0
        self._flushed_once = True
        return out
