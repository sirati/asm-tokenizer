"""Per-binary sorted-index reader + memmap-dir discovery (plan ALG-5 + ALG-8).

Two concerns wired in one file because they sit at the same abstraction
layer (per-binary on-disk index access) and the discovery helper's
returned :class:`LengthReduction` map is the construction input for
``SortedIndexReader``:

* :class:`SortedIndexReader` -- eager-load (``Path.read_bytes``) a
  single ``<binary>_sorted_<mode>_d<depth>.idx`` file and expose
  bucket-count + bucket-sample primitives the cross-binary sampler
  consumes.
* :func:`discover_indices` -- walk a memmap directory for files
  matching the canonical filename grammar and return the
  ``{binary_name -> [LengthReduction, ...]}`` map for a given depth.

Loading discipline is :meth:`Path.read_bytes` (NOT :func:`numpy.memmap`)
per plan ALG-5: per-binary index files are ~1 MB at the corpus scale we
target, so eager load is simpler than mmap lifecycle bookkeeping and
keeps the reader free of file-handle state.

No CLI / batch_decode imports here -- this module reads only the wire
format (via :func:`tokenizer.aligned_data.sorted_index._wire.parse_header`)
plus the typed :class:`LengthReduction` parameter (via
:func:`tokenizer.aligned_data.sorted_index._modes.parse_reduction`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from ._modes import parse_reduction
from ._types import LengthReduction
from ._wire import parse_header


__all__ = ["SortedIndexReader", "discover_indices"]


# Canonical filename grammar (plan D5): ``<binary>_sorted_<mode>_d<depth>.idx``
# where ``<mode>`` is ``max`` or ``p<NN>`` (two zero-padded digits) and
# ``<depth>`` is exactly three zero-padded digits so files lexsort by
# depth.
_FILENAME_RE = re.compile(
    r"^(?P<binary>.+)_sorted_(?P<mode>max|p\d{2})_d(?P<depth>\d{3})\.idx$",
)


class SortedIndexReader:
    """Per-binary sorted-index reader (plan ALG-5).

    Loads the index file eagerly via :meth:`Path.read_bytes` and
    precomputes a per-length body-offset table so :meth:`count_at` and
    :meth:`sample_section_indices` run in O(1) per call (modulo the
    sample-without-replacement RNG cost).

    The reader carries :attr:`reduction` and :attr:`depth` metadata so
    downstream code that mixes readers can verify they share the same
    reduction + depth -- but the reader itself does NOT validate that
    invariant (the file content is identical regardless of which
    reduction produced it; depth + reduction live only in the filename).
    """

    def __init__(
        self,
        path: Path,
        *,
        reduction: LengthReduction,
        depth: int,
    ) -> None:
        self._path = Path(path)
        self._blob = self._path.read_bytes()
        min_length, counts, body_offset = parse_header(self._blob)
        self._min_length = int(min_length)
        self._counts = counts
        self._body_offset = int(body_offset)
        self._reduction = reduction
        self._depth = int(depth)
        # Per-length body offsets, precomputed once (O(num_lengths)).
        # ``cumsum[i]`` is the count of body entries strictly BEFORE
        # length ``min_length + i``; ``cumsum[num_lengths]`` is the
        # total body entry count. We use ``uint64`` for the cumulative
        # arithmetic so the byte-offset multiply cannot overflow when
        # ``total_sections * 4`` exceeds u32 range.
        cumsum = np.concatenate(
            ([np.uint64(0)], np.cumsum(counts.astype(np.uint64))),
        )
        self._bucket_body_offsets = (
            np.uint64(body_offset) + np.uint64(4) * cumsum
        )

    # ------------------------------------------------------------------
    # Static metadata
    # ------------------------------------------------------------------
    @property
    def reduction(self) -> LengthReduction:
        return self._reduction

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def min_length(self) -> int:
        return self._min_length

    @property
    def max_length(self) -> int:
        """Largest length present, or 0 for an empty index.

        Mirrors the plan: ``min_length + num_lengths - 1`` when the
        body is non-empty.
        """
        if self._counts.size == 0:
            return 0
        return self._min_length + int(self._counts.size) - 1

    def total_sections(self) -> int:
        """Number of body entries (sum over every length bucket)."""
        return int(self._counts.sum())

    # ------------------------------------------------------------------
    # Bucket access
    # ------------------------------------------------------------------
    def count_at(self, length: int) -> int:
        """Bucket size at ``length``; returns 0 when out of range."""
        length_idx = length - self._min_length
        if length_idx < 0 or length_idx >= self._counts.size:
            return 0
        return int(self._counts[length_idx])

    def count_in_band(self, lo: int, hi: int) -> int:
        """Total bucket count for all lengths in ``[lo, hi]`` (inclusive).

        Sums the per-length ``_counts`` slice for the overlap of
        ``[lo, hi]`` with the reader's valid range
        ``[min_length, max_length]``.  Returns 0 when the band is
        entirely out of range or the overlap is empty.

        Length 0 is the index's EXCLUSION marker (0-variant and
        gated-out sections are stamped 0 by the builder), so the band
        is clamped to ``lo >= 1`` -- excluded sections are never
        eligible, no matter how low the band reaches.
        """
        lo = max(lo, 1)
        lo_idx = max(0, lo - self._min_length)
        hi_idx = min(self._counts.size - 1, hi - self._min_length)
        if lo_idx > hi_idx:
            return 0
        return int(self._counts[lo_idx : hi_idx + 1].sum())

    def sample_section_indices_in_band(
        self,
        lo: int,
        hi: int,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Uniform sample without replacement from all buckets in ``[lo, hi]``.

        Concatenates every section index from every length-bucket in the
        range ``[lo, hi]`` (intersected with the reader's valid range)
        into a single pool, then draws ``min(count, pool_size)`` entries
        uniformly without replacement via ``rng.choice``.

        Returns a fresh ``u32`` ndarray (never a view of the blob).
        Returns an empty array when the band pool is empty.

        Length 0 is the index's EXCLUSION marker (see
        :meth:`count_in_band`); the band is clamped to ``lo >= 1`` so
        excluded sections are never drawn.
        """
        lo = max(lo, 1)
        lo_idx = max(0, lo - self._min_length)
        hi_idx = min(self._counts.size - 1, hi - self._min_length)
        if lo_idx > hi_idx:
            return np.empty(0, dtype=np.uint32)

        # Collect all section indices from every bucket in the band.
        parts = []
        for idx in range(lo_idx, hi_idx + 1):
            bc = int(self._counts[idx])
            if bc == 0:
                continue
            body_offset = int(self._bucket_body_offsets[idx])
            bucket = np.frombuffer(
                self._blob,
                dtype=np.uint32,
                count=bc,
                offset=body_offset,
            )
            # Copy so the pool array is independent of the blob.
            parts.append(bucket.copy())

        if not parts:
            return np.empty(0, dtype=np.uint32)

        pool = np.concatenate(parts)
        pool_size = pool.size
        k = min(count, pool_size)
        if k == pool_size:
            return pool
        chosen = rng.choice(pool_size, size=k, replace=False)
        return pool[chosen].astype(np.uint32, copy=False)

    def sample_section_indices(
        self,
        target_length: int,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample ``min(count, bucket_count)`` section indices at ``target_length``.

        Returns a fresh ``u32`` ndarray; never a view of the eager
        ``read_bytes`` blob. When ``count >= bucket_count`` the entire
        bucket is returned (in stable-sort order, matching the body's
        layout); otherwise ``rng.choice`` produces a uniform sample
        without replacement.
        """
        length_idx = target_length - self._min_length
        if length_idx < 0 or length_idx >= self._counts.size:
            return np.empty(0, dtype=np.uint32)
        bucket_count = int(self._counts[length_idx])
        if bucket_count == 0:
            return np.empty(0, dtype=np.uint32)
        body_offset = int(self._bucket_body_offsets[length_idx])
        bucket = np.frombuffer(
            self._blob,
            dtype=np.uint32,
            count=bucket_count,
            offset=body_offset,
        )
        if count >= bucket_count:
            # Copy out of the eager blob so callers can mutate the
            # returned array without poisoning future bucket reads.
            return bucket.copy()
        chosen = rng.choice(bucket_count, size=count, replace=False)
        # ``rng.choice`` returns int64; ``bucket[chosen]`` materialises
        # a fresh u32 ndarray -- safe to hand back to callers.
        return bucket[chosen]


def discover_indices(
    memmap_dir: Path,
    *,
    depth: int,
) -> Dict[str, List[LengthReduction]]:
    """Scan ``memmap_dir`` for sorted-index files at the given depth.

    Walks every regular file in ``memmap_dir`` and matches against the
    canonical filename grammar
    ``<binary>_sorted_<mode>_d<depth>.idx``. Files that don't match the
    grammar (e.g. unrelated ``.idx`` files, sub-directories, mode
    strings that fail :func:`parse_reduction`) are skipped silently.
    Files matching the grammar but with a different depth are also
    skipped.

    Returns
    -------
    Dict[str, List[LengthReduction]]
        ``{binary_name -> [reduction, ...]}`` -- one entry per binary
        with at least one matching index file. Within a binary the
        reduction list reflects directory iteration order (file system
        dependent); callers needing a stable order should sort by
        :meth:`LengthReduction.filename_tag`.
    """
    by_binary: Dict[str, List[LengthReduction]] = {}
    depth_tag = f"{depth:03d}"
    memmap_dir = Path(memmap_dir)
    for entry in memmap_dir.iterdir():
        if not entry.is_file():
            continue
        m = _FILENAME_RE.match(entry.name)
        if m is None:
            continue
        if m.group("depth") != depth_tag:
            continue
        try:
            reduction = parse_reduction(m.group("mode"))
        except ValueError:
            continue
        by_binary.setdefault(m.group("binary"), []).append(reduction)
    return by_binary
