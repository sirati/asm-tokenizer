"""Pure wire encode/decode for the per-binary sorted-length index.

Mirrors ``translation_dataset.build_sorted_index_numpy`` but uses u32
(not u16) widths in the header because asm depth-3 splice lengths
routinely exceed 65535.

Layout (little-endian throughout):

* ``u32 min_length``    -- smallest length present (or 0 when empty).
* ``u32 num_lengths``   -- count of contiguous length buckets,
  ``max_length - min_length + 1`` (or 0 when empty).
* ``u32 counts[num_lengths]`` -- number of sections per length bucket
  starting at ``min_length`` (one slot per length, possibly zero).
* ``u32 body[N]`` -- original section indices in stable-sorted order,
  with ``N == sum(counts) == lengths.size``.

This module is *purely* a wire concern: no file I/O, no batch_decode
imports. Encode + ``parse_header`` are the only public names.
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np

__all__ = ["encode_sorted_index", "parse_header"]


def encode_sorted_index(lengths: np.ndarray) -> bytes:
    """Argsort + bucket + pack into the sorted-index wire format.

    Parameters
    ----------
    lengths:
        ``u32`` per-section length array; one entry per section in the
        binary's natural order. ``lengths[i]`` is the depth-N reduced
        length of section ``i``.

    Returns
    -------
    bytes
        The serialised index. For empty input (``lengths.size == 0``)
        returns the 8-byte empty header ``struct.pack("<II", 0, 0)``.

    Notes
    -----
    Sort key is ``np.argsort(lengths, kind="stable")`` so equal-length
    sections retain their original input order in the body. Counts are
    derived via ``np.bincount`` over ``sorted_lengths - min_length`` so
    every length bucket between ``min`` and ``max`` (inclusive) gets an
    explicit count slot, including zero-count gaps.
    """
    if lengths.size == 0:
        return struct.pack("<II", 0, 0)

    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]
    min_length = int(sorted_lengths[0])
    max_length = int(sorted_lengths[-1])
    num_lengths = max_length - min_length + 1
    counts = np.bincount(sorted_lengths - min_length, minlength=num_lengths)

    header = struct.pack("<II", min_length, num_lengths)
    header += counts.astype(np.uint32).tobytes()
    body = order.astype(np.uint32).tobytes()
    return header + body


def parse_header(blob: bytes) -> Tuple[int, np.ndarray, int]:
    """Parse the header off a sorted-index blob.

    Parameters
    ----------
    blob:
        The raw bytes produced by :func:`encode_sorted_index` (or any
        equivalent producer obeying the wire format).

    Returns
    -------
    (min_length, counts, body_offset)
        ``min_length`` is the smallest length present (``int``).
        ``counts`` is a zero-copy ``np.ndarray`` view of ``num_lengths``
        ``u32`` entries backed by ``blob``. ``body_offset`` is the byte
        offset at which the ``u32`` body begins (``8 + 4 * num_lengths``).

    For an empty index (``num_lengths == 0``) ``counts`` is a length-0
    view and ``body_offset == 8``.
    """
    min_length, num_lengths = struct.unpack_from("<II", blob, 0)
    counts = np.frombuffer(blob, dtype=np.uint32, count=num_lengths, offset=8)
    body_offset = 8 + 4 * num_lengths
    return min_length, counts, body_offset
