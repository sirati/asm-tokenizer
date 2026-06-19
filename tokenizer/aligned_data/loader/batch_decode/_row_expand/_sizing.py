"""Per-row length sizing (row_offsets cumsum without a flat array).

Single concern: expand per-unique-variant lengths to per-row lengths via
the :func:`build_per_row_variant_lookup` mapping, then cumsum into the
``u32[batch_size + 1]`` ``row_offsets`` -- the length-only mode shared by
every per-row concat concern (it materialises no flat payload).

Plan reference: ``batch_decode_plan.md`` ``## Stages -- algorithm
sketch`` Stage 2 (row-offset cumsum).
"""

from __future__ import annotations

import numpy as np


__all__ = ["row_offsets_from_per_variant_lengths"]


def row_offsets_from_per_variant_lengths(
    per_variant_lengths: np.ndarray,
    per_row_variant_idx: np.ndarray,
    is_padding: np.ndarray,
) -> np.ndarray:
    """Build ``u32[batch_size + 1]`` cumsum-of-per-row-lengths.

    Parameters
    ----------
    per_variant_lengths:
        ``u32[num_unique_variants]`` -- one entry per unique variant in
        the flat order produced by :func:`build_per_row_variant_lookup`.
        For scalar-count concerns (e.g. surviving identity / number
        counts) this IS the per-variant payload; for array concerns it
        is ``[a.shape[0] for a in per_variant_arrays]``.
    per_row_variant_idx:
        ``u32[batch_size]`` -- output of
        :func:`build_per_row_variant_lookup`.
    is_padding:
        ``bool[batch_size]`` -- output of
        :func:`build_per_row_variant_lookup`. Padding rows contribute
        zero length regardless of the (clamped) variant index.

    Returns
    -------
    np.ndarray
        ``u32[batch_size + 1]``; ``[0]`` is always 0; ``[i + 1] =
        [i] + per_row_length_at_i``.
    """

    batch_size = per_row_variant_idx.shape[0]
    if per_variant_lengths.shape[0] == 0:
        # No variants exist at all -- every row is either padding or
        # has clamped index 0 with no backing entry. Per-row length
        # is uniformly 0; skip the fancy-index to avoid an out-of-bounds
        # access on the empty per_variant_lengths array.
        per_row_lengths = np.zeros(batch_size, dtype=np.uint32)
    else:
        per_row_lengths = np.where(
            is_padding,
            np.uint32(0),
            per_variant_lengths[per_row_variant_idx],
        ).astype(np.uint32, copy=False)

    row_offsets = np.empty(batch_size + 1, dtype=np.uint32)
    row_offsets[0] = 0
    np.cumsum(per_row_lengths, out=row_offsets[1:])
    return row_offsets
