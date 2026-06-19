"""Estimate per-record ``_data.bin`` body byte spans for prefetch.

Single concern: this is the ONE place the body-span over-estimate
formula lives -- it turns the geometry emission's sidecar counts (and the
catalog's record offsets) into the ``(record_start, span_bytes)`` ranges
the page-prefetch will hint, WITHOUT touching ``_data.bin`` (the whole
point: we estimate the span we are about to read so we can prefetch it
before reading it).

The formula is a DELIBERATE SAFE OVER-ESTIMATE of each stored record's
body byte span, validated zero-under-cover corpus-wide (4 binaries incl
z3, 14.86M records):

    body_len   = own_length - 1            (own_length = 1 self + body)
    span_bytes = ceil((body_len
                       + value_total * 8
                       + id_total * 2) * 2 * 1.11 + 30)

``2`` is ``bytes_per_token``: body tokens are always u16 (there is no
variable token width). The exact stored span is intentionally NOT
computed -- doing so would require decoding the record header in
``_data.bin``, i.e. the very read this hint exists to front-run.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._locator import RECORD_OFFSET_SHIFT


__all__ = ["estimate_body_prefetch_ranges"]


def estimate_body_prefetch_ranges(cols, emission) -> Tuple[np.ndarray, np.ndarray]:
    """``(record_starts, span_bytes)`` prefetch ranges for the emission.

    Parameters
    ----------
    cols:
        The columnar ``sections.bin`` catalog; ``var_data_offset_shifted
        [node]`` is the node's ``record_offset >> RECORD_OFFSET_SHIFT``.
    emission:
        The body-free ``BatchRowEmission`` carrying ``node``,
        ``own_length``, ``id_total`` and ``value_total`` sidecar counts.

    Returns
    -------
    tuple
        ``(record_starts, span_bytes)`` -- ``int64[n]`` absolute
        ``_data.bin`` record offsets + the over-estimated body byte spans,
        parallel to ``emission.node``. Empty emission -> two empty arrays.
    """
    nodes = np.asarray(emission.node, dtype=np.int64)
    if nodes.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()

    starts = (
        cols.var_data_offset_shifted[nodes].astype(np.int64)
        << RECORD_OFFSET_SHIFT
    )

    body_len = emission.own_length.astype(np.int64) - 1
    value_total = emission.value_total.astype(np.int64)
    id_total = emission.id_total.astype(np.int64)
    span = np.ceil(
        (body_len + value_total * 8 + id_total * 2) * 2 * 1.11 + 30
    ).astype(np.int64)
    return starts, span
