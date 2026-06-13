"""Depth-0 contributing body-length parse + the build's length budget.

Single concern: the per-variant OWN length used by the per-root BFS
(:mod:`._bfs`) -- ``1 self-token + contributing-body-length(record)`` --
plus the no-cutoff budget constant the build asserts against. The splice
adjacency (which callees a node reaches) lives in :mod:`._adjacency`,
read live from the catalog memmap with no precomputed graph structure.

A node is one (section, variant) pair; its body length is the record's
post-promotion post-strip stream length, computed in bulk by
:func:`...loader.batch_decode._bulk_expand_lengths.
bulk_contributing_body_lengths` at unique-record granularity. The
variant-token row PREFIX deliberately stays OUTSIDE the index sums (the
historical ``_variant_lengths_at_depth`` contract never included it).
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)
from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
    bulk_contributing_body_lengths,
)
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections


__all__ = ["LARGE_CONTEXT_LEN", "_body_lengths"]


#: Same role as the historical walk budget: the build ASSERTS every
#: spliced length stays under this bound, because beyond it the legacy
#: stage-2 cutoff would have fired and the sorted index would silently
#: under-report (plan D-2.2).
LARGE_CONTEXT_LEN = 2**30

#: The record-offset shift packing ``data_offset_shifted`` (see
#: ``_matched_arm_loader``: real offset = ``data_offset_shifted << 4``).
_DATA_OFFSET_SHIFT = 4


def _body_lengths(
    cols: ColumnarSections, data_u8: np.ndarray
) -> np.ndarray:
    """``int64[total_vars]``: contributing BODY length per variant.

    ONE depth-0 parse at unique-record granularity. Of a node's three
    token contributors this is the only record-derived one; the other
    two are kept structurally separate:

    * the prepended self-token (exactly 1 per spliced call target) is
      composed by :func:`compute_node_lengths` at the DP site;
    * the variant-token row PREFIX (the v3 variant-axis run behind
      ``var_ref_offset`` in ``_variants.bin``) deliberately lives
      OUTSIDE the index sums -- the historical
      ``_variant_lengths_at_depth`` contract sums
      ``predicted_full_length`` over call targets, which never included
      the row prefix. A future consumer that wants prefix lengths gets
      a bulk ``n_tokens``-header reader next to the record layout owner
      (:mod:`tokenizer.variant_tokens.record`); it does NOT belong in
      this parse (the build's 3-sidecar contract, see
      :mod:`._builder`).
    """
    offsets = cols.var_data_offset_shifted.astype(np.int64) << _DATA_OFFSET_SHIFT
    uniq, inverse = np.unique(offsets, return_inverse=True)
    starts, counts = bulk_token_spans(data_u8, uniq)
    body = bulk_contributing_body_lengths(data_u8, starts, counts)
    return body[inverse]
