"""Dense front-matter columns built DIRECTLY from the batched expansion.

Single concern (the design-first sentence): *given the flat
:class:`...._batched_expand.BatchedExpansion` (the canonical CSR expansion
+ per-position InlineDecodeState fields) and the per-emitted-node
``surviving`` token count (the straddler-cut clip), produce the SAME flat
columns the stage-3 dense-byte-stream sites
(:mod:`...batch_decode._inline_bytes` 3a,
:mod:`...batch_decode._identity_decode` 3b,
:mod:`...batch_decode._number_decode._flat_segments` 3c,
:mod:`...batch_decode._bulk_bytes` sign) concatenate today by re-walking
the per-call_target ``Stage2Batch`` tree.*

Why this is well-posed (NOT a fresh decode): in the vector path the
``Stage2Batch`` those sites walk is itself ADAPTED from this exact
:class:`BatchedExpansion` + ``surviving`` array by
:mod:`._dense_adapter` -- one synthetic ``Stage2CallTarget`` per emitted
node, in emission order, with ``surviving_token_count == surviving[e]``.
The adapter lays one section per non-padding batch row in row order and
the row CSR ``row_offsets`` partitions ``[0, n_emitted)`` contiguously, so
the canonical stage-3 DFS call_target enumeration
(``sections -> variants -> call_targets``) is exactly the emitted-node
order ``e = 0 .. n_emitted - 1``. The "emission -> DFS-kept permutation"
the object-tree-elimination plan flags as the sharp edge is therefore the
IDENTITY here; the only non-trivial axis is the per-node surviving CLIP.

This module re-implements NO decode rule. It only SLICES + CONCATENATES
the batched flats the same way the per-call_target sites slice + concat
the per-call_target object views -- so the byte-identity gate cannot
diverge on decode logic. The carrier identification, byte-offset
arithmetic and per-:class:`TokenType` emission stay with the owning
kernels; they read these columns the same way they read the per-CT views.

Module boundary: owned at the scatter/expand boundary (it consumes
:class:`BatchedExpansion`); the only things crossing the boundary are flat
numpy arrays + per-node scalars. No caller sees a ``BatchedExpansion``
internal nor a per-CT dataclass.

The two CSR axes
----------------
``BatchedExpansion`` carries every per-node column as a flat CSR:

* RAW-space (``raw_record_offsets``): ``raw_tokens`` / ``real_mask`` /
  ``number_mask`` / ``runlen_number`` / ``is_negative_per_position``.
* DIGIT-cumsum space (``N + 1`` slots per node, packed at
  ``rec[i] + i .. rec[i + 1] + (i + 1)``): ``digit_cumsum``.
* EXPANDED-space (``node_offsets``): ``expanded`` / ``extra_value_v2_mask``
  / ``extra_f128_mask``.

The KEPT (``surviving > 0``) subset + the surviving clip are derived as
per-node CSR bases via the ``np.repeat``-over-per-segment-LENGTH
discipline (NEVER mark-and-cumsum: consecutive zero-length segments would
merge and shift every later segment id -- the #92 trap that bites the
sign array's ``surviving == 1`` empty-body segments).
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._dense_columns import (
    DenseColumns,
)
from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving_batched,
)

from ._batched_expand import BatchedExpansion


__all__ = ["DenseColumns", "build_dense_columns"]


# ---------------------------------------------------------------------------
# Build. The adapter computes ``surviving_identity_count`` /
# ``surviving_number_chunk_count`` via ``count_surviving_batched`` over the
# SAME ``expanded`` + ``node_offsets`` + ``surviving`` triple; we reuse
# that owned kernel verbatim so the two can never drift.
# ---------------------------------------------------------------------------


def build_dense_columns(
    batched: BatchedExpansion,
    raw_flat: np.ndarray,
    record_offsets: np.ndarray,
    surviving: np.ndarray,
) -> DenseColumns:
    """Build :class:`DenseColumns` from the batched expansion + the cut.

    Parameters
    ----------
    batched:
        The flat :class:`BatchedExpansion` for the emitted nodes (node
        ``e`` of the expansion IS emitted node ``e``).
    raw_flat:
        ``uint16[total_raw]`` -- the flat gathered RAW body stream the
        expansion was built over (``GatheredBodies.raw``). The expansion
        retains the per-position state masks but NOT the raw tokens
        themselves (3a / 3b read ``state.raw_tokens``), so the caller
        threads the same flat stream it passed to ``batched_expand``.
    record_offsets:
        ``int64[n_nodes + 1]`` -- the RAW-space CSR the expansion was
        built over (``GatheredBodies.record_offsets``). The expansion does
        not retain it, so the caller threads it (it owns the raw-space
        segmentation the per-node ``state`` views slice).
    surviving:
        ``int64[n_nodes]`` -- the per-emitted-node surviving token count
        from the straddler cut (:func:`._surviving.surviving_token_counts`).
        This IS each node's ``Stage2CallTarget.surviving_token_count`` (=
        ``partial_cut_length``).

    Returns
    -------
    DenseColumns
        The flat per-node columns + CSR + kept subset, in emitted-node
        (== DFS call_target) order.
    """
    raw_flat = np.asarray(raw_flat, dtype=np.uint16).reshape(-1)
    raw_offsets = np.asarray(record_offsets, dtype=np.int64).reshape(-1)
    node_offsets = np.asarray(batched.node_offsets, dtype=np.int64).reshape(-1)
    n_nodes = raw_offsets.shape[0] - 1

    surviving = np.asarray(surviving, dtype=np.int64).reshape(-1)
    if surviving.shape[0] != n_nodes:
        raise ValueError(
            f"surviving has {surviving.shape[0]} entries but the expansion "
            f"covers {n_nodes} nodes"
        )

    # DIGIT-cumsum CSR: node ``e``'s ``N + 1`` block sits at
    # ``raw_offsets[e] + e .. raw_offsets[e + 1] + (e + 1)`` (the per-node
    # trailing-slot packing the batched twin produces). Adding
    # ``arange(n_nodes + 1)`` to ``raw_offsets`` yields exactly those
    # block boundaries.
    digit_offsets = raw_offsets + np.arange(n_nodes + 1, dtype=np.int64)

    expanded = batched.expanded
    predicted_full_length = np.diff(node_offsets)
    is_cut = surviving < predicted_full_length

    # Surviving band cardinalities via the OWNED batched count kernel (the
    # same call the adapter makes) -- never re-implemented here.
    surviving_id, surviving_num = count_surviving_batched(
        expanded, node_offsets, surviving
    )

    # Kept subset = nodes with at least one surviving token, in DFS order.
    # ``np.flatnonzero`` preserves ascending node index == DFS order.
    kept_node_index = np.flatnonzero(surviving > 0).astype(np.int64)

    return DenseColumns(
        surviving_token_count=surviving,
        predicted_full_length=predicted_full_length.astype(np.int64),
        is_cut=is_cut,
        surviving_identity_count=np.asarray(surviving_id, dtype=np.int64),
        surviving_number_chunk_count=np.asarray(surviving_num, dtype=np.int64),
        raw_tokens=raw_flat,
        real_mask=batched.real_mask,
        number_mask=batched.number_mask,
        runlen_number=batched.runlen_number,
        is_negative_per_position=batched.is_negative_per_position,
        raw_offsets=raw_offsets,
        digit_cumsum=batched.digit_cumsum,
        digit_offsets=digit_offsets,
        expanded=expanded,
        extra_value_v2_mask=batched.extra_value_v2_mask,
        extra_f128_mask=batched.extra_f128_mask,
        node_offsets=node_offsets,
        kept_node_index=kept_node_index,
    )
