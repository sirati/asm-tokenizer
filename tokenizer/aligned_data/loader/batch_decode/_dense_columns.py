"""The flat dense front-matter the 4 stage-3 sites consume.

Single concern (the design-first sentence): *own the COLUMNAR input
contract the four stage-3 dense-byte-stream sites
(:mod:`._inline_bytes` 3a, :mod:`._identity_decode` 3b,
:mod:`._number_decode._flat_segments` 3c, :mod:`._bulk_bytes` sign)
read -- the per-node RAW / DIGIT / EXPANDED CSR columns + per-node
scalars + the kept-node DFS index, in emitted-node (== canonical
stage-3 DFS call_target) order.*

This type carries NO build logic and NO decode rule. Two builders feed
it, both proven byte-identical to the per-call_target tree front-matter:

* the vector dense path builds it DIRECTLY from the flat
  ``BatchedExpansion`` + the per-node ``surviving`` clip
  (:func:`...vector_batch._scatter._dense_columns.build_dense_columns`),
  collapsing the four redundant tree-walks to ONE columnar build; and
* the staged ``batch_decode`` path builds it from its real
  ``Stage2Batch`` with a SINGLE DFS walk
  (:func:`._flat_call_targets.dense_columns_from_stage2`).

The per-node view accessors (``node_raw_slice`` etc.) surface the same
column slices the per-call_target sites read off the adapter's
``Stage2CallTarget`` / ``Stage1CallTarget.state`` views -- a site reads
a node's columns the same way whichever builder produced this object.

Module boundary: this type is the stage-3 dense sites' input. The only
things crossing the boundary are flat numpy arrays + per-node scalars;
no site sees a ``BatchedExpansion`` nor a ``Stage2Batch`` internal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["DenseColumns"]


@dataclass(frozen=True)
class DenseColumns:
    """The flat dense front-matter the 4 stage-3 sites consume.

    Built once (per builder) in emitted-node order (== the canonical
    stage-3 DFS call_target order). Every array is a VIEW or a fresh
    concatenation over the source flats -- the same data the
    per-call_target sites slice off the adapter's ``Stage2CallTarget`` /
    ``Stage1CallTarget.state`` views.

    Per-node scalar columns (FULL DFS axis, length ``n_nodes``)
    -----------------------------------------------------------
    surviving_token_count:
        ``int64[n_nodes]`` -- node ``e``'s surviving prefix length (the
        adapter's ``Stage2CallTarget.surviving_token_count`` /
        ``partial_cut_length``); the cut clip.
    predicted_full_length:
        ``int64[n_nodes]`` -- node ``e``'s full ``expanded`` length
        (``node_offsets`` diff); ``Stage2CallTarget.predicted_full_length``.
    is_cut:
        ``bool[n_nodes]`` -- ``surviving_token_count < predicted_full_length``
        (the adapter's ``is_cut``).
    surviving_identity_count / surviving_number_chunk_count:
        ``int64[n_nodes]`` -- the per-node IDENTITY / NUMBER band
        cardinalities over the surviving prefix (the adapter's
        homonymous scalar fields).

    RAW-space CSR (FULL DFS axis)
    -----------------------------
    raw_tokens / real_mask / number_mask / runlen_number /
    is_negative_per_position:
        The flat raw-space arrays (VIEWS, no copy).
    raw_offsets:
        ``int64[n_nodes + 1]`` -- the raw-space CSR; node ``e`` owns
        ``raw_tokens[raw_offsets[e] : raw_offsets[e + 1]]``.
    digit_cumsum:
        The flat ``digit_cumsum`` (VIEW). Node ``e``'s ``N + 1`` block is
        ``digit_cumsum[digit_offsets[e] : digit_offsets[e + 1]]``.
    digit_offsets:
        ``int64[n_nodes + 1]`` -- the digit-cumsum CSR
        (``raw_offsets + arange(n_nodes + 1)``, the per-node ``+1``-slot
        packing).

    EXPANDED-space CSR (FULL DFS axis)
    ----------------------------------
    expanded / extra_value_v2_mask / extra_f128_mask:
        The flat expanded-space arrays (VIEWS).
    node_offsets:
        ``int64[n_nodes + 1]`` -- the expanded-space CSR.

    Kept subset (``surviving_token_count > 0``, DFS order)
    ------------------------------------------------------
    kept_node_index:
        ``int64[n_kept]`` -- the FULL-DFS node index of each kept node
        (the ``dfs_index`` / ``ct_index`` the per-DFS-call_target slice
        lists are keyed by). Equals ``flatten_call_targets`` / 3c /
        sign's ``ct_index``.
    """

    surviving_token_count: np.ndarray
    predicted_full_length: np.ndarray
    is_cut: np.ndarray
    surviving_identity_count: np.ndarray
    surviving_number_chunk_count: np.ndarray

    raw_tokens: np.ndarray
    real_mask: np.ndarray
    number_mask: np.ndarray
    runlen_number: np.ndarray
    is_negative_per_position: np.ndarray
    raw_offsets: np.ndarray
    digit_cumsum: np.ndarray
    digit_offsets: np.ndarray

    expanded: np.ndarray
    extra_value_v2_mask: np.ndarray
    extra_f128_mask: np.ndarray
    node_offsets: np.ndarray

    kept_node_index: np.ndarray

    # ------------------------------------------------------------------
    # Per-node VIEW accessors -- the lazy column read each stage-3 site
    # does off the adapter's ``Stage2CallTarget`` / ``state``. Object
    # reuse is irrelevant here (these are cheap numpy slices, no wrapper
    # materialisation); they exist so a site reads a node's columns
    # without re-deriving the CSR base every call.
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the full DFS enumeration (slice axis)."""
        return int(self.surviving_token_count.shape[0])

    def node_raw_slice(self, e: int) -> slice:
        """RAW-space slice of node ``e`` (``raw_tokens`` etc.)."""
        return slice(int(self.raw_offsets[e]), int(self.raw_offsets[e + 1]))

    def node_digit_slice(self, e: int) -> slice:
        """DIGIT-cumsum slice of node ``e`` (``N + 1`` slots)."""
        return slice(
            int(self.digit_offsets[e]), int(self.digit_offsets[e + 1])
        )

    def node_expanded_slice(self, e: int) -> slice:
        """EXPANDED-space slice of node ``e`` (``expanded`` / masks)."""
        return slice(int(self.node_offsets[e]), int(self.node_offsets[e + 1]))
