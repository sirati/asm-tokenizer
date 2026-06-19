"""Slim per-row ``Stage2Batch`` for the vector dense path (NO object tree).

Single concern (the design-first sentence): *given the per-row emission
CSR + the per-emitted-node surviving identity / number-chunk counts the
:class:`...._dense_columns.DenseColumns` already carries, produce the SLIM
``Stage2Batch`` the vector dense path's downstream stages read off --
``stage1.batch_idx_to_section_variant`` (the identity mapping) +
``identity_row_offsets`` + ``number_row_offsets`` -- WITHOUT building the
per-call-target / per-variant / per-section object tree.*

Why the tree is gone (step-5 object-tree elimination): post 5a/5b the
vector dense path reads EVERY per-node body COLUMNAR -- the four stage-3
emission sites + the per-source sign collection + the per-row remap + the
number-chunk stream all source the flat :class:`DenseColumns` /
:class:`...._remap_inputs.FlatRemapInputs` / :class:`...._number_chunk_
columns.NumberChunkColumns`, never the per-call-target ``Stage2CallTarget``
dataclasses. The ONLY remaining reader of the per-CT tree on this path was
:func:`...batch_decode._bulk_bytes._assemble_hierarchy`, building
``Stage3Batch.sections`` wrappers nothing downstream reads -- so the tree
is vestigial and ``build_bulk_bytes`` skips it when a pre-built ``dense``
is supplied. The vector path therefore needs only this SLIM batch's three
columnar arrays; the per-CT dataclass constructor loop (the ~24.7%
GIL-held per-batch cost) is dropped.

What the downstream readers see (the API surface crossing this boundary):

* :func:`...batch_decode._bulk_bytes.build_bulk_bytes` -- threads this
  ``Stage2Batch`` onto ``Stage3Batch.stage2`` (it builds NO hierarchy on
  the vector path).
* :func:`...batch_decode._dedup_walk.apply_per_row_remap` /
  :func:`...batch_decode._sidecar_concat.assemble_number_sidecars` -- read
  ``stage3.stage2.stage1.batch_idx_to_section_variant`` +
  ``stage3.stage2.number_row_offsets`` (the per-row variant lookup + the
  number-sidecar sizing). Both thread their ``variants_per_section`` /
  ``numbers`` columnar on this path, so neither walks ``.sections``.
* :func:`._dense.build_dense_sidecars` -- reads ``identity_row_offsets`` /
  ``number_row_offsets`` to re-expand the per-row arrays onto the batch.

None of those reach ``stage2.sections`` / ``stage1.sections`` -- so they
stay EMPTY here (the staged ``batch_decode`` path keeps its full tree).

The synthetic ``batch_idx_to_section_variant`` is the IDENTITY over
non-padding rows (section ``r`` == non-padding row ``r``, variant 0) so the
kernels' per-row offset cumsums enumerate the non-padding rows in order;
the orchestrator (:mod:`._dense`) re-expands onto the true batch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    Stage1Batch,
    Stage2Batch,
)

from .._types import BatchGeometry
from ._dense_columns import DenseColumns


__all__ = ["build_slim_stage2_batch"]


def build_slim_stage2_batch(
    geometry: BatchGeometry,
    dense: Optional[DenseColumns],
) -> Stage2Batch:
    """Build the SLIM (tree-free) ``Stage2Batch`` for the vector dense path.

    Parameters
    ----------
    geometry:
        The body-free prepass result. ``geometry.emission.row_offsets`` is
        the per-row node CSR (row ``r`` owns emitted nodes
        ``[row_offsets[r] : row_offsets[r + 1]]``); ``geometry.n_rows`` is
        the non-padding row count.
    dense:
        The shared :class:`._dense_columns.DenseColumns` (built ONCE from
        the ``BatchedExpansion``), carrying the per-emitted-node surviving
        identity / number-chunk counts (= the same counts the deleted per-CT
        adapter summed per row). ``None`` on the empty batch (no emitted
        node), where every per-row count is zero.

    Returns
    -------
    Stage2Batch
        Carries ``stage1.batch_idx_to_section_variant`` (the identity
        mapping) + ``identity_row_offsets`` + ``number_row_offsets``;
        ``sections`` / ``stage1.sections`` are EMPTY (the vector dense path
        reads no per-CT tree).
    """
    n_rows = int(geometry.n_rows)
    mapping = _identity_mapping(n_rows)
    identity_row_offsets, number_row_offsets = _row_offset_cumsums(
        geometry, dense, n_rows
    )
    stage1_batch = Stage1Batch(
        sections=[],
        batch_idx_to_section_variant=mapping,
        batch_size=n_rows,
    )
    return Stage2Batch(
        stage1=stage1_batch,
        sections=[],
        identity_row_offsets=identity_row_offsets,
        number_row_offsets=number_row_offsets,
    )


def _identity_mapping(n_rows: int) -> np.ndarray:
    """``u32[n_rows, 2]`` identity mapping (row r -> section r, variant 0)."""
    mapping = np.zeros((n_rows, 2), dtype=np.uint32)
    mapping[:, 0] = np.arange(n_rows, dtype=np.uint32)
    return mapping


def _row_offset_cumsums(
    geometry: BatchGeometry,
    dense: Optional[DenseColumns],
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(identity_row_offsets, number_row_offsets)`` -- u32[n_rows + 1].

    Per-row cumsum of the per-row surviving identity / number-chunk counts
    -- the SAME quantities the deleted per-CT adapter summed over each row's
    call_targets (``_stage2_variant``'s ``total_surviving_identity_count`` /
    ``total_surviving_number_chunk_count``), here reduced COLUMNAR from the
    per-emitted-node counts :class:`DenseColumns` carries + the per-row node
    CSR. ``np.add.reduceat`` over ``row_offsets[:-1]`` sums each row's node
    run; empty rows (``row_offsets[r] == row_offsets[r + 1]``) contribute
    zero (handled below -- ``reduceat`` is undefined on a zero-length
    segment, so we mask those rows back to zero).
    """
    identity_row_offsets = np.zeros(n_rows + 1, dtype=np.uint32)
    number_row_offsets = np.zeros(n_rows + 1, dtype=np.uint32)
    if n_rows == 0 or dense is None:
        return identity_row_offsets, number_row_offsets

    row_offsets = np.asarray(
        geometry.emission.row_offsets, dtype=np.int64
    ).reshape(-1)
    _row_band_cumsum(
        np.asarray(dense.surviving_identity_count, dtype=np.int64),
        row_offsets,
        out=identity_row_offsets,
    )
    _row_band_cumsum(
        np.asarray(dense.surviving_number_chunk_count, dtype=np.int64),
        row_offsets,
        out=number_row_offsets,
    )
    return identity_row_offsets, number_row_offsets


def _row_band_cumsum(
    per_node: np.ndarray,
    row_offsets: np.ndarray,
    *,
    out: np.ndarray,
) -> None:
    """Write the per-row CSR cumsum of ``per_node`` summed over node runs.

    ``out`` is ``u32[n_rows + 1]``; ``out[r + 1] - out[r]`` is the sum of
    ``per_node`` over row ``r``'s node run ``[row_offsets[r] :
    row_offsets[r + 1]]``. Computed via a node-prefix cumsum gathered at the
    row boundaries (``node_prefix[row_offsets]``) -- robust for empty rows
    (a zero-length run yields a zero delta) AND a trailing empty row (the
    boundary lands at ``len(per_node)``), with NONE of ``np.add.reduceat``'s
    zero-length-segment pitfalls. This reproduces the deleted per-CT
    adapter's per-row sum (``_stage2_variant``'s aggregate count, an empty
    call_target list summing to zero) exactly.
    """
    node_prefix = np.zeros(per_node.shape[0] + 1, dtype=np.int64)
    np.cumsum(per_node, out=node_prefix[1:])
    boundaries = node_prefix[row_offsets]  # int64[n_rows + 1]
    out[:] = boundaries.astype(np.uint32)
