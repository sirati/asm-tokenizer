"""Test-only: rebuild the full per-CT ``Stage3Batch`` tree (the gate oracle).

Step-5 object-tree elimination dropped the per-call-target ``Stage2Batch`` /
``Stage3Batch`` tree from the PRODUCTION vector dense path (``_dense`` builds
the slim :mod:`..._scatter._slim_stage2` batch + ``build_bulk_bytes`` skips
the Stage3 hierarchy when fed a columnar ``dense``). The ``test_*_equiv``
gates still need the tree-walk as their ORACLE -- so they rebuild it HERE,
from the SAME columnar inputs the production path computed
(``geometry`` + ``dense`` + ``catalog``), via the retained tree adapter
(:func:`..._scatter._dense_adapter.build_stage2_batch`) + a FULL
``build_bulk_bytes`` (``dense=None`` forces the hierarchy build). Asserting
the columnar production output equals THIS independently-rebuilt tree's walk
is exactly the equivalence the gates pin.

The adapter needs an :class:`...._scatter._expand.ExpandedBatch`; the only
fields it reads (``expanded`` + ``node_offsets``) AND the per-node surviving
counts it sums live on the captured :class:`DenseColumns`, so the oracle is
a pure function of the production columnar inputs (no re-decode).
"""

from __future__ import annotations

import contextlib
import unittest.mock as _mock

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._bulk_bytes import (
    build_bulk_bytes,
)
from tokenizer.aligned_data.loader.batch_decode._dense_columns import (
    DenseColumns,
)
from tokenizer.aligned_data.loader.batch_decode._types import Stage3Batch
from tokenizer.aligned_data.loader.vector_batch._scatter import (
    _dense as _dense_mod,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._catalog_columns import (
    CatalogColumns,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._dense_adapter import (
    build_stage2_batch,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._expand import (
    ExpandedBatch,
)
from tokenizer.aligned_data.loader.vector_batch._types import BatchGeometry


__all__ = ["build_full_tree_stage3", "capture_columnar_inputs"]


@contextlib.contextmanager
def capture_columnar_inputs():
    """Capture each non-empty batch's ``(geometry, dense, catalog)`` triple.

    Patches the production ``build_flat_remap_inputs`` seam (called once per
    non-empty :func:`...._scatter._dense.build_dense_sidecars` with exactly
    those three columnar inputs) to APPEND the triple, then delegate to the
    real builder unchanged. The equivalence gates then rebuild the full-tree
    ``Stage3Batch`` oracle from each captured triple via
    :func:`build_full_tree_stage3`. Yields the capture list (one entry per
    non-empty decoded batch, in call order).
    """
    captured: list[tuple[BatchGeometry, DenseColumns, CatalogColumns]] = []
    real = _dense_mod.build_flat_remap_inputs

    def _capturing(geometry, dense, catalog):
        captured.append((geometry, dense, catalog))
        return real(geometry, dense, catalog)

    with _mock.patch.object(
        _dense_mod, "build_flat_remap_inputs", _capturing
    ):
        yield captured


def build_full_tree_stage3(
    geometry: BatchGeometry,
    dense: DenseColumns,
    catalog: CatalogColumns,
) -> Stage3Batch:
    """Rebuild the full per-CT ``Stage3Batch`` tree from the columnar inputs.

    Parameters
    ----------
    geometry / catalog:
        The production prepass geometry + per-emitted-node catalog columns
        (the same objects ``_dense`` threads to the columnar builders).
    dense:
        The production :class:`DenseColumns` -- carries the expanded stream
        (``expanded`` + ``node_offsets``) the tree adapter slices per node
        AND the per-node surviving token count
        (``surviving_token_count``) it uses as each call_target's
        ``partial_cut_length``.

    Returns
    -------
    Stage3Batch
        The full-tree stage 3 (``sections`` populated), whose tree-walk the
        equivalence gates use as the oracle.
    """
    expanded = ExpandedBatch(
        expanded=np.asarray(dense.expanded),
        node_offsets=np.asarray(dense.node_offsets, dtype=np.int64),
    )
    surviving = np.asarray(dense.surviving_token_count, dtype=np.int64)
    # The adapter's per-CT tree carries the catalog FID fields + topology
    # the hierarchy walk reads; its per-node BODIES are slim singletons, but
    # the Stage3 slices ``_assemble_hierarchy`` attaches are computed from
    # ``dense`` (NOT the bodies) -- so this reproduces EXACTLY the tree the
    # production path assembled at HEAD (``build_bulk_bytes(slim_stage2,
    # dense)`` with the hierarchy ON).
    tree_stage2 = build_stage2_batch(
        geometry, expanded, catalog=catalog, surviving=surviving
    )
    return build_bulk_bytes(tree_stage2, dense, build_hierarchy=True)
