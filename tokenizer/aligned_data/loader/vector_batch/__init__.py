"""Vectorized batch dataloader -- the body-free geometry PREPASS (plan C1).

Public surface: :func:`compute_batch_geometry` -- the geometry-first
prepass that computes the entire ``[B, L]`` token layout + the dense-
sidecar reservations + the remembered-excluded backfill pool for a
batch, from SIDECARS ONLY (sections.bin + RLG3 geometry + _variants.bin),
with ZERO body decode and zero ``_data.bin`` read. It DEFINES the typed
:class:`BatchGeometry` result the later fused scatter (TC2) and entry
(TC3) consume.

Single-concern submodules:

* :mod:`._types` -- the typed result contract (DEFINED here, consumed by
  TC2 / TC3);
* :mod:`._inclusion` -- the shared once-only BFS over the catalog
  adjacency -> per-row ordered emitted nodes (BFS order) + the
  remembered-excluded pool;
* :mod:`._prefix` -- the body-free variant-prefix (``n_axis``) width from
  ``_variants.bin``;
* :mod:`._layout` -- the per-row token-column prefix-sum + the single
  straddler cut against ``L``;
* :mod:`._reserve` -- the per-row dense id / value reservation totals +
  offsets;
* :mod:`._geometry` -- the orchestrator that composes the above.
"""

from ._geometry import compute_batch_geometry
from ._types import (
    BatchGeometry,
    BatchRowEmission,
    BatchTokenLayout,
    DenseReservation,
)


__all__ = [
    "compute_batch_geometry",
    "BatchGeometry",
    "BatchRowEmission",
    "BatchTokenLayout",
    "DenseReservation",
]
