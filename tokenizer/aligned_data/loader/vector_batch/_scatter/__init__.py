"""Fused vectorized scatter for the vector-batch dataloader (plan C2).

Public surface: :func:`scatter_batch_tokens` -- given the body-free
:class:`...vector_batch.BatchGeometry` prepass result + the session's
``_data.bin`` / ``_variants.bin`` / catalog bytes, assemble the model-
facing ``tokens[B, L]`` tensor by reading each emitted node's record
ONCE (one batched gather) and scattering at the prepass's precomputed
BFS columns -- no per-edge body re-read, no per-row Python assembly.

Single-concern submodules:

* :mod:`._locator` -- emission node -> ``_data.bin`` token span.
* :mod:`._body_load` -- one vectorized gather of every node's raw stream.
* :mod:`._expand` -- per-node promotion / strip / shift (owned
  ``expand_tokens`` semantics) + self-token, flattened CSR.
* :mod:`._prefix_values` -- per-row variant-prefix ids (shifted),
  body-free from ``_variants.bin``.
* :mod:`._token_scatter` -- the single batched ``[B, L]`` scatter +
  straddler cut.
* :mod:`._surviving` -- the per-node surviving-column count under the
  straddler cut (shared by the token scatter + the dense pass).
* :mod:`._dense_adapter` -- flat emission -> staged ``Stage2Batch`` for
  the dense kernels.
* :mod:`._dense` -- the dense identity + numeric sidecar producer (reuses
  the ``batch_decode`` decode kernels; byte-identical with backfill OFF).
* :mod:`._scatter` -- the orchestrator.
"""

from ._dense import DenseSidecars, build_dense_sidecars
from ._scatter import ScatteredTokens, scatter_batch_tokens


__all__ = [
    "DenseSidecars",
    "ScatteredTokens",
    "build_dense_sidecars",
    "scatter_batch_tokens",
]
