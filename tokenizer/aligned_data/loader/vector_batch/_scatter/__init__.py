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
* :mod:`._scatter` -- the orchestrator.
"""

from ._scatter import ScatteredTokens, scatter_batch_tokens


__all__ = ["ScatteredTokens", "scatter_batch_tokens"]
