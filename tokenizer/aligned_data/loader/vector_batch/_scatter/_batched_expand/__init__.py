"""Batched promotion + strip + expand twin over the flat gathered bodies.

Single concern: run the per-node ``build_inline_decode_state`` +
``expand_tokens`` MATH (VC2 / F128 promotion, strip + shift, the
prepended calling-category self-token) as a few vectorized numpy passes
over the WHOLE flat CSR ``raw`` body stream at once -- the batched twin
of the per-node scalar drive in :mod:`.._expand`. This is to the expansion
what :func:`...batch_decode._bulk_expand_lengths.bulk_contributing_geometry`
is to the contributing-length scan: one boundary-aware pass instead of N
Python calls.

REUSE, NOT RE-IMPLEMENTATION (byte-identity contract): every rule this
package vectorizes is OWNED elsewhere and asserted equivalent by the
cross-check unit test + the corpus byte-identity gate --

* the per-stream masks + run-lengths + ``digit_cumsum`` +
  ``is_negative_per_position`` are the
  :class:`...decoded._inline_decode_state.InlineDecodeState` fields,
  reproduced boundary-aware in :mod:`._state_fields` so each per-node
  SLICE equals :func:`build_inline_decode_state` on that node's raw stream.
* VC2 / F128 promotion + strip + shift + self-token prepend are owned by
  :func:`...batch_decode._expand_tokens.expand_tokens`; the per-source
  chunk-count + ALG-2 NaN/Inf detection + the ``> 256`` keep predicate +
  the ``- 256`` shift are reproduced verbatim in :mod:`._rewrite`.

The malformed-stream guards (VC2 carrier at a node tail; F128 carrier
within 2 of a node tail) mirror the scalar asserts -- the same shapes
:func:`expand_tokens` rejects raise here.

Package layout:

* :mod:`._constants` -- the unified-vocab layout constants.
* :mod:`._state_fields` -- the boundary-aware InlineDecodeState fields.
* :mod:`._rewrite` -- the raw-stream promotion + strip-shift-prepend.
* :mod:`._expansion` -- the :class:`BatchedExpansion` record + orchestrator.
"""

from ._expansion import BatchedExpansion, batched_expand


__all__ = ["BatchedExpansion", "batched_expand"]
