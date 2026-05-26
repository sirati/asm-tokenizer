"""BatchDecodeBackend -- :class:`RenderBackend` over flat ``BatchDecodeResult`` tensors.

Phase B1 of the Wave-5 plan ships this sub-package; concrete classes
stay private (per plan decision #15) -- the public surface is the
:class:`BatchDecodeBackend` class re-exported here. The backend is
consumed by :mod:`tokenizer.inspector._backend_factory`'s opener for
the ``--memmap`` (stage-3) provider.

Plan reference: ``inspector-render-backends.md`` §6.
"""

from __future__ import annotations

from ._backend import BatchDecodeBackend


__all__ = ["BatchDecodeBackend"]
