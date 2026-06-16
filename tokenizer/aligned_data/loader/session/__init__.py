"""Per-binary, batch-scoped session handle (:class:`BinarySession`).

This package splits the session implementation across focused
submodules under the project's file-size cap, while preserving the
historical public import surface: ``from ...loader.session import
BinarySession`` keeps resolving unchanged.

  * :mod:`._session`        -- handle lifecycle + lazy openers +
                               record slicing + metadata accessors
                               (the ``BinarySession`` class itself).
  * :mod:`._matched_load`   -- matched-arm load + per-variant body
                               mixin.
  * :mod:`._unmatched_load` -- unmatched-arm load + section-lookup
                               mixin.
"""

from __future__ import annotations

from ._session import BinarySession, _close_memmap

__all__ = ["BinarySession", "_close_memmap"]
