"""Per-binary, batch-scoped session handle (:class:`BinarySession`).

This package splits the session implementation across focused
submodules under the project's file-size cap, while preserving the
historical public import surface: ``from ...loader.session import
BinarySession`` keeps resolving unchanged.

  * :mod:`._session`        -- handle lifecycle + record slicing +
                               metadata accessors (the
                               ``BinarySession`` class itself).
  * :mod:`._handles`        -- lazy memmap-handle acquisition mixin.
  * :mod:`._matched_load`   -- matched-arm load + per-variant body
                               mixin.
  * :mod:`._unmatched_load` -- unmatched-arm load + section-lookup
                               mixin.
"""

from __future__ import annotations

from ._handles import _close_memmap
from ._session import BinarySession

__all__ = ["BinarySession", "_close_memmap"]
