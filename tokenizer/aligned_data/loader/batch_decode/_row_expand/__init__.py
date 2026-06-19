"""Shared per-variant -> per-row expansion (package facade).

Single concern of the package: the per-row expansion primitives every
downstream stage module shares. Split into one submodule per concern --

* :mod:`._lookup` -- ``(section_idx, slot_idx) -> flat variant idx`` +
  padding mask (the per-row variant lookup).
* :mod:`._sizing` -- per-row length cumsum into ``row_offsets`` (the
  length-only mode, no flat payload).
* :mod:`._concat` -- per-row flat concatenation of per-variant payloads
  (list-input and already-concatenated buffer-input entries, sharing one
  vectorised fill core).

The public import surface is re-exported here so callers keep using
``from .._row_expand import build_per_row_variant_lookup`` etc.
unchanged after the split.
"""

from __future__ import annotations

from ._concat import concat_per_row, concat_per_row_from_buffer
from ._lookup import build_per_row_variant_lookup
from ._sizing import row_offsets_from_per_variant_lengths


__all__ = [
    "build_per_row_variant_lookup",
    "concat_per_row",
    "concat_per_row_from_buffer",
    "row_offsets_from_per_variant_lengths",
]
