"""Per-row band-emit walker for the BatchDecodeBackend.

Single concern: translate one row of :attr:`BatchDecodeResult.tokens`
+ its aligned sidecars into ``[RowSection, ...]`` -- variant header,
LOCAL_FUNC self-prepend (Function ID), and per-basic-block BODY
sections with the ``Block_Def`` + ``block_v2`` header pair consumed
silently (the parent tree row's label already encodes the block
index).

The implementation is split across the subpackage submodules:

* :mod:`._state` -- :class:`_WalkState` dataclass + per-instruction
  :class:`_InsnEmitPolicy` enum + the ``Category -> CallTargetType``
  reverse map.
* :mod:`._instruction` -- per-instruction text accumulation +
  bracket-aware text join (W3-11). Pre-paved for the R2c
  per-instruction collector wiring.
* :mod:`._dispatch` -- IDENTITY-band per-Category dispatch (block /
  function / jump-table-footer).
* :mod:`._driver` -- per-col loop + section transitions.

Plan reference: ``inspector-render-backends.md`` §6 + decisions
#16/#17/#18/#29/#30; ``inspector-followup.md`` W3-11 / W3-16 /
A-L2 H1+H2 / cluster #5 (subpackage split).
"""

from __future__ import annotations

from .._sections import RowSection
from ._driver import render_row_blocks


__all__ = ["RowSection", "render_row_blocks"]
