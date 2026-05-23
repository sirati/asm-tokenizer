"""Batch-vectorized v2 dataloader entry point.

Public surface for the staged batch-decode pipeline (see
``batch_decode_plan.md`` for the design). The pipeline runs in four
stages (load -> length-predict -> bulk-bytes -> assemble) whose
hierarchical handoff types live in :mod:`._types`.

Re-exports:

- :class:`BatchDecodeResult` -- user-facing flat-tensor result.
- :func:`batch_decode` -- end-to-end pipeline entry. Currently a stub
  raising ``NotImplementedError`` until the stage implementations are
  wired together.
- :class:`VariantPadding` -- runtime enum controlling how short sections
  pad into the linear batch layout.
- :class:`SectionPointerSpec` -- typed ``(arm, idx)`` section pointer
  request type.
"""

from __future__ import annotations

from ._entry import batch_decode
from ._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
)

__all__ = [
    "BatchDecodeResult",
    "SectionPointerSpec",
    "VariantPadding",
    "batch_decode",
]
