"""Batch-vectorized v2 dataloader entry point.

Public surface for the staged batch-decode pipeline (see
``batch_decode_plan.md`` for the design). The pipeline runs in four
stages (load -> length-predict -> bulk-bytes -> assemble) whose
hierarchical handoff types live in :mod:`._types`.

Re-exports are added by their owning phase. Phase 0b contributed the
staged dataclasses, the :class:`VariantPadding` policy enum, and the
:class:`SectionPointerSpec` request type. Subsequent phases (1-4) wire
up the per-stage implementations; phase 4 adds the ``batch_decode``
entry-point.
"""

from __future__ import annotations

from ._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
)

# TODO(phase 4): re-export the ``batch_decode`` entry-point once
# Phase 4 lands. It is intentionally absent from this package until
# then -- importing it from outside is expected to fail.

__all__ = [
    "BatchDecodeResult",
    "SectionPointerSpec",
    "VariantPadding",
]
