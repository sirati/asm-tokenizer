"""Sorted-index build/read/sample pipeline for length-bucketed batches.

Public surface for the matched-arm sorted-index files consumed by the
batch-decode dataloader. See ``sorted-index-builder.md`` (plan) for the
full design.

Re-exports the types + parser shipped so far; additional pieces
(builder, reader, sampler, batch helper) land in later phases.
"""

from __future__ import annotations

from ._modes import parse_reduction
from ._types import (
    LengthReduction,
    MultiBinaryBatchDecodeResult,
    MultiBinarySectionPointer,
    ReductionKind,
)


__all__ = [
    "LengthReduction",
    "MultiBinaryBatchDecodeResult",
    "MultiBinarySectionPointer",
    "ReductionKind",
    "parse_reduction",
]
