"""Sorted-index build/read/sample pipeline for length-bucketed batches.

Public surface for the matched-arm sorted-index files consumed by the
batch-decode dataloader. See ``sorted-index-builder.md`` (plan) for the
full design.

Re-exports the pieces shipped so far; additional surface (builder,
reader, sampler, batch helper) lands in later phases.
"""

from __future__ import annotations

from ._modes import parse_reduction
from ._types import (
    LengthReduction,
    MultiBinaryBatchDecodeResult,
    MultiBinarySectionPointer,
    ReductionKind,
)
from ._wire import encode_sorted_index, parse_header

__all__ = [
    "LengthReduction",
    "MultiBinaryBatchDecodeResult",
    "MultiBinarySectionPointer",
    "ReductionKind",
    "encode_sorted_index",
    "parse_header",
    "parse_reduction",
]
