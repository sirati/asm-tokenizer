"""Sorted-index build/read/sample pipeline for length-bucketed batches.

Public surface for the matched-arm sorted-index files consumed by the
batch-decode dataloader. See ``sorted-index-builder.md`` (plan) for the
full design.

Re-exports the pieces shipped so far; additional surface (builder,
reader, sampler, batch helper) lands in later phases.
"""

from __future__ import annotations

from ._builder import build_sorted_index_bytes, write_sorted_index_files
from ._length_compute import compute_reduced_lengths
from ._modes import parse_reduction
from ._reader import SortedIndexReader, discover_indices
from ._sampler import (
    MultiBinarySortedIndexSampler,
    open_length_bucketed_batch,
)
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
    "MultiBinarySortedIndexSampler",
    "ReductionKind",
    "SortedIndexReader",
    "build_sorted_index_bytes",
    "compute_reduced_lengths",
    "discover_indices",
    "encode_sorted_index",
    "open_length_bucketed_batch",
    "parse_header",
    "parse_reduction",
    "write_sorted_index_files",
]
