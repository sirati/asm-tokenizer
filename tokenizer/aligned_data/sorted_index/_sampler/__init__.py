"""Cross-binary sample -> per-binary decode -> concat pipeline.

The three submodules form one user-facing flow:

* :mod:`._sample` -- Reading-A unbiased sampler over per-binary
  :class:`SortedIndexReader` urns (plan ALG-3 + D6).
* :mod:`._concat` -- stitch per-binary :class:`BatchDecodeResult`
  instances into one :class:`MultiBinaryBatchDecodeResult` (plan
  ALG-6).
* :mod:`._batch` -- top-level length-bucketed batch helper that
  wires sampling + decoding + concatenation together (plan D7).
* :mod:`._engine` -- the selectable per-binary-group decode engine
  (batch_decode vs vector_batch) the batch helper dispatches through.

Two deterministic, rng-free enumeration siblings sit alongside the urn
samplers (each a MODULAR sibling, never a branch in the others):

* :mod:`._validation` -- the ordered shuffle/chunk/drop validation sampler
  (a deterministic per-section variant subset).
* :mod:`._exhaustive` -- the whole-corpus EXHAUSTIVE enumeration (every
  non-excluded section once, all variants, NO rng/shuffle/chunk/drop).
"""

from __future__ import annotations

from ._batch import decode_pointer_batch, open_length_bucketed_batch
from ._concat import _concat_results, _concat_row_offsets
from ._engine import DecodeEngine
from ._sample import (
    CrossSpecSortedIndexSampler,
    MultiBinarySortedIndexSampler,
    sample_section_pointers,
)
from ._exhaustive import ExhaustiveSectionSampler, all_section_pointers
from ._exhaustive_batch import open_exhaustive_batches
from ._validation import SequentialValidationSampler, ValidationBatch
from ._validation_batch import open_validation_batches


__all__ = [
    "CrossSpecSortedIndexSampler",
    "DecodeEngine",
    "ExhaustiveSectionSampler",
    "MultiBinarySortedIndexSampler",
    "SequentialValidationSampler",
    "ValidationBatch",
    "all_section_pointers",
    "decode_pointer_batch",
    "open_exhaustive_batches",
    "open_length_bucketed_batch",
    "open_validation_batches",
    "sample_section_pointers",
]
