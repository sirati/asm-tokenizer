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
from ._validation import SequentialValidationSampler, ValidationBatch
from ._validation_batch import open_validation_batches


__all__ = [
    "CrossSpecSortedIndexSampler",
    "DecodeEngine",
    "MultiBinarySortedIndexSampler",
    "SequentialValidationSampler",
    "ValidationBatch",
    "decode_pointer_batch",
    "open_length_bucketed_batch",
    "open_validation_batches",
    "sample_section_pointers",
]
