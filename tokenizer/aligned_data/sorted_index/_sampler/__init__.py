"""Cross-binary sample -> per-binary decode -> concat pipeline.

The three submodules form one user-facing flow:

* :mod:`._sample` -- Reading-A unbiased sampler over per-binary
  :class:`SortedIndexReader` urns (plan ALG-3 + D6).
* :mod:`._concat` -- stitch per-binary :class:`BatchDecodeResult`
  instances into one :class:`MultiBinaryBatchDecodeResult` (plan
  ALG-6).
* :mod:`._batch` -- top-level length-bucketed batch helper that
  wires sampling + decoding + concatenation together (plan D7).
"""

from __future__ import annotations

from ._batch import decode_pointer_batch, open_length_bucketed_batch
from ._concat import _concat_results, _concat_row_offsets
from ._sample import MultiBinarySortedIndexSampler, sample_section_pointers


__all__ = [
    "MultiBinarySortedIndexSampler",
    "decode_pointer_batch",
    "open_length_bucketed_batch",
    "sample_section_pointers",
]
