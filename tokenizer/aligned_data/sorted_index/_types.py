"""Typed dataclasses + enums for the sorted-index module.

This file defines the typed surface used across the sorted-index
pipeline (build, read, sample, cross-binary batch helper). The types
here are intentionally pure dataclasses + enums with no I/O, no
file-format awareness, and no batch-decode pipeline imports beyond the
two already-typed pieces (``SectionPointerSpec``, ``BatchDecodeResult``)
they extend.

Concerns owned by this module:

- :class:`ReductionKind` + :class:`LengthReduction` -- the typed
  parameter that controls how per-variant lengths collapse to a single
  per-section key length (plan D2).
- :class:`MultiBinarySectionPointer` -- typed cross-binary section
  pointer; replaces ad-hoc ``(binary_name, section_idx)`` tuples
  (plan D7).
- :class:`MultiBinaryBatchDecodeResult` -- extension of
  :class:`BatchDecodeResult` with a per-row ``binary_id`` sidecar so
  cross-binary batches retain provenance (plan D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
)


__all__ = [
    "ReductionKind",
    "LengthReduction",
    "MultiBinarySectionPointer",
    "MultiBinaryBatchDecodeResult",
]


class ReductionKind(Enum):
    """Discriminator for :class:`LengthReduction`.

    ``MAX`` collapses per-variant lengths via :func:`numpy.ndarray.max`;
    ``PERCENTILE`` collapses via :func:`numpy.percentile` with
    ``method="lower"`` so the result is always one of the input values.
    """

    MAX = "max"
    PERCENTILE = "percentile"


@dataclass(frozen=True)
class LengthReduction:
    """Per-section length aggregator across N variants (plan D2).

    A typed parameter understood by every layer of the sorted-index
    pipeline. ``filename_tag`` produces the canonical CLI/filename
    spelling (``"max"``, ``"p05"``, ``"p95"``) used by the wire-format
    regex; ``reduce`` performs the actual aggregation over a length
    vector for one section.

    Construction validation (:meth:`__post_init__`):

    - :attr:`ReductionKind.MAX` requires ``percentile is None``.
    - :attr:`ReductionKind.PERCENTILE` requires ``percentile`` in
      ``[1, 99]``. ``p100`` collapses to ``MAX`` at the parser
      boundary (:func:`._modes.parse_reduction`) -- direct
      construction with ``percentile=100`` is rejected so the
      canonical form is the only representation.
    """

    kind: ReductionKind
    percentile: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind is ReductionKind.MAX:
            if self.percentile is not None:
                raise ValueError(
                    f"MAX reduction takes no percentile; got {self.percentile!r}",
                )
            return
        # PERCENTILE
        if self.percentile is None:
            raise ValueError("PERCENTILE reduction requires a percentile value")
        if not (1 <= self.percentile <= 99):
            raise ValueError(
                f"percentile must be in [1, 99]; got {self.percentile!r}",
            )

    def filename_tag(self) -> str:
        """Canonical CLI/filename spelling.

        ``"max"`` for :attr:`ReductionKind.MAX`; zero-padded
        ``"p{percentile:02d}"`` for :attr:`ReductionKind.PERCENTILE` so
        files lexsort by percentile.
        """
        if self.kind is ReductionKind.MAX:
            return "max"
        # __post_init__ guarantees percentile is not None for PERCENTILE.
        assert self.percentile is not None
        return f"p{self.percentile:02d}"

    def reduce(self, lengths: np.ndarray) -> int:
        """Collapse a vector of per-variant lengths to one int key.

        Empty input -> ``0`` (0-variant sections stamp 0 directly;
        see plan ALG-1).
        """
        if lengths.size == 0:
            return 0
        if self.kind is ReductionKind.MAX:
            return int(lengths.max())
        # __post_init__ guarantees percentile is not None for PERCENTILE.
        assert self.percentile is not None
        return int(np.percentile(lengths, self.percentile, method="lower"))


@dataclass(frozen=True)
class MultiBinarySectionPointer:
    """Typed cross-binary section pointer (plan D7).

    Replaces ad-hoc ``(binary_name, section_idx)`` tuples used at the
    cross-binary sampler boundary. ``section_pointer`` carries the
    single-binary :class:`SectionPointerSpec` (which already encodes
    ``SectionKind.MATCHED`` + per-arm idx); ``binary_name`` selects
    which per-binary session to open.
    """

    binary_name: str
    section_pointer: SectionPointerSpec


@dataclass(frozen=True)
class MultiBinaryBatchDecodeResult:
    """Extension of :class:`BatchDecodeResult` with cross-binary identity.

    The inner :class:`BatchDecodeResult` is the concatenated per-binary
    result (see plan ALG-6); ``binary_id_per_row`` is a per-row index
    into ``binary_names`` so each batch row can be traced back to its
    source binary (plan D7).
    """

    inner: BatchDecodeResult
    binary_id_per_row: np.ndarray
    """``u32[batch_size]`` -- index into :attr:`binary_names`."""

    binary_names: List[str]
    """``binary_id -> binary_name`` reverse map; alphabetical order
    matching :attr:`MultiBinarySortedIndexSampler.binary_names`."""
