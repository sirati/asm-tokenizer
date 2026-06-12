"""Realized-token-length sidecars: standalone generator + mmap reader.

A self-contained pass that persists each variant's realized record-body
length (the contributing body length the bulk expand engine computes,
excluding the self/identity token, variant-axis tokens, and spliced
function tokens) into directly-addressable per-binary sidecar files, for
both the matched and unmatched arms.

Four files per binary, per arm:

* ``<binary>_lengths.bin`` / ``<binary>_unmatched_lengths.bin`` -- a
  ``u32`` body, one realized length per (section, variant), section-major.
* ``<binary>_lengths_index.bin`` / ``<binary>_unmatched_lengths_index.bin``
  -- the ``n_sections + 1`` ``u32`` CSR jump table addressing the body.

The generator runs as its own pass (CLI: ``python -m
tokenizer.aligned_data.realized_lengths``), BEFORE the sorted-index
build; the sorted-index builder will later consume these sidecars
instead of recomputing lengths.

Public API:

* :func:`generate_realized_lengths` -- write all four sidecars for one
  binary.
* :class:`RealizedLengths` -- lazy zero-copy mmap reader for one arm's
  pair.
* :func:`realized_lengths_present` / :func:`require_realized_lengths` --
  existence / pre-flight helpers so consumers fail with a clear "run the
  generator first" message.
* :data:`MATCHED_ARM` / :data:`UNMATCHED_ARM` / :data:`ARMS` -- the typed
  arm selectors threaded through the generator + reader.
"""

from __future__ import annotations

from ._format import ARMS, MATCHED_ARM, UNMATCHED_ARM, RealizedLengthsArm
from ._generate import generate_realized_lengths
from ._reader import (
    RealizedLengths,
    realized_lengths_present,
    require_realized_lengths,
)


__all__ = [
    "generate_realized_lengths",
    "RealizedLengths",
    "realized_lengths_present",
    "require_realized_lengths",
    "RealizedLengthsArm",
    "MATCHED_ARM",
    "UNMATCHED_ARM",
    "ARMS",
]
