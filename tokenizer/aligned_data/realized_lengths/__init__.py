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

Alongside the length sidecars, a SUPERSET realized-GEOMETRY pair per arm
(``<binary>_realized.bin`` / ``<binary>_unmatched_realized.bin`` +
``*_realized_index.bin``) stores the full
``bulk_contributing_geometry`` triple ``(body_len, id_count,
value_count)`` per (section, variant): three parallel ``u32`` blocks
under one shared CSR. The ``body_len`` block is byte-identical to the
matching ``_lengths.bin`` body (one shared dedup loop produces both).

The generator runs as its own pass (CLI: ``python -m
tokenizer.aligned_data.realized_lengths``), BEFORE the sorted-index
build; the sorted-index builder will later consume these sidecars
instead of recomputing lengths.

Public API:

* :func:`generate_realized_lengths` / :func:`generate_realized_geometry`
  -- write all four length sidecars / both geometry pairs for one binary.
* :class:`RealizedLengths` / :class:`RealizedGeometryReader` -- lazy
  zero-copy mmap readers for one arm's length / geometry pair.
* :func:`realized_lengths_present` / :func:`require_realized_lengths`
  (and the ``*_geometry`` twins) -- existence / pre-flight helpers so
  consumers fail with a clear "run the generator first" message.
* :data:`MATCHED_ARM` / :data:`UNMATCHED_ARM` / :data:`ARMS` (and the
  ``GEOMETRY_*`` twins) -- the typed arm selectors threaded through the
  generator + reader.
"""

from __future__ import annotations

from ._format import ARMS, MATCHED_ARM, UNMATCHED_ARM, RealizedLengthsArm
from ._generate import generate_realized_lengths
from ._geometry_format import (
    GEOMETRY_ARMS,
    GEOMETRY_MATCHED_ARM,
    GEOMETRY_UNMATCHED_ARM,
    RealizedGeometryArm,
)
from ._geometry_generate import generate_realized_geometry
from ._geometry_reader import (
    RealizedGeometryReader,
    realized_geometry_present,
    require_realized_geometry,
)
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
    "generate_realized_geometry",
    "RealizedGeometryReader",
    "realized_geometry_present",
    "require_realized_geometry",
    "RealizedGeometryArm",
    "GEOMETRY_MATCHED_ARM",
    "GEOMETRY_UNMATCHED_ARM",
    "GEOMETRY_ARMS",
]
