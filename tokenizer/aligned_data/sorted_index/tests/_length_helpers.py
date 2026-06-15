"""Shared test helpers for the matched-arm body-length sidecar input.

Single concern: give the sorted-index length tests the per-variant body
lengths the build now CONSUMES (instead of the retired ``_data.bin``
recompute) -- both the real production input (the generated realized-
length sidecar) and the self-contained bulk-engine reference the
byte-identity gate compares it against.

The build's length compute takes a ``body_lengths`` array (section-major
in ``cols.var_offsets`` order, EXCLUDING the self/identity token). These
two helpers produce that array two ways so every test drives the real
sidecar path while one gate proves the sidecar is byte-identical to the
bulk reference:

* :func:`sidecar_body_lengths` -- generate the realized-length sidecars
  for the fixture, then read the matched arm's flat ``lengths`` body
  (the exact bytes the builder consumes).
* :func:`reference_body_lengths` -- recompute the same array inline from
  ``bulk_token_spans`` + ``bulk_contributing_body_lengths`` over the
  unique ``var_data_offset_shifted << 4`` record offsets (the retired
  ``_resolve._body_lengths`` math, reconstructed so the gate is
  self-contained and never leans on production code under test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.binary_format._bulk_geometry import (
    bulk_token_spans,
)
from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
    bulk_contributing_body_lengths,
)
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.realized_lengths import (
    MATCHED_ARM,
    RealizedLengths,
    generate_realized_lengths,
)


_DATA_OFFSET_SHIFT = 4


def ensure_sidecar(base: Path, binary_name: str) -> Path:
    """Generate the realized-length sidecars (the Phase-4a precondition).

    The build now HARD-REQUIRES the matched-arm sidecar; fixtures lay
    down only the memmap dir, so tests that drive the builder must run
    this generator first (exactly as the dynrunner phase-4 dependency
    edge does in production). Returns ``base`` for call chaining.
    """
    generate_realized_lengths(base, binary_name)
    return base


def sidecar_body_lengths(base: Path, binary_name: str) -> np.ndarray:
    """The matched-arm body lengths the build consumes, via the sidecar.

    Generates the realized-length sidecars for ``binary_name`` under
    ``base`` (the Phase-4a pass) and returns the matched arm's flat
    ``int64`` ``lengths`` body -- the exact array
    :func:`build_sorted_index_bytes` reads.
    """
    ensure_sidecar(base, binary_name)
    with RealizedLengths.open(base, binary_name, MATCHED_ARM) as rlen:
        return np.asarray(rlen.lengths, dtype=np.int64)


def reference_body_lengths(
    cols: ColumnarSections, data_u8: np.ndarray
) -> np.ndarray:
    """Bulk-engine reference body length per variant (self-contained).

    The retired ``_resolve._body_lengths`` arithmetic, reconstructed
    inline: dedup the per-variant record offsets
    (``var_data_offset_shifted << 4``), measure each unique record's
    contributing body length with the bulk engine, scatter back. Used by
    the byte-identity gate to prove the sidecar matches without importing
    the code under test.
    """
    offsets = cols.var_data_offset_shifted.astype(np.int64) << _DATA_OFFSET_SHIFT
    uniq, inverse = np.unique(offsets, return_inverse=True)
    starts, counts = bulk_token_spans(data_u8, uniq)
    body = bulk_contributing_body_lengths(data_u8, starts, counts)
    return body[inverse].astype(np.int64)
