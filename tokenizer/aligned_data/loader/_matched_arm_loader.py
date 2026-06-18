"""Matched-arm loader (BIN catalog).

Single concern: assemble the matched ``SectionArm`` from
``<binary>_matched_index.bin`` (function-to-BIN-section locator) and
``<binary>_sections.bin`` (the BIN catalog parsed by
:mod:`tokenizer.aligned_data.matched_sections_bin`).

Layout split that makes this module possible:

* ``<binary>_matched_index.bin`` -- packed u40/u24 layout, no prelude.
  Each entry locates ONE function's section in ``sections.bin`` as
  ``(bin_offset, bin_section_length)`` (both 4-byte-aligned). Decoded
  via :func:`tokenizer.aligned_data.csv_section_index.read_csv_section_index_arrays`.
* ``<binary>_sections.bin`` -- 16-byte ``MSEC`` prelude + a stream of
  4-byte-aligned section records. Each section header carries the
  function's line number (FID), the call_target table, and N
  variant blocks. The reader-side codec is
  :func:`tokenizer.aligned_data.matched_sections_bin.parse_section_bin`.

The arm's per-function arrays come from walking each matched section's
BIN payload:

* ``func_names`` -- resolved from ``section.function_name_ptr`` via
  ``line_to_name``.
* ``starts`` -- flat per-VARIANT array of real ``_data.bin`` offsets,
  recovered from each variant block's ``data_offset_shifted << 4``.
* ``bin_starts`` / ``bin_lengths`` -- per-function locator into
  ``sections.bin`` (same arrays the matched_index.bin codec returns).

``select_random_function_by_length`` is a NotImplementedError stub for
the matched arm, so the length-band lookup tables collapse to empty
placeholders -- there is no per-function avg-length signal to feed
them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    ColumnarSections,
    parse_sections_columnar,
)

from ._sections_bin_walk import (
    read_sections_bin_blob,
    resolve_func_name_or_raise,
)


def _columnar_matched_sections(
    sections_bin: Path,
    bin_starts: np.ndarray,
    line_to_name: Dict[int, str],
) -> Tuple[List[str], ColumnarSections]:
    """Decode every matched section at ``bin_starts`` into columnar arrays.

    Returns ``(func_names, cols)`` where ``func_names[i]`` is the
    resolved function name for ``bin_starts[i]`` and ``cols`` is the
    flat columnar view of those sections (parallel, in encounter
    order). The matched arm needs only the section-level
    ``function_name_ptr`` (-> names) and the per-variant
    ``var_data_offset_shifted`` (-> ``starts``); the columnar decoder is
    the single vectorized source of truth for the ``sections.bin`` wire
    format, so this reuses it rather than walking 340k full
    :class:`Section` objects (each carrying its whole call_target +
    variant-block table) only to discard them.
    """
    func_names: List[str] = []
    if not sections_bin.exists() or len(bin_starts) == 0:
        return func_names, parse_sections_columnar(
            np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.int64)
        )
    raw, _blob = read_sections_bin_blob(sections_bin)
    cols = parse_sections_columnar(np.asarray(raw), bin_starts)
    fids = cols.function_name_ptr
    func_names = [
        resolve_func_name_or_raise(
            int(fids[i]), line_to_name, sections_bin, int(bin_starts[i])
        )
        for i in range(len(bin_starts))
    ]
    return func_names, cols


def _flat_variant_starts(cols: ColumnarSections) -> np.ndarray:
    """Recover the flat per-variant real ``_data.bin`` offsets.

    Each variant block's ``var_data_offset_shifted`` is the ``>> 4`` of
    the real ``_data.bin`` offset (16-byte record alignment). The
    columnar ``var_data_offset_shifted`` column is already in section-
    major, variant-minor order (CSR via ``var_offsets``) -- the same
    flatten order the per-section walk produced -- so recovering the
    real offsets is a single ``<< 4`` over that column. Keeping the
    real (post-shift) offsets here holds the arm's ``starts`` semantics
    in lockstep with ``unmatched_index.bin``-derived offsets.
    """
    return cols.var_data_offset_shifted.astype(np.int64) << 4


def load_matched_arm(
    sections_bin: Path,
    matched_index: Path,
    line_to_name: Dict[int, str],
    *,
    data_bin: Path,
):
    """Build the matched ``SectionArm`` from ``matched_index.bin`` + BIN catalog.

    Empty (no matched functions) -> the orchestrator's canonical
    ``_empty_arm()``. The matched_index is the function-to-section
    locator into ``sections.bin``; its entries are 4-byte aligned (the
    :class:`SectionWriter` pads each section trailer up to the next
    4-byte boundary). Per-variant data-bin positions are 16-byte
    aligned and recovered from each variant block's
    ``data_offset_shifted`` field.

    ``data_bin`` feeds the load-time per-arm sweep that asserts each
    record's on-wire ``entry_idx`` equals its flat-starts index; the
    sweep is a single chokepoint shared with the unmatched arm.
    """
    # Local import to break the import cycle between this module and
    # the orchestrator (``metadata_loader`` imports ``load_matched_arm``).
    from .metadata_loader import SectionArm, _empty_arm

    if not matched_index.exists():
        return _empty_arm()

    section_index = read_csv_section_index_arrays(matched_index)
    if section_index is None:
        return _empty_arm()
    bin_starts, bin_lengths = section_index

    func_names, cols = _columnar_matched_sections(
        sections_bin, bin_starts, line_to_name
    )

    # Flatten variants into per-record offsets. ``func_names`` stays
    # per-function (one entry per matched section); ``starts`` is
    # per-variant (one entry per variant block in encounter order).
    # No length or overlong flag -- the record at each offset is
    # self-describing.
    starts = _flat_variant_starts(cols)

    # ``select_random_function_by_length`` is a NotImplementedError
    # stub for the matched arm, so the length-band lookup tables have
    # no consumer; ship the canonical empty placeholders the
    # ``SectionArm`` dataclass expects.
    edge_indices = np.zeros(1, dtype=np.int32)
    count_per_length = np.zeros(1, dtype=np.int32)

    return SectionArm(
        starts=starts,
        edge_indices=edge_indices,
        count_per_length=count_per_length,
        func_names=func_names,
        section_starts=bin_starts,
        bin_starts=bin_starts,
        bin_lengths=bin_lengths,
    )
