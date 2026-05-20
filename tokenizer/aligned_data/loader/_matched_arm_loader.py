"""Matched-arm loader.

Single concern: assemble the matched ``SectionArm`` from the pre-v1
``matched_index.bin`` (function-to-CSV-section locator) and the v1
sections CSV whose variant rows carry inline indexer hex (per-variant
data-bin offsets).

Layout split that makes this module possible:

* ``<binary>_index.bin`` (matched arm) -- pre-v1 8-byte layout, no
  prelude. Each entry locates ONE function's section in the text CSV
  as ``(csv_offset, csv_section_length)`` (u40 + u24, both
  4-byte-aligned). Decoded via
  :func:`tokenizer.aligned_data.csv_section_index.read_csv_section_index_arrays`.
* ``<binary>_sections.csv`` -- function sections separated by blank
  rows. Header row first cell is the base64 line number into
  ``<binary>_function_names.txt``; subsequent rows (until blank) are
  3-cell variant rows ``[variant_ref, inlining_str, indexer_hex]``
  whose ``indexer_hex`` (8 hex chars = one u32 ``offset >> 4``)
  decodes to a single ``data_offset`` into ``<binary>_data.bin``. The
  record at that offset is self-describing -- its header carries every
  geometry field a reader needs -- so no length or overlong flag rides
  alongside the offset.

The previous matched-arm shape carried CSV byte positions in
``starts`` / ``lengths``. Post-restructuring those arrays hold
**data-bin positions per VARIANT** (one entry per
``write_function_binary_data`` call pass 1 emitted); the per-function
CSV-section locator moves to dedicated ``csv_starts`` / ``csv_lengths``
fields. ``func_names`` is per-function, resolved from the sidecar's
``line_to_name`` dict.

``select_random_function_by_length`` is a NotImplementedError stub for
the matched arm, so the length-band lookup tables collapse to empty
placeholders -- there is no per-function avg-length signal to feed
them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.csv_format import parse_function_line_no
from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.inline_indexer import decode_inline_indexer


def _slice_section_rows(
    sections_csv: Path,
    csv_starts: np.ndarray,
    csv_lengths: np.ndarray,
    line_to_name: Dict[int, str],
) -> Tuple[List[str], List[List[int]]]:
    """Per-function: slice the CSV section bytes, decode the header line
    number to a function name, decode each variant row's ``indexer_hex``.

    Returns ``(func_names, per_function_variant_offsets)`` where the
    inner list is ``[data_offset]`` for every variant of that function
    (one int per ``write_function_binary_data`` call pass 1 emitted).
    The CSV is opened ONCE via :func:`metadata_loader.open_sections_csv`
    so the v1 prelude check fires exactly once.
    """
    from .metadata_loader import open_sections_csv

    func_names: List[str] = []
    per_function: List[List[int]] = []
    if not sections_csv.exists() or len(csv_starts) == 0:
        return func_names, per_function

    f, content_offset = open_sections_csv(sections_csv)
    try:
        for i in range(len(csv_starts)):
            f.seek(int(csv_starts[i]) + content_offset)
            blob = f.read(int(csv_lengths[i]))
            func_name, variant_offsets = _parse_section_blob(
                blob, line_to_name, sections_csv
            )
            func_names.append(func_name)
            per_function.append(variant_offsets)
    finally:
        f.close()
    return func_names, per_function


def _parse_section_blob(
    blob: str,
    line_to_name: Dict[int, str],
    sections_csv: Path,
) -> Tuple[str, List[int]]:
    """Decode one CSV section blob.

    Header row: ``[base64_line_no, base64_called_funcs_csv]`` -> name
    via :func:`parse_function_line_no` + ``line_to_name``.

    Variant rows: ``[variant_ref, inlining_str, indexer_hex]``
    (3 cells). Each ``indexer_hex`` is 8 hex chars and decodes to one
    ``data_offset`` (the record at that offset is self-describing).
    Iteration stops at the first empty row (the trailing blank
    separator that the writer emits) or at EOF. Any 4-cell legacy row
    raises with a migration-pointing message; the legacy 16-hex-char
    inline indexer is caught one layer down by
    :func:`decode_inline_indexer` -- both are hard cutovers.
    """
    reader = csv.reader(blob.splitlines())
    rows = list(reader)
    if not rows:
        raise ValueError(
            f"{sections_csv}: empty section blob; re-run memmap_builder "
            f"to regenerate"
        )
    header = rows[0]
    if not header or not header[0]:
        raise ValueError(
            f"{sections_csv}: section header missing base64 line number; "
            f"re-run memmap_builder to regenerate"
        )
    line_no = parse_function_line_no(header[0])
    if line_no not in line_to_name:
        raise ValueError(
            f"{sections_csv}: header references line {line_no} which is "
            f"absent from the function-names sidecar; re-run memmap_builder "
            f"to regenerate"
        )
    func_name = line_to_name[line_no]

    variant_offsets: List[int] = []
    for row in rows[1:]:
        if not row:
            # Blank row = end-of-section separator.
            break
        if len(row) == 4:
            raise ValueError(
                f"{sections_csv}: legacy 4-cell variant row "
                f"{row!r} detected (expected 3 cells "
                f"[variant_ref, inlining_str, indexer_hex]); re-run "
                f"memmap_builder to regenerate at the current format"
            )
        if len(row) != 3:
            raise ValueError(
                f"{sections_csv}: variant row has {len(row)} cells "
                f"(expected 3); row={row!r}"
            )
        indexer_hex = row[2]
        variant_offsets.append(decode_inline_indexer(indexer_hex))
    return func_name, variant_offsets


def load_matched_arm(
    sections_csv: Path,
    matched_index: Path,
    line_to_name: Dict[int, str],
):
    """Build the matched ``SectionArm`` from the pre-v1 matched_index +
    inline-indexer-bearing sections CSV.

    Empty (no matched functions) -> the orchestrator's canonical
    ``_empty_arm()``. The matched_index is the function-to-CSV-section
    locator; its entries point at TEXT-file byte positions and obey
    the section CSV's 4-byte alignment rule -- they are not subject to
    the ``_data.bin`` 16-byte alignment. The per-variant data-bin
    positions ARE 16-byte-aligned (decoded from the 8-hex-char inline
    indexer in each variant row) and produce the per-record
    ``starts`` array.
    """
    # Local import to break the import cycle between this module and
    # the orchestrator (``metadata_loader`` imports ``load_matched_arm``).
    from .metadata_loader import SectionArm, _empty_arm

    if not matched_index.exists():
        return _empty_arm()

    section_index = read_csv_section_index_arrays(matched_index)
    if section_index is None:
        return _empty_arm()
    csv_starts, csv_lengths = section_index

    func_names, per_function = _slice_section_rows(
        sections_csv, csv_starts, csv_lengths, line_to_name
    )

    # Flatten variants into per-record offsets. ``func_names`` stays
    # per-function (one entry per matched section); ``starts`` is
    # per-variant (one entry per ``write_function_binary_data`` call
    # pass 1 made). No length or overlong flag -- the record at each
    # offset is self-describing.
    flat_starts: List[int] = [
        offset for offsets in per_function for offset in offsets
    ]
    starts = np.array(flat_starts, dtype=np.int64)

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
        section_starts=csv_starts.astype(np.int64),
        csv_starts=csv_starts.astype(np.int64),
        csv_lengths=csv_lengths.astype(np.uint32),
    )
