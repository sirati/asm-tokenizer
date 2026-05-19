"""Unmatched-arm loader.

Single concern: assemble the unmatched ``SectionArm`` from the v1
``<binary>_unmatched_index.bin`` (per-record data-bin locator, one
entry per function -- unmatched functions have a single record) and
the post-restructuring 5-cell unmatched sections CSV whose first cell
is a base64 line number into the function-names sidecar.

Per-record ``is_overlong`` is derived from the v1 sentinel marker
(``stored_length == SENTINEL_LENGTH``) at load time; downstream
consumers never re-encode the sentinel rule.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.csv_format import parse_function_line_no
from tokenizer.aligned_data.index_format import SENTINEL_LENGTH


def _walk_unmatched_rows(
    paths,
    line_to_name: Dict[int, str],
    open_sections_csv,
) -> Tuple[List[str], np.ndarray]:
    """Walk line-by-line, recording per-row CSV byte offsets (content-
    relative). Manual ``readline()`` (not ``csv.reader``) so ``tell()``
    stays accurate -- ``csv.reader`` buffers ahead. Each unmatched row
    is single-line by format. The first cell is a base64 line number
    into the function-names sidecar; resolved via ``line_to_name`` so
    callers see names. Hard cutover: a legacy 6-cell row raises with a
    migration-pointing message.
    """
    func_names: List[str] = []
    section_offsets: List[int] = []
    if not paths.sections_csv.exists():
        return func_names, np.zeros(0, dtype=np.int64)
    f, content_offset = open_sections_csv(paths.sections_csv)
    try:
        while True:
            row_start = f.tell() - content_offset
            line = f.readline()
            if not line:
                break
            row = list(csv.reader([line]))[0]
            if not row:
                continue
            if len(row) == 6:
                raise ValueError(
                    f"{paths.sections_csv}: legacy 6-cell unmatched "
                    f"row {row!r}; re-run memmap_builder to regenerate "
                    f"at the current format (5-cell layout with "
                    f"indexer_hex)"
                )
            if len(row) != 5:
                continue
            line_no = parse_function_line_no(row[0])
            if line_no not in line_to_name:
                raise ValueError(
                    f"{paths.sections_csv}: row references line {line_no} "
                    f"which is absent from the function-names sidecar; "
                    f"re-run memmap_builder to regenerate"
                )
            func_names.append(line_to_name[line_no])
            section_offsets.append(row_start)
    finally:
        f.close()
    return func_names, np.array(section_offsets, dtype=np.int64)


def load_unmatched_arm(
    paths,
    line_to_name: Dict[int, str],
):
    """Build the unmatched ``SectionArm`` from v1 index + 5-cell CSV.

    Empty (no unmatched functions) -> the orchestrator's canonical
    ``_empty_arm()``.
    """
    # Local imports break the import cycle between the orchestrator
    # (``metadata_loader``) and this module.
    from .metadata_loader import (
        SectionArm,
        _empty_arm,
        build_length_lookup_tables,
        load_index_once,
        load_unmatched_lengths,
        open_sections_csv,
    )

    if not paths.index_bin.exists():
        return _empty_arm()
    starts, lengths, avg_lengths = load_index_once(paths.index_bin)
    if starts is None or avg_lengths is None or lengths is None:
        return _empty_arm()

    is_overlong = (lengths == SENTINEL_LENGTH)
    real_lengths = load_unmatched_lengths(paths, starts, lengths)
    if len(real_lengths) > 0:
        edge_indices, count_per_length = build_length_lookup_tables(
            real_lengths, scale_factor=1
        )
    else:
        edge_indices = np.zeros(1, dtype=np.int32)
        count_per_length = np.zeros(1, dtype=np.int32)

    func_names, section_starts = _walk_unmatched_rows(
        paths, line_to_name, open_sections_csv
    )
    return SectionArm(
        starts=starts,
        lengths=lengths,
        edge_indices=edge_indices,
        count_per_length=count_per_length,
        func_names=func_names,
        section_starts=section_starts,
        csv_starts=section_starts,
        csv_lengths=np.zeros(0, dtype=np.uint32),
        avg_lengths=avg_lengths,
        is_overlong=is_overlong,
    )
