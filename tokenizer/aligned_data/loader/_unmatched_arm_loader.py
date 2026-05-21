"""Unmatched-arm loader (BIN catalog).

Single concern: assemble the unmatched ``SectionArm`` from
``<binary>_unmatched_index.bin`` (per-record data-bin locator, one
entry per unmatched ``_unmatched_data.bin`` record) and
``<binary>_sections.bin`` (the BIN catalog; unmatched sections are
emitted by ``write_unmatched_sections_pass2`` immediately after the
matched arm's sections in encounter order).

Records are self-describing in ``_unmatched_data.bin`` (the record
header carries ``token_count``), so the per-record index entry is a
bare offset and there is no length / sentinel / overlong shadow.

How the unmatched arm finds its sections in the shared BIN: the
builder always emits matched sections first, then unmatched. The
matched-arm locator (``matched_index.bin``) lists every matched
section's offset + length. The byte just past the last matched
section is the start of the unmatched region; from there the walker
streams sections via :func:`parse_section_bin` until EOF. An empty
matched arm starts the walk at the file-level prelude end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.matched_sections_bin import parse_section_bin

from ._sections_bin_walk import (
    read_sections_bin_blob,
    resolve_func_name_or_raise,
    unmatched_region_start,
)


def _walk_unmatched_sections(
    sections_bin: Path,
    region_start: int,
    line_to_name: Dict[int, str],
) -> Tuple[List[str], np.ndarray]:
    """Parse every section in ``[region_start, EOF)`` of ``sections_bin``.

    Returns ``(func_names, section_starts)`` where ``section_starts[i]``
    is the BIN offset of the i-th unmatched section and
    ``func_names[i]`` is its resolved function name. The walker honours
    encounter order so the i-th entry of both arrays describes the
    same section.
    """
    func_names: List[str] = []
    section_starts: List[int] = []
    if not sections_bin.exists():
        return func_names, np.zeros(0, dtype=np.int64)
    raw, blob = read_sections_bin_blob(sections_bin)
    end = len(raw)
    cursor = region_start
    while cursor < end:
        section, next_cursor = parse_section_bin(blob, cursor)
        func_names.append(
            resolve_func_name_or_raise(
                section.function_name_ptr, line_to_name,
                sections_bin, cursor,
            )
        )
        section_starts.append(cursor)
        cursor = next_cursor
    return func_names, np.array(section_starts, dtype=np.int64)


def load_unmatched_arm(
    paths,
    line_to_name: Dict[int, str],
    *,
    matched_index: Path,
):
    """Build the unmatched ``SectionArm`` from BIN walk + v1 data index.

    Empty (no unmatched functions) -> the orchestrator's canonical
    ``_empty_arm()``. ``paths.index_bin`` is the v1
    ``<binary>_unmatched_index.bin`` (per-record data-bin offsets);
    ``paths.sections_bin`` is the shared section catalog;
    ``matched_index`` locates the matched region so the unmatched
    walker knows where to start streaming sections.
    """
    # Local imports break the import cycle between the orchestrator
    # (``metadata_loader``) and this module.
    from .metadata_loader import (
        SectionArm,
        _empty_arm,
        build_length_lookup_tables,
        load_index_once,
        load_unmatched_lengths,
    )

    if not paths.index_bin.exists():
        return _empty_arm()
    starts = load_index_once(paths.index_bin)
    if starts is None:
        return _empty_arm()

    token_counts = load_unmatched_lengths(paths, starts)
    edge_indices, count_per_length = build_length_lookup_tables(
        token_counts, scale_factor=1
    )

    region_start = unmatched_region_start(matched_index)
    func_names, section_starts = _walk_unmatched_sections(
        paths.sections_bin, region_start, line_to_name
    )
    return SectionArm(
        starts=starts,
        edge_indices=edge_indices,
        count_per_length=count_per_length,
        func_names=func_names,
        section_starts=section_starts,
    )
