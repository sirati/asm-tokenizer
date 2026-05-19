"""Per-binary matched / unmatched metadata loading.

The two arms share schema; the only deltas -- length source feeding the
lookup tables and sections-CSV walker -- are isolated to ``_ArmSpec``
callables dispatched by the closed ``SectionKind`` enum. A third arm
adds an ``_ArmSpec`` entry, never an ``elif`` cascade.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, TextIO, Tuple

import numpy as np

from tokenizer.aligned_data.index_format import read_index_arrays
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

from ._index_decoding import record_token_count

# First line of every v1 sections / variants CSV. Comment-line marker so
# third-party CSV viewers ignore it; this reader requires it verbatim.
_SECTIONS_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


class SectionKind(Enum):
    """Matched (multi-version) vs unmatched (6-col single-row) arm."""

    MATCHED = "matched"
    UNMATCHED = "unmatched"


@dataclass(frozen=True)
class SectionArm:
    """Per-arm metadata mirroring legacy ``self.{matched,unmatched}_*``.

    ``section_starts``: per-row CSV byte offsets (content-offset-relative,
    for ``f.seek()`` of a handle from ``open_sections_csv``). Empty for
    matched (matched ``starts`` ARE CSV offsets); populated for unmatched
    to let callers O(1)-seek instead of linear-iterating.
    """

    starts: np.ndarray
    lengths: np.ndarray
    edge_indices: np.ndarray
    count_per_length: np.ndarray
    func_names: List[str] = field(default_factory=list)
    section_starts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    @property
    def count(self) -> int:
        return int(len(self.starts))


@dataclass(frozen=True)
class BinaryArmPaths:
    """Per-binary file paths for one arm."""

    sections_csv: Path
    index_bin: Path
    data_bin: Path


@dataclass(frozen=True)
class _ArmSpec:
    """Per-arm dispatch table: scale, length source, sections walker."""

    kind: SectionKind
    scale_factor: int
    length_source: Callable[
        [BinaryArmPaths, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    walk_sections: Callable[
        [BinaryArmPaths, np.ndarray], Tuple[List[str], np.ndarray]
    ]


# --- Module-level helpers (formerly ``BinaryDataset._*`` methods) ----------
# Free functions so they unit-test without a full dataset and so the
# hot path calls them without a method-resolution hop.


def open_sections_csv(path: Path) -> Tuple[TextIO, int]:
    """Open a v1 sections CSV; require the ``# format=N`` prelude line.

    Returns ``(handle, content_offset_bytes)``: seek-based callers add
    ``content_offset`` to stored offsets, iterating callers just read.
    Raises :class:`ValueError` with a migration-pointing message on any
    other first line. CRLF in the prelude is tolerated; the writer
    always emits LF.
    """
    f = open(path, "r", newline="", encoding="ascii")
    first_line = f.readline()
    expected = _SECTIONS_PRELUDE_LINE.rstrip("\n")
    if first_line.rstrip("\r\n") != expected:
        f.close()
        raise ValueError(
            f"{path}: missing or unsupported prelude; expected first line "
            f"{expected!r}, got {first_line!r}; re-run memmap_builder on the "
            f"per-binary CSVs to regenerate"
        )
    return f, f.tell()


def load_index_once(
    index_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a v1 ``*_index.bin`` into ``(starts, lengths, avg_lengths)``.

    Missing file -> ``(None, None, None)``. Delegates to
    :func:`read_index_arrays` which owns the wire format (16-byte
    prelude, alignment shift, sentinel marker). ``starts`` are real
    byte offsets (``int64``); ``lengths`` are real record byte lengths
    (``uint32``) with ``0`` flagging an overlong record.
    """
    arrays = read_index_arrays(index_path)
    if arrays is None:
        return None, None, None
    return arrays


def build_length_lookup_tables(
    avg_lengths: np.ndarray, scale_factor: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """``edge_indices[L]``: first idx whose length>=L. ``count_per_length[L]``:
    functions at exactly length L. Used for O(1) length-band sampling."""
    if len(avg_lengths) == 0:
        return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)
    actual_lengths = avg_lengths.astype(np.int32) * scale_factor
    max_length = int(actual_lengths.max())
    edge_indices = np.zeros(max_length + 2, dtype=np.int32)
    count_per_length = np.zeros(max_length + 1, dtype=np.int32)
    for length in actual_lengths:
        count_per_length[length] += 1
    current_idx = 0
    for length in range(max_length + 2):
        while current_idx < len(actual_lengths) and actual_lengths[current_idx] < length:
            current_idx += 1
        edge_indices[length] = current_idx
    return edge_indices, count_per_length


def load_unmatched_lengths(
    paths: BinaryArmPaths, starts: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Real token count per unmatched function.

    Delegates per-record decoding to :func:`record_token_count`, which
    owns the pad + overlong-field accounting in one place. ``lengths``
    carries the index reader's sentinel-marker convention (``0`` flags
    an overlong record).
    """
    if not paths.data_bin.exists() or len(starts) == 0:
        return np.array([], dtype=np.int32)

    data_memmap = np.memmap(str(paths.data_bin), dtype=np.uint8, mode="r")
    token_counts = [
        record_token_count(data_memmap, int(starts[i]), int(lengths[i]))
        for i in range(len(starts))
    ]
    del data_memmap
    return np.array(token_counts, dtype=np.int32)


# --- Per-arm walkers + length sources --------------------------------------
# Only the matched-vs-unmatched behavioural delta lives here; each pair
# plugs into ``_ArmSpec``.


def _matched_length_source(paths, starts, lengths, avg_lengths):
    """Matched: index's ``avg_lengths`` is authoritative (scale 16)."""
    return avg_lengths

def _unmatched_length_source(paths, starts, lengths, avg_lengths):
    """Unmatched: real token counts need per-record data-bin header reads."""
    return load_unmatched_lengths(paths, starts, lengths)


def _matched_walk_sections(
    paths: BinaryArmPaths, starts: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Seek to each stored CSV offset, read first line as the function
    name. ``section_starts`` empty -- matched ``starts`` ARE CSV offsets.
    """
    func_names: List[str] = []
    if not paths.sections_csv.exists() or len(starts) == 0:
        return func_names, np.zeros(0, dtype=np.int64)
    f, content_offset = open_sections_csv(paths.sections_csv)
    try:
        for i in range(len(starts)):
            f.seek(int(starts[i]) + content_offset)
            row = list(csv.reader([f.readline()]))[0]
            if row and len(row) >= 1:
                func_names.append(row[0])
    finally:
        f.close()
    return func_names, np.zeros(0, dtype=np.int64)


def _unmatched_walk_sections(
    paths: BinaryArmPaths, starts: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Walk line-by-line; record per-row CSV byte offsets (content-relative)
    so callers can O(1)-seek instead of linear-iterating. Uses manual
    ``readline()`` (not ``csv.reader``) for accurate ``tell()`` -- the
    reader buffers ahead. Each unmatched row is single-line by format.
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
            if row and len(row) == 6:
                func_names.append(row[0])
                section_offsets.append(row_start)
    finally:
        f.close()
    return func_names, np.array(section_offsets, dtype=np.int64)


_ARM_SPECS: dict[SectionKind, _ArmSpec] = {
    SectionKind.MATCHED: _ArmSpec(
        kind=SectionKind.MATCHED,
        scale_factor=16,
        length_source=_matched_length_source,
        walk_sections=_matched_walk_sections,
    ),
    SectionKind.UNMATCHED: _ArmSpec(
        kind=SectionKind.UNMATCHED,
        scale_factor=1,
        length_source=_unmatched_length_source,
        walk_sections=_unmatched_walk_sections,
    ),
}


def _empty_arm() -> SectionArm:
    """Canonical empty arm; dtypes match ``load_index_once`` output."""
    return SectionArm(
        starts=np.array([], dtype=np.int64),
        lengths=np.array([], dtype=np.uint32),
        edge_indices=np.zeros(1, dtype=np.int32),
        count_per_length=np.zeros(1, dtype=np.int32),
        func_names=[],
        section_starts=np.zeros(0, dtype=np.int64),
    )


def load_section_arm(kind: SectionKind, paths: BinaryArmPaths) -> SectionArm:
    """Build one ``SectionArm`` for the requested kind. Single
    implementation: the kind picks an ``_ArmSpec`` (length-source +
    walker); the scaffold around it is common across both arms.
    """
    spec = _ARM_SPECS[kind]
    if not paths.index_bin.exists():
        return _empty_arm()
    starts, lengths, avg_lengths = load_index_once(paths.index_bin)
    if starts is None or avg_lengths is None or lengths is None:
        return _empty_arm()

    length_array = spec.length_source(paths, starts, lengths, avg_lengths)
    if len(length_array) > 0:
        edge_indices, count_per_length = build_length_lookup_tables(
            length_array, scale_factor=spec.scale_factor
        )
    else:
        edge_indices = np.zeros(1, dtype=np.int32)
        count_per_length = np.zeros(1, dtype=np.int32)

    func_names, section_starts = spec.walk_sections(paths, starts)
    return SectionArm(
        starts=starts,
        lengths=lengths,
        edge_indices=edge_indices,
        count_per_length=count_per_length,
        func_names=func_names,
        section_starts=section_starts,
    )


def load_metadata(
    matched_paths: BinaryArmPaths, unmatched_paths: BinaryArmPaths
) -> Tuple[SectionArm, SectionArm]:
    """Compose both arms for one binary; pure function on paths."""
    return (
        load_section_arm(SectionKind.MATCHED, matched_paths),
        load_section_arm(SectionKind.UNMATCHED, unmatched_paths),
    )
