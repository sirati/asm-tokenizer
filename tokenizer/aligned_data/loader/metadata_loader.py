"""Per-binary matched / unmatched metadata loading.

Matched-arm specifics (BIN-section locator + ``sections.bin`` walk)
live in :mod:`_matched_arm_loader`; unmatched-arm specifics (per-
version data-bin locator + ``sections.bin`` walk + sidecar line-no
resolution) live in :mod:`_unmatched_arm_loader`. Both plug into a
closed ``SectionKind`` enum dispatch -- a third arm adds an
``_ArmSpec`` entry, never an ``elif`` cascade.

Matched arm: ``starts`` holds per-VARIANT ``_data.bin`` record
positions decoded from each section's variant blocks
(``data_offset_shifted << 4``); per-function BIN section positions
live in ``bin_starts`` / ``bin_lengths``, loaded via
:func:`csv_section_index.read_csv_section_index_arrays`. Records in
``_data.bin`` are self-describing -- header carries the geometry --
so the arm no longer shadows per-record lengths or overlong flags.
Function name resolution is via the ``<binary>_function_names.txt``
sidecar (loaded by ``BinaryDataset``) keyed on each BIN section's
``function_name_ptr`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO, Tuple

import numpy as np

from tokenizer.aligned_data.binary_format import record_token_count_from_memmap
from tokenizer.aligned_data.index_format import read_index_arrays
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

from ._matched_arm_loader import load_matched_arm

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

    Cardinality (post self-describing record header):

    * Per-RECORD (one entry per ``_data.bin`` record): ``starts``.
      Matched: one entry per VARIANT (flattened across functions).
      Unmatched: one per function. Records are self-describing in
      ``_data.bin`` -- the record header carries ``insn_len``,
      ``block_word_count`` and ``token_count`` -- so the arm no longer
      shadows per-record lengths or overlong flags.
    * Per-FUNCTION: ``func_names``, ``bin_starts``, ``bin_lengths``,
      ``edge_indices``, ``count_per_length``, ``section_starts``,
      ``count``.

    ``bin_starts`` / ``bin_lengths``: per-function locator into
    ``<binary>_sections.bin``. Matched: from
    ``<binary>_matched_index.bin`` (``csv_section_index``); unmatched:
    derived by walking the BIN past the matched region
    (``bin_lengths`` empty -- the unmatched walker only records starts,
    section width is recoverable by re-parsing). ``section_starts``
    aliases ``bin_starts``.

    ``edge_indices`` / ``count_per_length`` drive O(1) length-band
    sampling; both arms compute them from real token counts
    (unmatched: ``load_unmatched_lengths`` over the data records;
    matched: per-function aggregation owned by the matched loader).
    """

    starts: np.ndarray
    edge_indices: np.ndarray
    count_per_length: np.ndarray
    func_names: List[str] = field(default_factory=list)
    section_starts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    bin_starts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    bin_lengths: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.uint32)
    )

    @property
    def count(self) -> int:
        """Per-function count (driver for length-band sampling +
        session indexing).
        """
        return int(len(self.func_names))

    @property
    def record_count(self) -> int:
        """Per-record count (driver for validator pad/bounds checks
        + per-variant iteration)."""
        return int(len(self.starts))


@dataclass(frozen=True)
class BinaryArmPaths:
    """Per-binary file paths for one arm.

    ``sections_bin`` is the shared per-binary section catalog
    (``<binary>_sections.bin``); both arms' :class:`BinaryArmPaths`
    point at the same file because the bin holds matched + unmatched
    sections back-to-back. The matched arm reads it via the per-
    function locator in ``index_bin``; the unmatched arm walks it past
    the matched region.

    ``sections_csv`` is the debug-only text catalog kept for human
    inspection; the loader's hot path no longer reads it post-Phase 4
    cutover. The validator still uses it for CSV cross-checks.
    """

    sections_csv: Path
    sections_bin: Path
    index_bin: Path
    data_bin: Path


@dataclass(frozen=True)
class _ArmSpec:
    """Per-arm dispatch: one ``loader`` callable from per-arm paths +
    sidecar ``line_to_name`` + the matched-index locator (so the
    unmatched arm's walker can find the matched-region end) to a fully
    populated ``SectionArm``.
    """

    kind: SectionKind
    loader: Callable[..., "SectionArm"]


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


def load_index_once(index_path: Path) -> Optional[np.ndarray]:
    """Load a v1 ``*_index.bin`` into a single int64 ndarray of offsets.

    Missing file (or zero-entry file) -> ``None``. Delegates to
    :func:`read_index_arrays` which owns the wire format (16-byte
    prelude, alignment shift). The returned array contains the real
    ``_data.bin`` byte offsets (``stored << alignment_shift``); records
    are self-describing in ``_data.bin`` so the index entry no longer
    carries a length or sentinel field.
    """
    return read_index_arrays(index_path)


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
    paths: BinaryArmPaths, starts: np.ndarray
) -> np.ndarray:
    """Real token count per unmatched function.

    Records are self-describing in ``_data.bin``: the header at each
    record start carries the token count directly, so no companion
    ``lengths`` array is needed (the index entry is just an offset).
    Delegates per-record decoding to
    :func:`record_token_count_from_memmap`, which owns the header
    parse + width-tag dispatch in one place.
    """
    if not paths.data_bin.exists() or len(starts) == 0:
        return np.array([], dtype=np.int32)

    data_memmap = np.memmap(str(paths.data_bin), dtype=np.uint8, mode="r")
    token_counts = [
        record_token_count_from_memmap(data_memmap, int(starts[i]))
        for i in range(len(starts))
    ]
    del data_memmap
    return np.array(token_counts, dtype=np.int32)


# --- Per-arm loader dispatch ----------------------------------------------
# Each arm is fully owned by its own module. The dispatch is one entry
# per arm; a third arm is one entry, not an ``elif`` cascade. The
# unmatched arm needs to know where the matched region of
# ``<binary>_sections.bin`` ends -- the orchestrator threads the
# matched-arm index path through the loader so the unmatched walker
# can start at the correct offset.


def _load_matched(
    paths: BinaryArmPaths,
    line_to_name: Dict[int, str],
    *,
    matched_index: Path,
) -> SectionArm:
    # ``matched_index`` is always ``paths.index_bin`` on the matched
    # arm; accepting it via kw keeps the dispatch signature uniform.
    return load_matched_arm(paths.sections_bin, paths.index_bin, line_to_name)


def _load_unmatched(
    paths: BinaryArmPaths,
    line_to_name: Dict[int, str],
    *,
    matched_index: Path,
) -> SectionArm:
    from ._unmatched_arm_loader import load_unmatched_arm
    return load_unmatched_arm(paths, line_to_name, matched_index=matched_index)


_ARM_SPECS: dict[SectionKind, _ArmSpec] = {
    SectionKind.MATCHED: _ArmSpec(SectionKind.MATCHED, _load_matched),
    SectionKind.UNMATCHED: _ArmSpec(SectionKind.UNMATCHED, _load_unmatched),
}


def _empty_arm() -> SectionArm:
    """Canonical empty arm; every array dtype is preserved so downstream
    length / indexing arithmetic does not degrade.
    """
    return SectionArm(
        starts=np.array([], dtype=np.int64),
        edge_indices=np.zeros(1, dtype=np.int32),
        count_per_length=np.zeros(1, dtype=np.int32),
        func_names=[],
        section_starts=np.zeros(0, dtype=np.int64),
        bin_starts=np.zeros(0, dtype=np.int64),
        bin_lengths=np.zeros(0, dtype=np.uint32),
    )


def load_section_arm(
    kind: SectionKind,
    paths: BinaryArmPaths,
    line_to_name: Optional[Dict[int, str]] = None,
    *,
    matched_index: Optional[Path] = None,
) -> SectionArm:
    """Build one ``SectionArm`` for the requested kind via the per-arm
    loader. ``line_to_name`` resolves each BIN section's
    ``function_name_ptr`` to a function name; required whenever the
    arm actually reads sections (i.e. both matched and unmatched when
    their index file exists). Pass an empty dict only for the "no
    functions at all" path.

    ``matched_index`` is the matched-arm's locator file (which the
    unmatched arm consults to find the matched-region end in the
    shared ``sections.bin``). Defaults to ``paths.index_bin`` for the
    matched arm; on the unmatched arm the caller MUST supply the
    SIBLING matched-arm path explicitly (the unmatched arm's
    ``paths.index_bin`` is the per-record data-bin locator, not a
    section locator).
    """
    spec = _ARM_SPECS[kind]
    if matched_index is None:
        if kind is SectionKind.MATCHED:
            matched_index = paths.index_bin
        else:
            raise ValueError(
                "load_section_arm(UNMATCHED, ...) requires the "
                "``matched_index`` kwarg pointing at the matched-arm "
                "locator file; the unmatched arm's paths.index_bin "
                "is the per-record data-bin locator, not a section "
                "locator. Pass the matched arm's ``index_bin`` here."
            )
    return spec.loader(paths, line_to_name or {}, matched_index=matched_index)


def load_metadata(
    matched_paths: BinaryArmPaths,
    unmatched_paths: BinaryArmPaths,
    line_to_name: Optional[Dict[int, str]] = None,
) -> Tuple[SectionArm, SectionArm]:
    """Compose both arms for one binary; pure function on paths +
    sidecar lookup. The matched-arm index path is threaded into the
    unmatched-arm loader so the unmatched walker can find the matched-
    region end in the shared ``sections.bin``.
    """
    return (
        load_section_arm(
            SectionKind.MATCHED, matched_paths, line_to_name,
            matched_index=matched_paths.index_bin,
        ),
        load_section_arm(
            SectionKind.UNMATCHED, unmatched_paths, line_to_name,
            matched_index=matched_paths.index_bin,
        ),
    )
