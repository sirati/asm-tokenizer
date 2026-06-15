"""Shared ``sections.bin`` blob primitives.

Single concern: own the open + prelude-assert + memoryview-of-data-region
plumbing both arm loaders and :class:`BinarySession` previously inlined
three times, plus the per-FID name resolution wording the two arm
loaders previously duplicated verbatim.

Boundary contract:

* :func:`read_sections_bin_blob` — ``np.memmap`` the file (so
  ``parse_section_bin`` pages in only the touched section, not the whole
  catalog), validate its 16-byte ``MSEC`` prelude, return
  ``(memmap, memoryview)``. The ``np.memmap`` is pinned so the caller can
  keep the memoryview alive (the session does); callers that walk once and
  drop both let the mapping release by refcounting once no slice is
  exported.
* :func:`resolve_func_name_or_raise` — turn a parsed-section
  ``function_name_ptr`` (FID) into the resolved function name via the
  ``line_to_name`` sidecar; raise :class:`ValueError` with the
  consistent "re-run memmap_builder to regenerate" message both arms
  previously emitted.
* :func:`unmatched_region_start` — derive the BIN byte offset at which
  the unmatched-region walk starts, given the matched-arm locator
  file. Identical to what the unmatched arm builder and the loader
  test fixture previously inlined.

This module deliberately does NOT cache the blob across callers: pass-2
emits the BIN once and the matched-arm loader mmaps it once; the
unmatched-arm loader mmaps it once; the session mmaps it once and
pins the memoryview for the session lifetime. Cross-caller caching
would require a second concern (cache invalidation) this module
refuses to own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_bin import (
    Section,
    _SECTION_FID_MASK,
    parse_section_bin,
)
from tokenizer.aligned_data.memmap_format import (
    MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    assert_matched_sections_prelude,
)


def read_sections_bin_blob(path: Path) -> Tuple[np.memmap, memoryview]:
    """``mmap`` + prelude-validate ``path``; return the memmap + a memoryview.

    The BIN is NOT slurped into a ``bytes`` object: it is ``np.memmap``-ed
    so :func:`tokenizer.aligned_data.matched_sections_bin.parse_section_bin`
    pages in only the section(s) it actually touches. A per-batch
    :class:`BinarySession` opens a fresh blob per sampled binary; the old
    full read copied the ENTIRE catalog every time (z3's ``_sections.bin``
    is ~348MB), so the eager copy dominated per-batch memory while the much
    larger ``_data.bin`` was already lazy. The mmap drops that copy to the
    touched pages only.

    The returned ``memoryview`` covers the whole file (NOT just the data
    region) so callers can pass absolute file offsets straight through to
    ``parse_section_bin``. Returning the ``np.memmap`` alongside lets the
    caller pin the mapping for as long as the view (and any ``Section``
    parsed from a slice of it) needs to stay live -- e.g. for the lifetime
    of a :class:`BinarySession`; short-lived callers drop both together and
    the mapping is released by refcounting once no slice is exported.
    """
    mm = np.memmap(str(path), dtype=np.uint8, mode="r")
    assert_matched_sections_prelude(
        bytes(mm[:MATCHED_SECTIONS_BIN_PRELUDE_SIZE]), path=str(path)
    )
    return mm, memoryview(mm)


def walk_parsed_sections(
    blob: memoryview, region_start: int
) -> Iterator[Tuple[int, Section]]:
    """Yield ``(start, Section)`` for every section in ``[region_start, EOF)``.

    The pure structural walk: it streams sections via
    :func:`...matched_sections_bin.parse_section_bin` from ``region_start``
    to the end of ``blob`` and yields, in catalog order, each section's
    absolute start offset PAIRED WITH the ``Section`` the walk already
    parsed to find the next boundary. The walk owns the parse; it threads
    the result out rather than discarding it, so a consumer needing the
    section's fields (the unmatched-arm loader reads
    ``function_name_ptr`` + ``variants``) reuses this single parse instead
    of re-parsing every section. Consumers needing only the start offsets
    (the realized-lengths pass feeds them to the columnar parser) ignore
    the second element. ``parse_section_bin`` therefore runs exactly once
    per section per pass.

    The walk owns NO name-resolution concern. ``blob`` is the whole-file
    memoryview from :func:`read_sections_bin_blob` so the offsets are
    absolute. As a generator each section is parsed lazily as the consumer
    pulls it, so a one-shot consumer never holds more than the current
    ``Section`` live.
    """
    end = len(blob)
    cursor = region_start
    while cursor < end:
        section, next_cursor = parse_section_bin(blob, cursor)
        yield cursor, section
        cursor = next_cursor


def resolve_func_name_or_raise(
    fid: int,
    line_to_name: Dict[int, str],
    sections_bin: Path,
    cursor: int,
) -> str:
    """Resolve a section's ``function_name_ptr`` against the sidecar.

    ``cursor`` is the BIN byte offset of the section being resolved;
    it rides into the error message so a sidecar-drift failure points
    the user at the exact section that triggered the mismatch.
    Identical wording is what the matched + unmatched arm walkers
    used to inline -- centralising it here means a future tweak to
    the migration pointer changes one site, not three.

    Bit 31 of a section-header FID is the duplicated-section marker
    (:data:`...matched_sections_bin._SECTION_DUPLICATED_BIT`); the
    function-names sidecar is keyed on the CLEAN low-31-bit line
    number only. ``parse_section_bin`` already strips the bit from
    ``Section.function_name_ptr``, but this resolver is the single
    shared name lookup for every FID consumer, so it masks
    (``& _SECTION_FID_MASK``) on its own input contract too: a raw
    header FID resolves to the same name as its clean form instead of
    a spurious sidecar-drift raise. The mask is a no-op for any clean
    FID (real line numbers are ``< 2**31`` by construction -- bit 31
    is reserved), so the genuine sidecar-drift guard below is
    preserved for every truly-absent (post-mask) FID.
    """
    fid &= _SECTION_FID_MASK
    if fid not in line_to_name:
        raise ValueError(
            f"{sections_bin}: section at offset {cursor} "
            f"references function_name_ptr={fid} which is absent from "
            f"the function-names sidecar; re-run memmap_builder to "
            f"regenerate"
        )
    return line_to_name[fid]


def unmatched_region_start(matched_index: Path) -> int:
    """BIN byte offset at which the unmatched-region walk should begin.

    The builder emits matched sections first in encounter order, then
    unmatched sections, so the last matched section's end (``bin_offset
    + bin_section_length``) is the first unmatched section's start.
    Missing / empty matched index -> the walk begins at the BIN's
    file-level prelude end.
    """
    if not matched_index.exists():
        return MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    pair = read_csv_section_index_arrays(matched_index)
    if pair is None:
        return MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    bin_starts, bin_lengths = pair
    if len(bin_starts) == 0:
        return MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    last_start = int(bin_starts[-1])
    last_length = int(bin_lengths[-1])
    return last_start + last_length
