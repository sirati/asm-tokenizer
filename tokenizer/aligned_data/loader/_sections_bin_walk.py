"""Shared ``sections.bin`` blob primitives.

Single concern: own the open + prelude-assert + memoryview-of-data-region
plumbing both arm loaders and :class:`BinarySession` previously inlined
three times, plus the per-FID name resolution wording the two arm
loaders previously duplicated verbatim.

Boundary contract:

* :func:`read_sections_bin_blob` — read the whole file, validate its
  16-byte ``MSEC`` prelude, return ``(blob_bytes, memoryview)``. The
  ``bytes`` is pinned so the caller can keep the memoryview alive
  (the session does); callers that walk once and drop both let the
  ``bytes`` GC alongside the view.
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
emits the BIN once and the matched-arm loader reads it once; the
unmatched-arm loader reads it once; the session reads it once and
pins the memoryview for the session lifetime. Cross-caller caching
would require a second concern (cache invalidation) this module
refuses to own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.memmap_format import (
    MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    assert_matched_sections_prelude,
)


def read_sections_bin_blob(path: Path) -> Tuple[bytes, memoryview]:
    """Read + prelude-validate ``path``; return the raw bytes + a memoryview.

    The returned ``memoryview`` covers the whole file (NOT just the
    data region) so callers can pass absolute file offsets straight
    through to :func:`tokenizer.aligned_data.matched_sections_bin.parse_section_bin`.
    Returning the ``bytes`` alongside lets the caller pin the buffer
    for as long as the view needs to stay live (e.g. for the lifetime
    of a :class:`BinarySession`); short-lived callers discard both
    together.
    """
    raw = path.read_bytes()
    assert_matched_sections_prelude(raw, path=str(path))
    return raw, memoryview(raw)


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
    """
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
