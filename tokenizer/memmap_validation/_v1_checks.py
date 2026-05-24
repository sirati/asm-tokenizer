"""v1-format invariant checks for the memmap validator.

Single concern: assert the per-binary on-disk invariants the v1 writer
guarantees. Each check is a free function that takes the artefacts it
inspects and returns a ``list[str]`` of human-readable error messages
(empty on success). The validator orchestrates the calls and appends
results into its existing error list -- no shared state, no I/O hidden
behind class boundaries.

Layout / version knowledge crosses module boundaries only via:

  * ``aligned_data.memmap_format.MEMMAP_FORMAT_VERSION``
  * ``aligned_data.index_format.{RECORD_ALIGNMENT_does_not_live_here,
    read_index_prelude}`` (record alignment is exported by
    :mod:`aligned_data.binary_format`, not by ``index_format``)
  * ``aligned_data.binary_format.{RECORD_ALIGNMENT, MAX_HEADER_BYTES,
    parse_binary_header, derive_pad_placement, record_total_size}``
    -- the sole source of truth for record geometry
  * ``aligned_data.loader.metadata_loader.open_sections_csv`` (prelude
    consumption on sections + slim variants CSVs -- the helper is
    content-agnostic so the slim CSV reuses it)

If any of these constants/parsers change, the writer + reader + this
validator all swap together; we never re-derive layout here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from tokenizer.aligned_data.binary_format import (
    BLOCK_WORD_SIZE,
    MAX_HEADER_BYTES,
    RECORD_ALIGNMENT,
    derive_pad_placement,
    parse_binary_header,
    record_total_size,
)
from tokenizer.aligned_data.index_format import read_index_prelude
from tokenizer.aligned_data.loader.metadata_loader import open_sections_csv


def check_csv_prelude(path: Path, label: str) -> List[str]:
    """Open ``path`` via ``open_sections_csv``; surface any prelude error.

    ``open_sections_csv`` raises :class:`ValueError` on missing or
    wrong-version ``# format=N`` first line; we translate that to a
    validator-formatted entry instead of letting the traceback escape.
    A missing file is silently skipped because the path-existence check
    is a separate concern owned by the validator's main loop (an arm
    can legitimately have no sections at all).
    """
    if not path.exists():
        return []
    try:
        handle, _ = open_sections_csv(path)
    except ValueError as exc:
        return [f"{label}: prelude check failed: {exc}"]
    handle.close()
    return []


def check_index_prelude(path: Path, label: str) -> List[str]:
    """Read ``path``'s 16-byte prelude; surface magic / version errors."""
    if not path.exists():
        return []
    try:
        with open(path, "rb") as fh:
            read_index_prelude(fh)
    except ValueError as exc:
        return [f"{label}: prelude check failed: {exc}"]
    return []


def check_starts_alignment(starts: np.ndarray, label: str) -> List[str]:
    """Every index entry's resolved ``start`` must be ``RECORD_ALIGNMENT``-aligned.

    Records align to :data:`RECORD_ALIGNMENT` (= 16) bytes; the writer
    guarantees this by construction (offsets written as
    ``start >> ALIGNMENT_SHIFT`` and read back ``stored << shift``).
    A corrupted ``_index.bin`` or a stale prelude alignment_shift could
    silently land on an unaligned offset; the validator catches it.
    """
    if len(starts) == 0:
        return []
    bad = np.where(starts % RECORD_ALIGNMENT != 0)[0]
    return [
        f"{label}: index entry {int(i)} start {int(starts[int(i)])} "
        f"is not {RECORD_ALIGNMENT}-byte aligned"
        for i in bad
    ]


def _header_window(data: np.ndarray, start: int) -> np.ndarray:
    """Return up to :data:`MAX_HEADER_BYTES` bytes starting at ``start``.

    Real header may be shorter (ultrashort is 3 bytes, narrow normal
    tags are 7-9 bytes); :func:`parse_binary_header` reads only the
    bytes it needs from the slice.
    """
    end = min(start + MAX_HEADER_BYTES, len(data))
    return data[start:end]


def check_pad_bytes_zero(
    data_path: Path,
    starts: np.ndarray,
    label: str,
) -> List[str]:
    """Every record's pad regions (pre AND post block) must be ``\\x00``.

    The new record layout is::

        [prefix][insn_bytes][pre_pad][block_bytes][post_pad][tokens]

    ``derive_pad_placement(header)`` returns the split that the writer
    used; both regions are independently asserted zero. Iterates per
    record over a single ``np.memmap`` -- only the prefix + pad slices
    are paged in, never the record body.
    """
    if not data_path.exists() or len(starts) == 0 or data_path.stat().st_size == 0:
        return []
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            start = int(starts[i])
            header, prefix_bytes = parse_binary_header(_header_window(data, start))
            pre_pad, post_pad = derive_pad_placement(header)
            block_bytes = header.block_word_count * BLOCK_WORD_SIZE[header.block_enc]

            pre_pad_start = start + prefix_bytes + header.insn_len
            pre_pad_end = pre_pad_start + pre_pad
            post_pad_start = pre_pad_end + block_bytes
            post_pad_end = post_pad_start + post_pad

            err = _first_nonzero_in_region(
                data, pre_pad_start, pre_pad_end, "pre-pad", i, start, label
            )
            if err is not None:
                errors.append(err)
                continue
            err = _first_nonzero_in_region(
                data, post_pad_start, post_pad_end, "post-pad", i, start, label
            )
            if err is not None:
                errors.append(err)
    finally:
        del data
    return errors


def _first_nonzero_in_region(
    data: np.ndarray,
    region_start: int,
    region_end: int,
    region_name: str,
    record_idx: int,
    record_start: int,
    label: str,
) -> str | None:
    """Return a single error string if any byte in the region is non-zero.

    Empty regions (``region_start == region_end``) short-circuit cleanly
    -- a zero-pad split is the common case for records whose body
    geometry already lands on a 16-byte boundary.
    """
    if region_end <= region_start:
        return None
    region = data[region_start:region_end]
    if bool(np.all(region == 0)):
        return None
    return (
        f"{label}: record {record_idx} (start={record_start}) has non-zero "
        f"{region_name} bytes {bytes(region)!r} at [{region_start}:{region_end}]"
    )


def check_pad_consistency(
    data_path: Path,
    starts: np.ndarray,
    label: str,
) -> List[str]:
    """Each record's writer layout must match ``derive_pad_placement``.

    Re-derives the rule from the parsed header alone (the rule lives in
    :mod:`aligned_data.binary_format._pad` and is the single source of
    truth) and asserts that:

    * total record size from ``record_total_size(header)`` is a
      multiple of :data:`RECORD_ALIGNMENT`;
    * the ``(pre_pad, post_pad)`` split is the one that the rule
      prescribes given ``P`` (``(-U) % 16``) and ``B``
      (``(-(prefix + insn_len)) % block_align``).

    The rule is a single-line conditional inside ``derive_pad_placement``
    so this check effectively pins that ``derive_pad_placement(header)``
    is the same value the writer used when laying down the record; a
    mismatch means the writer + header geometry disagree (or tampering).
    """
    if not data_path.exists() or len(starts) == 0 or data_path.stat().st_size == 0:
        return []
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            start = int(starts[i])
            header, prefix_bytes = parse_binary_header(_header_window(data, start))
            pre_pad, post_pad = derive_pad_placement(header)

            block_bytes = header.block_word_count * BLOCK_WORD_SIZE[header.block_enc]
            unpadded_total = prefix_bytes + header.insn_len + block_bytes + 2 * header.token_count
            total_pad = (-unpadded_total) % RECORD_ALIGNMENT
            block_align = BLOCK_WORD_SIZE[header.block_enc]
            block_pad = (-(prefix_bytes + header.insn_len)) % block_align

            expected_pre, expected_post = (
                (block_pad, total_pad - block_pad)
                if block_pad <= total_pad
                else (total_pad, 0)
            )
            if (pre_pad, post_pad) != (expected_pre, expected_post):
                errors.append(
                    f"{label}: record {i} (start={start}) pad split "
                    f"({pre_pad}, {post_pad}) disagrees with rule "
                    f"({expected_pre}, {expected_post}) "
                    f"(insn_len={header.insn_len}, block_word_count={header.block_word_count}, "
                    f"block_enc={header.block_enc}, token_count={header.token_count})"
                )
                continue
            total = record_total_size(header)
            if total % RECORD_ALIGNMENT != 0:
                errors.append(
                    f"{label}: record {i} (start={start}) total size {total} "
                    f"is not a multiple of {RECORD_ALIGNMENT}"
                )
    finally:
        del data
    return errors


def check_record_bounds(
    data_path: Path,
    starts: np.ndarray,
    label: str,
) -> List[str]:
    """Every record must fit inside ``_data.bin`` (``start + total <= size``).

    A truncated ``_data.bin`` or a corrupted index entry pointing past
    the file would otherwise surface as a silent ``IndexError`` later.
    Total size is derived from the parsed header via
    :func:`record_total_size`. ``file_size`` budget reserves space
    for the u32-aligned trailing ``total_entries`` field so the
    ``start + total <= effective_size`` bound stays exact.
    """
    if not data_path.exists() or len(starts) == 0:
        return []
    file_size = data_path.stat().st_size
    if file_size == 0:
        return []
    # The trailer (u32-aligned ``total_entries``) sits at the tail of
    # the file; record bodies must not extend into it. We don't know
    # the exact pad-byte count without re-walking, but the trailer +
    # its pad is at most ``DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE + 3``
    # bytes, which is the tightest upper bound that doesn't require
    # parsing the last record's geometry.
    from tokenizer.aligned_data.memmap_format import (
        DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE,
    )
    effective_size = file_size - DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            start = int(starts[i])
            header, _ = parse_binary_header(_header_window(data, start))
            total = record_total_size(header)
            end = start + total
            if end > effective_size:
                errors.append(
                    f"{label}: record {i} (start={start}, total_size={total}) "
                    f"extends to {end} but file_size={file_size} "
                    f"(effective={effective_size} accounting for trailer)"
                )
    finally:
        del data
    return errors


def check_entry_idx_sequence(
    data_path: Path,
    starts: np.ndarray,
    label: str,
) -> List[str]:
    """Every record's ``entry_idx`` must equal its index in ``starts``.

    The writer stamps ``entry_idx = N`` on the Nth record it appends;
    a corrupted ``_index.bin`` or misaligned starts array would surface
    here as an off-by-one mismatch. Also cross-checks the trailing
    ``total_entries`` against ``len(starts)`` so a truncated index
    is flagged at the file level.
    """
    if not data_path.exists() or len(starts) == 0:
        return []
    from tokenizer.aligned_data.memmap_format import read_data_bin_trailer

    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        total_entries = read_data_bin_trailer(data)
        if total_entries != len(starts):
            errors.append(
                f"{label}: trailer total_entries={total_entries} disagrees "
                f"with len(starts)={len(starts)}"
            )
        for i in range(len(starts)):
            start = int(starts[i])
            header, _ = parse_binary_header(_header_window(data, start))
            if header.entry_idx != i:
                errors.append(
                    f"{label}: record {i} (start={start}) has "
                    f"entry_idx={header.entry_idx}, expected {i}"
                )
    finally:
        del data
    return errors


def run_v1_post_checks(
    matched_index: Path,
    unmatched_index: Path,
    matched_data: Path,
    unmatched_data: Path,
    matched_starts: np.ndarray,
    unmatched_starts: np.ndarray,
) -> List[str]:
    """Per-record invariants that run only after preludes have validated.

    The starts arrays come from the already-loaded ``SectionArm`` so the
    validator never re-opens an index file for this step. Each per-arm
    probe touches its ``_data.bin`` once via ``np.memmap``; records are
    self-describing so no companion ``_lengths`` array is required.
    """
    errors: List[str] = []
    errors.extend(check_starts_alignment(matched_starts, f"{matched_index} (matched)"))
    errors.extend(check_starts_alignment(unmatched_starts, f"{unmatched_index} (unmatched)"))
    for data_path, starts in (
        (matched_data, matched_starts),
        (unmatched_data, unmatched_starts),
    ):
        errors.extend(check_pad_bytes_zero(data_path, starts, str(data_path)))
        errors.extend(check_pad_consistency(data_path, starts, str(data_path)))
        errors.extend(check_record_bounds(data_path, starts, str(data_path)))
        errors.extend(check_entry_idx_sequence(data_path, starts, str(data_path)))
    return errors
