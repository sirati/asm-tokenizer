"""v1-format invariant checks for the memmap validator.

Single concern: assert the per-binary on-disk invariants the v1 writer
guarantees. Each check is a free function that takes the artefacts it
inspects and returns a ``list[str]`` of human-readable error messages
(empty on success). The validator orchestrates the calls and appends
results into its existing error list -- no shared state, no I/O hidden
behind class boundaries.

Layout / version knowledge crosses three module boundaries only:

  * ``aligned_data.memmap_format.MEMMAP_FORMAT_VERSION``
  * ``aligned_data.index_format.read_index_prelude``
  * ``aligned_data.binary_format.{HEADER_BYTES, OVERLONG_FIELD_BYTES,
    parse_binary_header}`` (record-header decoding)
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
    HEADER_BYTES,
    OVERLONG_FIELD_BYTES,
    compute_pad,
    parse_binary_header,
)
from tokenizer.aligned_data.index_format import MAX_NORMAL_REAL_LENGTH, read_index_prelude
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
    """Every index entry's resolved ``start`` must be 4-byte aligned.

    The post-shift writer guarantees this by construction (offsets are
    written as ``start >> 2`` and read back ``stored << 2``), but a
    corrupted ``_index.bin`` or a stale prelude alignment_shift could
    silently land on an unaligned offset; the validator catches it.
    """
    if len(starts) == 0:
        return []
    bad = np.where(starts % 4 != 0)[0]
    return [
        f"{label}: index entry {int(i)} start {int(starts[int(i)])} is not 4-byte aligned"
        for i in bad
    ]


def check_pad_bytes_zero(
    data_path: Path,
    starts: np.ndarray,
    lengths: np.ndarray,
    label: str,
) -> List[str]:
    """Every record's pad bytes between insn and block must be ``\\x00``.

    The writer always emits zeroed pad bytes; non-zero pad is either
    corruption or a writer regression. Iterates per record so this is
    O(n_entries) over the corpus; the check uses a single memmap and
    only touches the pad slice -- it does not materialise the whole
    record.
    """
    if not data_path.exists() or len(starts) == 0 or data_path.stat().st_size == 0:
        return []
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            start = int(starts[i])
            stored = int(lengths[i])
            _, is_overlong = resolve_record_length(data, start, stored)
            body_prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES if is_overlong else 0)
            header = parse_binary_header(data[start : start + HEADER_BYTES])
            if header.pad_size == 0:
                continue
            pad_start = start + body_prefix + header.insn_len
            pad_end = pad_start + header.pad_size
            pad = data[pad_start:pad_end]
            if not bool(np.all(pad == 0)):
                errors.append(
                    f"{label}: record {i} (start={start}) has non-zero pad bytes "
                    f"{bytes(pad)!r} at [{pad_start}:{pad_end}]"
                )
    finally:
        del data
    return errors


def check_sentinel_overlong_coupling(
    data_path: Path,
    starts: np.ndarray,
    lengths: np.ndarray,
    label: str,
) -> List[str]:
    """Index sentinel (length==0) <-> data record's overlong u24 field.

    For every entry the index marks as sentinel (the loader collapses
    ``length_shifted == 0x0000`` into ``lengths[i] == 0``), the record
    at ``start`` must carry a u24-shifted real length immediately after
    the 6-byte header. The check resolves the length via the shared
    decoder and asserts the result is in the overlong band (above the
    u16-shifted cap); a sentinel entry that decodes back to a normal
    length means writer + index disagree on the layout.
    """
    if not data_path.exists() or len(starts) == 0 or data_path.stat().st_size == 0:
        return []
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            if int(lengths[i]) != 0:
                continue
            start = int(starts[i])
            real_length, is_overlong = resolve_record_length(data, start, 0)
            if not is_overlong:
                errors.append(
                    f"{label}: index entry {i} flagged sentinel but record at "
                    f"start={start} did not resolve as overlong"
                )
            elif real_length <= MAX_NORMAL_REAL_LENGTH:
                errors.append(
                    f"{label}: index entry {i} sentinel/overlong-field mismatch: "
                    f"resolved real_length={real_length} fits the normal u16 cap "
                    f"({MAX_NORMAL_REAL_LENGTH}); the writer should not have used the sentinel"
                )
    finally:
        del data
    return errors


def check_pad_consistency(
    data_path: Path,
    starts: np.ndarray,
    lengths: np.ndarray,
    label: str,
) -> List[str]:
    """Every record's ``header.pad_size`` must equal ``compute_pad(...)``.

    Recovers token_count from body geometry then re-derives pad via the
    shared :func:`tokenizer.aligned_data.binary_format.compute_pad` --
    a mismatch means writer + header invariant disagree (or tampering).
    """
    if not data_path.exists() or len(starts) == 0 or data_path.stat().st_size == 0:
        return []
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            start = int(starts[i])
            real_length, is_overlong = resolve_record_length(data, start, int(lengths[i]))
            header = parse_binary_header(data[start : start + HEADER_BYTES])
            body_prefix = HEADER_BYTES + (OVERLONG_FIELD_BYTES if is_overlong else 0)
            token_count = (
                real_length - body_prefix - header.insn_len - header.pad_size - header.block_len
            ) // 2
            expected_pad = compute_pad(
                header.insn_len, header.block_len, token_count, is_overlong
            )
            if header.pad_size != expected_pad:
                errors.append(
                    f"{label}: record {i} (start={start}) pad_size={header.pad_size} "
                    f"disagrees with compute_pad={expected_pad} "
                    f"(insn_len={header.insn_len}, block_len={header.block_len}, "
                    f"token_count={token_count}, is_overlong={is_overlong})"
                )
    finally:
        del data
    return errors


def check_record_bounds(
    data_path: Path,
    starts: np.ndarray,
    lengths: np.ndarray,
    label: str,
) -> List[str]:
    """Every record must fit inside ``_data.bin`` (``start + real_length <= size``).

    A truncated ``_data.bin`` or a corrupted index entry pointing past
    the file would otherwise surface as a silent ``IndexError`` later.
    """
    if not data_path.exists() or len(starts) == 0:
        return []
    file_size = data_path.stat().st_size
    if file_size == 0:
        return []
    errors: List[str] = []
    data = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    try:
        for i in range(len(starts)):
            start = int(starts[i])
            real_length, _ = resolve_record_length(data, start, int(lengths[i]))
            end = start + real_length
            if end > file_size:
                errors.append(
                    f"{label}: record {i} (start={start}, real_length={real_length}) "
                    f"extends to {end} but file_size={file_size}"
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
    matched_lengths: np.ndarray,
    unmatched_starts: np.ndarray,
    unmatched_lengths: np.ndarray,
) -> List[str]:
    """Per-record invariants that run only after preludes have validated.

    The starts arrays come from the already-loaded ``SectionArm`` so the
    validator never re-opens an index file for this step. Pad + sentinel
    probes touch ``_data.bin`` once via ``np.memmap``.
    """
    errors: List[str] = []
    errors.extend(check_starts_alignment(matched_starts, f"{matched_index} (matched)"))
    errors.extend(check_starts_alignment(unmatched_starts, f"{unmatched_index} (unmatched)"))
    for data_path, index_path, starts, lengths in (
        (matched_data, matched_index, matched_starts, matched_lengths),
        (unmatched_data, unmatched_index, unmatched_starts, unmatched_lengths),
    ):
        errors.extend(check_pad_bytes_zero(data_path, starts, lengths, str(data_path)))
        errors.extend(
            check_sentinel_overlong_coupling(data_path, starts, lengths, str(index_path))
        )
        errors.extend(check_pad_consistency(data_path, starts, lengths, str(data_path)))
        errors.extend(check_record_bounds(data_path, starts, lengths, str(data_path)))
    return errors
