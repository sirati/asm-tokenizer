"""Pre-v1 layout codec for ``<binary>_matched_index.bin``.

Single concern: the on-disk byte stream that maps function index to a
(section CSV byte offset, section CSV byte length, average-length
bucket) triple. This file is a *function-to-CSV-section locator* — it
indexes a TEXT file (``<binary>_matched_sections.csv``), not the
``_data.bin`` record stream.

Layout (flat, no prelude, 8 bytes per entry, little-endian)::

    byte 0-3   u32  csv_section_start    text-file byte offset of the
                                         section header row
    byte 4-6   u24  csv_section_length   text-file byte length of the
                                         section
    byte 7     u8   avg_len_bucket       ``min(avg_len >> 4, 255)`` --
                                         consumed by length-conditioned
                                         function selection

Why pre-v1 (no ``IDX1`` prelude, no alignment shift, no sentinel /
overlong machinery): the v1 wire format (see ``index_format.py``)
exists to compactly index ``_data.bin`` records that the writer always
4-byte-aligns. Text-file byte positions in a CSV are not 4-aligned and
have no record-length cap that would benefit from the shift+sentinel
trick. The two layouts therefore stay in *separate* modules to keep
each codec's single concern intact: this file owns the locator into
the section CSV; ``index_format.py`` owns the locator into the v1 data
binary. Per-variant pointers into the data binary are carried inline
in the section CSV via the inline-indexer codec, not through this file.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.binary_format import IndexEntrySkip

ENTRY_SIZE: int = 8
MAX_CSV_OFFSET: int = (1 << 32) - 1
MAX_CSV_LENGTH: int = (1 << 24) - 1


def pack_csv_section_index_entry(csv_offset: int, csv_len: int, avg_len: int) -> bytes:
    """Pack one 8-byte pre-v1 entry.

    Raises :class:`IndexEntrySkip` with ``csv_offset_overflow`` /
    ``csv_length_overflow`` when the offset or length exceeds its
    on-wire cap. ``avg_len`` is clamped the same way as v1
    (``min(avg_len >> 4, 255)``) so the bucket value matches what
    length-conditioned selection expects across both indices.
    """
    if csv_offset > MAX_CSV_OFFSET:
        raise IndexEntrySkip("csv_offset_overflow", csv_offset)
    if csv_len > MAX_CSV_LENGTH:
        raise IndexEntrySkip("csv_length_overflow", csv_len)
    avg_len_clamped = min(avg_len >> 4, 255)
    return (
        struct.pack("<I", csv_offset)
        + struct.pack("<I", csv_len)[:3]
        + struct.pack("B", avg_len_clamped)
    )


def write_csv_section_index_entry(
    file_handle,
    csv_offset: int,
    csv_len: int,
    avg_len: int,
    *,
    func_name: str = "",
    error_log=None,
) -> None:
    """Pack + write one pre-v1 entry, mirroring v1 error-log policy.

    On :class:`IndexEntrySkip` from the packer: with ``error_log``
    supplied the exception is logged and the function returns without
    writing; without ``error_log`` it propagates. Successful path
    appends exactly :data:`ENTRY_SIZE` bytes.
    """
    try:
        entry_bytes = pack_csv_section_index_entry(csv_offset, csv_len, avg_len)
    except IndexEntrySkip as exc:
        if error_log is None:
            raise
        # Lazy import: ``tokenizer.memmap_builder`` pulls back into
        # ``aligned_data``; a top-level import would cycle.
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return
    file_handle.write(entry_bytes)


def _read_entries_uint8(path: Path) -> Optional[np.ndarray]:
    """Memmap ``path`` as ``(n_entries, ENTRY_SIZE)`` uint8. ``None`` if absent."""
    if not path.exists():
        return None
    filesize = path.stat().st_size
    if filesize % ENTRY_SIZE != 0:
        raise ValueError(
            f"{path}: file size {filesize} is not a multiple of "
            f"{ENTRY_SIZE} (pre-v1 matched_index.bin layout)"
        )
    n_entries = filesize // ENTRY_SIZE
    if n_entries == 0:
        return np.zeros((0, ENTRY_SIZE), dtype=np.uint8)
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(n_entries, ENTRY_SIZE))


def read_csv_section_index_arrays(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a pre-v1 ``matched_index.bin`` into column ndarrays.

    Returns ``(csv_starts, csv_lengths, avg_lengths)`` with dtypes
    ``int64``/``uint32``/``uint8``. Returns ``None`` if the file does
    not exist; raises :class:`ValueError` on a non-multiple-of-8 size.
    """
    raw = _read_entries_uint8(path)
    if raw is None:
        return None
    try:
        n_entries = raw.shape[0]
        if n_entries == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.uint32),
                np.zeros(0, dtype=np.uint8),
            )
        # u32 csv_start: view bytes [0:4] as little-endian uint32.
        csv_starts = (
            np.ascontiguousarray(raw[:, 0:4]).view(np.uint32).reshape(n_entries).astype(np.int64)
        )
        # u24 csv_len: pad bytes [4:7] with a zero column then view as u32.
        len_block = np.zeros((n_entries, 4), dtype=np.uint8)
        len_block[:, 0:3] = raw[:, 4:7]
        csv_lengths = len_block.view(np.uint32).reshape(n_entries)
        # u8 avg_len.
        avg_lengths = np.ascontiguousarray(raw[:, 7])
        return csv_starts, csv_lengths, avg_lengths
    finally:
        if isinstance(raw, np.memmap):
            del raw


def iter_csv_section_index_entries(
    path: Path,
) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(csv_start, csv_len, avg_len)`` per entry of a pre-v1 file."""
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(ENTRY_SIZE)
            if not raw:
                return
            if len(raw) != ENTRY_SIZE:
                raise ValueError(
                    f"{path}: truncated entry of {len(raw)} bytes; "
                    f"expected {ENTRY_SIZE}"
                )
            csv_start = int.from_bytes(raw[0:4], "little")
            csv_len = int.from_bytes(raw[4:7], "little")
            avg_len = raw[7]
            yield csv_start, csv_len, avg_len
