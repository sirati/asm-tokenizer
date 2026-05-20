"""Pre-v1 layout codec for ``<binary>_matched_index.bin``.

Single concern: the on-disk byte stream that maps function index to a
(section CSV byte offset, section CSV byte length) pair. This file is a
*function-to-CSV-section locator* — it indexes a TEXT file
(``<binary>_matched_sections.csv``), not the ``_data.bin`` record
stream.

Layout (flat, no prelude, 8 bytes per entry, packed little-endian into
64 bits)::

    bits  0-39   u40  csv_offset_shifted          ``csv_offset >> 2``
    bits 40-63   u24  csv_section_length_shifted  ``csv_section_length >> 2``

Both fields are stored as ``>> 2`` because matched-section starts and
lengths are written on a 4-byte boundary (the section CSV writer pads
the gap between sections with explicit ``\\n`` bytes); shifting recovers
two address bits per field. Real caps therefore are:

* ``csv_offset``         < ``(1 << 40) << 2``  (4 TiB per CSV)
* ``csv_section_length`` < ``(1 << 24) << 2``  (64 MiB per section)

Why pre-v1 (no ``IDX1`` prelude, no alignment shift in the format
sense): the v1 wire format (see ``index_format.py``) indexes
``_data.bin`` records that the writer 16-byte-aligns; this locator
indexes a text file. The two layouts therefore stay in *separate*
modules to keep each codec's single concern intact: this file owns the
locator into the section CSV; ``index_format.py`` owns the locator into
the v1 data binary. Per-variant pointers into the data binary are
carried inline in the section CSV via the inline-indexer codec, not
through this file.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Optional, TextIO, Tuple

import numpy as np

from tokenizer.aligned_data.binary_format import IndexEntrySkip

ENTRY_SIZE: int = 8

_CSV_OFFSET_BITS: int = 40
_CSV_LENGTH_BITS: int = 24
_CSV_ALIGN_SHIFT: int = 2
_CSV_ALIGN: int = 1 << _CSV_ALIGN_SHIFT  # = 4

# Real-value caps (exclusive upper bound — i.e. the first value that
# overflows). Stored fields are ``value >> _CSV_ALIGN_SHIFT``.
MAX_CSV_OFFSET: int = (1 << _CSV_OFFSET_BITS) << _CSV_ALIGN_SHIFT
MAX_CSV_LENGTH: int = (1 << _CSV_LENGTH_BITS) << _CSV_ALIGN_SHIFT

_CSV_OFFSET_MASK: int = (1 << _CSV_OFFSET_BITS) - 1
_CSV_LENGTH_MASK: int = (1 << _CSV_LENGTH_BITS) - 1


def pack_csv_section_index_entry(csv_offset: int, csv_section_length: int) -> bytes:
    """Pack one 8-byte entry.

    Both values must be 4-byte aligned (asserted). Raises
    :class:`IndexEntrySkip` with ``csv_offset_overflow`` /
    ``csv_length_overflow`` when the real value reaches its cap
    (``(1 << 40) << 2`` for offset, ``(1 << 24) << 2`` for length).
    Returns exactly :data:`ENTRY_SIZE` bytes, little-endian.
    """
    assert csv_offset % _CSV_ALIGN == 0, (
        f"csv_offset must be {_CSV_ALIGN}-byte aligned, got {csv_offset}"
    )
    assert csv_section_length % _CSV_ALIGN == 0, (
        f"csv_section_length must be {_CSV_ALIGN}-byte aligned, "
        f"got {csv_section_length}"
    )
    if csv_offset >= MAX_CSV_OFFSET:
        raise IndexEntrySkip("csv_offset_overflow", csv_offset)
    if csv_section_length >= MAX_CSV_LENGTH:
        raise IndexEntrySkip("csv_length_overflow", csv_section_length)
    stored_offset = csv_offset >> _CSV_ALIGN_SHIFT
    stored_length = csv_section_length >> _CSV_ALIGN_SHIFT
    packed = stored_offset | (stored_length << _CSV_OFFSET_BITS)
    return struct.pack("<Q", packed)


def unpack_csv_section_index_entry(entry_bytes: bytes) -> Tuple[int, int]:
    """Inverse of :func:`pack_csv_section_index_entry`.

    Returns ``(csv_offset, csv_section_length)`` as real (post-shift)
    integers. Input must be exactly :data:`ENTRY_SIZE` bytes.
    """
    if len(entry_bytes) != ENTRY_SIZE:
        raise ValueError(
            f"entry_bytes must be {ENTRY_SIZE} bytes, got {len(entry_bytes)}"
        )
    (packed,) = struct.unpack("<Q", entry_bytes)
    stored_offset = packed & _CSV_OFFSET_MASK
    stored_length = (packed >> _CSV_OFFSET_BITS) & _CSV_LENGTH_MASK
    return (stored_offset << _CSV_ALIGN_SHIFT, stored_length << _CSV_ALIGN_SHIFT)


def write_csv_section_index_entry(
    file_handle,
    csv_offset: int,
    csv_section_length: int,
    *,
    func_name: str = "",
    error_log: Optional[TextIO] = None,
) -> None:
    """Pack + write one entry, mirroring the v1 error-log policy.

    On :class:`IndexEntrySkip` from the packer: with ``error_log``
    supplied the exception is logged and the function returns without
    writing; without ``error_log`` it propagates. Successful path
    appends exactly :data:`ENTRY_SIZE` bytes.
    """
    try:
        entry_bytes = pack_csv_section_index_entry(csv_offset, csv_section_length)
    except IndexEntrySkip as exc:
        if error_log is None:
            raise
        # Lazy import: ``tokenizer.memmap_builder`` pulls back into
        # ``aligned_data``; a top-level import would cycle.
        from tokenizer.memmap_builder.error_log import write_error_log_entry

        write_error_log_entry(error_log, exc.reason, func_name, exc.value)
        return
    file_handle.write(entry_bytes)


def _read_entries_uint64(path: Path) -> Optional[np.ndarray]:
    """Memmap ``path`` as an ``(n_entries,)`` uint64. ``None`` if absent.

    Raises :class:`ValueError` with a ``re-run memmap_builder`` pointer
    if the file size is not a multiple of :data:`ENTRY_SIZE` (legacy
    stride mismatch from a prior on-disk layout).
    """
    if not path.exists():
        return None
    filesize = path.stat().st_size
    if filesize % ENTRY_SIZE != 0:
        raise ValueError(
            f"{path}: file size {filesize} is not a multiple of "
            f"{ENTRY_SIZE} (legacy matched_index.bin stride); "
            f"re-run memmap_builder to regenerate."
        )
    n_entries = filesize // ENTRY_SIZE
    if n_entries == 0:
        return np.zeros(0, dtype=np.uint64)
    return np.memmap(path, dtype=np.uint64, mode="r", shape=(n_entries,))


def read_csv_section_index_arrays(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load ``matched_index.bin`` into column ndarrays.

    Returns ``(csv_starts, csv_lengths)`` with dtypes ``int64`` (real
    byte offsets) and ``uint32`` (real byte lengths). Returns ``None``
    if the file does not exist. Raises :class:`ValueError` on stride
    mismatch (see :func:`_read_entries_uint64`).
    """
    raw = _read_entries_uint64(path)
    if raw is None:
        return None
    try:
        n_entries = raw.shape[0]
        if n_entries == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.uint32),
            )
        # Decode in vector form: low 40 bits → stored offset, next 24
        # bits → stored length. Multiply back by the alignment factor
        # to recover real byte positions / lengths.
        packed = np.asarray(raw, dtype=np.uint64)
        stored_offsets = (packed & np.uint64(_CSV_OFFSET_MASK)).astype(np.int64)
        stored_lengths = (
            (packed >> np.uint64(_CSV_OFFSET_BITS)) & np.uint64(_CSV_LENGTH_MASK)
        ).astype(np.uint32)
        csv_starts = stored_offsets << _CSV_ALIGN_SHIFT
        csv_lengths = stored_lengths << np.uint32(_CSV_ALIGN_SHIFT)
        return csv_starts, csv_lengths
    finally:
        if isinstance(raw, np.memmap):
            del raw


def iter_csv_section_index_entries(
    path: Path,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(csv_offset, csv_section_length)`` per entry in file order."""
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
            yield unpack_csv_section_index_entry(raw)
