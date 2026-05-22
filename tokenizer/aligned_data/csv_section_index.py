"""Codec for ``<binary>_matched_index.bin``: the function-to-section locator.

Single concern: the on-disk byte stream that maps function index to a
(section offset, section length) pair into ``<binary>_sections.bin``.
Pre-Phase-4 this file located CSV sections; post-cutover it locates
sections in the binary catalog. The codec bytes are unchanged because
sections in both formats are 4-byte aligned; only the referent (and the
kwarg names) move.

Layout (flat, no prelude, 8 bytes per entry, packed little-endian into
64 bits)::

    bits  0-39   u40  bin_offset_shifted          ``bin_offset >> 2``
    bits 40-63   u24  bin_section_length_shifted  ``bin_section_length >> 2``

Both fields are stored as ``>> 2`` because BIN sections are 4-byte
aligned (the :class:`SectionWriter` pads each section trailer up to a
4-byte boundary); shifting recovers two address bits per field. Real
caps therefore are:

* ``bin_offset``         < ``(1 << 40) << 2``  (4 TiB per BIN)
* ``bin_section_length`` < ``(1 << 24) << 2``  (64 MiB per section)

Why a separate codec from ``index_format.py``: ``index_format.py``
indexes ``_data.bin`` records that the writer 16-byte-aligns; this
locator indexes ``<binary>_sections.bin`` whose sections are 4-byte
aligned. The two layouts therefore stay in *separate* modules to keep
each codec's single concern intact: this file owns the locator into the
section catalog; ``index_format.py`` owns the locator into the v1 data
binary.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Optional, TextIO, Tuple

import numpy as np

from tokenizer.aligned_data.binary_format import IndexEntrySkip

ENTRY_SIZE: int = 8

_BIN_OFFSET_BITS: int = 40
_BIN_LENGTH_BITS: int = 24
_BIN_ALIGN_SHIFT: int = 2
_BIN_ALIGN: int = 1 << _BIN_ALIGN_SHIFT  # = 4

# Real-value caps (exclusive upper bound — i.e. the first value that
# overflows). Stored fields are ``value >> _BIN_ALIGN_SHIFT``.
MAX_BIN_OFFSET: int = (1 << _BIN_OFFSET_BITS) << _BIN_ALIGN_SHIFT
MAX_BIN_LENGTH: int = (1 << _BIN_LENGTH_BITS) << _BIN_ALIGN_SHIFT

_BIN_OFFSET_MASK: int = (1 << _BIN_OFFSET_BITS) - 1
_BIN_LENGTH_MASK: int = (1 << _BIN_LENGTH_BITS) - 1


def pack_csv_section_index_entry(bin_offset: int, bin_section_length: int) -> bytes:
    """Pack one 8-byte entry.

    Both values must be 4-byte aligned (asserted). Raises
    :class:`IndexEntrySkip` with ``bin_offset_overflow`` /
    ``bin_length_overflow`` when the real value reaches its cap
    (``(1 << 40) << 2`` for offset, ``(1 << 24) << 2`` for length).
    Returns exactly :data:`ENTRY_SIZE` bytes, little-endian.
    """
    assert bin_offset % _BIN_ALIGN == 0, (
        f"bin_offset must be {_BIN_ALIGN}-byte aligned, got {bin_offset}"
    )
    assert bin_section_length % _BIN_ALIGN == 0, (
        f"bin_section_length must be {_BIN_ALIGN}-byte aligned, "
        f"got {bin_section_length}"
    )
    if bin_offset >= MAX_BIN_OFFSET:
        raise IndexEntrySkip("bin_offset_overflow", bin_offset)
    if bin_section_length >= MAX_BIN_LENGTH:
        raise IndexEntrySkip("bin_length_overflow", bin_section_length)
    stored_offset = bin_offset >> _BIN_ALIGN_SHIFT
    stored_length = bin_section_length >> _BIN_ALIGN_SHIFT
    packed = stored_offset | (stored_length << _BIN_OFFSET_BITS)
    return struct.pack("<Q", packed)


def unpack_csv_section_index_entry(entry_bytes: bytes) -> Tuple[int, int]:
    """Inverse of :func:`pack_csv_section_index_entry`.

    Returns ``(bin_offset, bin_section_length)`` as real (post-shift)
    integers. Input must be exactly :data:`ENTRY_SIZE` bytes.
    """
    if len(entry_bytes) != ENTRY_SIZE:
        raise ValueError(
            f"entry_bytes must be {ENTRY_SIZE} bytes, got {len(entry_bytes)}"
        )
    (packed,) = struct.unpack("<Q", entry_bytes)
    stored_offset = packed & _BIN_OFFSET_MASK
    stored_length = (packed >> _BIN_OFFSET_BITS) & _BIN_LENGTH_MASK
    return (stored_offset << _BIN_ALIGN_SHIFT, stored_length << _BIN_ALIGN_SHIFT)


def write_csv_section_index_entry(
    file_handle,
    bin_offset: int,
    bin_section_length: int,
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
        entry_bytes = pack_csv_section_index_entry(bin_offset, bin_section_length)
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

    Returns ``(bin_starts, bin_lengths)`` with dtypes ``int64`` (real
    byte offsets into ``<binary>_sections.bin``) and ``uint32`` (real
    byte lengths). Returns ``None`` if the file does not exist. Raises
    :class:`ValueError` on stride mismatch (see
    :func:`_read_entries_uint64`).
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
        stored_offsets = (packed & np.uint64(_BIN_OFFSET_MASK)).astype(np.int64)
        stored_lengths = (
            (packed >> np.uint64(_BIN_OFFSET_BITS)) & np.uint64(_BIN_LENGTH_MASK)
        ).astype(np.uint32)
        bin_starts = stored_offsets << _BIN_ALIGN_SHIFT
        bin_lengths = stored_lengths << np.uint32(_BIN_ALIGN_SHIFT)
        return bin_starts, bin_lengths
    finally:
        if isinstance(raw, np.memmap):
            del raw


def iter_csv_section_index_entries(
    path: Path,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(bin_offset, bin_section_length)`` per entry in file order."""
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
