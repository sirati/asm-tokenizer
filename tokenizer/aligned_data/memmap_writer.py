"""Memmap-backed sequential writer for ``_data.bin`` records.

Single concern: write record bytes to a memmap'd file at the current
cursor, returning the byte offset. Reads back already-written bytes at
arbitrary earlier offsets without coherence issues — that's the property
the content-addressed dedup helper relies on when it has to disambiguate
a hash collision by byte-comparing against a previously-written record.

The mapping grows on demand: when the next write would exceed the
current mapping size, the underlying file is ``ftruncate``'d to the next
size (geometric growth, capped at 1 GiB increments past 1 GiB) and the
mapping is ``mmap.resize()``'d in place. ``finalize()`` truncates the
file back to ``cursor`` so the on-disk size matches the actual data.

Stdio buffering would make the readback path racy (`pwrite` after a
buffered `write` can read stale bytes); the memmap path bypasses the
userspace buffer entirely and reads go straight through the kernel page
cache the writes populated.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path

# Initial mapping size for a fresh bin. Linux file systems back the
# range with sparse blocks, so this is real virtual address space but
# (almost) no real disk until we actually write into it.
_INITIAL_SIZE: int = 64 * 1024 * 1024

# Growth past 1 GiB is linear (1 GiB per step) rather than geometric;
# at corpus scale a 64 GiB bin would double itself once into 128 GiB and
# waste the upper half of an mmap address space we'll never use.
_LINEAR_GROWTH_THRESHOLD: int = 1024 * 1024 * 1024
_LINEAR_GROWTH_STEP: int = 1024 * 1024 * 1024


class MemmapBinWriter:
    """Append-only memmap'd writer that supports random reads.

    ``write(record_bytes)`` appends at the current cursor and returns the
    byte offset of the first byte. ``read(offset, length)`` returns
    bytes at any earlier offset; the underlying memmap means writes
    become visible to reads immediately.

    The writer is NOT thread-safe — the memmap-builder pipeline is
    single-threaded per arm.
    """

    def __init__(self, path: Path, prelude_bytes: bytes = b"") -> None:
        self._path = Path(path)
        # O_RDWR | O_CREAT | O_TRUNC: fresh file every build. The bin's
        # offsets are emitted into index files + section CSVs of the
        # same build, so any prior bin is stale by definition.
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(self._fd, _INITIAL_SIZE)
        self._size = _INITIAL_SIZE
        self._mm = mmap.mmap(
            self._fd,
            self._size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self._cursor = 0
        if prelude_bytes:
            self._mm[: len(prelude_bytes)] = prelude_bytes
            self._cursor = len(prelude_bytes)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cursor(self) -> int:
        return self._cursor

    def write(self, data: bytes) -> int:
        """Append ``data`` at the current cursor. Return the start offset."""
        needed = self._cursor + len(data)
        if needed > self._size:
            self._grow(needed)
        start = self._cursor
        self._mm[start:needed] = data
        self._cursor = needed
        return start

    def read(self, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``offset`` from the mapping."""
        return bytes(self._mm[offset : offset + length])

    def truncate_to(self, position: int) -> None:
        """Roll the cursor back to ``position`` (e.g. on encoder skip).

        The mapping size is left alone; only the next ``write`` will
        overwrite the speculatively-written bytes. This mirrors the
        pre-refactor ``file.seek(pre_write_pos); file.truncate()``
        pattern that the writer used on cap-overflow skips.
        """
        if position > self._cursor:
            raise ValueError(
                f"truncate_to({position}) would move cursor forward from "
                f"{self._cursor}; refusing"
            )
        self._cursor = position

    def _grow(self, needed: int) -> None:
        if self._size < _LINEAR_GROWTH_THRESHOLD:
            new_size = max(needed, self._size * 2)
        else:
            new_size = self._size
            while new_size < needed:
                new_size += _LINEAR_GROWTH_STEP
        os.ftruncate(self._fd, new_size)
        self._mm.resize(new_size)
        self._size = new_size

    def finalize(self) -> None:
        """Flush, unmap, truncate to ``cursor``, close the fd.

        Idempotent: a second call is a no-op so the builder's
        ``contextlib.ExitStack`` cleanup can safely invoke it even
        after an explicit close in the happy path.
        """
        if self._fd < 0:
            return
        self._mm.flush()
        self._mm.close()
        os.ftruncate(self._fd, self._cursor)
        os.close(self._fd)
        self._fd = -1
