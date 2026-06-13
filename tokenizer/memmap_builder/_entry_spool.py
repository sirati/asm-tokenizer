"""On-disk spill for the per-arm pass-1 entry stream.

Pass 1 of the memmap builder (:mod:`tokenizer.memmap_builder.passes`)
produces one Python ``dict`` per surviving matched / unmatched function
variant. At corpus scale (z3: millions of functions) retaining the whole
stream as live Python objects across the pass-1 → pass-2 boundary costs
many GB of RSS — the nested ``unique_called`` / per-variant ``called`` /
``extern_libraries`` payloads dominate.

This module owns a single concern: **persist that entry stream to a
temp file during pass-1, replay it during pass-2 without ever holding
the whole stream in RAM.** The builder appends each entry as pass-1
emits it (then drops the live object), and pass-2 re-iterates the spool
one entry at a time.

Wire format: an append-only sequence of pickle-framed records, each a
``<u32 little-endian length><pickle payload>`` frame. Pickle is the
faithful transport here — the entry dicts carry
:class:`~tokenizer.aligned_data.call_target_type.CallTargetType` enum
members and :class:`~tokenizer.memmap_builder.builder.VersionKey` frozen
dataclasses nested inside lists/dicts; pickle round-trips all of them
byte-for-byte, so a replayed entry is ``==`` to the one appended. The
spool is opaque storage, not a data-interchange API: callers never
inspect the bytes, they only ``append`` and iterate back the same
objects.

The spool is re-iterable: each :meth:`EntrySpool.__iter__` re-opens the
file from the start, so the lookup-table build and the section pass can
each stream it independently. Object lifetime is one entry at a time —
the iterator deserialises on demand and the caller releases each entry
after its section is written.
"""

from __future__ import annotations

import os
import pickle
import struct
import tempfile
from typing import Iterator

# Each frame is length-prefixed with a 4-byte little-endian unsigned
# record length so the reader can slice exactly one pickle payload
# without a second pass or a sentinel scan. Pickle protocol is pinned
# to the highest available so frozen dataclasses + IntEnum members
# round-trip via the fast C path.
_LEN = struct.Struct("<I")
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class EntrySpool:
    """Append-only, re-iterable on-disk spill for one arm's pass-1 entries.

    Lifetime: constructed once per arm inside ``build_memmap_files``,
    appended to during pass 1, iterated during pass 2, then
    :meth:`close`-d (which deletes the backing temp file). Intended to
    be owned by the builder's ``contextlib.ExitStack`` so an exception
    in any phase still unlinks the temp file.

    The append handle stays open across the whole pass-1 walk; it is
    flushed + closed by the first iteration (or by :meth:`close`) so the
    on-disk bytes are complete before any reader opens the file.
    """

    def __init__(self, dir: "str | os.PathLike[str] | None" = None) -> None:
        # A named temp file (delete=False) so we own the unlink in
        # :meth:`close`; ``dir`` lets the caller co-locate the spill with
        # the output tree (or rely on the system temp dir).
        fd, path = tempfile.mkstemp(suffix=".entryspool", dir=dir)
        self._path = path
        self._write_handle = os.fdopen(fd, "wb")
        self._closed = False

    def append(self, entry: dict) -> None:
        """Serialise one pass-1 entry and append its framed bytes.

        The caller may drop its reference to ``entry`` immediately after;
        the spool owns the only durable copy from here on.
        """
        if self._write_handle is None:
            raise RuntimeError(
                "EntrySpool.append after the spool was sealed for reading"
            )
        payload = pickle.dumps(entry, protocol=_PICKLE_PROTOCOL)
        self._write_handle.write(_LEN.pack(len(payload)))
        self._write_handle.write(payload)

    def _seal(self) -> None:
        """Flush + close the append handle so the file is complete.

        Idempotent. Called lazily on first iteration (and by
        :meth:`close`) — the writer must be flushed before any reader
        opens the file, but the builder appends across the whole of
        pass 1 without an explicit seal call, so iteration seals on
        demand.
        """
        if self._write_handle is not None:
            self._write_handle.flush()
            self._write_handle.close()
            self._write_handle = None

    def __iter__(self) -> Iterator[dict]:
        """Yield every appended entry in append order, one at a time.

        Re-openable: each call streams the file from the start with its
        own handle, so multiple passes (lookup-table build, section
        emit) read independently. Deserialises on demand — only the
        currently-yielded entry is alive.
        """
        self._seal()
        with open(self._path, "rb") as fh:
            while True:
                header = fh.read(_LEN.size)
                if not header:
                    return
                if len(header) != _LEN.size:
                    raise EOFError(
                        "EntrySpool: truncated length header "
                        f"({len(header)} of {_LEN.size} bytes)"
                    )
                (size,) = _LEN.unpack(header)
                payload = fh.read(size)
                if len(payload) != size:
                    raise EOFError(
                        "EntrySpool: truncated payload "
                        f"({len(payload)} of {size} bytes)"
                    )
                yield pickle.loads(payload)

    def close(self) -> None:
        """Seal the writer (if open) and unlink the backing temp file.

        Idempotent so it is safe to register as an ``ExitStack``
        callback and still call explicitly.
        """
        if self._closed:
            return
        self._seal()
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        self._closed = True
