"""Per-binary ``<binary>_strings.bin`` sidecar — wire format owner.

The sidecar holds the raw bytes of every string referenced by a
``string_ptr`` metadata entry for a given binary. Keeping string
content out of the per-function metadata column keeps the CSV compact
(real binaries can carry hundreds of KB of string data) and lets
consumers that don't need string content ignore the sidecar entirely.

This module is the *single* place that knows the on-disk format. The
tokenizer pipeline (``main_loop.py``, ``output_staging.py``,
``dynrunner.tokenize.tokenizer_task``) treats the sidecar opaquely —
they open it, flow the path through their plumbing, and close it.
Readers in downstream phases (``memmap_builder``, function-matching)
call ``iter_sidecar_lines`` / ``read_sidecar_line`` when they need to
materialize a string's bytes.

Wire format
-----------
One string per line; consecutive lines separated by a literal ``0x0A``
byte. Each line is one string's bytes with **selective C-style
escaping for ASCII control characters** so the line separator can
never appear inside a string. Other bytes (anything ``>= 0x20`` except
``0x5C`` and ``0x7F``, plus everything ``>= 0x80``) pass through
unmodified — UTF-8, UTF-16-LE, UTF-16-BE, latin-1 lines all keep
their non-ASCII bytes intact, so different lines may legitimately be
in different encodings. The encoding label belongs in the per-
function metadata (``string_ptr`` triplets reference
``{line, start_offset, encoding}``), not in the file itself; that's
why the file is ``.bin`` rather than ``.txt``.

Escape table (CANONICAL — both ``escape_line`` and ``unescape_line``
implement exactly this mapping; the constant ``ESCAPE_TABLE`` below is
the single source of truth):

================  ================================================
Original byte     On-disk encoding
================  ================================================
``0x5C`` (``\\``) ``\\\\`` (2 bytes)
``0x09`` (``\\t``) ``\\t`` (2 bytes)
``0x0A`` (``\\n``) ``\\n`` (2 bytes) — REQUIRED; literal 0x0A is
                  the line separator
``0x0D`` (``\\r``) ``\\r`` (2 bytes)
``0x07`` (``\\a``) ``\\a`` (2 bytes)
``0x08`` (``\\b``) ``\\b`` (2 bytes)
``0x0B`` (``\\v``) ``\\v`` (2 bytes)
``0x0C`` (``\\f``) ``\\f`` (2 bytes)
``0x00``          ``\\0`` (2 bytes)
``0x01-0x06``,    ``\\xHH`` (4 bytes, LOWERCASE hex)
``0x0E-0x1F``,
``0x7F``
``>= 0x20``       raw, 1 byte
(except ``0x5C``,
``0x7F``)
``>= 0x80``       raw, 1 byte (UTF-8/UTF-16/latin-1 multi-byte
                  bodies pass through untouched)
================  ================================================

``start_offset`` in a ``string_ptr`` triplet indexes into the
*unescaped* bytes of the line — the offset is a substring offset
into the real string, not into its on-disk escaped form. Consumers
must unescape first, then index.

Dedup invariant
---------------
The writer keys its dedup index on ``(raw_bytes, encoding)``. A
string referenced from N functions appears once at one line;
substring references at different offsets within the same string all
share the same line number with different ``start_offset`` values.
Identical strings at different lines is a wire-format bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterator


# Single source of truth for the escape mapping (see module docstring).
# Maps a single raw byte (int) to its on-disk byte sequence.
ESCAPE_TABLE: dict[int, bytes] = {
    0x5C: b"\\\\",
    0x09: b"\\t",
    0x0A: b"\\n",
    0x0D: b"\\r",
    0x07: b"\\a",
    0x08: b"\\b",
    0x0B: b"\\v",
    0x0C: b"\\f",
    0x00: b"\\0",
}
# Bytes that get the generic ``\xHH`` 4-byte escape (lowercase hex).
_HEX_ESCAPE_BYTES: frozenset[int] = frozenset(
    list(range(0x01, 0x07)) + list(range(0x0E, 0x20)) + [0x7F]
)

# Inverse of ESCAPE_TABLE for the single-letter escapes. The ``\xHH``
# branch in ``unescape_line`` handles the rest of the controls.
_UNESCAPE_LETTER: dict[int, int] = {
    ord("\\"): 0x5C,
    ord("t"): 0x09,
    ord("n"): 0x0A,
    ord("r"): 0x0D,
    ord("a"): 0x07,
    ord("b"): 0x08,
    ord("v"): 0x0B,
    ord("f"): 0x0C,
    ord("0"): 0x00,
}

# Line separator on disk. A real 0x0A byte, written between (and
# after) entries by the writer.
_LINE_SEP = b"\n"


class StringSidecar:
    """Writer for the per-binary ``<binary>_strings.bin`` sidecar file.

    Opens the file in binary write mode (truncating any existing
    contents) on construction; flushes and closes on
    :py:meth:`close` / context-manager exit.

    Each :py:meth:`add` call registers a string and returns its
    zero-based line index. The same ``(raw_bytes, encoding)`` tuple
    deduplicates: a second ``add`` with the same key returns the
    original line index and writes nothing.

    The caller passes the **unescaped** raw string bytes; the writer
    applies :py:meth:`escape_line` internally before writing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: BinaryIO | None = path.open("wb")
        # (raw_bytes, encoding) → line index. Dedup is by exact bytes
        # AND encoding because the same byte sequence under different
        # encodings is semantically distinct (e.g. a UTF-16-LE "hi" and
        # a hypothetical 4-byte latin-1 happen to share bytes but mean
        # different strings).
        self._dedup: dict[tuple[bytes, str], int] = {}
        self._next_line: int = 0

    def add(self, raw_bytes: bytes, encoding: str) -> int:
        """Register a string; return its zero-based line index.

        Idempotent on ``(raw_bytes, encoding)``: a repeat call with the
        same key returns the original line and writes nothing.

        ``raw_bytes`` are the UNESCAPED bytes of the string; the
        writer escapes them on the fly. ``encoding`` is a label
        preserved in the per-function metadata column (``ascii``,
        ``utf-8``, ``utf-16-le``, ``utf-16-be``, ``latin-1``,
        ``unknown``, ...). It does NOT change how bytes are written —
        the wire format is encoding-agnostic.
        """
        if self._fh is None:
            raise ValueError("StringSidecar is closed")
        key = (raw_bytes, encoding)
        existing = self._dedup.get(key)
        if existing is not None:
            return existing
        escaped = self.escape_line(raw_bytes)
        # Wire-format invariant: the line separator must never appear
        # inside a written entry. If this trips, the escape table is
        # incomplete.
        assert 0x0A not in escaped, (
            "escape_line produced a literal 0x0A; escape table is broken"
        )
        line_index = self._next_line
        self._fh.write(escaped)
        self._fh.write(_LINE_SEP)
        self._dedup[key] = line_index
        self._next_line += 1
        return line_index

    def close(self) -> None:
        """Flush and close the underlying file. Idempotent."""
        if self._fh is None:
            return
        self._fh.flush()
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "StringSidecar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def escape_line(raw_bytes: bytes) -> bytes:
        """Apply the selective C-style escape table.

        The result is guaranteed not to contain a literal ``0x0A``
        byte (that's the line separator). Bytes ``>= 0x80`` pass
        through unmodified, so multi-byte UTF-8 / UTF-16 / latin-1
        sequences are preserved.
        """
        out = bytearray()
        for b in raw_bytes:
            mapped = ESCAPE_TABLE.get(b)
            if mapped is not None:
                out.extend(mapped)
            elif b in _HEX_ESCAPE_BYTES:
                # Lowercase hex per spec.
                out.extend(b"\\x%02x" % b)
            else:
                # b >= 0x20 (and not 0x5C, not 0x7F), or b >= 0x80.
                out.append(b)
        return bytes(out)

    @staticmethod
    def unescape_line(line_bytes: bytes) -> bytes:
        """Inverse of :py:meth:`escape_line`.

        Scans for backslash escapes and reconstructs the original
        bytes. Any non-escape byte passes through verbatim. The input
        must not contain a literal ``0x0A`` (callers strip the line
        separator first).
        """
        out = bytearray()
        i = 0
        n = len(line_bytes)
        while i < n:
            b = line_bytes[i]
            if b != 0x5C:
                out.append(b)
                i += 1
                continue
            # Backslash escape: peek the next byte.
            if i + 1 >= n:
                raise ValueError(
                    "truncated escape: trailing backslash at end of line"
                )
            nxt = line_bytes[i + 1]
            letter = _UNESCAPE_LETTER.get(nxt)
            if letter is not None:
                out.append(letter)
                i += 2
                continue
            if nxt == ord("x"):
                if i + 4 > n:
                    raise ValueError(
                        "truncated \\xHH escape near end of line"
                    )
                hex_bytes = line_bytes[i + 2 : i + 4]
                try:
                    val = int(hex_bytes.decode("ascii"), 16)
                except (UnicodeDecodeError, ValueError) as e:
                    raise ValueError(
                        f"malformed \\xHH escape: {hex_bytes!r}"
                    ) from e
                out.append(val)
                i += 4
                continue
            raise ValueError(
                f"unknown escape sequence \\{chr(nxt) if 0x20 <= nxt < 0x7F else nxt!r}"
            )
        return bytes(out)


def iter_sidecar_lines(path: Path) -> Iterator[bytes]:
    """Yield each UNESCAPED line in order from the sidecar at ``path``.

    Streams the file in binary mode, splits on the literal ``0x0A``
    line separator, and unescapes each entry before yielding.
    """
    with path.open("rb") as fh:
        # Python's iter-over-binary-file splits on b'\n', which is
        # exactly our line separator. Each yielded chunk includes the
        # trailing 0x0A except possibly the last; we strip uniformly.
        for raw_line in fh:
            if raw_line.endswith(_LINE_SEP):
                raw_line = raw_line[:-1]
            if not raw_line:
                # Trailing empty element from a file that ends with a
                # separator (which is the writer's normal behaviour).
                # Genuine empty strings round-trip as a zero-length
                # escaped form too, but our writer always writes a
                # separator AFTER each entry, so a zero-length chunk
                # right before EOF is just the post-last-entry tail —
                # NOT a real empty entry. The only way to distinguish
                # them is position: a real empty entry is followed by
                # a separator (and thus more file content); the
                # post-tail empty is at EOF. Python's file iterator
                # only yields the post-tail empty if the file doesn't
                # end with a separator at all, so we never see it
                # from a well-formed writer output, and treating any
                # empty chunk as a real empty entry is correct.
                yield b""
                continue
            yield StringSidecar.unescape_line(raw_line)


def read_sidecar_line(path: Path, line_index: int) -> bytes:
    """Read just the requested line; unescape and return raw bytes.

    Streaming scan from the start of the file — fine for one-shot
    lookups. Callers doing many lookups against the same sidecar
    should iterate via :py:func:`iter_sidecar_lines` once and build
    their own index instead of calling this in a loop.
    """
    if line_index < 0:
        raise IndexError(f"negative line_index: {line_index}")
    for idx, line in enumerate(iter_sidecar_lines(path)):
        if idx == line_index:
            return line
    raise IndexError(
        f"line_index {line_index} out of range for sidecar {path}"
    )
