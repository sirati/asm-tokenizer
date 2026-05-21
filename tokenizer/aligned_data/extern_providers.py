"""Per-binary extern-provider sidecar (``<binary>_extern_providers.txt``).

This module owns the wire format of the sidecar that maps each unique
library name encountered on a binary's extern call_targets to a stable
1-indexed line number. The line number is what writers stamp into a
call_target's ``function_section_ptr`` field for ``type=extern`` entries
whose library is known.

Wire format (UTF-8 text, LF newlines):

* Line 0 (the file's first line): ``# format=<MEMMAP_FORMAT_VERSION>\\n``
  prelude, mirroring the convention used by the other per-binary
  sidecars (see :mod:`tokenizer.memmap_builder.function_names`).
* Lines 1..N: one library name per line, in **encounter order** (the
  order in which :meth:`ExternProviderRegistry.add` first saw each
  unique library). Encounter order -- not alphabetical -- is the
  distinguishing property versus :class:`FunctionNamesRegistry`:
  callers want stable 1-indexed slots assigned as the build progresses
  so they can stamp a call_target's pointer immediately, without a
  finalize step.

Reserved sentinel: file-line ``0`` (i.e. the prelude line, which is
never a real library) doubles as the "library unknown" pointer value.
A ``function_section_ptr = 0`` on an extern call_target means the
writer had no library information for that call; the consumer treats
the library as unknown without consulting this sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


class ExternProviderRegistry:
    """Collect unique library names and emit the sidecar.

    Encounter-order, 1-indexed, idempotent on the library string. There
    is no finalize step: :meth:`add` is callable up to (and after) any
    :meth:`write_sidecar` call, and the on-disk artefact can be
    re-stamped at any time during the build.
    """

    def __init__(self) -> None:
        # dict preserves insertion order; value is the assigned 1-indexed
        # line number. We could derive the line number from positional
        # insertion, but storing it explicitly makes ``add`` O(1) idempotent
        # without re-walking the dict.
        self._line_no: dict[str, int] = {}

    def add(self, library: str) -> int:
        """Register ``library`` (if new) and return its 1-indexed line number.

        Idempotent: a second call with the same library string returns
        the existing line number and does not emit a new entry. Line
        numbers are assigned in encounter order starting at 1; line 0
        is reserved as the "library unknown" sentinel and is never
        emitted as a real row.
        """
        existing = self._line_no.get(library)
        if existing is not None:
            return existing
        line_no = len(self._line_no) + 1
        self._line_no[library] = line_no
        return line_no

    def write_sidecar(self, output_dir: Path, binary_name: str) -> Path:
        """Write ``<output_dir>/<binary_name>_extern_providers.txt``.

        The file starts with the ``# format=N`` prelude; the following
        lines are the registered libraries in encounter order. Returns
        the written path. Idempotent: a second call rewrites the same
        content (modulo any libraries added in between).
        """
        path = Path(output_dir) / f"{binary_name}_extern_providers.txt"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(_PRELUDE_LINE)
            for library in self._line_no:
                f.write(library)
                f.write("\n")
        return path


def iter_extern_providers(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_no_1indexed, library_name)`` for each registered library.

    Validates the ``# format=N`` prelude line; mismatch raises
    :class:`ValueError` with a migration-pointing message. CRLF in the
    prelude is tolerated (writer always emits LF). Library lines have
    their trailing newline stripped before yielding.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        expected = _PRELUDE_LINE.rstrip("\n")
        if first_line.rstrip("\r\n") != expected:
            raise ValueError(
                f"{path}: missing or unsupported prelude; expected first line "
                f"{expected!r}, got {first_line!r}; re-run memmap_builder on the "
                f"per-binary CSVs to regenerate"
            )
        line_no = 0
        for raw in f:
            line_no += 1
            yield line_no, raw.rstrip("\r\n")
