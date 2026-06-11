"""Resolve a raw binary file by exact-filename match across roots.

Single concern: given a binary's fullname (e.g.
``arm32-clang-3.5-O0_minigzip``) find the on-disk raw binary of that
exact basename under one of the user-supplied ``--binaries-root``
directories.  No DB, no axis parsing, no archive extraction.

Raw binaries on disk are named by their fullname (the full
``<isa>-<comp>-<compv>-<optim>_<program>`` string), not by the bare
program, so the lookup key is the fullname; the resolver itself is
agnostic to the key's shape (it matches whatever basename it is given).

The corpus' raw sources are incomplete and may be spread across several
roots, so the resolver walks each root lazily and caches the
basename → path index it builds.  A fullname with no matching file
resolves to ``None`` (the caller stores raw_size as NULL).  Archives are
never opened and large directories are never materialised into memory
beyond the basename index itself.
"""

from __future__ import annotations

import os
from pathlib import Path


class RawResolver:
    """Maps a raw-binary basename to its on-disk path across roots.

    The first time a lookup misses the in-memory index, the next
    unwalked root is walked (lazily, one root at a time) until the name
    is found or every root is exhausted.  Roots are walked at most once
    each; the basename → path index persists for the resolver's life.

    When several roots (or several files) share a basename the first one
    encountered wins; later collisions are ignored so the index stays a
    flat name → path map (the corpus has no duplicate fullname basenames
    across packages that matter for a size lookup).
    """

    def __init__(self, roots: list[Path]) -> None:
        self._roots: list[Path] = list(roots)
        self._walked = 0
        self._index: dict[str, Path] = {}

    def _walk_next_root(self) -> bool:
        """Walk the next not-yet-walked root into the index.  Returns
        ``False`` when every root has already been walked."""
        if self._walked >= len(self._roots):
            return False
        root = self._roots[self._walked]
        self._walked += 1
        if root.is_dir():
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    self._index.setdefault(filename, Path(dirpath) / filename)
        return True

    def resolve(self, fullname: str) -> Path | None:
        """Return the raw file whose basename equals ``fullname``, or
        ``None`` when no supplied root contains such a file."""
        while True:
            hit = self._index.get(fullname)
            if hit is not None:
                return hit
            if not self._walk_next_root():
                return None
