"""Lazy per-section function-name resolution.

Single concern: expose a matched-arm ``func_names`` sequence whose
single-index lookup (``names[idx]``) resolves ONE section's
``function_name_ptr`` from ``<binary>_sections.bin`` on demand, without
parsing the whole catalog. Full materialisation (iteration, ``==`` a
list, ``list(names)``) is deferred to first such access and memoised.

Why this exists: the vector_batch decode path reads ``func_names[idx]``
for only the handful of sections a batch samples, but the eager arm
build resolved EVERY section's name up front (a full columnar parse +
a Python resolve loop, ~2.1 s on z3). The vb path never iterates the
list, so resolving names lazily — one mmap-paged section header per
sampled idx — removes the dominant open cost while keeping the
``Sequence`` contract the validator / inspector / tests rely on.

Boundary contract: this is a read-only :class:`Sequence[str]`. It owns
no file handle; every single-index parse mmaps ``sections_bin`` afresh
(``read_sections_bin_blob`` pages in only the touched section) and the
mapping releases by refcounting once the parse returns. Equality and
iteration build (and cache) the full list via the supplied ``resolve_all``
thunk — the SAME resolve the eager path produced — so byte-for-byte
identical names cross the boundary regardless of access shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ._sections_bin_walk import (
    read_sections_bin_blob,
    resolve_func_name_or_raise,
)
from ..matched_sections_bin import parse_section_bin


class LazyFuncNames(Sequence):
    """A ``func_names`` view: per-idx single-section resolve, lazy-full list.

    ``section_offsets[i]`` is the BIN byte offset of the i-th section;
    ``names[i]`` parses ONLY that section's header (via the mmap-paged
    ``parse_section_bin``) and resolves its ``function_name_ptr`` against
    ``line_to_name``. The full list is built (once, memoised) by
    ``resolve_all`` on any whole-sequence access (iteration / equality /
    ``list()``), so consumers that walk every name pay the eager cost
    exactly once and only when they actually walk.
    """

    def __init__(
        self,
        sections_bin: Path,
        section_offsets: np.ndarray,
        line_to_name: Dict[int, str],
        resolve_all: Callable[[], List[str]],
    ) -> None:
        self._sections_bin = sections_bin
        self._section_offsets = section_offsets
        self._line_to_name = line_to_name
        self._resolve_all = resolve_all
        self._full: Optional[List[str]] = None

    def __len__(self) -> int:
        return int(len(self._section_offsets))

    def __getitem__(self, idx):
        # Slices (and any access once the full list is realised) go
        # through the materialised list so the result type matches a
        # plain ``list`` exactly.
        if self._full is not None:
            return self._full[idx]
        if isinstance(idx, slice):
            return self._materialise()[idx]
        n = len(self._section_offsets)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(
                f"func_names index {idx} out of range (have {n})"
            )
        return self._resolve_one(int(idx))

    def _resolve_one(self, idx: int) -> str:
        """Resolve a single section's name by paging in only its header."""
        offset = int(self._section_offsets[idx])
        mm, blob = read_sections_bin_blob(self._sections_bin)
        try:
            section, _end = parse_section_bin(blob, offset)
            return resolve_func_name_or_raise(
                int(section.function_name_ptr),
                self._line_to_name,
                self._sections_bin,
                offset,
            )
        finally:
            blob.release()
            del mm

    def _materialise(self) -> List[str]:
        if self._full is None:
            self._full = list(self._resolve_all())
        return self._full

    def __iter__(self):
        return iter(self._materialise())

    def __eq__(self, other) -> bool:
        if isinstance(other, LazyFuncNames):
            other = other._materialise()
        if isinstance(other, (list, tuple)):
            return self._materialise() == list(other)
        return NotImplemented

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self) -> str:
        state = "realised" if self._full is not None else "lazy"
        return f"LazyFuncNames(n={len(self)}, {state})"
