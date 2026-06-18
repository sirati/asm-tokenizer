"""Lazy memmap-handle acquisition for :class:`BinarySession`.

Single concern: open + validate the per-binary file handles on first
touch (``<binary>_sections.bin`` as a paged memoryview,
``_data.bin`` / ``_unmatched_data.bin``, ``_variants.bin``) and pin
each release to the session's :class:`~contextlib.ExitStack` so a
long-lived worker never leaks file descriptors. The handle lifecycle
(``__enter__`` / ``__exit__`` / ref-drop) stays on
:class:`BinarySession`; this mixin owns only the acquisition + the
prelude/fingerprint validation that gates a handle becoming usable.

Exposed as a mixin :class:`_HandlesMixin` so the openers stay on
:class:`BinarySession` (callers need not know about the split). Every
attribute it reads -- ``_stack``, ``_base_path``, ``_binary_name``,
``_vocab_manager``, and the cached ``_sections_bin_*`` / ``_data_*`` /
``_variants_mmap`` slots -- is owned by :class:`BinarySession` itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...matched_sections_bin import Section, parse_section_bin
from ...memmap_format import MATCHED_SECTIONS_BIN_PRELUDE_SIZE
from .._sections_bin_walk import read_sections_bin_blob


def _close_memmap(mmap_obj) -> None:
    # Pin mmap release to ExitStack vs GC -- long-lived workers leak fds.
    inner = getattr(mmap_obj, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:
            pass


class _HandlesMixin:
    """Mixin providing the lazy memmap openers + section parse.

    Every method reads attributes that :class:`BinarySession` owns;
    this class deliberately holds no state of its own. ``self`` is
    typed ``Any`` inside the bodies because the concrete attributes
    live on the subclass.
    """

    def _open_sections_bin(  # type: ignore[no-untyped-def]
        self,
    ) -> memoryview:
        """Lazy-``mmap`` the per-binary section catalog as a memoryview.

        The catalog is ``np.memmap``-ed (NOT slurped) so
        :func:`parse_section_bin` pages in only the section(s) a batch
        actually touches. A fresh :class:`BinarySession` is opened per
        sampled binary per batch; a full read would copy the ENTIRE
        catalog every time (z3's ``_sections.bin`` is ~348MB), so the
        eager copy dominated per-batch memory even though the far larger
        ``_data.bin`` was already lazy. The memoryview keeps parser
        slicing zero-copy and is pinned (with the backing ``np.memmap``)
        for the session lifetime. Prelude is validated on first open.
        """
        if self._stack is None:
            raise RuntimeError("BinarySession used outside its with-block")
        if self._sections_bin_view is not None:
            return self._sections_bin_view
        # ``matched_arm`` and ``unmatched_arm`` share the same BIN file;
        # which arm's path we resolve doesn't matter, but we walk through
        # the conventional per-binary filename for clarity.
        path = self._base_path / f"{self._binary_name}_sections.bin"
        # Pin the mmap so the view (and any Section sliced from it) stays
        # valid for the session lifetime; __exit__ releases the view then
        # drops this ref, so the mapping unmaps by refcounting with no
        # explicit close (no exported-pointer BufferError risk).
        mm, view = read_sections_bin_blob(path)
        self._sections_bin_blob = mm
        self._sections_bin_view = view
        return view

    def _sections_bin_u8(  # type: ignore[no-untyped-def]
        self,
    ) -> np.ndarray:
        """The section catalog as a 1-D ``uint8`` array (zero-copy).

        Ensures the catalog is mapped (via :py:meth:`_open_sections_bin`,
        which pins the prelude-validated ``np.memmap``) and surfaces that
        same backing ``np.memmap`` -- already a ``uint8`` ndarray over the
        whole file with absolute offsets. The vectorized columnar readers
        (:func:`...matched_sections_columnar.read_n_variants_columnar`)
        gather over a uint8 array, not the parser's ``memoryview``; this
        accessor hands them the cached mapping so a header-only batch read
        pages in only the touched section headers -- no full parse.
        """
        self._open_sections_bin()
        return self._sections_bin_blob

    def _parse_section_at(  # type: ignore[no-untyped-def]
        self, offset: int
    ) -> Section:
        """Parse one BIN section at the given byte offset.

        Single chokepoint: every slice call routes through here so the
        prelude assertion fires exactly once per session and the
        zero-copy memoryview is reused across calls.
        """
        if offset < MATCHED_SECTIONS_BIN_PRELUDE_SIZE:
            raise ValueError(
                f"section offset {offset} is inside the BIN prelude "
                f"(<{MATCHED_SECTIONS_BIN_PRELUDE_SIZE}); the index "
                f"file is corrupt"
            )
        blob = self._open_sections_bin()
        section, _end = parse_section_bin(blob, offset)
        return section

    def _open_data(  # type: ignore[no-untyped-def]
        self, kind: str
    ) -> np.ndarray:
        if self._stack is None:
            raise RuntimeError("BinarySession used outside its with-block")
        if self._data_mmap is not None:
            if self._data_kind != kind:
                raise RuntimeError(
                    f"BinarySession already opened {self._data_kind} data; "
                    f"cannot switch to {kind} mid-session"
                )
            return self._data_mmap
        suffix = "_unmatched_data.bin" if kind == "unmatched" else "_data.bin"
        path = self._base_path / f"{self._binary_name}{suffix}"
        mmap = np.memmap(str(path), dtype=np.uint8, mode="r")
        # Validate the 16-byte file-level prelude up front so a stale /
        # pre-prelude / wrong-format bin fails loud on open instead of
        # returning garbage records on first slice.
        from tokenizer.aligned_data.memmap_format import (
            DATA_BIN_PRELUDE_SIZE,
            NO_FINGERPRINT,
            assert_data_bin_prelude,
            read_bin_prelude_reserved,
            read_data_bin_trailer,
        )
        prelude = bytes(mmap[:DATA_BIN_PRELUDE_SIZE])
        assert_data_bin_prelude(prelude, path=str(path))
        # #27 safety net: a _data.bin built post-fingerprint carries the
        # identity of the unified vocab it was built against. If we hold a
        # fingerprinted vocab (loaded via the gate) and it disagrees, this
        # catalog is being decoded with the WRONG vocab -- _data.bin stores
        # unified-vocab token ids for the WHOLE stream, so a wrong vocab
        # silently remaps EVERY token (instructions, numbers, identities,
        # AND the variant axes), not just the prefix. Fail loud. Soft-skip
        # when either side lacks a fingerprint (pre-#27 bin, or a vocab not
        # loaded through the gate).
        catalog_fp = read_bin_prelude_reserved(prelude)
        if catalog_fp != NO_FINGERPRINT:
            vocab_fp = getattr(self._vocab_manager, "_vocab_fingerprint", None)
            if vocab_fp is not None and catalog_fp != vocab_fp:
                raise ValueError(
                    f"catalog<->vocab fingerprint mismatch for {path}: this "
                    f"_data.bin was built against a DIFFERENT unified_vocab "
                    f"(catalog={catalog_fp.hex()}) than the one loaded "
                    f"(vocab={vocab_fp.hex()}). Every token id in the stream "
                    f"resolves against the unified vocab, so decoding with the "
                    f"wrong one mis-decodes EVERY token, not just the variant "
                    f"axes. Load the unified_vocab.csv co-located with this memmap."
                )
        # The trailing ``total_entries`` u32 is the per-lookup
        # ``entry_idx < total_entries`` bound; read + cache it once
        # here so the hot path doesn't re-parse it per slice.
        self._data_total_entries = read_data_bin_trailer(mmap)
        self._stack.callback(_close_memmap, mmap)
        self._data_mmap = mmap
        self._data_kind = kind
        return mmap

    def _open_variants(  # type: ignore[no-untyped-def]
        self,
    ) -> Optional[np.ndarray]:
        if self._stack is None:
            raise RuntimeError("BinarySession used outside its with-block")
        if self._variants_mmap is not None:
            return self._variants_mmap
        path = self._base_path / f"{self._binary_name}_variants.bin"
        if not path.exists():
            return None
        mmap = np.memmap(str(path), dtype=np.uint8, mode="r")
        self._stack.callback(_close_memmap, mmap)
        self._variants_mmap = mmap
        return mmap

