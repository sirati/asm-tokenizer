"""Per-binary, batch-scoped handle lifecycle.

Single concern: own the three file handles a ``BinaryDataset`` uses
while serving a batch of slicing operations on ONE binary
(``<binary>_sections.bin``, ``_data.bin`` / ``_unmatched_data.bin``,
``_variants.bin``), and guarantee deterministic close on exit.

Lazy opens + a single ``contextlib.ExitStack``: handles nobody touches
stay closed; handles that DO open are unwound (in reverse order) by
the stack on ``__exit__``, even when a mid-batch slice raises.
``__exit__`` is idempotent.

This module does NOT load metadata (``metadata_loader``), parse
data-bin records (``aligned_data.io.parse_function_data_memmap``), or
own the variant-ref decoder (``variant_resolver``). Section-parsing
glue lives in ``_session_parsers``.

**Lifetime contract (egress copy)**: every ``FunctionData`` /
``MatchedFunction`` returned by a slice method is independent of the
session's open memmap handles -- :py:meth:`BinarySession._slice_data_record`
copies ``tokens`` / ``insn_runlength`` / ``block_runlength`` off the
zero-copy ``extract_arrays_from_data`` views before they reach the
caller. Callers may consume returned arrays freely after the ``with``
exits; per-record copy cost is negligible vs the memmap-paging the
reader already paid, and mirrors ``variant_resolver.get_variant_by_ref``
which already copies ``variant_tokens`` for the same reason.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ...binary_format import (
    MAX_HEADER_BYTES,
    extract_arrays_from_data,
    parse_binary_header,
    record_total_size,
)
from ...matched_sections_bin import Section, parse_section_bin
from ...memmap_format import MATCHED_SECTIONS_BIN_PRELUDE_SIZE
from .._sections_bin_walk import read_sections_bin_blob
from .._session_helpers import _BinarySessionHelpersMixin
from .._worker_guard import assert_main_process
from ..function_data import FunctionData
from ..matched_function import MatchedFunction
from ..variant_resolver import get_variant_by_ref as _resolve_variant_by_ref
from ._matched_load import _MatchedLoadMixin
from ._unmatched_load import _UnmatchedLoadMixin


def _close_memmap(mmap_obj) -> None:
    # Pin mmap release to ExitStack vs GC -- long-lived workers leak fds.
    inner = getattr(mmap_obj, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:
            pass


class BinarySession(
    _MatchedLoadMixin, _UnmatchedLoadMixin, _BinarySessionHelpersMixin
):
    """Context manager bundling the three per-binary handles.

    ``metadata`` is a pre-loaded bag (built by ``metadata_loader``).
    Accessed attribute-first, dict-fallback via :meth:`get_metadata`
    (public accessor wrapping ``_meta_get``). Expected keys/attrs:

      * ``matched_arm``        -- SectionArm: ``.starts`` (per-variant
                                  data-bin offsets), ``.bin_starts`` /
                                  ``.bin_lengths`` (per-function BIN
                                  catalog locator), ``.func_names``
      * ``unmatched_arm``      -- SectionArm: ``.starts`` (per-record
                                  data-bin offsets), ``.func_names``,
                                  ``.section_starts`` (per-function
                                  BIN catalog offsets)
      * ``offset_to_filename`` -- ``dict[int, str]``
      * ``line_to_name``       -- ``dict[int, str]`` for resolving
                                  unmatched ``call_target`` FIDs to
                                  function names.
      * ``line_to_provider``   -- ``dict[int, str]`` for resolving an
                                  EXTERN ``call_target.function_section_ptr``
                                  to its library / provider name
                                  (loaded from the per-binary
                                  ``<binary>_extern_providers.txt``).

    ``_data.bin`` records are self-describing -- their headers carry
    insn / block / token geometry -- so no companion ``lengths`` or
    ``is_overlong`` array crosses any boundary here. Section parsing
    happens against an ``np.memmap`` ``memoryview`` of
    ``<binary>_sections.bin`` (lazy per-section paging, not a full read);
    the BIN's prelude is validated on first open and a per-session
    memoryview is held until ``__exit__``.
    """

    def __init__(
        self,
        base_path: Path,
        binary_name: str,
        vocab_manager: Any,
        metadata: Any,
    ) -> None:
        self._base_path = Path(base_path)
        self._binary_name = binary_name
        self._vocab_manager = vocab_manager
        self._metadata = metadata

        self._sections_bin_blob: Optional[np.memmap] = None
        self._sections_bin_view: Optional[memoryview] = None
        self._data_mmap: Optional[np.ndarray] = None
        self._data_kind: Optional[str] = None
        self._data_total_entries: Optional[int] = None
        self._variants_mmap: Optional[np.ndarray] = None

        self._stack: Optional[ExitStack] = None
        self._closed: bool = False

    # --- lifecycle -------------------------------------------------

    def __enter__(self) -> "BinarySession":
        assert_main_process()
        self._stack = ExitStack()
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._closed:
            return False
        self._closed = True
        stack = self._stack
        self._stack = None
        # Drop refs BEFORE stack unwinds so stray mid-unwind slice calls
        # see a torn-down session, not a half-closed handle.
        view = self._sections_bin_view
        self._sections_bin_view = None
        self._sections_bin_blob = None
        self._data_mmap = None
        self._data_kind = None
        self._data_total_entries = None
        self._variants_mmap = None
        if view is not None:
            # memoryview.release() drops the export so the underlying
            # bytes object can be GC'd without warning.
            view.release()
        if stack is not None:
            stack.close()
        return False

    def close(self) -> None:
        self.__exit__(None, None, None)

    # --- public slice methods --------------------------------------

    def load_matched(self, idx: int) -> MatchedFunction:
        _section, _offset, matched = self._load_matched_section_and_variants(idx)
        return matched

    def load_unmatched(self, idx: int) -> FunctionData:
        _section, _offset, fd = self._load_unmatched_record_and_section(idx)
        return fd

    # --- internal load + section helpers ---------------------------
    #
    # The matched + unmatched ``load_*`` paths share a need with the
    # batch-decode pipeline: BOTH want the parsed :class:`Section` (for
    # call_target walking) and the BIN section offset (for cycle keys)
    # alongside the per-function data. Factoring those reads into
    # dedicated private helpers keeps ``load_matched`` /
    # ``load_unmatched`` byte-for-byte semantically identical (single
    # source of truth) while exposing the section + offset to
    # ``_load_*_for_splice`` without re-parsing.

    def _slice_data_record(self, data_mmap, offset: int):
        """Slice + parse + egress-copy one record (memmap-view detach).

        The record at ``offset`` is self-describing: its header carries
        every geometry field the body parser needs. Reads at most
        :data:`MAX_HEADER_BYTES` for the header, derives the total via
        :func:`record_total_size`, and slices the body via
        :func:`extract_arrays_from_data`. Arrays are copied so they
        outlive the session's ``_data.bin`` memmap (see class docstring
        lifetime contract).

        Per-lookup integrity check: the parsed header's ``entry_idx``
        must be ``< total_entries`` (the value the trailer of the bin
        stamps; read once at session-open). A failure raises
        :class:`ValueError` with the exact wording
        ``corrupt file: <filename> did not pass validation``.
        """
        header, prefix_bytes = parse_binary_header(
            bytes(data_mmap[offset : offset + MAX_HEADER_BYTES])
        )
        if header.entry_idx >= (self._data_total_entries or 0):
            raise ValueError(
                f"corrupt file: {self._data_filename()} did not pass validation"
            )
        total = record_total_size(header)
        record_bytes = bytes(data_mmap[offset : offset + total])
        insn_rl, block_rl, tokens = extract_arrays_from_data(
            record_bytes, header, prefix_bytes
        )
        return (
            np.array(insn_rl, copy=True),
            np.array(block_rl, copy=True),
            np.array(tokens, copy=True),
        )

    def _data_filename(self) -> str:
        """Return the active arm's ``_data.bin`` filename for error msgs.

        Single chokepoint that derives the path from the cached
        ``_data_kind`` so corrupt-file errors report the right file
        (matched vs unmatched arm).
        """
        suffix = (
            "_unmatched_data.bin"
            if self._data_kind == "unmatched"
            else "_data.bin"
        )
        return f"{self._binary_name}{suffix}"

    def require_vocab_manager(self) -> Any:
        """Return the vocab manager, or RAISE if this session is vocab-less.

        Variant-prefix assembly (the train/decode path) REQUIRES the unified
        vocab to resolve a binary's variant records into axis token IDs. A
        ``None`` vocab here is a construction error -- the session was opened
        without ``vocab_manager=`` (``AlignedDataLoader`` threads it
        automatically). Length/graph-only consumers never call this and may
        run vocab-less by design.

        Failing loud is deliberate: decoding variant-prefixed rows with no
        vocab would SILENTLY drop the prefix and corrupt every training row
        with no error. Explicit ``raise`` (not ``assert``) because
        ``python -O`` strips asserts and would resurrect the silent footgun
        in an optimised run -- exactly where it matters most.
        """
        if self._vocab_manager is None:
            raise ValueError(
                "BinarySession was opened without a vocab_manager but variant "
                "decoding was requested. Construct BinaryDataset(..., "
                "vocab_manager=<unified vm>) or load via AlignedDataLoader "
                "(auto-loads the co-located unified_vocab.csv). A vocab-less "
                "session is length/graph-only and must not decode prefixes."
            )
        return self._vocab_manager

    def get_variant_by_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        # Swallow resolver errors to ``None`` -- parsers want a sentinel
        # for "no variant available", not an exception aborting a batch
        # over one bad section row. The vocab-less case stays tolerant HERE
        # (length/graph paths resolve refs without needing the vocab); the
        # train/decode path guards loudly upstream via
        # ``require_vocab_manager`` at ``batch_decode``.
        if not ref or self._vocab_manager is None:
            return None
        variants_mmap = self._open_variants()
        if variants_mmap is None:
            return None
        offset_to_filename = self._meta_get("offset_to_filename") or {}
        try:
            return _resolve_variant_by_ref(
                ref, self._vocab_manager, variants_mmap, offset_to_filename
            )
        except (TypeError, ValueError, KeyError, IndexError, AssertionError):
            return None

    # --- lazy openers ----------------------------------------------

    def _open_sections_bin(self) -> memoryview:
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

    def _parse_section_at(self, offset: int) -> Section:
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

    def _open_data(self, kind: str) -> np.ndarray:
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

    def _open_variants(self) -> Optional[np.ndarray]:
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

    # --- internal helpers ------------------------------------------

    def _meta_get(self, key: str) -> Any:
        if self._metadata is None:
            return None
        if hasattr(self._metadata, key):
            return getattr(self._metadata, key)
        if isinstance(self._metadata, dict):
            return self._metadata.get(key)
        return None

    def get_metadata(self, key: str) -> Any:
        """Public accessor for the session's metadata bag.

        Returns the value stored at ``key`` -- attribute-first, with
        ``dict`` fallback for the legacy dict-shaped metadata --
        mirroring :meth:`_meta_get`. ``None`` when the key is absent
        from both shapes, so callers needing a default substitute
        ``session.get_metadata(k) or <default>``.

        This is the single supported boundary for inspector / tooling
        layers that need to read sidecar artefacts (``line_to_name``,
        ``offset_to_filename``, etc.) without reaching into
        ``self._metadata`` directly.
        """
        return self._meta_get(key)
