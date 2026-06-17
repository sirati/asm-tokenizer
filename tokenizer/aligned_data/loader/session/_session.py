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
from .._session_helpers import _BinarySessionHelpersMixin
from .._worker_guard import assert_main_process
from ..function_data import FunctionData
from ..matched_function import MatchedFunction
from ..variant_resolver import get_variant_by_ref as _resolve_variant_by_ref
from ._handles import _HandlesMixin
from ._matched_load import _MatchedLoadMixin
from ._unmatched_load import _UnmatchedLoadMixin


class BinarySession(
    _HandlesMixin,
    _MatchedLoadMixin,
    _UnmatchedLoadMixin,
    _BinarySessionHelpersMixin,
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
