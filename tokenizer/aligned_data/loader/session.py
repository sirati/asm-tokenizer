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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..binary_format import (
    MAX_HEADER_BYTES,
    extract_arrays_from_data,
    parse_binary_header,
    record_total_size,
)
from ..matched_sections_bin import Section, parse_section_bin
from ..memmap_format import MATCHED_SECTIONS_BIN_PRELUDE_SIZE
from ._sections_bin_walk import read_sections_bin_blob
from ._session_parsers import (
    arm_arrays,
    build_unmatched_function_data,
    parse_matched_section,
)
from ._session_helpers import _BinarySessionHelpersMixin
from ._worker_guard import assert_main_process
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .variant_resolver import get_variant_by_ref as _resolve_variant_by_ref


def _close_memmap(mmap_obj) -> None:
    # Pin mmap release to ExitStack vs GC -- long-lived workers leak fds.
    inner = getattr(mmap_obj, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:
            pass


class BinarySession(_BinarySessionHelpersMixin):
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

    ``_data.bin`` records are self-describing -- their headers carry
    insn / block / token geometry -- so no companion ``lengths`` or
    ``is_overlong`` array crosses any boundary here. Section parsing
    happens against an in-memory ``memoryview`` of
    ``<binary>_sections.bin``; the BIN's prelude is validated on first
    open and a per-session memoryview is held until ``__exit__``.
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

        self._sections_bin_blob: Optional[bytes] = None
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

    def _load_matched_section_and_variants(
        self, idx: int
    ) -> Tuple[Section, int, MatchedFunction]:
        """Parse the matched section at ``idx`` + build all its variants.

        Returns ``(section, section_offset, MatchedFunction)``. The
        ``Section`` is the parsed BIN catalog entry (call_targets +
        variant blocks); ``section_offset`` is the BIN byte offset
        from ``bin_starts[idx]``. Shared by :py:meth:`load_matched`
        and the batch-decode pipeline.
        """
        arm = self._meta_get("matched_arm")
        bin_starts, _bin_lengths = arm_arrays(arm, "matched", self._binary_name)
        if idx >= len(bin_starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        section_offset = int(bin_starts[idx])
        section = self._parse_section_at(section_offset)
        data_mmap = self._open_data("matched")
        func_names = getattr(arm, "func_names", None) or []
        if idx >= len(func_names):
            raise IndexError(
                f"matched arm func_names short of index {idx} "
                f"(have {len(func_names)})"
            )
        func_name = func_names[idx]
        matched = parse_matched_section(
            section,
            func_name=func_name,
            data_slice=lambda o: self._slice_data_record(data_mmap, o),
            resolve_ref=self.get_variant_by_ref,
        )
        return section, section_offset, matched

    def _load_unmatched_record_and_section(
        self, idx: int
    ) -> Tuple[Section, int, FunctionData]:
        """Parse the unmatched record at ``idx`` + its owning section.

        Returns ``(section, section_offset, FunctionData)`` where
        ``section_offset`` is the BIN byte offset of the owning section
        (NOT the per-record ``_unmatched_data.bin`` offset). Shared by
        :py:meth:`load_unmatched` and the batch-decode pipeline.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        start = int(starts[idx])
        data_mmap = self._open_data("unmatched")
        insn_rl, block_rl, tokens = self._slice_data_record(data_mmap, start)
        section, section_offset = self._unmatched_section_for_record(
            arm, idx, start
        )
        line_to_name = self._meta_get("line_to_name") or {}
        fd = build_unmatched_function_data(
            section,
            self._unmatched_func_name(arm, idx),
            start,
            tokens, insn_rl, block_rl,
            resolve_ref=self.get_variant_by_ref,
            line_to_name=line_to_name,
        )
        return section, section_offset, fd

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

    def get_variant_by_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        # Swallow resolver errors to ``None`` -- parsers want a sentinel
        # for "no variant available", not an exception aborting a batch
        # over one bad section row.
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
        """Lazy-load the per-binary section catalog as a memoryview.

        The BIN is small relative to ``_data.bin`` (sections carry
        header + call_targets + per-variant blocks but no token
        payload), so we read the whole file into memory once per
        session rather than mmap-ing it; the memoryview keeps parser
        slicing zero-copy. Prelude is validated on first open.
        """
        if self._stack is None:
            raise RuntimeError("BinarySession used outside its with-block")
        if self._sections_bin_view is not None:
            return self._sections_bin_view
        # ``matched_arm`` and ``unmatched_arm`` share the same BIN file;
        # which arm's path we resolve doesn't matter, but we walk through
        # the conventional per-binary filename for clarity.
        path = self._base_path / f"{self._binary_name}_sections.bin"
        # Pin the bytes so the view stays valid for the session lifetime.
        raw, view = read_sections_bin_blob(path)
        self._sections_bin_blob = raw
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
            assert_data_bin_prelude,
            read_data_bin_trailer,
        )
        assert_data_bin_prelude(bytes(mmap[:DATA_BIN_PRELUDE_SIZE]), path=str(path))
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

    def _unmatched_section_for_record(
        self, arm: Any, idx: int, start: int
    ) -> Tuple[Section, int]:
        """Resolve the BIN section that owns the per-record ``start``.

        The unmatched index is per-RECORD (one entry per
        ``_unmatched_data.bin`` record). The arm pre-computes
        ``record_to_section_idx[idx]`` at load time (O(M) once over
        the BIN walk); this dispatch is O(1) for the section-idx
        lookup plus O(log K) for the slot-base ``np.searchsorted``
        — negligible compared to the BIN parse this dispatch then
        triggers.

        Returns ``(section, section_offset)`` -- the parsed section and
        its BIN byte offset, so callers (notably the batch-decode pipeline)
        can use the offset as a cycle key without re-deriving it.

        Sanity-checks the section's variant at the matching slot:
        its ``data_offset_shifted << 4`` must equal ``start``. A
        mismatch surfaces as :class:`ValueError`, not a silent wrong-
        section return, since it indicates the unmatched_index.bin /
        sections.bin drifted apart at build time.
        """
        section_idx = self._unmatched_section_idx(arm, idx)
        section_starts = getattr(arm, "section_starts", None)
        section_offset = int(section_starts[section_idx])
        section = self._parse_section_at(section_offset)
        slot = idx - self._unmatched_record_slot_base(arm, section_idx)
        variant = section.variants[slot]
        if (variant.data_offset_shifted << 4) != start:
            raise ValueError(
                f"unmatched section[{section_idx}] variant [{slot}] "
                f"data_offset {variant.data_offset_shifted << 4} does "
                f"not match record offset {start} (record idx={idx})"
            )
        return section, section_offset

    def _unmatched_func_name(self, arm: Any, idx: int) -> str:
        """Per-record function name via the pre-cached mapping.

        Falls through to the ``unmatched_<idx>`` sentinel only when
        the section index points beyond ``func_names`` -- this
        indicates the function-names sidecar drifted from the BIN
        catalog at build time and is normally caught earlier by the
        arm-load FID-resolution check.
        """
        names = getattr(arm, "func_names", None) or []
        section_idx = self._unmatched_section_idx(arm, idx)
        if 0 <= section_idx < len(names):
            return names[section_idx]
        return f"unmatched_{idx}"

    def _unmatched_section_idx(self, arm: Any, idx: int) -> int:
        """Look up the per-record -> per-section index via the arm's
        pre-cached mapping. Raises :class:`IndexError` on out-of-range
        ``idx`` with the same wording the legacy section walk used.
        """
        mapping = getattr(arm, "record_to_section_idx", None)
        if mapping is None or len(mapping) == 0:
            raise IndexError(
                f"unmatched arm has no record_to_section_idx for record "
                f"{idx} on binary {self._binary_name}"
            )
        if idx < 0 or idx >= len(mapping):
            raise IndexError(
                f"unmatched record idx={idx} out of bounds (have "
                f"{len(mapping)} records on binary {self._binary_name})"
            )
        return int(mapping[idx])

    def _unmatched_record_slot_base(self, arm: Any, section_idx: int) -> int:
        """First record index belonging to section ``section_idx``.

        Derived once per call from the pre-cached mapping; the slot
        within the section is ``idx - base``. ``np.searchsorted`` on
        the contiguous-section mapping is O(log K) — the same
        derivation the legacy section-walk used to accumulate via the
        ``consumed`` counter, just sourced from the mapping instead
        of re-parsing the BIN.
        """
        mapping = arm.record_to_section_idx
        # np.searchsorted on the contiguous-section mapping returns
        # the first record index whose section_idx >= target; for an
        # exact match this is exactly the section's base record.
        return int(np.searchsorted(mapping, section_idx, side="left"))

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
