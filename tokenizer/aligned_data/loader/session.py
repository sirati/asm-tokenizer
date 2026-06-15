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
from ..index_format import ALIGNMENT_SHIFT
from ..matched_sections_bin import Section, parse_section_bin
from ..memmap_format import MATCHED_SECTIONS_BIN_PRELUDE_SIZE
from ._sections_bin_walk import read_sections_bin_blob
from ._session_parsers import (
    arm_arrays,
    build_unmatched_function_data,
    parse_matched_section,
    parse_matched_variant,
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

    def _matched_section_meta(self, idx: int) -> Tuple[Section, int]:
        """Parse a matched section's BIN catalog entry only (no bodies).

        Returns ``(section, section_offset)`` -- the same parsed
        :class:`Section` and BIN byte offset
        :py:meth:`_load_matched_section_and_variants` produces, but
        WITHOUT touching ``_data.bin`` (no per-variant body materialised).
        The callee walk's once-only inclusion decision keys solely on the
        callee ``section_offset`` and the parent's per-call J-resolution,
        so the body load is deferred to the survivors via
        :py:meth:`_load_matched_variant_body`.
        """
        arm = self._meta_get("matched_arm")
        bin_starts, _bin_lengths = arm_arrays(arm, "matched", self._binary_name)
        if idx >= len(bin_starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        section_offset = int(bin_starts[idx])
        section = self._parse_section_at(section_offset)
        return section, section_offset

    def _load_matched_variant_body(
        self, idx: int, variant_index: int, section: Section
    ) -> FunctionData:
        """Load ONE matched section variant body from ``_data.bin``.

        ``section`` is the already-parsed BIN catalog entry the caller
        obtained from :py:meth:`_matched_section_meta` (threaded through
        the callee walk's :class:`ResolvedCalleeMeta`), so this load does
        NOT re-parse ``_sections.bin`` -- it materialises only
        ``section.variants[variant_index]`` via the shared
        :func:`parse_matched_variant`. That is the same single-variant
        parse :py:meth:`_load_matched_section_and_variants` runs per
        variant, so the returned body is byte-identical to that path's
        ``MatchedFunction.variants[variant_index]``. ``idx`` is retained
        for the O(1) ``func_names[idx]`` lookup (the name carried on the
        body) and its bounds check. Raises :class:`IndexError` if
        ``variant_index`` is out of range.
        """
        arm = self._meta_get("matched_arm")
        if variant_index < 0 or variant_index >= len(section.variants):
            raise IndexError(
                f"matched function idx={idx} has {len(section.variants)} "
                f"variants; variant_index {variant_index} out of range"
            )
        func_names = getattr(arm, "func_names", None) or []
        if idx >= len(func_names):
            raise IndexError(
                f"matched arm func_names short of index {idx} "
                f"(have {len(func_names)})"
            )
        data_mmap = self._open_data("matched")
        return parse_matched_variant(
            section,
            section.variants[variant_index],
            func_name=func_names[idx],
            data_slice=lambda o: self._slice_data_record(data_mmap, o),
            resolve_ref=self.get_variant_by_ref,
        )

    def _load_unmatched_record_and_section(
        self, idx: int
    ) -> Tuple[Section, int, FunctionData]:
        """Parse the unmatched record at ``idx`` + its owning section.

        Returns ``(section, section_offset, FunctionData)`` where
        ``section_offset`` is the BIN byte offset of the owning section
        (NOT the per-record ``_unmatched_data.bin`` offset). Shared by
        :py:meth:`load_unmatched` and the batch-decode pipeline.

        The per-record ``idx`` maps to a ``(section base record, variant
        slot)`` pair via the arm's ``record_to_section_idx`` mapping; the
        body load delegates to :py:meth:`_load_unmatched_variant_body`,
        which slices the variant block's OWN
        ``data_offset_shifted << ALIGNMENT_SHIFT``. The writer emits the
        per-record index entries in encounter order but sorts the section's
        variant blocks by ``variant_ref_offset``, so the positional
        ``starts[idx]`` and the slot-J variant block are NOT guaranteed to
        coincide; slicing by the variant's own offset is the single robust
        source of truth (symmetric with the matched arm and the callee
        walk), with no residual dependence on emit-order==vref-order.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        section, section_offset = self._unmatched_section_for_record(arm, idx)
        # Per-record -> per-variant slot inside the owning section.
        # Unmatched sections store one record per variant; the slot is
        # the offset from the section's first-record idx in the arm's
        # ``record_to_section_idx`` mapping.
        section_idx = self._unmatched_section_idx(arm, idx)
        base = self._unmatched_record_slot_base(arm, section_idx)
        variant_slot = idx - base
        fd = self._load_unmatched_variant_body(base, variant_slot, section)
        return section, section_offset, fd

    def _load_unmatched_variant_body(
        self, idx: int, variant_index: int, section: Section
    ) -> FunctionData:
        """Load ONE unmatched section variant body, reusing the section.

        ``section`` is the already-parsed owning section the caller
        obtained from :py:meth:`_unmatched_section_meta` (threaded through
        the callee walk's :class:`ResolvedCalleeMeta`), so this load does
        NOT re-derive it via :py:meth:`_unmatched_section_for_record` (no
        ``_sections.bin`` re-parse).

        The data record is sliced at ``section.variants[variant_index]``'s
        OWN ``data_offset_shifted << ALIGNMENT_SHIFT`` -- the SAME way the
        matched arm slices (:func:`parse_matched_variant`). Unmatched
        sections store one DISTINCT body record per variant, so loading by
        the variant block's own offset (rather than the positional
        ``starts[base + variant_index]``) splices variant-``variant_index``'s
        body regardless of whether the index-entry order and the
        variant-block order coincide -- removing the silent dependence on
        the writer's emit-order==vref-order lock-step. ``idx`` is retained
        for the section-keyed ``func_names`` lookup (the name is a section
        property, shared by every variant) and its bounds check. Raises
        :class:`IndexError` if ``variant_index`` is out of range.

        For ``variant_index == 0`` this is byte-identical to the legacy
        first-record load: the section's first variant block carries the
        same ``data_offset_shifted`` as ``starts[base]``.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        if variant_index < 0 or variant_index >= len(section.variants):
            raise IndexError(
                f"unmatched section idx={idx} has {len(section.variants)} "
                f"variants; variant_index {variant_index} out of range"
            )
        start = (
            section.variants[variant_index].data_offset_shifted << ALIGNMENT_SHIFT
        )
        data_mmap = self._open_data("unmatched")
        insn_rl, block_rl, tokens = self._slice_data_record(data_mmap, start)
        line_to_name = self._meta_get("line_to_name") or {}
        return build_unmatched_function_data(
            section,
            self._unmatched_func_name(arm, idx),
            start,
            tokens, insn_rl, block_rl,
            variant_slot=variant_index,
            resolve_ref=self.get_variant_by_ref,
            line_to_name=line_to_name,
        )

    def _unmatched_section_meta(self, idx: int) -> Tuple[Section, int]:
        """Parse an unmatched record's owning section only (no body).

        Returns ``(section, section_offset)`` for the record at per-record
        ``idx`` -- the same parsed :class:`Section` and BIN section offset
        :py:meth:`_load_unmatched_record_and_section` produces, but
        WITHOUT slicing the ``_unmatched_data.bin`` record body. The callee
        walk defers the body load (sliced at the J-resolved variant block's
        own offset, via :py:meth:`_load_unmatched_variant_body`) to the
        surviving pairs.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        return self._unmatched_section_for_record(arm, idx)

    def _load_unmatched_section_and_all_variants(
        self, idx: int
    ) -> Tuple[Section, int, list]:
        """Parse the unmatched section at ``idx`` + build every variant's FunctionData.

        Mirrors :py:meth:`_load_matched_section_and_variants` for the
        unmatched arm: returns ``(section, section_offset, list[FunctionData])``
        where the per-variant list is parallel to ``section.variants``.
        Unmatched sections store one record per variant; each body is
        sliced at its variant block's own
        ``data_offset_shifted << ALIGNMENT_SHIFT`` via the shared
        :py:meth:`_load_unmatched_variant_body` (the single owner of the
        slice), so the result never depends on the writer's encounter-order
        index entries lining up with the sorted variant blocks. ``idx``
        MUST be the section's first-record idx (the value
        :py:meth:`_idx_for_section_offset` returns for the unmatched arm);
        a non-base record idx raises :class:`ValueError`.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        # Pin ``idx`` to the section's first-record slot. Loading a
        # non-base record into the per-section variants list would lose
        # the preceding slots, breaking the parallel
        # ``section.variants`` <-> returned list contract.
        section_idx = self._unmatched_section_idx(arm, idx)
        base = self._unmatched_record_slot_base(arm, section_idx)
        if idx != base:
            raise ValueError(
                f"unmatched section variants require first-record idx "
                f"(section[{section_idx}] base={base}); got idx={idx}"
            )
        section, section_offset = self._unmatched_section_for_record(arm, base)
        variants = [
            self._load_unmatched_variant_body(base, slot, section)
            for slot in range(len(section.variants))
        ]
        return section, section_offset, variants

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

    def _unmatched_section_for_record(
        self, arm: Any, idx: int
    ) -> Tuple[Section, int]:
        """Resolve the BIN section that owns the per-record ``idx``.

        The unmatched index is per-RECORD (one entry per
        ``_unmatched_data.bin`` record). The arm pre-computes
        ``record_to_section_idx[idx]`` at load time (O(M) once over
        the BIN walk); this dispatch is O(1) for the section-idx
        lookup — negligible compared to the BIN parse it then triggers.

        Returns ``(section, section_offset)`` -- the parsed section and
        its BIN byte offset, so callers (notably the batch-decode pipeline)
        can use the offset as a cycle key without re-deriving it.

        No positional ``starts[idx] == variant.data_offset_shifted << 4``
        drift check is performed: the body load slices each variant block's
        OWN ``data_offset_shifted`` (see :py:meth:`_load_unmatched_variant_body`),
        so the per-record index-entry offset is never used to locate a body.
        The writer emits index entries in encounter order but sorts variant
        blocks by ``variant_ref_offset``, so the two orders legitimately
        differ; an equality assertion against ``starts[idx]`` would FALSELY
        reject correctly-written corpora rather than guard corruption.
        """
        section_idx = self._unmatched_section_idx(arm, idx)
        section_starts = getattr(arm, "section_starts", None)
        section_offset = int(section_starts[section_idx])
        section = self._parse_section_at(section_offset)
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
