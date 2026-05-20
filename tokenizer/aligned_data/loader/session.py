"""Per-binary, batch-scoped handle lifecycle.

Single concern: own the three file handles a ``BinaryDataset`` uses
while serving a batch of slicing operations on ONE binary
(``_sections.csv``/``_unmatched_sections.csv``, ``_data.bin``,
``_variants.bin``), and guarantee deterministic close on exit.

Lazy opens + a single ``contextlib.ExitStack``: handles nobody touches
stay closed; handles that DO open are unwound (in reverse order) by
the stack on ``__exit__``, even when a mid-batch slice raises.
``__exit__`` is idempotent.

This module does NOT load metadata (``metadata_loader``), parse
data-bin records (``aligned_data.io.parse_function_data_memmap``), or
own the variant-ref decoder (``variant_resolver``). Row→FunctionData
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

import csv
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..binary_format import (
    MAX_HEADER_BYTES,
    extract_arrays_from_data,
    parse_binary_header,
    record_total_size,
)
from ._worker_guard import assert_main_process
from ._session_parsers import (
    arm_arrays,
    build_unmatched_function_data,
    parse_matched_section,
)
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .metadata_loader import open_sections_csv
from .variant_resolver import get_variant_by_ref as _resolve_variant_by_ref


def _close_memmap(mmap_obj) -> None:
    # Pin mmap release to ExitStack vs GC -- long-lived workers leak fds.
    inner = getattr(mmap_obj, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:
            pass


class BinarySession:
    """Context manager bundling the three per-binary handles.

    ``metadata`` is a pre-loaded bag (built by ``metadata_loader``).
    Accessed attribute-first, dict-fallback. Expected keys/attrs:

      * ``matched_arm``        -- SectionArm: ``.starts`` (per-variant
                                  data-bin offsets), ``.csv_starts`` /
                                  ``.csv_lengths`` (per-function CSV
                                  locator), ``.func_names``
      * ``unmatched_arm``      -- SectionArm: ``.starts`` (per-function
                                  data-bin offsets), ``.func_names``,
                                  ``.section_starts``
      * ``offset_to_filename`` -- ``dict[int, str]``

    ``_data.bin`` records are self-describing -- their headers carry
    insn / block / token geometry -- so no companion ``lengths`` or
    ``is_overlong`` array crosses any boundary here.
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

        self._sections_handle: Optional[Any] = None
        self._sections_content_offset: int = 0
        self._sections_kind: Optional[str] = None
        self._data_mmap: Optional[np.ndarray] = None
        self._data_kind: Optional[str] = None
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
        self._sections_handle = None
        self._sections_content_offset = 0
        self._sections_kind = None
        self._data_mmap = None
        self._data_kind = None
        self._variants_mmap = None
        if stack is not None:
            stack.close()
        return False

    def close(self) -> None:
        self.__exit__(None, None, None)

    # --- public slice methods --------------------------------------

    def load_matched(self, idx: int) -> MatchedFunction:
        arm = self._meta_get("matched_arm")
        csv_starts, csv_lengths = arm_arrays(arm, "matched", self._binary_name)
        if idx >= len(csv_starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        csv_start = int(csv_starts[idx])
        csv_length = int(csv_lengths[idx])
        sections = self._open_sections("matched")
        sections.seek(csv_start + self._sections_content_offset)
        section_data = sections.read(csv_length)
        data_mmap = self._open_data("matched")
        func_names = getattr(arm, "func_names", None) or []
        func_name_override = func_names[idx] if idx < len(func_names) else None
        return parse_matched_section(
            section_data,
            func_name_override=func_name_override,
            data_slice=lambda o: self._slice_data_record(data_mmap, o),
            resolve_ref=self.get_variant_by_ref,
        )

    def load_unmatched(self, idx: int) -> FunctionData:
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        start = int(starts[idx])
        data_mmap = self._open_data("unmatched")
        insn_rl, block_rl, tokens = self._slice_data_record(data_mmap, start)
        return build_unmatched_function_data(
            self._read_unmatched_row(arm, idx),
            idx,
            self._unmatched_func_name(arm, idx),
            start,
            tokens, insn_rl, block_rl,
            resolve_ref=self.get_variant_by_ref,
        )

    def _slice_data_record(self, data_mmap, offset: int):
        """Slice + parse + egress-copy one record (memmap-view detach).

        The record at ``offset`` is self-describing: its header carries
        every geometry field the body parser needs. Reads at most
        :data:`MAX_HEADER_BYTES` for the header, derives the total via
        :func:`record_total_size`, and slices the body via
        :func:`extract_arrays_from_data`. Arrays are copied so they
        outlive the session's ``_data.bin`` memmap (see class docstring
        lifetime contract).
        """
        header, prefix_bytes = parse_binary_header(
            bytes(data_mmap[offset : offset + MAX_HEADER_BYTES])
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

    def _open_sections(self, kind: str):
        if self._stack is None:
            raise RuntimeError("BinarySession used outside its with-block")
        if self._sections_handle is not None:
            if self._sections_kind != kind:
                raise RuntimeError(
                    f"BinarySession already opened {self._sections_kind} "
                    f"sections; cannot switch to {kind} mid-session"
                )
            return self._sections_handle
        suffix = "_sections.csv" if kind == "matched" else "_unmatched_sections.csv"
        path = self._base_path / f"{self._binary_name}{suffix}"
        # Prelude validation + content-offset accounting belong to
        # ``open_sections_csv`` (single v1 ``# format=N`` consumer).
        f, content_offset = open_sections_csv(path)
        self._stack.callback(f.close)
        self._sections_handle = f
        self._sections_kind = kind
        self._sections_content_offset = content_offset
        return f

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
        )
        assert_data_bin_prelude(bytes(mmap[:DATA_BIN_PRELUDE_SIZE]), path=str(path))
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

    def _read_unmatched_row(self, arm: Any, idx: int) -> Optional[List[str]]:
        # Row layout (5 cells): line_no_b64, variant_refs, called_b64,
        # inlining, indexer_hex (post matched-arm restructuring).
        section_starts = getattr(arm, "section_starts", None)
        sections_path = (
            self._base_path / f"{self._binary_name}_unmatched_sections.csv"
        )
        if section_starts is None or not sections_path.exists():
            return None
        sections = self._open_sections("unmatched")
        try:
            sections.seek(
                int(section_starts[idx]) + self._sections_content_offset
            )
        except (IndexError, ValueError):
            return None
        line = sections.readline()
        if not line:
            return None
        row = next(csv.reader([line]), None)
        return row if row and len(row) == 5 else None

    def _unmatched_func_name(self, arm: Any, idx: int) -> str:
        # Row first cell is base64-of-line-number; ``metadata_loader``
        # resolves via the function-names sidecar and surfaces names on
        # ``arm.func_names`` in row order. Placeholder fallback on miss.
        names = getattr(arm, "func_names", None) or []
        return names[idx] if 0 <= idx < len(names) else f"unmatched_{idx}"

    def _meta_get(self, key: str) -> Any:
        if self._metadata is None:
            return None
        if hasattr(self._metadata, key):
            return getattr(self._metadata, key)
        if isinstance(self._metadata, dict):
            return self._metadata.get(key)
        return None
