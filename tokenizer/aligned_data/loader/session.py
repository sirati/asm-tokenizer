"""Per-binary, batch-scoped handle lifecycle.

Single concern: own the three file handles a ``BinaryDataset`` uses
while serving a batch of slicing operations on ONE binary
(``_sections.csv``/``_unmatched_sections.csv``, ``_data.bin``,
``_variants.bin``), and guarantee deterministic close of every
opened handle on exit.

Lazy opens + a single ``contextlib.ExitStack`` give the two required
properties at once: handles nobody touches stay closed; handles that
DO open are unwound (in reverse order) by the stack on ``__exit__``,
even when a mid-batch slice raises. ``__exit__`` is idempotent per
the plan's "BinarySession exception safety is mandatory" requirement.

This module does NOT load metadata (``metadata_loader``), parse
data-bin records (``aligned_data.io.parse_function_data_memmap``), or
own the variant-ref decoder body
(``variant_resolver.get_variant_by_ref``). Row→FunctionData glue lives
in ``_session_parsers``.
"""

from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..io import parse_function_data_memmap as _parse_function_data_memmap
from ._session_parsers import (
    arm_arrays,
    build_unmatched_function_data,
    parse_matched_section,
)
from .function_data import FunctionData
from .matched_function import MatchedFunction
from .variant_resolver import get_variant_by_ref as _resolve_variant_by_ref


def _close_memmap(mmap_obj) -> None:
    # Pin mmap release to the ExitStack instead of GC; long-lived
    # dataloader workers otherwise accumulate file descriptors.
    inner = getattr(mmap_obj, "_mmap", None)
    if inner is not None:
        try:
            inner.close()
        except Exception:
            pass


class BinarySession:
    """Context manager bundling the three per-binary handles.

    ``metadata`` is a pre-loaded bag (built by ``metadata_loader``
    in 5B and the variant-CSV reader in 5D). Accessed attribute-first,
    dict-fallback so 5B's final shape choice is transparent. Expected
    keys/attrs:

      * ``matched_arm``        — SectionArm: ``.starts``, ``.lengths``
      * ``unmatched_arm``      — SectionArm: ``.starts``, ``.lengths``,
                                 ``.func_names``, ``.section_starts``
      * ``offset_to_filename`` — ``dict[int, str]``
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
        self._stack = ExitStack()
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._closed:
            return False
        self._closed = True
        stack = self._stack
        self._stack = None
        # Drop handle refs BEFORE the stack unwinds so a stray
        # mid-unwind slice call sees a torn-down session, not a
        # half-closed handle.
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
        starts, lengths = arm_arrays(arm, "matched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        start = int(starts[idx])
        length = int(lengths[idx])
        sections = self._open_sections("matched")
        sections.seek(start + self._sections_content_offset)
        section_data = sections.read(length)
        data_mmap = self._open_data("matched")
        return parse_matched_section(
            section_data,
            data_slice=lambda o, l: _parse_function_data_memmap(data_mmap, o, l),
            resolve_ref=self.get_variant_by_ref,
        )

    def load_unmatched(self, idx: int) -> FunctionData:
        arm = self._meta_get("unmatched_arm")
        starts, lengths = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        start = int(starts[idx])
        length = int(lengths[idx])
        data_mmap = self._open_data("unmatched")
        insn_rl, block_rl, tokens = _parse_function_data_memmap(
            data_mmap, start, length
        )
        row = self._read_unmatched_row(arm, idx)
        return build_unmatched_function_data(
            row, idx, start, length, tokens, insn_rl, block_rl,
            resolve_ref=self.get_variant_by_ref,
        )

    def get_variant_by_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        # Pure resolver raises on bad input (malformed hex, missing filename,
        # vocab miss). Session swallows them all to ``None`` because the
        # parser callers want a "no variant available" sentinel, not an
        # exception that aborts a whole batch over one bad section row.
        if not ref:
            return None
        if self._vocab_manager is None:
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
        f = open(path, "r", newline="", encoding="ascii")
        self._stack.callback(f.close)
        # Transparent v2 prelude consumption — mirrors the legacy
        # ``_open_sections_csv``; prelude width is a property of the
        # open handle, not of cached metadata.
        start_pos = f.tell()
        first_line = f.readline()
        if first_line.startswith("version="):
            self._sections_content_offset = f.tell() - start_pos
        else:
            f.seek(start_pos)
            self._sections_content_offset = 0
        self._sections_handle = f
        self._sections_kind = kind
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
        section_starts = getattr(arm, "section_starts", None)
        sections_path = (
            self._base_path / f"{self._binary_name}_unmatched_sections.csv"
        )
        if section_starts is None or not sections_path.exists():
            return None
        sections = self._open_sections("unmatched")
        try:
            sections.seek(int(section_starts[idx]))
        except (IndexError, ValueError):
            return None
        line = sections.readline()
        if not line:
            return None
        row = next(csv.reader([line]), None)
        return row if row and len(row) == 6 else None

    def _meta_get(self, key: str) -> Any:
        if self._metadata is None:
            return None
        if hasattr(self._metadata, key):
            return getattr(self._metadata, key)
        if isinstance(self._metadata, dict):
            return self._metadata.get(key)
        return None
