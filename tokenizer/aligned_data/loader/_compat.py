"""Cross-batch import shims for session.py.

When subtask 5A (``aligned_data.io`` split) and 5D (``variant_resolver``
module) have not yet merged into this worktree, the symbols
``parse_function_data_memmap`` and ``get_variant_by_ref`` are absent.
This module re-exports the real symbols when available and falls back
to inline implementations of the documented contracts otherwise so the
session module and its tests still work in isolation.

The integrator (5G) should delete this file once 5A and 5D are
merged and ``session.py`` imports from the canonical locations
directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

MISSING_INTERFACES: List[str] = []


try:
    from ..io import parse_function_data_memmap
except ImportError:  # pragma: no cover — covered indirectly
    MISSING_INTERFACES.append("aligned_data.io.parse_function_data_memmap")

    def parse_function_data_memmap(handle, offset, length):
        # 5A contract: ``handle`` is an open uint8 memmap of ``_data.bin``;
        # slice and parse via the existing header path.
        from ..io import parse_function_data_header

        return parse_function_data_header(handle[offset:offset + length])


try:
    from .variant_resolver import get_variant_by_ref
except ImportError:  # pragma: no cover
    MISSING_INTERFACES.append(
        "aligned_data.loader.variant_resolver.get_variant_by_ref"
    )

    def get_variant_by_ref(
        ref: str,
        vocab_manager: Any,
        variants_mmap: np.ndarray,
        offset_to_filename: Dict[int, str],
    ) -> Optional[Dict[str, Any]]:
        # 5D contract: pure function on its inputs; returns the
        # decoded dict (incl. ``filename`` + ``variant_tokens``) or
        # ``None`` on malformed/missing input.
        from tokenizer.variant_tokens.encoder import decode_record
        from tokenizer.variant_tokens.record import read_record

        if vocab_manager is None or variants_mmap is None:
            return None
        try:
            offset = int(ref, 16)
        except (TypeError, ValueError):
            return None
        try:
            tokens = read_record(variants_mmap, offset)
        except (AssertionError, IndexError, ValueError):
            return None
        decoded = decode_record(tokens, vocab_manager)
        decoded["filename"] = offset_to_filename.get(offset, "")
        decoded["variant_tokens"] = np.array(tokens, copy=True)
        return decoded
