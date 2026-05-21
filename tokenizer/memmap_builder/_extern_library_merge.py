"""Shared first-wins merge of per-variant ``extern_libraries`` dicts.

Both pass-1 walkers — the matched arm in :mod:`passes` and the
unmatched arm in :mod:`_pass2` — need to union the per-variant
``extern_libraries`` dicts (function_name → library) of a function
group into a single per-function authoritative map. The semantics are
identical across arms: first writer wins on a same-name-different-
library conflict, and a warning surfaces via the module logger so the
builder bug is visible without polluting the structured
``<binary>.error.log`` TSV (which is reserved for ``ALLOWED_REASONS``
cap-overflow rows; see :mod:`error_log`).

Centralising the merge here is the single source of truth for the
union policy: the only thing that differs between callers is how they
arrive at the iterable of per-variant dicts (matched walks
``Dict[variant_index, ParsedRecord]`` sorted by index; unmatched walks
a flat list of per-variant entry dicts).
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def merge_extern_libraries(
    per_variant_dicts: "Iterable[dict[str, str]]",
    *,
    func_name: str,
) -> "dict[str, str]":
    """Merge per-variant extern-library dicts, first-wins on conflict.

    Iterates ``per_variant_dicts`` once. For each ``name → library``
    pair: record it on first sight; on a same-name-different-library
    conflict, surface one ``logger.warning`` per occurrence naming the
    function, the extern name, the library that was kept (first
    encountered), and the library that was dropped, then continue with
    the first-wins value.

    Callers control iteration order (variant-index-sorted for
    determinism); the helper makes no assumption about per-variant
    ordering and produces the same merged map regardless of repeats.
    The returned dict is the per-function authoritative mapping
    consumed downstream by the BIN's call_target emitter.
    """
    merged: "dict[str, str]" = {}
    for per_variant in per_variant_dicts:
        for name, library in per_variant.items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = library
            elif existing != library:
                logger.warning(
                    "function %s extern library %s mismatch across variants: "
                    "kept %s, dropped %s",
                    func_name,
                    name,
                    existing,
                    library,
                )
    return merged
