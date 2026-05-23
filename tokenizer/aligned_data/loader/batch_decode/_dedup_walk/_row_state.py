"""Per-row dedup state.

Single concern: the mutable per-row state owned by the per-row dedup
walk (:mod:`._apply`). Holds per-FUNCTION-Category fresh-id counters,
per-COUNTER-Category running offsets, and the optional reverse mapping
(``counter_id -> function_name_ptr``) for the FID sidecar.

The three :class:`HashMapU32U16` instances live OUTSIDE this class —
caller-provided (:func:`apply_per_row_remap` reuses them across rows
via ``clean()`` per the plan's hot-path discipline).
"""

from __future__ import annotations

from typing import Optional

from tokenizer.tokens import Category

from ._constants import COUNTER_CATEGORIES, FUNCTION_CATEGORIES


__all__ = ["_RowState"]


class _RowState:
    """Mutable per-row dedup state.

    Lives for the duration of one row's walk; reset at the top of each
    row. Holds:

    * Per-FUNCTION-Category ``next_fresh_id`` counters.
    * Per-FUNCTION-Category reverse-mapping list (``counter_id ->
      function_name_ptr``) for the optional fid sidecar collection.
    * Per-COUNTER-Category running offsets.

    The 3 ``HashMapU32U16`` instances are NOT owned by ``_RowState`` —
    they are caller-provided (the parent reuses them across rows via
    ``clean()`` per the plan).
    """

    __slots__ = (
        "next_fresh_id",
        "counter_offset",
        "fid_inverse",
    )

    def __init__(self, collect_fid_sidecar: bool) -> None:
        self.next_fresh_id: dict[Category, int] = {
            cat: 0 for cat in FUNCTION_CATEGORIES
        }
        self.counter_offset: dict[Category, int] = {
            cat: 0 for cat in COUNTER_CATEGORIES
        }
        # ``counter_id -> function_name_ptr`` per FUNCTION Category;
        # only populated when the sidecar is requested. Indices align
        # with counter ids because fresh ids are minted densely.
        self.fid_inverse: Optional[dict[Category, list[int]]] = (
            {cat: [] for cat in FUNCTION_CATEGORIES}
            if collect_fid_sidecar
            else None
        )
