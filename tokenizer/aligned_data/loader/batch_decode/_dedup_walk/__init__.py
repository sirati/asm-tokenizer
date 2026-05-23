"""Stage 4 step 1 package: per-row identity remap walk.

Re-exports the public entry :func:`apply_per_row_remap` and the
shared Category-partition constants. The package is split into one
submodule per concern:

* :mod:`._constants` — Category partition tables + shifted-id map +
  call-target-type -> Category map + ``NOT_FOUND_U16`` sentinel.
* :mod:`._helpers` — :func:`_surviving_in_stream_token_ids` (shared
  between FUNCTION + COUNTER dispatch).
* :mod:`._row_state` — :class:`_RowState` per-row mutable state.
* :mod:`._function_remap` — ALG-3 FUNCTION-category dedup.
* :mod:`._counter_bump` — ALG-4 COUNTER-category offset bump.
* :mod:`._prepend_slot` — ALG-9 prepend self-counter write +
  LOCAL_FUNC root seed.
* :mod:`._apply` — :func:`apply_per_row_remap` public entry + the
  batch-row loop.

See :mod:`._apply` for the full algorithmic docstring.
"""

from __future__ import annotations

from ._apply import apply_per_row_remap
from ._constants import (
    COUNTER_CATEGORIES,
    FUNCTION_CATEGORIES,
    NOT_FOUND_U16,
    _CALL_TARGET_TYPE_TO_CATEGORY,
    _CATEGORY_TO_SHIFTED_ID,
)


__all__ = [
    "COUNTER_CATEGORIES",
    "FUNCTION_CATEGORIES",
    "NOT_FOUND_U16",
    "_CALL_TARGET_TYPE_TO_CATEGORY",
    "_CATEGORY_TO_SHIFTED_ID",
    "apply_per_row_remap",
]
