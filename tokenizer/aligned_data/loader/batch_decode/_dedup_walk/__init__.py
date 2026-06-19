"""Stage 4 step 1 package: per-row identity remap walk.

Re-exports the public entry :func:`apply_per_row_remap` and the
shared Category-partition constants. The ALG-3/4/9 remap LOGIC lives in
the Rust kernel ``dedup_hashmap.apply_remap_walk``; this package owns
the object-tree -> flat-int-array ADAPTER + the FID-sidecar pass-2
emission that the kernel's per-row output feeds. The package is split
into one submodule per concern:

* :mod:`._constants` — Category partition tables + shifted-id map +
  call-target-type -> Category map + per-partition slot codes +
  ``NOT_FOUND_U16`` sentinel.
* :mod:`._helpers` — :func:`_surviving_in_stream_token_ids` (the
  per-call-target in-stream identity-band token id extractor).
* :mod:`._flat_extract` — the object-tree -> flat-int-array extractor
  feeding the Rust kernel (:func:`extract_flat_remap_inputs`).
* :mod:`._apply` — :func:`apply_per_row_remap` public entry: extract
  flat arrays, call the kernel, emit the row-keyed FID sidecar.

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
