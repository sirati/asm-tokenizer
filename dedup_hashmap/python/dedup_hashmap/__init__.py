"""``dedup_hashmap`` — typed integer-keyed hashmaps for Python.

The package wraps a PyO3 extension that emits one ``HashMap<K><V>``
class per Cartesian product of supported integer key dtypes and
integer/float value dtypes (see :mod:`dedup_hashmap._native`). The
generated classes are exposed at the package top-level so existing
callers that import ``dedup_hashmap.HashMapU32U16`` keep working
unchanged after the package restructure.

On top of the raw classes, the package ships :class:`IntEnumHashMap`,
a pure-Python typed facade that auto-selects the narrowest backing
dtype which fits a caller-supplied :class:`enum.IntEnum` value range
and translates the scalar API to/from enum members at the boundary.
See :mod:`dedup_hashmap.typed` for the design rationale.
"""

from __future__ import annotations

from . import _native as _native
from ._native import *  # noqa: F401,F403 — re-export every generated class
from ._native import (
    LiveAdjacencyKernel,
    OnceOnlyInclusionKernel,
    apply_remap_walk,
    build_carrier_signs_kernel,
    build_flat_segments_kernel,
    build_identity_carriers_kernel,
    build_inline_bytes_kernel,
    build_node_ct_csr_kernel,
    build_number_idx_2d_kernel,
    compute_row_inclusions_kernel,
    segment_distinct_count,
)
from .typed import IntDtype, IntEnumHashMap, PlainBool, PlainInt


# ``_native.__all__`` is set by PyO3 and lists every generated
# ``HashMap<K><V>`` class; extend it with the pure-Python typed surface
# plus the free ``segment_distinct_count`` kernel so
# ``from dedup_hashmap import *`` covers all layers.
__all__ = list(_native.__all__) + [
    "IntDtype",
    "IntEnumHashMap",
    "LiveAdjacencyKernel",
    "OnceOnlyInclusionKernel",
    "PlainBool",
    "PlainInt",
    "apply_remap_walk",
    "build_carrier_signs_kernel",
    "build_flat_segments_kernel",
    "build_identity_carriers_kernel",
    "build_inline_bytes_kernel",
    "build_node_ct_csr_kernel",
    "build_number_idx_2d_kernel",
    "compute_row_inclusions_kernel",
    "segment_distinct_count",
]
