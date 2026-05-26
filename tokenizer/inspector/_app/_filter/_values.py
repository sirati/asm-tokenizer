"""Per-axis value discovery for the filter dialog.

Single concern: given a sequence of currently-loaded
:class:`RenderedVariant`s + a candidate axis tuple, return the sorted,
deduplicated set of values seen for each axis. The dialog renders one
checkable row per ``(axis, value)`` pair so the user can toggle that
particular value off.

The axis tuple itself is built by the existing
:mod:`tokenizer.inspector._app._order._axes` helpers (canonical-5 +
EXTRA_META discovery): the filter reuses the same axis surface as the
order dialog so a user-extended set (e.g. a new sidecar key) shows up
in both modals at once.

Missing-axis values map onto :data:`MISSING_VALUE_TOKEN` so the dialog
renders a literal ``"?"`` row when at least one loaded variant has no
value on that axis -- the row is checkable like every other value
(unchecking it filters out the variants whose axis is absent).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import natsort

from tokenizer.inspector._render._protocol import RenderedVariant

from .._order import AxisDescriptor, extract_axis_value
from ._config import MISSING_VALUE_TOKEN


__all__ = [
    "discover_axis_values",
    "discover_all_axis_values",
]


# Pre-built natsort key generator: same idiom as :mod:`._order._grouping`
# so a value sort like ``v9 < v10`` is consistent between the two modals.
_natural_sort_key = natsort.natsort_keygen()


def discover_axis_values(
    axis: AxisDescriptor,
    variants: Iterable[RenderedVariant],
) -> tuple[str, ...]:
    """Sorted, deduplicated values seen on ``axis`` across ``variants``.

    Missing values surface as the literal :data:`MISSING_VALUE_TOKEN`
    so the dialog row stays addressable (the user can disable
    missing-value variants explicitly). Natsort ordering places the
    ``"?"`` bucket last by sinking it via natsort's own ordering -- ``?``
    sorts AFTER digits + letters in the standard table, so no special
    last-bucket prefix is needed (cross-checked against natsort 7.x).
    """
    seen: set[str] = set()
    has_missing = False
    for rv in variants:
        raw = extract_axis_value(axis, rv)
        if raw is None:
            has_missing = True
        else:
            seen.add(str(raw))
    out = sorted(seen, key=_natural_sort_key)
    if has_missing:
        out.append(MISSING_VALUE_TOKEN)
    return tuple(out)


def discover_all_axis_values(
    axes: Sequence[AxisDescriptor],
    variants: Iterable[RenderedVariant],
) -> dict[AxisDescriptor, tuple[str, ...]]:
    """Per-axis value tuple for every axis in ``axes``.

    Single pass over ``variants`` per axis (a single combined pass is
    possible but the per-axis loop reads cleaner + the variant counts
    are small enough that the extra iterations don't matter).
    """
    # Materialise once so the per-axis iteration can re-traverse.
    variants_list = list(variants)
    return {
        axis: discover_axis_values(axis, variants_list)
        for axis in axes
    }
