"""Filter modal subpackage -- per-axis-value enable/disable.

Re-exports the public surface flat: pure-data config + result sum type
+ filter pass + value-discovery helpers live in :mod:`._config` and
:mod:`._values`; the Textual modal screen lives in :mod:`._dialog`. The
dialog re-export is deferred via PEP 562 ``__getattr__`` so importing
this subpackage does NOT pull in :mod:`textual` (mirrors the parent
``_app`` package's lazy-import discipline + the sibling :mod:`._order`
subpackage).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._config import (
    FilterAccepted,
    FilterCancelled,
    FilterConfig,
    FilterResult,
    MISSING_VALUE_TOKEN,
    apply_filter,
    function_has_passing_variants,
    missing_value_token,
)
from ._values import discover_all_axis_values, discover_axis_values


__all__ = [
    "FilterAccepted",
    "FilterCancelled",
    "FilterConfig",
    "FilterDialog",
    "FilterResult",
    "MISSING_VALUE_TOKEN",
    "apply_filter",
    "discover_all_axis_values",
    "discover_axis_values",
    "function_has_passing_variants",
    "missing_value_token",
]


if TYPE_CHECKING:
    from ._dialog import FilterDialog


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export for the Textual-dependent dialog."""
    if name == "FilterDialog":
        from ._dialog import FilterDialog

        return FilterDialog
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
