"""Order modal subpackage -- axis model + grouping pass + dialog widget.

Re-exports the public surface flat: the typed axis primitives + sum
types live in :mod:`._axes`; the grouping pass + group-node dataclass
live in :mod:`._grouping`; the Textual modal screen + reorderable
SelectionList live in :mod:`._dialog`. The dialog re-export is
deferred via PEP 562 ``__getattr__`` so importing this subpackage
does NOT pull in :mod:`textual` (mirrors the parent ``_app``
package's lazy-import discipline -- see
:mod:`tokenizer.inspector._app.__init__`).

Submodule split per cluster #7 (B-L2 H2, B-L4 M3, B-L5 H4): four
files, each below the 300-LOC cap, covering one concern each.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._axes import (
    AxisDescriptor,
    AxisKind,
    BITWIDTH_AXIS_KEY,
    OrderAccepted,
    OrderCancelled,
    OrderConfig,
    OrderResult,
    build_canonical_axes,
    build_extra_meta_axis,
    discover_extra_meta_keys,
    extract_axis_value,
)
from ._grouping import (
    VariantGroupNode,
    format_grouping_label,
    group_variants,
)


__all__ = [
    "AxisDescriptor",
    "AxisKind",
    "BITWIDTH_AXIS_KEY",
    "OrderAccepted",
    "OrderCancelled",
    "OrderConfig",
    "OrderDialog",
    "OrderResult",
    "VariantGroupNode",
    "build_canonical_axes",
    "build_extra_meta_axis",
    "discover_extra_meta_keys",
    "extract_axis_value",
    "format_grouping_label",
    "group_variants",
]


if TYPE_CHECKING:
    from ._dialog import OrderDialog


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export for the Textual-dependent dialog."""
    if name == "OrderDialog":
        from ._dialog import OrderDialog

        return OrderDialog
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
