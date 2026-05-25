"""Leaf node dataclasses for the inspector tree model.

Terminal nodes never expand into children; the UI gates expand on
``can_expand``. The two leaves are :class:`AsmLeaf` (one rendered
asm-like line) and :class:`ShowAllVariantsNode` (the sibling shown
under an inline-call when only some of the callee's variants matched
the caller's pin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._nodes_variant import VariantNode


__all__ = [
    "AsmLeaf",
    "ShowAllVariantsNode",
]


@dataclass
class AsmLeaf:
    """One asm-like line inside a block -- terminal."""

    text: str
    can_expand: bool = field(default=False, init=False)
    is_failed: bool = False
    # Per-row horizontal scroll memory; the UI saves the row's current
    # ``scroll_offset.x`` here on manual pan and restores it when the
    # cursor returns to this row. See :mod:`tokenizer.inspector._app`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list:
        """Terminal node -- callers must gate on ``can_expand``."""
        raise NotImplementedError(
            "AsmLeaf is terminal; gate expansion on can_expand"
        )


@dataclass
class ShowAllVariantsNode:
    """``[+] show all variants`` sibling under an InlineCallNode.

    Holds the variants of the callee that are NOT the caller's
    matching variant; expand yields them.
    """

    label: str
    other_variants: tuple["VariantNode", ...]
    can_expand: bool = field(default=True, init=False)
    is_failed: bool = False
    # Per-row horizontal scroll memory; see :class:`AsmLeaf`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list:
        return list(self.other_variants)
