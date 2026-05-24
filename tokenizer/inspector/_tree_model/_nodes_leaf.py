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
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.token_manager import VocabularyManager

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

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager | None" = None,
    ) -> list:
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

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager | None" = None,
    ) -> list:
        return list(self.other_variants)
