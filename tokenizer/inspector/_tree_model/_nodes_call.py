"""``InlineCallNode`` -- inline call site under a block.

Expandable only when the callee is LOCAL and the
:class:`InlineCallEntry` carried a resolvable
:class:`SectionPointerSpec`; PLT / EXTERN have no body to inline.
Expansion constructs a synthetic callee :class:`FunctionNode` against
the same :class:`BackendFactory` the parent tree-open used; the
returned variants are matched against ``self.variant_idx`` and the
unmatched variants are bundled under a :class:`ShowAllVariantsNode`
sibling (per plan section 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenizer.aligned_data.call_target_type import CallTargetType


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        FunctionHandle,
    )

    from ._nodes_variant import VariantNode


__all__ = ["InlineCallNode"]


@dataclass
class InlineCallNode:
    """Inline call from a block to another function.

    ``kind`` is the canonical
    :class:`~tokenizer.aligned_data.call_target_type.CallTargetType`
    enum (LOCAL / PLT / EXTERN). ``provider`` is the library /
    sidecar name appended after ``@`` on EXTERN rows; ``None`` for
    LOCAL / PLT and for EXTERN rows whose provider is unknown.
    ``callee_handle`` is ``None`` when the call has no addressable
    callee section (EXTERN, Phase-1 BatchDecode, FtlBackend per plan
    decision 24, or LOCAL/PLT row whose pointer failed to resolve).
    """

    factory: "BackendFactory"
    kind: CallTargetType
    counter_id: int
    callee_name: str
    callee_handle: "FunctionHandle | None"
    variant_idx: int
    provider: str | None
    is_failed: bool = False

    @property
    def can_expand(self) -> bool:
        # Single dispatch point; the UI gates the expand call on this.
        # Mirrors the legacy contract: only LOCAL with a resolved
        # callee section is expandable. PLT / EXTERN have no body to
        # inline; LOCAL without a resolved pointer (callee_handle is
        # None) cannot construct a FunctionNode either.
        return (
            self.kind is CallTargetType.LOCAL
            and self.callee_handle is not None
        )

    def expand(self) -> list:
        """Open the callee + surface the matching variant + others.

        Constructs a synthetic :class:`FunctionNode` against the same
        factory; ``expand`` on that node returns one
        :class:`VariantNode` per callee variant. The variant whose
        ``variant_idx`` equals ``self.variant_idx`` is the row the
        caller pinned to; the others are bundled under a
        :class:`ShowAllVariantsNode` sibling.

        When the caller's variant is not in the callee's surviving
        set (e.g. ``MISSING_VARIANT_INDEX`` -- the callee dropped that
        variant), the fallback is to surface all variants directly,
        matching the pre-Wave-5 ``_find_matching_variant_index``
        behaviour.
        """
        from ._nodes_function import FunctionNode
        from ._nodes_leaf import ShowAllVariantsNode

        if self.callee_handle is None:
            raise RuntimeError(
                "InlineCallNode.expand called on a non-expandable node "
                f"(kind={self.kind.name}); UI should gate on can_expand."
            )
        callee = FunctionNode(factory=self.factory, handle=self.callee_handle)
        all_variants = callee.expand()

        matched_idx_in_list = _find_matching_variant_index(
            all_variants, self.variant_idx
        )
        if matched_idx_in_list is None:
            # Caller's variant not in callee's surviving set -- fall
            # back to surfacing every variant directly.
            return list(all_variants)

        matched = all_variants[matched_idx_in_list]
        others = tuple(
            v for i, v in enumerate(all_variants) if i != matched_idx_in_list
        )
        if not others:
            return [matched]
        return [
            matched,
            ShowAllVariantsNode(
                label="show all variants", other_variants=others
            ),
        ]


def _find_matching_variant_index(
    variants: list["VariantNode"], target_variant_idx: int
) -> int | None:
    """Position in ``variants`` whose ``variant_idx`` matches the target.

    The pre-Wave-5 implementation matched on object identity of the
    underlying ``variant_block``; the post-Wave-5 typed contract uses
    the int ``variant_idx`` carried on every :class:`VariantNode`,
    which both backends pin to the same canonical value (FtlBackend's
    lockstep position, BatchDecodeBackend's per-section variant
    index) -- the Protocol is the single source of truth.
    """
    for i, v in enumerate(variants):
        if v.variant_idx == target_variant_idx:
            return i
    return None
