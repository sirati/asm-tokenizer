"""``InlineCallNode`` -- inline call site under a block.

Expandable whenever the :class:`InlineCallEntry` carried a resolvable
:class:`SectionPointerSpec`; LOCAL and PLT both qualify (the writer
treats them identically -- a PLT thunk is sectioned just like a local
function, its body is the small jump-thunk to the resolved extern).
EXTERN has no callee section at all so it never expands.
Expansion constructs a synthetic callee :class:`FunctionNode` against
the same :class:`BackendFactory` the parent tree-open used; the
matching variant's blocks are surfaced DIRECTLY (skipping the
intermediate variant-list level, per plan decision D2) and the other
variants are bundled under a :class:`ShowAllVariantsNode` sibling.
The fallback path (caller's pin not in the callee's surviving set,
e.g. ``MISSING_VARIANT_INDEX``) is to surface every variant as a
sibling, matching the pre-D2 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX


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
    # Caller-row variant_idx; used as a fallback pin when the
    # callee-side ``variant_idx`` is :data:`MISSING_VARIANT_INDEX`
    # (e.g. Function-ID self-references where no vkey pin is recorded).
    # Defaults to :data:`MISSING_VARIANT_INDEX` so existing constructors
    # without caller threading retain the pre-existing all-variants
    # fallback behaviour.
    caller_variant_idx: int = MISSING_VARIANT_INDEX
    is_failed: bool = False
    # Per-row horizontal scroll memory; see :mod:`tokenizer.inspector._app._tree_widget`.
    remembered_scroll_x: int = field(default=0, init=False)

    @property
    def can_expand(self) -> bool:
        # Single dispatch point; the UI gates the expand call on this.
        # LOCAL and PLT call_targets both point at a section the writer
        # emitted (the writer's ``_resolve_function_section_ptr`` treats
        # them identically -- PLT thunks are sectioned just like local
        # functions, their body being the small jump-thunk that calls
        # the resolved extern). EXTERN has no callee section so its
        # ``callee_handle`` is always ``None`` here; LOCAL/PLT without
        # a resolved pointer (cross-arm miss) likewise cannot construct
        # a FunctionNode.
        return self.callee_handle is not None

    def expand(self) -> list:
        """Open the callee + inline the matching variant's blocks.

        Constructs a synthetic :class:`FunctionNode` against the same
        factory; ``expand`` on that node returns one
        :class:`VariantNode` per callee variant. The pinned variant is
        chosen by ``self.variant_idx`` when it is not
        :data:`MISSING_VARIANT_INDEX`, otherwise by
        ``self.caller_variant_idx`` (the row that emitted this call
        site) — so Function-ID self-references and other no-vkey-pin
        cases still default to the caller's variant instead of
        surfacing the full variant list. That pinned variant's blocks
        are spliced in DIRECTLY as the InlineCallNode's children,
        skipping the intermediate variant-list level (plan decision D2).
        The other variants are bundled under a
        :class:`ShowAllVariantsNode` sibling appended after the
        spliced blocks so they remain reachable.

        When neither pin resolves to a surviving callee variant
        (e.g. both are :data:`MISSING_VARIANT_INDEX`, or the callee
        dropped both indices), the fallback is to surface every
        variant directly as VariantNode siblings, matching the
        pre-D2 contract.
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

        pinned = (
            self.variant_idx
            if self.variant_idx != MISSING_VARIANT_INDEX
            else self.caller_variant_idx
        )
        matched_idx_in_list = (
            None
            if pinned == MISSING_VARIANT_INDEX
            else _find_matching_variant_index(all_variants, pinned)
        )
        if matched_idx_in_list is None:
            # Caller's variant not in callee's surviving set -- fall
            # back to surfacing every variant directly.
            return list(all_variants)

        matched = all_variants[matched_idx_in_list]
        others = tuple(
            v for i, v in enumerate(all_variants) if i != matched_idx_in_list
        )
        # Splice the matched variant's blocks in directly (D2: skip the
        # intermediate variant-list level for the pinned variant). The
        # callee's :class:`FunctionNode` already owns the
        # :class:`RenderBackend` instance the :class:`VariantNode` was
        # threaded with, so ``matched.expand()`` resolves blocks via
        # the same backend cache — no extra factory.make call.
        children: list = list(matched.expand())
        if others:
            children.append(
                ShowAllVariantsNode(
                    label="show all variants", other_variants=others
                )
            )
        return children


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
