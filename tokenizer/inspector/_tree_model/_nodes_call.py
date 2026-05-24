"""``InlineCallNode`` -- inline call site under a block.

Expandable only when the callee is a local matched function whose
``function_section_ptr`` resolves through the parent's
:class:`DecodeContext.callee_arm_resolver` to a real
:class:`SectionPointerSpec` (PLT / EXTERN call sites have no body to
inline). Expansion fires a fresh :func:`batch_decode` for the callee
through :meth:`FunctionNode.expand` and surfaces the variant matching
the caller's per-call entry, plus a :class:`ShowAllVariantsNode`
sibling for the others.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec

from ._context import DecodeContext


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.token_manager import VocabularyManager

    from ._nodes_variant import VariantNode


__all__ = ["InlineCallNode"]


@dataclass
class InlineCallNode:
    """Inline call from a block to another function.

    Expandable only when the callee is a local matched function with
    an addressable section pointer; PLT / EXTERN have no body to
    inline. Expansion fires a fresh ``batch_decode`` for the callee
    per plan D2 and surfaces the variant matching the caller's
    per-call entry, plus a :class:`ShowAllVariantsNode` sibling for
    the others.

    ``kind`` is the canonical
    :class:`~tokenizer.aligned_data.call_target_type.CallTargetType`
    enum (LOCAL / PLT / EXTERN). ``provider`` is the library /
    sidecar name for the ``@<provider>`` suffix on EXTERN rows;
    ``None`` for LOCAL / PLT and for EXTERN rows whose provider is
    unknown.
    """

    kind: CallTargetType
    counter_id: int
    callee_name: str
    callee_section_pointer: SectionPointerSpec | None
    variant_idx: int
    provider: str | None
    decode_context: DecodeContext
    is_failed: bool = False

    @property
    def can_expand(self) -> bool:
        # Single dispatch point; the UI gates the expand call on this.
        return (
            self.kind is CallTargetType.LOCAL
            and self.callee_section_pointer is not None
        )

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager",
    ) -> list:
        """Decode the callee section and surface the matching variant
        + (when present) a ``ShowAllVariantsNode`` for the siblings."""
        from ._nodes_function import FunctionNode
        from ._nodes_leaf import ShowAllVariantsNode

        if self.callee_section_pointer is None:
            raise RuntimeError(
                "InlineCallNode.expand called on a non-expandable node "
                f"(kind={self.kind.name}); UI should gate on can_expand."
            )
        spec = self.callee_section_pointer
        # Reuse FunctionNode.expand -- the callee is inspected via the
        # exact same batch_decode contract as a top-level open.
        all_variants = FunctionNode(
            arm=spec.arm, idx=spec.idx, name=self.callee_name
        ).expand(session, vocab_manager=vocab_manager)

        matched_idx_in_list = _find_matching_variant_index(
            all_variants, self.variant_idx
        )
        if matched_idx_in_list is None:
            # Caller's variant not in callee's surviving set
            # (MISSING_VARIANT_INDEX-style drop) -- fall back to
            # surfacing all variants directly.
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
    variants: list["VariantNode"], target_section_variant_index: int
) -> int | None:
    """Position in ``variants`` whose section-slot matches the target.

    The caller's ``VariantBlock.per_call_entries`` stores a section
    slot index into the callee section; we map each VariantNode back
    to that slot and pick the match.
    """
    for i, v in enumerate(variants):
        if _variant_slot_in_section(v) == target_section_variant_index:
            return i
    return None


def _variant_slot_in_section(variant: "VariantNode") -> int:
    """Slot index of ``variant.variant_block`` inside its section.

    Object-identity scan; n_variants per section is small. The Phase 2
    design has a single
    :meth:`tokenizer.aligned_data.loader.session.BinarySession._parse_section_at`
    parse per section per session, so every :class:`VariantBlock` in
    play is the same object stored on ``section.variants`` -- identity
    holds by construction. Returning ``-1`` on identity miss surfaces
    as ``None`` from :func:`_find_matching_variant_index`, which
    legitimately means "this variant_block doesn't belong to this
    section" rather than a callable contract violation.
    """
    for i, vb in enumerate(variant.section.variants):
        if vb is variant.variant_block:
            return i
    return -1
