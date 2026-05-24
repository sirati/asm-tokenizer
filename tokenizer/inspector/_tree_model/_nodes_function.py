"""``FunctionNode`` -- the top-level matched/unmatched function row.

Owns the ONE mandatory :func:`batch_decode` call per function-open
(plan D2): pins ``include_fid_sidecar=True`` + ``keep_intermediate=
True`` so descendant nodes consume already-decoded data. Builds the
per-FunctionNode :class:`DecodeContext` and the session-bound callee-
arm resolver closure threaded to every descendant via the same
context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tokenizer.aligned_data.loader.batch_decode import batch_decode
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from ._context import DecodeContext, session_line_to_name


if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.batch_decode._types import (
        BatchDecodeResult,
    )
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.token_manager import VocabularyManager

    from ._nodes_variant import VariantNode


__all__ = [
    "FunctionNode",
    "build_variants_from_result",
]


def _context_len_for_variants(variants_lengths: list[int]) -> int:
    """``context_len`` sized to the longest variant + headroom (plan D2).

    Headroom (64) covers the variant-axis prefix + per-call prepend slot
    budget so no variant is mid-cut by the inspector's batch_decode.
    """
    longest = max(variants_lengths) if variants_lengths else 0
    return max(longest + 64, 64)


def _build_callee_arm_resolver(
    session: "BinarySession", arm: SectionKind
):
    """Closure that maps a section byte offset to a
    :class:`SectionPointerSpec` in the supplied ``arm``.

    Reaches into the session's mixin-private
    :meth:`_idx_for_section_offset` (the inverse lookup that the
    splice walker uses too -- see
    :mod:`tokenizer.aligned_data.loader.batch_decode._callee_walk._walker`).
    The render layer's :func:`_emit_call_entry` calls this with
    ``call_target.function_section_ptr`` for LOCAL/PLT call sites; a
    ``None`` return surfaces as a non-expandable inline-call node.

    The private-attr touch lives here (one place, one comment) instead
    of leaking through every descendant render path.
    """

    def resolver(section_offset: int) -> SectionPointerSpec | None:
        idx = session._idx_for_section_offset(section_offset, arm.value)
        if idx is None:
            return None
        return SectionPointerSpec(arm=arm, idx=idx)

    return resolver


def build_variants_from_result(
    result: "BatchDecodeResult",
    section_index_in_result: int,
    *,
    decode_context: DecodeContext,
) -> list["VariantNode"]:
    """Build VariantNodes from a batch_decode result.

    Reaches into ``result.intermediate.stage2.stage1`` for the parsed
    Section + per-variant FunctionData per plan D2. The caller pre-
    builds the :class:`DecodeContext` so every node shares it.
    """
    # Lazy import -- sibling _label.py + _nodes_variant.py may evolve
    # in parallel.
    from .._label import variant_label
    from ._nodes_variant import VariantNode

    if result.intermediate is None:
        raise RuntimeError(
            "batch_decode result missing .intermediate; FunctionNode "
            "requires keep_intermediate=True"
        )
    stage1_batch = result.intermediate.stage2.stage1
    stage1_section = stage1_batch.sections[section_index_in_result]
    section = stage1_section.section

    variants: list[VariantNode] = []
    for stage1_variant in stage1_section.variants:
        if stage1_variant.batch_idx is None:
            # Padding / dropped slot -- skip so batch_row_idx is always
            # a real row downstream.
            continue
        # Root body is at call_targets[0] per Stage1 layout. Inlined
        # callees follow but the inspector exposes them via
        # InlineCallNode, not via VariantNode siblings.
        root = stage1_variant.call_targets[0]
        variant_block = section.variants[stage1_variant.variant_idx]
        variants.append(
            VariantNode(
                function_data=root.function_data,
                section=section,
                variant_block=variant_block,
                batch_row_idx=int(stage1_variant.batch_idx),
                label=variant_label(root.function_data),
                decode_context=decode_context,
            )
        )
    return variants


@dataclass(frozen=True)
class FunctionNode:
    """Top-level node: one per matched function (plan D3).

    ``arm`` is the canonical
    :class:`~tokenizer.aligned_data.loader.metadata_loader.SectionKind`
    enum (MATCHED / UNMATCHED); only MATCHED is currently seeded by
    the UI but UNMATCHED works the same way once D3 relaxes.
    """

    arm: SectionKind
    idx: int
    name: str  # resolved by caller via line_to_name
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: "VocabularyManager",
    ) -> list["VariantNode"]:
        """Fire the one mandatory ``batch_decode`` call per plan D2.

        Returns one :class:`VariantNode` per surviving variant. Raises
        on failure -- the UI dispatcher wraps the call.
        """
        # Peek the section via the public load_* APIs to size
        # ``num_variants_per_section`` (real variant count, plan D2)
        # and ``context_len`` (longest variant body's token count).
        if self.arm is SectionKind.MATCHED:
            matched = session.load_matched(self.idx)
            variant_lengths = [len(v.tokens) for v in matched.variants]
            n_variants = len(matched.variants)
        else:
            fd = session.load_unmatched(self.idx)
            variant_lengths = [len(fd.tokens)]
            n_variants = 1

        if n_variants == 0:
            raise RuntimeError(
                f"function arm={self.arm.name} idx={self.idx} has no variants"
            )

        spec = SectionPointerSpec(arm=self.arm, idx=self.idx)
        result = batch_decode(
            session,
            [spec],
            num_variants_per_section=n_variants,
            context_len=_context_len_for_variants(variant_lengths),
            max_depth=0,
            include_fid_sidecar=True,
            keep_intermediate=True,
        )
        decode_context = DecodeContext(
            arm=self.arm,
            fid_sidecar=result.fid_sidecar,
            fid_row_offsets=result.fid_row_offsets,
            line_to_name=session_line_to_name(session),
            vocab_manager=vocab_manager,
            callee_arm_resolver=_build_callee_arm_resolver(session, self.arm),
        )
        return build_variants_from_result(
            result,
            section_index_in_result=0,
            decode_context=decode_context,
        )
