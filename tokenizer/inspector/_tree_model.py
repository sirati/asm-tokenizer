"""Tree-model node dataclasses for the inspector.

Pure model layer (no Textual imports): every node is a frozen
dataclass with ``can_expand: bool`` + ``expand(session, *,
vocab_manager) -> list[Node]``. The UI layer (``_app.py``) drives
expand and wraps it in a single try/except dispatcher per plan D8 —
node methods raise normally; ``is_failed`` is stamped by the UI.

Lazy expansion contract (per plan D2): only :class:`FunctionNode` and
:class:`InlineCallNode` invoke :func:`batch_decode` (every call uses
``include_fid_sidecar=True`` + ``keep_intermediate=True``).
:class:`VariantNode` / :class:`BlockNode` / :class:`InlineJumpNode`
consume data already in hand. PLT / EXT call sites are non-expandable
leaves (only ``can_expand`` for local-matched calls).

The :class:`DecodeContext` bundles the ``fid_sidecar`` /
``fid_row_offsets`` / ``line_to_name`` / vocab references threaded
from the parent :class:`FunctionNode` through every descendant so
each node carries ONE shared reference instead of four.

Lazy-import discipline: :mod:`._label` and :mod:`._render` are sibling
modules that may land in parallel; both are imported inside expand
bodies so this module loads even before they exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode import batch_decode
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.metadata_loader import SectionKind


if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.batch_decode._types import (
        BatchDecodeResult,
    )
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.aligned_data.matched_sections_bin import Section, VariantBlock
    from tokenizer.token_manager import VocabularyManager


__all__ = [
    "AsmLeaf",
    "BlockNode",
    "DecodeContext",
    "FunctionNode",
    "InlineCallNode",
    "InlineJumpNode",
    "Node",
    "ShowAllVariantsNode",
    "VariantNode",
]


# ---------------------------------------------------------------------------
# Decode-context bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeContext:
    """Per-FunctionNode batch-decode context threaded to descendants.

    Lifetime equals the parent FunctionNode's ``BatchDecodeResult``;
    holds only plain references to numpy views + mappings, plus the
    section-arm tag (:class:`SectionKind`) so descendant render calls
    that consult ``callee_arm_resolver(call_target, arm)`` have the
    parent's arm without re-deriving it.
    """

    arm: SectionKind
    fid_sidecar: Optional[np.ndarray]
    fid_row_offsets: Optional[np.ndarray]
    line_to_name: Mapping[int, str]
    vocab_manager: Optional["VocabularyManager"]


# ---------------------------------------------------------------------------
# Leaf nodes (non-expandable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsmLeaf:
    """One asm-like line inside a block — terminal."""

    text: str
    can_expand: bool = field(default=False, init=False)
    is_failed: bool = False

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["Node"]:
        """Leaves carry no children; defensive empty list so accidental
        expand calls from the UI never raise — ``can_expand`` is the
        intended gate."""
        return []


@dataclass(frozen=True)
class ShowAllVariantsNode:
    """`[+] show all variants` sibling under an InlineCallNode.

    Holds the variants of the callee that are NOT the caller's
    matching variant; expand yields them.
    """

    label: str
    other_variants: Tuple["VariantNode", ...]
    can_expand: bool = field(default=True, init=False)
    is_failed: bool = False

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["Node"]:
        return list(self.other_variants)


# ---------------------------------------------------------------------------
# Function -> Variant -> Block -> {Asm, InlineCall, InlineJump}
# ---------------------------------------------------------------------------


def _context_len_for_variants(variants_lengths: list[int]) -> int:
    """``context_len`` sized to the longest variant + headroom (plan D2).

    Headroom (64) covers the variant-axis prefix + per-call prepend slot
    budget so no variant is mid-cut by the inspector's batch_decode.
    """
    longest = max(variants_lengths) if variants_lengths else 0
    return max(longest + 64, 64)


def _build_variants_from_result(
    result: BatchDecodeResult,
    section_index_in_result: int,
    *,
    decode_context: DecodeContext,
) -> list["VariantNode"]:
    """Build VariantNodes from a batch_decode result.

    Reaches into ``result.intermediate.stage2.stage1`` for the parsed
    Section + per-variant FunctionData per plan D2. The caller pre-
    builds the :class:`DecodeContext` so every node shares it.
    """
    # Lazy import — sibling _label.py lands in parallel.
    from ._label import variant_label

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
            # Padding / dropped slot — skip so batch_row_idx is always
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


def _session_line_to_name(
    session: Optional["BinarySession"],
) -> Mapping[int, str]:
    """Extract ``line_to_name`` from a session via its public metadata
    accessor. Empty mapping when absent — name resolution then falls
    back to ``"?"`` per plan D4.
    """
    if session is None:
        return {}
    return session.get_metadata("line_to_name") or {}


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
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["VariantNode"]:
        """Fire the one mandatory ``batch_decode`` call per plan D2.

        Returns one :class:`VariantNode` per surviving variant. Raises
        on failure — the UI dispatcher wraps the call.
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
            line_to_name=_session_line_to_name(session),
            vocab_manager=vocab_manager,
        )
        return _build_variants_from_result(
            result,
            section_index_in_result=0,
            decode_context=decode_context,
        )


@dataclass(frozen=True)
class VariantNode:
    """One per variant of a function — wraps the per-variant FunctionData.

    No ``batch_decode`` call: the body was loaded as part of the parent
    :class:`FunctionNode`'s batch.
    """

    function_data: "FunctionData"
    section: "Section"
    variant_block: "VariantBlock"
    batch_row_idx: int
    label: str
    decode_context: DecodeContext
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["BlockNode"]:
        """Enumerate the variant body's blocks — no decode."""
        from ._label import block_preview

        ftl = _reconstruct_function_token_list(
            self.function_data, self.decode_context.vocab_manager
        )
        blocks: list[BlockNode] = []
        for block_idx, block in enumerate(ftl.iter_blocks(transient=True)):
            blocks.append(
                BlockNode(
                    function_data=self.function_data,
                    section=self.section,
                    variant_block=self.variant_block,
                    block_idx=block_idx,
                    batch_row_idx=self.batch_row_idx,
                    preview=block_preview(block),
                    decode_context=self.decode_context,
                )
            )
        return blocks


def _reconstruct_function_token_list(
    function_data: "FunctionData",
    vocab_manager: Optional["VocabularyManager"],
):
    """Single point that turns a FunctionData into an iterable
    FunctionTokenList; lazy-imports to avoid pulling tokens machinery
    at module load."""
    from tokenizer.function_token_list import FunctionTokenList

    return FunctionTokenList.reconstruct_func_from_raw_bytes(
        function_data.tokens,
        function_data.block_runlength,
        function_data.insn_runlength,
        vocab_manager=vocab_manager,
    )


@dataclass(frozen=True)
class BlockNode:
    """One per block within a variant body.

    Expansion lists the block's asm-like lines, inline calls, and
    inline jumps. NO ``batch_decode`` call.
    """

    function_data: "FunctionData"
    section: "Section"
    variant_block: "VariantBlock"
    block_idx: int
    batch_row_idx: int
    preview: str
    decode_context: DecodeContext
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["Node"]:
        """Map this block's render items into model nodes."""
        return _expand_block_body(self)


def _expand_block_body(block: "BlockNode") -> list["Node"]:
    """Render one block + lift its line items into model nodes.

    Shared by :meth:`BlockNode.expand` and
    :meth:`InlineJumpNode.expand` so the rendering path is single-
    sourced (an InlineJump is just "render the target block").
    """
    from ._render import (
        AsmLine,
        InlineCallEntry,
        InlineJumpEntry,
        render_block,
    )

    items = render_block(
        block.function_data,
        block.section,
        block.variant_block,
        block.block_idx,
        arm=block.decode_context.arm,
        fid_sidecar=block.decode_context.fid_sidecar,
        fid_row_offsets=block.decode_context.fid_row_offsets,
        line_to_name=block.decode_context.line_to_name,
        batch_row_idx=block.batch_row_idx,
        vocab_manager=block.decode_context.vocab_manager,
    )

    out: list[Node] = []
    for item in items:
        if isinstance(item, AsmLine):
            out.append(AsmLeaf(text=item.text))
        elif isinstance(item, InlineCallEntry):
            out.append(
                InlineCallNode(
                    kind=item.kind,
                    counter_id=item.counter_id,
                    callee_name=item.callee_name,
                    callee_section_pointer=item.callee_section_pointer,
                    variant_idx=item.variant_idx,
                    provider=item.provider,
                    decode_context=block.decode_context,
                )
            )
        elif isinstance(item, InlineJumpEntry):
            out.append(
                InlineJumpNode(
                    function_data=block.function_data,
                    section=block.section,
                    variant_block=block.variant_block,
                    batch_row_idx=block.batch_row_idx,
                    target_block_idx=item.target_block_idx,
                    decode_context=block.decode_context,
                )
            )
        else:
            # Unknown line-item kind = render/model contract drift.
            raise TypeError(
                f"unknown render line item type: {type(item).__name__}"
            )
    return out


@dataclass(frozen=True)
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
    callee_section_pointer: Optional[Tuple[SectionKind, int]]
    variant_idx: int
    provider: Optional[str]
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
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["Node"]:
        """Decode the callee section and surface the matching variant
        + (when present) a ``ShowAllVariantsNode`` for the siblings."""
        if self.callee_section_pointer is None:
            raise RuntimeError(
                "InlineCallNode.expand called on a non-expandable node "
                f"(kind={self.kind.name}); UI should gate on can_expand."
            )
        arm, callee_idx = self.callee_section_pointer
        # Reuse FunctionNode.expand — the callee is inspected via the
        # exact same batch_decode contract as a top-level open.
        all_variants = FunctionNode(
            arm=arm, idx=callee_idx, name=self.callee_name
        ).expand(session, vocab_manager=vocab_manager)

        matched_idx_in_list = _find_matching_variant_index(
            all_variants, self.variant_idx
        )
        if matched_idx_in_list is None:
            # Caller's variant not in callee's surviving set
            # (MISSING_VARIANT_INDEX-style drop) — fall back to
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
) -> Optional[int]:
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


@dataclass(frozen=True)
class InlineJumpNode:
    """Inline jump to another block in the SAME variant.

    Expansion renders the target block in place — no decode. Routes
    through ``_expand_block_body`` so jump-target rendering shares
    its implementation with :class:`BlockNode`.
    """

    function_data: "FunctionData"
    section: "Section"
    variant_block: "VariantBlock"
    batch_row_idx: int
    target_block_idx: int
    decode_context: DecodeContext
    is_failed: bool = False
    can_expand: bool = field(default=True, init=False)

    def expand(
        self,
        session: "BinarySession",
        *,
        vocab_manager: Optional["VocabularyManager"] = None,
    ) -> list["Node"]:
        # Synthetic BlockNode never enters the tree; only its children.
        synthetic = BlockNode(
            function_data=self.function_data,
            section=self.section,
            variant_block=self.variant_block,
            block_idx=self.target_block_idx,
            batch_row_idx=self.batch_row_idx,
            preview="",
            decode_context=self.decode_context,
        )
        return _expand_block_body(synthetic)


# Union of every concrete node type the model can produce; the UI
# layer pins this single name instead of sprinkling typing.Union.
Node = (
    AsmLeaf
    | BlockNode
    | FunctionNode
    | InlineCallNode
    | InlineJumpNode
    | ShowAllVariantsNode
    | VariantNode
)
