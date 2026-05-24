"""Block-renderer for the inspector tree.

Single concern: translate ONE pre-parsed
:class:`~tokenizer.token_lists.BlockTokenList` into an ordered list
of typed :class:`LineItem` s (:class:`AsmLine` +
:class:`InlineCallEntry` + :class:`InlineJumpEntry`).

Parsing of the parent :class:`FunctionTokenList` + the per-section
+ per-variant invariants (``kind_to_called_idx`` /
``variant_pins``) is the tree-model layer's concern -- this renderer
receives them already built and walks the block's instruction stream
once. Decoupling parse-from-render keeps the renderer cheap to call
per inline-jump expansion and obeys the CLAUDE.md "no re-parsing in
call chains" rule.

The caller threads in:

* ``block`` -- the parent variant's pre-parsed
  :class:`~tokenizer.token_lists.BlockTokenList`.
* ``section`` -- parent :class:`~tokenizer.aligned_data.matched_sections_bin.Section`
  (parsed once at function-open time per plan D2). Its
  ``call_targets`` list is the authority for "what does this call
  site go to": each LOCAL/PLT/EXT_FUNC token carries an
  encoder-allocated per-Category counter that maps DIRECTLY to the
  position of the K-th call_target of that ``CallTargetType`` in
  ``section.call_targets`` (see :func:`_kind_to_called_idx`).
* ``kind_to_called_idx`` -- per-kind index lists into
  ``section.call_targets`` (variant-level invariant; the tree model
  builds it once per variant).
* ``variant_pins`` -- this variant's ``called_idx ->
  section_variant_index`` table (variant-level invariant; built once
  per variant).
* ``line_to_name`` -- the per-binary
  ``<binary>_function_names.txt`` mapping, used to resolve the
  callee FID -> display name. Lives in
  :mod:`tokenizer.aligned_data.loader.function_names_loader`.
* ``callee_arm_resolver`` -- closure that maps a section byte offset
  (``CallTarget.function_section_ptr``) to a
  :class:`SectionPointerSpec` or ``None`` when the offset doesn't
  resolve to a known section. Built once per FunctionNode-open by the
  tree model; the renderer never reaches into session internals.

No Textual imports, no tree-node construction, no string-parsing of
``to_asm_like`` output. The discriminator between asm / call / jump
is the structured :class:`~tokenizer.tokens.TokenType` enum on each
metatoken (LOCAL_FUNC / PLT_FUNC / EXT_FUNC = call site; BLOCK_V2 =
within-function jump target).

This module owns ONLY the per-block walk; building tree nodes from
these line items is :mod:`tokenizer.inspector._tree_model`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    Section,
    VariantBlock,
)
from tokenizer.token_lists import BlockTokenList
from tokenizer.tokens import TokenType


__all__ = [
    "AsmLine",
    "InlineCallEntry",
    "InlineJumpEntry",
    "LineItem",
    "render_block",
]


# ---------------------------------------------------------------------------
# Typed line items (public API consumed by ``_tree_model``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsmLine:
    """A plain assembly-like line emitted for one instruction."""

    text: str


@dataclass(frozen=True)
class InlineCallEntry:
    """One inline call site under a block.

    ``counter_id`` is the encoder's per-Category counter (= the
    position of the K-th call_target of ``kind`` in
    ``section.call_targets``). ``callee_section_pointer`` is the
    :class:`SectionPointerSpec` the tree-model layer can hand to
    :func:`batch_decode` when the user expands this node;
    ``None`` for ext calls or for LOCAL/PLT call_targets whose
    ``function_section_ptr`` could not be resolved to a section.
    ``variant_idx`` is the callee variant the parent's
    ``per_call_entries`` pinned for THIS caller variant; equals
    :data:`MISSING_VARIANT_INDEX` when no per_call_entry exists
    (EXTERN) or when the callee section is reachable but lacks a
    variant matching the caller's vkey.

    ``kind`` is the canonical wire-format
    :class:`~tokenizer.aligned_data.call_target_type.CallTargetType`
    enum (LOCAL/PLT/EXTERN); the rendering layer
    (:mod:`tokenizer.inspector._label`) routes its per-kind label
    word off this same enum so no string-typed discriminator crosses
    this boundary. ``provider`` is the library / sidecar name
    appended after ``@`` for EXTERN rows; ``None`` for LOCAL/PLT
    and for EXTERN rows whose provider is unknown.
    """

    kind: CallTargetType
    counter_id: int
    callee_name: str
    callee_section_pointer: SectionPointerSpec | None
    variant_idx: int
    provider: str | None


@dataclass(frozen=True)
class InlineJumpEntry:
    """One within-function jump target referenced by an instruction."""

    target_block_idx: int


LineItem = AsmLine | InlineCallEntry | InlineJumpEntry


# ---------------------------------------------------------------------------
# Section-level invariants (built once per variant by the tree model)
# ---------------------------------------------------------------------------


def _kind_to_called_idx(section: Section) -> dict[CallTargetType, list[int]]:
    """Per-kind index lists into ``section.call_targets``.

    The matched-sections writer concatenates call_targets in
    non-decreasing :class:`CallTargetType` order (LOCAL -> PLT ->
    EXTERN) with stable encounter-order within each block (see
    :mod:`tokenizer.memmap_builder._pass2` ``_emit_section_call_targets``).
    That guarantees the K-th element of ``kind_to_idx[kind]`` is the
    K-th distinct callee of that ``CallTargetType`` IN THE SAME
    ORDER THE ENCODER'S PER-CATEGORY COUNTER WALKED THEM -- so
    ``counter_id`` from a LOCAL/PLT/EXT_FUNC token maps directly to
    ``section.call_targets[kind_to_idx[kind][counter_id]]``.

    This is variant-level invariant; the tree model
    (:mod:`tokenizer.inspector._tree_model._nodes_variant`) builds it
    once per :class:`VariantNode.expand` and threads the result down
    to every block render.
    """
    kind_to_idx: dict[CallTargetType, list[int]] = {
        k: [] for k in CallTargetType
    }
    for called_idx, ct in enumerate(section.call_targets):
        kind_to_idx[ct.type].append(called_idx)
    return kind_to_idx


def _variant_index_for_called_idx(variant_block: VariantBlock) -> dict[int, int]:
    """Build ``called_idx -> section_variant_index`` for this variant.

    A given ``called_idx`` appears AT MOST ONCE in
    ``per_call_entries`` per variant (the encoder collapses repeated
    same-callee calls within a variant to one per-call slot; see
    :mod:`tokenizer.memmap_builder._pass2`
    ``_emit_variant_per_call_entries``). EXTERN call_targets are
    filtered out before emission, so their ``called_idx`` is absent
    from the dict.

    Variant-level invariant -- the tree model builds it once per
    :class:`VariantNode.expand`.
    """
    return {
        int(called_idx): int(section_variant_index)
        for called_idx, section_variant_index in variant_block.per_call_entries
    }


# ---------------------------------------------------------------------------
# Per-token classification (typed dispatch, no string parsing)
# ---------------------------------------------------------------------------


_CALL_TOKEN_TYPES: dict[TokenType, CallTargetType] = {
    TokenType.LOCAL_FUNC: CallTargetType.LOCAL,
    TokenType.PLT_FUNC: CallTargetType.PLT,
    TokenType.EXT_FUNC: CallTargetType.EXTERN,
}


_JUMP_TOKEN_TYPES: frozenset[TokenType] = frozenset({TokenType.BLOCK_V2})


def _emit_call_entry(
    *,
    kind: CallTargetType,
    counter_id: int,
    section: Section,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
    line_to_name: Mapping[int, str],
) -> InlineCallEntry:
    """Build one :class:`InlineCallEntry` from a call-site token.

    All branches go through ONE typed dispatch: ``kind`` (a
    :class:`CallTargetType`) drives lookups via dicts
    (``kind_to_called_idx`` + ``variant_pins``) -- no per-kind
    ``if/elif`` chains. EXT vs LOCAL/PLT pointer-resolution differs
    ONLY in whether the arm resolver yields a section; that single
    ``Optional`` flows straight into ``callee_section_pointer``.

    An out-of-range ``counter_id`` for the kind raises
    :class:`IndexError` naturally from the list lookup -- the
    inspector is a diagnostic tool, surfacing a corrupt counter as a
    crash points at the encoder bug rather than papering over it with
    a ``"?"`` placeholder.

    ``provider`` is left ``None`` here: for EXTERN call_targets the
    library name is keyed by ``CallTarget.function_section_ptr`` into
    the per-binary ``<binary>_extern_providers.txt`` sidecar, which is
    not currently threaded into the render layer. The label layer
    falls back to ``"?"`` on a ``None`` provider so the EXTERN row's
    ``@?`` suffix shape is preserved until the sidecar is wired in.
    """
    called_idx = kind_to_called_idx[kind][counter_id]
    call_target = section.call_targets[called_idx]
    callee_section_pointer = callee_arm_resolver(
        int(call_target.function_section_ptr)
    )
    variant_idx = variant_pins.get(called_idx, MISSING_VARIANT_INDEX)
    callee_name = line_to_name.get(call_target.function_name_ptr, "?")

    return InlineCallEntry(
        kind=kind,
        counter_id=counter_id,
        callee_name=callee_name,
        callee_section_pointer=callee_section_pointer,
        variant_idx=variant_idx,
        provider=None,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_block(
    *,
    block: BlockTokenList,
    section: Section,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
) -> list[LineItem]:
    """Walk one block's instructions and emit typed line items.

    Iterates ``block.iter_insn`` and per instruction yields one
    :class:`AsmLine` plus -- in token-stream order WITHIN the
    instruction -- one :class:`InlineCallEntry` per
    LOCAL_FUNC/PLT_FUNC/EXT_FUNC metatoken and one
    :class:`InlineJumpEntry` per BLOCK_V2 metatoken. Token-type
    discrimination is via the structured :class:`TokenType` enum (no
    string parsing of asm).

    The parsed :class:`BlockTokenList` + the section-level invariants
    are produced ONCE by the tree-model layer's
    :func:`VariantNode.expand`; rebuilding them here would re-parse
    the parent :class:`FunctionTokenList` per call (CLAUDE.md "no
    re-parsing in call chains" rule).
    """
    return list(
        _walk_block_instructions(
            block,
            section=section,
            kind_to_called_idx=kind_to_called_idx,
            variant_pins=variant_pins,
            line_to_name=line_to_name,
            callee_arm_resolver=callee_arm_resolver,
        )
    )


def _walk_block_instructions(
    block: BlockTokenList,
    *,
    section: Section,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
) -> Iterable[LineItem]:
    """Per-instruction generator: one :class:`AsmLine` then any inline
    call/jump entries from the instruction's metatoken stream."""
    for insn in block.iter_insn(transient=True):
        yield AsmLine(text=insn.to_asm_like())
        yield from _emit_inline_entries(
            insn,
            section=section,
            kind_to_called_idx=kind_to_called_idx,
            variant_pins=variant_pins,
            line_to_name=line_to_name,
            callee_arm_resolver=callee_arm_resolver,
        )


def _emit_inline_entries(
    insn,
    *,
    section: Section,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
) -> Iterable[LineItem]:
    """Inline call/jump items for ONE instruction.

    Dispatch is one dict lookup (call kind) + one set membership
    (jump); identity payload reads via :meth:`InsnTokenList.iter_tokens`
    -- the same path :meth:`InsnTokenList.to_asm_like` walks, so the
    AsmLine and inline entries see a consistent token view.
    """
    for token in insn.iter_tokens():
        token_type = token.token_type
        kind = _CALL_TOKEN_TYPES.get(token_type)
        if kind is not None:
            yield _emit_call_entry(
                kind=kind,
                counter_id=int(token.id),
                section=section,
                kind_to_called_idx=kind_to_called_idx,
                variant_pins=variant_pins,
                callee_arm_resolver=callee_arm_resolver,
                line_to_name=line_to_name,
            )
            continue
        if token_type in _JUMP_TOKEN_TYPES:
            yield InlineJumpEntry(target_block_idx=int(token.id))
