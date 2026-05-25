"""Shared per-block rendering core for the inspector.

Single concern: translate ONE pre-parsed
:class:`~tokenizer.token_lists.BlockTokenList` into an ordered list
of :class:`AsmLine` items. Each :class:`AsmLine` carries its own
per-instruction openables tuple (:class:`InlineCallEntry` /
:class:`InlineJumpEntry`) on :attr:`AsmLine.openables` -- there is no
longer a top-level sibling stream of inline entries. Mirrors the
BatchDecode emit shape so the tree-model consumes ONE item kind
across both backends.

The typed line-item dataclasses + :data:`LineItem` / :data:`Openable`
union live in :mod:`tokenizer.inspector._render._protocol` (the Wave-5
shared boundary both rendering backends emit through); this module
re-exports them so a single in-process object identity flows through
the FTL path and the tree-model's ``isinstance(item, AsmLine)`` checks.

Parsing of the parent :class:`FunctionTokenList` + the per-section /
per-variant invariants (``kind_to_called_idx`` / ``variant_pins``) is
the tree-model layer's concern; this renderer receives them already
built (CLAUDE.md "no re-parsing in call chains" rule).

The caller threads in:

* ``block`` -- the parent variant's pre-parsed
  :class:`~tokenizer.token_lists.BlockTokenList`.
* ``section`` -- parent :class:`~tokenizer.aligned_data.matched_sections_bin.Section`
  (parsed once at function-open time per plan D2). Its
  ``call_targets`` list is the authority for "what does this call
  site go to": each LOCAL/PLT/EXT_FUNC token carries an
  encoder-allocated per-Category counter that maps DIRECTLY to the
  position of the K-th call_target of that ``CallTargetType`` in
  ``section.call_targets`` (see :func:`partition_call_target_kinds`).
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
* ``line_to_provider`` -- the per-binary
  ``<binary>_extern_providers.txt`` mapping (1-indexed line ->
  library / sidecar name), used to resolve an EXTERN call_target's
  ``function_section_ptr`` to the ``@libname`` suffix. Lives in
  :mod:`tokenizer.aligned_data.loader.extern_providers_loader`.
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

from typing import Callable, Iterable, Mapping, Protocol, Sequence

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    VariantBlock,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
    Openable,
)
from tokenizer.inspector._render._token_text import substitute_mem_chars
from tokenizer.token_lists import BlockTokenList
from tokenizer.tokens import TokenType


# Re-export the typed line items so legacy callers (and the test
# package at ``tokenizer.inspector.tests.test_render``) keep importing
# them from this module's public surface. The dataclasses themselves
# live in :mod:`._protocol` -- the Wave-5 shared boundary -- so the
# walker emits the same in-process objects both rendering backends
# (FTL + BatchDecode) and the tree model's ``isinstance`` checks see.
__all__ = [
    "AsmLine",
    "InlineCallEntry",
    "InlineJumpEntry",
    "LineItem",
    "RenderableCallTarget",
    "RenderableSection",
    "partition_call_target_kinds",
    "render_block",
]


def _render_insn_text(asm_like: str) -> str:
    """Apply the shared MEM substitution to an FTL ``to_asm_like`` string.

    :meth:`InsnTokenList.to_asm_like` joins each token's
    ``to_asm_like()`` output with single spaces; every token returns a
    single space-free atom, so splitting on ``" "`` recovers the atom
    stream and :func:`substitute_mem_chars` swaps any MEM-operand
    vocab-string/asm-value form for its polished display char (``[``,
    ``]``, ``+``, ``-``, ``*``, ``,``). FTL does NOT apply arch-prefix
    elision -- :meth:`PlatformTokenInner.to_asm_like` already strips it.
    """
    return " ".join(substitute_mem_chars(atom) for atom in asm_like.split(" "))


# ---------------------------------------------------------------------------
# Renderable-section Protocol
# ---------------------------------------------------------------------------


class RenderableCallTarget(Protocol):
    """Minimum-surface call-target view consumed by :func:`render_block`.

    Both the writer-side
    :class:`~tokenizer.aligned_data.matched_sections_bin.CallTarget`
    and the FTL-side :class:`FtlCallTarget` satisfy this Protocol. The
    renderer reads only the three fields below; pinning them in this
    Protocol prevents future call_target views from drifting.
    """

    type: CallTargetType
    function_name_ptr: int
    function_section_ptr: int


class RenderableSection(Protocol):
    """Minimum-surface section view consumed by :func:`render_block`.

    The renderer reads only ``call_targets``; each entry must satisfy
    :class:`RenderableCallTarget`. Both the writer-side
    :class:`~tokenizer.aligned_data.matched_sections_bin.Section` and
    the FTL :class:`FtlSectionView` qualify. Phase 2 views can opt in
    by exposing the same attribute.
    """

    call_targets: Sequence[RenderableCallTarget]


# ---------------------------------------------------------------------------
# Section-level invariants (built once per variant by the tree model)
# ---------------------------------------------------------------------------


def partition_call_target_kinds(
    call_target_types: Iterable[CallTargetType],
) -> dict[CallTargetType, list[int]]:
    """Per-:class:`CallTargetType` index lists into the source list.

    Generic over the source: callers pass the ``ct.type`` stream from
    whichever section-shape they hold (the writer-side
    :class:`Section`, the FTL-side :class:`FtlSectionView`, or any
    future Phase-2 view) and the helper returns a dict keyed by every
    :class:`CallTargetType` member, with the K-th list entry equal to
    the position of the K-th call_target of that type.

    Both inspector backends partition their call-target list this way
    (writer-side :class:`Section` for BatchDecode-rendered blocks,
    FTL :class:`FtlSectionView` for the CSV path). Lifting the walk
    out of each consumer keeps the encoder's LOCAL -> PLT -> EXTERN
    encounter-order invariant pinned in ONE place; the K-th
    ``kind_to_idx[kind]`` entry equals the K-th distinct callee of
    that ``CallTargetType`` in encoder-allocation order.

    This is variant-level invariant; the tree model
    (:mod:`tokenizer.inspector._tree_model._nodes_variant`) builds it
    once per :class:`VariantNode.expand` and threads the result down
    to every block render.
    """
    kind_to_idx: dict[CallTargetType, list[int]] = {
        k: [] for k in CallTargetType
    }
    for called_idx, ct_type in enumerate(call_target_types):
        kind_to_idx[ct_type].append(called_idx)
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


_EMPTY_PROVIDER_MAP: Mapping[int, str] = {}


def _provider_sources(
    line_to_provider: Mapping[int, str],
) -> Mapping[CallTargetType, Mapping[int, str]]:
    """Per-kind provider-mapping dispatch table.

    EXTERN call_targets key into ``line_to_provider`` via
    ``CallTarget.function_section_ptr`` (a 1-indexed line into the
    extern-providers sidecar). LOCAL/PLT call_targets use that same
    field as a section byte offset and have no library/provider, so
    they map to an empty dict whose ``.get`` always yields ``None``.
    Threading the empty dict instead of branching keeps the
    ``_emit_call_entry`` body free of per-kind ``if`` chains.
    """
    return {
        CallTargetType.LOCAL: _EMPTY_PROVIDER_MAP,
        CallTargetType.PLT: _EMPTY_PROVIDER_MAP,
        CallTargetType.EXTERN: line_to_provider,
    }


def _emit_call_entry(
    *,
    kind: CallTargetType,
    counter_id: int,
    section: RenderableSection,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
    line_to_name: Mapping[int, str],
    kind_to_provider_source: Mapping[CallTargetType, Mapping[int, str]],
) -> InlineCallEntry:
    """Build one :class:`InlineCallEntry` from a call-site token.

    All branches go through ONE typed dispatch: ``kind`` (a
    :class:`CallTargetType`) drives lookups via dicts
    (``kind_to_called_idx`` + ``variant_pins`` +
    ``kind_to_provider_source``) -- no per-kind ``if/elif`` chains.
    EXT vs LOCAL/PLT pointer-resolution differs ONLY in whether the
    arm resolver yields a section; that single ``Optional`` flows
    straight into ``callee_section_pointer``. EXT vs LOCAL/PLT
    provider lookup differs ONLY in which mapping
    ``kind_to_provider_source`` routes to (real sidecar mapping for
    EXTERN, empty dict for LOCAL/PLT).

    An out-of-range ``counter_id`` for the kind raises
    :class:`IndexError` naturally; the inspector is a diagnostic tool,
    so a corrupt counter crashes loud rather than papering over a
    likely encoder bug with a ``"?"`` placeholder.
    """
    called_idx = kind_to_called_idx[kind][counter_id]
    call_target = section.call_targets[called_idx]
    callee_section_pointer = callee_arm_resolver(
        int(call_target.function_section_ptr)
    )
    variant_idx = variant_pins.get(called_idx, MISSING_VARIANT_INDEX)
    callee_name = line_to_name.get(call_target.function_name_ptr, "?")
    provider = kind_to_provider_source[kind].get(
        int(call_target.function_section_ptr)
    )

    return InlineCallEntry(
        kind=kind,
        counter_id=counter_id,
        callee_name=callee_name,
        callee_section_pointer=callee_section_pointer,
        variant_idx=variant_idx,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_block(
    *,
    block: BlockTokenList,
    section: RenderableSection,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    line_to_provider: Mapping[int, str],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
) -> list[AsmLine]:
    """Walk one block's instructions and emit one :class:`AsmLine` each.

    Iterates ``block.iter_insn`` and per instruction emits ONE
    :class:`AsmLine` whose :attr:`AsmLine.openables` tuple carries --
    in token-stream order WITHIN the instruction -- one
    :class:`InlineCallEntry` per LOCAL_FUNC/PLT_FUNC/EXT_FUNC metatoken
    and one :class:`InlineJumpEntry` per BLOCK_V2 metatoken. Token-type
    discrimination is via the structured :class:`TokenType` enum (no
    string parsing of asm).

    Inline entries are NO LONGER yielded as top-level sibling items;
    they hang off their owning instruction's :class:`AsmLine` so the
    tree-model consumes one item kind across both rendering backends.

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
            kind_to_provider_source=_provider_sources(line_to_provider),
            callee_arm_resolver=callee_arm_resolver,
        )
    )


def _walk_block_instructions(
    block: BlockTokenList,
    *,
    section: RenderableSection,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    kind_to_provider_source: Mapping[CallTargetType, Mapping[int, str]],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
) -> Iterable[AsmLine]:
    """Per-instruction generator: one :class:`AsmLine` per instruction
    with its inline-call/jump entries attached as ``openables``."""
    for insn in block.iter_insn(transient=True):
        openables = _collect_inline_openables(
            insn,
            section=section,
            kind_to_called_idx=kind_to_called_idx,
            variant_pins=variant_pins,
            line_to_name=line_to_name,
            kind_to_provider_source=kind_to_provider_source,
            callee_arm_resolver=callee_arm_resolver,
        )
        yield AsmLine(
            text=_render_insn_text(insn.to_asm_like()),
            openables=openables,
        )


def _collect_inline_openables(
    insn,
    *,
    section: RenderableSection,
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    kind_to_provider_source: Mapping[CallTargetType, Mapping[int, str]],
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None],
) -> tuple[Openable, ...]:
    """Inline call/jump openables for ONE instruction.

    Dispatch is one dict lookup (call kind) + one set membership
    (jump); identity payload reads via :meth:`InsnTokenList.iter_tokens`
    -- the same path :meth:`InsnTokenList.to_asm_like` walks, so the
    AsmLine text and the openables tuple see a consistent token view.

    Returns the ordered tuple of openables; the caller attaches it
    onto the owning :class:`AsmLine` so the tree-model never sees
    sibling Inline*Entry items at the block level.
    """
    openables: list[Openable] = []
    for token in insn.iter_tokens():
        token_type = token.token_type
        kind = _CALL_TOKEN_TYPES.get(token_type)
        if kind is not None:
            openables.append(
                _emit_call_entry(
                    kind=kind,
                    counter_id=int(token.id),
                    section=section,
                    kind_to_called_idx=kind_to_called_idx,
                    variant_pins=variant_pins,
                    callee_arm_resolver=callee_arm_resolver,
                    line_to_name=line_to_name,
                    kind_to_provider_source=kind_to_provider_source,
                )
            )
            continue
        if token_type in _JUMP_TOKEN_TYPES:
            openables.append(InlineJumpEntry(target_block_idx=int(token.id)))
    return tuple(openables)
