"""Block-renderer for the inspector tree.

Single concern: translate ONE block of a Stage1
:class:`~tokenizer.aligned_data.loader.function_data.FunctionData`
into an ordered list of typed :class:`LineItem`s
(:class:`AsmLine` + :class:`InlineCallEntry` + :class:`InlineJumpEntry`).

The block is identified by ``block_idx`` within the function. The
caller threads in:

* ``section`` -- parent :class:`~tokenizer.aligned_data.matched_sections_bin.Section`
  (parsed once at function-open time per plan D2). Its
  ``call_targets`` list is the authority for "what does this call
  site go to": each LOCAL/PLT/EXT_FUNC token carries an
  encoder-allocated per-Category counter that maps DIRECTLY to the
  position of the K-th call_target of that ``CallTargetType`` in
  ``section.call_targets`` (see :func:`_kind_to_called_idx`).
* ``variant_block`` -- the parent's :class:`VariantBlock` for this
  variant; its ``per_call_entries`` table pins ONE
  ``section_variant_index`` per ``called_idx`` (per-variant; EXTERN
  call_targets carry no per_call_entry).
* ``line_to_name`` -- the per-binary
  ``<binary>_function_names.txt`` mapping, used to resolve the
  callee FID -> display name. Lives in
  :mod:`tokenizer.aligned_data.loader.function_names_loader`.

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
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Union

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    CallTarget,
    Section,
    VariantBlock,
)
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.tokens import TokenType

if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.token_manager import VocabularyManager


__all__ = [
    "AsmLine",
    "InlineCallEntry",
    "InlineJumpEntry",
    "KindLabel",
    "LineItem",
    "render_block",
]


# ---------------------------------------------------------------------------
# Typed line items (public API consumed by ``_tree_model``)
# ---------------------------------------------------------------------------


class KindLabel(Enum):
    """Rendering-label discriminator for inline call entries.

    The values mirror the literal strings used in the plan's mockup
    (``call local function K`` / ``call plt function K`` /
    ``call ext function K``); :func:`from_call_target_type` is the
    sole bridge between the codebase's typed
    :class:`CallTargetType` (LOCAL/PLT/EXTERN) and this rendering
    label. Distinct from :class:`CallTargetType` because the latter
    is wire-format with fixed integer values; this enum is a
    presentation concern.
    """

    LOCAL = "local"
    PLT = "plt"
    EXT = "ext"

    @classmethod
    def from_call_target_type(cls, call_type: CallTargetType) -> "KindLabel":
        """Translate the codebase's typed ``CallTargetType`` to the label.

        Closed enum dispatch -- new ``CallTargetType`` members fail
        loud here rather than silently rendering as the default.
        """
        return _CALL_TARGET_TYPE_TO_KIND[call_type]


_CALL_TARGET_TYPE_TO_KIND: dict[CallTargetType, KindLabel] = {
    CallTargetType.LOCAL: KindLabel.LOCAL,
    CallTargetType.PLT: KindLabel.PLT,
    CallTargetType.EXTERN: KindLabel.EXT,
}


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
    ``(arm, idx)`` pair the tree-model layer can hand to
    :func:`batch_decode` when the user expands this node;
    ``None`` for ext calls or for LOCAL/PLT call_targets whose
    ``function_section_ptr`` could not be resolved to a section.
    ``variant_idx`` is the callee variant the parent's
    ``per_call_entries`` pinned for THIS caller variant; equals
    :data:`MISSING_VARIANT_INDEX` when no per_call_entry exists
    (EXTERN) or when the callee section is reachable but lacks a
    variant matching the caller's vkey.
    """

    kind: KindLabel
    counter_id: int
    callee_name: str
    callee_section_pointer: Optional[tuple[SectionKind, int]]
    variant_idx: int


@dataclass(frozen=True)
class InlineJumpEntry:
    """One within-function jump target referenced by an instruction."""

    target_block_idx: int


LineItem = Union[AsmLine, InlineCallEntry, InlineJumpEntry]


# ---------------------------------------------------------------------------
# Section-level indices computed once per render_block call
# ---------------------------------------------------------------------------


def _kind_to_called_idx(section: Section) -> dict[KindLabel, list[int]]:
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
    """
    kind_to_idx: dict[KindLabel, list[int]] = {k: [] for k in KindLabel}
    for called_idx, ct in enumerate(section.call_targets):
        kind_to_idx[KindLabel.from_call_target_type(ct.type)].append(called_idx)
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
    """
    return {
        int(called_idx): int(section_variant_index)
        for called_idx, section_variant_index in variant_block.per_call_entries
    }


# ---------------------------------------------------------------------------
# Per-token classification (typed dispatch, no string parsing)
# ---------------------------------------------------------------------------


_CALL_TOKEN_TYPES: dict[TokenType, KindLabel] = {
    TokenType.LOCAL_FUNC: KindLabel.LOCAL,
    TokenType.PLT_FUNC: KindLabel.PLT,
    TokenType.EXT_FUNC: KindLabel.EXT,
}


_JUMP_TOKEN_TYPES: frozenset[TokenType] = frozenset({TokenType.BLOCK_V2})


def _emit_call_entry(
    *,
    kind: KindLabel,
    counter_id: int,
    section: Section,
    kind_to_called_idx: Mapping[KindLabel, list[int]],
    variant_pins: Mapping[int, int],
    arm: SectionKind,
    callee_arm_resolver,
    line_to_name: Mapping[int, str],
) -> InlineCallEntry:
    """Build one :class:`InlineCallEntry` from a call-site token.

    All branches go through ONE typed dispatch: ``kind`` (a
    :class:`KindLabel`) drives lookups via dicts (``kind_to_called_idx``
    + ``variant_pins``) -- no per-kind ``if/elif`` chains. EXT vs
    LOCAL/PLT pointer-resolution differs ONLY in whether the
    arm resolver yields a section; that single ``Optional`` flows
    straight into ``callee_section_pointer``.
    """
    called_idxs = kind_to_called_idx[kind]
    if 0 <= counter_id < len(called_idxs):
        called_idx = called_idxs[counter_id]
        call_target = section.call_targets[called_idx]
        callee_section_pointer = callee_arm_resolver(call_target, arm)
        variant_idx = variant_pins.get(called_idx, MISSING_VARIANT_INDEX)
        callee_name = line_to_name.get(call_target.function_name_ptr, "?")
    else:
        # Counter id has no matching call_target -- the token's id
        # outran the section's per-kind block. Surface as unresolved
        # rather than silently swallowing; the inspector is a
        # diagnostic tool, "?" is the legitimate display.
        callee_section_pointer = None
        variant_idx = MISSING_VARIANT_INDEX
        callee_name = "?"

    return InlineCallEntry(
        kind=kind,
        counter_id=counter_id,
        callee_name=callee_name,
        callee_section_pointer=callee_section_pointer,
        variant_idx=variant_idx,
    )


def _default_callee_arm_resolver(
    call_target: CallTarget, arm: SectionKind
) -> Optional[tuple[SectionKind, int]]:
    """Stub resolver used when the caller does not supply a session.

    Without a :class:`BinarySession` the inspector cannot map a BIN
    section byte-offset back to a per-arm idx (only the session's
    ``_idx_for_section_offset`` knows the per-arm starts arrays).
    Return ``None`` so downstream nodes render as non-expandable
    leaves; the real path threads a session-backed resolver in via
    the ``callee_arm_resolver`` parameter of :func:`render_block`.
    """
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_block(
    function_data: "FunctionData",
    section: Section,
    variant_block: VariantBlock,
    block_idx: int,
    *,
    arm: SectionKind,
    vocab_manager: "VocabularyManager",
    fid_sidecar: Optional[np.ndarray],
    fid_row_offsets: Optional[np.ndarray],
    line_to_name: Mapping[int, str],
    callee_arm_resolver=_default_callee_arm_resolver,
) -> list[LineItem]:
    """Walk one block's instructions and emit typed line items.

    Reconstructs the :class:`FunctionTokenList` once, iterates to
    ``block_idx``, then per instruction yields one :class:`AsmLine`
    plus -- in token-stream order WITHIN the instruction -- one
    :class:`InlineCallEntry` per LOCAL_FUNC/PLT_FUNC/EXT_FUNC
    metatoken and one :class:`InlineJumpEntry` per BLOCK_V2 metatoken.
    Token-type discrimination is via the structured
    :class:`TokenType` enum (no string parsing of asm).

    ``arm`` identifies the section's arm and is passed to
    ``callee_arm_resolver`` (default: stub returning ``None``; the
    tree model injects a session-backed resolver wrapping
    :meth:`BinarySession._idx_for_section_offset` so inline calls
    can be expanded). ``vocab_manager`` is required because
    :meth:`FunctionTokenList.reconstruct_func_from_raw_bytes` is
    non-functional without one (the spec omits it from the signature
    but the caller threads it from the session anyway).
    ``fid_sidecar`` / ``fid_row_offsets`` are accepted per the spec
    but unused here -- the per-function pre-remap path via
    ``section.call_targets[i].function_name_ptr -> line_to_name``
    resolves callee names; the sidecar is the batch-row post-remap
    path used by the tree model.
    """
    # Stage 1: build the FunctionTokenList view over the raw FunctionData.
    # ``reconstruct_func_from_raw_bytes`` is the existing single-place
    # codec; calling it here keeps the inspector's read path identical
    # to every other consumer of ``FunctionData`` (see
    # :mod:`tokenizer.memmap_validation._validator_mismatch_report`).
    func_tokens = FunctionTokenList.reconstruct_func_from_raw_bytes(
        function_data.tokens,
        function_data.block_runlength,
        function_data.insn_runlength,
        vocab_manager,
    )

    if block_idx < 0 or block_idx >= func_tokens.block_count:
        raise IndexError(
            f"block_idx={block_idx} out of bounds "
            f"(0 <= block_idx < {func_tokens.block_count})"
        )

    # Stage 2: precompute section-level indices once -- the per-instruction
    # walk only does O(1) dict lookups thereafter.
    kind_to_called_idx = _kind_to_called_idx(section)
    variant_pins = _variant_index_for_called_idx(variant_block)

    # Stage 3: iterate to the target block and walk its instructions.
    # ``iter_blocks(transient=True)`` reuses one BlockTokenList object,
    # matching the lazy-view discipline (no copy of metatoken arrays).
    target_block = None
    for i, block in enumerate(func_tokens.iter_blocks(transient=True)):
        if i == block_idx:
            target_block = block
            break
    if target_block is None:
        # Defensive -- the bounds check above guarantees we hit the
        # block, but iter_blocks could short-circuit on a corrupt
        # function (e.g. block_count > yielded count). Surface as a
        # typed error instead of silently returning [].
        raise RuntimeError(
            f"iter_blocks() yielded fewer than {block_idx + 1} blocks "
            f"for function with block_count={func_tokens.block_count}"
        )

    return list(
        _walk_block_instructions(
            target_block,
            arm=arm,
            section=section,
            kind_to_called_idx=kind_to_called_idx,
            variant_pins=variant_pins,
            line_to_name=line_to_name,
            callee_arm_resolver=callee_arm_resolver,
        )
    )


def _walk_block_instructions(
    block,
    *,
    arm: SectionKind,
    section: Section,
    kind_to_called_idx: Mapping[KindLabel, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    callee_arm_resolver,
) -> Iterable[LineItem]:
    """Per-instruction generator: one :class:`AsmLine` then any inline
    call/jump entries from the instruction's metatoken stream."""
    for insn in block.iter_insn(transient=True):
        yield AsmLine(text=insn.to_asm_like())
        yield from _emit_inline_entries(
            insn,
            arm=arm,
            section=section,
            kind_to_called_idx=kind_to_called_idx,
            variant_pins=variant_pins,
            line_to_name=line_to_name,
            callee_arm_resolver=callee_arm_resolver,
        )


def _emit_inline_entries(
    insn,
    *,
    arm: SectionKind,
    section: Section,
    kind_to_called_idx: Mapping[KindLabel, list[int]],
    variant_pins: Mapping[int, int],
    line_to_name: Mapping[int, str],
    callee_arm_resolver,
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
                arm=arm,
                callee_arm_resolver=callee_arm_resolver,
                line_to_name=line_to_name,
            )
            continue
        if token_type in _JUMP_TOKEN_TYPES:
            yield InlineJumpEntry(target_block_idx=int(token.id))
