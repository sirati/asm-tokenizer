"""Per-variant FTL parse cache for :class:`FtlBackend`.

Single concern: the once-per-(function, variant) parse pipeline --
take one :class:`ParsedRecord` and produce the in-memory state
:func:`tokenizer.inspector._render._render_block.render_block`
consumes (a :class:`BlockTokenList` tuple, the
:class:`FtlSectionView`, the per-kind ``kind_to_called_idx`` table,
the ``line_to_name`` + ``line_to_provider`` Mappings).

Plan v2 ``F-MED-10`` splits this concern out of ``_handles.py`` (the
prior plan revision tangled per-variant parse state into the handle
class). ``F-CRIT-2`` mandates that the
:class:`~tokenizer.token_manager.VocabularyManager` is bound at
construction time -- the per-variant CSV's own vocab travels with the
:class:`VariantState` and the backend never accepts an external
``(tokens, vocab)`` tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.parsed_record_iter import ParsedRecord
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager

from ._ftl_section_view import (
    FtlSectionView,
    build_section_view_from_record,
)


__all__ = [
    "VariantState",
    "build_variant_state",
]


@dataclass(frozen=True)
class VariantState:
    """All per-(function, variant) state the renderer consumes.

    ``record`` is the source :class:`ParsedRecord` -- kept for
    diagnostics + future Phase-2 consumers (e.g. strings sidecar
    rendering); the renderer itself reads only the derived fields
    below.

    ``vocab`` is the per-CSV :class:`VocabularyManager`. Bound at
    construction so a stray cross-vocab decode is impossible.

    ``ftl`` is the parsed :class:`FunctionTokenList` (single parse;
    the ``blocks`` tuple is materialised with ``transient=False`` so
    each block is a stash-safe distinct view -- the
    ``transient=True`` form reuses one mutable view across iterations).

    ``view`` is the typed FTL-side section view (see
    :mod:`._ftl_section_view`).

    ``kind_to_called_idx`` partitions the view's call_targets list by
    :class:`CallTargetType` -- one list per kind, holding the
    positions into ``view.call_targets`` of the K-th LOCAL / PLT /
    EXTERN call site (matches the encoder's per-Category counter
    walk).

    ``line_to_name`` maps the flat ``function_name_ptr`` (= position in
    the encoder-allocated ``called_funcs`` list) back to a display
    name. ``line_to_provider`` maps the 1-indexed EXTERN slot (matching
    ``FtlCallTarget.function_section_ptr`` for EXTERN entries) to its
    library string; LOCAL / PLT slots are absent.
    """

    record: ParsedRecord
    vocab: VocabularyManager
    ftl: FunctionTokenList
    view: FtlSectionView
    blocks: Tuple[BlockTokenList, ...]
    kind_to_called_idx: Mapping[CallTargetType, list[int]]
    line_to_name: Mapping[int, str]
    line_to_provider: Mapping[int, str]


def build_variant_state(
    record: ParsedRecord,
    vocab: VocabularyManager,
) -> VariantState:
    """Produce the per-variant state from one parsed record + vocab.

    Steps:

    1. Build the typed section view (see
       :func:`build_section_view_from_record`) -- includes the
       category-order assert per ``F-MED-14``.
    2. Partition the view's call_targets by
       :class:`CallTargetType` to derive the per-kind index lists
       :func:`render_block` consumes.
    3. Reconstruct the :class:`FunctionTokenList` from the record's
       runlength-encoded raw bytes (the same helper the v2
       dataloader uses) bound to the per-CSV ``vocab``; materialise
       block views with ``transient=False`` so stashes are safe.
    4. Build the flat ``line_to_name`` Mapping (flat-idx -> name) and
       the 1-indexed ``line_to_provider`` Mapping (EXTERN-slot ->
       library) per plan decisions #28 + the F-HIGH-4 fix.
    """
    view = build_section_view_from_record(record)
    kind_to_called_idx = _partition_kinds(view)
    ftl = FunctionTokenList.reconstruct_func_from_raw_bytes(
        record.tokens,
        record.block_runlength,
        record.insn_runlength,
        vocab_manager=vocab,
    )
    blocks = tuple(ftl.iter_blocks(transient=False))
    line_to_name: Dict[int, str] = {
        flat_idx: name for flat_idx, (name, _ct) in enumerate(record.called_funcs)
    }
    line_to_provider = _build_line_to_provider(
        record.called_funcs, record.extern_libraries
    )
    return VariantState(
        record=record,
        vocab=vocab,
        ftl=ftl,
        view=view,
        blocks=blocks,
        kind_to_called_idx=kind_to_called_idx,
        line_to_name=line_to_name,
        line_to_provider=line_to_provider,
    )


def _build_line_to_provider(
    called_funcs: list[tuple[str, CallTargetType]],
    extern_libraries: Mapping[str, str],
) -> Dict[int, str]:
    """Build the 1-indexed ``EXTERN-slot -> library`` lookup.

    Keys match :attr:`FtlCallTarget.function_section_ptr` for EXTERN
    entries (see :func:`build_section_view_from_record`). LOCAL / PLT
    callees do not appear -- the renderer routes their kinds through
    the empty provider mapping (see ``_render_block._provider_sources``).
    """
    line_to_provider: Dict[int, str] = {}
    extern_slot = 0
    for name, ct_type in called_funcs:
        if ct_type is CallTargetType.EXTERN:
            extern_slot += 1
            library = extern_libraries.get(name)
            if library is not None:
                line_to_provider[extern_slot] = library
    return line_to_provider


def _partition_kinds(view: FtlSectionView) -> Mapping[CallTargetType, list[int]]:
    """Per-kind index lists into ``view.call_targets``.

    Mirrors :func:`_render_block._kind_to_called_idx` but operates on
    the FTL-side view (the renderer's helper expects the writer-side
    :class:`Section`). The K-th element of ``kind_to_idx[kind]`` is
    the position of the K-th distinct call_target of that
    ``CallTargetType`` -- matches the encoder's per-Category counter.
    """
    kind_to_idx: Dict[CallTargetType, list[int]] = {k: [] for k in CallTargetType}
    for called_idx, ct in enumerate(view.call_targets):
        kind_to_idx[ct.type].append(called_idx)
    return kind_to_idx
