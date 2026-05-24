"""Typed FTL-side section view consumed by ``_render_block``.

Single concern: build the minimum-surface object the shared
:func:`tokenizer.inspector._render._render_block.render_block` body
needs in its ``section`` slot when rendering from FTL CSV parses --
NOT a fake :class:`~tokenizer.aligned_data.matched_sections_bin.Section`
shim.

Plan v2 audit ``F-HIGH-3`` mandates this: the renderer's section input
must be a typed view exposing exactly the fields the renderer reads
(``section.call_targets[i].function_name_ptr``,
``section.call_targets[i].function_section_ptr``), and nothing else.

``F-HIGH-4`` (the EXTERN provider bug) is fixed here: each EXTERN
``FtlCallTarget`` carries its 1-indexed slot in the
``parsed_record.extern_libraries`` ordering as
``function_section_ptr`` -- the v1 "uniform 0" silently collapsed
every EXTERN row to one provider. The renderer's
``line_to_provider.get(function_section_ptr)`` lookup
(``_render_block:_emit_call_entry``) reads the same field.

LOCAL / PLT call_targets have no provider; the renderer's
per-kind dispatch table threads the empty provider mapping for them
(``_render_block:_provider_sources``) so the value of their
``function_section_ptr`` is read but never produces a hit. We set it
to the flat-name index so a misrouted lookup would still land on
*something* and surface as a name-mismatch rather than a silent zero
fan-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.parsed_record_iter import ParsedRecord, _V2_CATEGORY_TYPES


__all__ = [
    "FtlCallTarget",
    "FtlSectionView",
    "build_section_view_from_record",
]


@dataclass(frozen=True)
class FtlCallTarget:
    """Typed call_target entry from one FTL parse.

    Mirrors the field set
    :func:`tokenizer.inspector._render._render_block.render_block`
    reads off ``section.call_targets[i]`` -- ``function_name_ptr`` and
    ``function_section_ptr``. ``type`` is exposed for completeness
    (the renderer routes per-kind via ``kind_to_called_idx``; ``type``
    isn't accessed on the call_target itself, but threading it keeps
    the view self-describing for tests + future Phase-2 consumers).

    ``is_matched`` is the dead field on the writer-side
    :class:`~tokenizer.aligned_data.matched_sections_bin.CallTarget`
    that the renderer also never reads; mirrored as ``False`` so the
    view doesn't grow a third surface inconsistent with the canonical
    dataclass.
    """

    function_name_ptr: int
    function_section_ptr: int
    type: CallTargetType
    is_matched: bool = False


@dataclass(frozen=True)
class FtlSectionView:
    """Typed view over one FTL :class:`ParsedRecord`'s call-target table.

    The renderer reads ``section.call_targets`` and indexes into it via
    the per-kind ``kind_to_called_idx`` table the caller supplies
    separately. Per the plan, FtlBackend builds both from the
    encoder-allocation-ordered ``ParsedRecord.called_funcs``.
    """

    call_targets: tuple[FtlCallTarget, ...]


def build_section_view_from_record(record: ParsedRecord) -> FtlSectionView:
    """Construct the typed view from one parsed FTL record.

    ``ParsedRecord.called_funcs`` is already in the encoder's
    LOCAL -> PLT -> EXTERN category order (see
    :data:`tokenizer.aligned_data.parsed_record_iter._V2_CATEGORY_TYPES`),
    with order preserved within each category. The flat position in
    ``called_funcs`` is therefore the ``function_name_ptr`` the
    renderer keys into via ``line_to_name``.

    EXTERN slot offsets: the per-category counter for EXTERN starts at
    1 (the encoder's ``extern_provider_line_no`` is 1-indexed; see
    :class:`~tokenizer.aligned_data.matched_sections_bin.CallTargetSpec`).
    So the K-th EXTERN entry (0-indexed) gets ``function_section_ptr =
    K + 1`` -- the same 1-indexed line the
    ``line_to_provider`` map is keyed by (built by the caller from
    ``record.extern_libraries``).

    ``F-MED-14`` runtime assert: the order of categories in
    ``called_funcs`` must match :data:`_V2_CATEGORY_TYPES`. The
    encoder's iterator (:func:`called_from_v2_metadata`) walks
    ``_V2_CATEGORY_TYPES`` in fixed order, so any mismatch points at
    a drift between this view and the parser. We assert at
    construction; the inspector is a diagnostic tool, surfacing the
    drift loudly is preferable to silently mis-routing every EXTERN
    row.
    """
    expected_order = tuple(t for _key, t in _V2_CATEGORY_TYPES)

    targets: list[FtlCallTarget] = []
    extern_slot = 0  # 0-indexed counter within EXTERN category
    seen_category_order: list[CallTargetType] = []
    last_category: CallTargetType | None = None

    for flat_idx, (_name, ct_type) in enumerate(record.called_funcs):
        if ct_type != last_category:
            seen_category_order.append(ct_type)
            last_category = ct_type
        if ct_type is CallTargetType.EXTERN:
            extern_slot += 1
            section_ptr = extern_slot
        else:
            # LOCAL / PLT: documented unused by the renderer; setting
            # to the flat-name index keeps the view diagnosable.
            section_ptr = flat_idx
        targets.append(
            FtlCallTarget(
                function_name_ptr=flat_idx,
                function_section_ptr=section_ptr,
                type=ct_type,
                is_matched=False,
            )
        )

    # F-MED-14 runtime assert: category order matches the parser's
    # fixed walk order (after deduping repeats since called_funcs has
    # already collapsed within-category duplicates per
    # ``called_from_v2_metadata``).
    seen_unique = tuple(
        t for i, t in enumerate(seen_category_order)
        if i == 0 or t != seen_category_order[i - 1]
    )
    # ``seen_unique`` is a SUBSEQUENCE of ``expected_order`` (some
    # categories may simply not appear in this function); subsequence
    # check via two-pointer walk.
    j = 0
    for t in seen_unique:
        while j < len(expected_order) and expected_order[j] != t:
            j += 1
        assert j < len(expected_order), (
            f"called_funcs category order {seen_unique!r} not a subsequence "
            f"of _V2_CATEGORY_TYPES order {expected_order!r}"
        )
        j += 1

    return FtlSectionView(call_targets=tuple(targets))
