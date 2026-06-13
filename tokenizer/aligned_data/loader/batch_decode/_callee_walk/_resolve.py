"""Per-(parent variant, call_target slot) callee resolution.

Single concern: resolve ONE call_target row of a parent section+variant
to its splice callee -- the callee :class:`Section`, the chosen callee
variant index, and that variant's :class:`FunctionData` -- or ``None``
when the row is not spliceable. This is exactly the walker's per-edge
J-resolution; it is UNCHANGED from the legacy DFS walker (the
fallback/override chain in :func:`choose_callee_variant`, the 0xFFFE
missing-vkey skip, the EXTERN / unresolved-pointer / unknown-offset
gates) -- only the once-only-visited and inlining-equivalence gates are
gone, having moved to the shared inclusion decider
(:mod:`...splice_inclusion`).

The returned ``callee_section.section_offset`` is the once-only dedup
key the BFS feeds the shared decider (the legacy active-path cycle key
was ``(arm, section_offset)``; the walk only ever splices within one
arm, so the offset alone identifies the callee function).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.decoded._variant_selection import (
    called_by_in_selection,
    choose_callee_variant,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import CallTarget, Section

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.session import BinarySession


__all__ = ["ResolvedCallee", "resolve_callee"]


@dataclass(frozen=True)
class ResolvedCallee:
    """One spliceable callee edge of a parent (section, variant).

    ``section_offset`` is the once-only dedup key; ``function_data`` is
    the chosen callee variant body (emission input); ``call_target_type``
    drives the self-prepend :class:`Category` at emission.
    """

    section: Section
    variant_idx: int
    function_data: "FunctionData"
    section_offset: int
    function_name_ptr: int
    call_target_type: CallTargetType


def resolve_callee(
    *,
    session: "BinarySession",
    arm: SectionKind,
    parent_section: Section,
    parent_variant_idx: int,
    parent_sibling_v_idxs: frozenset,
    called_idx: int,
    ct: CallTarget,
) -> Optional[ResolvedCallee]:
    """Resolve one call_target row to its callee, or ``None`` to skip.

    Skip reasons (matched against the legacy splice walker's gates,
    MINUS the now-shared visited / inlining gates):

    * Extern call site (``ct.type is CallTargetType.EXTERN``) -- D3
      prohibits inlining extern bodies.
    * Unresolved pointer (``ct.function_section_ptr == 0``).
    * Cross-arm or missing section (``_idx_for_section_offset`` ->
      ``None``).
    * No usable callee variant (``choose_callee_variant`` -> ``None`` --
      missing-vkey 0xFFFE at every fallback level).

    The callee section is loaded BEFORE the variant choice so the chosen
    callee variant body is available to splice.
    """
    if ct.type is CallTargetType.EXTERN:
        return None
    if ct.function_section_ptr == 0:
        return None

    callee_byte_offset = int(ct.function_section_ptr)
    called_by_set = called_by_in_selection(
        parent_section, parent_sibling_v_idxs, called_idx
    )

    callee_idx = session._idx_for_section_offset(callee_byte_offset, arm.value)
    if callee_idx is None:
        return None

    # Per-arm J-free load: matched loads the section + every variant
    # body; unmatched loads its single record + owning section. Both
    # surface the callee section the chosen variant body is read from.
    if arm is SectionKind.MATCHED:
        callee_section, _callee_section_offset, callee_matched = (
            session._load_matched_section_and_variants(callee_idx)
        )
    else:
        callee_fd, callee_section, _callee_section_offset = (
            session._load_unmatched_for_splice(callee_idx)
        )

    callee_variant_idx = choose_callee_variant(
        parent_section,
        parent_variant_idx,
        called_by_set,
        called_idx,
    )
    if callee_variant_idx is None:
        return None

    if arm is SectionKind.MATCHED:
        callee_fd = callee_matched.variants[callee_variant_idx]

    return ResolvedCallee(
        section=callee_section,
        variant_idx=callee_variant_idx,
        function_data=callee_fd,
        section_offset=int(callee_section.section_offset),
        function_name_ptr=int(ct.function_name_ptr),
        call_target_type=ct.type,
    )
