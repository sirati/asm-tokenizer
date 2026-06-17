"""Per-(parent variant, call_target slot) callee resolution.

Single concern: resolve ONE call_target row of a parent section+variant
to its splice callee. The resolution is in TWO stages so the body load
follows the once-only prune instead of preceding it:

* :func:`resolve_callee_metadata` -- the DECISION. Returns everything the
  BFS needs to drive the shared once-only inclusion decider (the callee
  :class:`Section`, its ``section_offset`` once-only dedup key, the
  J-resolved callee variant index, the per-arm load locator) WITHOUT
  touching ``_data.bin``. Derived from the cheap ``_sections.bin`` parse
  alone, since :func:`choose_callee_variant` reads only the PARENT's
  per-call entries and the callee body is never inspected to decide
  inclusion.
* :func:`load_callee_body` -- the LOAD. Materialises the chosen callee
  variant body, issued only for the survivor pairs the BFS actually
  emits + descends (the pruned / multi-parent-deduped edges never pay
  the body read + egress copy).

Both stages are UNCHANGED from the legacy DFS walker's J-resolution (the
fallback/override chain in :func:`choose_callee_variant`, the 0xFFFE
missing-vkey skip, the EXTERN / unresolved-pointer / unknown-offset
gates) -- only the once-only-visited and inlining-equivalence gates are
gone, having moved to the shared inclusion decider
(:mod:`...splice_inclusion`). Splitting decision from load changes WHEN
``_data.bin`` is read, never WHICH bytes splice.

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

from ._section_meta_memo import CalleeSectionMetaMemo

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from tokenizer.aligned_data.loader.function_data import FunctionData
    from tokenizer.aligned_data.loader.session import BinarySession


__all__ = ["ResolvedCalleeMeta", "resolve_callee_metadata", "load_callee_body"]


@dataclass(frozen=True)
class ResolvedCalleeMeta:
    """One spliceable callee edge of a parent (section, variant) -- the
    metadata-only decision (no body loaded).

    ``section_offset`` is the once-only dedup key the BFS feeds the
    shared decider; ``call_target_type`` drives the self-prepend
    :class:`Category` at emission. ``callee_idx`` is the per-arm load
    locator :func:`load_callee_body` uses to materialise the chosen
    ``variant_idx`` body once the prune retains this edge.

    ``is_matched`` is the resolving call_target's BIN flag (the callee's
    arm): ``True`` when the callee resides in the matched arm. It is read
    by the unmatched-outline inlining transform
    (:mod:`...splice_inclusion._unmatched_expand`) to decide whether an
    edge is a direct matched target or an unmatched outline to look
    THROUGH; it does NOT affect any other resolution gate.
    """

    section: Section
    variant_idx: int
    section_offset: int
    function_name_ptr: int
    call_target_type: CallTargetType
    callee_idx: int
    is_matched: bool


def resolve_callee_metadata(
    *,
    session: "BinarySession",
    arm: SectionKind,
    parent_section: Section,
    parent_variant_idx: int,
    parent_sibling_v_idxs: frozenset,
    called_idx: int,
    ct: CallTarget,
    section_meta_memo: "CalleeSectionMetaMemo",
) -> Optional[ResolvedCalleeMeta]:
    """Decide one call_target row's callee edge, or ``None`` to skip.

    No ``_data.bin`` touch: only the callee section's ``_sections.bin``
    catalog entry is parsed (for the ``section_offset`` dedup key and the
    next-level call_target table) and the parent's per-call entries drive
    :func:`choose_callee_variant`. Skip reasons (matched against the
    legacy splice walker's gates, MINUS the now-shared visited / inlining
    gates):

    * Extern call site (``ct.type is CallTargetType.EXTERN``) -- D3
      prohibits inlining extern bodies.
    * Unresolved pointer (``ct.function_section_ptr == 0``).
    * Cross-arm or missing section (``_idx_for_section_offset`` ->
      ``None``).
    * No usable callee variant (``choose_callee_variant`` -> ``None`` --
      missing-vkey 0xFFFE at every fallback level).
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

    # Per-arm metadata-only parse: matched parses the catalog entry;
    # unmatched parses the owning section of the first record. Neither
    # touches ``_data.bin`` -- the chosen variant body is loaded later
    # only for survivors via :func:`load_callee_body`. The per-walk memo
    # owns the meta-method dispatch + parses each ``(arm, idx)`` once, so
    # the same callee reached across sibling parents / BFS levels /
    # sampled variants does not re-run ``parse_section_bin``.
    callee_section, _callee_section_offset = section_meta_memo.section_meta(
        session, arm, callee_idx
    )

    callee_variant_idx = choose_callee_variant(
        parent_section,
        parent_variant_idx,
        called_by_set,
        called_idx,
    )
    if callee_variant_idx is None:
        return None

    return ResolvedCalleeMeta(
        section=callee_section,
        variant_idx=callee_variant_idx,
        section_offset=int(callee_section.section_offset),
        function_name_ptr=int(ct.function_name_ptr),
        call_target_type=ct.type,
        callee_idx=callee_idx,
        is_matched=bool(ct.is_matched),
    )


def load_callee_body(
    session: "BinarySession",
    arm: SectionKind,
    meta: ResolvedCalleeMeta,
) -> "FunctionData":
    """Materialise the chosen callee variant body for a retained edge.

    Issued only for the survivor pairs the BFS emits/descends. Both arms
    load exactly ``section.variants[meta.variant_idx]`` -- the same
    single-variant parse the all-variants path runs per variant. The
    unmatched arm stores one DISTINCT body record per variant, so the
    J-resolved ``variant_idx`` is threaded through (rather than always
    loading the first record) and the body is sliced at that variant
    block's own ``data_offset_shifted`` -- symmetric with the matched arm.

    The already-parsed ``meta.section`` is threaded into both arms'
    body loads so ``_sections.bin`` is NOT re-parsed here: the metadata
    stage (:func:`resolve_callee_metadata`) already paid the catalog
    parse, and re-deriving the section from the index would discard and
    rebuild it (a re-parse-in-call-chain violation).
    """
    if arm is SectionKind.MATCHED:
        return session._load_matched_variant_body(
            meta.callee_idx, meta.variant_idx, meta.section
        )
    return session._load_unmatched_variant_body(
        meta.callee_idx, meta.variant_idx, meta.section
    )
