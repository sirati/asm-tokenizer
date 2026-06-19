"""Section pointer resolution + RNG variant sampling for stage 1.

Single concern of this module: walk a request's
``list[SectionPointerSpec]``, dispatch each pointer through
:class:`BinarySession`'s per-arm load helpers, and pick the variant
indices that the downstream stage-1 wiring will load FunctionData /
InlineDecodeState for. Per the call-chain design the resolver also
harvests the per-sampled-variant :class:`FunctionData` from the same
load it issues for the :class:`Section`, so the wiring (1d) does NOT
re-parse the same section/record a second time.

What this module owns (the boundary):

* Input: a session + the request's section-pointer list + RNG knobs.
* Output: a parallel ``list[ResolvedSection]`` -- one entry per
  pointer, in the same order. Each entry carries the parsed
  :class:`Section`, the RNG-sampled variant indices in encounter order,
  and the parallel per-sampled-variant :class:`FunctionData` list.

What this module does NOT own:

* Callee discovery / DFS recursion -- handled by the sibling 1b
  module.
* ``batch_idx`` assignment (the ALG-10 padding-policy layout) --
  handled by the sibling 1c module.
* :class:`Stage1Variant` / :class:`Stage1CallTarget` construction --
  composed by the 1d wiring step.

``ResolvedSection`` is deliberately NOT placed on the pipeline-handoff
shape file (``_types.py``): that file describes the four stage-boundary
shapes the whole pipeline contracts on. ``ResolvedSection`` is a
sub-module helper -- the 1a -> 1b/1c handoff -- visible only inside
the stage-1 wiring step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

from ..metadata_loader import SectionKind

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..function_data import FunctionData
    from ..session import BinarySession

from ...matched_sections_bin import Section
from ._types import SectionPointerSpec
from ._variant_selection import CountThenRNGSelection


__all__ = [
    "ResolvedSection",
    "resolve_section_pointers",
]


@dataclass(frozen=True)
class ResolvedSection:
    """1a output: section identity + parsed Section + RNG-sampled variant indices
    + per-sampled-variant FunctionData.

    The downstream wiring (1d / Phase 4) constructs
    :class:`Stage1Section` from this plus 1b's per-variant callee walk
    results. :class:`ResolvedSection` itself does NOT contain
    :class:`Stage1Variant` instances -- it is the pre-load handoff.

    Attributes:
        arm: :attr:`SectionKind.MATCHED` or :attr:`SectionKind.UNMATCHED`;
            mirrors the originating :attr:`SectionPointerSpec.arm`.
        idx: Per-arm function / section idx; mirrors
            :attr:`SectionPointerSpec.idx`.
        section: The BIN's parsed :class:`Section` (call_targets table
            + variant blocks). Read ``section.section_offset`` for the
            BIN-side byte offset; do NOT store it as a separate field
            (single source of truth -- see the audit-fix commit that
            dropped ``Stage1Section.section_offset``).
        sampled_variant_indices: The RNG-selected variant indices in
            their existing encounter order (``_select_variant_indices``
            sorts its sampled output, and returns ``range(n)`` when the
            request covers every variant). Length is bounded by
            ``len(section.variants)`` regardless of arm -- unmatched
            sections store one record per variant and the loader
            harvests every record.
        function_data_per_sampled_variant: Parallel to
            ``sampled_variant_indices`` -- entry ``v`` is the
            :class:`FunctionData` for the variant body identified by
            ``sampled_variant_indices[v]``. Harvested from the same
            per-arm load that produced :attr:`section`, so the wiring
            does NOT re-parse the section to pick up the variant body.
            EMPTY when the resolver ran with ``load_bodies=False`` (the
            geometry-first vector_batch path, which never reads it).
    """

    arm: SectionKind
    idx: int
    section: Section
    sampled_variant_indices: List[int]
    function_data_per_sampled_variant: List["FunctionData"]


def resolve_section_pointers(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    rng: np.random.Generator,
    load_bodies: bool = True,
) -> List[ResolvedSection]:
    """Resolve each ``SectionPointerSpec`` and sample variant indices.

    For each pointer in ``section_pointers`` (in order):

    * Parse the pointer's :class:`Section` catalog ONCE
      (:func:`_parse_section_catalog`), body-free -- it reads
      ``_sections.bin`` but never ``_data.bin``.
    * Sample the variant indices off ``len(section.variants)``.
    * Load a :class:`FunctionData` body for the SAMPLED indices ONLY
      (:func:`_load_sampled_variant_bodies`), reusing the already-parsed
      catalog. The UNSAMPLED variants' body parse + ``category_counts``
      are never computed -- the dominant decode-path saving, since a
      request keeps ``num_variants_per_section`` of a section's full
      variant set. Both arms route through the session's per-arm plural
      body loader, so each kept body is byte-identical to the eager
      all-variants path's ``variants[idx]``.

    Variant index sampling uses
    :func:`_session_helpers._select_variant_indices` -- one source of
    truth for the rng-driven without-replacement sampling rule.

    Args:
        session: The :class:`BinarySession` holding the open binary.
        section_pointers: One :class:`SectionPointerSpec` per requested
            section. Order is preserved in the output list.
        num_variants_per_section: Upper bound on how many variants to
            sample per section. ``_select_variant_indices`` clamps to
            ``min(num_variants_per_section, n_variants)`` and returns
            ``range(n_variants)`` (no shuffle) when the bound covers
            every variant; otherwise it samples without replacement
            and sorts for determinism.
        rng: Sampling source. Passed through to
            :func:`_select_variant_indices` verbatim; callers seed for
            reproducibility.
        load_bodies: When ``True`` (default) the sampled variant bodies
            are loaded into ``function_data_per_sampled_variant`` (the
            decode + auto-size paths need them). When ``False`` only the
            catalog parse + variant-index sampling run; the body list is
            left EMPTY -- the geometry-first vector_batch path gathers its
            own bodies via the RLG3 geometry and never reads
            ``function_data_per_sampled_variant``, so its per-variant body
            parse + ``category_counts`` is dead work. The sampling +
            ``section`` + ``sampled_variant_indices`` are identical either
            way, so the rng draw stays in lockstep with the body-loading
            path.

    Returns:
        ``list[ResolvedSection]`` parallel to ``section_pointers``.
        For each entry the ``sampled_variant_indices`` list holds
        Python ``int`` (not ``numpy.int64``) so downstream consumers
        can use the values as plain list indices, and the
        ``function_data_per_sampled_variant`` list is the parallel
        per-variant body list.

    Raises:
        ValueError: If a pointer's ``arm`` is not a known
            :class:`SectionKind` member. (Passes through any
            :class:`IndexError` from the session's load helpers when a
            per-arm idx is out of range.)
    """

    resolved: List[ResolvedSection] = []
    for pointer in section_pointers:
        section = _parse_section_catalog(session, pointer)
        # Null-object collapse: a pointer that pins no explicit selection
        # falls back to the count-then-RNG strategy with the request's
        # ``num_variants_per_section``. The default branch hits
        # ``CountThenRNGSelection`` -> ``_select_variant_indices`` with the
        # identical arguments the resolver passed before this seam existed,
        # so the count path stays byte-identical. The validation path rides
        # an ``ExplicitIndicesSelection`` on the pointer instead -- this
        # resolver never learns which it got.
        selection = pointer.variant_selection or CountThenRNGSelection(
            num_variants_per_section
        )
        sampled = selection.select(n_variants=len(section.variants), rng=rng)
        # ``_select_variant_indices`` returns ``np.ndarray[int64]``;
        # convert each element to a Python ``int`` so the downstream
        # 1d wiring can use the indices as plain list indices without
        # numpy-typing surprises (e.g. ``MatchedFunction.variants`` is
        # a plain Python list).
        sampled_ints = [int(v) for v in sampled]
        # Body load is SAMPLED-ONLY: parse + category-count exactly the
        # variants the draw kept, not every ``section.variants`` entry.
        # ``_parse_section_catalog`` already paid the (body-free) catalog
        # parse, so each body is the single-variant load the lazy session
        # helper performs -- byte-identical to the eager all-variants path.
        resolved.append(
            ResolvedSection(
                arm=pointer.arm,
                idx=pointer.idx,
                section=section,
                sampled_variant_indices=sampled_ints,
                function_data_per_sampled_variant=_load_sampled_variant_bodies(
                    session, pointer, section, sampled_ints
                )
                if load_bodies
                else [],
            )
        )
    return resolved


def _load_sampled_variant_bodies(
    session: "BinarySession",
    pointer: SectionPointerSpec,
    section: Section,
    sampled_variant_indices: List[int],
) -> List["FunctionData"]:
    """Load the sampled variant bodies for one pointer's section.

    Arm-dispatch only: routes the whole sampled-slot set through the
    session's plural body loader for that arm
    (:py:meth:`BinarySession._load_matched_variant_bodies` /
    :py:meth:`_load_unmatched_variant_bodies`). The unmatched plural
    loader builds the section-wide resolve bundle ONCE for the section and
    threads it into every slot, so the whole-section variant resolve +
    call_target flatten happens once per section rather than once per
    sampled slot (O(V) vs O(V²)); the hoist is owned entirely by the
    session, so this resolver stays arm-agnostic and unaware of the
    bundle. Output is parallel to ``sampled_variant_indices`` and
    byte-identical to the per-slot path.

    Raises:
        ValueError: On an unknown :class:`SectionKind` member.
    """
    if pointer.arm is SectionKind.MATCHED:
        return session._load_matched_variant_bodies(
            pointer.idx, section, sampled_variant_indices
        )
    if pointer.arm is SectionKind.UNMATCHED:
        return session._load_unmatched_variant_bodies(
            pointer.idx, section, sampled_variant_indices
        )
    raise ValueError(f"unknown SectionKind: {pointer.arm!r}")


def _parse_section_catalog(
    session: "BinarySession", pointer: SectionPointerSpec
) -> Section:
    """Parse a single pointer's :class:`Section` catalog -- NO bodies.

    Arm-dispatch only: routes to the session's body-free catalog parse
    (:py:meth:`BinarySession._matched_section_meta` /
    :py:meth:`_unmatched_section_meta`), which reads ``_sections.bin``
    but never touches ``_data.bin``. The sampling step keys solely on
    ``len(section.variants)``, so the body load is deferred to the
    sampled survivors via :func:`_load_sampled_variant_bodies`.

    Kept module-private + arm-dispatch-only so
    :func:`resolve_section_pointers` stays a clean walk over the input
    list. The two catalog helpers share the ``(section, section_offset)``
    return shape; the offset is not needed here (the resolver reads
    ``section.section_offset`` directly), so only the section is surfaced.

    Raises:
        ValueError: On an unknown :class:`SectionKind` member.
    """
    if pointer.arm is SectionKind.MATCHED:
        section, _section_offset = session._matched_section_meta(pointer.idx)
        return section
    if pointer.arm is SectionKind.UNMATCHED:
        section, _section_offset = session._unmatched_section_meta(pointer.idx)
        return section
    raise ValueError(f"unknown SectionKind: {pointer.arm!r}")


