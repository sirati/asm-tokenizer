"""Section pointer resolution + RNG variant sampling for stage 1.

Single concern of this module: walk a request's
``list[SectionPointerSpec]``, dispatch each pointer through
:class:`BinarySession`'s per-arm load helpers, and pick the variant
indices that the downstream stage-1 wiring will load FunctionData /
InlineDecodeState for.

What this module owns (the boundary):

* Input: a session + the request's section-pointer list + RNG knobs.
* Output: a parallel ``list[ResolvedSection]`` -- one entry per
  pointer, in the same order. Each entry carries the parsed
  :class:`Section` plus the RNG-sampled variant indices in
  encounter order.

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
from .._session_splice import _select_variant_indices

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..session import BinarySession

from ...matched_sections_bin import Section
from ._types import SectionPointerSpec


__all__ = [
    "ResolvedSection",
    "resolve_section_pointers",
]


@dataclass(frozen=True)
class ResolvedSection:
    """1a output: section identity + parsed Section + RNG-sampled variant indices.

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
            request covers every variant). For unmatched sections the
            list has at most 1 entry by the matched_sections_bin
            invariant.
    """

    arm: SectionKind
    idx: int
    section: Section
    sampled_variant_indices: List[int]


def resolve_section_pointers(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    rng: np.random.Generator,
) -> List[ResolvedSection]:
    """Resolve each ``SectionPointerSpec`` and sample variant indices.

    For each pointer in ``section_pointers`` (in order):

    * :attr:`SectionKind.MATCHED` -- delegate to
      :py:meth:`BinarySession._load_matched_section_and_variants(idx)`
      which returns ``(Section, section_offset, MatchedFunction)``.
      The ``section_offset`` is read off ``section.section_offset``
      downstream; we discard the redundant tuple element.
    * :attr:`SectionKind.UNMATCHED` -- delegate to
      :py:meth:`BinarySession._load_unmatched_record_and_section(idx)`
      which returns ``(Section, section_offset, FunctionData)``. The
      :class:`FunctionData` is the single root-body load already done
      by the session; the 1d wiring step will reuse it via the same
      load helper. (Re-loading from the session is idempotent on the
      session's caches; we choose NOT to plumb the ``FunctionData``
      through here because that would couple the variant-sampling
      concern to the per-variant function-data load, which is 1d's
      job.)

    Variant index sampling uses
    :func:`_session_splice._select_variant_indices` (the same helper
    the decoded splicer uses), guaranteeing the legacy + new pipelines
    select identically given the same rng state.

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

    Returns:
        ``list[ResolvedSection]`` parallel to ``section_pointers``.
        For each entry the ``sampled_variant_indices`` list holds
        Python ``int`` (not ``numpy.int64``) so downstream consumers
        can use the values as plain list indices.

    Raises:
        ValueError: If a pointer's ``arm`` is not a known
            :class:`SectionKind` member. (Passes through any
            :class:`IndexError` from the session's load helpers when a
            per-arm idx is out of range.)
    """

    resolved: List[ResolvedSection] = []
    for pointer in section_pointers:
        section = _load_section_for_arm(session, pointer)
        sampled = _select_variant_indices(
            n_variants=len(section.variants),
            max_variants=num_variants_per_section,
            rng=rng,
        )
        # ``_select_variant_indices`` returns ``np.ndarray[int64]``;
        # convert each element to a Python ``int`` so the downstream
        # 1d wiring can use the indices as plain list indices without
        # numpy-typing surprises (e.g. ``MatchedFunction.variants`` is
        # a plain Python list).
        resolved.append(
            ResolvedSection(
                arm=pointer.arm,
                idx=pointer.idx,
                section=section,
                sampled_variant_indices=[int(v) for v in sampled],
            )
        )
    return resolved


def _load_section_for_arm(
    session: "BinarySession", pointer: SectionPointerSpec
) -> Section:
    """Dispatch a single pointer through the right per-arm loader.

    Kept module-private + arm-dispatch-only so
    :func:`resolve_section_pointers` is a clean walk over the input
    list. The two loader return tuples differ on their third element
    (``MatchedFunction`` vs ``FunctionData``); both are discarded here
    because variant sampling needs only the parsed :class:`Section`
    (its ``variants`` list length is the sampling input). The 1d
    wiring step re-issues the same load to harvest the variant
    function-data; the session caches make this cheap.

    Raises:
        ValueError: On an unknown :class:`SectionKind` member.
    """
    if pointer.arm is SectionKind.MATCHED:
        section, _section_offset, _matched = (
            session._load_matched_section_and_variants(pointer.idx)
        )
        return section
    if pointer.arm is SectionKind.UNMATCHED:
        section, _section_offset, _fd = (
            session._load_unmatched_record_and_section(pointer.idx)
        )
        return section
    raise ValueError(f"unknown SectionKind: {pointer.arm!r}")
