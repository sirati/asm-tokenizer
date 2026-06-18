"""Geometry-only section resolve for the vector_batch path.

Single concern: turn a request's ``list[SectionPointerSpec]`` into the
ARM + BIN section offset + RNG-sampled variant indices that the
vector_batch geometry / scatter pipeline consumes -- WITHOUT the full
per-section :func:`...matched_sections_bin.parse_section_bin` object
build that :func:`...batch_decode._resolve_pointers.resolve_section_pointers`
pays for the callee-walking ``batch_decode`` path.

Why a separate resolve (vs ``resolve_section_pointers`` with
``load_bodies=False``): the vector_batch dispatch reads ONLY
``rs.arm`` (arm routing), ``rs.section_offset`` (the columnar catalog
key), and ``rs.sampled_variant_indices`` (the post-sampling slots) --
see ``_dispatch._rows_to_catalog_nodes``. It never touches a
:class:`Section`'s call_target / variant-block tables. Variant sampling
itself keys solely on the per-section variant COUNT, and that count is
the section header's third u16 field -- recoverable for the whole batch
in one vectorized header read
(:func:`...matched_sections_columnar.read_n_variants_columnar`) instead
of ~B full ``parse_section_bin`` walks. This mirrors the proven columnar
cutover the matched/unmatched arm loaders already made (e12735a /
c482529): the columnar decoder is the single vectorized source of truth
for the ``sections.bin`` wire format, so this resolve reads from it
rather than hand-rolling a parser.

Byte-identity contract: the RNG draw is per-pointer, in input order, via
the SAME :func:`..._session_helpers._select_variant_indices` the shared
resolver calls with the SAME ``n_variants`` value -- so for a given
``rng`` the sampled indices are IDENTICAL to ``resolve_section_pointers``
(the vector_batch <-> batch_decode equivalence the entry harness pins).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

from ..metadata_loader import SectionKind
from .._session_helpers import _select_variant_indices
from ...matched_sections_columnar import read_n_variants_columnar
from ..batch_decode._types import SectionPointerSpec

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..session import BinarySession


__all__ = ["ResolvedSectionGeometry", "resolve_section_geometry"]


@dataclass(frozen=True)
class ResolvedSectionGeometry:
    """vector_batch resolve output: arm + BIN offset + sampled variants.

    The geometry-only counterpart to
    :class:`...batch_decode._resolve_pointers.ResolvedSection`. It carries
    NO parsed :class:`Section` (the vector_batch dispatch keys on the BIN
    byte offset, not the catalog object) and NO per-variant
    :class:`FunctionData` (the RLG3 geometry / scatter gathers bodies
    itself). The fields are exactly what ``_dispatch`` +
    ``compute_batch_idx_mapping`` read.

    Attributes:
        arm: The originating :attr:`SectionPointerSpec.arm` (matched /
            unmatched routing).
        section_offset: The section's BIN byte offset -- the universal
            key the dispatch maps to its columnar catalog position (the
            same value :class:`ResolvedSection.section`'s
            ``section_offset`` carries).
        sampled_variant_indices: The RNG-selected native variant indices
            in encounter order, identical to the shared resolver's draw
            for the same ``rng`` (see module docstring).
    """

    arm: SectionKind
    section_offset: int
    sampled_variant_indices: List[int]


def resolve_section_geometry(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    rng: np.random.Generator,
) -> List[ResolvedSectionGeometry]:
    """Resolve each pointer to ``(arm, section_offset, sampled_variants)``.

    Two passes over ``section_pointers`` (order preserved throughout):

    1. Parse-free section-offset gather -- arm-dispatch each pointer to
       the session's ``idx -> section_offset`` index lookup
       (:py:meth:`BinarySession._matched_section_offset` /
       :py:meth:`_unmatched_section_offset`); no ``_sections.bin`` parse.
    2. ONE vectorized header read
       (:func:`...matched_sections_columnar.read_n_variants_columnar`)
       over the gathered offsets -> the per-pointer variant counts, then
       per-pointer RNG variant sampling in input order.

    The RNG draw order + ``n_variants`` values match
    :func:`...batch_decode._resolve_pointers.resolve_section_pointers`
    exactly, so the sampled indices are byte-identical for a shared
    ``rng`` (the vector_batch / batch_decode sampling-parity contract).

    Args:
        session: The :class:`BinarySession` whose per-arm offset maps +
            section catalog this resolves against.
        section_pointers: One :class:`SectionPointerSpec` per requested
            section; output order mirrors it.
        num_variants_per_section: Upper bound on variants sampled per
            section (clamped to the section's count by
            :func:`_select_variant_indices`).
        rng: Sampling source; threaded verbatim to
            :func:`_select_variant_indices` so the draw stays in lockstep
            with the body-loading resolver.

    Returns:
        ``list[ResolvedSectionGeometry]`` parallel to ``section_pointers``.

    Raises:
        ValueError: On a pointer whose ``arm`` is not a known
            :class:`SectionKind` member.
    """
    offsets = np.array(
        [_section_offset(session, pointer) for pointer in section_pointers],
        dtype=np.int64,
    )
    n_variants_per_pointer = read_n_variants_columnar(
        session._sections_bin_u8(), offsets
    )
    resolved: List[ResolvedSectionGeometry] = []
    for pointer, section_offset, n_variants in zip(
        section_pointers, offsets.tolist(), n_variants_per_pointer.tolist()
    ):
        sampled = _select_variant_indices(
            n_variants=int(n_variants),
            max_variants=num_variants_per_section,
            rng=rng,
        )
        resolved.append(
            ResolvedSectionGeometry(
                arm=pointer.arm,
                section_offset=int(section_offset),
                sampled_variant_indices=[int(v) for v in sampled],
            )
        )
    return resolved


def _section_offset(
    session: "BinarySession", pointer: SectionPointerSpec
) -> int:
    """Parse-free BIN section offset for one pointer (arm-dispatch only).

    Mirrors :func:`...batch_decode._resolve_pointers._parse_section_catalog`'s
    arm dispatch but routes to the session's index-only offset lookup
    (:py:meth:`_matched_section_offset` / :py:meth:`_unmatched_section_offset`)
    -- the parse-free half of the meta helpers -- so no ``_sections.bin``
    section is materialised here.

    Raises:
        ValueError: On an unknown :class:`SectionKind` member.
    """
    if pointer.arm is SectionKind.MATCHED:
        return session._matched_section_offset(pointer.idx)
    if pointer.arm is SectionKind.UNMATCHED:
        return session._unmatched_section_offset(pointer.idx)
    raise ValueError(f"unknown SectionKind: {pointer.arm!r}")
