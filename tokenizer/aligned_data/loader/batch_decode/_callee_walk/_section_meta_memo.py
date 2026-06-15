"""Per-walk callee-section metadata parse memo.

Single concern: parse a given callee section's ``_sections.bin`` catalog
entry (the metadata-only :class:`Section` + its BIN byte offset) AT MOST
ONCE per batch-decode walk, keyed by ``(arm, idx)``.

Why this exists -- the re-parse it removes:

The level-synchronous callee walk resolves EVERY direct-call edge of
EVERY surviving frontier row at EVERY BFS level on the PRE-prune
metadata path (:func:`.._resolve.resolve_callee_metadata`). Each edge's
metadata resolution calls :py:meth:`BinarySession._matched_section_meta`
/ :py:meth:`_unmatched_section_meta` -> :py:meth:`_parse_section_at` ->
:func:`parse_section_bin` -- a full Python-loop parse of the section's
CallTarget list + VariantBlock list. The SAME callee section is reached
across sibling parents, across BFS levels, and across the section's
sampled variants, so on a dense call graph the identical
``(arm, idx)`` catalog entry is re-parsed many times over even before
the once-only decider prunes / dedups the edge.

This is NOT a persistent cross-call cache (which would violate the
re-parse-on-demand / no-parallel-index mandate). It is the sanctioned
"thread parsed state" pattern: a dict keyed by ``(arm, idx)``, created
when a walk begins, threaded through the resolver, and DROPPED when the
walk ends. It memoizes a within-one-walk parse of mmap'd bytes, never
state that outlives the walk.

API boundary: the memo OWNS the choice of meta method per arm. The
resolver calls :meth:`section_meta` with the arm + idx it already
resolved (via ``_idx_for_section_offset``); it never touches the
session's per-arm meta methods directly and never sees the memo's
internals. A walk that wants no memoization passes ``None`` and the
resolver falls through to the session methods unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import Section

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from tokenizer.aligned_data.loader.session import BinarySession


__all__ = ["CalleeSectionMetaMemo"]


class CalleeSectionMetaMemo:
    """Memoize ``(section, section_offset)`` catalog parses for one walk.

    Keyed by ``(arm, idx)`` -- the per-arm catalog index the resolver
    already holds. The key is bijective with the parsed section's BIN
    byte offset within an arm (``_idx_for_section_offset`` is the
    inverse), so memoizing on ``(arm, idx)`` is exactly "parse this
    section's catalog entry once per walk".

    The returned :class:`Section` is identical to the un-memoized
    session-meta call (same zero-copy parse of the same mmap bytes); the
    memo only skips the repeat ``parse_section_bin`` Python loop.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int], Tuple[Section, int]] = {}

    def section_meta(
        self,
        session: "BinarySession",
        arm: SectionKind,
        idx: int,
    ) -> Tuple[Section, int]:
        """Parsed ``(section, section_offset)`` for ``(arm, idx)``, once.

        First miss routes to the session's per-arm metadata-only parse
        (:py:meth:`BinarySession._matched_section_meta` /
        :py:meth:`_unmatched_section_meta`); subsequent hits for the same
        ``(arm, idx)`` within this walk return the cached parse.
        """
        key = (arm.value, idx)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if arm is SectionKind.MATCHED:
            meta = session._matched_section_meta(idx)
        else:
            meta = session._unmatched_section_meta(idx)
        self._cache[key] = meta
        return meta
