"""BatchDecode-side inline-jump openable resolvability gate.

Single concern: bridge the row walker's completed
:class:`._sections.RowSection` list to the shared per-line filter
(:mod:`tokenizer.inspector._render._jump_validity`). Derives the
addressable :attr:`BlockKind.BODY` block_idx set from the row's own
sections (no parallel indexing) and threads each section's items
through the shared helper.

The filter itself -- the per-:class:`AsmLine` openable rebuild + the
:class:`InlineJumpEntry` discriminator -- lives in the shared module
so the FtlBackend can apply the same gate on its own ``render_block``
output without re-implementing the discriminator (mirrors the
:meth:`AsmLeaf.can_expand` contract).

WHY this lives at the section layer here (concern boundary): the row
walker (:mod:`._row_walk`) populates :class:`._sections.RowSection`
items during its per-col loop. The set of BODY block_idxs is a
property of the COMPLETED section list -- known only after the walk
finishes. This module owns the post-walk pass that reads that set
and rebuilds sections with filtered openables.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List

from tokenizer.inspector._render._jump_validity import (
    filter_unresolvable_jump_openables_in_lines,
)
from tokenizer.inspector._render._protocol import BlockKind

from ._sections import RowSection


__all__ = ["filter_unresolvable_jump_openables"]


def filter_unresolvable_jump_openables(
    sections: List[RowSection],
) -> List[RowSection]:
    """Drop :class:`InlineJumpEntry` openables whose target has no BODY section.

    Returns a list of :class:`RowSection` whose :class:`AsmLine`
    openables tuples have any :class:`InlineJumpEntry` referencing a
    non-existent BODY block_idx removed. Sections whose items contain
    no such phantom openables are returned by reference (the common
    case for rows without jump tables, or rows whose targets all
    resolve).

    The validity set is derived directly from ``sections`` itself --
    every :attr:`BlockKind.BODY` section contributes its
    ``block_idx`` to the addressable set. No external lookup, no
    side cache.
    """
    body_block_idxs = _body_block_idx_set(sections)
    rebuilt: List[RowSection] = []
    any_changed = False
    for section in sections:
        new_section = _filter_section(section, body_block_idxs)
        if new_section is section:
            rebuilt.append(section)
        else:
            rebuilt.append(new_section)
            any_changed = True
    return rebuilt if any_changed else sections


def _body_block_idx_set(sections: Iterable[RowSection]) -> FrozenSet[int]:
    """Collect the addressable :attr:`BlockKind.BODY` block_idxs.

    A jump target is resolvable iff its ``target_block_idx`` appears
    here -- :meth:`RenderBackend.render_block` looks up by
    ``(BlockKind.BODY, block_idx)`` and only succeeds on a match.
    """
    return frozenset(
        s.block_idx for s in sections if s.kind is BlockKind.BODY
    )


def _filter_section(
    section: RowSection, body_block_idxs: FrozenSet[int],
) -> RowSection:
    """Return a section with each AsmLine's unresolvable jump openables dropped.

    Identity-preserving: if no AsmLine in the section needs
    filtering, the original section is returned by reference (frozen
    dataclasses share cheaply). Otherwise a fresh :class:`RowSection`
    is constructed carrying the same kind + block_idx, with the
    shared per-line helper applied to the items tuple.
    """
    new_items = filter_unresolvable_jump_openables_in_lines(
        section.items, body_block_idxs,
    )
    # Identity check: the helper preserves item identity where no
    # openables changed; if every item is the original object, the
    # tuple's contents match section.items element-for-element.
    if len(new_items) == len(section.items) and all(
        new_items[i] is section.items[i] for i in range(len(new_items))
    ):
        return section
    return RowSection(
        kind=section.kind, block_idx=section.block_idx, items=list(new_items),
    )
