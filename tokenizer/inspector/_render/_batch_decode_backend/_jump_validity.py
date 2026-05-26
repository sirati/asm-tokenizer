"""Inline-jump openable resolvability gate over completed row sections.

Single concern: drop :class:`InlineJumpEntry` openables whose
``target_block_idx`` is not addressable as a :attr:`BlockKind.BODY`
section in the row's completed section list. Keeps the placeholder
``jump block: N`` text on the :class:`AsmLine` (so the row's
diagnostic content survives), but removes the openable so the
:meth:`AsmLeaf.can_expand` gate flips off and the tree model never
calls :meth:`RenderBackend.render_block` for a non-existent target.

WHY this exists (writer-side root cause): the per-function BLOCK
identity namespace is a *superset* of body-block ids. The encoder's
:func:`tokenizer.fill_constant_candidates._emit_jump_table_footer`
allocates a fresh ``Category.BLOCK`` identity for every switch-table
target address (via :meth:`TokenResolver.get_identity`), including
targets that have NO corresponding body block in this variant (data-
only addresses the disassembler resolved but never made a block out
of, or cross-variant divergence where one variant's switch table
references blocks another variant lacks). Those phantom ids land in
the row writer's BLOCK_V2 stream verbatim — the row walker's
``_handle_block`` cannot tell phantom from real at emit time
(W3-16's per-block + per-instruction flags discriminate
header-vs-jump, not "exists as a body section").

WHERE this lives (concern boundary): the row walker
(:mod:`._row_walk`) populates :class:`._sections.RowSection` items
during its per-col loop. The set of BODY block_idxs is a property of
the COMPLETED section list — known only after the walk finishes.
This module owns the post-walk pass that reads that set and rebuilds
items with filtered openables. NO parallel indexing (per CLAUDE.md's
"no parallel indexing over self-describing data" rule): the BODY set
is read directly off ``sections``, never cached in side state.

WHAT the caller sees (API surface): one function over the walker's
output. The caller (:func:`._row_walk._driver.render_row_blocks`)
applies it once at end-of-walk and never touches per-instruction
state. The :class:`RowSection` / :class:`AsmLine` dataclasses are
frozen, so the helper rebuilds instances where filtering changes the
openables tuple; sections + lines with no unresolvable openables are
preserved by reference (cheap no-op when the row has no phantom
targets).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    InlineJumpEntry,
    LineItem,
    Openable,
)

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

    The validity set is derived directly from ``sections`` itself —
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


def _body_block_idx_set(sections: Iterable[RowSection]) -> frozenset[int]:
    """Collect the addressable :attr:`BlockKind.BODY` block_idxs.

    A jump target is resolvable iff its ``target_block_idx`` appears
    here -- :meth:`RenderBackend.render_block` looks up by
    ``(BlockKind.BODY, block_idx)`` and only succeeds on a match.
    """
    return frozenset(
        s.block_idx for s in sections if s.kind is BlockKind.BODY
    )


def _filter_section(
    section: RowSection, body_block_idxs: frozenset[int],
) -> RowSection:
    """Return a section with each AsmLine's unresolvable jump openables dropped.

    Identity-preserving: if no AsmLine in the section needs
    filtering, the original section is returned by reference (frozen
    dataclasses share cheaply). Otherwise a fresh :class:`RowSection`
    is constructed carrying the same kind + block_idx, with each
    AsmLine rebuilt only when its own openables tuple changed.
    """
    new_items: List[LineItem] = []
    any_changed = False
    for item in section.items:
        if isinstance(item, AsmLine):
            new_item = _filter_asm_line(item, body_block_idxs)
            if new_item is item:
                new_items.append(item)
            else:
                new_items.append(new_item)
                any_changed = True
        else:
            new_items.append(item)
    if not any_changed:
        return section
    return RowSection(
        kind=section.kind, block_idx=section.block_idx, items=new_items,
    )


def _filter_asm_line(
    line: AsmLine, body_block_idxs: frozenset[int],
) -> AsmLine:
    """Drop :class:`InlineJumpEntry` openables with no addressable BODY target.

    Identity-preserving: if every openable's target resolves (or the
    line carries no jump openables at all), the original
    :class:`AsmLine` is returned by reference. Otherwise a fresh
    line is built with the same ``text`` (the ``jump block: N``
    placeholder stays so the row's diagnostic content survives) and
    the filtered ``openables`` tuple.
    """
    filtered: Tuple[Openable, ...] = tuple(
        o for o in line.openables
        if not _is_unresolvable_jump(o, body_block_idxs)
    )
    if len(filtered) == len(line.openables):
        return line
    return AsmLine(text=line.text, openables=filtered)


def _is_unresolvable_jump(
    openable: Openable, body_block_idxs: frozenset[int],
) -> bool:
    """True iff ``openable`` is an :class:`InlineJumpEntry` whose
    ``target_block_idx`` is not an addressable BODY section.

    Discriminator: dataclass type (``isinstance``), matching the
    Openable union's typed pattern (no string-typed kind).
    """
    if not isinstance(openable, InlineJumpEntry):
        return False
    return openable.target_block_idx not in body_block_idxs
