"""Provider-agnostic inline-jump openable resolvability gate.

Single concern: drop :class:`InlineJumpEntry` openables whose
``target_block_idx`` is not an addressable :attr:`BlockKind.BODY`
section in the surrounding variant. Keeps the placeholder
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
the renderer's stream verbatim regardless of which backend produced
the per-block walk -- the per-instruction emitter cannot tell phantom
from real at that layer.

WHERE this lives (concern boundary): the per-backend caller derives
the validity set from its own data shape and threads the resulting
:class:`frozenset[int]` into this module. The BatchDecodeBackend reads
it off the row walker's completed :class:`RowSection` list; the
FtlBackend reads it off its parsed-variant ``state.blocks`` tuple.
Both then funnel through :func:`filter_unresolvable_jump_openables_in_lines`
(this module) -- one source of truth for the per-:class:`AsmLine`
filter logic. NO parallel indexing (per CLAUDE.md's "no parallel
indexing over self-describing data" rule): the validity set is a
typed argument, never a side cache owned by this module.

WHAT the caller sees (API surface): one function over a flat line
sequence. Identity-preserving: lines with no unresolvable openables
are returned by reference (frozen dataclass sharing is cheap).
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Tuple

from ._protocol import AsmLine, InlineJumpEntry, LineItem, Openable


__all__ = [
    "filter_unresolvable_jump_openables_in_lines",
    "is_unresolvable_jump_openable",
]


def filter_unresolvable_jump_openables_in_lines(
    lines: Iterable[LineItem],
    body_block_idxs: FrozenSet[int],
) -> Tuple[LineItem, ...]:
    """Drop :class:`InlineJumpEntry` openables with no addressable target.

    Returns a tuple of :class:`LineItem` whose :class:`AsmLine`
    openables tuples have any :class:`InlineJumpEntry` referencing a
    block_idx not in ``body_block_idxs`` removed. Lines with no
    phantom openables are returned by reference (the common case for
    rows without jump tables, or rows whose targets all resolve).

    The validity set is supplied by the caller -- each backend
    derives it from its own variant-level state (BatchDecode reads
    the per-row sections, FtlBackend reads the per-variant
    ``state.blocks`` tuple). This module owns only the per-line
    filter; the set-derivation concern stays in the backend.

    Non-:class:`AsmLine` :class:`LineItem` shapes pass through
    untouched. The :data:`LineItem` union currently narrows to
    :class:`AsmLine` only (see :mod:`._protocol`); the
    ``isinstance`` guard is defensive against future re-broadening.
    """
    filtered: list[LineItem] = []
    for item in lines:
        if isinstance(item, AsmLine):
            filtered.append(_filter_asm_line(item, body_block_idxs))
        else:
            filtered.append(item)
    return tuple(filtered)


def _filter_asm_line(
    line: AsmLine, body_block_idxs: FrozenSet[int],
) -> AsmLine:
    """Drop :class:`InlineJumpEntry` openables with no addressable target.

    Identity-preserving: if every openable's target resolves (or the
    line carries no jump openables at all), the original
    :class:`AsmLine` is returned by reference. Otherwise a fresh
    line is built with the same ``text`` (the ``jump block: N``
    placeholder stays so the row's diagnostic content survives) and
    the filtered ``openables`` tuple.
    """
    filtered: Tuple[Openable, ...] = tuple(
        o for o in line.openables
        if not is_unresolvable_jump_openable(o, body_block_idxs)
    )
    if len(filtered) == len(line.openables):
        return line
    return AsmLine(text=line.text, openables=filtered)


def is_unresolvable_jump_openable(
    openable: Openable, body_block_idxs: FrozenSet[int],
) -> bool:
    """True iff ``openable`` is an :class:`InlineJumpEntry` whose
    ``target_block_idx`` is not in ``body_block_idxs``.

    Discriminator: dataclass type (``isinstance``), matching the
    Openable union's typed pattern (no string-typed kind).
    """
    if not isinstance(openable, InlineJumpEntry):
        return False
    return openable.target_block_idx not in body_block_idxs
