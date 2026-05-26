"""Pure name-substring matchers over the inspector tree's root rows.

Single concern: given a sequence of top-level :class:`TreeNode` rows
(the FunctionNode rows mounted under the inspector tree root) and a
lower-cased needle, yield the rows whose :attr:`FunctionNode.name`
matches. Two flavours: :func:`first_function_match` returns the
single first hit (used by :class:`SearchBar` for the live-preview
jump + Enter-commit), :func:`iter_function_matches` yields every hit
in declared order (used by the App's ``n`` / ``shift+n`` walker).

Match policy: case-insensitive substring against
:attr:`FunctionNode.name` -- the typed handle's display name, the
single source of truth for the function row's identity. The composed
row label (which prepends ``function`` etc.) is intentionally NOT
matched against, so a user typing ``calloc`` does not accidentally
match the literal ``function`` prefix every row carries.

Kept module-scope helpers (no widget instantiation required) so unit
tests + the ``n`` / ``shift+n`` walker on the App both reach them
without dragging the textual stack along.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator

from tokenizer.inspector._tree_model import FunctionNode


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._tree_model import Node


__all__ = [
    "first_function_match",
    "iter_function_matches",
    "next_match_index",
]


def first_function_match(
    children: "Iterable[TreeNode[Node]]", needle_lower: str
) -> "TreeNode[Node] | None":
    """Return the first top-level FunctionNode row whose name matches.

    Substring match against :attr:`FunctionNode.name` (typed handle's
    display name -- the single source of truth for the function row's
    identity). Case-insensitive: ``needle_lower`` is the already-
    lowered query, and each candidate name is lowered before the
    ``in`` check.

    Non-FunctionNode rows (the root has none today, but a defensive
    isinstance keeps the helper robust if a future addition mounts
    non-function payloads directly under the root) are skipped.
    """
    for child in iter_function_matches(children, needle_lower):
        return child
    return None


def iter_function_matches(
    children: "Iterable[TreeNode[Node]]", needle_lower: str
) -> "Iterator[TreeNode[Node]]":
    """Yield every top-level FunctionNode row whose name matches.

    Walks the same FunctionNode-filtered iterator that
    :func:`first_function_match` uses but yields ALL matches in
    declared order so the App's ``n`` / ``shift+n`` jump bindings
    can step through them.
    """
    for child in children:
        model = child.data
        if not isinstance(model, FunctionNode):
            continue
        if needle_lower in model.name.lower():
            yield child


def next_match_index(
    matches: "list[TreeNode[Node]]",
    cursor: "TreeNode[Node] | None",
    cursor_line: int,
    *,
    forward: bool,
) -> int:
    """Pick the match-list index to land on, given the cursor row.

    Used by the App's ``n`` / ``shift+n`` walker. ``matches`` is the
    in-tree-row-order match list (caller materialises it from
    :func:`iter_function_matches`). When the cursor IS one of the
    matches, the next/previous slot is returned with wrap-around.
    When the cursor sits on a non-match row (or the root), the
    first match strictly after (or before) the cursor's
    ``cursor_line`` is returned, wrapping when the cursor is past
    the last (or before the first) match. Caller is responsible for
    pre-checking that ``matches`` is non-empty.

    Pure: takes the cursor's pre-read ``_line`` rather than
    re-reading from the (Textual-owned) node so the helper can be
    unit-tested without instantiating widgets.
    """
    if cursor is None:
        return 0 if forward else len(matches) - 1
    try:
        idx = matches.index(cursor)
    except ValueError:
        # Non-match cursor: scan by line index for the nearest match
        # in the requested direction, wrap when past the end.
        if forward:
            for i, candidate in enumerate(matches):
                if candidate._line > cursor_line:
                    return i
            return 0
        for i in range(len(matches) - 1, -1, -1):
            if matches[i]._line < cursor_line:
                return i
        return len(matches) - 1
    return (idx + (1 if forward else -1)) % len(matches)
