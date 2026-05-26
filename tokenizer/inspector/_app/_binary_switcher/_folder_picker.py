"""Folder picker modal: filesystem tree with green-marked data folders.

Single concern: present a tree rooted at a start path; lazy-expand
sub-directories on demand; green-mark folders whose
:func:`is_loadable_for_any` predicate is ``True`` (i.e. the folder
contains data for AT LEAST ONE provider — the picker is
provider-agnostic, picked-binary-first ordering means provider choice
happens later in the binary switcher dialog).

Each expandable folder exposes an ``[open this folder]`` child as its
first entry: selecting it commits that folder back to the caller.
Subfolders are listed below; expanding a subfolder recursively shows
ITS own ``[open this folder]`` row + its subfolders. The user can
drill arbitrarily deep, then commit whichever level they wanted via
``[open this folder]``. Plain subfolder rows themselves never commit
on click — they only browse — because the green-marking is the hint
that helps the user spot promising drill-down targets, not a
commit-on-click action.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

from rich.text import Text

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Tree
from textual.widgets._tree import TreeNode

from ._scan import is_loadable_for_any, list_child_directories


__all__ = [
    "FolderPickerDialog",
]


_GREEN_STYLE = "bold green"


class _PickerRow:
    """Marker base for picker tree payloads."""


class _FolderRow(_PickerRow):
    """Tree-node payload: one filesystem directory.

    ``loadable`` drives the green coloring (any-provider data
    detected); ``_populated`` is the lazy-expand flag toggled on first
    :class:`Tree.NodeExpanded`.
    """

    __slots__ = ("path", "loadable", "_populated")

    def __init__(self, path: Path, loadable: bool) -> None:
        self.path = path
        self.loadable = loadable
        self._populated = False


class _OpenFolderRow(_PickerRow):
    """Tree-node payload: ``[open this folder]`` — commits ``path``."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path


class FolderPickerDialog(ModalScreen[Optional[Path]]):
    """Filesystem tree modal for picking a data folder.

    The tree expands sub-directories lazily on first
    :class:`textual.widgets.Tree.NodeExpanded` per node. Green-marking
    applies whenever :func:`is_loadable_for_any` reports the folder
    holds data for at least one provider.

    Dismiss paths:

    * ``escape`` -> :class:`None` (cancel).
    * Select / Enter on an ``[open this folder]`` row -> dismiss
      with the enclosing folder's :class:`Path`.
    """

    CSS: ClassVar[str] = """
    FolderPickerDialog {
        align: center middle;
    }
    FolderPickerDialog > #picker-body {
        background: $panel;
        border: tall $accent;
        padding: 1 2;
        width: 80;
        height: auto;
        max-height: 80%;
    }
    FolderPickerDialog #picker-tree {
        height: auto;
        max-height: 30;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "pick_highlighted", "Pick", show=True),
    ]

    BINDING_GROUP_TITLE: ClassVar[str] = "Folder picker"

    def __init__(
        self,
        *,
        start_path: Path,
    ) -> None:
        super().__init__()
        # Resolve symlinks + ensure a directory.
        try:
            start_resolved = start_path.expanduser().resolve()
        except OSError:
            start_resolved = start_path
        if not start_resolved.is_dir():
            start_resolved = Path.home()
        self._start_path = start_resolved

    # --- compose --------------------------------------------------

    def compose(self) -> ComposeResult:
        tree: Tree[_PickerRow] = Tree(
            "Pick folder",
            id="picker-tree",
        )
        loadable = is_loadable_for_any(self._start_path)
        root_data = _FolderRow(self._start_path, loadable)
        # The root shows the absolute path so the user always knows
        # where they're anchored.
        tree.root.set_label(_format_folder_label(self._start_path, loadable, full=True))
        tree.root.data = root_data
        tree.root.allow_expand = True
        tree.root.expand()
        # Eagerly populate the root so the [open this folder] row is
        # visible without the user expanding the root first.
        self._populate_folder(tree.root)
        with Vertical(id="picker-body"):
            yield tree

    # --- event dispatch ------------------------------------------

    def on_tree_node_expanded(
        self, event: Tree.NodeExpanded[_PickerRow]
    ) -> None:
        """Populate the expanded node's children on first expand.

        Walks one directory level via :func:`list_child_directories`,
        prepending an ``[open this folder]`` row so the user can commit
        the current level without drilling further. Idempotent:
        re-expand after collapse does not duplicate children.
        """
        event.stop()
        self._populate_folder(event.node)

    def _populate_folder(self, node: TreeNode[_PickerRow]) -> None:
        """Idempotent child-mount: ``[open this folder]`` + subfolders."""
        data = node.data
        if not isinstance(data, _FolderRow) or data._populated:
            return
        data._populated = True
        # [open this folder] is always the first child so its position
        # is predictable for keyboard navigation; styling is green when
        # the folder itself is loadable (matches the parent label's
        # color hint).
        open_style = _GREEN_STYLE if data.loadable else "cyan"
        node.add_leaf(
            Text("[open this folder]", style=open_style),
            data=_OpenFolderRow(data.path),
        )
        for sub in list_child_directories(data.path):
            sub_loadable = is_loadable_for_any(sub)
            node.add(
                _format_folder_label(sub, sub_loadable),
                data=_FolderRow(sub, sub_loadable),
                allow_expand=True,
            )

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[_PickerRow]
    ) -> None:
        """``[open this folder]`` commits; folder rows only browse.

        The user's mental model: green folders are worth drilling INTO
        and worth committing FROM (via their ``[open this folder]``
        child). Plain folders are still browseable but never commit.
        """
        event.stop()
        data = event.node.data
        if isinstance(data, _OpenFolderRow):
            self.dismiss(data.path)
            return
        # _FolderRow / None: expand/collapse handled by the Tree widget.

    # --- actions --------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_pick_highlighted(self) -> None:
        """Enter-key handler: pick the cursor's row if it is committable."""
        tree = self.query_one("#picker-tree", Tree)
        cursor = tree.cursor_node
        if cursor is None:
            return
        data = cursor.data
        if isinstance(data, _OpenFolderRow):
            self.dismiss(data.path)


def _format_folder_label(path: Path, loadable: bool, *, full: bool = False) -> Text:
    """Folder label: green when loadable, default-styled otherwise.

    ``full=True`` renders the absolute path (used for the picker root
    so the user always sees the anchor); otherwise the basename
    suffices since the parent chain in the tree carries the prefix.
    """
    display = str(path) if full else (path.name or str(path))
    style = _GREEN_STYLE if loadable else ""
    return Text(display, style=style)
