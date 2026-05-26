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


class _ParentFolderRow(_PickerRow):
    """Tree-node payload: ``[parent folder]`` — re-roots picker at parent."""

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
        Binding("backspace", "go_up", "Parent folder", show=True),
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
        self._tree = tree
        self._seed_root(tree)
        with Vertical(id="picker-body"):
            yield tree

    def _seed_root(self, tree: Tree[_PickerRow]) -> None:
        """Wire the tree's root to ``self._start_path`` (re-root entry).

        Called from :meth:`compose` and again from :meth:`_rebase` when
        the user navigates up via ``[parent folder]``. Idempotent for the
        latter: ``tree.clear()`` zeroes the prior children before reseed.
        """
        loadable = is_loadable_for_any(self._start_path)
        root_data = _FolderRow(self._start_path, loadable)
        tree.root.set_label(
            _format_folder_label(self._start_path, loadable, full=True)
        )
        tree.root.data = root_data
        tree.root.allow_expand = True
        tree.root.expand()
        # Eagerly populate so the [parent folder] + [open this folder]
        # rows are visible without the user expanding the root first.
        self._populate_folder(tree.root)

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
        """Idempotent child-mount: parent (root only) + open + subfolders.

        ``[parent folder]`` appears only on the picker's root and only
        when a parent exists (i.e. we are not already at the filesystem
        root). ``[open this folder]`` appears only when the folder
        actually holds data for at least one provider — committing a
        non-loadable folder would just crash the loader downstream.
        """
        data = node.data
        if not isinstance(data, _FolderRow) or data._populated:
            return
        data._populated = True
        if node is self._tree.root:
            parent = data.path.parent
            if parent != data.path:
                node.add_leaf(
                    Text(f"[parent folder: {parent}]", style="cyan"),
                    data=_ParentFolderRow(parent),
                )
        if data.loadable:
            node.add_leaf(
                Text("[open this folder]", style=_GREEN_STYLE),
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
        """``[open this folder]`` commits; ``[parent folder]`` re-roots
        the picker one level up; plain folder rows only browse.

        The user's mental model: green folders are worth drilling INTO
        and worth committing FROM (via their ``[open this folder]``
        child). Plain folders are still browseable but never commit;
        navigating UP uses the dedicated ``[parent folder]`` row (or
        the ``backspace`` keybinding).
        """
        event.stop()
        data = event.node.data
        if isinstance(data, _OpenFolderRow):
            self.dismiss(data.path)
            return
        if isinstance(data, _ParentFolderRow):
            self._rebase(data.path)
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
        elif isinstance(data, _ParentFolderRow):
            self._rebase(data.path)

    def action_go_up(self) -> None:
        """Backspace: re-root picker at the current root's parent."""
        parent = self._start_path.parent
        if parent != self._start_path:
            self._rebase(parent)

    def _rebase(self, new_root: Path) -> None:
        """Reseed the tree at ``new_root`` (used by parent-folder nav)."""
        self._start_path = new_root
        tree = self._tree
        tree.clear()
        self._seed_root(tree)


def _format_folder_label(path: Path, loadable: bool, *, full: bool = False) -> Text:
    """Folder label: green when loadable, default-styled otherwise.

    ``full=True`` renders the absolute path (used for the picker root
    so the user always sees the anchor); otherwise the basename
    suffices since the parent chain in the tree carries the prefix.
    """
    display = str(path) if full else (path.name or str(path))
    style = _GREEN_STYLE if loadable else ""
    return Text(display, style=style)
