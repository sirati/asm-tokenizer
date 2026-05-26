"""Folder picker modal: filesystem tree with green-marked data folders.

Single concern: present a tree rooted at a start path; lazy-expand
sub-directories on demand; green-mark folders whose
:func:`is_loadable_for` predicate is ``True`` for the provider the
caller asked about. Dismiss with the user-confirmed :class:`Path` (or
``None`` on cancel).

Single click on a folder = expand. Double-click / Enter on a green
folder = confirm + dismiss. Plain (non-green) folders never dismiss —
they only browse, since the caller wants a path that contains data.

Green-marking is per-provider: a memmap-loadable folder is green when
opened via :attr:`LoaderProvider.MEMMAP`, a csv-loadable folder is
green when opened via :attr:`LoaderProvider.CSV`. The picker does NOT
discover both at once — the caller has already chosen which provider
they are picking a path for.
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

from ._provider import LoaderProvider
from ._scan import is_loadable_for, list_child_directories


__all__ = [
    "FolderPickerDialog",
]


_GREEN_STYLE = "bold green"


class _FolderRow:
    """Tree-node payload: one filesystem directory."""

    __slots__ = ("path", "loadable", "_populated")

    def __init__(self, path: Path, loadable: bool) -> None:
        self.path = path
        self.loadable = loadable
        # Children-populated flag — set ``True`` after the first
        # :meth:`Tree.NodeExpanded` walks the directory's child dirs.
        self._populated = False


class FolderPickerDialog(ModalScreen[Optional[Path]]):
    """Filesystem tree modal for picking a data folder.

    The tree expands sub-directories lazily on first
    :class:`textual.widgets.Tree.NodeExpanded` per node. Green-marking
    is applied at child-mount time per :func:`is_loadable_for` against
    the caller-supplied provider.

    Dismiss paths:

    * ``escape`` -> :class:`None` (cancel).
    * Enter / select on a GREEN folder -> dismiss with that
      folder's :class:`Path`.
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
        provider: LoaderProvider,
        start_path: Path,
    ) -> None:
        super().__init__()
        self._provider = provider
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
        tree: Tree[_FolderRow] = Tree(
            f"Pick folder for {self._provider.value}",
            id="picker-tree",
        )
        loadable = is_loadable_for(self._start_path, self._provider)
        root_data = _FolderRow(self._start_path, loadable)
        root_label = _format_folder_label(self._start_path, loadable)
        # Replace the default root with our own labelled root.
        tree.root.set_label(root_label)
        tree.root.data = root_data
        tree.root.allow_expand = True
        tree.root.expand()
        with Vertical(id="picker-body"):
            yield tree

    # --- event dispatch ------------------------------------------

    def on_tree_node_expanded(
        self, event: Tree.NodeExpanded[_FolderRow]
    ) -> None:
        """Populate the expanded node's children on first expand.

        Walks one directory level via :func:`list_child_directories`
        and adds one child node per sub-directory; each child gets
        ``allow_expand=True`` so the user can drill deeper. Idempotent:
        re-expand after collapse does not duplicate children.
        """
        event.stop()
        node = event.node
        data = node.data
        if data is None or data._populated:
            return
        data._populated = True
        for sub in list_child_directories(data.path):
            sub_loadable = is_loadable_for(sub, self._provider)
            node.add(
                _format_folder_label(sub, sub_loadable),
                data=_FolderRow(sub, sub_loadable),
                allow_expand=True,
            )

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[_FolderRow]
    ) -> None:
        """Selecting a green folder dismisses with its path.

        Non-green folders are no-op on selection — the user can still
        expand them via Textual's built-in expand binding to browse
        deeper.
        """
        event.stop()
        data = event.node.data
        if data is None or not data.loadable:
            return
        self.dismiss(data.path)

    # --- actions --------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_pick_highlighted(self) -> None:
        """Enter-key handler: pick the cursor's folder if it is green."""
        tree = self.query_one("#picker-tree", Tree)
        cursor = tree.cursor_node
        if cursor is None:
            return
        data = cursor.data
        if data is None or not data.loadable:
            return
        self.dismiss(data.path)


def _format_folder_label(path: Path, loadable: bool) -> Text:
    """Folder label: green if loadable, default-styled otherwise.

    Shows the basename only — the full path is implicit from the
    parent chain in the tree. The root node carries the absolute
    path string so the user always knows where they are anchored;
    that is handled by the caller (via :meth:`Tree.root.set_label`),
    not this helper.
    """
    display = path.name or str(path)
    style = _GREEN_STYLE if loadable else ""
    return Text(display, style=style)
