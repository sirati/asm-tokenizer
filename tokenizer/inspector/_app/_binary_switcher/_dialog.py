"""Binary switcher modal: provider tree + per-binary entries.

Single concern: present a :class:`textual.widgets.Tree` rooted at two
:class:`LoaderProvider`-discriminated children (memmap, csv), each
showing the current path + the binaries discovered there + a
``change path...`` entry, and dismiss with a :class:`SwitchTarget`
(or ``None`` on cancel).

Layout (one provider):

    ▼ memmap (stage 3)
       └── <current-path>
           ├── ▶ [open this folder]
           ├── ▶ <binary-1>
           ├── ▶ <binary-2>
           └── ▶ change path...

Green-marking: a folder containing loadable data is rendered in green
text. The ``[open this folder]`` entry is green when the folder is
loadable (auto-detect succeeds). Per-binary entries are always green
(they only show up when the folder is loadable). ``change path...`` is
default-styled — it opens the folder picker, not a switch.

The dialog does NOT execute the switch — it only yields a typed
:class:`SwitchTarget` that the App side consumes. Confirmation
("Proceed? Work will not be saved") is the responsibility of the
switch flow itself (this dialog dismisses immediately on accept).
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

from ._provider import LoaderProvider, SwitchTarget
from ._scan import FolderScanResult, scan_folder


__all__ = [
    "BinarySwitcherDialog",
]


# CSS-friendly green style applied via Rich Text.
_GREEN_STYLE = "bold green"


class _TreeNodePayload:
    """Marker class for the tree's data payload.

    Subclassed by the concrete payload variants below. The
    :class:`Tree` widget is generic over the payload type; using one
    base class keeps the type parameter narrow.
    """


class _ProviderRoot(_TreeNodePayload):
    """Tree-root payload for one provider (memmap / csv)."""

    __slots__ = ("provider",)

    def __init__(self, provider: LoaderProvider) -> None:
        self.provider = provider


class _PathRow(_TreeNodePayload):
    """Tree row: the current path under a provider."""

    __slots__ = ("provider", "path", "scan")

    def __init__(
        self,
        provider: LoaderProvider,
        path: Path,
        scan: FolderScanResult,
    ) -> None:
        self.provider = provider
        self.path = path
        self.scan = scan


class _OpenFolderRow(_TreeNodePayload):
    """Tree row: ``[open this folder]`` — picks all binaries in the path."""

    __slots__ = ("provider", "path", "scan")

    def __init__(
        self,
        provider: LoaderProvider,
        path: Path,
        scan: FolderScanResult,
    ) -> None:
        self.provider = provider
        self.path = path
        self.scan = scan


class _BinaryRow(_TreeNodePayload):
    """Tree row: one per-binary entry under a loadable path."""

    __slots__ = ("provider", "path", "binary")

    def __init__(
        self,
        provider: LoaderProvider,
        path: Path,
        binary: str,
    ) -> None:
        self.provider = provider
        self.path = path
        self.binary = binary


class _ChangePathRow(_TreeNodePayload):
    """Tree row: ``change path...`` — opens the folder picker."""

    __slots__ = ("provider",)

    def __init__(self, provider: LoaderProvider) -> None:
        self.provider = provider


# ---------------------------------------------------------------------------
# Modal screen
# ---------------------------------------------------------------------------


class BinarySwitcherDialog(ModalScreen[Optional[SwitchTarget]]):
    """Switch-binary modal — provider tree, dismisses with a target."""

    CSS: ClassVar[str] = """
    BinarySwitcherDialog {
        align: center middle;
    }
    BinarySwitcherDialog > #switcher-body {
        background: $panel;
        border: tall $accent;
        padding: 1 2;
        width: 80;
        height: auto;
        max-height: 80%;
    }
    BinarySwitcherDialog #switcher-tree {
        height: auto;
        max-height: 30;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    BINDING_GROUP_TITLE: ClassVar[str] = "Switch binary"

    def __init__(
        self,
        *,
        current_path: Optional[Path] = None,
        current_provider: Optional[LoaderProvider] = None,
    ) -> None:
        super().__init__()
        # Both provider subtrees seed against the App's single
        # ``current_path``; the user can re-point either independently
        # via ``change path...`` within the dialog session. The home
        # directory is a sensible fallback when no current path is
        # known (the mock-factory test case).
        anchor = current_path or Path.home()
        self._memmap_path: Path = anchor
        self._csv_path: Path = anchor
        self._current_provider = current_provider

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        tree: Tree[_TreeNodePayload] = Tree(
            "Switch binary", id="switcher-tree"
        )
        tree.root.expand()
        self._build_provider_branch(
            tree.root,
            LoaderProvider.MEMMAP,
            self._memmap_path,
            stage_label="stage 3",
        )
        self._build_provider_branch(
            tree.root,
            LoaderProvider.CSV,
            self._csv_path,
            stage_label="stage 1",
        )
        with Vertical(id="switcher-body"):
            yield tree

    def _build_provider_branch(
        self,
        root: TreeNode[_TreeNodePayload],
        provider: LoaderProvider,
        path: Optional[Path],
        *,
        stage_label: str,
    ) -> None:
        """Mount one provider's subtree under ``root``.

        Layout: ``<provider> (<stage_label>) -> <current-path>``; under
        the path, ``[open this folder]`` (when loadable) + per-binary
        rows + ``change path...``.
        """
        suffix = self._provider_label_suffix(provider)
        provider_label = Text(
            f"{provider.value} ({stage_label}){suffix}", style="bold"
        )
        provider_node = root.add(
            provider_label,
            data=_ProviderRoot(provider),
            expand=True,
        )
        if path is None:
            provider_node.add_leaf(
                Text("(no path set — pick one below)", style="dim"),
                data=None,
            )
            provider_node.add_leaf(
                Text("change path...", style="cyan"),
                data=_ChangePathRow(provider),
            )
            return

        scan = scan_folder(path, provider)
        path_label = self._format_path_label(path, scan.loadable)
        path_node = provider_node.add(
            path_label,
            data=_PathRow(provider, path, scan),
            expand=True,
        )
        if scan.loadable:
            path_node.add_leaf(
                Text("[open this folder]", style=_GREEN_STYLE),
                data=_OpenFolderRow(provider, path, scan),
            )
            for binary in scan.binaries:
                path_node.add_leaf(
                    Text(binary, style=_GREEN_STYLE),
                    data=_BinaryRow(provider, path, binary),
                )
        else:
            path_node.add_leaf(
                Text("(no loadable data in this folder)", style="dim"),
                data=None,
            )
        path_node.add_leaf(
            Text("change path...", style="cyan"),
            data=_ChangePathRow(provider),
        )

    def _provider_label_suffix(self, provider: LoaderProvider) -> str:
        """``" — current"`` marker for the provider the App is using now."""
        if self._current_provider is provider:
            return "  [current]"
        return ""

    @staticmethod
    def _format_path_label(path: Path, loadable: bool) -> Text:
        """Green-styled path when loadable; default style otherwise."""
        style = _GREEN_STYLE if loadable else ""
        return Text(str(path), style=style)

    # --- event dispatch ----------------------------------------------

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[_TreeNodePayload]
    ) -> None:
        """Route the selected tree node through the payload-dispatch."""
        event.stop()
        data = event.node.data
        if isinstance(data, _OpenFolderRow):
            self.dismiss(
                SwitchTarget(
                    provider=data.provider,
                    path=data.path,
                    binary=None,
                )
            )
            return
        if isinstance(data, _BinaryRow):
            self.dismiss(
                SwitchTarget(
                    provider=data.provider,
                    path=data.path,
                    binary=data.binary,
                )
            )
            return
        if isinstance(data, _ChangePathRow):
            self._open_folder_picker(data.provider)
            return
        # Provider-root / path / informational rows: no action.

    # --- folder picker integration -----------------------------------

    def _open_folder_picker(self, provider: LoaderProvider) -> None:
        """Push the folder picker; on confirm, rebuild this provider's
        subtree against the new path."""
        from ._folder_picker import FolderPickerDialog

        anchor = (
            self._memmap_path
            if provider is LoaderProvider.MEMMAP
            else self._csv_path
        )
        self.app.push_screen(
            FolderPickerDialog(provider=provider, start_path=anchor),
            lambda result: self._on_folder_picked(provider, result),
        )

    def _on_folder_picked(
        self, provider: LoaderProvider, new_path: Optional[Path]
    ) -> None:
        """Re-seed the dialog's tree against the newly-picked path.

        The picker returns ``None`` on cancel (no-op) or a directory
        path. The dialog rebuilds its tree in place so the user can
        keep browsing without re-opening the modal.
        """
        if new_path is None:
            return
        if provider is LoaderProvider.MEMMAP:
            self._memmap_path = new_path
        else:
            self._csv_path = new_path
        # Rebuild the tree wholesale.
        tree = self.query_one("#switcher-tree", Tree)
        tree.clear()
        self._build_provider_branch(
            tree.root,
            LoaderProvider.MEMMAP,
            self._memmap_path,
            stage_label="stage 3",
        )
        self._build_provider_branch(
            tree.root,
            LoaderProvider.CSV,
            self._csv_path,
            stage_label="stage 1",
        )

    # --- actions ---------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)
