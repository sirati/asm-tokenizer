"""Binary switcher modal: binary-first tree with provider children.

Single concern: present a :class:`textual.widgets.Tree` whose top-level
rows are (1) a header label echoing the current path, (2) one node per
binary discovered under that path with two children (memmap / csv —
each providing the corresponding loader stage), and (3) a
``change path...`` row that opens the folder picker. Dismiss with a
:class:`SwitchTarget` (or ``None`` on cancel).

Layout::

    ▼ Switch binary
       ├── /tmp/<current path>          (label only)
       ├── ▼ libz.so.1.2.11
       │   ├── memmap (stage 3)
       │   └── csv (stage 1)
       ├── ▼ minigzip
       │   ├── memmap (stage 3)
       │   └── csv (stage 1)
       └── change path...

The header is non-actionable. Each binary node lists exactly the
providers whose auto-detect finds that binary under the current path
(so a memmap-only directory hides the ``csv`` child rather than
producing a click that crashes at the resolver). ``change path...`` is
a top-level sibling of the binary nodes — it never lives nested under
a provider/binary — so the user can re-anchor before picking a binary.

The dialog does NOT execute the switch — it yields a typed
:class:`SwitchTarget` for the App side. There is no ``[open this
folder]`` row at this level: in a multi-binary directory the prior row
shape sent ``binary=None`` to the resolver which raised
:class:`SystemExit` ("--binary not given and <dir> contains multiple
binaries"). Binary-first ordering makes the picker unambiguous and
removes the crash by construction; per-folder commits now live inside
the folder picker only.
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


# Each provider's loader-stage label is a UI string keyed by the
# typed discriminator — no string-typed if/elif at the dispatch site.
_PROVIDER_STAGE_LABELS: dict[LoaderProvider, str] = {
    LoaderProvider.MEMMAP: "stage 3",
    LoaderProvider.CSV: "stage 1",
}


class _TreeNodePayload:
    """Marker class for the tree's data payload.

    Subclassed by the concrete payload variants below. The
    :class:`Tree` widget is generic over the payload type; using one
    base class keeps the type parameter narrow.
    """


class _PathHeaderRow(_TreeNodePayload):
    """Tree row: the current path shown as a header (no-op when clicked)."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path


class _BinaryRoot(_TreeNodePayload):
    """Tree row: one binary discovered under the current path.

    Children are :class:`_ProviderRow` instances (one per provider that
    has data for this binary). Clicking the binary node itself is a
    no-op; the user must drill into a provider child to commit.
    """

    __slots__ = ("path", "binary")

    def __init__(self, path: Path, binary: str) -> None:
        self.path = path
        self.binary = binary


class _ProviderRow(_TreeNodePayload):
    """Tree row: one (binary, provider) leaf — clicking commits the switch."""

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

    __slots__ = ()


# ---------------------------------------------------------------------------
# Modal screen
# ---------------------------------------------------------------------------


class BinarySwitcherDialog(ModalScreen[Optional[SwitchTarget]]):
    """Switch-binary modal — binary-first tree, dismisses with a target."""

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
        # Binary-first ordering means both providers share a single
        # anchor path: every binary is discovered under THAT path and
        # offers whichever providers have data for it there. The home
        # directory is a sensible fallback when the App has no current
        # path (mock-factory test case).
        self._anchor_path: Path = current_path or Path.home()
        self._current_provider = current_provider

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        tree: Tree[_TreeNodePayload] = Tree(
            "Switch binary", id="switcher-tree"
        )
        tree.root.expand()
        self._populate_tree(tree)
        with Vertical(id="switcher-body"):
            yield tree

    def _populate_tree(self, tree: Tree[_TreeNodePayload]) -> None:
        """Mount the binary-first contents under ``tree.root``.

        Layout: a non-actionable path header, one collapsible node per
        discovered binary (provider children attached), and the
        ``change path...`` sibling. Per-provider scans run once each;
        the cross-provider union drives binary enumeration so a binary
        present in only one provider still appears (with just that
        provider's child).
        """
        path = self._anchor_path
        # Header: shows the user where the dialog is anchored. Not
        # selectable because re-anchoring is via ``change path...``.
        tree.root.add_leaf(
            Text(str(path), style="bold"),
            data=_PathHeaderRow(path),
        )

        scans = {
            provider: scan_folder(path, provider)
            for provider in LoaderProvider
        }
        binaries = sorted(
            {b for scan in scans.values() for b in scan.binaries}
        )
        if binaries:
            for binary in binaries:
                self._mount_binary_node(tree.root, path, binary, scans)
        else:
            tree.root.add_leaf(
                Text(
                    "(no binaries detected — use 'change path...')",
                    style="dim",
                ),
                data=None,
            )

        tree.root.add_leaf(
            Text("change path...", style="cyan"),
            data=_ChangePathRow(),
        )

    def _mount_binary_node(
        self,
        root: TreeNode[_TreeNodePayload],
        path: Path,
        binary: str,
        scans: dict[LoaderProvider, FolderScanResult],
    ) -> None:
        """Add one binary subtree under ``root``.

        Each provider that has ``binary`` under ``path`` gets a child
        leaf; the leaf label suffixes ``[current]`` when both the path
        and provider match the App's currently-active selection so the
        user can see "you are here" at a glance.
        """
        binary_node = root.add(
            Text(binary, style=_GREEN_STYLE),
            data=_BinaryRoot(path, binary),
            expand=True,
        )
        for provider in LoaderProvider:
            if binary not in scans[provider].binaries:
                continue
            stage = _PROVIDER_STAGE_LABELS[provider]
            suffix = self._current_marker(path, provider)
            label = Text(
                f"{provider.value} ({stage}){suffix}",
                style=_GREEN_STYLE,
            )
            binary_node.add_leaf(
                label,
                data=_ProviderRow(provider, path, binary),
            )

    def _current_marker(
        self, path: Path, provider: LoaderProvider
    ) -> str:
        """``"  [current]"`` when (path, provider) match the App's state."""
        if (
            self._current_provider is provider
            and self._anchor_path == path
        ):
            return "  [current]"
        return ""

    # --- event dispatch ----------------------------------------------

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[_TreeNodePayload]
    ) -> None:
        """Route the selected tree node through the payload-dispatch."""
        event.stop()
        data = event.node.data
        if isinstance(data, _ProviderRow):
            self.dismiss(
                SwitchTarget(
                    provider=data.provider,
                    path=data.path,
                    binary=data.binary,
                )
            )
            return
        if isinstance(data, _ChangePathRow):
            self._open_folder_picker()
            return
        # _PathHeaderRow / _BinaryRoot / None: collapse/expand handled
        # by the Tree widget itself; no commit on selection.

    # --- folder picker integration -----------------------------------

    def _open_folder_picker(self) -> None:
        """Push the folder picker; on confirm, re-anchor + rebuild."""
        from ._folder_picker import FolderPickerDialog

        self.app.push_screen(
            FolderPickerDialog(start_path=self._anchor_path),
            self._on_folder_picked,
        )

    def _on_folder_picked(self, new_path: Optional[Path]) -> None:
        """Re-anchor the dialog's tree against the newly-picked path.

        Both provider subtrees re-scan against the new path. The picker
        returns ``None`` on cancel (no-op).
        """
        if new_path is None:
            return
        self._anchor_path = new_path
        tree = self.query_one("#switcher-tree", Tree)
        tree.clear()
        tree.root.expand()
        self._populate_tree(tree)

    # --- actions ---------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)
