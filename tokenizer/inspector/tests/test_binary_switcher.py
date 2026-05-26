"""Pilot tests for the binary-switcher modal + scan helpers.

Covers:

* :func:`scan_folder` reports loadable / non-loadable directories.
* :class:`BinarySwitcherDialog` builds a provider tree with the
  current path + per-binary rows + change-path entry.
* Click on a per-binary entry dismisses with a :class:`SwitchTarget`.
* Click on ``[open this folder]`` dismisses with ``binary=None``.

Gated on ``pytest.importorskip("textual")`` so the default ``nix
develop`` shell shows it as SKIPPED rather than as an import failure.
"""

from __future__ import annotations

import pytest


pytest.importorskip("textual")

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app import InspectorApp
from tokenizer.inspector._app._binary_switcher import (
    BinarySwitcherDialog,
    LoaderProvider,
    SwitchTarget,
)
from tokenizer.inspector._app._binary_switcher._scan import (
    FolderScanResult,
    binaries_in_folder,
    is_loadable_for,
    list_child_directories,
    scan_folder,
)
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
)


def _build_app(
    log_path: Path,
    *,
    memmap_path: Path | None = None,
    csv_path: Path | None = None,
) -> InspectorApp:
    handles = [FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="main")]
    factory = MagicMock(spec=BackendFactory)
    factory.handles = handles
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handles[0]
    backend.closed = False
    backend.variants.return_value = []
    backend.blocks.return_value = []
    backend.render_block.return_value = ()
    factory.make.return_value = backend
    return InspectorApp(
        factory=factory,
        log_path=log_path,
        memmap_path=memmap_path,
        csv_path=csv_path,
    )


# ---------------------------------------------------------------------------
# Scan helpers (filesystem-level)
# ---------------------------------------------------------------------------


def test_scan_folder_empty_returns_no_binaries():
    """An empty directory yields a non-loadable scan."""
    with tempfile.TemporaryDirectory() as td:
        result = scan_folder(Path(td), LoaderProvider.MEMMAP)
        assert isinstance(result, FolderScanResult)
        assert result.binaries == ()
        assert not result.loadable


def test_scan_folder_memmap_detects_function_names_sidecar():
    """A directory carrying ``<binary>_function_names.txt`` is loadable."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        (path / "ncat_function_names.txt").write_text("")
        (path / "nmap_function_names.txt").write_text("")
        result = scan_folder(path, LoaderProvider.MEMMAP)
        assert sorted(result.binaries) == ["ncat", "nmap"]
        assert result.loadable


def test_is_loadable_for_predicate_matches_scan():
    """The compact :func:`is_loadable_for` predicate agrees with
    :func:`scan_folder.loadable`."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        assert not is_loadable_for(path, LoaderProvider.MEMMAP)
        (path / "foo_function_names.txt").write_text("")
        assert is_loadable_for(path, LoaderProvider.MEMMAP)


def test_list_child_directories_skips_hidden_and_files():
    """Subdir enumeration ignores files + dotfiles."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        (path / "sub_a").mkdir()
        (path / "sub_b").mkdir()
        (path / ".hidden").mkdir()
        (path / "afile.txt").write_text("")
        result = list_child_directories(path)
        assert [p.name for p in result] == ["sub_a", "sub_b"]


# ---------------------------------------------------------------------------
# BinarySwitcherDialog
# ---------------------------------------------------------------------------


def test_binary_switcher_b_opens_dialog():
    """``b`` pushes the :class:`BinarySwitcherDialog` modal."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(log_path, memmap_path=Path(td))
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                assert isinstance(
                    app.screen_stack[-1], BinarySwitcherDialog
                )

    asyncio.run(runner())


def test_binary_switcher_lists_loadable_binaries():
    """A loadable memmap dir surfaces per-binary rows + the
    ``[open this folder]`` row."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "ncat_function_names.txt").write_text("")
            (path / "nmap_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(log_path, memmap_path=path)
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                assert isinstance(dialog, BinarySwitcherDialog)
                from textual.widgets import Tree

                tree = dialog.query_one("#switcher-tree", Tree)
                # Walk every label string in the tree, concatenated.
                labels: list[str] = []

                def walk(node):
                    label = node.label
                    labels.append(
                        label.plain if hasattr(label, "plain") else str(label)
                    )
                    for child in node.children:
                        walk(child)

                walk(tree.root)
                joined = " | ".join(labels)
                assert "ncat" in joined
                assert "nmap" in joined
                assert "[open this folder]" in joined
                assert "change path..." in joined

    asyncio.run(runner())


def test_binary_switcher_dismiss_with_target_on_binary_select():
    """Selecting a per-binary row dismisses with a typed
    :class:`SwitchTarget` carrying the binary name."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "ncat_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(log_path, memmap_path=path)
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                assert isinstance(dialog, BinarySwitcherDialog)
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _BinaryRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                # Find the ncat row.
                target_node = None

                def walk(node):
                    nonlocal target_node
                    if isinstance(node.data, _BinaryRow) and node.data.binary == "ncat":
                        target_node = node
                        return
                    for child in node.children:
                        walk(child)

                walk(tree.root)
                assert target_node is not None, "ncat row not in tree"
                results: list = []
                dialog.dismiss = lambda v: results.append(v)  # type: ignore
                from textual.widgets import Tree as TreeWidget
                event = TreeWidget.NodeSelected(target_node)
                dialog.on_tree_node_selected(event)
                assert len(results) == 1
                target = results[0]
                assert isinstance(target, SwitchTarget)
                assert target.provider is LoaderProvider.MEMMAP
                assert target.binary == "ncat"
                assert target.path == path

    asyncio.run(runner())


def test_binary_switcher_open_this_folder_yields_no_binary():
    """``[open this folder]`` dismisses with ``binary=None``."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "ncat_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(log_path, memmap_path=path)
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _OpenFolderRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                target_node = None

                def walk(node):
                    nonlocal target_node
                    if isinstance(node.data, _OpenFolderRow):
                        target_node = node
                        return
                    for child in node.children:
                        walk(child)

                walk(tree.root)
                assert target_node is not None
                results: list = []
                dialog.dismiss = lambda v: results.append(v)  # type: ignore
                from textual.widgets import Tree as TreeWidget
                event = TreeWidget.NodeSelected(target_node)
                dialog.on_tree_node_selected(event)
                assert len(results) == 1
                target = results[0]
                assert target.binary is None
                assert target.path == path

    asyncio.run(runner())
