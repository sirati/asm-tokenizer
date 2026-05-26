"""Pilot tests for the binary-switcher modal + scan helpers.

Covers:

* :func:`scan_folder` reports loadable / non-loadable directories.
* :class:`BinarySwitcherDialog` builds a binary-first tree with the
  current path header + per-binary nodes (provider children) +
  ``change path...`` as a top-level sibling.
* Click on a per-provider leaf dismisses with a :class:`SwitchTarget`.
* No ``[open this folder]`` row at the main dialog level (lifted into
  the folder picker), so multi-binary paths never produce a row that
  would crash at the resolver.
* :class:`FolderPickerDialog` greens any-provider-loadable subfolders
  and exposes ``[open this folder]`` rows at every drill level.

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
    FolderPickerDialog,
    LoaderProvider,
    SwitchTarget,
    perform_switch,
)
from tokenizer.inspector._app._binary_switcher._provider import (
    resolve_provider_dirs,
)
from tokenizer.inspector._app._binary_switcher._scan import (
    FolderScanResult,
    binaries_in_folder,
    is_loadable_for,
    is_loadable_for_any,
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
    path: Path | None = None,
    provider: LoaderProvider | None = None,
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
        path=path,
        provider=provider,
    )


def _walk_tree(node) -> list:
    """Yield every (node, label_plain) in a textual Tree (root-first)."""
    out: list = []

    def _go(n):
        label = n.label
        out.append((n, label.plain if hasattr(label, "plain") else str(label)))
        for child in n.children:
            _go(child)

    _go(node)
    return out


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


def test_is_loadable_for_any_unions_across_providers():
    """The any-provider predicate is True whenever ANY provider matches."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        assert not is_loadable_for_any(path)
        (path / "foo_function_names.txt").write_text("")
        assert is_loadable_for_any(path)


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
            app = _build_app(
                log_path, path=Path(td), provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                assert isinstance(
                    app.screen_stack[-1], BinarySwitcherDialog
                )

    asyncio.run(runner())


def test_binary_switcher_top_level_layout_is_binary_first():
    """Top level: path header + binary nodes + ``change path...``;
    binary nodes carry provider children, NO ``[open this folder]``
    row at the main dialog level."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "ncat_function_names.txt").write_text("")
            (path / "nmap_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(
                log_path, path=path, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                assert isinstance(dialog, BinarySwitcherDialog)
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _BinaryRoot,
                    _ChangePathRow,
                    _PathHeaderRow,
                    _ProviderRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                top = list(tree.root.children)
                # First: header label.
                assert isinstance(top[0].data, _PathHeaderRow)
                assert top[0].data.path == path
                # Last: change path... sibling at TOP level.
                assert isinstance(top[-1].data, _ChangePathRow)
                # Middle: one node per binary.
                bin_nodes = [n for n in top if isinstance(n.data, _BinaryRoot)]
                assert sorted(n.data.binary for n in bin_nodes) == [
                    "ncat",
                    "nmap",
                ]
                # Each binary node has a memmap child (matching the
                # only provider that has data here).
                for bnode in bin_nodes:
                    children = list(bnode.children)
                    assert children, f"{bnode.data.binary} has no provider rows"
                    for child in children:
                        assert isinstance(child.data, _ProviderRow)
                        assert child.data.provider is LoaderProvider.MEMMAP
                        assert child.data.binary == bnode.data.binary
                # No [open this folder] anywhere in the dialog tree.
                joined = " | ".join(lbl for _, lbl in _walk_tree(tree.root))
                assert "[open this folder]" not in joined

    asyncio.run(runner())


def test_binary_switcher_multi_binary_path_does_not_crash_on_open():
    """The user's reproducer: a multi-binary path opens cleanly; the
    only commit rows are per-binary leaves so no ``binary=None`` ever
    reaches the resolver."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            # Four binaries — matches the user's failing case shape.
            for name in ("libz.so.1.2.11", "minigzip", "minigzip64", "minigzipsh"):
                (path / f"{name}_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(
                log_path, path=path, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                assert isinstance(dialog, BinarySwitcherDialog)
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _ProviderRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                # Every commit-able row carries an explicit binary name.
                for node, _ in _walk_tree(tree.root):
                    if isinstance(node.data, _ProviderRow):
                        assert node.data.binary is not None
                        assert node.data.binary != ""

    asyncio.run(runner())


def test_binary_switcher_dismiss_with_target_on_provider_select():
    """Selecting a provider leaf dismisses with a typed
    :class:`SwitchTarget` carrying the binary name."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "ncat_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(
                log_path, path=path, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                assert isinstance(dialog, BinarySwitcherDialog)
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _ProviderRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                target_node = None
                for node, _ in _walk_tree(tree.root):
                    if (
                        isinstance(node.data, _ProviderRow)
                        and node.data.binary == "ncat"
                        and node.data.provider is LoaderProvider.MEMMAP
                    ):
                        target_node = node
                        break
                assert target_node is not None, "ncat memmap leaf missing"
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


def test_binary_switcher_change_path_is_top_level_sibling():
    """The ``change path...`` row is a direct child of the tree root
    (sibling of the binary nodes), not nested inside any provider."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "ncat_function_names.txt").write_text("")
            log_path = path / "tui.log"
            app = _build_app(
                log_path, path=path, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _ChangePathRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                # Find the change-path row + its parent.
                change_rows = [
                    n
                    for n in tree.root.children
                    if isinstance(n.data, _ChangePathRow)
                ]
                assert len(change_rows) == 1, (
                    "change path... should appear exactly once at top level"
                )
                # Confirm there is NO change-path row deeper in the tree.
                deep = [
                    node
                    for node, _ in _walk_tree(tree.root)
                    if isinstance(node.data, _ChangePathRow)
                    and node is not change_rows[0]
                ]
                assert deep == []

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# FolderPickerDialog
# ---------------------------------------------------------------------------


def test_folder_picker_green_marks_loadable_subdir():
    """A subdir containing function-names sidecar gets a green label."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            loadable = base / "with_data"
            loadable.mkdir()
            (loadable / "foo_function_names.txt").write_text("")
            empty = base / "without_data"
            empty.mkdir()
            log_path = base / "tui.log"
            app = _build_app(
                log_path, path=base, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                dialog = FolderPickerDialog(start_path=base)
                await app.push_screen(dialog)
                await pilot.pause()
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._folder_picker import (
                    _FolderRow,
                )

                tree = dialog.query_one("#picker-tree", Tree)
                loadable_node = None
                empty_node = None
                for child in tree.root.children:
                    if isinstance(child.data, _FolderRow):
                        if child.data.path == loadable:
                            loadable_node = child
                        elif child.data.path == empty:
                            empty_node = child
                assert loadable_node is not None
                assert empty_node is not None
                assert loadable_node.data.loadable is True
                assert empty_node.data.loadable is False

    asyncio.run(runner())


def test_folder_picker_open_this_folder_present_at_root():
    """The picker root shows an ``[open this folder]`` row before any
    subfolders, so the user can commit the current level without
    drilling."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "sub_a").mkdir()
            log_path = base / "tui.log"
            app = _build_app(
                log_path, path=base, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                dialog = FolderPickerDialog(start_path=base)
                await app.push_screen(dialog)
                await pilot.pause()
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._folder_picker import (
                    _OpenFolderRow,
                )

                tree = dialog.query_one("#picker-tree", Tree)
                children = list(tree.root.children)
                assert len(children) >= 2
                # First child is [open this folder].
                assert isinstance(children[0].data, _OpenFolderRow)
                assert children[0].data.path == base

    asyncio.run(runner())


def test_folder_picker_open_this_folder_dismisses_with_path():
    """Selecting ``[open this folder]`` dismisses with the enclosing
    folder's path."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            log_path = base / "tui.log"
            app = _build_app(
                log_path, path=base, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                dialog = FolderPickerDialog(start_path=base)
                await app.push_screen(dialog)
                await pilot.pause()
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._folder_picker import (
                    _OpenFolderRow,
                )

                tree = dialog.query_one("#picker-tree", Tree)
                open_node = next(
                    child
                    for child in tree.root.children
                    if isinstance(child.data, _OpenFolderRow)
                )
                results: list = []
                dialog.dismiss = lambda v: results.append(v)  # type: ignore
                from textual.widgets import Tree as TreeWidget
                event = TreeWidget.NodeSelected(open_node)
                dialog.on_tree_node_selected(event)
                assert results == [base]

    asyncio.run(runner())


def test_folder_picker_subfolder_expansion_recursively_offers_open():
    """Expanding a subfolder shows ITS own ``[open this folder]`` row
    + its subfolders, so the user can drill arbitrarily deep."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            mid = base / "mid"
            mid.mkdir()
            inner = mid / "inner"
            inner.mkdir()
            log_path = base / "tui.log"
            app = _build_app(
                log_path, path=base, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                dialog = FolderPickerDialog(start_path=base)
                await app.push_screen(dialog)
                await pilot.pause()
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._folder_picker import (
                    _FolderRow,
                    _OpenFolderRow,
                )

                tree = dialog.query_one("#picker-tree", Tree)
                mid_node = next(
                    child
                    for child in tree.root.children
                    if isinstance(child.data, _FolderRow)
                    and child.data.path == mid
                )
                mid_node.expand()
                await pilot.pause()
                mid_children = list(mid_node.children)
                # First child of mid: [open this folder] for mid.
                assert isinstance(mid_children[0].data, _OpenFolderRow)
                assert mid_children[0].data.path == mid
                # Inner subfolder is also listed.
                inner_rows = [
                    c
                    for c in mid_children
                    if isinstance(c.data, _FolderRow) and c.data.path == inner
                ]
                assert len(inner_rows) == 1

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# perform_switch
# ---------------------------------------------------------------------------


def test_perform_switch_preserves_order_config():
    """:func:`perform_switch` keeps ``app._order_config`` across the swap."""
    from tokenizer.inspector._app._order import (
        AxisDescriptor,
        AxisKind,
        OrderConfig,
    )

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            path1 = Path(td) / "src1"
            path1.mkdir()
            (path1 / "foo_function_names.txt").write_text("")
            path2 = Path(td) / "src2"
            path2.mkdir()
            (path2 / "bar_function_names.txt").write_text("")
            log_path = Path(td) / "tui.log"
            app = _build_app(
                log_path, path=path1, provider=LoaderProvider.MEMMAP
            )

            # Set an order config so we can verify preservation.
            axis = AxisDescriptor(
                kind=AxisKind.POSITIONAL,
                key="arch_",
                label="arch",
            )
            app._order_config = OrderConfig(
                ordered_axes=(axis,),
                grouping_axes=frozenset(),
            )

            # Stub the opener dispatch so the test doesn't need real
            # memmap data (the real loader requires *_sections.bin etc.).
            new_factory = MagicMock(spec=BackendFactory)
            new_factory.handles = []
            new_factory.close = MagicMock()

            from tokenizer.inspector._app._binary_switcher import _switch

            original = _switch._OPENERS[LoaderProvider.MEMMAP]
            _switch._OPENERS[LoaderProvider.MEMMAP] = lambda p, b: new_factory
            try:
                async with app.run_test() as pilot:
                    await pilot.pause()
                    target = SwitchTarget(
                        provider=LoaderProvider.MEMMAP,
                        path=path2,
                        binary="bar",
                    )
                    perform_switch(app, target)
                    await pilot.pause()
                    assert app._order_config is not None
                    assert app._order_config.ordered_axes == (axis,)
                    assert app._factory is new_factory
                    assert app._current_path == path2
                    assert app._current_provider is LoaderProvider.MEMMAP
            finally:
                _switch._OPENERS[LoaderProvider.MEMMAP] = original

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Subdir-fallback scan (corpus_root/memmap/<bins> + corpus_root/<csv-bins>)
# ---------------------------------------------------------------------------


def test_resolve_provider_dirs_lists_in_place_and_subdir():
    """The candidate-dir helper returns (path, path/<provider.value>)."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        memmap_candidates = resolve_provider_dirs(base, LoaderProvider.MEMMAP)
        assert memmap_candidates == [base, base / "memmap"]
        csv_candidates = resolve_provider_dirs(base, LoaderProvider.CSV)
        assert csv_candidates == [base, base / "csv"]


def test_scan_folder_finds_memmap_bins_in_subdir():
    """When bins live under ``<corpus>/memmap/``, scanning the corpus
    root for the memmap provider still discovers them, and the
    effective dir is the subdir, not the root."""
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (memmap_subdir / "libz_function_names.txt").write_text("")
        (memmap_subdir / "minigzip_function_names.txt").write_text("")
        result = scan_folder(corpus, LoaderProvider.MEMMAP)
        assert sorted(result.binaries) == ["libz", "minigzip"]
        assert result.loadable
        assert result.binary_to_dir["libz"] == memmap_subdir
        assert result.binary_to_dir["minigzip"] == memmap_subdir


def test_scan_folder_in_place_wins_over_subdir():
    """When a binary lives BOTH in-place and in the subdir, the
    in-place location wins (matches the unified-vocab gate's
    explicit-copy-beats-inherited rule)."""
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        # Same binary name in both locations -> in-place wins.
        (corpus / "shared_function_names.txt").write_text("")
        (memmap_subdir / "shared_function_names.txt").write_text("")
        # Subdir-only binary still surfaces.
        (memmap_subdir / "only_in_subdir_function_names.txt").write_text("")
        result = scan_folder(corpus, LoaderProvider.MEMMAP)
        assert sorted(result.binaries) == ["only_in_subdir", "shared"]
        assert result.binary_to_dir["shared"] == corpus
        assert result.binary_to_dir["only_in_subdir"] == memmap_subdir


def test_binaries_in_folder_returns_sorted_union():
    """``binaries_in_folder`` returns the sorted name union across the
    in-place dir and the provider subdir."""
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (corpus / "a_function_names.txt").write_text("")
        (memmap_subdir / "z_function_names.txt").write_text("")
        (memmap_subdir / "m_function_names.txt").write_text("")
        names = binaries_in_folder(corpus, LoaderProvider.MEMMAP)
        assert names == ["a", "m", "z"]


def test_is_loadable_for_any_handles_subdir_fallback():
    """The folder picker's green-marking predicate honors the
    subdir-fallback so a corpus root whose data lives one level
    down still lights up."""
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        assert not is_loadable_for_any(corpus)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (memmap_subdir / "x_function_names.txt").write_text("")
        assert is_loadable_for_any(corpus)
        assert is_loadable_for(corpus, LoaderProvider.MEMMAP)
        assert not is_loadable_for(corpus, LoaderProvider.CSV)


def test_binary_switcher_dialog_shows_both_providers_with_mixed_layout():
    """User's reported failure case: ``<corpus>/memmap/<bins>`` plus
    ``<corpus>/<csv-bins-direct>``. The switcher dialog must surface
    BOTH provider rows per binary, and the memmap row's effective
    path must point at the memmap subdir, not the corpus root."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)
            memmap_subdir = corpus / "memmap"
            memmap_subdir.mkdir()
            # Memmap bins live one level down.
            (memmap_subdir / "libz_function_names.txt").write_text("")
            # CSV bins live directly at the corpus root.
            (corpus / "x64-gcc-13.2.0-O2_libz_output.csv").write_text("")
            log_path = corpus / "tui.log"
            app = _build_app(
                log_path, path=corpus, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                assert isinstance(dialog, BinarySwitcherDialog)
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _BinaryRoot,
                    _ProviderRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                bin_nodes = [
                    n
                    for n in tree.root.children
                    if isinstance(n.data, _BinaryRoot)
                ]
                assert [n.data.binary for n in bin_nodes] == ["libz"]
                provider_rows = [
                    child.data
                    for child in bin_nodes[0].children
                    if isinstance(child.data, _ProviderRow)
                ]
                # Both providers visible — the corpus-root layout
                # surfaces memmap (via subdir) AND csv (in-place).
                by_provider = {row.provider: row for row in provider_rows}
                assert LoaderProvider.MEMMAP in by_provider
                assert LoaderProvider.CSV in by_provider
                # Effective paths thread the right loader location.
                assert by_provider[LoaderProvider.MEMMAP].path == memmap_subdir
                assert by_provider[LoaderProvider.CSV].path == corpus
                # Anchor stays at the corpus root for next-dialog seed.
                assert (
                    by_provider[LoaderProvider.MEMMAP].anchor_path == corpus
                )
                assert by_provider[LoaderProvider.CSV].anchor_path == corpus

    asyncio.run(runner())


def test_binary_switcher_dialog_with_both_providers_in_subdirs():
    """Both providers' bins in their own subdirs:
    ``<corpus>/memmap/<memmap-bins>`` + ``<corpus>/csv/<csv-bins>``.
    Memmap discovery is non-recursive so its effective dir IS the
    subdir; CSV discovery rglobs so the in-place anchor already
    reaches into ``csv/`` (effective dir stays at the anchor by the
    in-place-wins rule)."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)
            memmap_subdir = corpus / "memmap"
            csv_subdir = corpus / "csv"
            memmap_subdir.mkdir()
            csv_subdir.mkdir()
            (memmap_subdir / "minigzip_function_names.txt").write_text("")
            (csv_subdir / "x64-gcc-13.2.0-O2_minigzip_output.csv").write_text("")
            log_path = corpus / "tui.log"
            app = _build_app(
                log_path, path=corpus, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _BinaryRoot,
                    _ProviderRow,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                bin_nodes = [
                    n
                    for n in tree.root.children
                    if isinstance(n.data, _BinaryRoot)
                ]
                assert [n.data.binary for n in bin_nodes] == ["minigzip"]
                provider_rows = [
                    child.data
                    for child in bin_nodes[0].children
                    if isinstance(child.data, _ProviderRow)
                ]
                by_provider = {row.provider: row for row in provider_rows}
                # Memmap effective path = subdir (non-recursive
                # discovery needs the exact dir to find sidecars).
                assert by_provider[LoaderProvider.MEMMAP].path == memmap_subdir
                # CSV effective path = anchor (rglob from anchor
                # already descends into ``csv/`` so the in-place
                # candidate wins per the search policy).
                assert by_provider[LoaderProvider.CSV].path == corpus
                # Anchor stays the corpus root regardless.
                assert (
                    by_provider[LoaderProvider.MEMMAP].anchor_path == corpus
                )
                assert by_provider[LoaderProvider.CSV].anchor_path == corpus

    asyncio.run(runner())


def test_binary_switcher_dialog_shows_no_binaries_for_unrelated_dir():
    """When ``path`` contains neither in-place nor subdir data, the
    dialog renders the empty-state message instead of fabricating
    binary rows."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)
            # An unrelated file + an unrelated subdir; no provider data.
            (corpus / "README.md").write_text("hi")
            (corpus / "logs").mkdir()
            (corpus / "logs" / "run.log").write_text("noise")
            log_path = corpus / "tui.log"
            app = _build_app(
                log_path, path=corpus, provider=LoaderProvider.MEMMAP
            )
            async with app.run_test() as pilot:
                await pilot.press("b")
                await pilot.pause()
                dialog = app.screen_stack[-1]
                from textual.widgets import Tree
                from tokenizer.inspector._app._binary_switcher._dialog import (
                    _BinaryRoot,
                )

                tree = dialog.query_one("#switcher-tree", Tree)
                bin_nodes = [
                    n
                    for n in tree.root.children
                    if isinstance(n.data, _BinaryRoot)
                ]
                assert bin_nodes == []


def test_perform_switch_threads_effective_path_to_opener():
    """``perform_switch`` calls the opener with ``target.path``
    (effective) while storing ``target.anchor_path`` on the App so
    re-opening the dialog still anchors at the corpus root."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)
            memmap_subdir = corpus / "memmap"
            memmap_subdir.mkdir()
            (memmap_subdir / "foo_function_names.txt").write_text("")
            log_path = corpus / "tui.log"
            app = _build_app(
                log_path, path=corpus, provider=LoaderProvider.MEMMAP
            )

            captured: list[Path] = []
            new_factory = MagicMock(spec=BackendFactory)
            new_factory.handles = []
            new_factory.close = MagicMock()

            def fake_opener(path: Path, binary: str | None) -> BackendFactory:
                captured.append(path)
                return new_factory

            from tokenizer.inspector._app._binary_switcher import _switch

            original = _switch._OPENERS[LoaderProvider.MEMMAP]
            _switch._OPENERS[LoaderProvider.MEMMAP] = fake_opener
            try:
                async with app.run_test() as pilot:
                    await pilot.pause()
                    target = SwitchTarget(
                        provider=LoaderProvider.MEMMAP,
                        path=memmap_subdir,
                        binary="foo",
                        anchor_path=corpus,
                    )
                    perform_switch(app, target)
                    await pilot.pause()
                    # Opener saw the effective subdir.
                    assert captured == [memmap_subdir]
                    # App-side state tracks the anchor, not the subdir.
                    assert app._current_path == corpus
                    assert app._current_provider is LoaderProvider.MEMMAP
            finally:
                _switch._OPENERS[LoaderProvider.MEMMAP] = original

    asyncio.run(runner())


def test_switch_target_anchor_defaults_to_path():
    """Legacy two-argument :class:`SwitchTarget` construction (no
    ``anchor_path``) defaults the anchor to ``path`` so older callers
    keep their pre-subdir semantics."""
    target = SwitchTarget(
        provider=LoaderProvider.MEMMAP,
        path=Path("/tmp/foo"),
        binary="x",
    )
    assert target.anchor_path == Path("/tmp/foo")


# ---------------------------------------------------------------------------
# CLI resolver subdir-fallback ($ python -m tokenizer.inspector PATH ...)
# ---------------------------------------------------------------------------


def test_resolve_binary_memmap_falls_back_to_memmap_subdir():
    """``_resolve_binary_memmap`` returns the effective subdir when
    bins live in ``<path>/memmap/`` and the user-supplied path is the
    corpus root."""
    from tokenizer.inspector._args import _resolve_binary_memmap

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (memmap_subdir / "bar_function_names.txt").write_text("")
        resolved = _resolve_binary_memmap(corpus, "bar")
        assert resolved.path == memmap_subdir
        assert resolved.name == "bar"


def test_resolve_binary_memmap_in_place_takes_precedence():
    """When the binary lives BOTH in-place and in the subdir, the
    in-place dir wins (matches the discovery union semantics)."""
    from tokenizer.inspector._args import _resolve_binary_memmap

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (corpus / "dup_function_names.txt").write_text("")
        (memmap_subdir / "dup_function_names.txt").write_text("")
        resolved = _resolve_binary_memmap(corpus, "dup")
        assert resolved.path == corpus


def test_resolve_binary_memmap_auto_detect_in_subdir():
    """``--binary`` omitted: a single binary in the memmap subdir
    succeeds and returns its effective dir."""
    from tokenizer.inspector._args import _resolve_binary_memmap

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (memmap_subdir / "only_function_names.txt").write_text("")
        resolved = _resolve_binary_memmap(corpus, None)
        assert resolved.path == memmap_subdir
        assert resolved.name == "only"


def test_resolve_binary_csv_handles_subdir_via_recursion():
    """:func:`discover_binaries_csv` rglobs, so a CSV under
    ``<corpus>/csv/`` is reachable from ``<corpus>`` directly — the
    in-place candidate wins per the search policy and the resolver
    returns ``<corpus>`` as the effective dir. The FTL loader rglobs
    from there too, so passing the anchor IS the correct loader path.
    """
    from tokenizer.inspector._args import _resolve_binary_csv

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        csv_subdir = corpus / "csv"
        csv_subdir.mkdir()
        (csv_subdir / "x64-gcc-13.2.0-O2_qux_output.csv").write_text("")
        resolved = _resolve_binary_csv(corpus, "qux")
        assert resolved.path == corpus
        assert resolved.name == "qux"


def test_parse_args_populates_effective_path():
    """The CLI's parse_args fills ``ns.effective_path`` with the
    resolver-side dir so ``__main__`` can pass it to the factory
    while keeping ``ns.path`` as the user's anchor."""
    from tokenizer.inspector._args import parse_args

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        memmap_subdir = corpus / "memmap"
        memmap_subdir.mkdir()
        (memmap_subdir / "main_function_names.txt").write_text("")
        ns = parse_args([str(corpus), "--memmap"])
        assert ns.path == corpus
        assert ns.effective_path == memmap_subdir
        assert ns.binary == "main"
