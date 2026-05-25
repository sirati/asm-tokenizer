"""Pilot tests for the inspector tree's auto-expand-single-child rule.

Universal rule (per user-stated UX): when a parent's
:meth:`Node.expand` returns exactly one expandable child, the
intermediate selection row carries no information -- the user would
always click through it. The tree dispatcher
(:meth:`InspectorApp._on_node_expanded`) auto-expands the lone child
so the deeper content surfaces without an extra keypress. The cascade
is recursive: a chain of 1-child wrappers unfolds end-to-end.

The tests below drive :class:`InspectorApp` via a Textual Pilot and
exercise:

* exactly-one-child + expandable -> auto-expand fires
* exactly-one-child + terminal -> auto-expand does NOT fire
* two-or-more children (heterogeneous) -> auto-expand does NOT fire
* multi-level chain of single-child wrappers -> cascades all levels
* top-level FunctionNode list -> never auto-expanded (the
  ``compose()`` root has no model, dispatcher returns early)
* idempotent vs. capture-on-rebuild's restore walk

The file is gated on ``pytest.importorskip("textual")`` so the default
``nix develop`` shell (textual-free) shows it as SKIPPED rather than
as an import failure. Lives in its own module per the project's
file-size cap (sibling :mod:`test_app_error_handling` is already
near the 400-LOC threshold).
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
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    BlockKind,
    FunctionHandle,
    RenderBackend,
)
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    BlockNode,
    FunctionNode,
)


# ---------------------------------------------------------------------------
# Helpers (mirror the patterns in ``test_app_error_handling`` so test
# fixtures stay consistent across the inspector test surface).
# ---------------------------------------------------------------------------


def _make_factory(names: list[str]) -> MagicMock:
    """Minimal :class:`BackendFactory` mock with the methods the dispatcher uses."""
    handles = [
        FunctionHandle(arm=SectionKind.MATCHED, idx=i, name=name)
        for i, name in enumerate(names)
    ]
    factory = MagicMock(spec=BackendFactory)
    factory.handles = handles
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handles[0] if handles else None
    backend.closed = False
    backend.variants.return_value = []
    backend.blocks.return_value = []
    backend.render_block.return_value = ()
    factory.make.return_value = backend
    return factory


def _build_app(names: list[str], log_path: Path) -> InspectorApp:
    return InspectorApp(factory=_make_factory(names), log_path=log_path)


def _root_tree_node(app: InspectorApp):
    """Return ``(tree, first_FunctionNode_tree_node)``."""
    from tokenizer.inspector._app import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    return tree, tree.root.children[0]


async def _expand_function_with_children(app: InspectorApp, pilot, children: list):
    """Stub the root FunctionNode's ``expand`` + drive expand-then-pause.

    Returns ``(tree, fn_tree_node, child_tree_nodes)``. Mirrors the
    helper in :mod:`test_app_error_handling` so the test surface is
    consistent.
    """
    tree, fn_tree_node = _root_tree_node(app)
    fn_model: FunctionNode = fn_tree_node.data
    fn_model.expand = MagicMock(return_value=children)
    fn_tree_node.expand()
    await pilot.pause()
    return tree, fn_tree_node, list(fn_tree_node.children)


def _make_expandable_block(*, expand_returns: list) -> BlockNode:
    """Synthetic :class:`BlockNode` whose ``expand`` returns the given list.

    A :class:`BlockNode` has ``can_expand=True`` by default; the
    factory/backend refs are mocks because the dispatcher never reads
    through them once ``expand`` is stubbed.
    """
    block = BlockNode(
        factory=MagicMock(),
        backend=MagicMock(),
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=0,
        preview="x",
    )
    block.expand = MagicMock(return_value=expand_returns)
    return block


# ---------------------------------------------------------------------------
# Single-child auto-expand: the core rule.
# ---------------------------------------------------------------------------


def test_single_expandable_child_auto_expands():
    """FunctionNode with exactly one expandable child -> child auto-expands."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # A single :class:`BlockNode` child whose own ``expand``
                # returns no grandchildren (so the cascade stops cleanly).
                lone_block = _make_expandable_block(expand_returns=[])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [lone_block]
                )
                assert len(children) == 1
                only_child = children[0]
                # The lone child was auto-expanded by the dispatcher; the
                # spy was invoked exactly once (the auto-expand cascade,
                # not a user keypress).
                assert only_child.is_expanded is True
                lone_block.expand.assert_called_once()

    asyncio.run(runner())


def test_single_terminal_child_does_not_auto_expand():
    """FunctionNode with exactly one ``can_expand=False`` child -> no auto-expand."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # AsmLeaf with no openables is terminal (can_expand=False).
                lone_leaf = AsmLeaf(text="just-one")
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [lone_leaf]
                )
                assert len(children) == 1
                only_child = children[0]
                # No auto-expand: nothing to surface under a terminal row.
                assert only_child.is_expanded is False

    asyncio.run(runner())


def test_two_children_does_not_auto_expand():
    """FunctionNode with two children -> neither auto-expanded (user picks)."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Two expandable children -> the selection node carries
                # information (which one?); we leave it to the user.
                first = _make_expandable_block(expand_returns=[])
                second = _make_expandable_block(expand_returns=[])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [first, second]
                )
                assert len(children) == 2
                assert children[0].is_expanded is False
                assert children[1].is_expanded is False
                first.expand.assert_not_called()
                second.expand.assert_not_called()

    asyncio.run(runner())


def test_heterogeneous_children_does_not_auto_expand():
    """Mixed children (1 expandable + 1 leaf) -> no auto-skip.

    Strict interpretation of "exactly one child": count is on the
    mounted child list, not on the expandable subset. This preserves
    the pinned-variant fix's heterogeneous shape (N BlockNodes +
    1 ShowAllVariantsNode under an InlineCallNode -- multiple children,
    no auto-skip) without a special case.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                expandable = _make_expandable_block(expand_returns=[])
                leaf = AsmLeaf(text="terminal-sibling")
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [expandable, leaf]
                )
                assert len(children) == 2
                # The lone expandable did NOT pre-empt the leaf row's
                # visibility under the parent.
                assert children[0].is_expanded is False
                expandable.expand.assert_not_called()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Recursion: a chain of 1-child wrappers unfolds end-to-end.
# ---------------------------------------------------------------------------


def test_three_level_chain_auto_expands_all():
    """3-level chain of 1-child nodes -> every wrapper auto-expanded.

    Models the Calloc case: Function ID has 1 child -> auto-expand;
    that child is an AsmLeaf with one openable that resolves to an
    InlineCallNode with 1 pinned variant -> auto-expand; the variant
    has multiple blocks (or 1 block here for the chain test) -> the
    cascade stops at the terminal level.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Build bottom-up so each parent's expand spy returns the
                # next-level child. The leaf at the bottom has zero
                # grandchildren so the cascade has a clear base case.
                leaf = _make_expandable_block(expand_returns=[])
                middle = _make_expandable_block(expand_returns=[leaf])
                top = _make_expandable_block(expand_returns=[middle])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [top]
                )

                # 1 child at the FunctionNode level -> top auto-expanded.
                assert len(children) == 1
                top_tree_node = children[0]
                assert top_tree_node.is_expanded is True
                top.expand.assert_called_once()

                # Top's lone child is middle -> middle auto-expanded.
                assert len(top_tree_node.children) == 1
                middle_tree_node = top_tree_node.children[0]
                assert middle_tree_node.is_expanded is True
                middle.expand.assert_called_once()

                # Middle's lone child is leaf -> leaf auto-expanded.
                assert len(middle_tree_node.children) == 1
                leaf_tree_node = middle_tree_node.children[0]
                assert leaf_tree_node.is_expanded is True
                leaf.expand.assert_called_once()

    asyncio.run(runner())


def test_chain_stops_when_branch_becomes_multi_child():
    """Cascade stops at the level whose expand returns 2+ children."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # top -> middle (auto) -> [block_a, block_b] (NO auto)
                block_a = _make_expandable_block(expand_returns=[])
                block_b = _make_expandable_block(expand_returns=[])
                middle = _make_expandable_block(expand_returns=[block_a, block_b])
                top = _make_expandable_block(expand_returns=[middle])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [top]
                )

                top_tree_node = children[0]
                middle_tree_node = top_tree_node.children[0]
                # The 2-child level mounts both, but neither auto-expands.
                assert len(middle_tree_node.children) == 2
                for grandchild in middle_tree_node.children:
                    assert grandchild.is_expanded is False
                block_a.expand.assert_not_called()
                block_b.expand.assert_not_called()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Top-level FunctionNode list is never auto-expanded.
# ---------------------------------------------------------------------------


def test_root_with_single_function_does_not_auto_expand():
    """One FunctionNode under the tree root -> NOT auto-expanded.

    The ``compose()`` root has ``data is None``, so the dispatcher
    returns early before the auto-expand check ever runs against it.
    This preserves the user-facing affordance: the root function list
    is the inspector's navigation entry point and must stay
    collapsible regardless of how few binaries are loaded.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["only_fn"], log_path)
            async with app.run_test() as pilot:
                await pilot.pause()
                _tree, fn_tree_node = _root_tree_node(app)
                # Root has 1 FunctionNode; it must remain user-collapsed
                # (no automatic expand off the back of compose()).
                assert fn_tree_node.is_expanded is False

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Idempotence: auto-expand does not re-fire on an already-expanded child.
# ---------------------------------------------------------------------------


def test_auto_expand_idempotent_when_child_already_expanded():
    """A pre-expanded lone child -> auto-expand stays a no-op.

    Mirrors the capture-on-rebuild interaction: if another code path
    already expanded the child before the dispatcher's auto-expand
    block ran (e.g. via the Order modal's identity-set restore), we
    must NOT post a duplicate :class:`Tree.NodeExpanded`. The guard
    is :attr:`TreeNode.is_expanded`.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                lone_block = _make_expandable_block(expand_returns=[])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [lone_block]
                )
                only_child = children[0]
                # After the first expand cycle, child is already expanded
                # by the auto-expand pass + the spy was called once.
                assert only_child.is_expanded is True
                lone_block.expand.assert_called_once()

                # Manually call expand() on the already-expanded child;
                # Textual posts another NodeExpanded which re-enters the
                # dispatcher. The dispatcher re-runs model.expand (it
                # remove_children + remounts), so the spy gets a SECOND
                # call. The auto-expand guard's job is then to NOT
                # post a third call. Verify the cascade settles at 2
                # (no infinite loop).
                only_child.expand()
                await pilot.pause()
                assert lone_block.expand.call_count == 2

    asyncio.run(runner())
