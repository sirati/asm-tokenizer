"""Pilot tests for the inspector tree's collapse-single-child-chain rule.

Universal rule (per user-stated UX): when a parent's
:meth:`Node.expand` returns exactly one expandable child, that child
is a content-free "selection of one" wrapper -- the inspector REMOVES
it from the rendered tree and surfaces its own expand result in its
place. The collapse is recursive: a chain of 1-expandable-child
wrappers unfolds end-to-end until either zero, two-or-more, or a
terminal child is reached.

Contrast with the prior in-tree-auto-expand variant: that flavour
kept wrappers visible (just initially expanded). The current contract
is stricter -- wrappers must be GONE so the deepest content sits
directly under the parent that actually carries information.

The tests below drive :class:`InspectorApp` via a Textual Pilot and
exercise:

* exactly-one-expandable-child -> wrapper REMOVED, grandchildren mounted
* exactly-one-terminal-child -> kept as-is (no chain to collapse into)
* two-or-more children (homogeneous or mixed) -> no collapse
* multi-level chain of single-child wrappers -> collapses fully
* top-level FunctionNode list -> never collapsed (root has no model)
* failure mid-chain -> wrapper kept visible, flagged ``is_failed``
* recursion depth bound -> defensive cutoff against pathological loops

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
from tokenizer.inspector._app._auto_expand import collapse_single_child_chains
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
# Single-child collapse: wrapper removed, grandchildren mounted.
# ---------------------------------------------------------------------------


def test_single_expandable_child_collapsed_grandchildren_mounted():
    """FunctionNode with one expandable child -> wrapper REMOVED, grandchildren under FunctionNode."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Grandchildren: two terminal leaves so the chain stops
                # (multi-child level) and gets mounted as-is.
                grand_a = AsmLeaf(text="grandchild-a")
                grand_b = AsmLeaf(text="grandchild-b")
                wrapper = _make_expandable_block(
                    expand_returns=[grand_a, grand_b]
                )
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [wrapper]
                )
                # The wrapper is GONE; the two grandchildren are now
                # the FunctionNode's direct children.
                assert len(children) == 2
                child_models = [c.data for c in children]
                assert child_models == [grand_a, grand_b]
                # The wrapper's expand was invoked during collapse.
                wrapper.expand.assert_called_once()

    asyncio.run(runner())


def test_single_terminal_child_kept_in_place():
    """FunctionNode with exactly one ``can_expand=False`` child -> child stays as-is.

    A terminal lone child has no deeper content to surface; the
    collapse cannot "expand into" it, so the leaf is mounted directly
    under the parent (without any wrapper between them since the
    FunctionNode IS the parent already).
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                lone_leaf = AsmLeaf(text="just-one")
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [lone_leaf]
                )
                # 1 child mounted, NOT expanded (terminal).
                assert len(children) == 1
                only_child = children[0]
                assert only_child.data is lone_leaf
                assert only_child.is_expanded is False

    asyncio.run(runner())


def test_two_children_no_collapse():
    """FunctionNode with two children -> both mounted, neither auto-expanded."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Two expandable children -> the selection node carries
                # information (which one?); the user picks.
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


def test_heterogeneous_children_no_collapse():
    """Mixed children (1 expandable + 1 leaf) -> no collapse.

    Strict interpretation of "exactly one child": count is on the
    mounted child list, not on the expandable subset. This preserves
    the pinned-variant fix's heterogeneous shape (N BlockNodes +
    1 ShowAllVariantsNode under an InlineCallNode -- multiple children,
    no collapse) without a special case.
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
                # Both rows surfaced under FunctionNode; the expandable
                # one was NOT auto-expanded.
                assert children[0].is_expanded is False
                expandable.expand.assert_not_called()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Recursion: a chain of 1-expandable-child wrappers collapses fully.
# ---------------------------------------------------------------------------


def test_three_level_chain_collapses_to_deepest_content():
    """3-level chain of 1-child wrappers -> all wrappers REMOVED.

    Models the Calloc case: Function ID has 1 child (AsmLeaf) -> collapse;
    that AsmLeaf has 1 openable resolving to an InlineCallNode -> collapse;
    the InlineCallNode surfaces blocks. Only the deepest mounted level
    becomes visible under Function ID; the two intermediate wrappers
    disappear entirely.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Build bottom-up: the bottom level is multi-child so the
                # chain has a clear stopping point with visible content.
                final_a = AsmLeaf(text="final-a")
                final_b = AsmLeaf(text="final-b")
                middle = _make_expandable_block(expand_returns=[final_a, final_b])
                top = _make_expandable_block(expand_returns=[middle])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [top]
                )

                # The two final leaves are mounted DIRECTLY under the
                # FunctionNode; ``top`` and ``middle`` wrappers are GONE.
                assert len(children) == 2
                assert [c.data for c in children] == [final_a, final_b]
                top.expand.assert_called_once()
                middle.expand.assert_called_once()

    asyncio.run(runner())


def test_chain_stops_when_branch_becomes_multi_child():
    """Cascade stops at the level whose expand returns 2+ children.

    The multi-child level's siblings get mounted under the original
    parent (the wrappers above the multi-child level are gone).
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # top -> middle (collapses) -> [block_a, block_b] (NO collapse)
                block_a = _make_expandable_block(expand_returns=[])
                block_b = _make_expandable_block(expand_returns=[])
                middle = _make_expandable_block(expand_returns=[block_a, block_b])
                top = _make_expandable_block(expand_returns=[middle])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [top]
                )

                # Top + middle wrappers GONE; block_a + block_b are
                # mounted as direct children of the FunctionNode.
                assert len(children) == 2
                assert [c.data for c in children] == [block_a, block_b]
                for grandchild in children:
                    assert grandchild.is_expanded is False
                block_a.expand.assert_not_called()
                block_b.expand.assert_not_called()

    asyncio.run(runner())


def test_chain_bottoms_out_at_empty_expand():
    """A chain whose deepest wrapper returns ``[]`` -> parent shows no children."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                # Lone wrapper whose expand returns nothing -- the chain
                # bottoms out at empty; the parent mounts zero children.
                lone = _make_expandable_block(expand_returns=[])
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [lone]
                )
                assert children == []
                lone.expand.assert_called_once()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Top-level FunctionNode list is never collapsed.
# ---------------------------------------------------------------------------


def test_root_with_single_function_does_not_collapse():
    """One FunctionNode under the tree root -> never auto-opened or removed.

    The ``compose()`` root has ``data is None``, so the dispatcher
    returns early before the collapse check ever runs against it.
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
                # (no automatic expand off the back of compose()) and
                # visible (the root is NOT subject to collapse).
                assert fn_tree_node.is_expanded is False
                assert fn_tree_node.data is not None

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Failure mid-chain: wrapper kept visible with is_failed=True.
# ---------------------------------------------------------------------------


def test_failed_wrapper_kept_visible_for_error_surfacing():
    """If a wrapper's ``expand`` raises during collapse, the wrapper is kept.

    The dispatcher's ``_safe_expand_one`` callback logs + flips
    ``is_failed`` on the wrapper; the collapse helper sees ``None``
    and stops at this level, returning the wrapper as-is. The user
    sees the wrapper with the ``[*]`` glyph; expanding it manually
    re-runs the failing ``expand`` through the main dispatcher path,
    which attaches the dim-red error leaf.
    """

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            app = _build_app(["main"], log_path)
            async with app.run_test() as pilot:
                bad_wrapper = _make_expandable_block(expand_returns=[])
                bad_wrapper.expand = MagicMock(
                    side_effect=RuntimeError("simulated failure")
                )
                _tree, _fn, children = await _expand_function_with_children(
                    app, pilot, [bad_wrapper]
                )
                # The wrapper is mounted (collapse stopped on failure)
                # with is_failed flipped by the dispatcher's callback.
                assert len(children) == 1
                assert children[0].data is bad_wrapper
                assert bad_wrapper.is_failed is True

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Recursion-depth bound: pathological loops are cut off defensively.
# ---------------------------------------------------------------------------


def test_collapse_respects_depth_limit():
    """An infinite 1-child chain is bounded by ``depth_limit``.

    Drive the standalone helper (not via Pilot) so we can detect the
    EXACT number of ``expand_one`` calls and assert the bound holds.
    """
    # Pathological model whose ``expand`` returns itself wrapped in a
    # single-child list -- a chain that would loop forever without the
    # depth bound.
    loop_node = _make_expandable_block(expand_returns=[])
    loop_node.expand = MagicMock(return_value=[loop_node])

    call_count = [0]

    def expand_one(node):
        call_count[0] += 1
        return list(node.expand())

    out = collapse_single_child_chains(
        [loop_node], expand_one=expand_one, depth_limit=5
    )
    # The chain hits the bound after exactly ``depth_limit`` expansions;
    # the result is the most-recent single-child list we walked into.
    assert call_count[0] == 5
    # And the helper still returns a list (not None / not raised).
    assert isinstance(out, list)


def test_collapse_helper_handles_failure_via_none():
    """``expand_one`` returning ``None`` halts the chain at that level.

    Direct standalone-helper test so we don't depend on Textual + the
    dispatcher's full plumbing for the failure-return contract.
    """
    wrapper = _make_expandable_block(expand_returns=[])
    # ``expand_one`` returning None mirrors a failed wrapper expand.
    out = collapse_single_child_chains(
        [wrapper], expand_one=lambda _n: None
    )
    # Wrapper stays in the returned list so the dispatcher can mount
    # it with the failure flag.
    assert out == [wrapper]
