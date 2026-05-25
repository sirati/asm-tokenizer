"""Pilot tests for the inspector Order modal + grouping pass.

Covers:

* :class:`OrderConfig` value-equality (cluster #10 -- pure data
  descriptors, two builds compare equal).
* :func:`extract_axis_value` per :class:`AxisKind` (POSITIONAL,
  BITWIDTH, EXTRA_META).
* :func:`group_variants` with zero / one grouping axis.
* :class:`OrderDialog` accept via ``ctrl+s`` -> :class:`OrderAccepted`.
* :class:`OrderDialog` accept via :class:`Button` press.
* :class:`OrderDialog` escape -> :class:`OrderCancelled`.
* Expand-state preservation: open dialog, reorder + accept, verify
  previously-expanded variant surfaces.

The file is gated on ``pytest.importorskip("textual")`` +
``pytest.importorskip("natsort")`` so the default ``nix develop`` shell
shows it as SKIPPED rather than as an import failure.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")
pytest.importorskip("natsort")

import asyncio
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app import InspectorApp
from tokenizer.inspector._app._order import (
    AxisDescriptor,
    AxisKind,
    BITWIDTH_AXIS_KEY,
    OrderAccepted,
    OrderCancelled,
    OrderConfig,
    OrderDialog,
    VariantGroupNode,
    build_canonical_axes,
    build_extra_meta_axis,
    extract_axis_value,
    group_variants,
)
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    BlockKind,
    FunctionHandle,
    RenderBackend,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._tree_model import FunctionNode, VariantNode
from tokenizer.variant_info import VariantIdentity
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rv(
    *,
    variant_idx: int,
    arch: str = "x64",
    compiler: str = "clang",
    cver: str = "10",
    opt: str = "O2",
    extra: dict[str, str] | None = None,
    pkg: str = "nping",
) -> RenderedVariant:
    """Build a minimal :class:`RenderedVariant` for grouping tests."""
    label_axes = types.MappingProxyType(
        {
            ARCH_PREFIX: arch,
            COMP_PREFIX: compiler,
            CVER_PREFIX: cver,
            OPT_PREFIX: opt,
        }
    )
    extra_metadata = types.MappingProxyType(dict(extra or {}))
    identity = VariantIdentity(
        arch=arch,
        compiler=compiler,
        compiler_version=cver,
        opt=opt,
        pkg=pkg,
        variant_id=variant_idx,
    )
    return RenderedVariant(
        variant_idx=variant_idx,
        label_axes=label_axes,
        extra_metadata=extra_metadata,
        variant_identity=identity,
    )


def _make_variant_node(rv: RenderedVariant) -> VariantNode:
    factory = MagicMock(spec=BackendFactory)
    backend = MagicMock(spec=RenderBackend)
    return VariantNode(
        factory=factory,
        backend=backend,
        variant_idx=rv.variant_idx,
        label_axes=rv.label_axes,
    )


def _make_factory_with_rvs(
    *,
    name: str,
    rvs: list[RenderedVariant],
) -> MagicMock:
    """Build a factory whose backend reports ``rvs`` via ``variants()``."""
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name=name)
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    backend = MagicMock(spec=RenderBackend)
    backend.handle = handle
    backend.closed = False
    backend.variants.return_value = rvs
    # Each variant has ONE body block so the variant row is expandable
    # without crashing the dispatcher.
    backend.blocks.return_value = [
        RenderedBlock(kind=BlockKind.BODY, block_idx=0, preview="block-preview")
    ]
    backend.render_block.return_value = ()
    factory.make.return_value = backend
    return factory


def _root_tree_node(app: InspectorApp):
    from tokenizer.inspector._app import _InspectorTree

    tree = app.query_one("#tree", _InspectorTree)
    return tree, tree.root.children[0]


# ---------------------------------------------------------------------------
# Pure-unit tests (no Pilot) -- axis model + grouping pass.
# ---------------------------------------------------------------------------


def test_order_config_equality_value_based():
    """Two configs built from separate factory calls compare equal
    (cluster #10): :class:`AxisDescriptor` is pure data."""
    axes_a = build_canonical_axes()
    axes_b = build_canonical_axes()
    assert axes_a == axes_b  # tuple of frozen dataclasses

    cfg_a = OrderConfig(
        ordered_axes=axes_a, grouping_axes=frozenset({axes_a[0]})
    )
    cfg_b = OrderConfig(
        ordered_axes=axes_b, grouping_axes=frozenset({axes_b[0]})
    )
    assert cfg_a == cfg_b
    assert hash(cfg_a) == hash(cfg_b)


def test_extract_axis_value_positional():
    rv = _make_rv(variant_idx=0, arch="x64", opt="O2")
    arch_axis = AxisDescriptor(kind=AxisKind.POSITIONAL, key=ARCH_PREFIX, label="arch")
    opt_axis = AxisDescriptor(kind=AxisKind.POSITIONAL, key=OPT_PREFIX, label="opt")
    assert extract_axis_value(arch_axis, rv) == "x64"
    assert extract_axis_value(opt_axis, rv) == "O2"


def test_extract_axis_value_bitwidth():
    bitwidth_axis = AxisDescriptor(
        kind=AxisKind.BITWIDTH, key=BITWIDTH_AXIS_KEY, label="32/64"
    )
    rv64 = _make_rv(variant_idx=0, arch="x64")
    rv32 = _make_rv(variant_idx=1, arch="x86")
    assert extract_axis_value(bitwidth_axis, rv64) == "64"
    assert extract_axis_value(bitwidth_axis, rv32) == "32"


def test_extract_axis_value_extra_meta():
    axis = build_extra_meta_axis("sanitizer")
    rv_with = _make_rv(variant_idx=0, extra={"sanitizer": "address"})
    rv_without = _make_rv(variant_idx=1, extra={})
    assert extract_axis_value(axis, rv_with) == "address"
    assert extract_axis_value(axis, rv_without) is None


def test_group_variants_no_grouping_returns_flat_sorted():
    rvs = [
        _make_rv(variant_idx=0, opt="O2"),
        _make_rv(variant_idx=1, opt="O0"),
        _make_rv(variant_idx=2, opt="O1"),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}

    axes = build_canonical_axes()
    config = OrderConfig(ordered_axes=axes, grouping_axes=frozenset())
    grouped = group_variants(variants, rendered_by_variant, config)

    # All flat VariantNodes, sorted by axes (last axis = opt natsort).
    assert all(isinstance(c, VariantNode) for c in grouped)
    assert [c.variant_idx for c in grouped] == [1, 2, 0]  # O0, O1, O2


def test_group_variants_one_grouping_axis_nests_by_arch():
    rvs = [
        _make_rv(variant_idx=0, arch="x64"),
        _make_rv(variant_idx=1, arch="arm64"),
        _make_rv(variant_idx=2, arch="x64"),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}

    axes = build_canonical_axes()
    arch_axis = next(a for a in axes if a.kind is AxisKind.POSITIONAL and a.key == ARCH_PREFIX)
    config = OrderConfig(
        ordered_axes=axes, grouping_axes=frozenset({arch_axis})
    )
    grouped = group_variants(variants, rendered_by_variant, config)

    assert all(isinstance(c, VariantGroupNode) for c in grouped)
    # natsort: "arm64" sorts before "x64"
    assert [c.axis_value for c in grouped] == ["arm64", "x64"]
    # arm64 bucket: one variant (idx=1)
    arm_group = grouped[0]
    assert arm_group.axis == arch_axis
    assert len(arm_group.children) == 1
    assert arm_group.children[0].variant_idx == 1
    # x64 bucket: two variants (idx=0, 2 -- both x64)
    x64_group = grouped[1]
    assert len(x64_group.children) == 2


def test_group_variants_missing_value_sinks_to_question_bucket():
    """An EXTRA_META value missing on one variant -> ``"?"`` bucket
    placed after the populated ones."""
    rvs = [
        _make_rv(variant_idx=0, extra={"san": "addr"}),
        _make_rv(variant_idx=1, extra={}),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}

    san_axis = build_extra_meta_axis("san")
    config = OrderConfig(
        ordered_axes=(san_axis,), grouping_axes=frozenset({san_axis})
    )
    grouped = group_variants(variants, rendered_by_variant, config)

    assert len(grouped) == 2
    assert grouped[0].axis_value == "addr"
    assert grouped[1].axis_value == "?"


# ---------------------------------------------------------------------------
# Modal dialog tests (Textual Pilot).
# ---------------------------------------------------------------------------


def test_order_dialog_ctrl_s_yields_order_accepted():
    """Accept via ``ctrl+s`` dismisses with :class:`OrderAccepted`."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                dialog = OrderDialog(candidate_axes=build_canonical_axes())
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.press("ctrl+s")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], OrderAccepted)
                assert isinstance(results[0].config, OrderConfig)

    asyncio.run(runner())


def test_order_dialog_escape_yields_order_cancelled():
    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                dialog = OrderDialog(candidate_axes=build_canonical_axes())
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], OrderCancelled)

    asyncio.run(runner())


def test_order_dialog_accept_button_click():
    """Clicking the ``[Accept]`` button dismisses with
    :class:`OrderAccepted` (cluster #11: Enter is shadowed by the
    SelectionList check toggle, so we route accept off the button)."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                results: list = []
                dialog = OrderDialog(candidate_axes=build_canonical_axes())
                app.push_screen(dialog, results.append)
                await pilot.pause()
                await pilot.click("#accept")
                await pilot.pause()
                assert len(results) == 1
                assert isinstance(results[0], OrderAccepted)

    asyncio.run(runner())


def test_order_binding_opens_dialog():
    """Pressing ``o`` pushes :class:`OrderDialog` onto the screen stack."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            factory = _make_factory_with_rvs(name="main", rvs=[])
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                await pilot.press("o")
                await pilot.pause()
                # The active screen should be the OrderDialog.
                assert isinstance(app.screen, OrderDialog)
                await pilot.press("escape")
                await pilot.pause()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Expand-state preservation (capture-on-rebuild).
# ---------------------------------------------------------------------------


def test_expand_state_preserved_across_regroup():
    """Open a variant row, accept a new OrderConfig that flips
    grouping_axes on -- the previously-open variant survives the
    regroup and is still expanded under its new group ancestor."""

    async def runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "tui.log"
            rvs = [
                _make_rv(variant_idx=0, arch="x64", compiler="clang"),
                _make_rv(variant_idx=1, arch="arm64", compiler="gcc"),
            ]
            factory = _make_factory_with_rvs(name="main", rvs=rvs)
            app = InspectorApp(factory=factory, log_path=log_path)
            async with app.run_test() as pilot:
                tree, fn_tree_node = _root_tree_node(app)
                fn_model: FunctionNode = fn_tree_node.data
                # Expand function -> 2 VariantNodes mounted directly.
                fn_tree_node.expand()
                await pilot.pause()
                assert len(fn_tree_node.children) == 2

                # Expand the second variant (arm64 / gcc by default-
                # natsort order before any grouping is applied -- list
                # iteration is in backend insertion order, idx 0 then 1).
                target_variant_tree_node = fn_tree_node.children[1]
                target_variant_model = target_variant_tree_node.data
                assert isinstance(target_variant_model, VariantNode)
                target_variant_tree_node.expand()
                await pilot.pause()
                assert target_variant_tree_node.is_expanded

                # Apply a config that groups by arch -> rebuild fires.
                axes = build_canonical_axes()
                arch_axis = next(
                    a for a in axes
                    if a.kind is AxisKind.POSITIONAL and a.key == ARCH_PREFIX
                )
                new_config = OrderConfig(
                    ordered_axes=axes, grouping_axes=frozenset({arch_axis})
                )
                # Drive the same code path the dialog dismiss callback uses.
                app._on_order_dialog_dismissed(OrderAccepted(config=new_config))
                await pilot.pause()

                # The function node now has two VariantGroupNode
                # children (one per arch bucket); the previously-
                # opened variant must surface under its bucket and be
                # expanded again.
                group_children = list(fn_tree_node.children)
                assert all(
                    isinstance(child.data, VariantGroupNode)
                    for child in group_children
                )
                # Find the variant tree node carrying our identity.
                found_expanded = False
                for group_tree_node in group_children:
                    for variant_tree_node in group_tree_node.children:
                        v_model = variant_tree_node.data
                        if (
                            isinstance(v_model, VariantNode)
                            and v_model.variant_idx == target_variant_model.variant_idx
                        ):
                            if variant_tree_node.is_expanded:
                                found_expanded = True
                assert found_expanded, (
                    "previously-expanded variant did not survive the regroup"
                )

    asyncio.run(runner())
