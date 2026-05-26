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


def test_group_variants_arch_collapses_to_family_when_bitwidth_co_groups():
    """When BOTH arch + bitwidth are grouping axes, the arch level
    surfaces the family-display name (``arm`` / ``mips`` / ``x86``)
    instead of the raw ``arm32`` / ``arm64`` / ... values; the
    bitwidth level still shows ``32`` / ``64`` unchanged."""
    rvs = [
        _make_rv(variant_idx=0, arch="arm32"),
        _make_rv(variant_idx=1, arch="arm64"),
        _make_rv(variant_idx=2, arch="mips32"),
        _make_rv(variant_idx=3, arch="mips64"),
        _make_rv(variant_idx=4, arch="x86"),
        _make_rv(variant_idx=5, arch="x64"),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}

    axes = build_canonical_axes()
    arch_axis = next(
        a for a in axes if a.kind is AxisKind.POSITIONAL and a.key == ARCH_PREFIX
    )
    bitwidth_axis = next(
        a for a in axes
        if a.kind is AxisKind.BITWIDTH and a.key == BITWIDTH_AXIS_KEY
    )
    config = OrderConfig(
        ordered_axes=axes,
        grouping_axes=frozenset({arch_axis, bitwidth_axis}),
    )
    grouped = group_variants(variants, rendered_by_variant, config)

    # Top level: 3 family buckets (alphabetical natsort: arm, mips, x86).
    assert all(isinstance(c, VariantGroupNode) for c in grouped)
    assert [c.axis_value for c in grouped] == ["arm", "mips", "x86"]
    # Each family has two bitwidth sub-buckets (32, 64) carrying one
    # variant each.
    for family_group, expected_bitwidth_pairs in zip(
        grouped,
        [
            [("32", 0), ("64", 1)],   # arm32 -> 0, arm64 -> 1
            [("32", 2), ("64", 3)],   # mips32 -> 2, mips64 -> 3
            [("32", 4), ("64", 5)],   # x86 -> 4, x64 -> 5
        ],
    ):
        assert family_group.axis == arch_axis
        bw_children = family_group.children
        assert all(isinstance(c, VariantGroupNode) for c in bw_children)
        assert [c.axis_value for c in bw_children] == [
            bw for bw, _ in expected_bitwidth_pairs
        ]
        for bw_group, (_, expected_variant_idx) in zip(
            bw_children, expected_bitwidth_pairs
        ):
            assert bw_group.axis == bitwidth_axis
            assert len(bw_group.children) == 1
            assert bw_group.children[0].variant_idx == expected_variant_idx


def test_group_variants_arch_keeps_raw_value_when_bitwidth_not_grouping():
    """arch-only grouping (no bitwidth co-grouping) keeps the raw
    bitness-bearing arch value -- the family collapse is gated on
    bitwidth being a sibling grouping axis."""
    rvs = [
        _make_rv(variant_idx=0, arch="arm32"),
        _make_rv(variant_idx=1, arch="arm64"),
    ]
    variants = [_make_variant_node(rv) for rv in rvs]
    rendered_by_variant = {rv.variant_idx: rv for rv in rvs}

    axes = build_canonical_axes()
    arch_axis = next(
        a for a in axes if a.kind is AxisKind.POSITIONAL and a.key == ARCH_PREFIX
    )
    config = OrderConfig(
        ordered_axes=axes, grouping_axes=frozenset({arch_axis})
    )
    grouped = group_variants(variants, rendered_by_variant, config)

    assert [c.axis_value for c in grouped] == ["arm32", "arm64"]


def test_collect_suppressed_axes_walks_variant_group_node_ancestors():
    """Positional :class:`VariantGroupNode` ancestors contribute their
    axis ``key`` to the suppression set; BITWIDTH + EXTRA_META
    ancestors do NOT (those axes are not in the canonical positional
    label so dropping a column wouldn't make sense)."""
    from tokenizer.inspector._app._application import _collect_suppressed_axes

    axes = build_canonical_axes()
    arch_axis = next(a for a in axes if a.kind is AxisKind.POSITIONAL and a.key == ARCH_PREFIX)
    cver_axis = next(a for a in axes if a.key == CVER_PREFIX)
    opt_axis = next(a for a in axes if a.key == OPT_PREFIX)
    bitwidth_axis = next(a for a in axes if a.kind is AxisKind.BITWIDTH)
    extra_axis = build_extra_meta_axis("sanitizer")

    rv = _make_rv(variant_idx=0)

    def _mock_tree_node(data, parent):
        node = MagicMock()
        node.data = data
        node.parent = parent
        return node

    # Synthetic chain: root (None data) <- function (None data) <- cver group
    # <- opt group <- the freshly-expanded opt-group node (still the
    # caller). The walk reads ``cursor.data`` + ``cursor.parent`` only.
    root = _mock_tree_node(None, None)
    fn_node = _mock_tree_node(None, root)
    cver_group = VariantGroupNode(
        axis=cver_axis, axis_value="5.0", children=[], rendered_by_variant={0: rv}
    )
    cver_tree_node = _mock_tree_node(cver_group, fn_node)
    opt_group = VariantGroupNode(
        axis=opt_axis, axis_value="O0", children=[], rendered_by_variant={0: rv}
    )
    opt_tree_node = _mock_tree_node(opt_group, cver_tree_node)

    # Caller passes the EXPANDED node (the opt group). Walk includes
    # the expanded node itself + every ancestor.
    suppressed = _collect_suppressed_axes(opt_tree_node)
    assert suppressed == frozenset({CVER_PREFIX, OPT_PREFIX})

    # Un-grouped path: expanding the function itself (no group ancestors)
    # yields an empty frozenset -- legacy behavior preserved.
    assert _collect_suppressed_axes(fn_node) == frozenset()

    # BITWIDTH + EXTRA_META group ancestors do NOT contribute to the
    # positional-axis suppression set.
    bw_group = VariantGroupNode(
        axis=bitwidth_axis, axis_value="32", children=[], rendered_by_variant={0: rv}
    )
    bw_tree_node = _mock_tree_node(bw_group, fn_node)
    extra_group = VariantGroupNode(
        axis=extra_axis, axis_value="address", children=[], rendered_by_variant={0: rv}
    )
    extra_tree_node = _mock_tree_node(extra_group, bw_tree_node)
    assert _collect_suppressed_axes(extra_tree_node) == frozenset()

    # Mixed chain: ARCH positional under BITWIDTH derived -- only ARCH
    # is suppressed (BITWIDTH does not occupy a positional column).
    arch_group = VariantGroupNode(
        axis=arch_axis, axis_value="arm32", children=[], rendered_by_variant={0: rv}
    )
    arch_tree_node = _mock_tree_node(arch_group, bw_tree_node)
    assert _collect_suppressed_axes(arch_tree_node) == frozenset({ARCH_PREFIX})


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
# apply_grouping: structural-gate regression tests.
#
# The dispatcher hands every model.expand() result through
# :func:`_order_hooks.apply_grouping`. The gate is structural ("are these
# children a homogeneous VariantNode sibling set?"), not a model-type
# allowlist, so InlineCallNode.expand's no-pin / missing-variant fallback
# path -- which returns a list of VariantNodes -- gets the same sort +
# group treatment as FunctionNode / ShowAllVariantsNode.
# ---------------------------------------------------------------------------


def _make_variant_node_with_backend(
    rv: RenderedVariant, backend: MagicMock
) -> VariantNode:
    """Like :func:`_make_variant_node` but threads a SHARED backend so the
    ``backend.variants()`` lookup inside :func:`apply_grouping` sees the
    full sibling set (not just this one variant)."""
    factory = MagicMock(spec=BackendFactory)
    return VariantNode(
        factory=factory,
        backend=backend,
        variant_idx=rv.variant_idx,
        label_axes=rv.label_axes,
    )


def _make_inline_call_fallback_variant_nodes(
    rvs: list[RenderedVariant],
) -> list[VariantNode]:
    """Build the kind of sibling list an :class:`InlineCallNode`'s
    fallback path returns: every callee variant as a flat
    :class:`VariantNode`, all sharing the callee's :class:`RenderBackend`
    (the dual-session-aware backend ``InlineCallNode.expand`` constructs
    via the callee :class:`FunctionNode`)."""
    callee_backend = MagicMock(spec=RenderBackend)
    callee_backend.variants.return_value = rvs
    return [_make_variant_node_with_backend(rv, callee_backend) for rv in rvs]


def _stub_app(order_config: OrderConfig | None) -> MagicMock:
    """Stub the only :class:`InspectorApp` surface :func:`apply_grouping`
    reads: the active :class:`OrderConfig` (or ``None``)."""
    app = MagicMock(spec=InspectorApp)
    app._order_config = order_config
    return app


def test_apply_grouping_natsorts_inline_call_fallback_variants():
    """Regression: under an :class:`InlineCallNode` whose pin misses
    (the callee dropped that variant), :meth:`InlineCallNode.expand`
    surfaces every callee variant as a flat :class:`VariantNode`. The
    dispatcher MUST still route that list through
    :func:`apply_grouping` so the siblings are natsort-ordered even
    when no :class:`OrderConfig` is active -- previously they leaked
    out in dataset order because the gate was a model-type allowlist
    (FunctionNode / ShowAllVariantsNode only).
    """
    from tokenizer.inspector._app._order_hooks import apply_grouping
    from tokenizer.inspector._tree_model import InlineCallNode

    # Dataset-order RVs that are NOT in natsort order (mixed compilers +
    # cvers -- natsort should put clang < gcc, and within compiler sort
    # by cver: 3.5 < 7 -- matching the user's repro).
    rvs = [
        _make_rv(variant_idx=0, compiler="gcc", cver="5", opt="O0"),
        _make_rv(variant_idx=1, compiler="clang", cver="7", opt="O3"),
        _make_rv(variant_idx=2, compiler="clang", cver="3.5", opt="O2"),
        _make_rv(variant_idx=3, compiler="gcc", cver="5", opt="Os"),
        _make_rv(variant_idx=4, compiler="clang", cver="3.5", opt="Os"),
    ]
    children = _make_inline_call_fallback_variant_nodes(rvs)

    # Stand-in model -- :func:`apply_grouping` no longer keys off the
    # model-instance type, so an :class:`InlineCallNode` works the same
    # as a :class:`FunctionNode`.
    model = MagicMock(spec=InlineCallNode)
    app = _stub_app(order_config=None)
    out = apply_grouping(app, model, children)

    # All flat VariantNodes (no grouping), natsort-ordered by the full
    # canonical axis chain. Canonical chain: ARCH, COMP, CVER, OPT,
    # BITWIDTH -- arch is uniform so comp tie-breaks first, then cver,
    # then opt. Expected natsort: clang 3.5 O2, clang 3.5 Os, clang 7
    # O3, gcc 5 O0, gcc 5 Os -> idx [2, 4, 1, 0, 3].
    assert all(isinstance(c, VariantNode) for c in out)
    assert [c.variant_idx for c in out] == [2, 4, 1, 0, 3]


def test_apply_grouping_groups_inline_call_fallback_variants_by_compiler():
    """Same scenario as the natsort regression, but with an
    :class:`OrderConfig` that groups by ``compiler``: the InlineCallNode
    fallback's flat :class:`VariantNode` list must surface as
    :class:`VariantGroupNode` siblings (3 clang + 2 gcc -> 2 groups)."""
    from tokenizer.inspector._app._order import AxisKind
    from tokenizer.inspector._app._order_hooks import apply_grouping
    from tokenizer.inspector._tree_model import InlineCallNode

    rvs = [
        _make_rv(variant_idx=0, compiler="gcc", cver="5", opt="O0"),
        _make_rv(variant_idx=1, compiler="clang", cver="7", opt="O3"),
        _make_rv(variant_idx=2, compiler="clang", cver="3.5", opt="O2"),
        _make_rv(variant_idx=3, compiler="gcc", cver="5", opt="Os"),
        _make_rv(variant_idx=4, compiler="clang", cver="3.5", opt="Os"),
    ]
    children = _make_inline_call_fallback_variant_nodes(rvs)

    axes = build_canonical_axes()
    comp_axis = next(
        a for a in axes
        if a.kind is AxisKind.POSITIONAL and a.key == COMP_PREFIX
    )
    config = OrderConfig(
        ordered_axes=axes, grouping_axes=frozenset({comp_axis})
    )
    model = MagicMock(spec=InlineCallNode)
    app = _stub_app(order_config=config)
    out = apply_grouping(app, model, children)

    # Two top-level groups (clang + gcc) -- the model-type gate fix
    # routes this list through :func:`group_variants` just like a
    # :class:`FunctionNode` sibling set.
    assert all(isinstance(c, VariantGroupNode) for c in out)
    assert [g.axis_value for g in out] == ["clang", "gcc"]
    clang_group, gcc_group = out
    assert {v.variant_idx for v in clang_group.children} == {1, 2, 4}
    assert {v.variant_idx for v in gcc_group.children} == {0, 3}


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
                # opened variant must surface under its bucket. With
                # the single-child-chain collapse policy, each group's
                # lone variant is folded into the group's direct
                # children -- so the group hosting the previously-
                # opened variant must itself be expanded (showing the
                # collapsed-past variant's blocks/leaves) rather than
                # containing an inner expanded variant row.
                group_children = list(fn_tree_node.children)
                assert all(
                    isinstance(child.data, VariantGroupNode)
                    for child in group_children
                )
                # Locate the group whose model wraps our target
                # variant identity; it must be expanded so the user
                # sees the same content they had before the regroup.
                found_expanded_group = False
                for group_tree_node in group_children:
                    group_model = group_tree_node.data
                    contains_target = any(
                        isinstance(c, VariantNode)
                        and c.variant_idx == target_variant_model.variant_idx
                        for c in group_model.children
                    )
                    if contains_target and group_tree_node.is_expanded:
                        found_expanded_group = True
                assert found_expanded_group, (
                    "group hosting the previously-expanded variant "
                    "did not auto-expand across the regroup"
                )

    asyncio.run(runner())
