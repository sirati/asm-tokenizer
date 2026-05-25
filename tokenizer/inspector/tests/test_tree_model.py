"""Unit tests for the inspector tree-model ``expand`` contracts.

Covers the post-Wave-5 ``BackendFactory`` + ``RenderBackend`` contract:
the :meth:`FunctionNode.expand` factory dispatch, the
:class:`RenderedVariant` -> :class:`VariantNode` mapping, idempotent
re-expansion, :class:`InlineCallNode.can_expand` gating per
:class:`CallTargetType`, :class:`AsmLeaf` terminal contract, and the
central ``_on_node_expanded`` dispatcher's failure-path
``is_failed=True`` stamping (the latter ``importorskip`` 'textual').
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
    BlockKind,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    BlockNode,
    FunctionNode,
    InlineCallNode,
    VariantNode,
)
from tokenizer.variant_info import VariantIdentity
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers -- typed RenderBackend / BackendFactory mocks via MagicMock(spec=...).
# ---------------------------------------------------------------------------


def _make_label_axes() -> dict[str, str | None]:
    """One canonical-order label_axes Mapping (MappingProxy-wrapped)."""
    return types.MappingProxyType(
        {
            ARCH_PREFIX: "x86",
            COMP_PREFIX: "clang",
            CVER_PREFIX: "8.0",
            OPT_PREFIX: "O3",
        }
    )


def _make_variant_identity(variant_idx: int) -> VariantIdentity:
    return VariantIdentity(
        arch="x86", compiler="clang", compiler_version="8.0", opt="O3",
        pkg="", variant_id=variant_idx,
    )


def _make_rendered_variant(variant_idx: int) -> RenderedVariant:
    return RenderedVariant(
        variant_idx=variant_idx,
        label_axes=_make_label_axes(),
        extra_metadata=types.MappingProxyType({}),
        variant_identity=_make_variant_identity(variant_idx),
    )


def _make_handle(idx: int = 0, name: str = "main") -> FunctionHandle:
    return FunctionHandle(arm=SectionKind.MATCHED, idx=idx, name=name)


def _make_backend(n_variants: int, n_blocks: int = 0) -> MagicMock:
    """A ``RenderBackend``-spec'd mock seeded with ``n_variants`` variants
    and ``n_blocks`` blocks per variant."""
    backend = MagicMock(spec=RenderBackend)
    backend.handle = _make_handle()
    backend.closed = False
    backend.variants.return_value = [
        _make_rendered_variant(i) for i in range(n_variants)
    ]
    backend.blocks.return_value = [
        RenderedBlock(
            kind=BlockKind.BODY, block_idx=i, preview=f"preview {i}",
        )
        for i in range(n_blocks)
    ]
    backend.render_block.return_value = ()
    return backend


def _make_factory(backend: MagicMock | None = None) -> MagicMock:
    """A ``BackendFactory``-spec'd mock whose ``.make`` returns the
    supplied backend (or a fresh empty one)."""
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [_make_handle()]
    factory.make.return_value = backend if backend is not None else _make_backend(0)
    return factory


# ---------------------------------------------------------------------------
# FunctionNode.expand -- the factory dispatch + idempotence
# ---------------------------------------------------------------------------


def test_function_node_expand_calls_factory_make_with_handle():
    """``FunctionNode.expand`` MUST construct exactly one backend per
    open via :meth:`BackendFactory.make`, passing the typed handle."""
    handle = _make_handle(idx=3, name="foo")
    backend = _make_backend(n_variants=2)
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    factory.make.return_value = backend

    node = FunctionNode(factory=factory, handle=handle)
    node.expand()

    assert factory.make.call_count == 1
    factory.make.assert_called_with(handle)


def test_function_node_expand_returns_one_variant_node_per_variant():
    """Every :class:`RenderedVariant` the backend reports maps to
    exactly one :class:`VariantNode` child."""
    backend = _make_backend(n_variants=3)
    handle = _make_handle()
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    factory.make.return_value = backend

    node = FunctionNode(factory=factory, handle=handle)
    children = node.expand()

    assert len(children) == 3
    assert all(isinstance(child, VariantNode) for child in children)
    # variant_idx is threaded from the RenderedVariant onto the model
    # node so descendants can re-key into backend.blocks/render_block.
    assert [c.variant_idx for c in children] == [0, 1, 2]


def test_function_node_expand_is_idempotent():
    """Re-expanding the same :class:`FunctionNode` is observably the
    same: same number of children, factory.make called per expand
    (the cached backend ref is invalidated by collapse + re-expand).
    """
    backend = _make_backend(n_variants=2)
    handle = _make_handle()
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    factory.make.return_value = backend

    node = FunctionNode(factory=factory, handle=handle)
    first = node.expand()
    second = node.expand()

    assert len(first) == 2
    assert len(second) == 2
    assert factory.make.call_count == 2


def test_function_node_expand_closes_prior_backend_on_reexpand():
    """Re-expansion MUST close the previously cached backend before
    constructing the new one; otherwise every collapse + re-expand
    (e.g. dialog accept) leaks one backend instance.
    """
    first_backend = _make_backend(n_variants=1)
    second_backend = _make_backend(n_variants=1)
    handle = _make_handle()
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [handle]
    factory.make.side_effect = [first_backend, second_backend]

    node = FunctionNode(factory=factory, handle=handle)
    node.expand()
    assert first_backend.close.call_count == 0
    node.expand()
    assert first_backend.close.call_count == 1
    assert second_backend.close.call_count == 0


# ---------------------------------------------------------------------------
# VariantNode.expand -- delegates to backend.blocks
# ---------------------------------------------------------------------------


def test_variant_node_expand_calls_backend_blocks():
    """Variant expansion reads :meth:`RenderBackend.blocks(v)` and
    surfaces one :class:`BlockNode` per :class:`RenderedBlock`."""
    backend = _make_backend(n_variants=1, n_blocks=4)
    factory = _make_factory(backend)
    node = VariantNode(
        factory=factory,
        backend=backend,
        variant_idx=0,
        label_axes=_make_label_axes(),
    )

    children = node.expand()

    backend.blocks.assert_called_once_with(0)
    assert len(children) == 4
    assert all(isinstance(c, BlockNode) for c in children)
    assert [c.block_idx for c in children] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# BlockNode.expand -- translates LineItems into model nodes
# ---------------------------------------------------------------------------


def test_block_node_expand_translates_asm_line_to_leaf():
    """A single :class:`AsmLine` LineItem becomes one :class:`AsmLeaf`."""
    from tokenizer.inspector._render._protocol import AsmLine

    backend = _make_backend(n_variants=1)
    backend.render_block.return_value = (AsmLine(text="nop"),)
    factory = _make_factory(backend)
    node = BlockNode(
        factory=factory,
        backend=backend,
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=0,
        preview="preview",
    )

    children = node.expand()

    backend.render_block.assert_called_once_with(0, BlockKind.BODY, 0)
    assert len(children) == 1
    assert isinstance(children[0], AsmLeaf)
    assert children[0].text == "nop"


def test_block_node_expand_only_produces_asmleaf_children_post_r2():
    """Post-R2 contract (plan W3-2 W4-amended cluster #3): BlockNode
    children are :class:`AsmLeaf` only -- inline call sites, jump
    targets, and number-precision sidecars no longer surface as
    sibling top-level rows. The openables ride on
    :attr:`AsmLine.openables` and surface as children of the leaf at
    leaf-expand time."""
    from tokenizer.inspector._render._protocol import (
        AsmLine,
        InlineCallEntry,
        InlineJumpEntry,
    )

    call_entry = InlineCallEntry(
        kind=CallTargetType.LOCAL,
        counter_id=2,
        callee_name="callee",
        callee_section_pointer=SectionPointerSpec(
            arm=SectionKind.MATCHED, idx=7
        ),
        variant_idx=0,
        provider=None,
    )
    jump_entry = InlineJumpEntry(target_block_idx=3)
    backend = _make_backend(n_variants=1)
    backend.render_block.return_value = (
        AsmLine(text="call foo", openables=(call_entry,)),
        AsmLine(text="jmp .L3", openables=(jump_entry,)),
        AsmLine(text="nop"),
    )
    factory = _make_factory(backend)
    node = BlockNode(
        factory=factory,
        backend=backend,
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=0,
        preview="preview",
    )

    children = node.expand()

    # All three rows are AsmLeaf -- nothing else surfaces at the top
    # level under a BlockNode now.
    assert len(children) == 3
    assert all(isinstance(c, AsmLeaf) for c in children)

    # The call-carrying leaf is expandable; expanding it produces the
    # InlineCallNode (the 1-arm dispatch).
    call_leaf = children[0]
    assert call_leaf.can_expand is True
    grandchildren = call_leaf.expand()
    assert len(grandchildren) == 1
    call = grandchildren[0]
    assert isinstance(call, InlineCallNode)
    assert call.can_expand is True
    assert call.callee_handle is not None
    assert call.callee_handle.arm is SectionKind.MATCHED
    assert call.callee_handle.idx == 7
    assert call.callee_handle.name == "callee"

    # The plain ``nop`` row is terminal -- no openables, no expand.
    assert children[2].can_expand is False


# ---------------------------------------------------------------------------
# InlineCallNode.can_expand -- LOCAL + handle is the ONE expandable shape.
# ---------------------------------------------------------------------------


def _make_inline_call(
    kind: CallTargetType, handle: FunctionHandle | None
) -> InlineCallNode:
    return InlineCallNode(
        factory=_make_factory(),
        kind=kind,
        counter_id=0,
        callee_name="callee",
        callee_handle=handle,
        variant_idx=0,
        provider=None,
    )


def test_inline_call_node_can_expand_for_local_with_handle():
    node = _make_inline_call(
        CallTargetType.LOCAL, _make_handle(idx=1, name="callee")
    )
    assert node.can_expand is True


def test_inline_call_node_cannot_expand_for_local_without_handle():
    node = _make_inline_call(CallTargetType.LOCAL, None)
    assert node.can_expand is False


@pytest.mark.parametrize("kind", [CallTargetType.PLT, CallTargetType.EXTERN])
def test_inline_call_node_cannot_expand_for_plt_or_extern(kind):
    """PLT / EXTERN have no inlineable body regardless of pointer
    presence."""
    node = _make_inline_call(kind, _make_handle(idx=1, name="callee"))
    assert node.can_expand is False


# ---------------------------------------------------------------------------
# AsmLeaf -- terminal, gating on ``can_expand``.
# ---------------------------------------------------------------------------


def test_asm_leaf_cannot_expand():
    assert AsmLeaf(text="nop").can_expand is False


def test_asm_leaf_expand_raises():
    """Calling :meth:`AsmLeaf.expand` is a contract violation; the
    docstring tells callers to gate on ``can_expand``."""
    with pytest.raises(NotImplementedError):
        AsmLeaf(text="nop").expand()


# ---------------------------------------------------------------------------
# Expand-error path through the central app dispatcher.
# Importing ``_app.py`` pulls in ``textual``; skip the test cleanly when
# the dep isn't available (default ``nix develop`` shell).
# ---------------------------------------------------------------------------


def test_expand_error_path_marks_is_failed_via_app_dispatcher():
    """The inspector dispatcher is the ONE try/except wrapping model
    ``expand()``. On exception it stamps ``is_failed=True`` on the
    model + attaches a dim-red error leaf + refreshes the prefix glyph.
    """
    pytest.importorskip("textual")

    from pathlib import Path
    import tempfile

    from tokenizer.inspector._app import InspectorApp

    # Force the FunctionNode's expand to raise; the dispatcher must
    # catch and stamp ``is_failed`` without bubbling.
    factory = _make_factory(_make_backend(0))
    handle = _make_handle()
    fn = FunctionNode(factory=factory, handle=handle)
    boom = RuntimeError("synthetic expand failure")

    def _raise() -> list:  # match signature: arg-less
        raise boom

    fn.expand = _raise  # type: ignore[method-assign]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "inspector.log"
        app = InspectorApp(factory=factory, log_path=log_path)

        # Hand-roll an event with the surface the dispatcher touches:
        # ``stop()``, ``.node.data``, ``.node.remove_children()``,
        # ``.node.add_leaf()``, ``.node.refresh()``.
        tree_node = MagicMock(name="TreeNode")
        tree_node.data = fn
        event = MagicMock(name="NodeExpanded")
        event.node = tree_node

        app._on_node_expanded(event)

    assert fn.is_failed is True
    # Tree node should have had its error-leaf attached + refresh called.
    tree_node.add_leaf.assert_called_once()
    tree_node.refresh.assert_called()


# ---------------------------------------------------------------------------
# Sibling-set column alignment: the dispatcher stamps ``aligned_label``
# on the variants returned by FunctionNode.expand so axis values align
# in columns across siblings.
# ---------------------------------------------------------------------------


def test_function_expand_via_dispatcher_stamps_aligned_variant_labels():
    """End-to-end pilot: a FunctionNode with multi-width axis values
    across siblings flows through the central expand dispatcher and the
    resulting :class:`VariantNode` children carry pre-aligned labels
    with per-axis columns padded to the sibling-set max."""
    pytest.importorskip("textual")

    import types as _types
    from pathlib import Path
    import tempfile

    from tokenizer.inspector._app import InspectorApp

    # Build two variants with different per-axis widths: arch col widens
    # to 5 (arm32 vs x86), comp col widens to 5 (gcc vs clang).
    def _axes(arch: str, comp: str, cver: str, opt: str):
        return _types.MappingProxyType(
            {
                ARCH_PREFIX: arch,
                COMP_PREFIX: comp,
                CVER_PREFIX: cver,
                OPT_PREFIX: opt,
            }
        )

    rendered_variants = [
        RenderedVariant(
            variant_idx=0,
            label_axes=_axes("arm32", "gcc", "5", "O0"),
            extra_metadata=_types.MappingProxyType({}),
            variant_identity=VariantIdentity(
                arch="arm32", compiler="gcc", compiler_version="5",
                opt="O0", pkg="", variant_id=0,
            ),
        ),
        RenderedVariant(
            variant_idx=1,
            label_axes=_axes("x86", "clang", "7", "O3"),
            extra_metadata=_types.MappingProxyType({}),
            variant_identity=VariantIdentity(
                arch="x86", compiler="clang", compiler_version="7",
                opt="O3", pkg="", variant_id=1,
            ),
        ),
    ]
    backend = MagicMock(spec=RenderBackend)
    backend.handle = _make_handle()
    backend.closed = False
    backend.variants.return_value = rendered_variants
    factory = _make_factory(backend)
    handle = _make_handle()
    fn = FunctionNode(factory=factory, handle=handle)

    captured: list = []

    tree_node = MagicMock(name="TreeNode")
    tree_node.data = fn
    tree_node.add = lambda _label, data, allow_expand: captured.append(data)
    event = MagicMock(name="NodeExpanded")
    event.node = tree_node

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "inspector.log"
        app = InspectorApp(factory=factory, log_path=log_path)
        app._on_node_expanded(event)

    assert len(captured) == 2
    assert all(isinstance(v, VariantNode) for v in captured)
    # Aligned labels: arch col = 5, comp col = 5, cver col = 2; -opt
    # trailing column unpadded. See aligned_variant_labels unit tests.
    assert captured[0].aligned_label == "arm32 gcc   v5 -O0"
    assert captured[1].aligned_label == "x86   clang v7 -O3"
