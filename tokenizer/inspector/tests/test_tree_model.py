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
from typing import Mapping
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
    ShowAllVariantsNode,
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


# --- Distinct-axes variants for caller-identity-fallback testing. -----
# The caller-fallback path matches the callee's variants on the
# canonical-4 axes ``(arch, compiler, compiler_version, opt)``; the
# uniform-axes :func:`_make_rendered_variant` above would collide every
# callee variant on the same match key, so the tests below build
# variants whose axes actually differ across the list.

_AXIS_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    ("x86", "clang", "8.0", "O3"),
    ("arm32", "clang", "5.0", "O0"),
    ("x86", "gcc", "9.0", "O2"),
)


def _make_axis_variant_identity(variant_idx: int) -> VariantIdentity:
    arch, compiler, cver, opt = _AXIS_VARIANTS[variant_idx]
    return VariantIdentity(
        arch=arch, compiler=compiler, compiler_version=cver, opt=opt,
        pkg="", variant_id=variant_idx,
    )


def _make_axis_label_axes(variant_idx: int) -> Mapping[str, str | None]:
    arch, compiler, cver, opt = _AXIS_VARIANTS[variant_idx]
    return types.MappingProxyType(
        {
            ARCH_PREFIX: arch, COMP_PREFIX: compiler,
            CVER_PREFIX: cver, OPT_PREFIX: opt,
        }
    )


def _make_axis_rendered_variant(variant_idx: int) -> RenderedVariant:
    return RenderedVariant(
        variant_idx=variant_idx,
        label_axes=_make_axis_label_axes(variant_idx),
        extra_metadata=types.MappingProxyType({}),
        variant_identity=_make_axis_variant_identity(variant_idx),
    )


def _make_distinct_axes_backend(
    n_variants: int, n_blocks: int = 0,
) -> MagicMock:
    """``RenderBackend`` mock whose variants carry DISTINCT canonical-4
    axes (one per :data:`_AXIS_VARIANTS` entry) so caller-identity
    matching can pick out a single variant deterministically."""
    assert n_variants <= len(_AXIS_VARIANTS), (
        f"_AXIS_VARIANTS only has {len(_AXIS_VARIANTS)} entries"
    )
    backend = MagicMock(spec=RenderBackend)
    backend.handle = _make_handle()
    backend.closed = False
    backend.variants.return_value = [
        _make_axis_rendered_variant(i) for i in range(n_variants)
    ]
    backend.blocks.return_value = [
        RenderedBlock(
            kind=BlockKind.BODY, block_idx=i, preview=f"preview {i}",
        )
        for i in range(n_blocks)
    ]
    backend.render_block.return_value = ()
    return backend


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


def test_inline_call_node_can_expand_for_plt_with_handle():
    """PLT call_targets resolve to a sectioned PLT thunk (the writer
    treats LOCAL and PLT identically) -- so a PLT entry with a
    resolved ``callee_handle`` is expandable into the small jump-thunk
    body that calls the resolved extern."""
    node = _make_inline_call(
        CallTargetType.PLT, _make_handle(idx=1, name="callee")
    )
    assert node.can_expand is True


def test_inline_call_node_cannot_expand_for_plt_without_handle():
    """A PLT call whose thunk section was not resolvable (cross-arm
    miss / dropped by pass-1) carries ``callee_handle=None`` and is
    therefore not expandable -- the gate is "has a handle", not "is
    LOCAL"."""
    node = _make_inline_call(CallTargetType.PLT, None)
    assert node.can_expand is False


def test_inline_call_node_cannot_expand_for_extern():
    """EXTERN has no callee section at all -- the resolver returns
    ``None`` unconditionally, so ``callee_handle`` is ``None`` and the
    node is non-expandable. The AsmLeaf 1-arm dispatch constructs the
    InlineCallNode with ``callee_handle=None`` for EXTERN regardless of
    any handle the test fixture supplies."""
    node = _make_inline_call(CallTargetType.EXTERN, None)
    assert node.can_expand is False


# ---------------------------------------------------------------------------
# InlineCallNode.expand -- pinned-variant splice + MISSING fallback (D2).
# ---------------------------------------------------------------------------


def test_inline_call_expand_splices_pinned_variant_blocks_directly():
    """Plan decision D2: an InlineCallNode whose ``variant_idx`` matches
    one of the callee's variants surfaces THAT variant's blocks as the
    InlineCallNode's children directly, skipping the intermediate
    variant-list level. The other (unmatched) variants are reachable
    via a trailing :class:`ShowAllVariantsNode` sibling.
    """
    # Three callee variants (idx 0, 1, 2); each yields 2 blocks.
    callee_backend = _make_backend(n_variants=3, n_blocks=2)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=1,  # pin variant idx 1
        provider=None,
    )

    children = node.expand()

    # First N children are BlockNode (the pinned variant's blocks);
    # final child is the ShowAllVariantsNode bundling the OTHER two
    # variants (idx 0, 2). No VariantNode appears among the pinned-
    # variant siblings -- the variant-list level is skipped.
    block_children = [c for c in children if isinstance(c, BlockNode)]
    show_all_children = [c for c in children if isinstance(c, ShowAllVariantsNode)]

    assert len(block_children) == 2
    assert all(c.variant_idx == 1 for c in block_children)
    assert [c.block_idx for c in block_children] == [0, 1]
    assert len(show_all_children) == 1
    # Trailing sibling -- last child slot.
    assert isinstance(children[-1], ShowAllVariantsNode)
    # Other variants in the bundle: idx 0 + idx 2 (NOT the pinned 1).
    other_idxs = sorted(v.variant_idx for v in show_all_children[0].other_variants)
    assert other_idxs == [0, 2]


def test_inline_call_expand_pinned_variant_solo_no_show_all_sibling():
    """When the callee has exactly one variant and the caller pins to
    it, the InlineCallNode's children are just the variant's blocks --
    no :class:`ShowAllVariantsNode` is appended (the bundle would be
    empty)."""
    callee_backend = _make_backend(n_variants=1, n_blocks=3)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=0,
        provider=None,
    )

    children = node.expand()

    assert len(children) == 3
    assert all(isinstance(c, BlockNode) for c in children)
    assert not any(isinstance(c, ShowAllVariantsNode) for c in children)


def test_inline_call_expand_plt_yields_thunk_body_blocks():
    """Regression: a PLT call whose thunk section resolves to a
    backend with 1 variant + N blocks expands into those N blocks
    directly -- not into an empty body (the prior gate restricted
    expansion to LOCAL kinds, leaving PLT rows expandable-looking but
    yielding zero children). The user observed the empty-body
    symptom on ``▼ bl plt function 0: calloc`` -> ``plt function 0:
    calloc`` with no further content; PLT thunks ARE sectioned by the
    writer (small jump-thunk to the resolved extern) so the inline
    expand must surface that body.
    """
    callee_backend = _make_backend(n_variants=1, n_blocks=2)
    callee_handle = _make_handle(idx=7, name="calloc")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.PLT,
        counter_id=0,
        callee_name="calloc",
        callee_handle=callee_handle,
        variant_idx=0,
        provider=None,
    )

    assert node.can_expand is True
    children = node.expand()
    # PLT thunk body surfaced -- non-empty list of BlockNodes, no
    # ShowAllVariantsNode (single-variant callee).
    assert len(children) == 2
    assert all(isinstance(c, BlockNode) for c in children)
    assert [c.block_idx for c in children] == [0, 1]


def test_inline_call_expand_missing_variant_falls_back_to_all_variants():
    """When the caller's variant_idx is not in the callee's surviving
    set (e.g. :data:`MISSING_VARIANT_INDEX` -- the callee dropped that
    variant), the fallback path surfaces EVERY callee variant as a
    sibling :class:`VariantNode` -- matching the pre-D2 contract.
    """
    from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX

    callee_backend = _make_backend(n_variants=3, n_blocks=1)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=MISSING_VARIANT_INDEX,
        provider=None,
    )

    children = node.expand()

    assert len(children) == 3
    assert all(isinstance(c, VariantNode) for c in children)
    assert [c.variant_idx for c in children] == [0, 1, 2]


def test_inline_call_expand_falls_back_to_caller_variant_when_pin_missing():
    """``caller_variant_identity`` rescues the no-pin case (e.g.
    Function-ID self-references) by defaulting the inline-call to a
    callee variant whose canonical-4 build axes ``(arch, compiler,
    compiler_version, opt)`` match the caller's — instead of
    surfacing the full variant list as :class:`VariantNode` siblings.
    Pins :attr:`variant_idx` to :data:`MISSING_VARIANT_INDEX` and
    :attr:`caller_variant_identity` to the canonical-4 of one of the
    callee's variants; the expand result must splice THAT variant's
    blocks directly + bundle the rest under :class:`ShowAllVariantsNode`.
    """
    from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX

    callee_backend = _make_distinct_axes_backend(n_variants=3, n_blocks=2)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    # Caller identity matches callee variant_idx=2's axes ("x86", "gcc",
    # "9.0", "O2"); ``pkg`` + ``variant_id`` differ (intentionally —
    # the match key drops both fields per the canonical-4 contract).
    caller_identity = VariantIdentity(
        arch="x86", compiler="gcc", compiler_version="9.0", opt="O2",
        pkg="caller_pkg", variant_id=42,
    )

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=MISSING_VARIANT_INDEX,  # no callee-side pin
        provider=None,
        caller_variant_identity=caller_identity,
    )

    children = node.expand()

    # Splice the axis-matched variant (idx 2) directly; others go in
    # the ShowAllVariantsNode bundle.
    block_children = [c for c in children if isinstance(c, BlockNode)]
    show_all_children = [c for c in children if isinstance(c, ShowAllVariantsNode)]
    assert len(block_children) == 2
    assert all(c.variant_idx == 2 for c in block_children)
    assert len(show_all_children) == 1
    other_idxs = sorted(v.variant_idx for v in show_all_children[0].other_variants)
    assert other_idxs == [0, 1]


def test_inline_call_expand_prefers_explicit_pin_over_caller_fallback():
    """When both :attr:`variant_idx` and :attr:`caller_variant_identity`
    are present, the explicit callee-side pin wins. The caller
    fallback only activates when the pin is
    :data:`MISSING_VARIANT_INDEX`. The explicit pin's integer is a
    valid intra-section reference recorded by the encoder (per_call_
    entries), so position-by-``variant_idx`` is correct here.
    """
    callee_backend = _make_distinct_axes_backend(n_variants=3, n_blocks=2)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    # Caller identity matches variant_idx=2's axes — but the explicit
    # pin to variant_idx=1 should override.
    caller_identity = VariantIdentity(
        arch="x86", compiler="gcc", compiler_version="9.0", opt="O2",
        pkg="caller_pkg", variant_id=0,
    )

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=1,  # explicit pin
        provider=None,
        caller_variant_identity=caller_identity,  # fallback (ignored)
    )

    children = node.expand()

    block_children = [c for c in children if isinstance(c, BlockNode)]
    assert all(c.variant_idx == 1 for c in block_children)


def test_inline_call_expand_both_missing_surfaces_all_variants():
    """Both :attr:`variant_idx` :data:`MISSING_VARIANT_INDEX` AND
    :attr:`caller_variant_identity` ``None`` -> no pinned variant
    resolves; fallback surfaces every callee variant as a
    :class:`VariantNode` sibling, matching the pre-D2 contract.
    """
    from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX

    callee_backend = _make_distinct_axes_backend(n_variants=3, n_blocks=1)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=MISSING_VARIANT_INDEX,
        provider=None,
        caller_variant_identity=None,
    )

    children = node.expand()

    assert len(children) == 3
    assert all(isinstance(c, VariantNode) for c in children)
    assert [c.variant_idx for c in children] == [0, 1, 2]


def test_inline_call_expand_caller_identity_not_in_callee_surviving_set():
    """Cross-arch edge case: the caller's canonical-4 build axes may
    not be present in the callee's surviving set (e.g. the callee
    only has variants for OTHER arches). Pin is
    :data:`MISSING_VARIANT_INDEX`, fallback identity is
    ``mips/clang/14.0/O1`` (not present in callee's
    ``{x86,arm32,x86} clang/gcc 5.0/8.0/9.0 ...``). Result must fall
    through to the all-variants surface — never a KeyError, never a
    partial splice.
    """
    from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX

    callee_backend = _make_distinct_axes_backend(n_variants=3, n_blocks=1)
    callee_handle = _make_handle(idx=7, name="callee")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    caller_identity = VariantIdentity(
        arch="mips", compiler="clang", compiler_version="14.0", opt="O1",
        pkg="caller_pkg", variant_id=0,
    )

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_handle=callee_handle,
        variant_idx=MISSING_VARIANT_INDEX,
        provider=None,
        caller_variant_identity=caller_identity,
    )

    children = node.expand()

    assert len(children) == 3
    assert all(isinstance(c, VariantNode) for c in children)
    assert [c.variant_idx for c in children] == [0, 1, 2]


def test_inline_call_expand_cross_arm_avoids_arch_incompatible_variant():
    """Regression: the prior implementation matched the caller's
    ``variant_idx`` directly against the callee's ``variant_idx`` list.
    For a cross-arm call (caller in MATCHED arm idx 0, callee section
    in UNMATCHED arm whose variant_idx 0 is an x86 stub), this picked
    the arch-incompatible variant_idx 0 instead of the variant that
    actually shares the caller's build axes. Identity-based matching
    fixes this by keying on the canonical-4 axes — the caller's arm32
    identity finds the callee's arm32 variant regardless of its
    position in the callee's variant_idx space.
    """
    from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX

    # Callee unmatched-arm variants: idx 0 is x86 (the "stub" the
    # user-observed bug landed on), idx 1 is arm32 (the variant the
    # caller actually wants), idx 2 is a third unrelated arch.
    callee_backend = _make_distinct_axes_backend(n_variants=3, n_blocks=2)
    callee_handle = _make_handle(idx=7, name="die")
    factory = MagicMock(spec=BackendFactory)
    factory.handles = [callee_handle]
    factory.make.return_value = callee_backend

    # Caller is arm32 clang 5.0 -O0 (= callee variant_idx=1 by axes).
    caller_identity = VariantIdentity(
        arch="arm32", compiler="clang", compiler_version="5.0", opt="O0",
        pkg="caller_pkg", variant_id=0,
    )

    node = InlineCallNode(
        factory=factory,
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="die",
        callee_handle=callee_handle,
        variant_idx=MISSING_VARIANT_INDEX,
        provider=None,
        caller_variant_identity=caller_identity,
    )

    children = node.expand()

    # CRITICAL: must NOT land on variant_idx 0 (the x86 stub). Must
    # land on variant_idx 1 (the arm32 variant the caller's axes
    # actually match).
    block_children = [c for c in children if isinstance(c, BlockNode)]
    assert len(block_children) == 2
    assert all(c.variant_idx == 1 for c in block_children), (
        f"caller-arch arm32 should match callee variant_idx=1 (arm32), "
        f"NOT variant_idx=0 (x86 stub); got block_children variant_idxs "
        f"{[c.variant_idx for c in block_children]}"
    )
    show_all_children = [
        c for c in children if isinstance(c, ShowAllVariantsNode)
    ]
    assert len(show_all_children) == 1
    other_idxs = sorted(
        v.variant_idx for v in show_all_children[0].other_variants
    )
    assert other_idxs == [0, 2]


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
