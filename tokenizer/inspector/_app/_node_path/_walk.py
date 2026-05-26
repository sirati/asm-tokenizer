"""Tree-walk capture: derive a :class:`NodePath` for any
:class:`TreeNode` and produce the :class:`CapturedExpandState`
snapshot consumed by the dispatcher's post-mount restore hook.

Single concern: the read side of the capture-on-rebuild contract.
:func:`node_key_for` is the per-model identity dispatcher (one
``match`` arm per tree-model node class, no string discriminators);
:func:`node_path_for` walks the parent chain to compose a full path;
:func:`capture_expand_state` produces the snapshot the
:mod:`._match` matcher consumes.

The :class:`CapturedExpandState` carries TWO parallel views of the
same set of paths -- full (group keys preserved) and logical (group
keys elided). The logical projection is the source of truth for
content-row matching across regroups; the full set is kept so a
re-grouping with identical axes still re-expands the same group
buckets (see :mod:`._match` for the rules).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
)
from tokenizer.inspector._render._protocol import (
    InlineCallEntry,
    InlineJumpEntry,
)
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    BlockNode,
    FunctionNode,
    InlineCallNode,
    InlineJumpNode,
    NumberPrecisionLeaf,
    ShowAllVariantsNode,
    VariantNode,
)

from .._order import VariantGroupNode
from ._keys import (
    AsmLeafTerminalKey,
    AsmLeafWrapperCallKey,
    AsmLeafWrapperJumpKey,
    AsmLeafWrapperNumberKey,
    BlockKey,
    FunctionKey,
    InlineCallKey,
    InlineJumpKey,
    NodeKey,
    NodePath,
    NumberPrecisionKey,
    ShowAllVariantsKey,
    VariantGroupKey,
    VariantKey,
)


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from tokenizer.inspector._tree_model import Node


__all__ = [
    "CapturedExpandState",
    "capture_expand_state",
    "logical_path_of",
    "node_key_for",
    "node_path_for",
]


# ---------------------------------------------------------------------------
# Per-node identity dispatch.
# ---------------------------------------------------------------------------


def node_key_for(
    model: "Node",
    *,
    sibling_index: int,
) -> Optional[NodeKey]:
    """Compute the typed :class:`NodeKey` for a tree-model node.

    ``sibling_index`` is the position of this node among its parent's
    children -- used only as the fallback identity for terminal
    :class:`AsmLeaf` rows. For every other node kind the identity is
    derived purely from the model fields.

    Returns ``None`` when the node is the root container (no payload
    -- :attr:`TreeNode.data` is ``None``); callers stop walking at
    that point.
    """
    match model:
        case FunctionNode():
            return FunctionKey(handle=model.handle)
        case VariantNode():
            # The :class:`VariantIdentity` is sourced from the
            # backend's :class:`RenderedVariant` table; the variant
            # row may pre-date the rebuild walk (so we look it up via
            # the parent backend at capture time -- see
            # :func:`variant_key_via_backend`).
            return variant_key_via_backend(model)
        case VariantGroupNode():
            return VariantGroupKey(
                axis_kind=model.axis.kind,
                axis_key=model.axis.key,
                axis_value=model.axis_value,
            )
        case BlockNode():
            return BlockKey(kind=model.kind, block_idx=model.block_idx)
        case InlineJumpNode():
            return InlineJumpKey(target_block_idx=model.target_block_idx)
        case InlineCallNode():
            return InlineCallKey(
                kind=model.kind, counter_id=model.counter_id
            )
        case AsmLeaf():
            return _asm_leaf_key(model, sibling_index=sibling_index)
        case ShowAllVariantsNode():
            return ShowAllVariantsKey()
        case NumberPrecisionLeaf():
            return NumberPrecisionKey(text=model.text)
        case _:
            # Unknown / unrepresentable -- caller treats as opaque
            # and stops accumulating identity for this branch.
            return None


def variant_key_via_backend(model: VariantNode) -> Optional[VariantKey]:
    """Resolve :class:`VariantIdentity` for a model node via its backend.

    The :class:`VariantNode` carries only ``variant_idx`` + the
    flattened ``label_axes`` -- the canonical
    :class:`VariantIdentity` lives on :class:`RenderedVariant`. The
    backend's ``variants()`` is metadata-only + cached, so the lookup
    is cheap. Exposed at module scope so the :mod:`._match`
    descendancy walker can re-use the same backend lookup.
    """
    backend = getattr(model, "backend", None)
    if backend is None:
        return None
    try:
        for rv in backend.variants():
            if rv.variant_idx == model.variant_idx:
                return VariantKey(identity=rv.variant_identity)
    except Exception:
        # Backend mid-failure -- treat as opaque (capture path drops
        # this entry); cursor + expand restoration still works for
        # ancestors that resolved successfully.
        return None
    return None


def _asm_leaf_key(leaf: AsmLeaf, *, sibling_index: int) -> NodeKey:
    """Per-arm identity for an :class:`AsmLeaf`.

    Matches the leaf's 3-arm expand contract:

    * Terminal (0 openables) -> sibling-index fallback (cursor only).
    * Single-openable wrapper -> openable-typed key matching the
      node the leaf would expand into.
    """
    if not leaf.openables:
        return AsmLeafTerminalKey(sibling_index=sibling_index)
    # Wrapper carries exactly one openable (either the natural 1-arm
    # case OR a 2+-case wrapper produced by ``_wrap_openable_as_node``).
    openable = leaf.openables[0]
    match openable:
        case InlineCallEntry():
            return AsmLeafWrapperCallKey(
                kind=openable.kind, counter_id=openable.counter_id
            )
        case InlineJumpEntry():
            return AsmLeafWrapperJumpKey(
                target_block_idx=openable.target_block_idx
            )
        case InlineNumberPrecisionEntry():
            return AsmLeafWrapperNumberKey(full_text=openable.full_text)
        case _:
            # Unknown openable type -- closed union, drift signal.
            return AsmLeafTerminalKey(sibling_index=sibling_index)


# ---------------------------------------------------------------------------
# Tree-walk capture: collect expanded NodePaths + cursor NodePath.
# ---------------------------------------------------------------------------


def node_path_for(tree_node: "TreeNode[Node]") -> Optional[NodePath]:
    """Walk parent-chain to build the :class:`NodePath` for this tree node.

    Returns ``None`` when any intermediate node yields a ``None``
    key (e.g. the root container, or a node whose backend lookup
    failed); callers treat that branch as un-identifiable.
    """
    keys: list = []
    cursor: Optional["TreeNode[Node]"] = tree_node
    while cursor is not None:
        model = cursor.data
        if model is None:
            # Hit the root container -- stop accumulating.
            break
        # The sibling index is read off the parent's children list;
        # only needed for terminal AsmLeaf so the lookup is cheap (and
        # parents-without-children are the root itself, handled above).
        sibling_index = _sibling_index(cursor)
        key = node_key_for(model, sibling_index=sibling_index)
        if key is None:
            return None
        keys.append(key)
        cursor = cursor.parent
    if not keys:
        return None
    keys.reverse()
    return tuple(keys)


def _sibling_index(tree_node: "TreeNode[Node]") -> int:
    """Position of ``tree_node`` among its parent's children.

    Returns ``0`` when ``tree_node`` is the root (no parent); callers
    only use the index for terminal :class:`AsmLeaf` rows so the
    root-case fallback is unreachable in practice.
    """
    parent = tree_node.parent
    if parent is None:
        return 0
    for i, child in enumerate(parent.children):
        if child is tree_node:
            return i
    return 0


def logical_path_of(path: NodePath) -> NodePath:
    """Strip every :class:`VariantGroupKey` from a path.

    Group keys are organisational wrappers: they identify a bucket,
    not a logical row. Eliding them gives a path whose entries
    identify the LOGICAL row (function -> variant -> block -> ...)
    independent of which grouping axes are active.

    The :func:`should_expand_for_capture` matcher uses logical
    paths so a row's expand-state survives any regroup that
    re-routes the row under a different group ancestry. Group rows
    themselves keep full-path equality as a separate match channel
    (see :mod:`._match`).
    """
    return tuple(k for k in path if not isinstance(k, VariantGroupKey))


@dataclass(frozen=True)
class CapturedExpandState:
    """Capture-on-rebuild payload threaded through the dispatcher.

    Two parallel views of the same set of expanded tree nodes:

    * :attr:`full_paths` -- with :class:`VariantGroupKey` entries
      preserved. Used to re-expand a group whose exact axis +
      bucket survived the regroup.
    * :attr:`logical_paths` -- with group keys elided. Used to
      re-expand content rows (variants / blocks / calls / ...)
      across regroups that rewrap them under different group
      ancestry.

    :attr:`cursor_path` is the cursor row's full path (may be
    ``None`` if the cursor is on a transient row whose path
    doesn't resolve, e.g. an error-leaf). Cursor restoration
    matches on the logical projection for the same robustness
    reason content expansion does.
    """

    full_paths: "frozenset[NodePath]"
    logical_paths: "frozenset[NodePath]"
    cursor_path: Optional[NodePath]


def capture_expand_state(
    root: "TreeNode[Node]",
    cursor_node: "Optional[TreeNode[Node]]",
) -> CapturedExpandState:
    """Capture every expanded :class:`TreeNode`'s :class:`NodePath`.

    Walks the tree under ``root`` (excluding the root container
    itself) and collects:

    * Every expanded tree node's :class:`NodePath` (full).
    * The logical projection of each captured path (group keys
      elided) so content-row matching survives a regroup that
      moves the row under different group ancestry.
    * The cursor row's path (may be ``None``).

    The cursor's logical path is also folded into
    :attr:`logical_paths` so the auto-expand walk surfaces every
    ancestor of the cursor target -- a row whose parent has not
    been expanded is not mounted, and an un-mounted row cannot
    receive the cursor.
    """
    full: set[NodePath] = set()
    stack: list["TreeNode[Node]"] = list(root.children)
    while stack:
        node = stack.pop()
        if node.is_expanded:
            path = node_path_for(node)
            if path is not None:
                full.add(path)
        stack.extend(node.children)
    cursor_path = (
        node_path_for(cursor_node) if cursor_node is not None else None
    )
    logical_set = {logical_path_of(p) for p in full}
    if cursor_path is not None:
        # Cursor restoration requires every ancestor of the target
        # row to be expanded; folding the cursor's logical path into
        # the auto-expand set lets the standard
        # :func:`should_expand_for_capture` matcher surface those
        # ancestors as a side effect.
        logical_set.add(logical_path_of(cursor_path))
    return CapturedExpandState(
        full_paths=frozenset(full),
        logical_paths=frozenset(logical_set),
        cursor_path=cursor_path,
    )
