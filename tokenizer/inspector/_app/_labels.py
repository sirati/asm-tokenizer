"""Node-typed label dispatcher for the inspector tree rows.

Single concern: translate one model :class:`Node` into its visible
``rich.text.Text`` label by dispatching on the closed Node union.
Atomic ``data -> str`` mapping lives in :mod:`tokenizer.inspector._label`;
this module is the thin Node-aware bridge between the model layer and
the Textual tree widget.
"""

from __future__ import annotations

from rich.text import Text

from tokenizer.inspector._label import (
    function_label,
    inline_call_label,
    inline_jump_label,
    variant_label_from_axes,
)
from tokenizer.inspector._render._protocol import BlockKind
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    BlockNode,
    FunctionNode,
    InlineCallNode,
    InlineJumpNode,
    Node,
    NumberPrecisionLeaf,
    ShowAllVariantsNode,
    VariantNode,
)


__all__ = ["_compose_label", "_block_node_label", "_BLOCK_KIND_LABELS"]


# ---------------------------------------------------------------------------
# Label composition (UI side -- plan-drift audit explicitly accepts that
# the BlockNode label composes ``"Block: <i>   <preview>"`` here, since
# the model carries ``block_idx`` + ``preview`` separately).
# ---------------------------------------------------------------------------


# Per-kind label policy for :class:`BlockNode` rows. The model node
# carries the typed :class:`BlockKind` discriminator; the UI side
# composes the row label per-kind without ``isinstance``-on-string
# discriminators.
_BLOCK_KIND_LABELS: dict[BlockKind, str] = {
    BlockKind.VARIANT_HEADER: "Variant Header",
    BlockKind.FUNCTION_ID: "Function ID",
}


def _block_node_label(node: "BlockNode") -> str:
    """Compose the row text for a :class:`BlockNode`.

    Dispatches off :attr:`BlockNode.kind`: the two non-body kinds
    render their fixed section name; :attr:`BlockKind.BODY` composes
    ``Block: <idx>   <preview>`` (the legacy shape).
    """
    fixed = _BLOCK_KIND_LABELS.get(node.kind)
    if fixed is not None:
        return fixed
    return f"Block: {node.block_idx}   {node.preview}"


def _compose_label(node: Node) -> Text:
    """Translate one model node into its visible label text.

    Dispatch is by ``isinstance`` on the seven concrete node types
    (no string compares on type names). The BlockNode label composes
    its two model fields (``block_idx`` + ``preview``) into the row
    text here on the UI side -- the model deliberately splits them.
    """
    if isinstance(node, FunctionNode):
        return Text(function_label(node.name))
    if isinstance(node, VariantNode):
        return Text(variant_label_from_axes(node.label_axes))
    if isinstance(node, BlockNode):
        return Text(_block_node_label(node))
    if isinstance(node, InlineCallNode):
        return Text(
            inline_call_label(
                node.kind, node.counter_id, node.callee_name, node.provider
            )
        )
    if isinstance(node, InlineJumpNode):
        return Text(inline_jump_label(node.target_block_idx))
    if isinstance(node, AsmLeaf):
        return Text(node.text)
    if isinstance(node, NumberPrecisionLeaf):
        return Text(node.text)
    if isinstance(node, ShowAllVariantsNode):
        return Text(node.label)
    # Closed Node union; any miss is a model/UI contract drift.
    raise TypeError(f"unsupported node type: {type(node).__name__}")
