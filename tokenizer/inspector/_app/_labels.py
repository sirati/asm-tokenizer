"""Node-typed label dispatcher for the inspector tree rows.

Single concern: translate one model :class:`Node` into its visible
``rich.text.Text`` label by dispatching on the closed Node union.
Atomic ``data -> str`` mapping lives in :mod:`tokenizer.inspector._label`;
this module is the thin Node-aware bridge between the model layer and
the Textual tree widget.
"""

from __future__ import annotations

from rich.style import Style
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
    InlineCallMissingVariantLeaf,
    InlineCallNode,
    InlineJumpNode,
    Node,
    NumberPrecisionLeaf,
    ShowAllVariantsNode,
    VariantNode,
)

from ._order import VariantGroupNode, format_grouping_label


# Dim red style for informational error rows (e.g. the missing-pin
# leaf under an :class:`InlineCallNode` fallback). Kept identical to
# the dispatcher's ``_ERR_STYLE`` in :mod:`._application` so a model-
# data shape issue surfaced as content and a caught-exception error
# leaf paint the same red tint -- the two paths share the visual
# vocabulary without sharing the ``is_failed`` flag (which carries
# additional "re-run the failing expand on next click" semantics).
_ERR_STYLE: Style = Style(color="red", dim=True)

# Style applied to a :class:`FunctionNode` row whose variants are all
# filtered out by the active :class:`FilterConfig`. The row stays
# visible (so the user can see the function exists) but loses its
# expand triangle (mounted with ``allow_expand=False``) and renders
# dim so the eye skims past it. Single point of control: every
# "function filtered to nothing" rendering must reach this style via
# :func:`_compose_label_filtered_out` -- no concatenated escapes, no
# inline styling at the call site.
_FILTERED_OUT_STYLE: Style = Style(dim=True)


# Muted style for the per-block-row asm preview suffix appended after
# the ``"Block: <i>"`` / ``"Jump table: <i>"`` prefix. ``dim`` lets the
# user visually distinguish the canonical row label (block index) from
# the asm-text preview that sits beside it -- the preview is auxiliary
# context, not the row's identity.
_BLOCK_PREVIEW_STYLE: Style = Style(dim=True)


__all__ = [
    "_BLOCK_KIND_INDEXED_PREFIXES",
    "_BLOCK_KIND_LABELS",
    "_BLOCK_PREVIEW_STYLE",
    "_ERR_STYLE",
    "_FILTERED_OUT_STYLE",
    "_block_node_label",
    "_compose_label",
    "_compose_label_filtered_out",
]


# ---------------------------------------------------------------------------
# Label composition (UI side -- plan-drift audit explicitly accepts that
# the BlockNode label composes ``"Block: <i>   <preview>"`` here, since
# the model carries ``block_idx`` + ``preview`` separately).
# ---------------------------------------------------------------------------


# Per-kind label policy for :class:`BlockNode` rows. The model node
# carries the typed :class:`BlockKind` discriminator; the UI side
# composes the row label per-kind without ``isinstance``-on-string
# discriminators. The non-indexed kinds (variant prefix / function
# id) carry a fixed-string label; the indexed kinds (body block,
# jump-table footer) compose ``"<prefix>: <idx>   <preview>"`` via
# the per-kind prefix dict below.
_BLOCK_KIND_LABELS: dict[BlockKind, str] = {
    BlockKind.VARIANT_HEADER: "Variant Header",
    BlockKind.FUNCTION_ID: "Function ID",
}

# Per-kind row-prefix word for the indexed section kinds. The prefix
# selects the user-visible label noun: body blocks render as
# ``"Block: <idx>"`` and jump-table footers render as
# ``"Jump table: <idx>"`` (per user directive: "list jump tables just
# as you list blocks"). Centralising the dispatch here avoids forking
# the label-compose path on the section discriminator.
_BLOCK_KIND_INDEXED_PREFIXES: dict[BlockKind, str] = {
    BlockKind.BODY: "Block",
    BlockKind.JUMP_TABLE: "Jump table",
}


def _block_node_label(node: "BlockNode", *, show_preview: bool = True) -> Text:
    """Compose the row text for a :class:`BlockNode` as styled ``Text``.

    Dispatches off :attr:`BlockNode.kind`: the non-indexed kinds
    (variant prefix / function id) render their fixed section name;
    the indexed kinds (body block / jump-table footer) compose a
    ``"<prefix>: <idx>"`` base with -- when ``show_preview`` is True
    AND the preview is non-empty -- a ``"   <preview>"`` muted-style
    suffix appended in :data:`_BLOCK_PREVIEW_STYLE`. ``show_preview``
    is False on already-expanded block rows (the user is looking at
    the content; the preview is redundant) and globally toggleable
    via the App-level ``p`` binding -- the per-row branch lives here
    so the label composer's single concern stays "translate node to
    Text".

    When :attr:`BlockNode.aligned_prefix_width` is set (stamped by the
    App-side sibling-set pass in
    :func:`tokenizer.inspector._app._application._stamp_aligned_block_prefix_width`),
    the ``"<prefix>: <idx>"`` chunk is left-padded to that width so the
    preview suffix of every indexed sibling row starts at the same
    column. ``None`` falls back to the unpadded form (single-block
    unit tests, non-indexed kinds).
    """
    fixed = _BLOCK_KIND_LABELS.get(node.kind)
    if fixed is not None:
        return Text(fixed)
    prefix = _BLOCK_KIND_INDEXED_PREFIXES[node.kind]
    base = f"{prefix}: {node.block_idx}"
    if node.aligned_prefix_width is not None:
        base = base.ljust(node.aligned_prefix_width)
    text = Text(base)
    if show_preview and node.preview:
        text.append("   ")
        text.append(node.preview, style=_BLOCK_PREVIEW_STYLE)
    return text


def _compose_label(node: Node, *, show_block_preview: bool = True) -> Text:
    """Translate one model node into its visible label text.

    Dispatch is by ``isinstance`` on the seven concrete node types
    (no string compares on type names). The BlockNode label composes
    its two model fields (``block_idx`` + ``preview``) into the row
    text here on the UI side -- the model deliberately splits them.

    ``show_block_preview`` is forwarded to :func:`_block_node_label`
    when the node is a :class:`BlockNode`. Other node types ignore
    it (they have no preview concept). Callers set it to False on
    already-expanded block rows OR when the App-level preview toggle
    is off; defaults to True so callers that don't care preserve the
    legacy "always show preview" behaviour.
    """
    if isinstance(node, FunctionNode):
        return Text(function_label(node.name))
    if isinstance(node, VariantNode):
        # ``aligned_label`` is pre-computed by the sibling-set-aware
        # stamp in :mod:`tokenizer.inspector._app._application` when
        # variants flow through the expand dispatcher; falls back to
        # the unaligned form for nodes built outside that context
        # (e.g. single-variant unit tests).
        if node.aligned_label is not None:
            return Text(node.aligned_label)
        return Text(variant_label_from_axes(node.label_axes))
    if isinstance(node, BlockNode):
        return _block_node_label(node, show_preview=show_block_preview)
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
    if isinstance(node, InlineCallMissingVariantLeaf):
        # The leaf is informational, not a caught exception, so the
        # ``is_failed`` prefix-glyph dispatch in the tree widget does
        # NOT fire here. We render a literal ``[*] `` glyph inline in
        # the label text + the message body, both in dim red so the
        # row reads as a fallback signal under the parent InlineCall.
        return Text(f"[*] {node.message}", style=_ERR_STYLE)
    if isinstance(node, ShowAllVariantsNode):
        return Text(node.label)
    if isinstance(node, VariantGroupNode):
        return Text(format_grouping_label(node.axis, node.axis_value))
    # Closed Node union; any miss is a model/UI contract drift.
    raise TypeError(f"unsupported node type: {type(node).__name__}")


def _compose_label_filtered_out(node: Node) -> Text:
    """:func:`_compose_label` result re-styled with :data:`_FILTERED_OUT_STYLE`.

    Single concern: paint a node whose visible content is correct but
    whose semantic state is "filtered to nothing" -- the row stays
    addressable but reads as inactive. Reuses :func:`_compose_label` so
    the underlying text composition (FunctionNode prefix, variant
    aligned label, etc.) cannot diverge from the un-styled path.
    """
    text = _compose_label(node)
    text.stylize(_FILTERED_OUT_STYLE)
    return text
