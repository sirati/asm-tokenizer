"""Status bar widget: cursor breadcrumb (left) + filter summary (right).

Single concern: a one-row :class:`textual.widgets.Static` widget
mounted at the bottom of the inspector viewport. Two text segments
joined left/right:

* **Left** -- inline breadcrumb of the cursor's ancestor chain, brief
  and NOT a tree view. Format: ``Function > Variant > Block`` (or
  whichever ancestor chain applies to the current cursor). Example:
  ``Calloc > arm32 clang v5.0 -O0 > Block: 3``.
* **Right** -- active filter summary. Format:
  ``filter: -arch:x86 -comp:gcc`` when any axis disables a value;
  empty when no filter is active.

The pure-text helpers (:func:`breadcrumb_for_cursor` /
:func:`filter_summary`) live at module scope so unit tests can drive
them without a running app. The widget class is the thin Textual
adapter that wires the cursor watcher + filter-config watch into a
single :meth:`Static.update` call.

The widget reads the cursor via the :class:`_InspectorTree` widget it
was constructed with (passed in via :meth:`set_tree`) instead of
querying the app each refresh; this keeps the per-refresh path off the
query DOM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

from rich.text import Text
from textual.widgets import Static

from tokenizer.inspector._label import variant_label_from_axes
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
from tokenizer.inspector._render._protocol import BlockKind

from ._filter import FilterConfig
from ._order import VariantGroupNode


if TYPE_CHECKING:
    from textual.widgets._tree import TreeNode

    from ._tree_widget import _InspectorTree


__all__ = [
    "StatusBar",
    "breadcrumb_for_cursor",
    "filter_summary",
    "node_breadcrumb_segment",
]


# Separator between breadcrumb segments. Single chevron + spaces reads
# the same way browser breadcrumbs do; the typography is intentional
# (a real ``>`` instead of an arrow glyph so the row remains plain ASCII).
_BREADCRUMB_SEP = " > "


# Maximum cell width for an :class:`AsmLeaf` breadcrumb segment. Asm
# lines can be long; the breadcrumb stays single-line so truncation
# keeps the bar readable. Truncation reuses Python slicing (the bar's
# Rich rendering owns the ellipsis suffix below).
_ASM_LEAF_MAX_CHARS = 40
_ASM_LEAF_TRUNCATION_SUFFIX = "..."


# ---------------------------------------------------------------------------
# Pure breadcrumb composition
# ---------------------------------------------------------------------------


def node_breadcrumb_segment(node: Node) -> Optional[str]:
    """One breadcrumb segment for ``node`` (or ``None`` to skip).

    Dispatch by ``isinstance`` on the closed :class:`Node` union plus
    :class:`VariantGroupNode` (which lives in :mod:`._order` outside the
    union). Per-segment policy:

    * :class:`FunctionNode` -> ``node.name`` (e.g. ``Calloc``).
    * :class:`VariantNode` -> the un-aligned axis label string (so
      group-suppressed ancestors don't leak into the breadcrumb's
      single-line form).
    * :class:`BlockNode` -> ``"Block: <idx>"`` for body blocks, the
      fixed label (``"Variant Header"`` / ``"Function ID"``) for the
      non-body kinds. The block-preview text is intentionally dropped
      from the breadcrumb -- it's already visible on the cursor row.
    * :class:`InlineCallNode` -> ``"<word> function <K>: <name>"`` (the
      same shape :func:`tokenizer.inspector._label.inline_call_label`
      renders, without the provider suffix to keep the breadcrumb
      short).
    * :class:`InlineJumpNode` -> ``"jump block: <target>"``.
    * :class:`ShowAllVariantsNode` -> ``"show all variants"``.
    * :class:`InlineCallMissingVariantLeaf` -> the message itself.
    * :class:`AsmLeaf` -> the (possibly-truncated) line text.
    * :class:`NumberPrecisionLeaf` -> the precision text.
    * :class:`VariantGroupNode` -> ``None`` (intermediate group rows
      add noise to the inline form; the variant they parent already
      carries the axis value the user is looking at).

    Returns ``None`` for nodes that should be skipped (currently only
    :class:`VariantGroupNode`) so the caller filters in one pass.
    """
    if isinstance(node, FunctionNode):
        return node.name
    if isinstance(node, VariantGroupNode):
        return None
    if isinstance(node, VariantNode):
        # Use the unaligned form: the breadcrumb is single-line and the
        # column-alignment that ``aligned_label`` carries assumes a
        # vertically-stacked sibling set. ``suppressed_axes`` defaults
        # to empty so every axis present on the variant surfaces here.
        return variant_label_from_axes(node.label_axes)
    if isinstance(node, BlockNode):
        if node.kind is BlockKind.BODY:
            return f"Block: {node.block_idx}"
        # Match the labels module's per-kind names.
        return {
            BlockKind.VARIANT_HEADER: "Variant Header",
            BlockKind.FUNCTION_ID: "Function ID",
        }.get(node.kind, f"block:{node.block_idx}")
    if isinstance(node, InlineCallNode):
        return f"call {node.counter_id}: {node.callee_name}"
    if isinstance(node, InlineJumpNode):
        return f"jump block: {node.target_block_idx}"
    if isinstance(node, ShowAllVariantsNode):
        return node.label
    if isinstance(node, InlineCallMissingVariantLeaf):
        return node.message
    if isinstance(node, AsmLeaf):
        text = node.text
        if len(text) > _ASM_LEAF_MAX_CHARS:
            return text[: _ASM_LEAF_MAX_CHARS - len(_ASM_LEAF_TRUNCATION_SUFFIX)] + _ASM_LEAF_TRUNCATION_SUFFIX
        return text
    if isinstance(node, NumberPrecisionLeaf):
        return node.text
    # Defensive: any future Node subclass we forgot. Returning None
    # avoids a hard crash in the status bar render path (a non-fatal
    # gap is preferable to taking down the TUI).
    return None


def breadcrumb_for_cursor(tree_node: "Optional[TreeNode[Node]]") -> str:
    """Walk the cursor's ancestor chain and join the per-node segments.

    The walk follows :attr:`TreeNode.parent` from the cursor up to the
    root, collecting one segment per non-``None`` model along the way.
    The root tree node has ``data is None`` (per
    :class:`_InspectorTree`'s seeding) so it's skipped naturally. The
    list is reversed so the deepest segment renders last (matching the
    drill-down direction the user reads top-to-bottom in the tree).

    Returns an empty string when the cursor is on the root or there
    are no model-bearing ancestors -- the status bar's left half will
    render empty in that case.
    """
    if tree_node is None:
        return ""
    segments: list[str] = []
    cursor: "Optional[TreeNode[Node]]" = tree_node
    while cursor is not None:
        data = cursor.data
        if data is not None:
            seg = node_breadcrumb_segment(data)
            if seg is not None and seg != "":
                segments.append(seg)
        cursor = cursor.parent
    if not segments:
        return ""
    segments.reverse()
    return _BREADCRUMB_SEP.join(segments)


# ---------------------------------------------------------------------------
# Filter summary
# ---------------------------------------------------------------------------


# Per-axis label override for the brief filter summary form. The axis
# descriptor's ``.label`` is human-readable but long for some axes
# (e.g. ``"compiler"`` / ``"version"``); the brief summary uses the
# canonical short prefix names users would type at a CLI. Falls back to
# ``axis.label`` for unknown axes (EXTRA_META keys are passed through
# verbatim).
_BRIEF_AXIS_LABEL_OVERRIDES: dict[str, str] = {
    "arch": "arch",
    "compiler": "comp",
    "version": "cver",
    "opt": "opt",
    "32/64": "bw",
}


def _brief_axis_label(axis_label: str) -> str:
    return _BRIEF_AXIS_LABEL_OVERRIDES.get(axis_label, axis_label)


def filter_summary(config: "Optional[FilterConfig]") -> str:
    """Render the brief filter status.

    Empty / ``None`` config -> ``""`` (the status bar hides the right
    half).

    Non-empty config -> ``"filter: -arch:x86 -comp:gcc"`` style. The
    per-axis fragment lists every disabled value joined by ``,`` so a
    multi-value disable reads as ``-arch:x86,arm32``. Axes are sorted
    by their brief label for deterministic output.
    """
    if config is None or config.is_empty():
        return ""
    fragments: list[tuple[str, str]] = []
    for axis, disabled_values in config.disabled.items():
        if not disabled_values:
            continue
        brief = _brief_axis_label(axis.label)
        values_joined = ",".join(sorted(disabled_values))
        fragments.append((brief, f"-{brief}:{values_joined}"))
    if not fragments:
        return ""
    fragments.sort(key=lambda pair: pair[0])
    return "filter: " + " ".join(frag for _, frag in fragments)


# ---------------------------------------------------------------------------
# Status bar widget
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """One-line status bar at the bottom of the inspector viewport.

    Left segment: breadcrumb of the cursor's ancestor chain.
    Right segment: active filter summary.

    The widget owns no app state -- :meth:`refresh_state` is called by
    the App layer with the current cursor + filter config; the widget
    renders a single Rich ``Text`` line with the two halves on opposite
    edges of the row.

    The Tree handle is stashed via :meth:`set_tree` so refresh can read
    ``cursor_node`` directly without going through ``app.query_one``.
    The dim style matches the rest of the chrome (search prompt,
    horizontal-scroll marker).
    """

    CSS: ClassVar[str] = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $panel;
        color: $foreground;
    }
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._tree: "Optional[_InspectorTree]" = None
        self._filter_config: "Optional[FilterConfig]" = None

    def set_tree(self, tree: "_InspectorTree") -> None:
        """Stash the tree handle so :meth:`refresh_state` reads cursor."""
        self._tree = tree

    def set_filter_config(self, config: "Optional[FilterConfig]") -> None:
        """Update the active filter config + re-render."""
        self._filter_config = config
        self.refresh_state()

    def refresh_state(self) -> None:
        """Recompose the status bar text from current cursor + filter."""
        cursor_node = (
            None if self._tree is None else self._tree.cursor_node
        )
        left = breadcrumb_for_cursor(cursor_node)
        right = filter_summary(self._filter_config)
        self.update(_compose_status_line(left, right, self.size.width))


def _compose_status_line(
    left: str, right: str, viewport_width: int
) -> Text:
    """Compose a left/right-justified single-line status row.

    The Rich :class:`Text` is built as ``left + spacer + right`` where
    ``spacer`` is the padding needed to push ``right`` to the right
    edge of the available viewport. When the combined text exceeds the
    viewport, the left fragment is truncated (with an ellipsis) so the
    right fragment stays visible -- the filter summary is the
    information the user is most likely to need at a glance after
    setting an active filter.
    """
    if viewport_width <= 0:
        # Pre-mount / zero-sized viewport: render the raw join so the
        # composition still produces visible content for tests that
        # inspect the rendered text.
        return Text(f"{left}   {right}".strip())
    if not right:
        return Text(_truncate_to(left, viewport_width))
    if not left:
        # Right-align by padding from the left.
        pad = max(0, viewport_width - _cell_len(right))
        return Text((" " * pad) + right)
    # Both halves: leave at least one space between them; truncate the
    # left fragment when the combination overflows.
    right_len = _cell_len(right)
    available_for_left = max(0, viewport_width - right_len - 1)
    left_trunc = _truncate_to(left, available_for_left)
    spacer = max(1, viewport_width - _cell_len(left_trunc) - right_len)
    return Text(left_trunc + (" " * spacer) + right)


def _cell_len(text: str) -> int:
    """Cheap cell-len proxy: ``len`` for the ASCII-heavy strings used here.

    The breadcrumb segments + the filter summary are emitted as plain
    ASCII; a Rich measure would be overkill for the status bar's
    fixed-width single-row content.
    """
    return len(text)


def _truncate_to(text: str, max_cells: int) -> str:
    """Truncate ``text`` to at most ``max_cells`` cells with an ellipsis.

    Returns an empty string when ``max_cells <= 0``. When the suffix
    alone wouldn't fit (``max_cells`` smaller than the suffix length),
    falls back to a flat truncation without the suffix marker.
    """
    if max_cells <= 0:
        return ""
    if len(text) <= max_cells:
        return text
    suffix = "..."
    if max_cells <= len(suffix):
        return text[:max_cells]
    return text[: max_cells - len(suffix)] + suffix
