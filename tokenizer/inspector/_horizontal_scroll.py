"""Truncation-marker helper for the inspector's tree labels.

Single concern: append a dim ' >>' to a rendered tree-label Text when
the label's cell-width extends past the viewport's current rightmost
visible column. Pure -- no Widget subclass, no Textual app coupling.
The caller (the inspector's Tree subclass in ``_app.py``) drives this
from its ``render_label`` override.

The Tree natively horizontal-scrolls (see Textual's
``widgets/_tree.py``); the marker is a cosmetic overlay so the user
knows scrolling will reveal more content. We DO NOT clip the label --
Textual's ``Tree.render_line`` already crops to the viewport. The
marker is appended unconditionally when the label is wider than the
viewport's right edge; Textual's cropping then keeps only the
rightmost cells of the marker that fall inside the visible window.

This module also owns the visual-cosmetics policy for the
"expand-failed" node-glyph (plan D8). Keeping it here avoids
``_app.py`` hardcoding '[*]' / 'bold red' directly; the Tree subclass
asks this module for the (text, style) pair.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text


__all__ = [
    "MARKER_TEXT",
    "MARKER_STYLE",
    "apply_truncation_marker",
    "assemble_failed_glyph",
]


# The trailing overflow marker. ``" >>"`` (leading space) so it
# visually separates from preceding content even when the label
# happens to end on a non-space cell.
MARKER_TEXT: str = " >>"

# Dim style for the marker -- the marker is a hint, not content; the
# dim attribute keeps it from competing with the actual label text.
MARKER_STYLE: Style = Style(dim=True)


# The "failed expand" glyph (plan D8). A '[*]' prefix in bold red so
# the user spots a failed expansion at a glance. ``_app.py`` consumes
# this via :func:`assemble_failed_glyph`, NEVER inlines the strings.
_FAILED_GLYPH_TEXT: str = "[*] "
_FAILED_GLYPH_STYLE: Style = Style(color="red", bold=True)


def apply_truncation_marker(
    label: Text,
    viewport_width: int,
    scroll_x: int,
) -> Text:
    """Return a Text with ``MARKER_TEXT`` appended when the label spills past the viewport.

    The rightmost visible column of the viewport is
    ``viewport_width + scroll_x``; a label whose cell-width exceeds
    that column has content the user cannot see, so the marker is
    appended. Textual's ``Tree.render_line`` then crops the augmented
    label to the viewport, which naturally keeps only whichever marker
    cells fall inside the visible window.

    Args:
        label: Fully-assembled tree-row label (prefix + node text).
        viewport_width: ``Tree.size.width`` at render time.
        scroll_x: ``Tree.scroll_offset.x`` at render time.

    Returns:
        ``label`` unchanged (no marker) when
        ``label.cell_len <= viewport_width + scroll_x`` OR
        ``viewport_width <= 0`` (unmounted / zero-size). Otherwise a
        fresh :class:`~rich.text.Text` (via :meth:`Text.copy`) with
        :data:`MARKER_TEXT` appended in :data:`MARKER_STYLE`.

    The input is NEVER mutated -- Textual's ``TreeNode`` caches labels
    across renders, so an in-place ``append`` on the cached label
    would compound a new marker on every paint.
    """
    if viewport_width <= 0:
        return label
    rightmost_visible_column = viewport_width + scroll_x
    if label.cell_len <= rightmost_visible_column:
        return label
    augmented = label.copy()
    augmented.append(MARKER_TEXT, style=MARKER_STYLE)
    return augmented


def assemble_failed_glyph(base_style: Style) -> tuple[str, Style]:
    """Prefix glyph + style for a node whose ``expand()`` raised (plan D8).

    Returns the ``(text, style)`` tuple consumable by
    :meth:`Text.assemble`. The style is ``base_style`` merged with
    :data:`_FAILED_GLYPH_STYLE` via Rich's :class:`Style` ``+`` operator,
    so any caller-supplied attributes (e.g. the tree's per-node
    highlight) carry through underneath the failed-glyph emphasis.

    The caller substitutes this for Textual's default
    ``ICON_NODE`` / ``ICON_NODE_EXPANDED`` prefix when the tree node's
    payload indicates a failed expansion.
    """
    return _FAILED_GLYPH_TEXT, base_style + _FAILED_GLYPH_STYLE
