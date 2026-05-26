"""Help-modal screen rendering live bindings via Textual's BindingsTable.

Single concern: surface the currently-active key bindings to the user
without a hand-maintained help string. The content auto-derives from
:meth:`textual.screen.Screen.active_bindings` of the screen UNDER the
modal -- so any future binding added to :class:`InspectorApp` or
:class:`_InspectorTree` shows up here for free, grouped under each
namespace's ``BINDING_GROUP_TITLE`` ClassVar.

The modal owns no application logic; it dismisses on ``escape`` via
Textual's canonical :meth:`textual.screen.Screen.action_dismiss`.

The :class:`BindingsTable` subclass below exists because Textual's
upstream widget hard-codes ``self.screen.active_bindings`` as the
source -- inside a :class:`ModalScreen` that resolves to the modal
itself (just the ``escape`` binding). We override one method to read
``app.screen_stack[-2]`` (the inspector screen beneath the modal)
instead; every other concern -- component styles, key-display
formatting, ``BINDING_GROUP_TITLE`` grouping -- stays the upstream's.

The widget is mounted directly on the modal screen (no
:class:`textual.containers.Container` wrapper). Textual's auto-width
propagation collapses to width 0 when an auto-sized parent wraps an
auto-sized :class:`BindingsTable` child, so the modal screen's
``align: center middle`` rule centres the widget itself -- and the
border + padding styling lives on the widget rather than a wrapper.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING, ClassVar

from rich import box
from rich.table import Table
from rich.text import Text

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets._key_panel import BindingsTable


if TYPE_CHECKING:
    from textual.screen import Screen


__all__ = ["HelpScreen"]


class _UnderlyingScreenBindingsTable(BindingsTable):
    """Retarget the binding source to the screen under the modal."""

    DEFAULT_CSS: ClassVar[str] = """
    _UnderlyingScreenBindingsTable {
        width: auto;
        height: auto;
        padding: 1 2;
        background: $surface;
    }
    """

    def render_bindings_table(self) -> Table:  # type: ignore[override]
        return _render_active_bindings_table(self, self._source_screen())

    def _source_screen(self) -> "Screen[object]":
        """Pick the screen whose bindings should be rendered.

        Default = the screen stacked below the modal (the inspector
        screen). Falls back to ``self.screen`` when the stack is
        shallow (e.g. a standalone test pushing the modal directly).
        """
        stack = self.app.screen_stack
        return stack[-2] if len(stack) >= 2 else self.screen


def _render_active_bindings_table(
    widget: BindingsTable, source: "Screen[object]"
) -> Table:
    """Render ``source.active_bindings`` as a Rich table.

    Mirrors the body of :meth:`BindingsTable.render_bindings_table`
    upstream (``textual/widgets/_key_panel.py``) with one change:
    bindings are pulled from ``source`` rather than ``widget.screen``.
    Component styles, key-display, and group titles delegate to
    ``widget`` so styling stays Textual-canonical.
    """
    bindings = source.active_bindings.values()
    key_style = widget.get_component_rich_style("bindings-table--key")
    divider_transparent = (
        widget.get_component_styles("bindings-table--divider").color.a == 0
    )
    table = Table(
        padding=(0, 0),
        show_header=False,
        box=box.SIMPLE if divider_transparent else box.HORIZONTALS,
        border_style=widget.get_component_rich_style("bindings-table--divider"),
    )
    table.add_column("", justify="right")

    header_style = widget.get_component_rich_style("bindings-table--header")
    description_style = widget.get_component_rich_style("bindings-table--description")
    get_key_display = widget.app.get_key_display
    previous_namespace: object = None

    for namespace, _bindings in groupby(bindings, key=itemgetter(0)):
        table_bindings = list(_bindings)
        if not table_bindings:
            continue

        if namespace.BINDING_GROUP_TITLE:
            title = Text(namespace.BINDING_GROUP_TITLE, end="", style=header_style)
            table.add_row("", title)

        action_to_bindings: defaultdict[str, list[Binding]] = defaultdict(list)
        for _ns, binding, _enabled, _tooltip in table_bindings:
            if not binding.system:
                action_to_bindings[binding.action].append(binding)

        for multi_bindings in action_to_bindings.values():
            primary = multi_bindings[0]
            keys_display = " ".join(
                dict.fromkeys(get_key_display(b) for b in multi_bindings)
            )
            description = Text.from_markup(
                primary.description, end="", style=description_style
            )
            if primary.tooltip:
                if primary.description:
                    description.append(" ")
                description.append(primary.tooltip, "dim")
            table.add_row(Text(keys_display, style=key_style), description)

        if namespace != previous_namespace:
            table.add_section()
        previous_namespace = namespace

    return table


class HelpScreen(ModalScreen[None]):
    """Modal listing every active binding via :class:`BindingsTable`.

    Content auto-generates from the screen stacked under the modal --
    adding a binding to :class:`InspectorApp` (or any focused widget
    in its hierarchy) automatically surfaces here. Dismisses on
    ``escape`` via :meth:`textual.screen.Screen.action_dismiss`.
    """

    DEFAULT_CSS: ClassVar[str] = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > #help-scroll {
        max-width: 90%;
        max-height: 90%;
        width: auto;
        height: auto;
        border: thick $primary;
        background: $surface;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    BINDING_GROUP_TITLE: ClassVar[str] = "Help"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-scroll"):
            yield _UnderlyingScreenBindingsTable()
