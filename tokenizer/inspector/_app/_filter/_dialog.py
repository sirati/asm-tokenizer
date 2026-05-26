"""Filter modal: per-axis :class:`SelectionList` for value enable/disable.

Single concern: present a :class:`textual.screen.ModalScreen` whose
body is a vertical stack of one :class:`SelectionList[str]` per
candidate axis, where every distinct value seen across the loaded
variants gets one checkable row. ``space`` toggles the checkbox
(SelectionList built-in); ``ctrl+s`` / ``alt+a`` OR the ``[Accept]``
:class:`Button` dismisses with :class:`FilterAccepted(config)`;
``escape`` / ``alt+c`` dismisses with :class:`FilterCancelled()`
(mirrors :class:`OrderDialog` so the two modals share keybindings).

Construction: the dialog takes a ``Mapping[AxisDescriptor, tuple[str, ...]]``
(axis -> seen values, built by :mod:`._values.discover_all_axis_values`)
plus an optional prior :class:`FilterConfig`. Rows are seeded with the
prior config: a value present in the prior disabled set lands UNCHECKED
(filtered out); every other value lands CHECKED (visible). New
axes/values land CHECKED by default.

Accept path collects the UNCHECKED values per axis -- those are the
"disabled" values that go into the new :class:`FilterConfig`.

Min-one-selected invariant: each per-axis :class:`SelectionList` is a
:class:`_MinOneSelectionList` subclass that refuses the toggle which
would drop the last selected value. An axis with zero values checked
would mean "filter accepts nothing on this axis", which makes the
filter useless; the subclass keeps the last selection pinned so the
user can never reach that degenerate state.

Focus: :meth:`on_mount` focuses the first axis :class:`SelectionList`
so the dialog opens keyboard-ready (no extra Tab press).
"""

from __future__ import annotations

from typing import ClassVar, Mapping, Optional, Sequence

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, SelectionList
from textual.widgets._selection_list import Selection

from .._order import AxisDescriptor
from ._config import (
    FilterAccepted,
    FilterCancelled,
    FilterConfig,
    FilterResult,
)


__all__ = ["FilterDialog"]


# Per-axis SelectionList id prefix; the accept path queries each one by
# composed id to recover its checked-state list.
_AXIS_LIST_ID_PREFIX = "filter-axis-list-"


class _MinOneSelectionList(SelectionList[str]):
    """:class:`SelectionList` that refuses to deselect the last checked row.

    Single concern: pin the "at least one value remains selected" axis
    invariant at the toggle boundary. A deselect that would drop the
    selected count to zero is silently rejected (the option stays
    checked); every other toggle behaves exactly like the upstream
    widget.

    The override targets :meth:`SelectionList._toggle` -- the single
    code path the spacebar key, click-on-row, and programmatic
    :meth:`SelectionList.toggle` all funnel through. Overriding here
    (rather than fielding the public :class:`SelectionList.SelectionToggled`
    message + reverting) means the rejected toggle never produces a
    transient zero-selected state visible to subscribers.
    """

    def _toggle(self, value: str) -> bool:  # type: ignore[override]
        if value in self._selected and len(self._selected) == 1:
            # Dropping the last selected value would leave the axis
            # with zero checked rows; reject the toggle (no-op).
            return False
        return super()._toggle(value)


class FilterDialog(ModalScreen[FilterResult]):
    """Modal screen: per-axis value enable/disable.

    Dismiss paths:

    * ``escape`` -> :class:`FilterCancelled`.
    * ``ctrl+s`` -> :class:`FilterAccepted(config)`.
    * :class:`Button` press with ``id="filter-accept"`` -> same as ``ctrl+s``.

    The result sum type is pattern-matched by the App's
    :func:`tokenizer.inspector._app._filter_hooks.on_filter_dialog_dismissed`
    callback.

    Axes with no discovered values (empty value tuple) are skipped:
    rendering an empty SelectionList would invite a confused user to
    "toggle nothing". The skipped axis still exists in the candidate
    set; if a later expand surfaces values, the next ``f``-press will
    show it.
    """

    CSS: ClassVar[str] = """
    FilterDialog {
        align: center middle;
    }
    FilterDialog > #filter-body {
        background: $panel;
        border: tall $accent;
        padding: 1 2;
        width: 80;
        height: 1fr;
        margin: 3 0;
    }
    FilterDialog #filter-scroll {
        height: 1fr;
    }
    FilterDialog .filter-axis-section {
        height: auto;
        margin-bottom: 1;
    }
    FilterDialog .filter-axis-header {
        text-style: bold;
    }
    FilterDialog .filter-axis-list {
        height: auto;
        max-height: 8;
    }
    FilterDialog #filter-buttons {
        height: 3;
        align-horizontal: right;
    }
    """

    BINDING_GROUP_TITLE: ClassVar[str] = "Filter dialog"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "[u]C[/]ancel", show=True),
        Binding("alt+c", "cancel", "[u]C[/]ancel", show=True),
        Binding("ctrl+s", "accept", "[u]A[/]ccept", show=True),
        Binding("alt+a", "accept", "[u]A[/]ccept", show=True),
    ]

    def __init__(
        self,
        *,
        axis_values: Mapping[AxisDescriptor, Sequence[str]],
        prior_config: Optional[FilterConfig] = None,
    ) -> None:
        super().__init__()
        # Materialise the axis-value tuples so each axis maps to a
        # stable ordered sequence (the dialog renders rows in this
        # order and the accept path keys back to the same axes).
        self._axes_in_order: tuple[AxisDescriptor, ...] = tuple(axis_values.keys())
        self._values_by_axis: dict[AxisDescriptor, tuple[str, ...]] = {
            axis: tuple(values) for axis, values in axis_values.items()
        }
        # Per-axis disabled value sets from the prior config (or empty).
        self._prior_disabled: dict[AxisDescriptor, frozenset[str]] = {}
        if prior_config is not None:
            for axis in self._axes_in_order:
                self._prior_disabled[axis] = prior_config.disabled_for(axis)

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        # Button row lives INSIDE the scrollable area so the dialog
        # adapts to any terminal height: at large heights the
        # ``height: auto`` outer body shrinks to its content and the
        # buttons sit just below the last axis row (no scroll needed);
        # at small heights the scroll caps at the modal max-height and
        # the buttons scroll into view rather than getting clipped at
        # the dialog's bottom edge (the latter is what happens when the
        # buttons live outside the scroll on a viewport too short to
        # hold both).
        with Vertical(id="filter-body"):
            with VerticalScroll(id="filter-scroll"):
                for index, axis in enumerate(self._axes_in_order):
                    values = self._values_by_axis[axis]
                    if not values:
                        continue
                    disabled_prior = self._prior_disabled.get(axis, frozenset())
                    # Re-seat the prior-disabled values to land CHECKED
                    # when EVERY value would otherwise land unchecked:
                    # an axis whose prior config disabled every value is
                    # an out-of-band state the min-one invariant would
                    # not let the user reach via the dialog. We snap it
                    # back to "all checked" so the user has a usable
                    # starting point.
                    if disabled_prior and disabled_prior.issuperset(values):
                        disabled_prior = frozenset()
                    with Vertical(classes="filter-axis-section"):
                        yield Label(
                            axis.label, classes="filter-axis-header"
                        )
                        yield _MinOneSelectionList(
                            *(
                                Selection(
                                    prompt=value,
                                    value=value,
                                    initial_state=(value not in disabled_prior),
                                )
                                for value in values
                            ),
                            id=_axis_list_id(index),
                            classes="filter-axis-list",
                        )
                with Horizontal(id="filter-buttons"):
                    yield Button(
                        "[u]A[/]ccept", id="filter-accept", variant="primary"
                    )
                    yield Button("[u]C[/]ancel", id="filter-cancel")

    # --- focus -----------------------------------------------------

    def on_mount(self) -> None:
        """Land keyboard focus on the first axis :class:`SelectionList`.

        The default ModalScreen focus path lands on the first focusable
        widget in compose order. Stating the focus explicitly here
        survives any future re-ordering of the compose tree (e.g.
        toolbar-style accept buttons mounted above the body) and gives
        the user a keyboard-ready dialog without an extra Tab press.

        When every axis has zero values (no expanded function has
        reported a row yet), the dialog renders no SelectionList at all
        -- in that case the loop falls through and the default focus
        path handles it (the Accept button takes focus and ``alt+a`` /
        ``alt+c`` still work via the screen-level BINDINGS).
        """
        for index in range(len(self._axes_in_order)):
            try:
                sel_list = self.query_one(
                    f"#{_axis_list_id(index)}", _MinOneSelectionList
                )
            except Exception:
                continue
            sel_list.focus()
            return

    # --- actions ---------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(FilterCancelled())

    def action_accept(self) -> None:
        self.dismiss(FilterAccepted(config=self._build_config()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "filter-accept":
            self.action_accept()
        elif event.button.id == "filter-cancel":
            self.action_cancel()

    # --- accept-time config construction --------------------------

    def _build_config(self) -> FilterConfig:
        disabled: dict[AxisDescriptor, frozenset[str]] = {}
        for index, axis in enumerate(self._axes_in_order):
            values = self._values_by_axis[axis]
            if not values:
                continue
            sel_list = self.query_one(
                f"#{_axis_list_id(index)}", SelectionList
            )
            checked_set = set(sel_list.selected)
            disabled_values = frozenset(
                value for value in values if value not in checked_set
            )
            if disabled_values:
                disabled[axis] = disabled_values
        return FilterConfig.build(disabled)


def _axis_list_id(index: int) -> str:
    """Stable per-axis SelectionList id (used by compose + accept lookup)."""
    return f"{_AXIS_LIST_ID_PREFIX}{index}"
