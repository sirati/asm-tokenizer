"""Order modal: reorderable + checkboxable :class:`SelectionList`.

Single concern: present a :class:`textual.screen.ModalScreen` whose
body is a :class:`SelectionList[AxisDescriptor]`. ``space`` toggles
the checkbox (SelectionList built-in); ``alt+up`` / ``alt+down``
reorder the highlighted axis via :meth:`SelectionList.clear_options` +
:meth:`SelectionList.add_options` (cluster B-L4 M4 -- single-call
replacement that preserves selection state). ``ctrl+s`` OR the
``[Accept]`` :class:`Button` dismisses with
:class:`OrderAccepted(config)`; ``escape`` dismisses with
:class:`OrderCancelled()` (cluster #11 / B-L3 H1 -- Enter is shadowed
by SelectionList's check toggle, so we route accept off ctrl+s + a
visible Button).

The dialog is constructed with the candidate axis tuple
(``canonical-5 + EXTRA_META`` discovered by the App layer) + the
existing :class:`OrderConfig` carrying the previous run's ordering +
grouping checks. Reconcile-against-candidates on construction: new
candidates land at the end UNCHECKED; stale candidates drop silently.

Focus: :meth:`on_mount` focuses the axis :class:`SelectionList` so
the dialog opens keyboard-ready (no extra Tab press), mirroring the
:class:`FilterDialog` focus-on-mount pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Sequence

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, SelectionList
from textual.widgets._selection_list import Selection

from ._axes import (
    AxisDescriptor,
    OrderAccepted,
    OrderCancelled,
    OrderConfig,
    OrderResult,
)


__all__ = [
    "OrderDialog",
    "_ReorderableSelectionList",
]


# ---------------------------------------------------------------------------
# Internal selection-list subclass owning the alt+up/down reorder bindings.
# ---------------------------------------------------------------------------


class _ReorderableSelectionList(SelectionList[int]):
    """:class:`SelectionList` keyed on a stable int handle per axis.

    Axes carry value-equal :class:`AxisDescriptor`s that would also
    work as the SelectionList payload, but :meth:`SelectionList.toggle`
    + ``_selected`` track values by dict key -- keying on a per-row
    int handle lets the dialog reorder rows without the SelectionList
    losing track of which handles are selected (the handle stays
    constant across the rebuild).

    The dialog owns the ``handle -> AxisDescriptor`` lookup so accept
    can recover the user's pure-data ordering.

    Adds ``alt+up`` / ``alt+down`` to swap the highlighted row with
    its neighbour. The swap path:

    1. Capture the current ``selected`` set (handles) + the ordered
       handle list off ``self._values``.
    2. Swap positions in the handle list.
    3. :meth:`clear_options` (resets ``_selected`` + ``_values``) then
       :meth:`add_options` re-adds with the previous-state initial-
       state tuple per row.
    4. Move the highlight to the swapped row's new position.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("shift+up", "move_up", "Move up", show=False),
        Binding("shift+down", "move_down", "Move down", show=False),
    ]

    def action_move_up(self) -> None:
        self._swap(offset=-1)

    def action_move_down(self) -> None:
        self._swap(offset=1)

    def _swap(self, *, offset: int) -> None:
        current = self.highlighted
        if current is None:
            return
        target = current + offset
        if target < 0 or target >= self.option_count:
            return

        # Snapshot the current handle list in display order, plus the
        # selected-handle set, BEFORE the rebuild clears them.
        ordered_handles: list[int] = [
            self.get_option_at_index(i).value
            for i in range(self.option_count)
        ]
        ordered_labels: list[object] = [
            self.get_option_at_index(i).prompt
            for i in range(self.option_count)
        ]
        selected_set = set(self.selected)

        # Swap rows in the snapshot.
        ordered_handles[current], ordered_handles[target] = (
            ordered_handles[target],
            ordered_handles[current],
        )
        ordered_labels[current], ordered_labels[target] = (
            ordered_labels[target],
            ordered_labels[current],
        )

        # Single-call replacement: clear + add (cluster B-L4 M4).
        self.clear_options()
        self.add_options(
            [
                Selection(prompt=label, value=handle, initial_state=(handle in selected_set))
                for handle, label in zip(ordered_handles, ordered_labels)
            ]
        )
        # Move the highlight to the swapped row's new position.
        self.highlighted = target


# ---------------------------------------------------------------------------
# Modal screen
# ---------------------------------------------------------------------------


class OrderDialog(ModalScreen[OrderResult]):
    """Modal screen: reorder + check the candidate axes.

    Dismiss paths:

    * ``escape`` -> :class:`OrderCancelled`.
    * ``ctrl+s`` -> :class:`OrderAccepted(config)`.
    * :class:`Button` press with ``id="accept"`` -> same as ``ctrl+s``.

    The result sum type is pattern-matched by the App's
    :meth:`InspectorApp._on_order_dialog_dismissed` callback.
    """

    CSS: ClassVar[str] = """
    OrderDialog {
        align: center middle;
    }
    OrderDialog > #order-body {
        background: $panel;
        border: tall $accent;
        padding: 1 2;
        width: 60;
        height: auto;
        max-height: 80%;
    }
    OrderDialog #order-scroll {
        height: auto;
        max-height: 24;
    }
    OrderDialog #order-list {
        height: auto;
    }
    OrderDialog #order-hint {
        color: $text-muted;
        padding: 0 1;
    }
    OrderDialog #order-buttons {
        height: 3;
        align-horizontal: right;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "[u]C[/]ancel", show=True),
        Binding("alt+c", "cancel", "[u]C[/]ancel", show=True),
        Binding("ctrl+s", "accept", "[u]A[/]ccept", show=True),
        Binding("alt+a", "accept", "[u]A[/]ccept", show=True),
    ]

    def __init__(
        self,
        *,
        candidate_axes: Sequence[AxisDescriptor],
        prior_config: Optional[OrderConfig] = None,
    ) -> None:
        """Build the dialog from ``candidate_axes`` + ``prior_config``.

        Reconcile-against-candidates:

        * Axes in ``prior_config.ordered_axes`` AND in ``candidate_axes``
          land first, in their previous order, with their previous
          grouping check preserved.
        * Axes in ``candidate_axes`` but NOT in the prior config land at
          the end, unchecked.
        * Axes in the prior config but NOT in ``candidate_axes`` drop
          silently.
        """
        super().__init__()
        candidate_set = set(candidate_axes)
        prior_ordered = (
            prior_config.ordered_axes if prior_config is not None else ()
        )
        prior_grouping = (
            prior_config.grouping_axes if prior_config is not None else frozenset()
        )

        ordered: list[AxisDescriptor] = []
        seen: set[AxisDescriptor] = set()
        for axis in prior_ordered:
            if axis in candidate_set and axis not in seen:
                ordered.append(axis)
                seen.add(axis)
        for axis in candidate_axes:
            if axis not in seen:
                ordered.append(axis)
                seen.add(axis)

        self._axes_in_order: tuple[AxisDescriptor, ...] = tuple(ordered)
        # Per-row stable int handle the SelectionList tracks.
        self._handle_to_axis: dict[int, AxisDescriptor] = {
            i: axis for i, axis in enumerate(self._axes_in_order)
        }
        self._initial_checked: frozenset[int] = frozenset(
            i for i, axis in self._handle_to_axis.items()
            if axis in prior_grouping
        )

    # --- compose ---------------------------------------------------

    def compose(self) -> ComposeResult:
        sel_list = _ReorderableSelectionList(
            *(
                Selection(
                    prompt=axis.label,
                    value=handle,
                    initial_state=(handle in self._initial_checked),
                )
                for handle, axis in self._handle_to_axis.items()
            ),
            id="order-list",
        )
        # Button row lives INSIDE the scrollable area so the dialog
        # adapts to any terminal height: at large heights the
        # ``height: auto`` outer body shrinks to its content and the
        # buttons sit just below the list (no scroll needed); at small
        # heights the scroll caps at the modal max-height and the
        # buttons scroll into view rather than getting clipped at the
        # dialog's bottom edge.
        with Vertical(id="order-body"):
            with VerticalScroll(id="order-scroll"):
                yield sel_list
                yield Label("Ordering: SHIFT + ↓↑", id="order-hint")
                with Horizontal(id="order-buttons"):
                    yield Button("[u]A[/]ccept", id="accept", variant="primary")
                    yield Button("[u]C[/]ancel", id="cancel")

    # --- focus -----------------------------------------------------

    def on_mount(self) -> None:
        """Land keyboard focus on the axis :class:`SelectionList`.

        The dialog's primary interaction surface is the per-axis
        checklist + reorder widget; focusing it on mount gives the
        user a keyboard-ready dialog (no extra Tab press) and mirrors
        the :class:`FilterDialog` focus-on-mount pattern.

        When the candidate-axes tuple is empty the SelectionList
        renders no rows but the widget still mounts -- focusing it is
        still the right call (the screen-level ``alt+a`` / ``alt+c``
        bindings stay live).
        """
        sel_list = self.query_one("#order-list", _ReorderableSelectionList)
        sel_list.focus()

    # --- actions ---------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(OrderCancelled())

    def action_accept(self) -> None:
        self.dismiss(OrderAccepted(config=self._build_config()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "accept":
            self.action_accept()
        elif event.button.id == "cancel":
            self.action_cancel()

    # --- accept-time config construction --------------------------

    def _build_config(self) -> OrderConfig:
        sel_list = self.query_one("#order-list", _ReorderableSelectionList)
        ordered_handles: list[int] = [
            sel_list.get_option_at_index(i).value
            for i in range(sel_list.option_count)
        ]
        selected_handles: frozenset[int] = frozenset(sel_list.selected)

        ordered_axes = tuple(
            self._handle_to_axis[handle] for handle in ordered_handles
        )
        grouping_axes = frozenset(
            self._handle_to_axis[handle] for handle in selected_handles
        )
        return OrderConfig(
            ordered_axes=ordered_axes, grouping_axes=grouping_axes
        )
