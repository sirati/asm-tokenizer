"""Quit-confirmation modal.

Single concern: prompt the user before exiting the inspector. ``q``
on the App pushes this modal; the user either confirms (dismissing
with ``True``) or cancels (``False`` / Escape).
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


__all__ = ["QuitConfirmScreen"]


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal asking "really quit?" — dismiss True to exit, False to stay."""

    DEFAULT_CSS: ClassVar[str] = """
    QuitConfirmScreen {
        align: center middle;
    }
    QuitConfirmScreen > #quit-body {
        background: $panel;
        border: thick $primary;
        padding: 1 2;
        width: auto;
        height: auto;
    }
    QuitConfirmScreen #quit-buttons {
        align: center middle;
        height: auto;
        padding-top: 1;
    }
    QuitConfirmScreen Button {
        margin: 0 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("alt+a", "accept", "Accept", show=True),
        Binding("alt+c", "cancel", "Cancel", show=False),
        Binding("y", "accept", "Yes", show=False),
        Binding("n", "cancel", "No", show=False),
    ]

    BINDING_GROUP_TITLE: ClassVar[str] = "Quit"

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-body"):
            yield Label("Really quit?")
            with Horizontal(id="quit-buttons"):
                yield Button("Accept", id="quit-accept", variant="primary")
                yield Button("Cancel", id="quit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-accept":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
