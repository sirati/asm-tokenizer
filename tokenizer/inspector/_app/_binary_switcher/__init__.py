"""Binary switcher subpackage — switch-binary modal + folder picker.

Single concern: present the user a way to pick a different binary
(memmap or CSV) at runtime, tear down the current
:class:`BackendFactory`, and reseed the tree.

This shell only carries the :func:`open_binary_switcher` entry point
the App-level menu binding wires against. Subsequent submodules
(provider tree, folder picker, switch action) land in follow-up
commits — keeping each below the 400 LOC cap and isolating the
textual-dependent dialog code behind PEP 562 lazy ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = [
    "open_binary_switcher",
]


if TYPE_CHECKING:
    from tokenizer.inspector._app._application import InspectorApp


def open_binary_switcher(app: "InspectorApp") -> None:
    """Stub entry point for the binary-switcher modal.

    Phase 1 lands the menu bar + binding shell only — the dialog is
    not yet implemented. The click / hotkey still resolves through
    :meth:`InspectorApp.action_open_binary_switcher` so the menu is
    fully testable; this stub surfaces a notification so the user
    sees something happen, and a follow-up commit replaces this body
    with the real dialog push.
    """
    app.notify(
        "Binary switcher coming soon — pick via --memmap-dir / --csv-dir for now.",
        title="Switch binary",
    )
