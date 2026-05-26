"""Binary switcher subpackage — switch-binary modal + folder picker.

Single concern: present the user a way to pick a different binary
(memmap or CSV) at runtime, tear down the current
:class:`BackendFactory`, and reseed the tree. Submodules:

* :mod:`._provider` — typed discriminator + switch-target dataclass.
* :mod:`._scan` — filesystem-side detection of loadable data folders.
* :mod:`._dialog` — :class:`BinarySwitcherDialog` modal screen.

The package boundary keeps each submodule below the 400 LOC cap and
keeps :mod:`textual`-dependent code behind the PEP 562 lazy
``__getattr__`` — importing this subpackage costs only the pure-data
types until the dialog actually opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = [
    "BinarySwitcherDialog",
    "LoaderProvider",
    "SwitchTarget",
    "open_binary_switcher",
]


if TYPE_CHECKING:
    from tokenizer.inspector._app._application import InspectorApp

    from ._dialog import BinarySwitcherDialog
    from ._provider import LoaderProvider, SwitchTarget


def open_binary_switcher(app: "InspectorApp") -> None:
    """Push the :class:`BinarySwitcherDialog` modal against the App state.

    The dialog reads the App's current paths + provider so the
    "memmap" / "output.csv" trees seed against the right initial
    directory. Result handling is dispatched to the App-side
    :meth:`InspectorApp._on_binary_switcher_dismissed` callback once
    the dialog dismisses.
    """
    from ._dialog import BinarySwitcherDialog

    dialog = BinarySwitcherDialog(
        current_memmap_path=getattr(app, "_current_memmap_path", None),
        current_csv_path=getattr(app, "_current_csv_path", None),
        current_provider=getattr(app, "_current_provider", None),
    )
    app.push_screen(
        dialog, lambda r: _on_binary_switcher_dismissed(app, r)
    )


def _on_binary_switcher_dismissed(app: "InspectorApp", target) -> None:
    """Handle the dialog's dismiss value.

    Phase 2 lands the dialog only — the actual factory swap arrives in
    a follow-up commit. For now the dismiss target is surfaced as a
    notification so the user sees what they picked and the dialog
    contract is observable end-to-end.
    """
    if target is None:
        return
    app.notify(
        f"Selected: provider={target.provider.value}, "
        f"path={target.path}, binary={target.binary}",
        title="Switch binary",
    )


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export for the Textual-dependent dialog modules."""
    if name in ("LoaderProvider", "SwitchTarget"):
        from ._provider import LoaderProvider, SwitchTarget

        return {
            "LoaderProvider": LoaderProvider,
            "SwitchTarget": SwitchTarget,
        }[name]
    if name == "BinarySwitcherDialog":
        from ._dialog import BinarySwitcherDialog

        return BinarySwitcherDialog
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
