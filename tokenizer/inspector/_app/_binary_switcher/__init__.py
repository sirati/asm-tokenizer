"""Binary switcher subpackage — switch-binary modal + folder picker.

Single concern: present the user a way to pick a different binary
(memmap or CSV) at runtime, tear down the current
:class:`BackendFactory`, and reseed the tree. Submodules:

* :mod:`._provider` — typed discriminator + switch-target dataclass.
* :mod:`._scan` — filesystem-side detection of loadable data folders.
* :mod:`._dialog` — :class:`BinarySwitcherDialog` modal screen.
* :mod:`._folder_picker` — :class:`FolderPickerDialog` modal screen.
* :mod:`._switch` — App-side switch action (factory swap +
  ``_order_config`` preservation).

The package boundary keeps each submodule below the 400 LOC cap and
keeps :mod:`textual`-dependent code behind the PEP 562 lazy
``__getattr__`` — importing this subpackage costs only the pure-data
types until the dialog actually opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = [
    "BinarySwitcherDialog",
    "FolderPickerDialog",
    "LoaderProvider",
    "SwitchTarget",
    "open_binary_switcher",
    "perform_switch",
]


if TYPE_CHECKING:
    from ._dialog import BinarySwitcherDialog
    from ._folder_picker import FolderPickerDialog
    from ._provider import LoaderProvider, SwitchTarget
    from ._switch import open_binary_switcher, perform_switch


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
    if name == "FolderPickerDialog":
        from ._folder_picker import FolderPickerDialog

        return FolderPickerDialog
    if name in ("open_binary_switcher", "perform_switch"):
        from ._switch import open_binary_switcher, perform_switch

        return {
            "open_binary_switcher": open_binary_switcher,
            "perform_switch": perform_switch,
        }[name]
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
