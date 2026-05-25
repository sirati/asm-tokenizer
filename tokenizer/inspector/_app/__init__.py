"""Textual ``App`` package driving the inspector TUI.

Single concern: re-export the public surface (``InspectorApp`` +
``run_inspector``) from the per-concern submodules. The package
boundary keeps each submodule below the 300 LOC cap and isolates the
textual imports to :mod:`._tree_widget` + :mod:`._application` so the
default ``nix develop`` shell pays the import cost only when the TUI
is actually launched.

Re-exports use PEP 562 ``__getattr__`` so accessing ``InspectorApp`` /
``run_inspector`` / ``_InspectorTree`` defers the ``textual.app`` +
widget imports until they are actually needed. ``_InspectorTree`` is
intentionally absent from :data:`__all__` -- it stays accessible to
the tests that pin the widget by name, but ``from ... import *``
expansion only carries the two public entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = ["InspectorApp", "run_inspector"]


if TYPE_CHECKING:
    from tokenizer.inspector._app._application import (
        InspectorApp,
        run_inspector,
    )
    from tokenizer.inspector._app._tree_widget import _InspectorTree


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export for the inspector ``_app`` package.

    Defers the ``textual`` (+ ``rich``-via-widget) import chain until
    callers actually touch ``InspectorApp`` / ``run_inspector`` /
    ``_InspectorTree``. ``__main__`` reaches the public app entry
    points through this shim; tests pin ``_InspectorTree`` by name and
    so resolve through the same path.
    """
    if name in ("InspectorApp", "run_inspector"):
        from tokenizer.inspector._app._application import (
            InspectorApp,
            run_inspector,
        )

        return {"InspectorApp": InspectorApp, "run_inspector": run_inspector}[name]
    if name == "_InspectorTree":
        from tokenizer.inspector._app._tree_widget import _InspectorTree

        return _InspectorTree
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
