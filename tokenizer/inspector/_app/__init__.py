"""Textual ``App`` package driving the inspector TUI.

Single concern: re-export the public surface (``InspectorApp`` +
``run_inspector``) from the per-concern submodules. The package
boundary keeps each submodule below the 300 LOC cap and isolates the
textual imports to :mod:`._tree_widget` + :mod:`._application` so the
default ``nix develop`` shell pays the import cost only when the TUI
is actually launched.

Re-exports are eager today; a PEP 562 ``__getattr__`` shim replaces
them in a follow-up step so that callers who only need ``run_inspector``
do not trigger the ``textual.app`` import indirectly through this
package's ``__init__``.
"""

from __future__ import annotations

from tokenizer.inspector._app._application import InspectorApp, run_inspector
from tokenizer.inspector._app._tree_widget import _InspectorTree


__all__ = ["InspectorApp", "run_inspector"]
