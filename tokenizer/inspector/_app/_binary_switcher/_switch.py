"""App-side switch action: factory swap + tree reseed + state preservation.

Single concern: own the runtime-switch-binary entry point + the
``BackendFactory`` swap that happens after the user confirms. The
dialog is data-in / data-out — it yields a :class:`SwitchTarget` (or
``None`` on cancel) and this module does everything else.

State preservation: ``app._order_config`` is carried across the swap
verbatim. Tree expand-state is dropped (the new binary's function
list is a fresh seed, so retaining tree cursors makes no sense). This
matches the "Proceed? Work will not be saved" confirmation the user
sees before the swap.

Opener dispatch lives here because the opener-table is a string-typed
lookup against :class:`LoaderProvider` values — co-locating it with
the switch action keeps the discriminator -> opener mapping in one
place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from tokenizer.inspector._render._backend_factory import (
    make_batch_decode_factory,
    make_ftl_factory,
)
from tokenizer.inspector._render._protocol import BackendFactory

from ._provider import LoaderProvider, SwitchTarget


if TYPE_CHECKING:
    from tokenizer.inspector._app._application import InspectorApp


__all__ = [
    "open_binary_switcher",
    "perform_switch",
]


# ---------------------------------------------------------------------------
# Opener dispatch (LoaderProvider -> factory builder)
# ---------------------------------------------------------------------------


def _open_memmap(path: Path, binary: Optional[str]) -> BackendFactory:
    """Memmap opener entry point used by the switch action.

    Resolves ``binary=None`` via the canonical
    :mod:`tokenizer.inspector._args` resolver so single-binary
    directories auto-detect; raises :class:`SystemExit` on failure,
    which the dialog surface catches and renders inline.
    """
    from tokenizer.inspector._args import _resolve_binary_memmap

    resolved = _resolve_binary_memmap(path, binary)
    return make_batch_decode_factory(path, resolved)


def _open_csv(path: Path, binary: Optional[str]) -> BackendFactory:
    """CSV opener entry point used by the switch action.

    Mirrors :func:`_open_memmap`; uses the CSV-side resolver so
    ``binary=None`` succeeds for single-binary CSV directories.
    """
    from tokenizer.inspector._args import _resolve_binary_csv

    resolved = _resolve_binary_csv(path, binary)
    return make_ftl_factory(path, resolved)


_OPENERS: dict[LoaderProvider, Callable[[Path, Optional[str]], BackendFactory]] = {
    LoaderProvider.MEMMAP: _open_memmap,
    LoaderProvider.CSV: _open_csv,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def open_binary_switcher(app: "InspectorApp") -> None:
    """Push the :class:`BinarySwitcherDialog` modal against the App state.

    The dialog reads the App's current paths + provider so the
    "memmap" / "output.csv" trees seed against the right initial
    directory. Result handling is dispatched to
    :func:`_on_binary_switcher_dismissed` once the dialog dismisses.
    """
    from ._dialog import BinarySwitcherDialog

    dialog = BinarySwitcherDialog(
        current_memmap_path=app._current_memmap_path,
        current_csv_path=app._current_csv_path,
        current_provider=app._current_provider,
    )
    app.push_screen(dialog, lambda r: _on_binary_switcher_dismissed(app, r))


def _on_binary_switcher_dismissed(
    app: "InspectorApp", target: Optional[SwitchTarget]
) -> None:
    """Confirm + perform the swap when ``target`` is non-``None``.

    The dialog already prompted the user via its own
    :class:`textual.screen.ModalScreen` confirmation flow — at this
    point the contract is "the user committed to the switch". The
    actual swap goes through :func:`perform_switch`.
    """
    if target is None:
        return
    perform_switch(app, target)


def perform_switch(app: "InspectorApp", target: SwitchTarget) -> None:
    """Tear down the current factory + open ``target`` + reseed the tree.

    Preserves ``app._order_config`` verbatim across the swap so the
    user's grouping / sort choices survive. Tree cursors + expand
    state are dropped (the new binary's functions are fresh seeds).

    On opener failure (missing sidecar, malformed CSV, ...), the
    exception propagates to the dispatcher so the user sees the error
    inline; the previous factory remains closed and the App switches
    to an empty handle list — the user can retry via the same modal.
    """
    opener = _OPENERS[target.provider]
    new_factory = opener(target.path, target.binary)

    # Tear down the previous factory's shared state.
    try:
        app._factory.close()
    except Exception:  # pragma: no cover -- defensive
        # The new factory is already constructed; surfacing a
        # teardown error here would leak the new factory. Best-effort
        # close + continue with the swap.
        app._log.exception(
            "factory close failed during binary switch; continuing"
        )

    app._factory = new_factory
    app._current_provider = target.provider
    if target.provider is LoaderProvider.MEMMAP:
        app._current_memmap_path = target.path
    else:
        app._current_csv_path = target.path

    # Clear any pending auto-expand state from the previous binary.
    app._pending_auto_expand.clear()

    # Reseed the tree with the new factory's handles.
    _reseed_tree(app)


def _reseed_tree(app: "InspectorApp") -> None:
    """Replace every root :class:`FunctionNode` tree entry with the new
    factory's handles.

    Walks the tree's root, removes all children, then re-adds one entry
    per handle the new factory published. Mirrors
    :meth:`InspectorApp.compose`'s seed loop so the rebuilt root looks
    identical to a fresh app start.
    """
    from tokenizer.inspector._app._tree_widget import _InspectorTree
    from tokenizer.inspector._app._labels import _compose_label

    tree = app.query_one("#tree", _InspectorTree)
    tree.root.remove_children()
    for handle in app._factory.handles:
        fn_node = app._build_root_function_node(handle)
        tree.root.add(
            _compose_label(fn_node),
            data=fn_node,
            allow_expand=True,
        )
    if not tree.root.is_expanded:
        tree.root.expand()
