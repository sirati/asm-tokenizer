"""``python -m tokenizer.inspector`` entry point.

Single concern: parse CLI args via :mod:`._args`, build a
:class:`BackendFactory` for the chosen provider (memmap or stage-1
CSV) against the unified ``PATH`` argument via the openers in
:mod:`tokenizer.inspector._render._backend_factory`, and hand the
factory off to the Textual app entry
(:func:`tokenizer.inspector._app.run_inspector`).

The factory dispatch is a typed dict-of-callable keyed by the
:class:`LoaderProvider` enum carried on the parsed namespace: no
string-typed flag, no inline ``if/elif`` ladder.

``_app`` is the only module that imports :mod:`textual`; this file
imports it lazily inside :func:`main` so unit tests that import
:mod:`tokenizer.inspector.__main__` (e.g. to exercise argparse) don't
trip the textual-free default shell rule.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from tokenizer.inspector._app._binary_switcher._provider import LoaderProvider
from tokenizer.inspector._render._backend_factory import (
    make_batch_decode_factory,
    make_ftl_factory,
)
from tokenizer.inspector._render._protocol import BackendFactory

from ._args import parse_args

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-provider opener return type
# ---------------------------------------------------------------------------


@dataclass
class _OpenedBackend:
    """Bundle yielded by every per-provider opener.

    ``factory`` is the live :class:`BackendFactory`; ``stack`` owns
    its shutdown (the caller drives ``with stack:``). ``path`` is
    the source directory used for the inspector log path AND seeds
    the binary-switcher dialog's "current path" indicator.
    ``provider`` is the typed discriminator the App stores so the
    switcher dialog renders the right provider as ``[current]``.
    """

    factory: BackendFactory
    stack: contextlib.ExitStack
    path: Path
    provider: LoaderProvider


# ---------------------------------------------------------------------------
# Openers
# ---------------------------------------------------------------------------


def _open_memmap_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Open the memmap-mode backend.

    The factory owns its :class:`BinarySession` lifetime; we register
    only ``factory.close`` so the caller's ``with stack:`` block
    drives the single shutdown hook (mirrors the CSV-mode opener).
    """
    path: Path = ns.path
    binary_name: str = ns.binary
    _log.info(
        "inspector: opening memmap backend for binary=%s in %s",
        binary_name,
        path,
    )
    factory = make_batch_decode_factory(path, binary_name)
    stack = contextlib.ExitStack()
    stack.callback(factory.close)
    return _OpenedBackend(
        factory=factory,
        stack=stack,
        path=path,
        provider=LoaderProvider.MEMMAP,
    )


def _open_csv_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Open the FTL-CSV mode backend.

    Constructs the per-binary :class:`CsvIndex`-backed factory; its
    :meth:`close` is registered on the returned ExitStack so callers
    release the parsed-record + vocab cache via ``with stack:``.
    """
    path: Path = ns.path
    binary_name: str = ns.binary
    _log.info(
        "inspector: opening csv backend for binary=%s in %s",
        binary_name,
        path,
    )
    factory = make_ftl_factory(path, binary_name)
    stack = contextlib.ExitStack()
    stack.callback(factory.close)
    return _OpenedBackend(
        factory=factory,
        stack=stack,
        path=path,
        provider=LoaderProvider.CSV,
    )


# Typed dispatch: ``ns.provider`` drives the opener. Mirrors the
# ``MetadataLookup`` dict-of-callable pattern -- no inline if/elif at
# this layer.
_OPENERS: Dict[LoaderProvider, Callable[[argparse.Namespace], _OpenedBackend]] = {
    LoaderProvider.MEMMAP: _open_memmap_backend,
    LoaderProvider.CSV: _open_csv_backend,
}


def _open_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Dispatch to the per-provider opener.

    The argparse mutex group guarantees ``ns.provider`` is set to a
    :class:`LoaderProvider`; the matching key in :data:`_OPENERS`
    selects the concrete opener.
    """
    return _OPENERS[ns.provider](ns)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)

    # Open the backend BEFORE importing ``_app`` so any discovery-side
    # error (no surviving CSVs, missing memmap sidecar) surfaces
    # without requiring textual on the default ``nix develop`` shell.
    opened = _open_backend(ns)

    # Lazy import: ``_app`` pulls in :mod:`textual`. Importing it at
    # module-level would block ``python -m tokenizer.inspector --help``
    # in any environment without textual installed.
    from ._app import run_inspector

    with opened.stack:
        log_path = opened.path / ".tui-inspector.log"
        return run_inspector(
            factory=opened.factory,
            log_path=log_path,
            path=opened.path,
            provider=opened.provider,
        )


if __name__ == "__main__":
    sys.exit(main())
