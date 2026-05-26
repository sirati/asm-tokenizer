"""``python -m tokenizer.inspector`` entry point.

Single concern: parse CLI args via :mod:`._args`, build a
:class:`BackendFactory` for the chosen source (``--memmap-dir`` or
``--csv-dir``) via the openers in
:mod:`tokenizer.inspector._render._backend_factory`, and hand the
factory off to the Textual app entry
(:func:`tokenizer.inspector._app.run_inspector`).

The factory dispatch is a typed dict-of-callable: no string-typed
``--backend`` flag, no inline ``if/elif`` ladder -- which source the
user picked is the discriminator that the openers consume.

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

from tokenizer.inspector._render._backend_factory import (
    make_batch_decode_factory,
    make_ftl_factory,
)
from tokenizer.inspector._render._protocol import BackendFactory

from ._args import parse_args

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source opener return type
# ---------------------------------------------------------------------------


@dataclass
class _OpenedBackend:
    """Bundle yielded by every per-source opener.

    ``factory`` is the live :class:`BackendFactory`; ``stack`` owns
    its shutdown (the caller drives ``with stack:``). ``log_dir`` is
    the source directory used for the inspector log path.
    ``memmap_path`` / ``csv_path`` carry the source path so the
    binary-switcher dialog knows which provider was opened (exactly
    one is set; the other is ``None``).
    """

    factory: BackendFactory
    stack: contextlib.ExitStack
    log_dir: Path
    memmap_path: Path | None = None
    csv_path: Path | None = None


# ---------------------------------------------------------------------------
# Openers
# ---------------------------------------------------------------------------


def _open_memmap_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Open the memmap-mode backend.

    The factory owns its :class:`BinarySession` lifetime; we register
    only ``factory.close`` so the caller's ``with stack:`` block
    drives the single shutdown hook (mirrors the CSV-mode opener).
    """
    memmap_dir: Path = ns.memmap_dir
    binary_name: str = ns.binary
    _log.info(
        "inspector: opening memmap backend for binary=%s in %s",
        binary_name,
        memmap_dir,
    )
    factory = make_batch_decode_factory(memmap_dir, binary_name)
    stack = contextlib.ExitStack()
    stack.callback(factory.close)
    return _OpenedBackend(
        factory=factory,
        stack=stack,
        log_dir=memmap_dir,
        memmap_path=memmap_dir,
    )


def _open_csv_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Open the FTL-CSV mode backend.

    Constructs the per-binary :class:`CsvIndex`-backed factory; its
    :meth:`close` is registered on the returned ExitStack so callers
    release the parsed-record + vocab cache via ``with stack:``.
    """
    csv_dir: Path = ns.csv_dir
    binary_name: str = ns.binary
    _log.info(
        "inspector: opening csv backend for binary=%s in %s",
        binary_name,
        csv_dir,
    )
    factory = make_ftl_factory(csv_dir, binary_name)
    stack = contextlib.ExitStack()
    stack.callback(factory.close)
    return _OpenedBackend(
        factory=factory,
        stack=stack,
        log_dir=csv_dir,
        csv_path=csv_dir,
    )


# Typed dispatch: which mutex-group flag is set drives the opener.
# Mirrors the ``MetadataLookup`` dict-of-callable pattern -- no
# inline if/elif at this layer.
_OPENERS: Dict[str, Callable[[argparse.Namespace], _OpenedBackend]] = {
    "memmap_dir": _open_memmap_backend,
    "csv_dir": _open_csv_backend,
}


def _open_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Dispatch to the per-source opener.

    The argparse mutex group guarantees exactly one of
    ``ns.memmap_dir`` / ``ns.csv_dir`` is set; the matching key in
    :data:`_OPENERS` selects the concrete opener.
    """
    for key, opener in _OPENERS.items():
        if getattr(ns, key) is not None:
            return opener(ns)
    raise SystemExit(
        "internal error: argparse mutex group reported no source flag "
        "set; this should be unreachable."
    )


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
        log_path = opened.log_dir / ".tui-inspector.log"
        return run_inspector(
            factory=opened.factory,
            log_path=log_path,
            memmap_path=opened.memmap_path,
            csv_path=opened.csv_path,
        )


if __name__ == "__main__":
    sys.exit(main())
