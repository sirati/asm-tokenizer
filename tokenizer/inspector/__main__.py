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

Phase B2 transitional note: :class:`InspectorApp` still consumes
``(dataset, session)`` for the memmap path; Phase C1 rewires it to
consume ``BackendFactory.handles`` + ``BackendFactory.make`` directly.
Until then, the memmap opener returns its dataset + session alongside
the factory so the call site stays backwards-compatible; the csv-dir
opener raises :class:`NotImplementedError` since the model layer has
no FTL-aware ``expand`` path yet.

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
from typing import Callable, Dict, Optional

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession
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

    ``legacy_dataset`` / ``legacy_session`` are the Phase B2 bridge
    fields: the model layer's :meth:`FunctionNode.expand` still expects
    a :class:`BinarySession` + a :class:`BinaryDataset` (Phase C1
    cuts that over to :class:`BackendFactory.make`). The memmap opener
    populates both; the csv opener leaves them ``None`` and the
    fallback raises before the call site reads them.
    """

    factory: BackendFactory
    stack: contextlib.ExitStack
    log_dir: Path
    legacy_dataset: Optional[BinaryDataset]
    legacy_session: Optional[BinarySession]


# ---------------------------------------------------------------------------
# Openers
# ---------------------------------------------------------------------------


def _open_memmap_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Open the memmap-mode backend.

    The :class:`BinarySession` is entered into the returned ExitStack
    so the caller's ``with`` block drives clean shutdown.
    """
    memmap_dir: Path = ns.memmap_dir
    binary_name: str = ns.binary
    _log.info(
        "inspector: opening memmap backend for binary=%s in %s",
        binary_name,
        memmap_dir,
    )
    factory, dataset, session = make_batch_decode_factory(
        memmap_dir, binary_name
    )
    stack = contextlib.ExitStack()
    stack.enter_context(session)
    stack.callback(factory.close)
    return _OpenedBackend(
        factory=factory,
        stack=stack,
        log_dir=memmap_dir,
        legacy_dataset=dataset,
        legacy_session=session,
    )


def _open_csv_backend(ns: argparse.Namespace) -> _OpenedBackend:
    """Open the FTL-CSV mode backend.

    Constructs the per-binary :class:`CsvIndex`-backed factory; its
    :meth:`close` is registered on the returned ExitStack so callers
    release the parsed-record + vocab cache via ``with stack:``.

    NOTE (Phase B2): the inspector TUI side of FTL is not wired yet;
    the model layer still expects a :class:`BinarySession` (Phase C1
    cuts the TUI over to ``BackendFactory.make``). This opener
    intentionally raises :class:`NotImplementedError` at the CLI
    boundary so users get a clear message instead of a deep
    :class:`AttributeError` from the un-rewired
    :class:`InspectorApp`.
    """
    csv_dir: Path = ns.csv_dir
    binary_name: str = ns.binary
    _log.info(
        "inspector: opening csv backend for binary=%s in %s",
        binary_name,
        csv_dir,
    )
    # Build the factory eagerly so any discovery-side error (no
    # surviving CSVs, vocab load failure) surfaces here rather than
    # under the NotImplementedError. Closed immediately in the
    # finally block to release the just-opened CsvIndex.
    factory = make_ftl_factory(csv_dir, binary_name)
    try:
        raise NotImplementedError(
            "--csv-dir inspector TUI integration pending Phase C1 "
            "(BackendFactory wired into InspectorApp.compose + "
            "FunctionNode.expand). The factory + handles list construct "
            "successfully; only the TUI bridge is missing."
        )
    finally:
        factory.close()


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

    # Open the backend BEFORE importing ``_app`` so the csv-mode
    # NotImplementedError surfaces without requiring textual on the
    # default ``nix develop`` shell.
    opened = _open_backend(ns)

    # Lazy import: ``_app`` pulls in :mod:`textual`. Importing it at
    # module-level would block ``python -m tokenizer.inspector --help``
    # in any environment without textual installed (e.g. the default
    # ``nix develop`` shell), which the plan explicitly forbids.
    from ._app import run_inspector

    with opened.stack:
        log_path = opened.log_dir / ".tui-inspector.log"
        # Phase B2: ``_app`` still consumes ``(dataset, session)``;
        # Phase C1 rewires it to consume ``BackendFactory`` directly.
        # The memmap opener populates the legacy fields above so this
        # call site keeps working until C1 lands.
        assert opened.legacy_dataset is not None, (
            "memmap opener must populate legacy_dataset; csv opener "
            "raises NotImplementedError before reaching here"
        )
        assert opened.legacy_session is not None, (
            "memmap opener must populate legacy_session; csv opener "
            "raises NotImplementedError before reaching here"
        )
        return run_inspector(
            dataset=opened.legacy_dataset,
            session=opened.legacy_session,
            log_path=log_path,
        )


if __name__ == "__main__":
    sys.exit(main())
