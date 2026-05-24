"""``python -m tokenizer.inspector`` entry point.

Single concern: parse CLI args via :mod:`._args`, build the per-binary
loader state (which owns the unified vocab + metadata loading via the
public :class:`BinaryDataset`), open a :class:`BinarySession` for that
binary inside a ``with`` block so its ExitStack-driven cleanup runs
even on mid-render exceptions, and hand the live session off to the
Textual app entry (:func:`tokenizer.inspector._app.run_inspector`).

``_app`` is the only module that imports :mod:`textual`; this file
imports it lazily inside :func:`main` so unit tests that import
:mod:`tokenizer.inspector.__main__` (e.g. to exercise argparse) don't
trip the textual-free default shell rule.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession

from ._args import parse_args

_log = logging.getLogger(__name__)


def _open_session(memmap_dir: Path, binary_name: str) -> tuple[BinaryDataset, BinarySession]:
    """Build the per-binary dataset shell + a fresh session.

    Wraps :class:`BinaryDataset` because that class is the public API
    that owns "load unified vocab + both section arms + function-names
    sidecar + variant offset map" wiring — reproducing that wiring
    here would duplicate logic the loader already centralises. The
    session is returned un-entered; the caller drives ``__enter__`` /
    ``__exit__`` via ``with``.
    """
    dataset = BinaryDataset(memmap_dir, binary_name)
    return dataset, dataset.open_session()


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    memmap_dir: Path = ns.memmap_dir
    binary_name: str = ns.binary

    _log.info(
        "inspector: opening session for binary=%s in %s",
        binary_name,
        memmap_dir,
    )

    # Lazy import: ``_app`` pulls in :mod:`textual`. Importing it at
    # module-level would block ``python -m tokenizer.inspector --help``
    # in any environment without textual installed (e.g. the default
    # ``nix develop`` shell), which the plan explicitly forbids.
    from ._app import run_inspector

    dataset, session = _open_session(memmap_dir, binary_name)
    with session:
        log_path = memmap_dir / ".tui-inspector.log"
        return run_inspector(
            dataset=dataset, session=session, log_path=log_path
        )


if __name__ == "__main__":
    sys.exit(main())
