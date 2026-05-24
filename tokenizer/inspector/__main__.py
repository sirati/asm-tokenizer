"""``python -m tokenizer.inspector`` entry point.

Single concern: parse CLI args via :mod:`._args`, build the per-binary
loader state (which owns the unified vocab + metadata loading via the
public :class:`BinaryDataset`), open a :class:`BinarySession` for that
binary inside a ``with`` block so its ExitStack-driven cleanup runs
even on mid-render exceptions, and (for now) print a placeholder line
to stdout. The Textual UI lands in a later phase under ``_app.py``;
this entry deliberately runs WITHOUT textual installed so the default
``nix develop`` shell stays free of that dependency.
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


def _matched_count(dataset: BinaryDataset) -> int:
    """Number of matched functions in the binary (placeholder copy)."""
    # ``BinaryDataset`` publishes the per-arm function count under the
    # legacy ``<prefix>_count`` attribute (see ``_publish_arm``); the
    # inspector seeds its tree from this number per plan D3.
    return dataset.matched_count


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    memmap_dir: Path = ns.memmap_dir
    binary_name: str = ns.binary

    _log.info(
        "inspector: opening session for binary=%s in %s",
        binary_name,
        memmap_dir,
    )

    dataset, session = _open_session(memmap_dir, binary_name)
    with session:
        count = _matched_count(dataset)
        print(
            f"tree would go here — {count} matched functions in "
            f"{binary_name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
