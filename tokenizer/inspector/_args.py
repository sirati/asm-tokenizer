"""Argparse spec + binary auto-detection.

Single concern: produce a fully-resolved :class:`argparse.Namespace`
ready for the runner. The only filesystem touch is the per-source
discovery glob backing ``--binary`` auto-detection -- every other
concern (opening files, building the session, rendering) lives
elsewhere.

CLI shape: one positional ``PATH`` argument identifies the directory
to open; a mutually-exclusive provider flag picks the backend that
reads it.

* ``--memmap`` (alias ``--stage3``) -- read ``PATH`` as a per-binary
  memmap directory (``*_sections.bin``, ``*_data.bin``,
  ``*_function_names.txt``, ``unified_vocab.csv``). The auto-detect
  anchor is ``<binary>_function_names.txt``: every binary in a memmap
  directory has exactly one such sidecar (the function-names registry).
* ``--stage1`` -- read ``PATH`` as a per-variant ``<base>_output.csv``
  tree. The auto-detect anchor is :func:`VariantInfo.from_csv`'s
  ``pkg`` field: every CSV's filename encodes the binary name as the
  ``pkg`` axis.

Anchoring on those files rather than e.g. ``*_sections.bin`` keeps
detection robust against the empty-arm edge case where one of the data
bins might be absent (memmap mode), and against accidental cross-binary
CSV mixing (stage-1 mode).

The parsed Namespace carries a typed :class:`LoaderProvider`
discriminator (see :mod:`._app._binary_switcher._provider`) so
downstream consumers route off the enum rather than a string-typed
flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from tokenizer.inspector._app._binary_switcher._provider import LoaderProvider
from tokenizer.variant_info import VariantInfo


# Sidecar suffix carried by every binary in memmap mode; see
# :mod:`tokenizer.aligned_data.loader.function_names_loader`.
_FUNCTION_NAMES_SUFFIX = "_function_names.txt"


def build_parser() -> argparse.ArgumentParser:
    """Return the inspector's argparse parser.

    Pure: no I/O, no environment reads. The Namespace produced by
    ``parse_args`` is fed to the resolver below to fill in
    ``--binary`` when omitted.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tokenizer.inspector",
        description=(
            "Interactive inspector for batch_decode results: opens a "
            "per-binary backend (memmap or FTL-CSV) and renders a "
            "navigable tree of matched functions, their variants, "
            "blocks, and inline calls."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        metavar="PATH",
        help=(
            "Directory to open. Interpreted by the backend selected "
            "via --memmap/--stage3 (per-binary memmap artefacts) or "
            "--stage1 (per-variant <base>_output.csv tree)."
        ),
    )
    provider = parser.add_mutually_exclusive_group(required=True)
    provider.add_argument(
        "--memmap",
        "--stage3",
        dest="provider",
        action="store_const",
        const=LoaderProvider.MEMMAP,
        help=(
            "Read PATH as a memmap directory (stage-3 artefacts: "
            "*_sections.bin, *_data.bin, *_variants.bin, "
            "*_function_names.txt, unified_vocab.csv). Mutually "
            "exclusive with --stage1."
        ),
    )
    provider.add_argument(
        "--stage1",
        dest="provider",
        action="store_const",
        const=LoaderProvider.CSV,
        help=(
            "Read PATH as a per-variant <base>_output.csv tree "
            "(flat or nested layout). Mutually exclusive with "
            "--memmap/--stage3."
        ),
    )
    parser.add_argument(
        "--binary",
        default=None,
        metavar="NAME",
        help=(
            "Name of the binary to inspect. In memmap mode this is "
            "the prefix used in per-binary sidecars (e.g. 'nmap' for "
            "'nmap_sections.bin'). In stage-1 mode it is the ``pkg`` "
            "field of each variant CSV. Omit to auto-detect when "
            "exactly one binary lives in PATH."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_binaries(memmap_dir: Path) -> List[str]:
    """List binary names present in a memmap ``memmap_dir``.

    The anchor is the function-names sidecar (one per binary); names
    are the prefix before ``_function_names.txt`` in sorted order so
    the auto-detect error message is deterministic.
    """
    if not memmap_dir.is_dir():
        return []
    names = [
        p.name[: -len(_FUNCTION_NAMES_SUFFIX)]
        for p in memmap_dir.glob(f"*{_FUNCTION_NAMES_SUFFIX}")
        if p.is_file()
    ]
    names.sort()
    return names


def discover_binaries_csv(csv_dir: Path) -> List[str]:
    """List binary names present in a CSV ``csv_dir``.

    Recursively globs ``*_output.csv`` so both the flat (memmap-builder
    input) and nested (tokenize-worker output) layouts are covered;
    derives the ``pkg`` field via :func:`VariantInfo.from_csv` per
    audit F-MED-13 (no parallel filename parser). CSVs whose
    ``VariantInfo`` cannot be derived (malformed filename, missing
    sidecar fallback) are silently skipped so a single malformed file
    does not poison auto-detect; if no usable CSV survives the caller
    surfaces a fail-loud error.
    """
    if not csv_dir.is_dir():
        return []
    names: set[str] = set()
    for path in csv_dir.rglob("*_output.csv"):
        if not path.is_file():
            continue
        try:
            info = VariantInfo.from_csv(path)
        except ValueError:
            continue
        names.add(info.pkg)
    return sorted(names)


# ---------------------------------------------------------------------------
# Per-provider binary resolution
# ---------------------------------------------------------------------------


def _resolve_binary_memmap(memmap_dir: Path, requested: str | None) -> str:
    """Memmap-side ``--binary`` resolution.

    Validates the directory exists and -- when ``requested`` is given
    -- that its function-names sidecar is on disk (otherwise the
    loader silently yields a zero-match session, masking typos). When
    ``requested`` is omitted, requires exactly one binary; zero or
    multiple is a user error with a candidate list.
    """
    if not memmap_dir.is_dir():
        raise SystemExit(f"memmap directory not found: {memmap_dir}")
    if requested:
        sidecar = memmap_dir / f"{requested}{_FUNCTION_NAMES_SUFFIX}"
        if not sidecar.is_file():
            candidates = discover_binaries(memmap_dir)
            raise SystemExit(
                f"binary {requested!r} not found in {memmap_dir}; "
                f"available: {candidates}."
            )
        return requested
    candidates = discover_binaries(memmap_dir)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"--binary not given and no binaries detected in "
            f"{memmap_dir} (looked for files matching "
            f"'*{_FUNCTION_NAMES_SUFFIX}')."
        )
    listing = ", ".join(candidates)
    raise SystemExit(
        f"--binary not given and {memmap_dir} contains multiple "
        f"binaries; pass --binary <name> to pick one. "
        f"Candidates: {listing}."
    )


def _resolve_binary_csv(csv_dir: Path, requested: str | None) -> str:
    """CSV-side ``--binary`` resolution.

    Mirrors :func:`_resolve_binary_memmap`: directory must exist;
    when ``requested`` is given, at least one CSV must report a
    matching ``pkg``; when omitted, exactly one distinct ``pkg``
    must be present.
    """
    if not csv_dir.is_dir():
        raise SystemExit(f"csv directory not found: {csv_dir}")
    candidates = discover_binaries_csv(csv_dir)
    if requested:
        if requested not in candidates:
            raise SystemExit(
                f"binary {requested!r} not found in {csv_dir}; "
                f"available (from VariantInfo.pkg): {candidates}."
            )
        return requested
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"--binary not given and no binaries detected in "
            f"{csv_dir} (looked for *_output.csv recursively)."
        )
    listing = ", ".join(candidates)
    raise SystemExit(
        f"--binary not given and {csv_dir} contains multiple "
        f"binaries; pass --binary <name> to pick one. "
        f"Candidates: {listing}."
    )


# Public name kept for backwards compat (used by tests + older callers).
def resolve_binary(memmap_dir: Path, requested: str | None) -> str:
    """Memmap-side resolver (backwards-compat alias)."""
    return _resolve_binary_memmap(memmap_dir, requested)


# ---------------------------------------------------------------------------
# Provider -> resolver dispatch (typed, no string-typed if/elif)
# ---------------------------------------------------------------------------


_RESOLVERS = {
    LoaderProvider.MEMMAP: _resolve_binary_memmap,
    LoaderProvider.CSV: _resolve_binary_csv,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse + resolve. Returns a namespace whose ``binary`` is set.

    The mutex group guarantees ``ns.provider`` is set to a
    :class:`LoaderProvider`; the resolver is selected off that enum.
    """
    parser = build_parser()
    ns = parser.parse_args(argv)
    resolver = _RESOLVERS[ns.provider]
    ns.binary = resolver(ns.path, ns.binary)
    return ns
