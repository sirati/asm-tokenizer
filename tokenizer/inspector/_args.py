"""Argparse spec + binary auto-detection.

Single concern: produce a fully-resolved :class:`argparse.Namespace`
ready for the runner. The only filesystem touch is the per-binary
sidecar glob backing ``--binary`` auto-detection — every other concern
(opening files, building the session, rendering) lives elsewhere.

The auto-detect anchor is ``<binary>_function_names.txt``: every binary
in a memmap directory has exactly one such sidecar (the function-names
registry), so ``glob('*_function_names.txt')`` yields one entry per
binary in the directory. Anchoring on this file rather than e.g.
``_sections.bin`` keeps the detection robust against the empty-arm
edge case where one of the data bins might be absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Sidecar suffix carried by every binary; see
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
            "BinarySession against a memmap directory and (in later "
            "phases) renders a navigable tree of matched functions, "
            "their variants, blocks, and inline calls."
        ),
    )
    parser.add_argument(
        "--memmap-dir",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Directory containing the per-binary memmap artefacts "
            "(*_sections.bin, *_data.bin, *_variants.bin, "
            "*_function_names.txt, unified_vocab.csv)."
        ),
    )
    parser.add_argument(
        "--binary",
        default=None,
        metavar="NAME",
        help=(
            "Name of the binary to inspect (the prefix used in the "
            "per-binary sidecars, e.g. 'nmap' for 'nmap_sections.bin'). "
            "Omit to auto-detect when exactly one binary lives in "
            "--memmap-dir."
        ),
    )
    return parser


def discover_binaries(memmap_dir: Path) -> list[str]:
    """List binary names present in ``memmap_dir``.

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


def resolve_binary(memmap_dir: Path, requested: str | None) -> str:
    """Return the binary name to open.

    Validates both user-facing flags at the CLI boundary: the
    ``memmap_dir`` must exist, and when ``requested`` is given its
    function-names sidecar must exist (otherwise the loader silently
    yields a zero-match session, masking typos). When ``requested``
    is omitted, scan ``memmap_dir`` for binaries and require exactly
    one; zero or multiple is a user error reported via
    :class:`SystemExit` with a candidate list.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse + resolve. Returns a namespace whose ``binary`` is set."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    ns.binary = resolve_binary(ns.memmap_dir, ns.binary)
    return ns
