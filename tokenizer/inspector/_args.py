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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from tokenizer.inspector._app._binary_switcher._provider import (
    LoaderProvider,
    resolve_provider_dirs,
)
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
# Subdir-aware discovery: union over in-place + provider-subdir candidates
# ---------------------------------------------------------------------------


# Per-provider, the strictly-in-place discovery function. Same shape
# across providers so the subdir-search policy is generic.
_IN_PLACE_DISCOVERY: Dict[LoaderProvider, Callable[[Path], List[str]]] = {
    LoaderProvider.MEMMAP: discover_binaries,
    LoaderProvider.CSV: discover_binaries_csv,
}


def discover_binaries_with_paths(
    path: Path, provider: LoaderProvider
) -> Dict[str, Path]:
    """Map each binary discoverable under ``path`` to its effective dir.

    Walks every candidate dir from :func:`resolve_provider_dirs`
    (in-place first, then ``path/<provider.subdir_name>/``) and
    accumulates ``binary_name -> dir_containing_its_files``. In-place
    wins on collision, matching the unified-vocab gate's "explicit copy
    beats inherited" rule. Returns an empty dict when neither candidate
    contains data for ``provider``.

    Callers that only need the name list can union the keys of this
    map; callers that need to actually open the files (CLI resolver,
    backend opener) read the value side to thread the correct dir into
    the loader.
    """
    discover = _IN_PLACE_DISCOVERY[provider]
    binary_to_dir: Dict[str, Path] = {}
    # Iterate candidates in the policy order so the first-seen entry
    # wins (in-place precedes subdir).
    for candidate in resolve_provider_dirs(path, provider):
        if not candidate.is_dir():
            continue
        for name in discover(candidate):
            binary_to_dir.setdefault(name, candidate)
    return binary_to_dir


# ---------------------------------------------------------------------------
# Per-provider binary resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBinary:
    """Resolver result: the effective dir + the binary name.

    ``path`` is the dir the loader should consume (either the
    user-supplied anchor or its provider-subdir). ``name`` is the
    binary name to focus on. The split lets the CLI / dialog keep the
    user's anchor intact (for next-dialog seeding) while the opener
    threads the loader-side path.
    """

    path: Path
    name: str


def _candidate_dirs_for(path: Path, provider: LoaderProvider) -> List[Path]:
    """Helper: candidate dirs from :func:`resolve_provider_dirs`, filtered.

    Returns only the candidates that exist on disk + are directories.
    Keeps the policy order intact (in-place first), so the resolver's
    first-hit semantics agree with
    :func:`discover_binaries_with_paths`'s in-place-wins rule.
    """
    return [c for c in resolve_provider_dirs(path, provider) if c.is_dir()]


def _resolve_binary_memmap(
    memmap_dir: Path, requested: str | None
) -> ResolvedBinary:
    """Memmap-side ``--binary`` resolution.

    Searches the in-place dir first, then ``memmap_dir/memmap/``.
    Validates that ``requested`` (when given) has its function-names
    sidecar on disk in EITHER candidate; otherwise the loader silently
    yields a zero-match session, masking typos. When ``requested`` is
    omitted, requires exactly one binary across both candidates; zero
    or multiple is a user error with a candidate list. The returned
    :class:`ResolvedBinary.path` names whichever candidate carries the
    binary's files (so the caller threads it into the loader).
    """
    if not memmap_dir.is_dir():
        raise SystemExit(f"memmap directory not found: {memmap_dir}")
    binary_to_dir = discover_binaries_with_paths(
        memmap_dir, LoaderProvider.MEMMAP
    )
    if requested:
        if requested not in binary_to_dir:
            raise SystemExit(
                f"binary {requested!r} not found in {memmap_dir}; "
                f"available: {sorted(binary_to_dir)}."
            )
        return ResolvedBinary(path=binary_to_dir[requested], name=requested)
    if len(binary_to_dir) == 1:
        name, dir_ = next(iter(binary_to_dir.items()))
        return ResolvedBinary(path=dir_, name=name)
    if not binary_to_dir:
        searched = ", ".join(
            str(c) for c in resolve_provider_dirs(memmap_dir, LoaderProvider.MEMMAP)
        )
        raise SystemExit(
            f"--binary not given and no binaries detected in "
            f"{memmap_dir} (looked for files matching "
            f"'*{_FUNCTION_NAMES_SUFFIX}' under: {searched})."
        )
    listing = ", ".join(sorted(binary_to_dir))
    raise SystemExit(
        f"--binary not given and {memmap_dir} contains multiple "
        f"binaries; pass --binary <name> to pick one. "
        f"Candidates: {listing}."
    )


def _resolve_binary_csv(
    csv_dir: Path, requested: str | None
) -> ResolvedBinary:
    """CSV-side ``--binary`` resolution.

    Mirrors :func:`_resolve_binary_memmap`: searches the in-place dir
    first, then ``csv_dir/csv/``; directory must exist; ``requested``
    (when given) must appear in at least one candidate's CSV union;
    when omitted, exactly one distinct ``pkg`` must be present across
    candidates.
    """
    if not csv_dir.is_dir():
        raise SystemExit(f"csv directory not found: {csv_dir}")
    binary_to_dir = discover_binaries_with_paths(csv_dir, LoaderProvider.CSV)
    if requested:
        if requested not in binary_to_dir:
            raise SystemExit(
                f"binary {requested!r} not found in {csv_dir}; "
                f"available (from VariantInfo.pkg): {sorted(binary_to_dir)}."
            )
        return ResolvedBinary(path=binary_to_dir[requested], name=requested)
    if len(binary_to_dir) == 1:
        name, dir_ = next(iter(binary_to_dir.items()))
        return ResolvedBinary(path=dir_, name=name)
    if not binary_to_dir:
        searched = ", ".join(
            str(c) for c in resolve_provider_dirs(csv_dir, LoaderProvider.CSV)
        )
        raise SystemExit(
            f"--binary not given and no binaries detected in "
            f"{csv_dir} (looked for *_output.csv recursively under: "
            f"{searched})."
        )
    listing = ", ".join(sorted(binary_to_dir))
    raise SystemExit(
        f"--binary not given and {csv_dir} contains multiple "
        f"binaries; pass --binary <name> to pick one. "
        f"Candidates: {listing}."
    )


# Public name kept for backwards compat (used by tests + older callers).
def resolve_binary(memmap_dir: Path, requested: str | None) -> str:
    """Memmap-side resolver (backwards-compat alias).

    Returns only the binary name string, matching the pre-subdir-search
    contract. Callers that need the effective dir use
    :func:`_resolve_binary_memmap` directly.
    """
    return _resolve_binary_memmap(memmap_dir, requested).name


# ---------------------------------------------------------------------------
# Provider -> resolver dispatch (typed, no string-typed if/elif)
# ---------------------------------------------------------------------------


_RESOLVERS: Dict[
    LoaderProvider, Callable[[Path, "str | None"], ResolvedBinary]
] = {
    LoaderProvider.MEMMAP: _resolve_binary_memmap,
    LoaderProvider.CSV: _resolve_binary_csv,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse + resolve. Returns a namespace whose ``binary`` is set.

    The mutex group guarantees ``ns.provider`` is set to a
    :class:`LoaderProvider`; the resolver is selected off that enum.

    Post-resolve the namespace carries two paths:

    * ``ns.path`` -- the user's anchor (unchanged from argparse).
    * ``ns.effective_path`` -- where the binary's files actually live
      (either ``ns.path`` itself or the provider's subdir under it).

    Downstream the opener consumes ``effective_path`` while the App
    stores ``path`` as the dialog's anchor so the next switch opens
    against the corpus root, not the subdir.
    """
    parser = build_parser()
    ns = parser.parse_args(argv)
    resolver = _RESOLVERS[ns.provider]
    resolved = resolver(ns.path, ns.binary)
    ns.binary = resolved.name
    ns.effective_path = resolved.path
    return ns
