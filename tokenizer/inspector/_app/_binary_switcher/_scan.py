"""Filesystem-side detection of loadable data folders.

Single concern: given a directory, answer "does this folder contain
loadable data for provider X?" Delegates to the canonical discovery
helpers already living in :mod:`tokenizer.inspector._args` so the
detection logic stays in lockstep with the CLI-side auto-detect.

The scan walks only the directory's immediate children + the
provider's canonical subdir (per :func:`resolve_provider_dirs`) —
arbitrary recursion is the folder picker's concern (one level at a
time, lazily expanded by the user). For CSV detection a single
``rglob`` is acceptable inside each candidate dir because
:func:`discover_binaries_csv` already covers both flat + nested
layouts; without it we would miss the nested layout's outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import List, Mapping

from tokenizer.inspector._args import discover_binaries_with_paths

from ._provider import LoaderProvider


__all__ = [
    "FolderScanResult",
    "binaries_in_folder",
    "is_loadable_for",
    "is_loadable_for_any",
    "list_child_directories",
    "scan_folder",
]


@dataclass(frozen=True)
class FolderScanResult:
    """Per-provider scan result for one folder.

    ``loadable`` is ``True`` when the folder contains at least one
    binary discoverable by the provider's auto-detect (including the
    provider-subdir fallback); ``binaries`` is the sorted list of
    discovered binary names (empty when not loadable).
    ``binary_to_dir`` maps each discovered binary name to the
    effective dir it lives in (either ``path`` itself or
    ``path/<provider.subdir_name>/``) so callers that commit a switch
    can thread the loader-side path. Both backends emit deterministic
    sorted output so the UI tree order is stable across re-scans.
    """

    provider: LoaderProvider
    path: Path
    binaries: tuple[str, ...]
    binary_to_dir: Mapping[str, Path]

    @property
    def loadable(self) -> bool:
        """``True`` iff at least one binary lives in this folder."""
        return bool(self.binaries)


def binaries_in_folder(path: Path, provider: LoaderProvider) -> List[str]:
    """Discover binaries reachable from ``path`` for the given provider.

    Walks both the in-place dir and the provider's canonical subdir
    (``path/<provider.subdir_name>/``) — see
    :func:`tokenizer.inspector._args.discover_binaries_with_paths` for
    the full policy. Returns the sorted union of binary names; the
    empty list when ``path`` is not a directory or carries no data
    for ``provider`` in either candidate.
    """
    if not path.is_dir():
        return []
    return sorted(discover_binaries_with_paths(path, provider))


def scan_folder(path: Path, provider: LoaderProvider) -> FolderScanResult:
    """Bundle the subdir-aware scan + provider into a typed record.

    Convenience wrapper for the dialog's tree-build loop: one call per
    folder + provider yields the per-cell "is this green?" + "which
    children?" + "which effective dir do its files live in?" answer.
    """
    binary_to_dir = (
        discover_binaries_with_paths(path, provider) if path.is_dir() else {}
    )
    return FolderScanResult(
        provider=provider,
        path=path,
        binaries=tuple(sorted(binary_to_dir)),
        binary_to_dir=MappingProxyType(dict(binary_to_dir)),
    )


def is_loadable_for(path: Path, provider: LoaderProvider) -> bool:
    """Compact predicate for the folder picker's green-marking.

    Same logic as :func:`scan_folder` but avoids constructing the
    binaries tuple when only the boolean answer is needed.
    """
    return bool(binaries_in_folder(path, provider))


def is_loadable_for_any(path: Path) -> bool:
    """``True`` iff ``path`` is loadable for AT LEAST ONE provider.

    Provider-agnostic predicate used by the folder picker once the
    binary-switcher dialog lifted ``change path...`` out of per-provider
    subtrees: the picker no longer knows which backend the user will
    pick, so any-provider data suffices to mark a folder as worth
    visiting. Short-circuits on the first match for cheap traversal of
    cold subdirectories.
    """
    for provider in LoaderProvider:
        if binaries_in_folder(path, provider):
            return True
    return False


def list_child_directories(path: Path) -> List[Path]:
    """Sorted list of immediate sub-directories of ``path``.

    Single concern: filesystem enumeration for the folder picker's
    tree-expand. Skips files, hidden dot-directories (``.*``) and
    permission errors; the picker should never crash on a permission-
    denied subdir.
    """
    if not path.is_dir():
        return []
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name)
    except (PermissionError, OSError):
        return []
    return [p for p in entries if p.is_dir() and not p.name.startswith(".")]
