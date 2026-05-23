"""Per-binary memmap-directory scan + post-discovery selection.

Single concern: turn a memmap directory path into a sorted list of
binary names that the matched-arm batch-decode driver will exercise.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

_INDEX_SUFFIX = "_index.bin"
_UNMATCHED_INDEX_SUFFIX = "_unmatched_index.bin"


def discover_binaries(memmap_dir: Path) -> List[str]:
    """Return sorted binary names present in ``memmap_dir``.

    A binary is recognised by the presence of ``<name>_index.bin`` (the
    matched-arm index sidecar emitted by the memmap builder). The
    ``<name>_unmatched_index.bin`` companion is skipped so each binary
    is returned once via its matched arm.
    """
    names: List[str] = []
    for entry in memmap_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(_UNMATCHED_INDEX_SUFFIX):
            continue
        if not name.endswith(_INDEX_SUFFIX):
            continue
        names.append(name[: -len(_INDEX_SUFFIX)])
    return sorted(names)


def filter_binaries(
    binary_names: Sequence[str],
    *,
    only: Optional[str],
    max_binaries: Optional[int],
) -> List[str]:
    """Apply the ``--only`` allow-list then the ``--max-binaries`` cap.

    Order is preserved (so the discovery sort still drives
    reproducibility) and the cap is applied AFTER the allow-list so
    ``--only`` semantics are exact.
    """
    selected = list(binary_names)
    if only is not None:
        keep = {x.strip() for x in only.split(",") if x.strip()}
        selected = [n for n in selected if n in keep]
    if max_binaries is not None:
        selected = selected[: max_binaries]
    return selected
