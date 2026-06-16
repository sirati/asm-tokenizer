"""Per-binary memmap-directory scan + post-discovery selection.

Single concern: turn a memmap directory path into a sorted list of
binary names (recognised by their matched-arm ``<name>_index.bin``
sidecar) plus the ``--only`` / ``--max-binaries`` post-selection.

Owned by the ``tokenizer.aligned_data`` package because every consumer
operates on the aligned-data memmap layout and this scan already keys
off the realized_lengths arm grammars: the realized-length and
sorted-index generators (their ``__main__`` discovery) and the dynrunner
phase-4 ``build_index`` task all select their binary set through here,
and ``tools.run_batch_smoke`` re-exports it for the same scan. Living in
``aligned_data`` keeps it inside the production image (``tools`` is not
packaged), so the mesh workers can import it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

# Import the arm grammar from the realized_lengths ``_format`` submodule
# (not the package ``__init__``, whose generator pulls in sorted_index
# and risks a circular import): ``_format`` is a leaf module.
from tokenizer.aligned_data.realized_lengths._format import ARMS
from tokenizer.aligned_data.realized_lengths._geometry_format import GEOMETRY_ARMS

_INDEX_SUFFIX = "_index.bin"
_UNMATCHED_INDEX_SUFFIX = "_unmatched_index.bin"

# Realized-length sidecars also end in ``_index.bin`` but are NOT
# binary-existence signals -- they co-exist in the memmap dir once the
# realized-lengths pass has run (the sorted-index build's Phase-4a
# precondition). BOTH sidecar families must be excluded: the length-CSR
# arms (``_lengths_index.bin`` / ``_unmatched_lengths_index.bin``) AND
# the realized-geometry RLG3 arms (``_realized_index.bin`` /
# ``_unmatched_realized_index.bin``). Sourced from BOTH arm grammars so
# this exclusion never drifts from the generator's filenames.
_REALIZED_LENGTHS_INDEX_SUFFIXES = tuple(
    arm.index_suffix for arm in ARMS
) + tuple(arm.index_suffix for arm in GEOMETRY_ARMS)


def discover_binaries(memmap_dir: Path) -> List[str]:
    """Return sorted binary names present in ``memmap_dir``.

    A binary is recognised by the presence of ``<name>_index.bin`` (the
    matched-arm index sidecar emitted by the memmap builder). The
    ``<name>_unmatched_index.bin`` companion and the realized-length CSR
    sidecars are skipped so each binary is returned once via its matched
    arm.
    """
    names: List[str] = []
    for entry in memmap_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(_UNMATCHED_INDEX_SUFFIX):
            continue
        if name.endswith(_REALIZED_LENGTHS_INDEX_SUFFIXES):
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
