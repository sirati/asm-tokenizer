"""Loader for the per-binary ``<binary>_extern_providers.txt`` sidecar.

Mirrors :class:`tokenizer.aligned_data.extern_providers.ExternProviderRegistry`
on the read side: validates the ``# format=N`` prelude (delegated to
:func:`tokenizer.aligned_data.extern_providers.iter_extern_providers`)
and materializes the encounter-ordered ``(line_no, library)`` stream
into a ``line_to_provider`` lookup dict.

The 1-indexed line number is what extern call_target
``function_section_ptr`` slots store; consumers (e.g. the inspector's
``InlineCallEntry.provider`` resolution) read the dict at that key to
obtain the library name. Line ``0`` is the reserved "library unknown"
sentinel and never appears as a key in the returned mapping.

Hard cutover: a missing or wrong-version prelude raises
:class:`ValueError` with a migration-pointing message, matching the
sibling :mod:`function_names_loader` policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from tokenizer.aligned_data.extern_providers import iter_extern_providers
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION


def load_extern_providers(path: Path) -> Dict[int, str]:
    """Load the extern-providers sidecar at ``path``.

    Returns a ``line_to_provider`` dict with 1-indexed line numbers.
    The first line of the file must be exactly
    ``# format=<MEMMAP_FORMAT_VERSION>``; any deviation raises
    :class:`ValueError` (delegated to :func:`iter_extern_providers`).
    A missing file raises :class:`ValueError` with a migration-pointing
    message so callers cannot silently fall back to an empty mapping
    when the sidecar should be present.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{path}: extern-providers sidecar missing; re-run memmap_builder "
            f"to regenerate the sidecar at format_version={MEMMAP_FORMAT_VERSION}"
        )
    return {line_no: library for line_no, library in iter_extern_providers(path)}
