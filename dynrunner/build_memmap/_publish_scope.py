"""The build_memmap per-binary publish scope -- defined once.

Single concern: own the ``build_memmap/<binary_name>`` scope literal the
memmap worker hands to ``staged_publish`` and the directory that scope
resolves to on disk. The publishing side (``build_memmap.worker``) and
any reading side (the composite pipeline resolving where a binary's
memmap landed) share this one definition so the layout decision is never
duplicated.

The scope is mode-asymmetric exactly as ``tokenizer.output_staging``
documents: container deployment republishes under ``<scope>/`` so the
files land at ``<root>/build_memmap/<name>/``; standalone ignores the
scope and the files land flat at ``<root>/``. ``memmap_dir_for`` is the
inverse of that decision (via ``published_path``), so a reader binds to
wherever the worker actually placed the sidecars in either mode.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.output_staging import published_path


def build_memmap_scope(binary_name: str) -> str:
    """The ``staged_publish`` scope for one binary's memmap artifacts."""
    return f"build_memmap/{binary_name}"


def memmap_dir_for(output_root: Path, binary_name: str) -> Path:
    """The directory a binary's memmap sidecars land in under ``output_root``.

    ``<output_root>/build_memmap/<name>/`` in container mode, ``output_root``
    itself in standalone mode -- mirroring the worker's publish layout.
    Resolved through ``published_path`` so the mode decision lives in one
    place; an empty filename yields the publish DIRECTORY directly.
    """
    return published_path(output_root, build_memmap_scope(binary_name), "")
