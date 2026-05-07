"""Sidecar-tarball extraction: ``.tar.zst`` → on-disk binary path.

Single concern: turn a sidecar archive into a ready-to-tokenize binary
file on the local filesystem. The dataset convention is "exactly one
binary per archive" (see ``src/dataset/<pkg>/clang10_*_<8hex>.tar.zst``
— each archive contains a single regular file, the package binary). On
the rare chance an archive carries auxiliary files (manpages, debug
symbols), the largest regular file is selected — heuristic borrowed
from the dataset producer's invariant that the binary dominates archive
size.

The extractor owns ONE boundary: ``(tarball_path, scratch_dir, pkg) →
binary_path``. It does not own scratch-dir lifecycle (the caller decides
when to clean up — typically the same lifetime as ``staged_publish``)
nor binary-content interpretation (the tokenizer library reads
whatever's at the returned path).

Python 3.14's ``tarfile`` reads ``r:zst`` natively via the stdlib
``compression.zstd`` module, so no third-party dependency is required.
The extraction filter is ``data`` (PEP 706 default for new archives)
which strips setuid bits, link targets escaping the destination, and
device nodes — defensive even though these archives are produced by a
trusted nix derivation.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

__all__ = ["extract_binary"]

_logger = logging.getLogger(__name__)

# Member names whose **stem** (or full name) matches the package are
# preferred when picking the binary out of a multi-file archive. The
# dataset's invariant is single-binary archives, so this disambiguator
# only matters for hypothetical future variants; keeping the rule simple
# (exact basename match) avoids ambiguity-driven heuristics.
def _pick_member(members: list[tarfile.TarInfo], pkg: str) -> tarfile.TarInfo:
    """Pick the binary member from a tarball's regular-file members.

    Selection rules, in priority order:
    * Exact basename match against ``pkg``.
    * Largest regular file (the binary dominates archive size in the
      sidecar dataset's convention).

    Raises ``ValueError`` if no regular file is present — an empty or
    directory-only archive cannot be tokenized.
    """
    regular_files = [m for m in members if m.isfile()]
    if not regular_files:
        raise ValueError("tarball contains no regular files")

    for m in regular_files:
        if Path(m.name).name == pkg:
            return m

    return max(regular_files, key=lambda m: m.size)


def extract_binary(tarball_path: Path, scratch_dir: Path, pkg: str) -> Path:
    """Extract the binary member of ``tarball_path`` into ``scratch_dir``;
    return the on-disk path of the extracted file.

    ``scratch_dir`` must already exist; the caller (``staged_publish`` in
    the worker) owns its lifecycle. The archive is opened in ``r:zst``
    mode (Python 3.14's stdlib zstd-backed reader) and a single member
    is extracted via the ``data`` filter so setuid bits, escaping
    symlinks, and device nodes are stripped.

    The returned path is ``scratch_dir`` joined with the member's
    archive-relative path (preserving any internal subdir layout the
    archive uses, e.g. ``./hello`` → ``scratch_dir/hello``).
    """
    with tarfile.open(tarball_path, "r:zst") as tf:
        members = tf.getmembers()
        chosen = _pick_member(members, pkg)
        _logger.info(
            "extracting %s from %s into %s (size=%d)",
            chosen.name,
            tarball_path,
            scratch_dir,
            chosen.size,
        )
        tf.extract(chosen, path=scratch_dir, filter="data")

    extracted = scratch_dir / chosen.name
    if not extracted.is_file():
        raise FileNotFoundError(
            f"expected extracted binary at {extracted} not found "
            f"after extracting {tarball_path}"
        )
    return extracted
