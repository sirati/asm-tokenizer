"""Sidecar-tarball extraction: ``.tar.zst`` → on-disk binary paths.

Single concern: turn a sidecar archive into ready-to-tokenize binary
files on the local filesystem. A package may contain any number of
binaries (executables and shared libraries), so the extractor yields
*every* regular-file member as a ``(extracted_path, member_name)``
pair — the worker fans out one tokenization run per member.

The extractor owns ONE boundary: ``(tarball_path, scratch_dir) →
Iterator[(binary_path, member_name)]``. It does not own scratch-dir
lifecycle (the caller decides when to clean up — typically the same
lifetime as ``staged_publish``) nor binary-content interpretation (the
tokenizer library reads whatever's at each yielded path). Symlinks,
directories, and device nodes are skipped (filtered to ``m.isfile()``)
so callers don't have to special-case archive layout.

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
from collections.abc import Iterator
from pathlib import Path

__all__ = ["extract_all_binaries"]

_logger = logging.getLogger(__name__)


def extract_all_binaries(
    tarball_path: Path,
    scratch_dir: Path,
) -> Iterator[tuple[Path, str]]:
    """Extract every regular-file member of ``tarball_path`` into
    ``scratch_dir``; yield ``(extracted_path, archive_member_name)``
    pairs in archive order.

    ``scratch_dir`` must already exist; the caller (``staged_publish``
    in the worker) owns its lifecycle. The archive is opened in
    ``r:zst`` mode (Python 3.14's stdlib zstd-backed reader). Each
    regular-file member is extracted via the ``data`` filter so setuid
    bits, escaping symlinks, and device nodes are stripped.

    The yielded ``extracted_path`` is ``scratch_dir`` joined with the
    member's archive-relative path (preserving any internal subdir
    layout the archive uses, e.g. ``./hello`` → ``scratch_dir/hello``).
    The yielded ``archive_member_name`` is the member name verbatim
    from the archive (e.g. ``./hello`` or ``libssl.so.1.1``); callers
    that need just the basename must call ``Path(name).name`` themselves.

    Members whose ``isfile()`` is false (directories, symlinks, device
    nodes, hardlinks) are skipped silently — they're never tokenization
    targets. An archive with zero regular-file members yields nothing;
    the caller decides whether that is a soft skip or an error.
    """
    with tarfile.open(tarball_path, "r:zst") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            _logger.info(
                "extracting %s from %s into %s (size=%d)",
                member.name,
                tarball_path,
                scratch_dir,
                member.size,
            )
            tf.extract(member, path=scratch_dir, filter="data")
            extracted = scratch_dir / member.name
            if not extracted.is_file():
                raise FileNotFoundError(
                    f"expected extracted binary at {extracted} not found "
                    f"after extracting {tarball_path}"
                )
            yield extracted, member.name
