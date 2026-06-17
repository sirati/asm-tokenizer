"""A vanished/stale log sink must degrade to stderr, never crash the worker.

Regression for the secondary-2 crash-loop (2026-06-17): a shared bind-mounted
log dir deleted on the host left the in-container path a *stale mount handle*,
so `log_file.parent.mkdir(parents=True, exist_ok=True)` re-raised
`FileExistsError` (pathlib's `exist_ok` re-check `is_dir()` returns False on the
stale handle) and the tokenize worker died BEFORE Ready, looping forever.
`resilient_file_handler` must absorb any such `OSError` and fall back to stderr.
"""

import logging
from pathlib import Path

import pytest

from shared.logging_utils import resilient_file_handler


def test_happy_path_returns_file_handler(tmp_path):
    log_file = tmp_path / "sub" / "worker.log"
    handler = resilient_file_handler(log_file, level=logging.INFO)
    try:
        assert isinstance(handler, logging.FileHandler)
        assert log_file.exists()
    finally:
        handler.close()


def test_unwritable_parent_falls_back_to_stderr(tmp_path):
    # Parent resolves to a regular FILE -> mkdir raises OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    log_file = blocker / "worker.log"
    handler = resilient_file_handler(log_file, level=logging.INFO)
    try:
        assert not isinstance(handler, logging.FileHandler)
        assert isinstance(handler, logging.StreamHandler)  # stderr fallback
    finally:
        handler.close()


def test_stale_mount_fileexistserror_falls_back(tmp_path, monkeypatch):
    # Exact reproduced failure: the stale mount makes mkdir(exist_ok=True)
    # re-raise FileExistsError. The helper must not propagate it.
    log_file = tmp_path / "stale" / "worker.log"
    target_parent = log_file.parent
    real_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == target_parent:
            raise FileExistsError(17, "File exists")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    handler = resilient_file_handler(log_file, level=logging.INFO)  # must NOT raise
    try:
        assert not isinstance(handler, logging.FileHandler)
        assert isinstance(handler, logging.StreamHandler)
    finally:
        handler.close()
