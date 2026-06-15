"""Tests for the ephemeral Ghidra project-directory lifecycle.

Pure filesystem concern — no JVM, so these run everywhere. The ephemeral
root is redirected into ``tmp_path`` so nothing touches the real ``/tmp``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.disasm.ghidra_provider import project_workspace as pw


@pytest.fixture
def redirected_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(pw, "_ephemeral_root", lambda: tmp_path)
    return tmp_path


def test_create_returns_fresh_existing_dir(redirected_root: Path) -> None:
    ws = pw.create_project_workspace()
    assert ws.is_dir()
    assert ws.parent.name == pw._WORKSPACE_PARENT
    assert ws.name.startswith(pw._WORKSPACE_PREFIX)
    assert ws.parent == redirected_root / pw._WORKSPACE_PARENT


def test_create_yields_unique_dirs(redirected_root: Path) -> None:
    a = pw.create_project_workspace()
    b = pw.create_project_workspace()
    assert a != b
    assert a.is_dir() and b.is_dir()


def test_remove_deletes_workspace_and_contents(redirected_root: Path) -> None:
    ws = pw.create_project_workspace()
    (ws / "proj.gpr").write_text("x")
    (ws / "proj.rep").mkdir()
    pw.remove_project_workspace(ws)
    assert not ws.exists()


def test_remove_is_idempotent(redirected_root: Path) -> None:
    ws = pw.create_project_workspace()
    pw.remove_project_workspace(ws)
    pw.remove_project_workspace(ws)  # already gone — must not raise
    assert not ws.exists()


def test_remove_refuses_wrong_parent(tmp_path: Path) -> None:
    # Right prefix, wrong parent dir name -> must be refused, not deleted.
    stray = tmp_path / "not-ghidra-projects" / f"{pw._WORKSPACE_PREFIX}x"
    stray.mkdir(parents=True)
    pw.remove_project_workspace(stray)
    assert stray.exists()


def test_remove_refuses_wrong_prefix(tmp_path: Path) -> None:
    # Right parent dir name, wrong prefix -> must be refused, not deleted.
    stray = tmp_path / pw._WORKSPACE_PARENT / "something-important"
    stray.mkdir(parents=True)
    pw.remove_project_workspace(stray)
    assert stray.exists()
