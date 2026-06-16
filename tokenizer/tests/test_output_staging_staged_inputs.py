"""``staged_inputs`` copies NFS inputs to node-local scratch in container
mode and is a no-op in standalone mode.

It is the input-side mirror of ``staged_publish``: container deployment
stages each requested source under ``/app/out-tmp/staged-inputs/<scope>/``
(mirroring the source's absolute layout so distinct sources never
collide) and removes the scope subtree on exit — clean OR exception;
standalone yields the original paths unchanged so reads happen in place.

The deployment mode is forced via ``is_container_deployment`` and the
tmpfs root is redirected via ``_SLURM_OUT_TMP`` so both branches run
deterministically without an ``/app/out-tmp`` mount on the test host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tokenizer.output_staging as staging
from tokenizer.output_staging import staged_inputs


@pytest.fixture
def out_tmp(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the tmpfs root to a tmp dir and force container mode."""
    root = tmp_path / "out-tmp"
    root.mkdir()
    monkeypatch.setattr(staging, "_SLURM_OUT_TMP", root)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    return root


@pytest.fixture
def standalone(monkeypatch) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)


def _make_source(base: Path, rel: str, content: bytes) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_container_copies_to_scratch_and_yields_by_key(
    tmp_path: Path, out_tmp: Path
) -> None:
    nfs = tmp_path / "nfs"
    csv = _make_source(nfs, "bin/a.csv", b"csv-data")
    mapping = _make_source(nfs, "vocab/a.mapping.b64c", b"map-data")
    sources = {"csv": csv, "mapping": mapping}

    with staged_inputs(sources, scope="binA") as local:
        # Same keys, distinct local paths under the scratch root.
        assert set(local) == {"csv", "mapping"}
        scope_root = out_tmp / "staged-inputs" / "binA"
        for key, src in sources.items():
            assert scope_root in local[key].parents
            assert local[key] != src
            # Content faithfully copied.
            assert local[key].read_bytes() == src.read_bytes()
        # Distinct sources never collide.
        assert local["csv"] != local["mapping"]


def test_container_removes_scratch_on_clean_exit(
    tmp_path: Path, out_tmp: Path
) -> None:
    csv = _make_source(tmp_path / "nfs", "a.csv", b"x")
    with staged_inputs({"csv": csv}, scope="binA") as local:
        assert local["csv"].exists()
    assert not (out_tmp / "staged-inputs" / "binA").exists()


def test_container_removes_scratch_on_exception(
    tmp_path: Path, out_tmp: Path
) -> None:
    csv = _make_source(tmp_path / "nfs", "a.csv", b"x")
    with pytest.raises(RuntimeError):
        with staged_inputs({"csv": csv}, scope="binA"):
            raise RuntimeError("boom")
    # Cleanup runs in the finally even on exception.
    assert not (out_tmp / "staged-inputs" / "binA").exists()


def test_distinct_sources_same_basename_do_not_collide(
    tmp_path: Path, out_tmp: Path
) -> None:
    nfs = tmp_path / "nfs"
    one = _make_source(nfs, "v1/_meta.json", b"one")
    two = _make_source(nfs, "v2/_meta.json", b"two")
    with staged_inputs({"a": one, "b": two}, scope="binA") as local:
        assert local["a"] != local["b"]
        assert local["a"].read_bytes() == b"one"
        assert local["b"].read_bytes() == b"two"


def test_standalone_is_noop_yields_originals(
    tmp_path: Path, standalone: None
) -> None:
    csv = _make_source(tmp_path / "nfs", "a.csv", b"x")
    sources = {"csv": csv}
    with staged_inputs(sources, scope="binA") as local:
        # No copying: the original paths pass through unchanged.
        assert local == sources
        assert local["csv"] == csv


def test_standalone_does_not_touch_out_tmp(
    tmp_path: Path, monkeypatch
) -> None:
    # Even with a redirected tmpfs root present, standalone mode must not
    # create any scratch under it.
    root = tmp_path / "out-tmp"
    root.mkdir()
    monkeypatch.setattr(staging, "_SLURM_OUT_TMP", root)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)
    csv = _make_source(tmp_path / "nfs", "a.csv", b"x")
    with staged_inputs({"csv": csv}, scope="binA"):
        pass
    assert list(root.iterdir()) == []
