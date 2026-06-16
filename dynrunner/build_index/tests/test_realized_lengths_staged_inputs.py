"""The realized_lengths worker reads the binary's memmap inputs from
node-local scratch and publishes the output sidecars back to NFS.

In container mode ``_generate_and_publish`` must stage every existing
input sidecar via ``staged_inputs`` (so the generator's random-access
``_data.bin`` reads + ``_sections.bin``/``_index.bin`` re-reads hit local
tmpfs, never the shared filesystem), run the generator against the LOCAL
dir, and publish ONLY the generator's returned output paths to the
unchanged NFS ``memmap_dir`` via an explicit-dst ``task.publish``.
Standalone mode stays a pass-through (the local dir IS the NFS dir, the
generator runs in place, nothing is published).

``generate_realized_lengths`` is stubbed so the test pins the staging +
publish routing contract without invoking the full dedup compute;
deployment mode + the tmpfs root are forced via the staging module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tokenizer.output_staging as staging
import dynrunner.build_index.realized_lengths_worker as worker


class _FakeTask:
    """Minimal Task stand-in recording explicit-dst publish calls."""

    def __init__(self) -> None:
        self.published: list[tuple[Path, Path]] = []

    def publish(self, src, dst=None, *, key=None) -> None:
        self.published.append((Path(src), Path(dst)))


@pytest.fixture
def out_tmp(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "out-tmp"
    root.mkdir()
    monkeypatch.setattr(staging, "_SLURM_OUT_TMP", root)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    return root


def _seed_inputs(memmap_dir: Path, binary_name: str) -> None:
    """Write the four per-binary memmap input sidecars with marker bytes."""
    memmap_dir.mkdir(parents=True, exist_ok=True)
    for suffix in worker._INPUT_SUFFIXES:
        (memmap_dir / f"{binary_name}{suffix}").write_bytes(
            f"{binary_name}{suffix}".encode()
        )


def test_container_runs_on_local_copies_and_publishes_to_nfs(
    tmp_path: Path, out_tmp: Path
) -> None:
    memmap_dir = tmp_path / "out-network" / "binA"
    _seed_inputs(memmap_dir, "binA")
    scratch = out_tmp / "staged-inputs" / "realized_lengths" / "binA"

    seen: dict = {}

    def fake_generate(base_path, binary_name):
        # The generator must run against the LOCAL staged copies, never
        # the NFS sources. Its base_path is the mirrored scratch dir, and
        # every input it would read is present + byte-identical there.
        base_path = Path(base_path)
        assert base_path != memmap_dir
        assert scratch in base_path.parents or base_path == scratch
        seen["base"] = base_path
        for suffix in worker._INPUT_SUFFIXES:
            staged = base_path / f"{binary_name}{suffix}"
            assert staged.read_bytes() == f"{binary_name}{suffix}".encode()
        # Write the four output sidecars into the LOCAL dir (as the real
        # generator does) and return their paths.
        out_paths = []
        for name in (
            "binA_lengths.bin",
            "binA_lengths_index.bin",
            "binA_unmatched_lengths.bin",
            "binA_unmatched_lengths_index.bin",
        ):
            p = base_path / name
            p.write_bytes(b"OUT")
            out_paths.append(p)
        return {
            "matched": out_paths[:2],
            "unmatched": out_paths[2:],
        }

    task = _FakeTask()
    import unittest.mock as mock

    with mock.patch.object(worker, "generate_realized_lengths", fake_generate):
        result = worker._generate_and_publish(task, memmap_dir, "binA")

    # The generator base_path was the node-local scratch, not the NFS dir.
    assert seen["base"] != memmap_dir
    # Every returned output was published to the unchanged NFS location.
    dsts = {dst for _src, dst in task.published}
    assert dsts == {
        memmap_dir / "binA_lengths.bin",
        memmap_dir / "binA_lengths_index.bin",
        memmap_dir / "binA_unmatched_lengths.bin",
        memmap_dir / "binA_unmatched_lengths_index.bin",
    }
    # Each published src is the local copy the generator returned.
    for src, dst in task.published:
        assert seen["base"] in src.parents
        assert src.name == dst.name
    # The returned dict is surfaced verbatim for the handler's logging.
    assert set(result) == {"matched", "unmatched"}


def test_container_stages_only_existing_inputs(
    tmp_path: Path, out_tmp: Path
) -> None:
    # An empty unmatched arm: no _unmatched_data.bin on NFS. Staging must
    # skip the absent input (no copy error) and the generator still runs.
    memmap_dir = tmp_path / "out-network" / "binB"
    memmap_dir.mkdir(parents=True)
    for suffix in ("_index.bin", "_sections.bin", "_data.bin"):
        (memmap_dir / f"binB{suffix}").write_bytes(b"x")

    def fake_generate(base_path, binary_name):
        base_path = Path(base_path)
        # The three present inputs are staged; the absent one is not.
        assert (base_path / "binB_data.bin").exists()
        assert not (base_path / "binB_unmatched_data.bin").exists()
        p = base_path / "binB_lengths.bin"
        p.write_bytes(b"OUT")
        return {"matched": [p]}

    task = _FakeTask()
    import unittest.mock as mock

    with mock.patch.object(worker, "generate_realized_lengths", fake_generate):
        worker._generate_and_publish(task, memmap_dir, "binB")

    assert [dst for _s, dst in task.published] == [memmap_dir / "binB_lengths.bin"]


def test_standalone_runs_in_place_and_skips_publish(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)
    memmap_dir = tmp_path / "binC"
    _seed_inputs(memmap_dir, "binC")

    def fake_generate(base_path, binary_name):
        # Standalone: the generator runs against the NFS dir directly.
        assert Path(base_path) == memmap_dir
        p = memmap_dir / "binC_lengths.bin"
        p.write_bytes(b"OUT")
        return {"matched": [p]}

    task = _FakeTask()
    import unittest.mock as mock

    with mock.patch.object(worker, "generate_realized_lengths", fake_generate):
        worker._generate_and_publish(task, memmap_dir, "binC")

    # No publish in standalone — the output is already at its final
    # location (local dir == NFS memmap_dir).
    assert task.published == []
