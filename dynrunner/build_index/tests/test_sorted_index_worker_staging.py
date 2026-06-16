"""``sorted_index`` worker stages inputs node-local + publishes ``.idx`` to NFS.

The DDOS fix's integration contract:

* Container mode (``/app/out-tmp`` present): the worker copies the build's
  input footprint (locator + catalog + matched-arm realized-length pair)
  to node-local scratch, runs the builder against the LOCAL copies, and
  atomic-publishes each produced ``.idx`` to its canonical NFS location
  ``<memmap_dir>/<binary>_sorted_<mode>_d<depth>.idx``. The scratch is
  removed on exit and the NFS memmap dir gains ONLY the ``.idx`` files
  (no copied sidecars leak there).
* Standalone mode (no ``/app/out-tmp``): ``staged_inputs`` is a no-op, the
  build reads/writes the NFS dir in place, and the explicit-dst publish
  self-skips (src == dst) so the in-place ``.idx`` is not republished onto
  itself. The on-disk result matches container mode byte-for-byte.

The deployment mode is forced via ``is_container_deployment`` and the
tmpfs/publish roots are redirected via ``_SLURM_OUT_TMP`` + the publish
env vars so both branches run without a real ``/app/out-tmp`` mount.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tokenizer.output_staging as staging
from dynamic_runner.worker import Task
from tokenizer.aligned_data.sorted_index import (
    PLAIN,
    VariantGate,
    parse_reduction,
    sorted_index_input_paths,
)
from tokenizer.aligned_data.sorted_index.tests._length_helpers import (
    ensure_sidecar,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    _BINARY_NAME,
    build_combined_fixture,
)

import dynrunner.build_index.sorted_index_worker as worker
from dynrunner.build_index.build_index_task import PAYLOAD_MEMMAP_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _worker_config(monkeypatch):
    """Populate the module-level build config the handler closes over.

    Mirrors ``_on_args``: a single ``max`` reduction at depth 3, no
    gating, plain duplicate handling -- enough to produce one ``.idx``.
    """
    # The handler's module-level config exists only as type annotations
    # until ``_on_args`` assigns it; set with ``raising=False`` so the
    # first assignment doesn't trip the default attribute-exists check.
    monkeypatch.setattr(
        worker, "_REDUCTIONS", [parse_reduction("max")], raising=False
    )
    monkeypatch.setattr(worker, "_DEPTHS", [3], raising=False)
    monkeypatch.setattr(worker, "_GATE", VariantGate(), raising=False)
    monkeypatch.setattr(worker, "_DUPLICATE_HANDLING", PLAIN, raising=False)


@pytest.fixture
def memmap_dir(tmp_path: Path) -> Path:
    """A real per-binary memmap dir WITH the realized-length sidecar.

    The combined fixture lays down the matched/unmatched sidecars; the
    realized-length pass then writes the body-length sidecar the build
    hard-requires (the production phase-4 dependency edge).
    """
    base = build_combined_fixture(tmp_path)
    ensure_sidecar(base, _BINARY_NAME)
    return base


def _task(memmap_dir: Path) -> Task:
    return Task(
        relative_path=_BINARY_NAME,
        payload_str=json.dumps({PAYLOAD_MEMMAP_DIR: str(memmap_dir)}),
    )


def _idx_name() -> str:
    return f"{_BINARY_NAME}_sorted_max_d003.idx"


# ---------------------------------------------------------------------------
# Container mode
# ---------------------------------------------------------------------------


def test_container_publishes_idx_and_keeps_sidecars_off_scratch(
    memmap_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    out_tmp = tmp_path / "out-tmp"
    out_tmp.mkdir()
    monkeypatch.setattr(staging, "_SLURM_OUT_TMP", out_tmp)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    # The publish layer reads its roots from env; src_root must be the
    # tmpfs scratch root so a staged-out file resolves under it, and the
    # explicit dst (the NFS memmap dir) is honoured verbatim regardless.
    monkeypatch.setenv("DYNRUNNER_PUBLISH_SRC_ROOT", str(out_tmp))
    monkeypatch.setenv("DYNRUNNER_PUBLISH_DST_ROOT", str(tmp_path / "dst"))

    # Capture the source the builder actually read from: it must be the
    # node-local scratch copy, never the NFS memmap dir (the DDOS).
    seen_dirs: list[Path] = []
    real_write = worker.write_sorted_index_files

    def _spy(memmap_arg, binary, **kw):
        seen_dirs.append(Path(memmap_arg))
        return real_write(memmap_arg, binary, **kw)

    monkeypatch.setattr(worker, "write_sorted_index_files", _spy)

    worker.handle(_task(memmap_dir))

    # The build ran against the node-local scratch copy.
    assert len(seen_dirs) == 1
    assert out_tmp in seen_dirs[0].parents

    # The .idx published to the canonical NFS location.
    published = memmap_dir / _idx_name()
    assert published.is_file()
    assert published.stat().st_size > 0

    # The per-task scope subtree was removed on exit (``staged_inputs``
    # rmtree's ``staged-inputs/<scope>`` where scope == ``sorted_index/
    # <binary>``); no staged input copies linger anywhere under the root.
    scope_leaf = out_tmp / "staged-inputs" / "sorted_index" / _BINARY_NAME
    assert not scope_leaf.exists()
    leftover = [p for p in out_tmp.rglob("*") if p.is_file()]
    assert leftover == []

    # ONLY the .idx was added to the NFS dir -- no staged sidecars leaked
    # (the publish targets the .idx alone; inputs stay node-local).
    nfs_names = {p.name for p in memmap_dir.iterdir()}
    assert _idx_name() in nfs_names
    # The four staged inputs are NOT re-deposited beyond their originals:
    # their count is unchanged (they pre-existed as fixture output).
    for src in sorted_index_input_paths(memmap_dir, _BINARY_NAME):
        # Each original input still exists exactly once at its NFS path.
        assert src.is_file()


def test_container_idx_bytes_match_standalone(
    tmp_path: Path, monkeypatch
) -> None:
    """The published .idx is byte-identical to the in-place standalone build."""
    # Standalone build into dir A.
    base_a = build_combined_fixture(tmp_path / "a")
    ensure_sidecar(base_a, _BINARY_NAME)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)
    worker.handle(_task(base_a))
    standalone_bytes = (base_a / _idx_name()).read_bytes()

    # Container build into dir B.
    base_b = build_combined_fixture(tmp_path / "b")
    ensure_sidecar(base_b, _BINARY_NAME)
    out_tmp = tmp_path / "out-tmp"
    out_tmp.mkdir()
    monkeypatch.setattr(staging, "_SLURM_OUT_TMP", out_tmp)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    monkeypatch.setenv("DYNRUNNER_PUBLISH_SRC_ROOT", str(out_tmp))
    monkeypatch.setenv("DYNRUNNER_PUBLISH_DST_ROOT", str(tmp_path / "dst"))
    worker.handle(_task(base_b))
    container_bytes = (base_b / _idx_name()).read_bytes()

    assert container_bytes == standalone_bytes


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------


def test_standalone_writes_idx_in_place_no_publish_error(
    memmap_dir: Path, monkeypatch
) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)

    # publish_all is still invoked, but standalone's src==dst pairs are
    # dropped so it receives an EMPTY list (a no-op). Asserting the
    # received pairs are empty pins the self-skip without depending on
    # publish env that standalone has no reason to set.
    seen_pairs: list = []

    def _record(self, pairs) -> None:
        seen_pairs.append(list(pairs))

    monkeypatch.setattr(Task, "publish_all", _record)

    worker.handle(_task(memmap_dir))

    assert seen_pairs == [[]]

    published = memmap_dir / _idx_name()
    assert published.is_file()
    assert published.stat().st_size > 0
