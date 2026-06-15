"""``BuildIndexTask.discover_items`` / ``items_for_binary`` contract.

Pins the phase-4 orchestration boundary:

* one phase ``index`` with the two types ``realized_lengths`` +
  ``sorted_index``, ``may_be_empty=True``, no phase dependency;
* per binary, exactly two items — a realized-length item and a
  sorted-index item that DEPENDS on it (``TaskInfo.task_depends_on``
  carries the rlen task's id as a same-phase ``TaskDep``);
* discovery scans ``<name>_index.bin`` (the memmap builder's matched-arm
  sidecar) and honours ``--only`` / ``--max-binaries``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_runner import TaskDep

from dynrunner.build_index.build_index_task import (
    BuildIndexTask,
    PHASE_ID,
    REALIZED_LENGTHS_TYPE,
    SORTED_INDEX_TYPE,
    _rlen_task_id,
    _sidx_task_id,
)


def _args(**overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    BuildIndexTask().add_task_arguments(parser)
    base = ["--mode", "p75", "--depth", "0", "--depth", "3"]
    for k, v in overrides.items():
        base.extend([f"--{k.replace('_', '-')}", str(v)])
    return parser.parse_args(base)


def _make_memmap_dir(tmp_path: Path, names: list[str]) -> Path:
    memmap = tmp_path / "memmap"
    memmap.mkdir()
    for name in names:
        # discover_binaries recognises a binary by its matched-arm
        # `<name>_index.bin`; the content is irrelevant to discovery.
        (memmap / f"{name}_index.bin").write_bytes(b"x")
        # An unmatched-arm sidecar must NOT be counted as a separate
        # binary.
        (memmap / f"{name}_unmatched_index.bin").write_bytes(b"x")
    return memmap


def test_single_phase_two_types_no_dep_may_be_empty() -> None:
    phases = BuildIndexTask().get_phases()
    assert len(phases) == 1
    phase = phases[0]
    assert phase.phase_id == PHASE_ID
    assert phase.depends_on == ()
    assert phase.may_be_empty is True
    assert {t.type_id for t in phase.types} == {
        REALIZED_LENGTHS_TYPE,
        SORTED_INDEX_TYPE,
    }


def test_items_for_binary_emits_dependent_pair() -> None:
    items = BuildIndexTask().items_for_binary("hello")
    assert len(items) == 2
    by_type = {it.type_id: it for it in items}
    rlen = by_type[REALIZED_LENGTHS_TYPE]
    sidx = by_type[SORTED_INDEX_TYPE]

    assert rlen.task_id == _rlen_task_id("hello") == "rlen:hello"
    assert sidx.task_id == _sidx_task_id("hello") == "sidx:hello"
    # rlen has no prerequisite; sidx depends on rlen (same phase).
    assert rlen.task_depends_on == ()
    assert sidx.task_depends_on == (TaskDep(task_id="rlen:hello"),)
    # Both carry the binary_name as the opaque path + phase tag.
    assert str(rlen.path) == "hello"
    assert str(sidx.path) == "hello"
    assert rlen.phase_id == sidx.phase_id == PHASE_ID
    assert rlen.binary_name == sidx.binary_name == "hello"


def test_unique_task_ids_across_types() -> None:
    """The framework rejects (phase_id, task_id) collisions; the two
    items for one binary must carry distinct ids."""
    items = BuildIndexTask().items_for_binary("dup")
    ids = {it.task_id for it in items}
    assert len(ids) == 2


def test_discover_scans_index_bin_and_filters(tmp_path: Path) -> None:
    memmap = _make_memmap_dir(tmp_path, ["alpha", "beta", "gamma"])
    task = BuildIndexTask()

    all_items = list(task.discover_items(memmap, _args()))
    # 3 binaries × 2 types.
    assert len(all_items) == 6
    rlen_names = sorted(
        it.binary_name for it in all_items if it.type_id == REALIZED_LENGTHS_TYPE
    )
    assert rlen_names == ["alpha", "beta", "gamma"]

    only = list(task.discover_items(memmap, _args(only="beta")))
    assert len(only) == 2
    assert {it.binary_name for it in only} == {"beta"}

    capped = list(task.discover_items(memmap, _args(max_binaries=1)))
    assert len(capped) == 2  # 1 binary × 2 types


def test_sorted_index_depends_on_its_own_rlen_per_binary(tmp_path: Path) -> None:
    """Each sorted-index item references ONLY its own binary's rlen id —
    no cross-binary edges."""
    memmap = _make_memmap_dir(tmp_path, ["a", "b"])
    items = list(BuildIndexTask().discover_items(memmap, _args()))
    for it in items:
        if it.type_id == SORTED_INDEX_TYPE:
            assert it.task_depends_on == (
                TaskDep(task_id=_rlen_task_id(it.binary_name)),
            )


def test_worker_argv_forwards_sorted_index_config_only_to_sidx() -> None:
    task = BuildIndexTask()
    args = _args(min_variants=2)
    rlen_argv = task.build_worker_command_args(
        REALIZED_LENGTHS_TYPE, args, Path("/s"), Path("/o"), False
    )
    sidx_argv = task.build_worker_command_args(
        SORTED_INDEX_TYPE, args, Path("/s"), Path("/o"), False
    )
    # Realized-length worker needs no per-run config.
    assert rlen_argv == []
    # Sorted-index worker gets the modes/depths/gate verbatim.
    assert sidx_argv == [
        "--mode", "p75",
        "--depth", "0",
        "--depth", "3",
        "--min-variants", "2",
    ]
