"""``BuildIndexTask.discover_items`` / ``items_for_binary`` contract.

Pins the phase-4 orchestration boundary:

* one phase ``index`` with the two types ``realized_lengths`` +
  ``sorted_index``, ``may_be_empty=True``, no phase dependency;
* per binary, exactly two items — a realized-length item and a
  sorted-index item that DEPENDS on it (``TaskInfo.task_depends_on``
  carries the rlen task's id as a same-phase ``TaskDep``);
* discovery scans ``<name>_index.bin`` (the memmap builder's matched-arm
  sidecar) -- FLAT (sidecars directly in the dir) or NESTED one level
  (per-binary subdirs, the container publish layout) -- and honours
  ``--only`` / ``--max-binaries``;
* each item's payload carries that binary's resolved ``memmap_dir`` so
  the worker reads the sidecars from the right place per layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from dynamic_runner import TaskDep

from dynrunner.build_index.build_index_task import (
    BuildIndexTask,
    PAYLOAD_BINARY_NAME,
    PAYLOAD_MEMMAP_DIR,
    PHASE_ID,
    REALIZED_LENGTHS_TYPE,
    SORTED_INDEX_TYPE,
    _rlen_task_id,
    _sidx_task_id,
)
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_ARMS,
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
    items = BuildIndexTask().items_for_binary("hello", Path("/m/hello"))
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
    # Both items carry the binary's memmap_dir on the payload so the
    # worker reads the sidecars from the right place per layout.
    for it in (rlen, sidx):
        assert it.payload[PAYLOAD_BINARY_NAME] == "hello"
        assert it.payload[PAYLOAD_MEMMAP_DIR] == str(Path("/m/hello"))


def test_unique_task_ids_across_types() -> None:
    """The framework rejects (phase_id, task_id) collisions; the two
    items for one binary must carry distinct ids."""
    items = BuildIndexTask().items_for_binary("dup", Path("/m/dup"))
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
    # Flat layout: every item's memmap_dir is the scanned dir itself.
    for it in all_items:
        assert it.payload[PAYLOAD_MEMMAP_DIR] == str(memmap)

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


def _make_nested_memmap_dir(tmp_path: Path, names: list[str]) -> Path:
    """A NESTED (container-publish) layout: one ``<name>/`` subdir per
    binary, each holding its own ``<name>_index.bin`` matched-arm sidecar
    (plus the unmatched companion that must NOT count as a binary)."""
    parent = tmp_path / "build_memmap"
    parent.mkdir()
    for name in names:
        sub = parent / name
        sub.mkdir()
        (sub / f"{name}_index.bin").write_bytes(b"x")
        (sub / f"{name}_unmatched_index.bin").write_bytes(b"x")
    return parent


def test_discover_nested_multi_binary_resolves_each_subdir(
    tmp_path: Path,
) -> None:
    """Pointed at the PARENT of per-binary subdirs (the container publish
    layout), discovery finds every binary one level down and each item's
    payload memmap_dir resolves to that binary's OWN subdir."""
    parent = _make_nested_memmap_dir(tmp_path, ["nameA", "nameB"])
    items = list(BuildIndexTask().discover_items(parent, _args()))

    # 2 binaries × 2 types, both names found.
    assert len(items) == 4
    rlen_names = sorted(
        it.binary_name for it in items if it.type_id == REALIZED_LENGTHS_TYPE
    )
    assert rlen_names == ["nameA", "nameB"]

    # Each item's memmap_dir is its OWN per-binary subdir, not the parent.
    for it in items:
        assert it.payload[PAYLOAD_MEMMAP_DIR] == str(parent / it.binary_name)


def test_discover_nested_excludes_geometry_sidecar(tmp_path: Path) -> None:
    """A realized-geometry ``_realized_index.bin`` sidecar living in a
    per-binary subdir must NOT be discovered as a separate binary -- the
    GEOMETRY_ARMS exclusion holds in the nested layout exactly as flat."""
    parent = _make_nested_memmap_dir(tmp_path, ["real"])
    sub = parent / "real"
    # Drop a realized-geometry sidecar (its suffix ends in _index.bin but
    # is a length/geometry sidecar, not a binary-existence signal).
    geom_suffix = GEOMETRY_ARMS[0].index_suffix
    (sub / f"real{geom_suffix}").write_bytes(b"x")

    items = list(BuildIndexTask().discover_items(parent, _args()))
    names = {it.binary_name for it in items}
    # Only the matched-arm "real" binary; the geometry sidecar's stem is
    # never surfaced as its own binary.
    assert names == {"real"}


def test_discover_nested_only_and_max_binaries(tmp_path: Path) -> None:
    """``--only`` / ``--max-binaries`` narrow the nested set on name,
    preserving the per-binary subdir resolution for survivors."""
    parent = _make_nested_memmap_dir(tmp_path, ["a", "b", "c"])

    only = list(BuildIndexTask().discover_items(parent, _args(only="b")))
    assert {it.binary_name for it in only} == {"b"}
    for it in only:
        assert it.payload[PAYLOAD_MEMMAP_DIR] == str(parent / "b")

    capped = list(BuildIndexTask().discover_items(parent, _args(max_binaries=1)))
    # Sorted discovery → "a" is the survivor; 1 binary × 2 types.
    assert {it.binary_name for it in capped} == {"a"}


def test_mode_and_depth_required_at_dispatch() -> None:
    """The dispatcher declares --mode/--depth required (mirroring the
    sorted-index worker's required=True): omitting either fails loud at
    parse time, so a missing flag can never silently yield an incomplete
    worker argv and zero .idx."""
    for argv in ([], ["--mode", "max"], ["--depth", "0"]):
        parser = argparse.ArgumentParser()
        BuildIndexTask().add_task_arguments(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


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
