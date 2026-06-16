"""Phase-4 composition + per-binary phase3→4 overlap in ``FullPipelineTask``.

Pins:

* ``get_phases`` returns four phases; ``index`` carries
  ``depends_on=("memmap",)`` but ``barrier=False`` (the pipelined edge,
  so it does not barrier behind full-phase-3 drain) and is
  ``may_be_empty=True``;
* the index phase's two types route to the user output ROOT for both
  ``--source`` and ``--output`` (the publish layer's staging scope owns
  the per-binary ``build_memmap/<name>`` sub-pathing, so there is NO
  ``memmap/`` prefix — that segment is phantom);
* ``_spawn_phase_items(memmap)`` co-spawns each binary's two index items
  in the SAME batch as its memmap item; the rlen item carries the
  cross-phase ``memmap`` dep, the sorted-index item keeps its intra-
  phase rlen dep; and a binary's index items are never double-spawned;
* the dropped ``task_completed_listener`` is gone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dynamic_runner import TaskDep

from dynrunner.binary_selection import BinaryIdentifier, TaskInfo
from dynrunner.build_index.build_index_task import (
    PAYLOAD_MEMMAP_DIR,
    REALIZED_LENGTHS_TYPE,
    SORTED_INDEX_TYPE,
    _rlen_task_id,
)
from dynrunner.build_memmap._publish_scope import memmap_dir_for
from dynrunner.build_memmap.memmap_builder_task import (
    _PHASE_ID as MEMMAP_PHASE_ID,
    _TYPE_ID as MEMMAP_TYPE_ID,
)
from dynrunner.full_pipeline.full_pipeline_task import FullPipelineTask
from dynrunner.full_pipeline.phase_routing import (
    BUILD_INDEX_PHASE,
    BUILD_MEMMAP_PHASE,
)


class _SpyHandle:
    """Captures every ``spawn_tasks`` batch; returns no errors."""

    def __init__(self) -> None:
        self.batches: list[list] = []

    def spawn_tasks(self, items):
        self.batches.append(list(items))
        return []


def _composite_args() -> argparse.Namespace:
    fp = FullPipelineTask()
    parser = argparse.ArgumentParser()
    fp.add_task_arguments(parser)
    return parser.parse_args(
        ["--unified-vocab", "unified_vocab.csv", "--mode", "p75", "--depth", "0"]
    )


def _memmap_item(binary_name: str) -> TaskInfo:
    """A synthetic memmap TaskInfo with ``task_id == binary_name`` (the
    shape ``MemmapBuilderTask.discover_items`` emits)."""
    return TaskInfo(
        path=Path("memmap_sentinel"),
        size=0,
        identifier=BinaryIdentifier(
            binary_name=binary_name,
            platform="",
            compiler="",
            version="",
            opt_level="",
        ),
        phase_id=MEMMAP_PHASE_ID,
        type_id=MEMMAP_TYPE_ID,
        affinity_id=None,
        task_id=binary_name,
        payload={"binary_name": binary_name},
    )


def test_four_phases_index_pipelined_edge_and_may_be_empty() -> None:
    phases = FullPipelineTask().get_phases()
    by_id = {p.phase_id: p for p in phases}
    assert set(by_id) == {"tokenize", "unify_vocab", "memmap", BUILD_INDEX_PHASE}

    index = by_id[BUILD_INDEX_PHASE]
    # Pipelined edge: depends_on memmap, but barrier relaxed → no
    # all-of-phase-3 barrier; phase may legitimately have no run-start items.
    assert index.depends_on == ("memmap",)
    assert index.barrier is False
    assert index.may_be_empty is True
    assert {t.type_id for t in index.types} == {
        REALIZED_LENGTHS_TYPE,
        SORTED_INDEX_TYPE,
    }
    # The chained phases keep their linear dependency AND the default
    # whole-of-upstream barrier.
    assert by_id["unify_vocab"].depends_on == ("tokenize",)
    assert by_id["unify_vocab"].barrier is True
    assert by_id["memmap"].depends_on == ("unify_vocab",)
    assert by_id["memmap"].barrier is True


def test_index_types_route_to_user_output_root() -> None:
    fp = FullPipelineTask()
    args = _composite_args()
    for type_id in (REALIZED_LENGTHS_TYPE, SORTED_INDEX_TYPE):
        argv = fp.build_worker_command_args(
            type_id, args, "/root/out", "/root/out", False
        )
        # Source + output both point at the user output ROOT — NOT a
        # ``memmap/`` subdir. The publish layer mirrors each worker's
        # staging scope (``build_memmap/<name>``) under the run output
        # root, so the index read path must anchor at that root; a
        # ``memmap/`` prefix is phantom (publish never creates it) and
        # would mis-anchor the read against artefacts that land at
        # ``<out>/build_memmap/<name>``.
        assert "--source" in argv and "--output" in argv
        src = argv[argv.index("--source") + 1]
        out = argv[argv.index("--output") + 1]
        assert src == "/root/out"
        assert out == "/root/out"


def test_sorted_index_worker_argv_carries_mode_and_depth() -> None:
    """Closes the threading coverage hole: the sorted-index worker argv
    MUST carry the dispatcher's --mode/--depth (the worker declares them
    required; an omitted pair yields zero .idx). The realized-length
    worker takes no mode/depth (it ignores them), so it is excluded.

    Asserts the actual reduction/depth VALUES the dispatcher parsed reach
    the worker argv, not merely that the flags are present."""
    fp = FullPipelineTask()
    args = _composite_args()  # parsed with --mode p75 --depth 0
    argv = fp.build_worker_command_args(
        SORTED_INDEX_TYPE, args, "/root/out", "/root/out", False
    )
    assert "--mode" in argv and "--depth" in argv
    assert argv[argv.index("--mode") + 1] == "p75"
    assert argv[argv.index("--depth") + 1] == "0"
    # The realized-length worker is configured solely by --source/--output.
    rlen_argv = fp.build_worker_command_args(
        REALIZED_LENGTHS_TYPE, args, "/root/out", "/root/out", False
    )
    assert "--mode" not in rlen_argv and "--depth" not in rlen_argv


def test_composite_parse_fails_loud_when_mode_depth_omitted() -> None:
    """The dispatcher must FAIL at parse time when --mode/--depth are
    omitted (mirroring the sorted-index worker's required=True), rather
    than silently emitting an incomplete worker argv that crashes the
    worker mid-run and produces no .idx."""
    fp = FullPipelineTask()
    parser = argparse.ArgumentParser()
    fp.add_task_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--unified-vocab", "unified_vocab.csv"])


def test_build_index_items_decorate_cross_phase_dep() -> None:
    """Each binary's rlen item gains the cross-phase ``memmap`` edge;
    the sorted-index item keeps its intra-phase rlen edge unchanged; and
    every item carries the binary's resolved ``memmap_dir`` payload."""
    fp = FullPipelineTask()
    fp._user_source = Path("/root/src")  # noqa: SLF001
    fp._user_output = Path("/root/out")  # noqa: SLF001
    memmap_items = [_memmap_item("hello"), _memmap_item("world")]
    index_items = fp._build_index_items_for_memmap(memmap_items)  # noqa: SLF001

    # 2 binaries × 2 types.
    assert len(index_items) == 4

    # The memmap phase publishes each binary under <out>/build_memmap/<name>
    # (the staging scope, mirrored under the run output ROOT — no memmap/
    # segment); each binary's memmap_dir resolves through the shared scope
    # helper against that root.
    expected_memmap_dir = str(memmap_dir_for(Path("/root/out"), "hello"))

    for name in ("hello", "world"):
        rlen = next(
            it
            for it in index_items
            if it.type_id == REALIZED_LENGTHS_TYPE and it.binary_name == name
        )
        sidx = next(
            it
            for it in index_items
            if it.type_id == SORTED_INDEX_TYPE and it.binary_name == name
        )
        # rlen gains the CROSS-PHASE memmap edge (task_id == binary_name).
        assert TaskDep(task_id=name, phase_id=BUILD_MEMMAP_PHASE) in (
            rlen.task_depends_on
        )
        # sidx keeps its INTRA-PHASE rlen edge (no phase_id qualifier).
        assert sidx.task_depends_on == (TaskDep(task_id=_rlen_task_id(name)),)
        # Each binary's memmap_dir rides the payload, resolved against the
        # memmap phase's output root (the user output root, NOT a memmap/
        # subdir — matching where the publish layer writes its artefacts).
        for it in (rlen, sidx):
            assert it.payload[PAYLOAD_MEMMAP_DIR] == str(
                memmap_dir_for(Path("/root/out"), name)
            )
    # Sanity: the helper is the single resolver, not a hand-built literal.
    assert expected_memmap_dir == str(
        memmap_dir_for(Path("/root/out"), "hello")
    )


def test_cospawn_batch_contains_memmap_and_index() -> None:
    """``_spawn_phase_items(memmap)`` injects ONE batch carrying BOTH the
    memmap items and the co-spawned index items."""
    fp = FullPipelineTask()
    handle = _SpyHandle()
    fp._primary_handle = handle  # noqa: SLF001
    fp._user_source = Path("/root/src")  # noqa: SLF001
    fp._user_output = Path("/root/out")  # noqa: SLF001
    fp._args = _composite_args()  # noqa: SLF001

    memmap_items = [_memmap_item("hello"), _memmap_item("world")]
    # Stub the memmap child's discovery so the test does not need a real
    # output tree; the composite still builds + decorates index items.
    fp._memmap.discover_items = lambda source_dir, args: list(memmap_items)  # noqa: SLF001

    fp._spawn_phase_items(BUILD_MEMMAP_PHASE)  # noqa: SLF001

    assert len(handle.batches) == 1
    batch = handle.batches[0]
    # 2 memmap + 4 index (2 binaries × 2 types) = 6 items in one batch.
    assert len(batch) == 6
    memmap_in_batch = [it for it in batch if it.phase_id == MEMMAP_PHASE_ID]
    index_in_batch = [it for it in batch if it.phase_id == BUILD_INDEX_PHASE]
    assert len(memmap_in_batch) == 2
    assert len(index_in_batch) == 4
    # The cross-phase predecessor each rlen names is present in the SAME
    # batch (the known-set that validates the dep at spawn time).
    batch_task_ids = {it.task_id for it in batch}
    for rlen in (it for it in index_in_batch if it.type_id == REALIZED_LENGTHS_TYPE):
        dep = next(
            d for d in rlen.task_depends_on if d.phase_id == BUILD_MEMMAP_PHASE
        )
        assert dep.task_id in batch_task_ids


def test_no_double_spawn_per_binary_index_item() -> None:
    """A binary's index items appear exactly once across the run — the
    memmap batch is fired once per phase, and re-firing would be the only
    double-spawn vector (guarded by a single ``on_phase_end``)."""
    fp = FullPipelineTask()
    handle = _SpyHandle()
    fp._primary_handle = handle  # noqa: SLF001
    fp._user_source = Path("/root/src")  # noqa: SLF001
    fp._user_output = Path("/root/out")  # noqa: SLF001
    fp._args = _composite_args()  # noqa: SLF001

    memmap_items = [_memmap_item("hello")]
    fp._memmap.discover_items = lambda source_dir, args: list(memmap_items)  # noqa: SLF001

    fp._spawn_phase_items(BUILD_MEMMAP_PHASE)  # noqa: SLF001

    all_index_task_ids = [
        it.task_id
        for batch in handle.batches
        for it in batch
        if it.phase_id == BUILD_INDEX_PHASE
    ]
    # rlen:hello + sidx:hello, each exactly once (no duplicates).
    assert len(all_index_task_ids) == len(set(all_index_task_ids))
    assert len(all_index_task_ids) == 2


def test_completion_listener_is_gone() -> None:
    """The per-binary completion listener was dropped in favour of the
    same-batch co-spawn; the method + its state must no longer exist."""
    assert not hasattr(FullPipelineTask, "task_completed_listener")
    assert not hasattr(FullPipelineTask(), "_memmap_task_ids")
    assert not hasattr(FullPipelineTask(), "_index_spawned_binaries")
