"""Phase-4 composition + per-binary phase3→4 overlap in ``FullPipelineTask``.

Pins:

* ``get_phases`` returns four phases; ``index`` carries NO ``depends_on``
  (so it does not barrier behind full-phase-3 drain) and is
  ``may_be_empty=True``;
* the index phase's two types route to the ``memmap/`` subdir for both
  ``--source`` and ``--output``;
* ``task_completed_listener`` spawns a binary's two index items exactly
  when THAT binary's memmap task completes — and never for non-memmap
  terminals, never twice for the same binary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dynrunner.build_index.build_index_task import (
    REALIZED_LENGTHS_TYPE,
    SORTED_INDEX_TYPE,
    _rlen_task_id,
)
from dynrunner.full_pipeline.full_pipeline_task import FullPipelineTask
from dynrunner.full_pipeline.phase_routing import BUILD_INDEX_PHASE


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


def test_four_phases_index_independent_and_may_be_empty() -> None:
    phases = FullPipelineTask().get_phases()
    by_id = {p.phase_id: p for p in phases}
    assert set(by_id) == {"tokenize", "unify_vocab", "memmap", BUILD_INDEX_PHASE}

    index = by_id[BUILD_INDEX_PHASE]
    # No phase dependency on memmap → no all-of-phase-3 barrier.
    assert index.depends_on == ()
    assert index.may_be_empty is True
    assert {t.type_id for t in index.types} == {
        REALIZED_LENGTHS_TYPE,
        SORTED_INDEX_TYPE,
    }
    # The chained phases keep their linear dependency.
    assert by_id["unify_vocab"].depends_on == ("tokenize",)
    assert by_id["memmap"].depends_on == ("unify_vocab",)


def test_index_types_route_to_memmap_subdir() -> None:
    fp = FullPipelineTask()
    args = _composite_args()
    for type_id in (REALIZED_LENGTHS_TYPE, SORTED_INDEX_TYPE):
        argv = fp.build_worker_command_args(
            type_id, args, "/root/out", "/root/out", False
        )
        # Source + output both point at the memmap/ subdir.
        assert "--source" in argv and "--output" in argv
        src = argv[argv.index("--source") + 1]
        out = argv[argv.index("--output") + 1]
        assert src == str(Path("/root/out") / "memmap")
        assert out == str(Path("/root/out") / "memmap")


def _prime_memmap_ids(fp: FullPipelineTask, names: list[str]) -> _SpyHandle:
    """Wire a spy handle + seed the recorded memmap task-ids the way the
    composite would after spawning phase 3."""
    handle = _SpyHandle()
    fp._primary_handle = handle  # noqa: SLF001 - test reaches recorded state
    fp._memmap_task_ids.update(names)  # noqa: SLF001
    return handle


def test_listener_spawns_index_on_memmap_completion() -> None:
    fp = FullPipelineTask()
    handle = _prime_memmap_ids(fp, ["hello", "world"])

    fp.task_completed_listener("hello", True, None, None)
    assert len(handle.batches) == 1
    spawned = handle.batches[0]
    assert {it.type_id for it in spawned} == {
        REALIZED_LENGTHS_TYPE,
        SORTED_INDEX_TYPE,
    }
    assert all(it.binary_name == "hello" for it in spawned)
    # The sorted-index item depends on hello's realized-length item.
    sidx = next(it for it in spawned if it.type_id == SORTED_INDEX_TYPE)
    assert sidx.task_depends_on[0].task_id == _rlen_task_id("hello")


def test_listener_ignores_non_memmap_terminals() -> None:
    fp = FullPipelineTask()
    handle = _prime_memmap_ids(fp, ["hello"])

    # A tokenize/unify/index terminal (id not in the memmap set) is a no-op.
    fp.task_completed_listener("unify_vocab", True, None, None)
    fp.task_completed_listener("rlen:hello", True, None, None)
    fp.task_completed_listener(None, True, None, None)
    assert handle.batches == []


def test_listener_is_idempotent_per_binary() -> None:
    fp = FullPipelineTask()
    handle = _prime_memmap_ids(fp, ["hello"])

    fp.task_completed_listener("hello", True, None, None)
    # A duplicate completion signal (retry / failover replay) must not
    # re-spawn.
    fp.task_completed_listener("hello", True, None, None)
    assert len(handle.batches) == 1


def test_listener_spawns_on_failed_memmap_too() -> None:
    """Barrier-on-completion: a FAILED memmap still hands off to phase 4
    (the index workers surface their own missing-input miss)."""
    fp = FullPipelineTask()
    handle = _prime_memmap_ids(fp, ["broken"])

    fp.task_completed_listener("broken", False, "non_recoverable", "boom")
    assert len(handle.batches) == 1
    assert all(it.binary_name == "broken" for it in handle.batches[0])
