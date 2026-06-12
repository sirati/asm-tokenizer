"""Assert ``TokenizerTask.build_worker_command_args`` threads the
per-node worker count to the worker as ``--workers-per-node <N>``.

The framework re-runs this method per-secondary against that node's
forwarded argv, so resolving ``--cores`` via ``parse_cores`` here lands
the correct per-node count in the worker's CLI — which the worker turns
into a ceil-divided Ghidra JVM processor cap.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from dynrunner.tokenize.tokenizer_task import TokenizerTask


def _args(cores: str) -> Namespace:
    # Mirrors the framework's ``--cores`` spec string surface; other
    # attributes the method touches default to harmless values.
    return Namespace(cores=cores, simulate_errors=None)


def test_emits_workers_per_node_for_exact_core_spec(tmp_path: Path) -> None:
    cmd = TokenizerTask().build_worker_command_args(
        "tokenizer", _args("7"), tmp_path, tmp_path, skip_existing=False
    )
    assert "--workers-per-node" in cmd
    idx = cmd.index("--workers-per-node")
    # ``parse_cores("7")`` is the exact-N form → 7 workers per node.
    assert cmd[idx + 1] == "7"


def test_workers_per_node_value_is_a_positive_int_string(tmp_path: Path) -> None:
    # ``"0"`` = all detected cores; whatever this host resolves to, it
    # must be a parseable positive int so the worker's argparse accepts
    # ``type=int`` and the cap computation stays well-defined.
    cmd = TokenizerTask().build_worker_command_args(
        "tokenizer", _args("0"), tmp_path, tmp_path, skip_existing=False
    )
    idx = cmd.index("--workers-per-node")
    assert int(cmd[idx + 1]) >= 1
