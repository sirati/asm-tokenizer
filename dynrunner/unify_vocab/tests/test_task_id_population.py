"""Assert ``VocabUnifierTask.discover_items`` emits ``TaskInfo.task_id``
populated with the per-run sentinel.

The framework's memprofile sampler keys output filenames on the per-task
identity string carried by ``TaskInfo.task_id``. Vocab unification is
single-task-per-run; the type_id sentinel ``"unify_vocab"`` is the
natural canonical identifier (same shape as the ``unified_vocab.csv``
done-marker the task already emits).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dynrunner.binary_selection import add_asm_selection_arguments
from dynrunner.unify_vocab.vocab_unifier_task import VocabUnifierTask


# Canonical 4-axis stem (matches the framework's default
# ``platform-compiler-version-optimisationlevel_binaryname`` file_format).
_STEM = "x64-gcc-7-Os_minigzip"


def _build_args(source_dir: Path, output_dir: Path) -> argparse.Namespace:
    """Construct a Namespace with all defaults the selection layer expects.

    Uses the same parser the task registers in production so we don't
    duplicate the defaults block.
    """
    parser = argparse.ArgumentParser()
    add_asm_selection_arguments(parser)
    VocabUnifierTask().add_private_task_arguments(parser)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--list-files", action="store_true")
    return parser.parse_args(
        [
            "--source", str(source_dir),
            "--output", str(output_dir),
        ]
    )


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """Produce one ``_output.csv`` the unifier walks see."""
    source = tmp_path / "source"
    source.mkdir()
    (source / f"{_STEM}_output.csv").write_bytes(b"")
    output = tmp_path / "output"
    output.mkdir()
    return source


def test_task_id_equals_sentinel(synthetic_csv: Path, tmp_path: Path) -> None:
    source = synthetic_csv
    output = tmp_path / "output"
    args = _build_args(source, output)
    task = VocabUnifierTask()
    items = list(task.discover_items(source, args))
    assert items, "discover_items should emit one TaskInfo for the synthetic CSV"
    [item] = items
    assert item.task_id == "unify_vocab"
