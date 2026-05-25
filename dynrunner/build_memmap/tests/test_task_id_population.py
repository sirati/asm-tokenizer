"""Assert ``MemmapBuilderTask.discover_items`` emits ``TaskInfo.task_id``
populated with the binary_name group identity.

The framework's memprofile sampler keys output filenames on the per-task
identity string carried by ``TaskInfo.task_id``. For the memmap builder,
exactly one TaskInfo is emitted per ``binary_name`` group — so the
group's name is the natural canonical identifier, matching the
``_index.bin`` filename slot produced by ``get_output_filename_pattern``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dynrunner.build_memmap.memmap_builder_task import MemmapBuilderTask
from dynrunner.binary_selection import add_asm_selection_arguments


# Canonical 4-axis stem (matches the framework's default
# ``platform-compiler-version-optimisationlevel_binaryname`` file_format).
_STEM = "x64-gcc-7-Os_minigzip"


def _build_args(source_dir: Path, output_dir: Path) -> argparse.Namespace:
    """Construct a Namespace with all defaults the selection layer expects.

    Uses the same parser the task registers in production so we don't
    duplicate the defaults block. The framework owns ``--source`` /
    ``--output`` / ``--list-files``; we register them here so
    ``process_selection_arguments`` reads them off the Namespace.
    ``--unified-vocab`` is required by
    ``MemmapBuilderTask.add_private_task_arguments`` but unused in
    ``discover_items``; pass a placeholder.
    """
    parser = argparse.ArgumentParser()
    add_asm_selection_arguments(parser)
    MemmapBuilderTask().add_private_task_arguments(parser)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--list-files", action="store_true")
    return parser.parse_args(
        [
            "--source", str(source_dir),
            "--output", str(output_dir),
            "--unified-vocab", str(source_dir / "unified_vocab.csv"),
        ]
    )


@pytest.fixture
def synthetic_pair(tmp_path: Path) -> tuple[Path, str]:
    """Produce a CSV + mapping + meta triple that the discover walks match."""
    source = tmp_path / "source"
    source.mkdir()
    binary_name = "minigzip"
    csv = source / f"{_STEM}_output.csv"
    mapping = source / f"{_STEM}_output.mapping.b64c"
    meta = source / f"{_STEM}_meta.json"
    for p in (csv, mapping, meta):
        p.write_bytes(b"")
    output = tmp_path / "output"
    output.mkdir()
    return source, binary_name


def test_task_id_equals_binary_name(synthetic_pair: tuple[Path, str], tmp_path: Path) -> None:
    source, binary_name = synthetic_pair
    output = tmp_path / "output"
    args = _build_args(source, output)
    task = MemmapBuilderTask()
    items = list(task.discover_items(source, args))
    assert items, "discover_items should emit one TaskInfo for the synthetic pair"
    [item] = items
    assert item.task_id == binary_name
