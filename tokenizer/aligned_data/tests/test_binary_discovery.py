"""``tokenizer.aligned_data.binary_discovery`` scan + selection contract.

Pins the relocated discovery helpers (moved out of ``tools.batch_smoke``
so they ship in the production image): a binary is recognised by its
matched-arm ``<name>_index.bin`` sidecar; the ``_unmatched_index.bin``
companion and the realized-length CSR index sidecars (keyed off the
``ARMS`` grammar) are excluded; ``--only`` / ``--max-binaries`` select in
sorted order, allow-list before cap.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.binary_discovery import (
    discover_binaries,
    filter_binaries,
)
from tokenizer.aligned_data.realized_lengths._format import ARMS
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_ARMS,
)


def _touch(directory: Path, name: str) -> None:
    (directory / name).write_bytes(b"x")


def test_discover_recognises_matched_arm_index_bin(tmp_path: Path) -> None:
    _touch(tmp_path, "alpha_index.bin")
    _touch(tmp_path, "beta_index.bin")
    # Sorted, de-duplicated by matched arm.
    assert discover_binaries(tmp_path) == ["alpha", "beta"]


def test_discover_excludes_unmatched_companion(tmp_path: Path) -> None:
    _touch(tmp_path, "gamma_index.bin")
    _touch(tmp_path, "gamma_unmatched_index.bin")
    # The unmatched companion must not surface gamma a second time.
    assert discover_binaries(tmp_path) == ["gamma"]


def test_discover_excludes_realized_sidecars_both_grammars(tmp_path: Path) -> None:
    _touch(tmp_path, "delta_index.bin")
    # Once the realized passes have run, BOTH sidecar families co-exist:
    # the length-CSR arms AND the realized-geometry RLG3 arms. They end in
    # ``_index.bin`` but are NOT binary signals — sourced from both arm
    # grammars so the exclusion never drifts from the generators.
    for arm in (*ARMS, *GEOMETRY_ARMS):
        _touch(tmp_path, f"delta{arm.index_suffix}")
    assert discover_binaries(tmp_path) == ["delta"]


def test_discover_ignores_non_index_and_subdirs(tmp_path: Path) -> None:
    _touch(tmp_path, "eps_index.bin")
    _touch(tmp_path, "eps_data.bin")
    _touch(tmp_path, "notes.txt")
    (tmp_path / "subdir").mkdir()
    assert discover_binaries(tmp_path) == ["eps"]


def test_shipped_consumers_do_not_import_from_tools() -> None:
    """Guard the relocation: the production-image modules that select
    their binary set MUST NOT import from ``tools`` (it is not packaged —
    such an import crashed the mesh workers). They must import the scan
    from the owned ``tokenizer.aligned_data.binary_discovery`` instead."""
    import importlib

    for module_name in (
        "dynrunner.build_index.build_index_task",
        "tokenizer.aligned_data.sorted_index.__main__",
        "tokenizer.aligned_data.realized_lengths.__main__",
    ):
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text()
        assert "from tools" not in source and "import tools" not in source, (
            f"{module_name} imports from tools (absent in the prod image)"
        )
        # ...and it reaches the owned discovery module.
        assert "tokenizer.aligned_data.binary_discovery" in source


def test_filter_only_then_max_binaries() -> None:
    names = ["a", "b", "c", "d"]
    # --only allow-list applied first.
    assert filter_binaries(names, only="b,d", max_binaries=None) == ["b", "d"]
    # --max-binaries cap applied AFTER --only, preserving order.
    assert filter_binaries(names, only="b,c,d", max_binaries=2) == ["b", "c"]
    # No filters → identity (order preserved).
    assert filter_binaries(names, only=None, max_binaries=None) == names
    # Cap alone.
    assert filter_binaries(names, only=None, max_binaries=1) == ["a"]
