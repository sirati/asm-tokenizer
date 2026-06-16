"""``tokenizer.aligned_data.binary_discovery`` scan + selection contract.

Pins the relocated discovery helpers (moved out of ``tools.batch_smoke``
so they ship in the production image): a binary is recognised by its
matched-arm ``<name>_index.bin`` sidecar; the ``_unmatched_index.bin``
companion and the realized-length CSR index sidecars (keyed off the
``ARMS`` grammar) are excluded; ``--only`` / ``--max-binaries`` select in
sorted order, allow-list before cap.

Two layouts are auto-detected by entry kind in one scan: FLAT (sidecars
directly in the dir) and NESTED one level (per-binary subdirs, the
container publish layout). Each discovered binary carries the directory
its sidecars live in (``DiscoveredBinary.memmap_dir``).
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.binary_discovery import (
    DiscoveredBinary,
    discover_binaries,
    filter_binaries,
)
from tokenizer.aligned_data.realized_lengths._format import ARMS
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_ARMS,
)


def _touch(directory: Path, name: str) -> None:
    (directory / name).write_bytes(b"x")


def _names(binaries) -> list[str]:
    return [b.name for b in binaries]


def test_discover_recognises_matched_arm_index_bin(tmp_path: Path) -> None:
    _touch(tmp_path, "alpha_index.bin")
    _touch(tmp_path, "beta_index.bin")
    # Sorted, de-duplicated by matched arm.
    found = discover_binaries(tmp_path)
    assert _names(found) == ["alpha", "beta"]
    # Flat layout: each binary's memmap_dir is the scanned directory.
    for b in found:
        assert b.memmap_dir == tmp_path


def test_discover_excludes_unmatched_companion(tmp_path: Path) -> None:
    _touch(tmp_path, "gamma_index.bin")
    _touch(tmp_path, "gamma_unmatched_index.bin")
    # The unmatched companion must not surface gamma a second time.
    assert _names(discover_binaries(tmp_path)) == ["gamma"]


def test_discover_excludes_realized_sidecars_both_grammars(tmp_path: Path) -> None:
    _touch(tmp_path, "delta_index.bin")
    # Once the realized passes have run, BOTH sidecar families co-exist:
    # the length-CSR arms AND the realized-geometry RLG3 arms. They end in
    # ``_index.bin`` but are NOT binary signals — sourced from both arm
    # grammars so the exclusion never drifts from the generators.
    for arm in (*ARMS, *GEOMETRY_ARMS):
        _touch(tmp_path, f"delta{arm.index_suffix}")
    assert _names(discover_binaries(tmp_path)) == ["delta"]


def test_discover_ignores_non_index_and_empty_subdirs(tmp_path: Path) -> None:
    _touch(tmp_path, "eps_index.bin")
    _touch(tmp_path, "eps_data.bin")
    _touch(tmp_path, "notes.txt")
    # An empty subdir contributes no binary (nested recursion finds no
    # matched-arm sidecar one level down).
    (tmp_path / "subdir").mkdir()
    assert _names(discover_binaries(tmp_path)) == ["eps"]


def test_discover_nested_per_binary_subdirs(tmp_path: Path) -> None:
    """NESTED layout: per-binary subdirs each holding their own
    ``<name>_index.bin``. Discovery finds every binary one level down and
    pairs it with its own subdir."""
    for name in ("nameA", "nameB"):
        sub = tmp_path / name
        sub.mkdir()
        _touch(sub, f"{name}_index.bin")
        _touch(sub, f"{name}_unmatched_index.bin")

    found = discover_binaries(tmp_path)
    assert _names(found) == ["nameA", "nameB"]
    by_name = {b.name: b for b in found}
    assert by_name["nameA"].memmap_dir == tmp_path / "nameA"
    assert by_name["nameB"].memmap_dir == tmp_path / "nameB"


def test_discover_nested_excludes_realized_sidecars(tmp_path: Path) -> None:
    """The realized/unmatched sidecar exclusions hold in a nested subdir
    exactly as at the top level (same shared exclusion helper)."""
    sub = tmp_path / "real"
    sub.mkdir()
    _touch(sub, "real_index.bin")
    _touch(sub, "real_unmatched_index.bin")
    for arm in (*ARMS, *GEOMETRY_ARMS):
        _touch(sub, f"real{arm.index_suffix}")
    found = discover_binaries(tmp_path)
    assert _names(found) == ["real"]
    assert found[0].memmap_dir == sub


def test_discover_mixed_flat_and_nested(tmp_path: Path) -> None:
    """A flat sidecar at the top level and a nested per-binary subdir
    coexist: both are discovered, each with its correct memmap_dir."""
    _touch(tmp_path, "flat_index.bin")
    sub = tmp_path / "nested"
    sub.mkdir()
    _touch(sub, "nested_index.bin")

    found = discover_binaries(tmp_path)
    by_name = {b.name: b for b in found}
    assert set(by_name) == {"flat", "nested"}
    assert by_name["flat"].memmap_dir == tmp_path
    assert by_name["nested"].memmap_dir == sub


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


def _disc(name: str) -> DiscoveredBinary:
    return DiscoveredBinary(name=name, memmap_dir=Path("/m") / name)


def test_filter_only_then_max_binaries() -> None:
    binaries = [_disc(n) for n in ("a", "b", "c", "d")]
    # --only allow-list applied first.
    assert _names(filter_binaries(binaries, only="b,d", max_binaries=None)) == [
        "b",
        "d",
    ]
    # --max-binaries cap applied AFTER --only, preserving order.
    assert _names(
        filter_binaries(binaries, only="b,c,d", max_binaries=2)
    ) == ["b", "c"]
    # No filters → identity (order preserved); memmap_dir survives.
    identity = filter_binaries(binaries, only=None, max_binaries=None)
    assert identity == binaries
    # Cap alone.
    assert _names(filter_binaries(binaries, only=None, max_binaries=1)) == ["a"]
