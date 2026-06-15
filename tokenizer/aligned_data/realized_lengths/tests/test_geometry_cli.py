"""CLI smoke: the geometry sidecars are emitted ALONGSIDE the length pair.

The single CLI run writes BOTH sidecar families per binary; this pins
that the geometry pair is present (per arm) for every discovered binary
without disturbing the length pair the existing ``test_cli`` covers.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tokenizer.aligned_data.realized_lengths import (
    GEOMETRY_MATCHED_ARM,
    GEOMETRY_UNMATCHED_ARM,
    MATCHED_ARM,
    RealizedGeometryReader,
    realized_geometry_present,
    realized_lengths_present,
)
from tokenizer.aligned_data.realized_lengths.__main__ import main as cli_main
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
)


def _two_binary_dir(tmp_path: Path) -> Path:
    src_a = build_combined_fixture(tmp_path / "src_a")
    src_b = build_many_variant_section_fixture(tmp_path / "src_b")
    shared = tmp_path / "shared"
    shared.mkdir()
    for src, new_name in ((src_a, "binA"), (src_b, "binB")):
        for f in src.iterdir():
            if not f.is_file():
                continue
            assert f.name.startswith("sortbin")
            shutil.copy(f, shared / f"{new_name}{f.name[len('sortbin'):]}")
    return shared


def test_cli_writes_geometry_alongside_lengths(tmp_path: Path) -> None:
    shared = _two_binary_dir(tmp_path)
    rc = cli_main(["--input-dir", str(shared)])
    assert rc == 0
    for name in ("binA", "binB"):
        for arm in (GEOMETRY_MATCHED_ARM, GEOMETRY_UNMATCHED_ARM):
            assert realized_geometry_present(shared, name, arm)
        # The length pair is still present too (families coexist).
        assert realized_lengths_present(shared, name, MATCHED_ARM)
        r = RealizedGeometryReader.open(shared, name, GEOMETRY_MATCHED_ARM)
        try:
            assert r.n_sections > 0
            assert r.body_lengths.size == r.id_counts.size == r.value_counts.size
        finally:
            r.close()
