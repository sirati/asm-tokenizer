"""CLI smoke for ``python -m tokenizer.aligned_data.realized_lengths``.

In-process invocation via :func:`__main__.main` (no fresh interpreter):
a minimal run over a tmp fixture dir writes the four sidecars per binary
and they read back; ``--only`` / ``--max-binaries`` scope the work like
the sorted_index CLI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tokenizer.aligned_data.realized_lengths import (
    MATCHED_ARM,
    UNMATCHED_ARM,
    RealizedLengths,
    realized_lengths_present,
)
from tokenizer.aligned_data.realized_lengths.__main__ import main as cli_main
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
)


def _two_binary_dir(tmp_path: Path) -> Path:
    """Lay down two distinct binaries (binA, binB) in one memmap dir."""
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


def test_cli_writes_sidecars_for_each_binary(tmp_path: Path) -> None:
    shared = _two_binary_dir(tmp_path)
    rc = cli_main(["--input-dir", str(shared)])
    assert rc == 0
    for name in ("binA", "binB"):
        for arm in (MATCHED_ARM, UNMATCHED_ARM):
            assert realized_lengths_present(shared, name, arm)
        r = RealizedLengths.open(shared, name, MATCHED_ARM)
        try:
            assert r.n_sections > 0
        finally:
            r.close()


def test_cli_only_filters(tmp_path: Path) -> None:
    shared = _two_binary_dir(tmp_path)
    rc = cli_main(["--input-dir", str(shared), "--only", "binA"])
    assert rc == 0
    assert realized_lengths_present(shared, "binA", MATCHED_ARM)
    assert not realized_lengths_present(shared, "binB", MATCHED_ARM)


def test_cli_max_binaries_caps(tmp_path: Path) -> None:
    shared = _two_binary_dir(tmp_path)
    rc = cli_main(["--input-dir", str(shared), "--max-binaries", "1"])
    assert rc == 0
    # Discovery sorts names, so binA is processed, binB capped out.
    assert realized_lengths_present(shared, "binA", MATCHED_ARM)
    assert not realized_lengths_present(shared, "binB", MATCHED_ARM)
