"""Tests for :mod:`sorted_index.__main__` CLI.

In-process invocation via :func:`sorted_index.__main__.main` so the
tests don't pay the cold-start of a fresh interpreter.  Coverage:

* Smoke: a minimal invocation writes files matching the canonical
  grammar.
* ``--only`` allow-list filters discovered binaries.
* ``--max-binaries`` caps the per-invocation work.
* Multiple ``--mode`` flags produce multiple ``.idx`` files (one per
  reduction) in one invocation -- the §D8 cost-amortising property at
  the CLI boundary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

from tokenizer.aligned_data.sorted_index.__main__ import main as cli_main

from .fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
)


_BINARY_NAME = "sortbin"

_FILENAME_RE = re.compile(
    r"^(?P<binary>.+)_sorted_(?P<mode>max|p\d{2})_d(?P<depth>\d{3})\.idx$",
)


# ---------------------------------------------------------------------------
# Multi-binary helpers
# ---------------------------------------------------------------------------


def _build_two_binary_dir(tmp_path: Path) -> Path:
    """Lay down two distinct binaries in one memmap dir under ``tmp_path``.

    The two fixtures own independent corpora but write into the same
    ``binary_name = "sortbin"`` so reusing them in one shared memmap
    dir would collide.  Instead each fixture is built in its own
    sub-temp; then the matched-set sidecars are copied across with
    distinct binary-name prefixes so CLI discovery sees two binaries.
    """
    src_a = build_combined_fixture(tmp_path / "src_a")
    src_b = build_many_variant_section_fixture(tmp_path / "src_b")

    shared = tmp_path / "shared_memmap"
    shared.mkdir()
    for src, new_name in ((src_a, "binA"), (src_b, "binB")):
        for f in src.iterdir():
            if not f.is_file():
                continue
            assert f.name.startswith(_BINARY_NAME + "_"), (
                f"fixture filename {f.name!r} does not start with the "
                f"documented binary name prefix"
            )
            new_filename = new_name + f.name[len(_BINARY_NAME):]
            (shared / new_filename).write_bytes(f.read_bytes())
    return shared


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_cli_smoke_writes_canonical_filenames(tmp_path: Path) -> None:
    """One-binary --mode max --depth 3 writes exactly one matching .idx."""
    base = build_combined_fixture(tmp_path)
    rc = cli_main([
        "--input-dir", str(base),
        "--mode", "max",
        "--depth", "3",
    ])
    assert rc == 0
    written = sorted(p.name for p in base.glob("*_sorted_*.idx"))
    assert written == [f"{_BINARY_NAME}_sorted_max_d003.idx"]


def test_cli_multiple_modes_produce_multiple_files(tmp_path: Path) -> None:
    """K --mode flags produce K .idx files in one CLI invocation (§D8)."""
    base = build_combined_fixture(tmp_path)
    rc = cli_main([
        "--input-dir", str(base),
        "--mode", "max",
        "--mode", "p50",
        "--mode", "p95",
        "--depth", "3",
    ])
    assert rc == 0
    written: List[Path] = sorted(base.glob("*_sorted_*.idx"))
    written_names = {p.name for p in written}
    assert written_names == {
        f"{_BINARY_NAME}_sorted_max_d003.idx",
        f"{_BINARY_NAME}_sorted_p50_d003.idx",
        f"{_BINARY_NAME}_sorted_p95_d003.idx",
    }
    for path in written:
        m = _FILENAME_RE.match(path.name)
        assert m is not None
        assert m.group("depth") == "003"


def test_cli_output_dir(tmp_path: Path) -> None:
    """``--output-dir`` routes .idx files away from the memmap dir."""
    base = build_combined_fixture(tmp_path)
    out_dir = tmp_path / "elsewhere"
    rc = cli_main([
        "--input-dir", str(base),
        "--mode", "max",
        "--depth", "3",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    assert out_dir.is_dir()
    assert (out_dir / f"{_BINARY_NAME}_sorted_max_d003.idx").is_file()
    assert list(base.glob("*_sorted_*.idx")) == []


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_cli_only_filters_to_specified_binary(tmp_path: Path) -> None:
    """``--only`` allow-lists a subset of discovered binaries."""
    shared = _build_two_binary_dir(tmp_path)
    rc = cli_main([
        "--input-dir", str(shared),
        "--mode", "max",
        "--depth", "3",
        "--only", "binA",
    ])
    assert rc == 0
    written = sorted(p.name for p in shared.glob("*_sorted_*.idx"))
    assert written == ["binA_sorted_max_d003.idx"]


def test_cli_only_comma_separated(tmp_path: Path) -> None:
    """``--only A,B`` allow-lists both binaries."""
    shared = _build_two_binary_dir(tmp_path)
    rc = cli_main([
        "--input-dir", str(shared),
        "--mode", "max",
        "--depth", "3",
        "--only", "binA,binB",
    ])
    assert rc == 0
    written = sorted(p.name for p in shared.glob("*_sorted_*.idx"))
    assert written == [
        "binA_sorted_max_d003.idx",
        "binB_sorted_max_d003.idx",
    ]


def test_cli_max_binaries_caps(tmp_path: Path) -> None:
    """``--max-binaries N`` caps after the allow-list."""
    shared = _build_two_binary_dir(tmp_path)
    rc = cli_main([
        "--input-dir", str(shared),
        "--mode", "max",
        "--depth", "3",
        "--max-binaries", "1",
    ])
    assert rc == 0
    written = sorted(p.name for p in shared.glob("*_sorted_*.idx"))
    # discover_binaries returns sorted names; binA sorts before binB.
    assert written == ["binA_sorted_max_d003.idx"]


def test_cli_max_binaries_after_only(tmp_path: Path) -> None:
    """``--max-binaries`` applies AFTER ``--only`` per plan CLI contract."""
    shared = _build_two_binary_dir(tmp_path)
    rc = cli_main([
        "--input-dir", str(shared),
        "--mode", "max",
        "--depth", "3",
        "--only", "binB",
        "--max-binaries", "5",
    ])
    assert rc == 0
    written = sorted(p.name for p in shared.glob("*_sorted_*.idx"))
    assert written == ["binB_sorted_max_d003.idx"]


# ---------------------------------------------------------------------------
# stdout
# ---------------------------------------------------------------------------


def test_cli_announces_files_on_stdout(
    tmp_path: Path, capsys
) -> None:
    """Each produced file is announced on stdout, one path per line."""
    base = build_combined_fixture(tmp_path)
    rc = cli_main([
        "--input-dir", str(base),
        "--mode", "max",
        "--mode", "p95",
        "--depth", "3",
    ])
    assert rc == 0
    captured = capsys.readouterr().out.splitlines()
    expected = {
        str(base / f"{_BINARY_NAME}_sorted_max_d003.idx"),
        str(base / f"{_BINARY_NAME}_sorted_p95_d003.idx"),
    }
    assert set(captured) == expected


# ---------------------------------------------------------------------------
# Multi-depth + gating + dedup CLI flags
# ---------------------------------------------------------------------------


def test_cli_multiple_depths_produce_one_file_per_pair(tmp_path: Path) -> None:
    """K --mode x M --depth flags produce K*M .idx files in one invocation."""
    base = build_combined_fixture(tmp_path)
    rc = cli_main([
        "--input-dir", str(base),
        "--mode", "max",
        "--mode", "p95",
        "--depth", "1",
        "--depth", "3",
    ])
    assert rc == 0
    written = {p.name for p in base.glob("*_sorted_*.idx")}
    assert written == {
        f"{_BINARY_NAME}_sorted_max_d001.idx",
        f"{_BINARY_NAME}_sorted_max_d003.idx",
        f"{_BINARY_NAME}_sorted_p95_d001.idx",
        f"{_BINARY_NAME}_sorted_p95_d003.idx",
    }


def test_cli_min_variants_shrinks_index(tmp_path: Path) -> None:
    """``--min-variants`` excludes failing sections (length-0 bucket).

    The combined fixture has sections with variant counts {0,1,4,1,2}.
    With ``--min-variants 2`` only the 4- and 2-variant sections keep a
    real (>0) length; the others fall into the length-0 bucket. The
    bucket at length 0 therefore grows relative to the ungated build.
    """
    from tokenizer.aligned_data.sorted_index import (
        LengthReduction,
        ReductionKind,
        SortedIndexReader,
    )

    base = build_combined_fixture(tmp_path)
    ungated_dir = tmp_path / "ungated"
    gated_dir = tmp_path / "gated"
    common = ["--input-dir", str(base), "--mode", "max", "--depth", "3"]
    assert cli_main([*common, "--output-dir", str(ungated_dir)]) == 0
    assert cli_main(
        [*common, "--output-dir", str(gated_dir), "--min-variants", "2"]
    ) == 0

    red = LengthReduction(kind=ReductionKind.MAX)
    fname = f"{_BINARY_NAME}_sorted_max_d003.idx"
    ungated = SortedIndexReader(ungated_dir / fname, reduction=red, depth=3)
    gated = SortedIndexReader(gated_dir / fname, reduction=red, depth=3)
    # Both index all 5 sections; the gate moves the 3 failing ones into
    # the length-0 bucket (func_zero already there, solo_a + caller_fn
    # join it).
    assert ungated.total_sections() == 5
    assert gated.total_sections() == 5
    assert gated.count_at(0) == 3
    assert ungated.count_at(0) == 1


def test_cli_adjust_for_duplicates_smoke(tmp_path: Path) -> None:
    """``--adjust-for-duplicates`` runs end-to-end and writes a valid index."""
    base = build_combined_fixture(tmp_path)
    rc = cli_main([
        "--input-dir", str(base),
        "--mode", "p50",
        "--depth", "3",
        "--adjust-for-duplicates",
    ])
    assert rc == 0
    assert (base / f"{_BINARY_NAME}_sorted_p50_d003.idx").is_file()


def test_cli_min_unique_greater_than_min_variants_errors(
    tmp_path: Path,
) -> None:
    """``--min-variants-unique`` > ``--min-variants`` is a CLI error.

    :func:`argparse.ArgumentParser.error` exits with code 2; the
    unsatisfiable-gate message originates from :class:`VariantGate`.
    """
    base = build_combined_fixture(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli_main([
            "--input-dir", str(base),
            "--mode", "max",
            "--depth", "3",
            "--min-variants", "4",
            "--min-variants-unique", "6",
        ])
    assert exc.value.code == 2
