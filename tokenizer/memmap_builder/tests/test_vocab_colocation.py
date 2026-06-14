"""``build_memmap_files`` co-locates the unified vocab into the catalog dir.

A memmap catalog is only self-describing if it ships with the EXACT
unified vocab that assigned its uint16 ids. The loader's gate
(``resolve_unified_vocab_path``) auto-resolves the vocab from the
memmap directory alone — but only if the build wrote a copy there.
Without it a consumer is forced to supply a vocab path by hand and can
silently feed a stale/mismatched one, decoding the same variant record
to the wrong axes.

This test exercises the LIVE builder against a tiny real corpus (same
fixture shape as ``test_builder_smoke``: real v2 per-binary CSVs + a
real v3 unified vocab produced by ``unify_vocab``) and asserts:

  * the gate resolves a vocab from the catalog dir with NO explicit
    path, and
  * the resolved file is byte-identical to the source unified vocab.

Un-fakeable: it goes through the real ``build_memmap_files`` write path
and the real ``resolve_unified_vocab_path`` search policy — not a
hand-placed file or a mocked gate.
"""

from __future__ import annotations

import csv
import filecmp
from pathlib import Path
from typing import List

from tokenizer.aligned_data.loader.unified_vocab_gate import (
    resolve_unified_vocab_path,
)
from tokenizer.memmap_builder.builder import (
    BinaryVersionInfo,
    build_memmap_files,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


# Same padding line as the smoke test so the synthesised CSV is
# structurally close to a real per-binary file (one newline outside the
# 64-byte tail ``read_last_line_of_file`` excludes).
_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def _write_per_binary_csv(csv_path: Path, platform: str) -> None:
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh, lineterminator="\n")
        save_vocabulary(vm, writer)


def _build_synthetic_corpus(tmp_path: Path) -> tuple[List[Path], Path]:
    csv_files: List[Path] = []
    for basename, arch in [
        ("x64-gcc-13.2.0-O2_pkga", "x64"),
        ("arm64-clang-15.0.0-O3_pkgb", "arm64"),
    ]:
        path = tmp_path / f"{basename}_output.csv"
        _write_per_binary_csv(path, platform=arch)
        csv_files.append(path)

    unified_vocab_path = tmp_path / "unified_vocab.csv"
    unify_vocab(csv_files, unified_vocab_path)
    return csv_files, unified_vocab_path


def _versions_for(csv_files: List[Path]) -> List[BinaryVersionInfo]:
    return [
        BinaryVersionInfo(
            path=csv_files[0],
            mapping_path=csv_files[0].with_suffix(".mapping.b64c"),
            arch="x64",
            compiler="gcc",
            compilerversion="13.2.0",
            opt="O2",
            pkg="pkga",
            filename="x64-gcc-13.2.0-O2_pkga",
        ),
        BinaryVersionInfo(
            path=csv_files[1],
            mapping_path=csv_files[1].with_suffix(".mapping.b64c"),
            arch="arm64",
            compiler="clang",
            compilerversion="15.0.0",
            opt="O3",
            pkg="pkgb",
            filename="arm64-clang-15.0.0-O3_pkgb",
        ),
    ]


def test_build_colocates_gate_resolvable_byte_identical_vocab(
    tmp_path: Path,
) -> None:
    """After a build, the gate resolves the catalog's own vocab with no
    explicit path, and the resolved file is byte-identical to the source.

    The source vocab lives OUTSIDE the catalog dir (a distinct ``src/``)
    so the resolution can only succeed via the co-located copy the
    builder wrote — not by the gate happening to find the source.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    csv_files, unified_vocab_path = _build_synthetic_corpus(src_dir)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    versions = _versions_for(csv_files)

    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    # Gate auto-resolves from the catalog dir alone — no explicit path.
    resolved = resolve_unified_vocab_path(output_dir)
    assert resolved == output_dir / "unified_vocab.csv"

    # The co-located copy is the in-directory candidate (precedence over
    # any parent), and it is byte-identical to the source vocab the
    # build was performed against.
    assert resolved.parent == output_dir
    assert filecmp.cmp(resolved, unified_vocab_path, shallow=False), (
        "co-located unified_vocab.csv must be a byte-faithful copy of the "
        "source vocab the catalog was built against"
    )


def test_build_colocation_is_idempotent_when_source_is_in_output_dir(
    tmp_path: Path,
) -> None:
    """When the source vocab already lives at the gate's target location
    (vocab-source == output dir), the build must not raise a same-file
    copy error and the vocab must remain resolvable + intact.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    csv_files, unified_vocab_path = _build_synthetic_corpus(output_dir)
    # Sanity: the source vocab IS at the gate target location here.
    assert unified_vocab_path == output_dir / "unified_vocab.csv"
    before = unified_vocab_path.read_bytes()

    versions = _versions_for(csv_files)
    build_memmap_files(versions, output_dir, "demo", unified_vocab_path)

    resolved = resolve_unified_vocab_path(output_dir)
    assert resolved == output_dir / "unified_vocab.csv"
    assert resolved.read_bytes() == before
