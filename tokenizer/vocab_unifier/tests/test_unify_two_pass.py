"""Integration test for the two-pass ``unify_vocab``.

Builds a small synthetic corpus of per-binary v2 vocab CSVs (with
filename-derived variant identity), runs ``unify_vocab``, then reads
the produced ``unified_vocab.csv`` back through
``load_unified_vocab_manager`` and asserts:

* ``format_version == MEMMAP_FORMAT_VERSION``
* the ``value_negative`` postfix sign marker is pinned at id 256 (the
  first slot after the reserved-digit range, eagerly registered by the
  unified VM constructor before any caller-driven registrations)
* variant-axis tokens populate ids ``[257, 257+n_variants)`` — one
  past the marker
* instruction-representative tokens populate ids above the variant
  block
* per-binary ``.mapping.b64c`` files exist

The synthetic per-binary CSVs are produced by writing a real
``VocabularyManager(platform=<arch>, format_version=2)`` through
``save_vocabulary`` — exercises the same writer path the tokenize
worker uses in production.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.compact_base64_utils import base64_to_ndarray
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


# Padding line so the file body exceeds the 64-byte tail
# ``read_last_line_of_file`` excludes (it searches the body *before*
# the last 64 bytes for a newline). Keeps the synthesised CSV
# structurally close to a real per-binary file (header + body row(s)
# + vocab def line).
_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"


def _write_per_binary_csv(
    csv_path: Path,
    platform: str,
    block_ids: list[int],
) -> None:
    """Write a synthetic v2 per-binary CSV at ``csv_path``.

    Registers a handful of ``Block_V2`` instruction tokens onto a
    platform-scoped v2 ``VocabularyManager``; saves the vocab def
    line as the file's final row, preceded by a single padding line
    so ``read_last_line_of_file`` can find a newline outside the
    64-byte tail."""
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in block_ids:
        vm.Block_V2(bid)

    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh, lineterminator='\n')
        save_vocabulary(vm, writer)


def test_unify_vocab_emits_unified_with_variant_block(tmp_path: Path) -> None:
    """Two per-binary v2 CSVs -> one unified CSV (at
    ``MEMMAP_FORMAT_VERSION``) with variant + instruction id bands."""
    csv_files = []
    # Filename schema: <platform>-<compiler>-<version>-<opt>_<pkg>_output.csv
    # (default format string: platform-compiler-version-optimisationlevel_binaryname).
    # Default platform/compiler/version regex is `[^-_]+` so underscored
    # aliases like x86_64 fail the parser; production filenames use the
    # already-collapsed alias (x64, arm64).
    for basename, arch in [
        ("x64-gcc-13.2.0-O2_pkga", "x64"),
        ("arm64-clang-15.0.0-O3_pkgb", "arm64"),
    ]:
        path = tmp_path / f"{basename}_output.csv"
        _write_per_binary_csv(path, platform=arch, block_ids=[0, 1, 2])
        csv_files.append(path)

    out_csv = tmp_path / "unified_vocab.csv"
    unify_vocab(csv_files, out_csv)

    loaded = load_unified_vocab_manager(out_csv)
    assert loaded is not None
    assert loaded.format_version == MEMMAP_FORMAT_VERSION

    # The eagerly-pinned `value_negative` marker occupies id 256; the
    # caller-driven variant-axis block starts one past it.
    assert loaded.get_token_id("value_negative") == 256
    assert loaded.id_to_token_type[256] == TokenType.VALUE_NEGATIVE
    variant_block_start = 257

    # Variant tokens: 2 archs + 2 compilers + 2 cver + 2 opts = 8
    expected_variants = {
        "arch:x64", "arch:arm64",
        "comp:gcc", "comp:clang",
        "cver:gcc:13.2.0", "cver:clang:15.0.0",
        "opt:O2", "opt:O3",
    }
    n_variants = len(expected_variants)
    for offset, token in enumerate(sorted(expected_variants)):
        tid = loaded.get_token_id(token)
        assert tid == variant_block_start + offset, (
            f"variant {token!r} got id {tid}, "
            f"expected {variant_block_start + offset}"
        )
        assert loaded.id_to_token_type[tid] == TokenType.VARIANT_AXIS

    # First instruction-id band must start exactly above the variants.
    # Block_V2 registers a `block_v2` representative token, so we know
    # ids variant_block_start+n_variants and up are non-variant.
    first_after_variants = variant_block_start + n_variants
    assert loaded.id_to_token_type[first_after_variants] != TokenType.VARIANT_AXIS

    # mapping.b64c sidecars must exist alongside each input CSV.
    for csv_file in csv_files:
        mapping_file = csv_file.with_suffix(".mapping.b64c")
        assert mapping_file.exists(), f"missing mapping file: {mapping_file}"
        mappings = base64_to_ndarray(mapping_file.read_text(encoding="ascii"))
        # Reserved-digit prefix (0..255) is identity-mapped.
        assert (mappings[:256] == list(range(256))).all(), (
            "reserved-digit prefix is not identity-mapped"
        )
        # Every entry must have been resolved (no -1s).
        assert (mappings >= 0).all(), "unresolved id in mapping array"


def test_unify_vocab_rejects_v1_per_binary_csv(tmp_path: Path) -> None:
    """A v1 per-binary input must hard-fail with a clear error."""
    csv_path = tmp_path / "x64-gcc-13.2.0-O2_pkg_output.csv"
    vm_v1 = VocabularyManager(platform="x64", format_version=1)
    vm_v1.Block(0)  # legacy v1 block token
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh, lineterminator='\n')
        save_vocabulary(vm_v1, writer)

    out_csv = tmp_path / "unified_vocab.csv"
    with pytest.raises(ValueError, match="format_version=2"):
        unify_vocab([csv_path], out_csv)
