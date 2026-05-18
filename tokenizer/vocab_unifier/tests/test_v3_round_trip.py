"""v3 unified vocab save/load round-trip.

Construct an in-memory v3 ``VocabularyManager`` with a handful of
variant-axis + instruction tokens, write it through ``save_vocabulary``
to a tempfile, read it back through ``load_unified_vocab_manager``, and
assert id_to_token + format_version + token types survive intact.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager
from tokenizer.vocab_unifier.saver import save_vocabulary


def _build_v3_vm_with_variants_and_blocks(
    variant_strings: list[str],
    block_ids: list[int],
) -> VocabularyManager:
    """Construct a unified v3 VM by registering variants then a few
    block-id instruction tokens (a v2-shaped representative) so the
    saved row exercises both id bands."""
    vm = VocabularyManager(platform=None, format_version=3)
    for token in variant_strings:
        vm.Variant_Axis(token)
    # A handful of v2 instruction-shape tokens with token_to_platform set.
    # `Block_V2(block_id)` registers the integer as a block token; pairs
    # with the unified VM's platform-list machinery (platform=None means
    # the unified shape with token_to_platform present).
    for bid in block_ids:
        vm.Block_V2(bid)
    return vm


def _write_unified_csv(vm: VocabularyManager, csv_path: Path) -> None:
    """Mirror the writer call inside ``unify_vocab`` — the unified
    file is exactly one row (the vocab def), written via a single
    ``csv.writer.writerow`` call. No header line."""
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        save_vocabulary(vm, writer)


def test_v3_round_trip_preserves_id_to_token(tmp_path: Path) -> None:
    variant_strings = [
        "arch:x64", "arch:arm64",
        "comp:gcc", "comp:clang",
        "cver:gcc:13.2.0", "cver:clang:15.0.0",
        "opt:O2", "opt:O3",
        "hardening:full",
    ]
    block_ids = [0, 1, 7]
    vm = _build_v3_vm_with_variants_and_blocks(variant_strings, block_ids)

    csv_path = tmp_path / "unified_vocab.csv"
    _write_unified_csv(vm, csv_path)

    loaded = load_unified_vocab_manager(csv_path)
    assert loaded is not None, "v3 unified vocab failed to load"
    assert loaded.format_version == 3
    # Total id space must be preserved byte-for-byte.
    assert list(loaded.id_to_token) == list(vm.id_to_token)
    # And the type array (the saver normalises against TokenType.UNRESOLVED).
    assert list(loaded.id_to_token_type) == list(vm.id_to_token_type)


def test_v3_round_trip_variant_ids_are_contiguous_from_256(
    tmp_path: Path,
) -> None:
    """Variants land at [256, 256+n_variants); blocks land above."""
    variant_strings = ["arch:x64", "arch:arm64", "comp:gcc"]
    vm = _build_v3_vm_with_variants_and_blocks(variant_strings, [0, 1])

    csv_path = tmp_path / "unified_vocab.csv"
    _write_unified_csv(vm, csv_path)
    loaded = load_unified_vocab_manager(csv_path)
    assert loaded is not None

    # The first 256 IDs are the reserved-digit placeholder slots.
    for i in range(256):
        assert loaded.id_to_token_type[i] == TokenType.UNRESOLVED, (
            f"reserved digit slot {i} has type "
            f"{loaded.id_to_token_type[i]}"
        )
    # Variants come next.
    for offset, token_str in enumerate(variant_strings):
        assert loaded.id_to_token[256 + offset] == token_str
        assert loaded.id_to_token_type[256 + offset] == TokenType.VARIANT_AXIS
    # Then any instruction tokens.
    assert len(loaded.id_to_token) > 256 + len(variant_strings)


def test_v3_round_trip_variant_token_resolves_to_inner_class(
    tmp_path: Path,
) -> None:
    """A loaded v3 VM can resolve a variant id back to a
    ``VariantAxisToken`` via the dispatch table — confirms Inner-class
    wiring survives the save/load."""
    vm = _build_v3_vm_with_variants_and_blocks(["arch:x64", "opt:O2"], [0])

    csv_path = tmp_path / "unified_vocab.csv"
    _write_unified_csv(vm, csv_path)
    loaded = load_unified_vocab_manager(csv_path)
    assert loaded is not None

    arch_id = loaded.get_token_id("arch:x64")
    assert arch_id != -1
    # Dispatch table check — VARIANT_AXIS must route to Variant_Axis.
    assert loaded.get_token_class_for_type(TokenType.VARIANT_AXIS) \
        is loaded.Variant_Axis
    # Round-trip a single-id token slice back to the Inner class.
    token = loaded.Variant_Axis._from_token_ids([arch_id])
    assert token.to_string() == "arch:x64"


def test_v3_trailer_cell_is_3(tmp_path: Path) -> None:
    """The CSV row's trailing cell pair must be ``format_version, 3``
    so a v2-only loader sees the mismatch up-front."""
    vm = _build_v3_vm_with_variants_and_blocks(["arch:x64"], [0])
    csv_path = tmp_path / "unified_vocab.csv"
    _write_unified_csv(vm, csv_path)

    rows = list(csv.reader(csv_path.open(encoding="ascii")))
    # File is exactly one CSV row (the vocab def) — see
    # _write_unified_csv comment.
    assert len(rows) == 1
    vocab_row = rows[0]
    # Unified layout base = 13; trailer adds 2 cells.
    assert len(vocab_row) == 15
    assert vocab_row[13] == "format_version"
    assert vocab_row[14] == "3"
