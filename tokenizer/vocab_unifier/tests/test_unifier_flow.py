"""Unifier-flow layout tests (post-canonical-blocks rewrite).

Concerns covered:

1. **Canonical-block prefix.** Every unifier-produced unified vocab
   carries the number block (``valued_const_v2`` ... ``float128``) at
   slots 257..263 and the identity block (``block_v2`` ... ``rw_data_ptr``)
   at slots 264..271, in source-declaration order.

2. **Axis-grouped variant tail.** Variant-axis tokens land at the TAIL
   of the unified vocab in axis-grouped order: positional axes
   (``arch`` -> ``comp`` -> ``cver`` -> ``opt``) first, then sidecar-key
   axes in alphabetical-by-prefix order — regardless of CSV iteration
   order or alphabetical key order. Critical case: positional axes
   come BEFORE alphabetically-earlier sidecar keys (e.g. ``aaa:foo``
   lands AFTER ``opt:O2``).

3. **Head-of-vocab assert.** ``VocabularyManager.from_vocab`` enforces
   the canonical layout on every unified VM it constructs; a wire row
   whose slot 257 is not ``valued_const_v2`` is rejected.

Synthetic per-binary CSVs are produced via the production
``save_vocabulary`` writer (no body assumptions), matching the
fixtures used in ``test_unify_two_pass.py``.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType
from tokenizer.vocab_unifier.loader import (
    load_unified_vocab_manager,
    load_vocab_manager_csv_row_bytes,
)
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


_PADDING_LINE = "function_name,binary_addr," + ("x" * 64) + "\n"

_RESERVED = VocabularyManager._V2_RESERVED_TOKEN_COUNT  # 257
_EAGER_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272
_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256

# Canonical NUMBER + IDENTITY block in source-declaration order — the
# wire-format strings every post-Phase-4 unifier writes at slots
# 257..271. Centralised here so the assertions track the registration
# sequence without re-encoding it per test.
_CANONICAL_BLOCK_NAMES = (
    "valued_const_v2",
    "float16", "bfloat16", "float32", "float64", "float80", "float128",
    "block_v2",
    "local_func", "plt_func", "ext_func", "string_ptr",
    "jump_table", "ro_data_ptr", "rw_data_ptr",
)
assert len(_CANONICAL_BLOCK_NAMES) == _EAGER_END - _RESERVED


def _write_per_binary_csv(
    csv_path: Path,
    platform: str,
    block_ids: list[int],
) -> None:
    """Synthesise a per-binary v2 CSV via the production writer."""
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in block_ids:
        vm.Block_V2(bid)
    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        fh.write(_PADDING_LINE)
        writer = csv.writer(fh, lineterminator='\n')
        save_vocabulary(vm, writer)


def _write_legacy_filename(dir_path: Path, basename: str, platform: str) -> Path:
    """Write a per-binary v2 CSV with the legacy 4-axis filename schema.

    Filename body content is irrelevant beyond the protocol shape; the
    vocab-def trailer is the line ``load_vocab_manager`` consumes.
    """
    path = dir_path / f"{basename}_output.csv"
    _write_per_binary_csv(path, platform=platform, block_ids=[0])
    return path


def _write_sidecar(csv_path: Path, *, arch: str, compiler_family: str,
                    compiler_version: str, optimization: str, pkg: str,
                    extra_metadata: dict | None = None) -> None:
    base = csv_path.name.removesuffix("_output.csv")
    sidecar = csv_path.with_name(base + "_meta.json")
    sidecar.write_text(json.dumps({
        "arch": arch,
        "compiler_family": compiler_family,
        "compiler_version": compiler_version,
        "optimization": optimization,
        "pkg": pkg,
        "extra_metadata": extra_metadata or {},
    }))


# ---------------------------------------------------------------------------
# 1. Canonical-block prefix
# ---------------------------------------------------------------------------


def test_unifier_writes_canonical_number_and_identity_blocks(tmp_path: Path) -> None:
    """1-CSV unify — assert slots 257..271 carry the canonical
    number + identity tokens in source-declaration order, and the
    accompanying block-range attributes match."""
    csv_path = _write_legacy_filename(tmp_path, "x64-gcc-13.2.0-O2_hello", "x64")
    out_csv = tmp_path / "unified_vocab.csv"
    unify_vocab([csv_path], out_csv)

    loaded = load_unified_vocab_manager(out_csv)
    assert loaded is not None
    assert loaded.format_version == MEMMAP_FORMAT_VERSION

    # Canonical block at the exact fixed slots.
    for offset, name in enumerate(_CANONICAL_BLOCK_NAMES):
        slot = _RESERVED + offset
        assert loaded.id_to_token[slot] == name, (
            f"slot {slot}: expected {name!r}, got {loaded.id_to_token[slot]!r}"
        )

    # Published range attributes match the source-of-truth constants.
    assert loaded.number_block_range == (
        VocabularyManager._V2_NUMBER_BLOCK_START,
        VocabularyManager._V2_NUMBER_BLOCK_START
        + VocabularyManager._V2_NUMBER_BLOCK_COUNT,
    )
    assert loaded.identity_block_range == (
        VocabularyManager._V2_IDENTITY_BLOCK_START,
        VocabularyManager._V2_IDENTITY_BLOCK_START
        + VocabularyManager._V2_IDENTITY_BLOCK_COUNT,
    )


def test_unifier_lands_instruction_reps_at_eager_block_end(tmp_path: Path) -> None:
    """First post-canonical-block slot (id 272) is reserved for the
    first instruction representative or the variant tail — slot 272 in
    a minimal 1-CSV corpus must therefore NOT carry one of the canonical
    block names."""
    csv_path = _write_legacy_filename(tmp_path, "x64-gcc-13.2.0-O2_hello", "x64")
    out_csv = tmp_path / "unified_vocab.csv"
    unify_vocab([csv_path], out_csv)

    loaded = load_unified_vocab_manager(out_csv)
    assert loaded is not None

    assert _EAGER_END < len(loaded.id_to_token), (
        "unified vocab has no slots past the canonical block — fixture "
        "should produce at least one variant axis token"
    )
    assert loaded.id_to_token[_EAGER_END] not in _CANONICAL_BLOCK_NAMES


# ---------------------------------------------------------------------------
# 2. Axis-grouped variant tail
# ---------------------------------------------------------------------------


def test_variant_tail_is_axis_grouped_with_positional_first(tmp_path: Path) -> None:
    """Multi-CSV unify — variant block at the tail, positional axes
    (arch, comp, cver, opt) ALWAYS before any sidecar-axis prefix even
    when the sidecar prefix is alphabetically earlier (e.g. ``aaa:`` <
    ``opt:``).

    Constructs a 2-CSV corpus with an ``extra_metadata`` sidecar
    carrying the key ``aaa`` so the test would fail the moment the
    iterator falls back to plain alphabetical order.
    """
    # CSV 1: legacy filename, no sidecar -> 4 positional axes.
    csv1 = _write_legacy_filename(tmp_path, "x64-gcc-13.2.0-O2_pkga", "x64")
    # CSV 2: same filename schema + sidecar adding an ``aaa`` metadata
    # axis — alphabetically before every positional prefix.
    csv2 = _write_legacy_filename(tmp_path, "arm64-clang-15.0.0-O3_pkgb", "arm64")
    _write_sidecar(
        csv2,
        arch="arm64", compiler_family="clang", compiler_version="15.0.0",
        optimization="-O3", pkg="pkgb",
        extra_metadata={"aaa": "marker"},
    )

    out_csv = tmp_path / "unified_vocab.csv"
    unify_vocab([csv1, csv2], out_csv)

    loaded = load_unified_vocab_manager(out_csv)
    assert loaded is not None

    expected_tail_order = [
        # Positional first, alphabetical within axis.
        "arch:arm64", "arch:x64",
        "comp:clang", "comp:gcc",
        "cver:clang:15.0.0", "cver:gcc:13.2.0",
        "opt:O2", "opt:O3",
        # Sidecar key (``aaa``) lands AFTER all positional axes despite
        # being alphabetically earlier than ``arch``, ``comp``, etc.
        "aaa:marker",
    ]
    n_variants = len(expected_tail_order)
    total = len(loaded.id_to_token)
    tail_start = total - n_variants

    for offset, token in enumerate(expected_tail_order):
        slot = tail_start + offset
        assert loaded.id_to_token[slot] == token, (
            f"variant tail slot {slot}: expected {token!r}, "
            f"got {loaded.id_to_token[slot]!r}"
        )
        assert loaded.id_to_token_type[slot] == TokenType.VARIANT_AXIS


def test_variant_tail_order_is_stable_across_csv_order(tmp_path: Path) -> None:
    """Variant tail order is determined by axis-grouped iteration on
    the deduplicated inventory, NOT by CSV iteration order. Two runs
    of the unifier over the same corpus (reversed) produce the same
    head-to-tail vocab layout."""
    csv1 = _write_legacy_filename(tmp_path, "x64-gcc-13.2.0-O2_pkga", "x64")
    csv2 = _write_legacy_filename(tmp_path, "arm64-clang-15.0.0-O3_pkgb", "arm64")

    out_a = tmp_path / "unified_a.csv"
    out_b = tmp_path / "unified_b.csv"
    unify_vocab([csv1, csv2], out_a)
    # Reverse order must produce the same on-disk vocab — variant inventory
    # is a set, registration uses axis-grouped iteration, instruction-rep
    # merge is deterministic per CSV. Mapping files clobber on the second
    # call but the unified-vocab byte content is what's under test.
    unify_vocab([csv2, csv1], out_b)

    vm_a = load_unified_vocab_manager(out_a)
    vm_b = load_unified_vocab_manager(out_b)
    assert vm_a is not None
    assert vm_b is not None
    assert vm_a.id_to_token == vm_b.id_to_token, (
        "variant tail (and full vocab) drifted between two runs over the "
        "same corpus — registration order is no longer deterministic"
    )


# ---------------------------------------------------------------------------
# 3. Head-of-vocab assert in from_vocab
# ---------------------------------------------------------------------------


def _craft_bad_head_row() -> bytes:
    """Build a unified-style wire row whose slot 257 carries ``wrong_token``
    instead of ``valued_const_v2``. Trailer declares format_version=1
    (a supported version) so the loader reaches ``from_vocab`` and the
    head-of-vocab assertion is the failure path under test."""
    # The serialised vocab cell is the post-strip suffix (slots >= 257).
    # Putting ``wrong_token`` first makes slot 257 = ``wrong_token`` after
    # the loader reconstitutes the protocol-reserved prefix.
    real_tokens = ["wrong_token"]
    serialized_vocab = ",".join(real_tokens)

    empty_int8 = ndarray_to_base64(np.zeros(1, dtype=np.int8))
    empty_int = ndarray_to_base64(np.array([], dtype=np.int_))

    row = [
        "vocabulary",
        serialized_vocab,
        f"_id_to_token_type norm:{0 + TokenType.UNRESOLVED}",
        empty_int8,
        f"_platform_instruction_type_cache norm:{0 + PlatformInstructionTypes.UNRESOLVED}",
        empty_int8,
        "_lit_start_cache",
        empty_int,
        "_lit_end_cache",
        empty_int,
        "platforms norm:-1",
        "x64,arm64",
        ndarray_to_base64(np.zeros(1, dtype=np.int8)),
        "format_version", "1",
    ]
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(row)
    return buf.getvalue().encode("ascii")


def test_from_vocab_rejects_non_canonical_head_of_unified_vocab() -> None:
    """``from_vocab`` enforces the head-of-vocab invariant for every
    unified VM (platform=None, format_version in {1,2}): slot 257 must
    be ``valued_const_v2``. A wire row whose head differs raises
    AssertionError on load."""
    raw = _craft_bad_head_row()
    with pytest.raises(AssertionError, match="valued_const_v2"):
        load_vocab_manager_csv_row_bytes(raw, "unified")


def test_from_vocab_rejects_degenerate_short_unified_vocab() -> None:
    """``from_vocab`` length-guards the head-of-vocab assert against a
    degenerate ``vocab_list`` (digits + ``value_negative`` only, length
    ``_V2_RESERVED_TOKEN_COUNT``). Without the guard the subsequent
    ``id_to_token[_V2_NUMBER_BLOCK_START]`` lookup raises IndexError
    instead of the cleaner AssertionError; the guard fires here with the
    length message naming the actual + required vocab sizes."""
    digits = [f"digit_{i:02X}" for i in range(_DIGIT_COUNT)]
    vocab_list = digits + ["value_negative"]
    assert len(vocab_list) == _RESERVED

    pitc = np.full(_RESERVED, PlatformInstructionTypes.AGNOSTIC, dtype=np.int8)
    itt = np.full(_RESERVED, TokenType.UNRESOLVED, dtype=np.int8)
    itt[VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID] = TokenType.VALUE_NEGATIVE
    empty_int = np.array([], dtype=np.int_)

    with pytest.raises(AssertionError, match="too short for canonical layout"):
        VocabularyManager.from_vocab(
            platform=None,
            vocab_list=vocab_list,
            platform_instruction_type_cache=pitc,
            id_to_token_type=itt,
            lit_start_cache=empty_int,
            lit_end_cache=empty_int,
            platform_list=[],
            token_to_platform=np.full(_RESERVED, -1, dtype=np.int8),
            format_version=1,
        )
