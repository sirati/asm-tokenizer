"""Unit tests for serialized vocab-size extraction.

Builds a tiny per-binary output CSV with the project's own
``save_vocabulary`` writer so the test exercises the real wire format,
then asserts :func:`count_vocab` returns the serialized token count
(excluding the 257 protocol-reserved slots) and degrades to ``None`` on
unreadable / non-vocab / unknown-ISA inputs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType
from tokenizer.vocab_unifier.saver import save_vocabulary

from scripts.collect_stats.vocab import count_vocab

_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_RESERVED = VocabularyManager._V2_RESERVED_TOKEN_COUNT


def _write_output_csv(path: Path, *, n_real_tokens: int, platform: str) -> None:
    """Write a per-binary output CSV whose last row is a real vocab-def
    produced by ``save_vocabulary``, preceded by a couple of fake token
    rows (so the tail-scan must skip past them)."""
    real = [f"{platform}_tok{i}" for i in range(n_real_tokens)]
    vocab_list = (
        [f"digit_{i:02X}" for i in range(_DIGIT_COUNT)] + ["value_negative"] + real
    )
    total = len(vocab_list)
    id_to_token_type = np.full(total, TokenType.UNRESOLVED, dtype=np.int8)
    id_to_token_type[_DIGIT_COUNT] = TokenType.VALUE_NEGATIVE
    for i in range(_RESERVED, total):
        id_to_token_type[i] = TokenType.PLATFORM
    vm = VocabularyManager.from_vocab(
        platform=platform,
        vocab_list=vocab_list,
        id_to_token_type=id_to_token_type,
        platform_instruction_type_cache=np.full(
            total, PlatformInstructionTypes.AGNOSTIC, dtype=np.int8
        ),
        lit_start_cache=np.array([], dtype=np.int_),
        lit_end_cache=np.array([], dtype=np.int_),
        platform_list=None,
        token_to_platform=None,
        format_version=2,
    )
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["token_row", "0", "1", "2"])
        w.writerow(["token_row", "3", "4", "5"])
        save_vocabulary(vm, w)


def test_count_matches_serialized_token_count(tmp_path: Path) -> None:
    """The count equals n_real_tokens — the digit + value_negative prefix
    is protocol-reserved and never serialised."""
    path = tmp_path / "x64-clang-3.5-O0_prog_output.csv"
    _write_output_csv(path, n_real_tokens=42, platform="x64")
    assert count_vocab(path) == 42


def test_count_equals_id_to_token_minus_reserved(tmp_path: Path) -> None:
    """Cross-check: the count equals ``len(id_to_token) - 257`` of a
    round-tripped manager (the definition documented in vocab.py)."""
    from tokenizer.vocab_unifier.loader import load_vocab_manager

    path = tmp_path / "arm32-gcc-4.8-O0_prog_output.csv"
    _write_output_csv(path, n_real_tokens=13, platform="arm32")
    vm = load_vocab_manager(path)
    assert vm is not None
    assert count_vocab(path) == len(vm.id_to_token) - _RESERVED == 13


def test_count_dashed_isa_platform_detect(tmp_path: Path) -> None:
    """``armv7l-hf`` resolves to a per-binary platform so the vocab row
    validates and counts."""
    path = tmp_path / "armv7l-hf-clang-10.0.1-Oz_hello_output.csv"
    _write_output_csv(path, n_real_tokens=7, platform="arm32")
    assert count_vocab(path) == 7


def test_unknown_isa_prefix_returns_none(tmp_path: Path) -> None:
    """No known ISA prefix ⇒ no platform ⇒ NULL (not 0, not a crash)."""
    path = tmp_path / "weirdname-noisa_thing_output.csv"
    _write_output_csv(path, n_real_tokens=5, platform="x64")
    assert count_vocab(path) is None


def test_non_vocab_csv_returns_none(tmp_path: Path) -> None:
    """A file whose last line is not a vocab-def row ⇒ None."""
    path = tmp_path / "x64-clang-3.5-O0_prog_output.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["token_row", "0", "1"])
        w.writerow(["token_row", "2", "3"])
    assert count_vocab(path) is None
