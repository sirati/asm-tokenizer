"""Tests for the per-binary ``_function_ranges.txt`` debug sidecar.

The sidecar mirrors the per-binary output CSV's FUNCTION rows: one
line per FUNCTION row in the same order, recording the function's
``<min_addr:hex>,<max_addr:hex>``. Coverage:

1. Wire format — header line, lowercase hex, no ``0x`` prefix, newline-
   terminated, comma-separated.
2. ``iter_sidecar_lines`` round-trip — parser yields the same int pairs
   the writer emitted, skipping the header.
3. End-to-end integration via ``fill_constant_candidates`` — the range
   the sidecar would record matches ``entry`` + body extent computed
   from a known block layout.
4. Path derivation via ``derive_sidecar_path`` — the sidecar lands
   next to the CSV with the canonical ``_function_ranges.txt`` suffix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.function_range_sidecar import (
    FunctionRangeSidecar,
    iter_sidecar_lines,
)
from tokenizer.output_filename import derive_sidecar_path


def test_sidecar_writes_header_and_lines(tmp_path: Path) -> None:
    """The writer emits ``# format=1`` first, then one line per
    ``add()`` call, lowercase hex with ``,`` separator and ``\\n``
    terminator. No trailing ``0x`` prefix.
    """
    path = tmp_path / "demo_function_ranges.txt"
    sidecar = FunctionRangeSidecar(path)
    sidecar.add(0x401000, 0x4011A0)
    sidecar.add(0x4012F0, 0x401400)
    sidecar.add(0xDEADBEEF, 0xDEADC000)
    sidecar.close()

    content = path.read_text(encoding="ascii")
    expected = (
        "# format=1\n"
        "401000,4011a0\n"
        "4012f0,401400\n"
        "deadbeef,deadc000\n"
    )
    assert content == expected, f"sidecar content mismatch: {content!r}"


def test_sidecar_line_count_tracks_adds(tmp_path: Path) -> None:
    """``line_count`` reflects only data lines (header excluded)."""
    sidecar = FunctionRangeSidecar(tmp_path / "x_function_ranges.txt")
    assert sidecar.line_count == 0
    sidecar.add(0x1000, 0x1100)
    assert sidecar.line_count == 1
    sidecar.add(0x2000, 0x2200)
    assert sidecar.line_count == 2
    sidecar.close()


def test_sidecar_rejects_negative_addresses(tmp_path: Path) -> None:
    """VMAs are unsigned; negative inputs are rejected to catch caller
    bugs at the write site rather than silently emitting ``-1`` as hex.
    """
    sidecar = FunctionRangeSidecar(tmp_path / "neg_function_ranges.txt")
    with pytest.raises(ValueError, match="negative address"):
        sidecar.add(-1, 0x100)
    with pytest.raises(ValueError, match="negative address"):
        sidecar.add(0x100, -1)
    sidecar.close()


def test_sidecar_closed_writes_raise(tmp_path: Path) -> None:
    """Adding after ``close()`` raises so use-after-close is loud."""
    sidecar = FunctionRangeSidecar(tmp_path / "closed_function_ranges.txt")
    sidecar.close()
    with pytest.raises(ValueError, match="closed"):
        sidecar.add(0x100, 0x200)


def test_sidecar_close_is_idempotent(tmp_path: Path) -> None:
    """Double-close must not raise; the main_loop's broad ``try/except``
    envelope can land on ``close()`` more than once on degraded paths.
    """
    sidecar = FunctionRangeSidecar(tmp_path / "idem_function_ranges.txt")
    sidecar.add(0x100, 0x200)
    sidecar.close()
    sidecar.close()  # must not raise


def test_iter_sidecar_lines_roundtrip(tmp_path: Path) -> None:
    """Parser yields exactly the int pairs the writer recorded, in
    order, with the header transparently skipped.
    """
    path = tmp_path / "roundtrip_function_ranges.txt"
    pairs = [
        (0x401000, 0x4011A0),
        (0x4012F0, 0x401400),
        (0xDEADBEEF, 0xDEADC000),
        (0, 0),
    ]
    sidecar = FunctionRangeSidecar(path)
    for mn, mx in pairs:
        sidecar.add(mn, mx)
    sidecar.close()

    parsed = list(iter_sidecar_lines(path))
    assert parsed == pairs, f"roundtrip mismatch: {parsed!r} vs {pairs!r}"


def test_iter_sidecar_lines_ignores_comments_and_blanks(tmp_path: Path) -> None:
    """Forward compatibility: extra ``#``-prefixed lines and blank
    lines in the data section must not break the parser.
    """
    path = tmp_path / "comments_function_ranges.txt"
    path.write_text(
        "# format=1\n"
        "# extra-comment-line\n"
        "401000,4011a0\n"
        "\n"
        "# mid-stream comment\n"
        "4012f0,401400\n",
        encoding="ascii",
    )
    parsed = list(iter_sidecar_lines(path))
    assert parsed == [(0x401000, 0x4011A0), (0x4012F0, 0x401400)]


def test_iter_sidecar_lines_rejects_missing_comma(tmp_path: Path) -> None:
    """A data line without a ``,`` separator is a wire-format bug —
    the parser surfaces it immediately instead of silently emitting a
    truncated pair.
    """
    path = tmp_path / "bad_function_ranges.txt"
    path.write_text("# format=1\n401000\n", encoding="ascii")
    with pytest.raises(ValueError, match="missing comma"):
        list(iter_sidecar_lines(path))


def test_derive_sidecar_path_strips_output_csv(tmp_path: Path) -> None:
    """``_output.csv`` is stripped; the sidecar lands next to the CSV
    under the canonical ``<base>`` prefix.
    """
    csv = tmp_path / "x86_64-clang-9-Os_minigzip_output.csv"
    sidecar_path = derive_sidecar_path(csv, "_function_ranges.txt")
    assert sidecar_path == tmp_path / "x86_64-clang-9-Os_minigzip_function_ranges.txt"


def test_derive_sidecar_path_handles_variant_suffix(tmp_path: Path) -> None:
    """Variant-suffix CSVs keep the variant tag in the sidecar name
    (the suffix is part of ``<base>``).
    """
    csv = tmp_path / "x86_64-clang-9-Os_minigzip__deadbeef_output.csv"
    sidecar_path = derive_sidecar_path(csv, "_function_ranges.txt")
    assert sidecar_path.name == "x86_64-clang-9-Os_minigzip__deadbeef_function_ranges.txt"


def test_derive_sidecar_path_defensive_fallback(tmp_path: Path) -> None:
    """When the CSV doesn't end in ``_output.csv``, the helper strips
    only the trailing extension — defensive fallback for test fixtures
    and ad-hoc paths.
    """
    csv = tmp_path / "weird.csv"
    sidecar_path = derive_sidecar_path(csv, "_function_ranges.txt")
    assert sidecar_path == tmp_path / "weird_function_ranges.txt"


# ---------------------------------------------------------------------------
# Integration with fill_constant_candidates: the (func_min, func_max)
# threaded through the function-analysis return tuple matches what the
# sidecar would record for a known block layout.
# ---------------------------------------------------------------------------

from tokenizer.fill_constant_candidates import fill_constant_candidates
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenResolver
from tokenizer.tests.test_block_identity_roundtrip import (
    _EmptyBlock,
    _EmptyFunction,
)


def test_fill_constant_candidates_returns_function_range() -> None:
    """The new (func_min, func_max) tail of the return tuple matches
    ``entry`` and ``max(block.addr + block.size)`` across the function's
    blocks — the exact values the sidecar must record.
    """
    vm = VocabularyManager(platform="x86_64", format_version=2)
    resolver = TokenResolver()
    entry = 0x401000
    blocks = [
        _EmptyBlock(addr=entry, size=0x40),
        _EmptyBlock(addr=0x401080, size=0x20),
        _EmptyBlock(addr=0x401200, size=0x10),
    ]
    func = _EmptyFunction(blocks=blocks)

    result = fill_constant_candidates(
        func_addr=entry,
        func=func,
        instr_sets=None,
        lookup=None,
        text_start=0,
        text_end=0x10000,
        resolver=resolver,
        vocab_manager=vm,
        arch_provider=None,
        func_tokens=FunctionTokenList(num_blocks=4, vocab_manager=vm),
        disasm_provider=None,
    )
    assert result is not None
    *_, func_min_addr, func_max_addr = result
    assert func_min_addr == entry
    assert func_max_addr == max(b.addr + b.size for b in blocks)


def test_sidecar_records_match_fill_constant_candidates(tmp_path: Path) -> None:
    """End-to-end: feed the (func_min, func_max) values
    ``fill_constant_candidates`` returns into the sidecar and verify
    the resulting file parses back to the same pairs.

    Mirrors the main_loop integration without requiring a real Task /
    provider fixture: the sidecar's contract is purely
    ``add(min, max) -> one line``; the values themselves come from
    ``fill_constant_candidates``.
    """
    vm = VocabularyManager(platform="x86_64", format_version=2)
    resolver = TokenResolver()

    # Two synthetic functions, distinct entry addresses and body sizes.
    # Each function has >= 2 empty blocks: ``fill_constant_candidates``
    # short-circuits to ``None`` on a single zero-instruction block (a
    # degenerate-function guard unrelated to this feature).
    functions = [
        (
            0x401000,
            [_EmptyBlock(addr=0x401000, size=0x80), _EmptyBlock(addr=0x401100, size=0x40)],
        ),
        (
            0x402000,
            [_EmptyBlock(addr=0x402000, size=0x30), _EmptyBlock(addr=0x402050, size=0x10)],
        ),
    ]

    expected: list[tuple[int, int]] = []
    sidecar = FunctionRangeSidecar(tmp_path / "demo_function_ranges.txt")
    # One buffer reused across both functions — mirrors main_loop's
    # grow-only reuse (fill_constant_candidates resets it on entry).
    func_tokens = FunctionTokenList(num_blocks=4, vocab_manager=vm)
    for entry, blocks in functions:
        result = fill_constant_candidates(
            func_addr=entry,
            func=_EmptyFunction(blocks=blocks),
            instr_sets=None,
            lookup=None,
            text_start=0,
            text_end=0x10000,
            resolver=resolver,
            vocab_manager=vm,
            arch_provider=None,
            func_tokens=func_tokens,
            disasm_provider=None,
        )
        assert result is not None
        *_, mn, mx = result
        sidecar.add(mn, mx)
        expected.append((mn, mx))
    sidecar.close()

    parsed = list(iter_sidecar_lines(tmp_path / "demo_function_ranges.txt"))
    assert parsed == expected, f"sidecar lines do not match producer values: {parsed!r} vs {expected!r}"
    # Sanity: line count == number of input functions (co-stepping
    # contract — one line per function).
    assert len(parsed) == len(functions)
