"""Unit tests for ``tokenizer.inspector._label`` helpers.

Pure-Python (no Textual, no I/O); covers the four positional axes,
FID-to-name resolution, block preview truncation, inline call/jump
labels, and the POSITIONAL_PREFIXES axis-order invariant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.inspector._label import (
    block_preview,
    function_label,
    inline_call_label,
    inline_jump_label,
    resolve_function_name_for_fid,
    variant_label,
)
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    POSITIONAL_PREFIXES,
)


# ---------------------------------------------------------------------------
# function_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("main", "local function main"),
        (None, "function ?"),
        # Empty string is a name (not None); helper takes the resolved-name
        # branch verbatim. Pins existing behavior — degrade gracefully.
        ("", "local function "),
        ("weird name with spaces", "local function weird name with spaces"),
    ],
)
def test_function_label(name, expected):
    assert function_label(name) == expected


# ---------------------------------------------------------------------------
# variant_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (
            {"arch": "x86", "compiler": "clang", "compilerversion": "8.0", "opt": "O3"},
            "x86 clang v8.0 -O3",
        ),
        (
            {"arch": "arm64", "compiler": "gcc", "compilerversion": "11", "opt": "O0"},
            "arm64 gcc v11 -O0",
        ),
        # All four axes missing -> all "?".
        ({}, "? ? v? -?"),
        # Single missing axis -> "?" only for that axis.
        (
            {"compiler": "clang", "compilerversion": "8.0", "opt": "O3"},
            "? clang v8.0 -O3",
        ),
        (
            {"arch": "x86", "compilerversion": "8.0", "opt": "O3"},
            "x86 ? v8.0 -O3",
        ),
        (
            {"arch": "x86", "compiler": "clang", "opt": "O3"},
            "x86 clang v? -O3",
        ),
        (
            {"arch": "x86", "compiler": "clang", "compilerversion": "8.0"},
            "x86 clang v8.0 -?",
        ),
    ],
)
def test_variant_label(metadata, expected):
    fd = MagicMock()
    fd.metadata = metadata
    assert variant_label(fd) == expected


def test_variant_label_coerces_non_string_values():
    """Non-string axis values are str()-coerced (e.g. int cver)."""
    fd = MagicMock()
    fd.metadata = {"arch": "x86", "compiler": "clang", "compilerversion": 11, "opt": "O2"}
    assert variant_label(fd) == "x86 clang v11 -O2"


# ---------------------------------------------------------------------------
# resolve_function_name_for_fid
# ---------------------------------------------------------------------------


def test_resolve_function_name_hit():
    assert resolve_function_name_for_fid(1, {1: "main"}) == "main"


def test_resolve_function_name_miss():
    assert resolve_function_name_for_fid(99, {1: "main"}) is None


def test_resolve_function_name_empty_map():
    assert resolve_function_name_for_fid(0, {}) is None


def test_resolve_function_name_multiple_entries():
    line_to_name = {1: "main", 2: "helper", 42: "answer"}
    assert resolve_function_name_for_fid(2, line_to_name) == "helper"
    assert resolve_function_name_for_fid(42, line_to_name) == "answer"
    assert resolve_function_name_for_fid(100, line_to_name) is None


# ---------------------------------------------------------------------------
# block_preview
# ---------------------------------------------------------------------------


def _block_with_asm(asm: str) -> MagicMock:
    block = MagicMock()
    block.to_asm_like.return_value = asm
    return block


def test_block_preview_short_returns_full_string():
    block = _block_with_asm("mov eax, ebx")
    assert block_preview(block) == "mov eax, ebx"


def test_block_preview_at_boundary_returns_full_string():
    asm = "a" * 80
    assert block_preview(_block_with_asm(asm)) == asm


def test_block_preview_long_truncates_to_max_chars():
    asm = "a" * 200
    preview = block_preview(_block_with_asm(asm))
    assert preview == "a" * 80
    assert len(preview) == 80


def test_block_preview_custom_max_chars():
    asm = "abcdefghijklmnop"
    assert block_preview(_block_with_asm(asm), max_chars=5) == "abcde"


def test_block_preview_does_not_append_overflow_marker():
    """Truncation is raw — the UI layer owns the ``>>`` marker, not us."""
    asm = "x" * 200
    preview = block_preview(_block_with_asm(asm))
    assert ">>" not in preview
    assert not preview.endswith(">")


def test_block_preview_empty_string():
    assert block_preview(_block_with_asm("")) == ""


# ---------------------------------------------------------------------------
# inline_call_label
# ---------------------------------------------------------------------------


def test_inline_call_label_local():
    assert (
        inline_call_label(CallTargetType.LOCAL, 0, "foo")
        == "local function 0: foo"
    )


def test_inline_call_label_plt():
    assert (
        inline_call_label(CallTargetType.PLT, 1, "printf")
        == "plt function 1: printf"
    )


def test_inline_call_label_extern_with_provider():
    assert (
        inline_call_label(CallTargetType.EXTERN, 2, "puts", "libc.so")
        == "ext function 2: puts@libc.so"
    )


def test_inline_call_label_extern_without_provider():
    """No provider -> ``@?`` fallback, preserving the visual shape."""
    assert (
        inline_call_label(CallTargetType.EXTERN, 3, "exit")
        == "ext function 3: exit@?"
    )


def test_inline_call_label_local_unknown_callee():
    assert (
        inline_call_label(CallTargetType.LOCAL, 5, None)
        == "local function 5: ?"
    )


def test_inline_call_label_plt_unknown_callee():
    assert (
        inline_call_label(CallTargetType.PLT, 7, None)
        == "plt function 7: ?"
    )


def test_inline_call_label_extern_unknown_callee_with_provider():
    assert (
        inline_call_label(CallTargetType.EXTERN, 9, None, "libm.so")
        == "ext function 9: ?@libm.so"
    )


def test_inline_call_label_extern_unknown_callee_and_provider():
    assert (
        inline_call_label(CallTargetType.EXTERN, 11, None)
        == "ext function 11: ?@?"
    )


def test_inline_call_label_provider_ignored_for_non_extern():
    """LOCAL / PLT never emit an ``@provider`` suffix even if one is supplied."""
    assert (
        inline_call_label(CallTargetType.LOCAL, 0, "foo", provider="should_ignore")
        == "local function 0: foo"
    )
    assert (
        inline_call_label(CallTargetType.PLT, 1, "printf", provider="should_ignore")
        == "plt function 1: printf"
    )


# ---------------------------------------------------------------------------
# inline_jump_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx,expected", [(0, "jump block: 0"), (42, "jump block: 42")])
def test_inline_jump_label(idx, expected):
    assert inline_jump_label(idx) == expected


# ---------------------------------------------------------------------------
# POSITIONAL_PREFIXES sanity — pin axis order
# ---------------------------------------------------------------------------


def test_positional_prefixes_canonical_order_and_length():
    """Catches an upstream refactor that reorders the positional axes."""
    assert POSITIONAL_PREFIXES == (ARCH_PREFIX, COMP_PREFIX, CVER_PREFIX, OPT_PREFIX)
    assert len(POSITIONAL_PREFIXES) == 4


def test_variant_label_uses_positional_prefix_order():
    """The label order mirrors POSITIONAL_PREFIXES — arch, comp, cver, opt."""
    fd = MagicMock()
    fd.metadata = {
        "arch": "ARCHVAL",
        "compiler": "COMPVAL",
        "compilerversion": "CVERVAL",
        "opt": "OPTVAL",
    }
    label = variant_label(fd)
    # Per-axis substrings in the expected positional order.
    parts = label.split(" ")
    assert parts == ["ARCHVAL", "COMPVAL", "vCVERVAL", "-OPTVAL"]
