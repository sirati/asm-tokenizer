"""Unit tests for ``tokenizer.inspector._label`` helpers.

Pure-Python (no Textual, no I/O); covers the four positional axes,
FID-to-name resolution, block preview shape, inline call/jump
labels, and the POSITIONAL_PREFIXES axis-order invariant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.inspector._label import (
    aligned_variant_labels,
    block_preview_from_asm_texts,
    function_label,
    inline_call_label,
    inline_jump_label,
    resolve_function_name_for_fid,
    variant_label,
    variant_label_from_axes,
)
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    POSITIONAL_PREFIXES,
)


def _axes(arch=None, comp=None, cver=None, opt=None):
    return {
        ARCH_PREFIX: arch,
        COMP_PREFIX: comp,
        CVER_PREFIX: cver,
        OPT_PREFIX: opt,
    }


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
# block_preview_from_asm_texts -- full-length join; no fixed-char cap
# ---------------------------------------------------------------------------


def test_block_preview_from_asm_texts_short_joins_full_string():
    """Short input joins with ``"; "`` and returns unchanged."""
    assert (
        block_preview_from_asm_texts(["mov eax, ebx", "ret"])
        == "mov eax, ebx; ret"
    )


def test_block_preview_from_asm_texts_long_returns_full_string():
    """No fixed-char cap -- the FULL joined string flows out so the
    tree widget's per-row horizontal-scroll can pan past the viewport
    edge. A fixed-char truncation here would strip the scrollable
    content before the user reaches it."""
    texts = ["instr" + str(i) for i in range(100)]
    out = block_preview_from_asm_texts(texts)
    assert out == "; ".join(texts)
    # No truncation -- the full join, regardless of length.
    assert len(out) > 80


def test_block_preview_from_asm_texts_does_not_append_overflow_marker():
    """The UI layer (:func:`apply_truncation_marker`) owns the ``>>``
    marker; this pure helper never appends one."""
    texts = ["x" * 100 for _ in range(5)]
    out = block_preview_from_asm_texts(texts)
    assert ">>" not in out
    assert not out.endswith(">")


def test_block_preview_from_asm_texts_empty_iterable():
    assert block_preview_from_asm_texts([]) == ""


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


# ---------------------------------------------------------------------------
# aligned_variant_labels — sibling-set column alignment
# ---------------------------------------------------------------------------


def test_aligned_variant_labels_pads_to_widest_value_per_axis():
    """The user-spec example: ``arm32`` widens the arch column to 5 so
    ``x86`` becomes ``x86  ``; ``clang`` widens the compiler column to
    5 so ``gcc`` becomes ``gcc  ``."""
    rows = [
        _axes(arch="arm32", comp="gcc", cver="5", opt="O0"),
        _axes(arch="x86", comp="clang", cver="7", opt="O3"),
    ]
    aligned = aligned_variant_labels(rows)
    # arch column width = max(len("arm32"), len("x86")) = 5
    # comp column width = max(len("gcc"), len("clang")) = 5
    # cver column width = max(len("v5"), len("v7")) = 2
    # opt column = trailing, not padded
    assert aligned == (
        "arm32 gcc   v5 -O0",
        "x86   clang v7 -O3",
    )


def test_aligned_variant_labels_empty_input_returns_empty_tuple():
    assert aligned_variant_labels([]) == ()


def test_aligned_variant_labels_single_row_self_aligns():
    """A one-element sibling set: each column's max == its only value,
    so no padding is inserted (apart from the trailing-column rule)."""
    rows = [_axes(arch="x86", comp="clang", cver="8.0", opt="O3")]
    aligned = aligned_variant_labels(rows)
    assert aligned == ("x86 clang v8.0 -O3",)


def test_aligned_variant_labels_matches_variant_label_when_columns_uniform():
    """When every sibling's per-axis value has the same width, the
    aligned form collapses to the same string :func:`variant_label_from_axes`
    would emit — no extra whitespace is introduced."""
    rows = [
        _axes(arch="x86", comp="clang", cver="8", opt="O0"),
        _axes(arch="x86", comp="clang", cver="8", opt="O3"),
    ]
    aligned = aligned_variant_labels(rows)
    for row, line in zip(rows, aligned):
        assert line == variant_label_from_axes(row)


def test_aligned_variant_labels_missing_axes_still_pad():
    """``None`` axes render as ``?`` with their per-axis prefix (e.g.
    ``v?`` / ``-?``); the column width is computed from the rendered
    string so ``?`` siblings stay aligned with longer real values."""
    rows = [
        _axes(arch="x86", comp=None, cver="8.0", opt="O3"),
        _axes(arch="arm64", comp="gcc", cver=None, opt="O2"),
    ]
    aligned = aligned_variant_labels(rows)
    # arch width = 5 (arm64), comp width = 3 (gcc), cver width = 4 (v8.0)
    assert aligned == (
        "x86   ?   v8.0 -O3",
        "arm64 gcc v?   -O2",
    )


def test_aligned_variant_labels_trailing_axis_not_right_padded():
    """The last column (``-opt``) gets no trailing whitespace — column
    padding only fills the gap before the NEXT column."""
    rows = [
        _axes(arch="x86", comp="gcc", cver="5", opt="O0"),
        _axes(arch="x86", comp="gcc", cver="5", opt="Olong"),
    ]
    aligned = aligned_variant_labels(rows)
    # Neither row ends with extra spaces, even though -O0 is shorter
    # than -Olong.
    for line in aligned:
        assert not line.endswith(" ")


def test_aligned_variant_labels_preserves_input_order():
    """Output order is lockstep with input — pass-through, no sorting."""
    rows = [
        _axes(arch="z80", comp="gcc", cver="1", opt="O0"),
        _axes(arch="aaa", comp="gcc", cver="1", opt="O0"),
        _axes(arch="mid", comp="gcc", cver="1", opt="O0"),
    ]
    aligned = aligned_variant_labels(rows)
    # Each output starts with the corresponding input's arch token
    # (post-padding), in the input order.
    assert aligned[0].startswith("z80")
    assert aligned[1].startswith("aaa")
    assert aligned[2].startswith("mid")


# ---------------------------------------------------------------------------
# aligned_variant_labels — grouping-ancestor axis suppression
# ---------------------------------------------------------------------------


def test_aligned_variant_labels_suppressed_axes_dropped_from_every_row():
    """When the caller passes the canonical-prefix keys disclosed by a
    grouping-ancestor chain, those axis columns are dropped from each
    row's rendered label — so the variant row stops repeating values
    the user already sees on the group rows above."""
    rows = [
        _axes(arch="arm32", comp="clang", cver="5.0", opt="O0"),
        _axes(arch="arm64", comp="clang", cver="5.0", opt="O0"),
        _axes(arch="x86", comp="gcc", cver="5.0", opt="O0"),
    ]
    aligned = aligned_variant_labels(
        rows, suppressed_axes=frozenset({CVER_PREFIX, OPT_PREFIX})
    )
    # Only arch + compiler columns survive; widths recomputed.
    assert aligned == (
        "arm32 clang",
        "arm64 clang",
        "x86   gcc",
    )


def test_aligned_variant_labels_all_axes_suppressed_yields_empty_rows():
    """4-deep grouping covers every positional axis; each variant
    row's label collapses to ``""`` — the tree structure has already
    disclosed everything and the row's prefix glyph alone marks the
    leaf."""
    rows = [
        _axes(arch="x86", comp="clang", cver="8.0", opt="O3"),
        _axes(arch="x86", comp="clang", cver="8.0", opt="O3"),
    ]
    aligned = aligned_variant_labels(
        rows,
        suppressed_axes=frozenset(
            {ARCH_PREFIX, COMP_PREFIX, CVER_PREFIX, OPT_PREFIX}
        ),
    )
    assert aligned == ("", "")


def test_aligned_variant_labels_suppress_single_middle_axis():
    """Dropping a middle axis preserves the surrounding axes' order
    (``arch comp v? -opt`` minus ``comp`` -> ``arch v? -opt``)."""
    rows = [
        _axes(arch="x86", comp="clang", cver="8.0", opt="O3"),
        _axes(arch="arm64", comp="clang", cver="9.1", opt="O0"),
    ]
    aligned = aligned_variant_labels(
        rows, suppressed_axes=frozenset({COMP_PREFIX})
    )
    # arch + cver + opt only. arch width = 5 (arm64). cver width = 4 (v8.0 / v9.1).
    assert aligned == (
        "x86   v8.0 -O3",
        "arm64 v9.1 -O0",
    )


def test_aligned_variant_labels_unknown_suppressed_key_is_no_op():
    """A suppressed key not in :data:`POSITIONAL_PREFIXES` is silently
    ignored — the canonical-prefix filter is the source of truth for
    column membership."""
    rows = [_axes(arch="x86", comp="clang", cver="8.0", opt="O3")]
    aligned = aligned_variant_labels(
        rows, suppressed_axes=frozenset({"not-a-real-prefix:"})
    )
    assert aligned == ("x86 clang v8.0 -O3",)


def test_variant_label_from_axes_suppressed_axis_dropped():
    """Non-aligned fallback path honours the same suppression policy."""
    axes = _axes(arch="x86", comp="clang", cver="8.0", opt="O3")
    label = variant_label_from_axes(
        axes, suppressed_axes=frozenset({OPT_PREFIX, CVER_PREFIX})
    )
    assert label == "x86 clang"


def test_variant_label_from_axes_all_axes_suppressed_returns_empty():
    """All-positional suppression collapses the fallback label to ``""``."""
    axes = _axes(arch="x86", comp="clang", cver="8.0", opt="O3")
    label = variant_label_from_axes(
        axes,
        suppressed_axes=frozenset(
            {ARCH_PREFIX, COMP_PREFIX, CVER_PREFIX, OPT_PREFIX}
        ),
    )
    assert label == ""


def test_variant_label_from_axes_default_suppressed_axes_preserves_legacy():
    """Default ``suppressed_axes`` is empty — un-grouped behavior is
    byte-identical to the pre-suppression rendering."""
    axes = _axes(arch="x86", comp="clang", cver="8.0", opt="O3")
    assert variant_label_from_axes(axes) == "x86 clang v8.0 -O3"


def test_aligned_variant_labels_default_suppressed_axes_preserves_legacy():
    """Default ``suppressed_axes`` is empty — un-grouped behavior is
    byte-identical to the pre-suppression alignment."""
    rows = [
        _axes(arch="arm32", comp="gcc", cver="5", opt="O0"),
        _axes(arch="x86", comp="clang", cver="7", opt="O3"),
    ]
    assert aligned_variant_labels(rows) == (
        "arm32 gcc   v5 -O0",
        "x86   clang v7 -O3",
    )
