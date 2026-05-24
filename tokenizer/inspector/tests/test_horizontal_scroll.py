"""Tests for :mod:`tokenizer.inspector._horizontal_scroll`.

Pins the truncation-marker contract used by the inspector tree's
``render_label`` override. The marker is a cosmetic ``" >>"`` hint
appended when the rendered label's cell-width spills past the
viewport's rightmost visible column. The tests cover:

* the spill/no-spill boundary (label width vs ``viewport_width + scroll_x``);
* the unmounted/zero-viewport short-circuit;
* the mutation-immune contract (TreeNode caches labels across renders,
  so the helper must return a fresh :class:`Text` when augmenting);
* the dim styling of the marker;
* the failed-glyph (text, style) tuple, including base-style composition.

No Textual dependency -- the helper is pure rich.
"""

from __future__ import annotations

import pytest
from rich.style import Style
from rich.text import Text

from tokenizer.inspector._horizontal_scroll import (
    MARKER_STYLE,
    MARKER_TEXT,
    apply_truncation_marker,
    assemble_failed_glyph,
)


class TestApplyTruncationMarkerNoSpill:
    """Configurations where the label fits inside the visible window.

    Identity passthrough is the contract -- the helper returns the
    input :class:`Text` unchanged (same object) when no marker is
    needed, so callers can cheaply detect "nothing to do".
    """

    @pytest.mark.parametrize(
        ("plain", "viewport_width", "scroll_x"),
        [
            # 1. label fits comfortably inside the viewport.
            ("hello", 80, 0),
            # 3. scroll_x widens the visible window: cols 30..110 cover
            #    a 100-char label entirely.
            ("x" * 100, 80, 30),
            # 5. unmounted / zero-size viewport short-circuits to the
            #    identity passthrough even on a long label.
            ("x" * 200, 0, 0),
        ],
    )
    def test_label_returned_unchanged(
        self, plain: str, viewport_width: int, scroll_x: int
    ) -> None:
        label = Text(plain)
        out = apply_truncation_marker(label, viewport_width, scroll_x)
        # Identity is the strict contract -- a fresh copy would force
        # callers to discard cached references unnecessarily.
        assert out is label
        assert out.plain == plain


class TestApplyTruncationMarkerSpill:
    """Configurations where the label overflows the visible window."""

    @pytest.mark.parametrize(
        ("plain", "viewport_width", "scroll_x"),
        [
            # 2. label wider than viewport, no scroll.
            ("x" * 200, 80, 0),
            # 4. scroll_x widens to cols 19..99; col 100 (the label's
            #    last char) is still off-screen, so the marker fires.
            ("x" * 100, 80, 19),
        ],
    )
    def test_marker_appended(
        self, plain: str, viewport_width: int, scroll_x: int
    ) -> None:
        label = Text(plain)
        out = apply_truncation_marker(label, viewport_width, scroll_x)
        # Spill case must return a *different* Text (no in-place mutation).
        assert out is not label
        assert out.plain.endswith(MARKER_TEXT)
        # plain text grew by exactly len(MARKER_TEXT) cells.
        assert len(out.plain) == len(plain) + len(MARKER_TEXT)


class TestApplyTruncationMarkerImmutable:
    """The input label MUST NEVER be mutated.

    Textual's ``TreeNode`` caches labels across renders. An in-place
    ``append`` on the cached label would compound a fresh marker on
    every paint, so the helper copies before appending.
    """

    def test_input_plain_text_preserved(self) -> None:
        original = Text("x" * 200)
        original_plain = original.plain
        original_span_count = len(original.spans)
        apply_truncation_marker(original, 80, 0)
        assert original.plain == original_plain
        assert len(original.spans) == original_span_count


class TestApplyTruncationMarkerStyling:
    """Marker carries :data:`MARKER_STYLE` (dim) so it stays visually
    subordinate to the actual label content."""

    def test_appended_span_has_marker_style(self) -> None:
        label = Text("x" * 200)
        out = apply_truncation_marker(label, 80, 0)
        # Locate the appended span -- it covers the trailing
        # ``len(MARKER_TEXT)`` cells and carries MARKER_STYLE.
        marker_start = len(label.plain)
        marker_end = marker_start + len(MARKER_TEXT)
        trailing = [
            span
            for span in out.spans
            if span.start == marker_start and span.end == marker_end
        ]
        assert len(trailing) == 1
        assert trailing[0].style == MARKER_STYLE


class TestModuleConstants:
    """Pins the public surface so consumers can rely on the strings."""

    def test_marker_text_is_double_chevron_with_leading_space(self) -> None:
        assert MARKER_TEXT == " >>"

    def test_marker_style_is_dim(self) -> None:
        assert MARKER_STYLE.dim is True


class TestAssembleFailedGlyph:
    """The ``[*]`` failed-expand prefix glyph (plan D8).

    Returns a ``(text, style)`` tuple consumable by
    :meth:`Text.assemble`; the style composes the caller's base on top
    of the bold-red emphasis so per-node highlight carries through.
    """

    def test_returns_text_and_style_tuple(self) -> None:
        text, style = assemble_failed_glyph(Style())
        assert text == "[*] "
        # bold + red are the failed-emphasis attributes.
        assert style.bold is True
        assert style.color is not None
        assert style.color.name == "red"

    def test_base_style_composes_under_failed_emphasis(self) -> None:
        # The caller's base style (e.g. cursor / per-node highlight)
        # must carry through; Rich's ``Style + Style`` operator merges
        # right-on-top, so attributes set only on the base persist.
        base = Style(italic=True)
        _, style = assemble_failed_glyph(base)
        assert style.italic is True
        # And the failed emphasis still wins on overlapping fields.
        assert style.bold is True
        assert style.color is not None
        assert style.color.name == "red"
