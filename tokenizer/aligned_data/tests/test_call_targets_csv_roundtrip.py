"""Round-trip unit tests for the Phase 4.1 call-target CSV helpers.

Pins :func:`tokenizer.aligned_data.csv_format.format_call_targets_dict`
↔ :func:`tokenizer.aligned_data.metadata.parse_call_targets` and
:func:`tokenizer.aligned_data.csv_format.format_called_line_nos_typed`
↔ :func:`tokenizer.aligned_data.metadata.parse_called_line_nos_typed`
against the on-disk cell wire format. The pair is the single source
of truth for the section CSV header; a writer / reader drift in
either direction would silently corrupt the validator's CSV cross-
check, so the round-trip is locked here.

Single concern: format → parse equivalence per cell shape. Builder-
side dedup, ordering, and arm-specific cell composition stay covered
by their own pass-2 tests; the helpers under test are pure on their
inputs.
"""

from __future__ import annotations

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import (
    format_call_targets_dict,
    format_called_line_nos_typed,
)
from tokenizer.aligned_data.metadata import (
    parse_call_targets,
    parse_called_line_nos_typed,
)


# ---------------------------------------------------------------------------
# format_call_targets_dict <-> parse_call_targets
# ---------------------------------------------------------------------------


def test_call_targets_roundtrip_matched_arm_cell():
    """Matched-arm cell: ``idx,hex_offset,is_matched`` triples round-trip
    through format → parse to the same triple list (no ``comp_set``
    fragment on the matched side)."""
    triples = [
        [0, 0x10, 1],
        [3, 0xABCDEF, 0],
        [7, 0x0, 1],
    ]
    encoded = format_call_targets_dict(triples)
    decoded = parse_call_targets(encoded)
    assert decoded == triples


def test_call_targets_roundtrip_unmatched_arm_cell_with_comp_set():
    """Unmatched-arm cell prepends ``-comp_set`` to ``idx``; the parser
    discards the fragment and recovers the bare ``[idx, offset,
    is_matched]`` triples. The format helper itself doesn't emit the
    comp_set decoration (the builder grafts it on at write time), so
    we hand-build the wire form to exercise the parser's strip path.
    """
    wire = "0-42,10,1;5-99,abcdef,0;9-7,4,1"
    decoded = parse_call_targets(wire)
    assert decoded == [
        [0, 0x10, 1],
        [5, 0xABCDEF, 0],
        [9, 0x4, 1],
    ]


def test_call_targets_roundtrip_empty_cell():
    """Empty input -> empty list -> empty input (round-trip stays
    closed on the degenerate case)."""
    assert parse_call_targets("") == []
    assert format_call_targets_dict([]) == ""


# ---------------------------------------------------------------------------
# format_called_line_nos_typed <-> parse_called_line_nos_typed
# ---------------------------------------------------------------------------


def _type_char_pairs():
    """One ``(line_no, type)`` per L/P/E discriminator; the line_nos
    are intentionally spread across the base64-codec's byte boundaries
    so the round-trip exercises the codec, not a single happy value.
    """
    return [
        (1, CallTargetType.LOCAL),
        (255, CallTargetType.PLT),
        (65535, CallTargetType.EXTERN),
    ]


def test_called_line_nos_typed_roundtrip_all_three_types():
    """Every type discriminator (LOCAL/PLT/EXTERN -> L/P/E) survives
    the format → parse pair. The base64-encoded line-no half is
    delegated to the line_no_codec, which has its own coverage; this
    test pins the type-tag wiring alongside it.
    """
    pairs = _type_char_pairs()
    encoded = format_called_line_nos_typed(pairs)
    decoded = parse_called_line_nos_typed(encoded)
    assert decoded == pairs


def test_called_line_nos_typed_roundtrip_empty_cell():
    """Empty sequence formats to ``""`` and parses back to ``[]``;
    closure on the degenerate case."""
    assert format_called_line_nos_typed([]) == ""
    assert parse_called_line_nos_typed("") == []


def test_called_line_nos_typed_missing_type_tag_raises():
    """An entry without a ``:<type_char>`` suffix is a builder bug
    (the Phase-3 fix added the tag so PLT and EXTERN call-sites with
    the same callee name no longer silently coalesce). The parser
    must surface it as :class:`ValueError`, not return a half-decoded
    pair.
    """
    # Encode one valid pair so the cell has structure, then drop the
    # ``:L`` suffix off the second entry to simulate a stale builder
    # output landing in the validator.
    valid = format_called_line_nos_typed([(1, CallTargetType.LOCAL)])
    untyped = "AAEB"  # base64 line_no, no ":<char>" suffix
    wire = f"{valid},{untyped}"
    with pytest.raises(ValueError, match="missing type tag"):
        parse_called_line_nos_typed(wire)


def test_called_line_nos_typed_unknown_type_char_raises():
    """An unrecognised type character (anything outside ``L``/``P``/
    ``E``) is a wire-format corruption; the parser must raise
    :class:`ValueError`, not silently fall through.
    """
    valid = format_called_line_nos_typed([(1, CallTargetType.LOCAL)])
    bogus = "AAEB:X"
    wire = f"{valid},{bogus}"
    with pytest.raises(ValueError, match="unknown type char"):
        parse_called_line_nos_typed(wire)
