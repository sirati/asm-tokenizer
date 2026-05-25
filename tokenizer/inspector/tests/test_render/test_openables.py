"""Shape tests for :data:`Openable` + :attr:`AsmLine.openables`.

The Wave-5 R1a additive extension of :mod:`tokenizer.inspector._render._protocol`
adds a typed sidecar to :class:`AsmLine`. These tests pin the contract
downstream consumers (row-walker emission + tree-model leaf expansion)
will couple to:

* the new :class:`InlineNumberPrecisionEntry` dataclass shape
  (typed :class:`TokenType` + pre-rendered text),
* :data:`Openable` Union membership (frozen-dataclass identity is
  the single discriminator -- there is no ``OpenableKind`` enum),
* :attr:`AsmLine.openables` default + element-type plumbing.

The dataclasses are all ``frozen=True`` -- mutation must raise
``FrozenInstanceError`` so the row-walker can hand the same
in-process instance to multiple tree-model consumers without
defensive copying.
"""

from __future__ import annotations

import dataclasses

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.inspector._render import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    InlineNumberPrecisionEntry,
    Openable,
)
from tokenizer.tokens import TokenType


# ---------------------------------------------------------------------------
# AsmLine.openables -- default + element typing
# ---------------------------------------------------------------------------


def test_asm_line_openables_defaults_to_empty_tuple():
    """A bare :class:`AsmLine` carries an empty ``openables`` -- no
    sidecar = leaf row in the tree-model. The default is a tuple (not
    a list) so the frozen-dataclass immutability extends to the
    sidecar payload."""
    line = AsmLine(text="mov rax, 0x1")
    assert line.openables == ()
    assert isinstance(line.openables, tuple)


def test_asm_line_openables_preserves_supplied_tuple():
    """Caller-supplied tuple flows through verbatim (no re-wrapping /
    sorting / dedup -- the row walker is the producer and decides the
    order)."""
    jump_a = InlineJumpEntry(target_block_idx=3)
    jump_b = InlineJumpEntry(target_block_idx=7)
    line = AsmLine(text="<jump_table 2>", openables=(jump_a, jump_b))
    assert line.openables == (jump_a, jump_b)


# ---------------------------------------------------------------------------
# Frozen-dataclass invariants
# ---------------------------------------------------------------------------


def test_asm_line_is_frozen():
    """Mutating :attr:`AsmLine.openables` after construction is a
    contract violation -- downstream consumers may have already cached
    the tuple identity. ``FrozenInstanceError`` enforces."""
    line = AsmLine(text="nop")
    with pytest.raises(dataclasses.FrozenInstanceError):
        line.openables = (InlineJumpEntry(target_block_idx=0),)  # type: ignore[misc]


def test_inline_call_entry_is_frozen():
    entry = InlineCallEntry(
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="foo",
        callee_section_pointer=None,
        variant_idx=0,
        provider=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.counter_id = 1  # type: ignore[misc]


def test_inline_jump_entry_is_frozen():
    entry = InlineJumpEntry(target_block_idx=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.target_block_idx = 3  # type: ignore[misc]


def test_inline_number_precision_entry_is_frozen():
    entry = InlineNumberPrecisionEntry(
        token_type=TokenType.FLOAT64, full_text="f64:3.141592653589793"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.full_text = "f64:0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InlineNumberPrecisionEntry shape -- typed fields only
# ---------------------------------------------------------------------------


def test_inline_number_precision_entry_dataclass_field_set():
    """Pin the wire-shape: ``token_type`` + ``full_text`` only.

    The integrated plan W3-1 W4-amendment dropped the speculative
    ``chunks`` field (no consumer for per-chunk decomposition). Adding
    a field back here would silently expand the producer contract --
    this test catches the drift.
    """
    expected = {"token_type", "full_text"}
    actual = {f.name for f in InlineNumberPrecisionEntry.__dataclass_fields__.values()}
    assert actual == expected


def test_inline_number_precision_entry_constructs_with_typed_fields():
    """``token_type`` is a :class:`TokenType` enum member; ``full_text``
    is the pre-rendered display string. Both backends will produce
    these; the consumer reads :attr:`full_text` verbatim."""
    entry = InlineNumberPrecisionEntry(
        token_type=TokenType.FLOAT80, full_text="f80:1.2345678901234567890"
    )
    assert entry.token_type is TokenType.FLOAT80
    assert entry.full_text == "f80:1.2345678901234567890"


# ---------------------------------------------------------------------------
# Openable union -- frozen-dataclass identity is the discriminator
# ---------------------------------------------------------------------------


def test_openable_union_members():
    """The :data:`Openable` Union covers exactly the three sidecar
    dataclass types. Drift here would silently change the tree-model
    dispatch surface (per W3-2 W4-amended: NO ``OpenableKind`` enum,
    NO ``openable_kind`` property -- ``isinstance`` over frozen
    dataclasses is the SOLE discriminator)."""
    members = set(Openable.__args__)
    assert members == {
        InlineCallEntry,
        InlineJumpEntry,
        InlineNumberPrecisionEntry,
    }


def test_openable_isinstance_dispatch():
    """The ``isinstance`` discriminator works on each concrete shape --
    pinned because the tree-model's ``_wrap_openable_as_node`` helper
    (landing in a later R2 phase) routes purely off these checks."""
    call = InlineCallEntry(
        kind=CallTargetType.EXTERN,
        counter_id=4,
        callee_name="printf",
        callee_section_pointer=None,
        variant_idx=0,
        provider="libc",
    )
    jump = InlineJumpEntry(target_block_idx=5)
    precision = InlineNumberPrecisionEntry(
        token_type=TokenType.FLOAT128, full_text="f128:0"
    )

    assert isinstance(call, InlineCallEntry)
    assert not isinstance(call, InlineJumpEntry)
    assert not isinstance(call, InlineNumberPrecisionEntry)

    assert isinstance(jump, InlineJumpEntry)
    assert not isinstance(jump, InlineCallEntry)
    assert not isinstance(jump, InlineNumberPrecisionEntry)

    assert isinstance(precision, InlineNumberPrecisionEntry)
    assert not isinstance(precision, InlineCallEntry)
    assert not isinstance(precision, InlineJumpEntry)


def test_no_openable_kind_enum_or_property():
    """W3-2 W4-amendment: there is intentionally NO ``OpenableKind``
    enum and NO ``openable_kind`` property on any of the three
    Openable dataclasses. Frozen-dataclass identity is the SINGLE
    discriminator -- three parallel discriminators (enum + property +
    isinstance) was dead code per the audit cluster."""
    import tokenizer.inspector._render as render_pkg

    assert not hasattr(render_pkg, "OpenableKind")

    for cls in (InlineCallEntry, InlineJumpEntry, InlineNumberPrecisionEntry):
        assert not hasattr(cls, "openable_kind"), (
            f"{cls.__name__}.openable_kind must not exist (W3-2 W4-amended)"
        )


def test_asm_line_carries_mixed_openables():
    """One :class:`AsmLine` can carry openables of multiple types --
    e.g. an instruction with both a call and a number precision
    sidecar. The tree-model walks the tuple in order; type dispatch
    happens per-element."""
    call = InlineCallEntry(
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="local_fn",
        callee_section_pointer=None,
        variant_idx=0,
        provider=None,
    )
    precision = InlineNumberPrecisionEntry(
        token_type=TokenType.VALUED_CONST_V2, full_text="v:0x0123456789abcdef (...)"
    )
    line = AsmLine(text="call local_fn", openables=(call, precision))
    assert len(line.openables) == 2
    assert isinstance(line.openables[0], InlineCallEntry)
    assert isinstance(line.openables[1], InlineNumberPrecisionEntry)
