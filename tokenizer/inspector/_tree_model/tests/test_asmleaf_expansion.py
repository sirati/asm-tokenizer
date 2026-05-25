"""Unit tests for the :class:`AsmLeaf` 3-arm expand contract.

Covers the post-R2e openables-sidecar shape (cluster #3 of the
integrated plan):

* 0 openables -> terminal leaf, ``can_expand == False``.
* 1 openable  -> 1-arm dispatch: ``expand`` returns the openable's
                  target node directly (no wrapper row).
* 2+ openables -> wrap-each: ``expand`` returns one wrapper-AsmLeaf
                  per openable; each wrapper itself follows the 1-arm
                  rule on its lone sidecar entry.

Match-statement dispatch on openable dataclass identity is exercised
via the three concrete Openable types (:class:`InlineCallEntry`,
:class:`InlineJumpEntry`, :class:`InlineNumberPrecisionEntry`).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._label import inline_call_label, inline_jump_label
from tokenizer.inspector._render._protocol import (
    BackendFactory,
    InlineCallEntry,
    InlineJumpEntry,
    RenderBackend,
)
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    InlineCallNode,
    InlineJumpNode,
    NumberPrecisionLeaf,
)
from tokenizer.tokens import TokenType


# ---------------------------------------------------------------------------
# Fixtures -- typed openable builders + parent-ref mocks.
# ---------------------------------------------------------------------------


def _make_inline_call_entry(
    callee_name: str = "callee",
    callee_section_pointer: SectionPointerSpec | None = None,
) -> InlineCallEntry:
    return InlineCallEntry(
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name=callee_name,
        callee_section_pointer=callee_section_pointer,
        variant_idx=0,
        provider=None,
    )


def _make_inline_jump_entry(target_block_idx: int = 3) -> InlineJumpEntry:
    return InlineJumpEntry(target_block_idx=target_block_idx)


def _make_precision_entry(full_text: str = "v:1F (31)") -> InlineNumberPrecisionEntry:
    return InlineNumberPrecisionEntry(
        token_type=TokenType.VALUED_CONST_V2, full_text=full_text
    )


# ---------------------------------------------------------------------------
# 0-arm: ``openables == ()`` -- terminal, no expand.
# ---------------------------------------------------------------------------


def test_asmleaf_no_openables_cannot_expand():
    """A leaf with no sidecar entries is terminal; can_expand is False."""
    leaf = AsmLeaf(text="nop")
    assert leaf.can_expand is False


def test_asmleaf_no_openables_expand_raises():
    """The 0-arm contract: calling ``expand`` raises (UI gates on
    ``can_expand``)."""
    leaf = AsmLeaf(text="nop")
    with pytest.raises(NotImplementedError):
        leaf.expand()


# ---------------------------------------------------------------------------
# 1-arm: one openable -> one target node directly.
# ---------------------------------------------------------------------------


def test_asmleaf_single_inline_call_produces_inline_call_node():
    """A leaf carrying ONE :class:`InlineCallEntry` expands directly
    to an :class:`InlineCallNode` (no wrapper row)."""
    factory = MagicMock(spec=BackendFactory)
    entry = _make_inline_call_entry(
        callee_section_pointer=SectionPointerSpec(
            arm=SectionKind.MATCHED, idx=7
        )
    )
    leaf = AsmLeaf(
        text="call foo", openables=(entry,), factory=factory
    )

    assert leaf.can_expand is True
    children = leaf.expand()

    assert len(children) == 1
    call = children[0]
    assert isinstance(call, InlineCallNode)
    # Callee handle carries the section pointer's (arm, idx) + the
    # entry's callee_name (mirrors the legacy translation contract).
    assert call.callee_handle is not None
    assert call.callee_handle.arm is SectionKind.MATCHED
    assert call.callee_handle.idx == 7
    assert call.callee_handle.name == "callee"


def test_asmleaf_single_inline_jump_produces_inline_jump_node():
    """A leaf carrying ONE :class:`InlineJumpEntry` expands directly
    to an :class:`InlineJumpNode` (no wrapper row), with the parent
    BlockNode's factory + backend + variant_idx threaded through."""
    factory = MagicMock(spec=BackendFactory)
    backend = MagicMock(spec=RenderBackend)
    entry = _make_inline_jump_entry(target_block_idx=5)
    leaf = AsmLeaf(
        text="jmp .L5",
        openables=(entry,),
        factory=factory,
        backend=backend,
        variant_idx=2,
    )

    children = leaf.expand()

    assert len(children) == 1
    jump = children[0]
    assert isinstance(jump, InlineJumpNode)
    assert jump.target_block_idx == 5
    assert jump.variant_idx == 2
    # The factory/backend threaded through unchanged so the jump can
    # render its target block via the same backend cache.
    assert jump.factory is factory
    assert jump.backend is backend


def test_asmleaf_single_number_precision_produces_terminal_leaf():
    """A leaf carrying ONE :class:`InlineNumberPrecisionEntry` expands
    to a terminal :class:`NumberPrecisionLeaf` carrying the
    ``full_text`` (no further expand)."""
    entry = _make_precision_entry(full_text="v:DEADBEEF (3735928559)")
    leaf = AsmLeaf(text="v:DEAD (57005)", openables=(entry,))

    children = leaf.expand()

    assert len(children) == 1
    precision = children[0]
    assert isinstance(precision, NumberPrecisionLeaf)
    assert precision.text == "v:DEADBEEF (3735928559)"
    # Terminal: no further expand.
    assert precision.can_expand is False


# ---------------------------------------------------------------------------
# 2+-arm: each openable becomes one wrapper-AsmLeaf row.
# ---------------------------------------------------------------------------


def test_asmleaf_two_openables_produces_two_wrapper_rows():
    """A leaf carrying TWO openables (one call + one jump) expands
    into TWO wrapper :class:`AsmLeaf` rows; each row carries exactly
    its own openable so its own ``expand`` falls through the 1-arm
    path."""
    factory = MagicMock(spec=BackendFactory)
    backend = MagicMock(spec=RenderBackend)
    call_entry = _make_inline_call_entry(
        callee_section_pointer=SectionPointerSpec(
            arm=SectionKind.MATCHED, idx=1
        )
    )
    jump_entry = _make_inline_jump_entry(target_block_idx=4)
    leaf = AsmLeaf(
        text="call+jmp combo",
        openables=(call_entry, jump_entry),
        factory=factory,
        backend=backend,
        variant_idx=0,
    )

    children = leaf.expand()

    assert len(children) == 2
    for child in children:
        # Each wrapper IS an AsmLeaf with a single openable; its own
        # expand goes through the 1-arm dispatch.
        assert isinstance(child, AsmLeaf)
        assert len(child.openables) == 1
        assert child.can_expand is True

    # First wrapper carries the call entry; expanding produces an
    # InlineCallNode.
    first_grandchildren = children[0].expand()
    assert len(first_grandchildren) == 1
    assert isinstance(first_grandchildren[0], InlineCallNode)

    # Second wrapper carries the jump entry; expanding produces an
    # InlineJumpNode.
    second_grandchildren = children[1].expand()
    assert len(second_grandchildren) == 1
    assert isinstance(second_grandchildren[0], InlineJumpNode)


def test_wrapper_rows_use_canonical_label_helpers():
    """The wrapper-AsmLeaf row label (2+-arm case) routes per-openable
    label assembly through the canonical
    :func:`tokenizer.inspector._label.inline_call_label` /
    :func:`tokenizer.inspector._label.inline_jump_label` helpers --
    NOT a hand-rolled local format. Single source of truth for row
    labels lives in :mod:`tokenizer.inspector._label`; the 1-arm path
    (via :class:`InlineCallNode` / :class:`InlineJumpNode`) and the
    2+-arm wrapper path MUST agree on the visible string so a future
    canonical-helper edit propagates to both call sites.
    """
    call_entry = InlineCallEntry(
        kind=CallTargetType.EXTERN,
        counter_id=4,
        callee_name="printf",
        callee_section_pointer=None,
        variant_idx=0,
        provider="libc.so",
    )
    jump_entry = _make_inline_jump_entry(target_block_idx=9)
    precision_entry = _make_precision_entry(full_text="v:DEAD (57005)")
    leaf = AsmLeaf(
        text="multi", openables=(call_entry, jump_entry, precision_entry)
    )

    children = leaf.expand()

    assert children[0].text == inline_call_label(
        call_entry.kind,
        call_entry.counter_id,
        call_entry.callee_name,
        call_entry.provider,
    )
    assert children[1].text == inline_jump_label(jump_entry.target_block_idx)
    # Number-precision rows have no canonical helper: their label IS
    # the pre-rendered full_text carried on the entry (passthrough).
    assert children[2].text == precision_entry.full_text


def test_asmleaf_three_openables_produces_three_wrapper_rows():
    """The 2+-arm rule generalises: N openables -> N wrapper rows
    (one of each Openable subtype to exercise the match-statement
    dispatch across all three branches)."""
    call_entry = _make_inline_call_entry()
    jump_entry = _make_inline_jump_entry()
    precision_entry = _make_precision_entry()
    leaf = AsmLeaf(
        text="multi", openables=(call_entry, jump_entry, precision_entry)
    )

    children = leaf.expand()

    assert len(children) == 3
    # All wrappers carry exactly one openable; verify each wrapper's
    # single-openable identity matches the corresponding source entry.
    assert children[0].openables == (call_entry,)
    assert children[1].openables == (jump_entry,)
    assert children[2].openables == (precision_entry,)
