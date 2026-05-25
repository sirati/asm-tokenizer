"""Unit tests pinning ``tokenizer.arch.x86.operands.tokenize_operand_immediate``'s
sign-handling contract.

The post-refactor invariant is "the v2 valued_const emitter is the sole
owner of sign decomposition". The x86 immediate path previously rebound
``op.imm`` to ``abs(op.imm)`` and handed the unsigned magnitude to
``constant_handler.process_constant_v2``, dropping the sign on the floor;
the v2 emitter then emitted ``[Valued_Const_V2(magnitude)]`` with no
postfix ``value_negative``. The fix routes the signed ``raw_imm`` through
unmodified so the emitter can decompose it into
``[Valued_Const_V2(|v|), Value_Negative()]``.

These tests mirror the ``operands_base`` sign-tests in spirit: a real
``VocabularyManager`` + ``ConstantHandler`` plus duck-typed mocks for the
``InstructionView`` / ``OperandView`` / ``MetadataLookup`` collaborators
(the Protocols at ``tokenizer.disasm.types``/``metadata`` are
``runtime_checkable`` but not enforced at call sites we exercise here).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from tokenizer.arch.x86.operands import tokenize_operand_immediate
from tokenizer.constant_handler.core import ConstantHandler
from tokenizer.disasm.metadata import AddressKind
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, TokenResolver


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def vm() -> VocabularyManager:
    """A v2 VocabularyManager on the x86_64 platform so the eagerly-
    pinned ``value_negative`` postfix marker is reachable via
    ``vm.Value_Negative()``."""
    return VocabularyManager(platform="x86_64", format_version=2)


@pytest.fixture
def handler(vm: VocabularyManager) -> ConstantHandler:
    """A real ``ConstantHandler``; the precedence walk is unaltered by
    this refactor — only the caller's pre-emitter sign-flattening was
    removed — so the full v2 entry point must remain reachable for the
    test to exercise the post-refactor path end-to-end."""
    resolver = TokenResolver()
    constant_dict: dict[str, list[str]] = {}
    block_ranges = np.empty((0, 2), dtype=np.uint64)
    return ConstantHandler(vm, resolver, constant_dict, block_ranges)


class _UnknownLookup:
    """Stub ``MetadataLookup`` reporting ``AddressKind.NONE`` for every
    query. With ``kind == NONE`` every v2 precedence-list address
    predicate (steps 2-10) short-circuits, so ``process_constant_v2``
    lands at step 11 (``_emit_valued_const``) — the path under test for
    sign decomposition."""

    def __init__(self) -> None:
        self._view = SimpleNamespace(kind=AddressKind.NONE, start_addr=None)

    def lookup(self, addr: int):
        return self._view


@pytest.fixture
def lookup() -> _UnknownLookup:
    return _UnknownLookup()


def _ids(tokens) -> List[int]:
    """Flatten a list of ``Tokens`` into the concatenated id stream."""
    out: List[int] = []
    for t in tokens:
        out.extend(int(x) for x in t.get_token_ids().tolist())
    return out


def _mem_minus_id(vm: VocabularyManager) -> int:
    # Resolved only to assert MEM_MINUS does NOT leak into the post-refactor
    # x86 immediate token stream. This file does not pin MEM_MINUS as a
    # sign marker — that role migrated to the postfix ``value_negative``
    # metatoken in the v2 valued_const emitter; MEM_MINUS remains the
    # addressing-operator minus inside ``mem[ ... ]mem`` (and the correct
    # emission for any future non-disasm-sourced minus).
    return int(vm.MemoryOperand(MemoryOperandSymbol.MINUS).get_token_ids().tolist()[0])


def _value_negative_id(vm: VocabularyManager) -> int:
    return int(vm.Value_Negative().get_token_ids().tolist()[0])


def _valued_const_v2_type_id(vm: VocabularyManager) -> int:
    """Type-id at the head of a ``valued_const_v2`` token (the rest are
    inline magnitude-digit bytes < 256)."""
    return int(vm.Valued_Const_V2(0).get_token_ids().tolist()[0])


def _make_imm_op(imm: int, fp_type=None):
    """``OperandView``-shaped duck mock; the x86 immediate-path helper
    only reads ``op.imm`` and ``op.fp_type``."""
    return SimpleNamespace(imm=imm, fp_type=fp_type)


def _make_insn(mnemonic: str):
    """``InstructionView``-shaped duck mock; the x86 immediate-path
    helper only reads ``insn.mnemonic``."""
    return SimpleNamespace(mnemonic=mnemonic)


# --------------------------------------------------------------------------
# Sign-handling contract on the small-immediate (<=2-hex-digit) branch.
# --------------------------------------------------------------------------


def test_negative_small_immediate_emits_value_negative_postfix(handler, vm, lookup):
    """``sub rsp, -0x8`` shape: a negative 1-byte immediate must surface
    a postfix ``value_negative`` metatoken and no leading ``MEM_MINUS``.
    This is the regression case for the pre-refactor
    ``imm_val = abs(op.imm)`` rebind, which dropped the sign before
    calling ``process_constant_v2``."""
    out = tokenize_operand_immediate(
        addressing_control_flow_instructions=set(),
        arithmetic_instructions={"sub"},
        insn=_make_insn("sub"),
        lookup=lookup,
        op=_make_imm_op(-0x8),
        func_max_addr=0x1000,
        func_min_addr=0x0,
        constant_handler=handler,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, (
        f"MEM_MINUS must not appear in the immediate token stream; got {ids!r}"
    )
    assert _value_negative_id(vm) in ids, (
        f"value_negative metatoken missing from negative-immediate stream; got {ids!r}"
    )
    # Structural ordering: magnitude byte 0x8 precedes the postfix marker.
    vneg_at = ids.index(_value_negative_id(vm))
    assert 0x8 in ids[:vneg_at], (
        f"magnitude byte 0x8 must precede value_negative in {ids!r}"
    )


def test_positive_small_immediate_emits_no_value_negative(handler, vm, lookup):
    """Positive 1-byte immediate (``sub rsp, 0x8`` shape): no postfix
    ``value_negative`` and no ``MEM_MINUS`` (mirror case)."""
    out = tokenize_operand_immediate(
        addressing_control_flow_instructions=set(),
        arithmetic_instructions={"sub"},
        insn=_make_insn("sub"),
        lookup=lookup,
        op=_make_imm_op(0x8),
        func_max_addr=0x1000,
        func_min_addr=0x0,
        constant_handler=handler,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, f"MEM_MINUS leaked: {ids!r}"
    assert _value_negative_id(vm) not in ids, (
        f"value_negative leaked into positive immediate: {ids!r}"
    )
    assert 0x8 in ids, f"magnitude byte 0x8 missing from {ids!r}"


# --------------------------------------------------------------------------
# Sign-handling contract on the wider (>2-hex-digit) arithmetic branch.
# --------------------------------------------------------------------------


def test_negative_arithmetic_wide_immediate_emits_value_negative(handler, vm, lookup):
    """Wider negative arithmetic immediate (``sub rsp, -0x1234`` shape):
    falls into the ``imm_val_hex_len <= 32`` arithmetic branch. The sign
    must reach the v2 emitter, producing the postfix ``value_negative``.
    """
    out = tokenize_operand_immediate(
        addressing_control_flow_instructions=set(),
        arithmetic_instructions={"sub"},
        insn=_make_insn("sub"),
        lookup=lookup,
        op=_make_imm_op(-0x1234),
        func_max_addr=0x10000,
        func_min_addr=0x0,
        constant_handler=handler,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, f"MEM_MINUS leaked: {ids!r}"
    assert _value_negative_id(vm) in ids, (
        f"value_negative missing from wide negative arithmetic immediate: {ids!r}"
    )
    # Shape: valued_const_v2 type id precedes value_negative.
    vct_at = ids.index(_valued_const_v2_type_id(vm))
    vneg_at = ids.index(_value_negative_id(vm))
    assert vct_at < vneg_at, (
        f"expected valued_const_v2 < value_negative in {ids!r}"
    )


# Note: the opaque-fallback branch (unknown mnemonic) is intentionally
# NOT covered here. Exercising it would require a richer
# ``AddressMetadataView`` stub (the predicate chain inspects fields like
# ``is_vtable`` that ``_UnknownLookup`` does not populate) and would
# duplicate the rebind-correctness signal already pinned by the small-
# immediate and wide-arithmetic cases above.


# --------------------------------------------------------------------------
# Control-flow path: intra-function jump target → ``block_v2``.
# --------------------------------------------------------------------------


class _LocalFunctionLookup:
    """Stub ``MetadataLookup`` reporting ``AddressKind.LOCAL_FUNCTION`` with
    a fixed ``start_addr`` for every query.

    Models the Ghidra-side metadata stamp for a control-flow target whose
    physical address falls inside the calling function's body: the
    function-range entry is the function's first instruction, and any
    interior address shares the same ``meta`` (since the lookup is
    range-based). The v2 precedence walk distinguishes entry vs interior
    via ``_is_function_entry(meta, value)``.
    """

    def __init__(self, start_addr: int) -> None:
        self._view = SimpleNamespace(
            kind=AddressKind.LOCAL_FUNCTION,
            start_addr=start_addr,
        )

    def lookup(self, addr: int):
        return self._view


def test_intra_function_jump_target_emits_block_v2(handler, vm):
    """x86 control-flow target strictly inside the calling function's body
    must reach the v2 precedence walk's step 4 (``block_v2``), not be
    short-circuited to step 11 (``valued_const_v2``).

    Regression: the legacy ``is_arithmetic=True`` pre-classification at
    this call site (mirrored from the shared ``operands_base`` path)
    contradicted ``precedence.md`` step 4. The fix collapses the
    pre-classification, handing both intra-function and cross-function
    targets to the constant_handler with ``is_arithmetic=False`` so the
    predicate dispatch decides.
    """
    func_start = 0x1000
    func_end = 0x2000
    intra_target = 0x1100  # strictly inside [func_start, func_end), != entry

    out = tokenize_operand_immediate(
        addressing_control_flow_instructions={"jmp"},
        arithmetic_instructions=set(),
        insn=_make_insn("jmp"),
        lookup=_LocalFunctionLookup(start_addr=func_start),
        op=_make_imm_op(intra_target),
        func_max_addr=func_end,
        func_min_addr=func_start,
        constant_handler=handler,
    )

    ids = _ids(out)
    block_v2_type_id = int(vm.Block_V2(0).get_token_ids().tolist()[0])
    valued_const_v2_type_id = int(vm.Valued_Const_V2(0).get_token_ids().tolist()[0])
    assert block_v2_type_id in ids, (
        f"intra-function jump target must emit block_v2 (id {block_v2_type_id}); "
        f"got {ids!r}"
    )
    assert valued_const_v2_type_id not in ids, (
        f"intra-function jump target must NOT collapse to valued_const_v2 "
        f"(id {valued_const_v2_type_id}); got {ids!r}"
    )
