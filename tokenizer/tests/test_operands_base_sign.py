"""Unit tests pinning the arch-base operand tokenizer's sign-handling contract.

After the sign-ownership refactor, ``tokenize_operand_immediate_generic``
and ``tokenize_operand_memory_base_disp`` (``tokenizer/arch/operands_base.py``)
no longer pre-flatten signed values to ``abs(...)`` nor pre-emit a leading
``MEM_MINUS`` token. They pass the signed integer straight to
``constant_handler.process_constant_v2``, which (via ``_emit_valued_const``)
is the sole owner of sign decomposition (postfix ``value_negative``
metatoken on the ``valued_const_v2`` token).

These tests construct a minimal-surface fixture: a real
``VocabularyManager`` + ``TokenResolver`` + ``ConstantHandler`` (the
constant handler's precedence walk is unaltered by this refactor; only
the input value's sign is now preserved by the caller). The
``InstructionView`` / ``OperandView`` / ``MetadataLookup`` collaborators
are duck-typed mocks because the Protocols at
``tokenizer.disasm.types``/``metadata`` are ``runtime_checkable`` but
not actually type-checked at call time — only the attributes/methods
that the operands_base helpers read are populated.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from tokenizer.arch.operands_base import (
    tokenize_operand_immediate_generic,
    tokenize_operand_memory_base_disp,
)
from tokenizer.constant_handler.core import ConstantHandler
from tokenizer.disasm.metadata import AddressKind
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, TokenResolver


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def vm() -> VocabularyManager:
    """A v2 VocabularyManager with a real platform string so that
    ``get_registry_token`` (used by the memory-disp path) produces
    valid platform-prefixed register tokens."""
    return VocabularyManager(platform="riscv64", format_version=2)


@pytest.fixture
def handler(vm: VocabularyManager) -> ConstantHandler:
    """A real ``ConstantHandler``. The precedence walk is unaltered by
    this refactor; only the caller's pre-emitter sign-flattening is
    being removed, so the full v2 entry point must remain reachable for
    the test to exercise the post-refactor path end-to-end."""
    resolver = TokenResolver()
    block_ranges = np.empty((0, 2), dtype=np.uint64)
    return ConstantHandler(vm, resolver, block_ranges)


class _UnknownLookup:
    """Stub ``MetadataLookup`` that always reports ``AddressKind.NONE``.

    With ``kind == NONE`` every v2 precedence-list address predicate
    (steps 2-10) short-circuits, so ``process_constant_v2`` lands at
    step 11 (``_emit_valued_const``) — the path under test for sign
    decomposition. The view stays minimal: only ``kind`` and
    ``start_addr`` are consulted by the immediate-path internal-jump
    heuristic.
    """

    def __init__(self) -> None:
        self._view = SimpleNamespace(kind=AddressKind.NONE, start_addr=None)

    def lookup(self, addr: int):  # signed Python int by the new contract
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
    # immediate / disp token stream. This file does not pin MEM_MINUS as a
    # sign marker — that role migrated to the postfix ``value_negative``
    # metatoken in the v2 valued_const emitter; MEM_MINUS remains the
    # addressing-operator minus inside ``mem[ ... ]mem`` (and the correct
    # emission for any future non-disasm-sourced minus).
    return int(vm.MemoryOperand(MemoryOperandSymbol.MINUS).get_token_ids().tolist()[0])


def _mem_plus_id(vm: VocabularyManager) -> int:
    return int(vm.MemoryOperand(MemoryOperandSymbol.PLUS).get_token_ids().tolist()[0])


def _value_negative_id(vm: VocabularyManager) -> int:
    return int(vm.Value_Negative().get_token_ids().tolist()[0])


def _valued_const_v2_type_id(vm: VocabularyManager) -> int:
    """The single type-id at the head of a ``valued_const_v2`` token
    (the rest are inline-digit magnitude bytes < 256)."""
    return int(vm.Valued_Const_V2(0).get_token_ids().tolist()[0])


# --------------------------------------------------------------------------
# Mock builders (duck-typed views — Protocols at disasm.types are
# runtime_checkable but not enforced at call sites we exercise here).
# --------------------------------------------------------------------------


def _make_imm_op(imm: int, fp_type=None):
    """An ``OperandView``-shaped duck mock carrying a plain integer
    immediate. The immediate-path helper only reads ``op.imm`` +
    ``op.fp_type``."""
    return SimpleNamespace(imm=imm, fp_type=fp_type)


def _make_mem_op(
    *,
    base_name: str,
    base_id: int,
    disp: int,
    resolved_target=None,
    fp_type=None,
):
    """An ``OperandView``-shaped duck mock with a base+disp memory
    sub-view. Only the fields the disp-path helper reads are populated:
    ``op.mem.{base.is_absent, base.name, base.id, disp, resolved_target}``
    and ``op.fp_type``."""
    base = SimpleNamespace(is_absent=False, name=base_name, id=base_id)
    mem = SimpleNamespace(base=base, disp=disp, resolved_target=resolved_target)
    return SimpleNamespace(mem=mem, fp_type=fp_type)


def _make_insn(mnemonic: str = "c.addi"):
    """An ``InstructionView``-shaped duck mock; the immediate-path
    helper only reads ``insn.mnemonic`` (in the >2-hex-digits branch)."""
    return SimpleNamespace(mnemonic=mnemonic)


# --------------------------------------------------------------------------
# Immediate path: no MEM_MINUS prefix; postfix value_negative on negatives.
# --------------------------------------------------------------------------


def test_negative_immediate_does_not_emit_mem_minus_prefix(handler, vm, lookup):
    """``c.addi sp, -0x10`` shape: a negative 1-byte immediate must
    not surface a leading ``MEM_MINUS`` token. The sign migrates to the
    postfix ``value_negative`` metatoken emitted by the v2 valued_const
    emitter.
    """
    out = tokenize_operand_immediate_generic(
        addressing_control_flow_instructions=set(),
        arithmetic_instructions={"c.addi"},
        insn=_make_insn("c.addi"),
        lookup=lookup,
        op=_make_imm_op(-0x10),
        func_max_addr=0x1000,
        func_min_addr=0x0,
        constant_handler=handler,
        vocab_manager=vm,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, (
        f"MEM_MINUS must not appear in the immediate token stream; got {ids!r}"
    )
    # The postfix ``value_negative`` metatoken (id 256) MUST be present.
    assert _value_negative_id(vm) in ids, (
        f"value_negative metatoken (id 256) missing from {ids!r}"
    )
    # Structural ordering: the magnitude bytes (0x10 = 16) precede the
    # postfix ``value_negative``.
    vneg_at = ids.index(_value_negative_id(vm))
    assert 0x10 in ids[:vneg_at], (
        f"magnitude byte 0x10 must precede value_negative in {ids!r}"
    )


def test_positive_immediate_emits_no_value_negative_and_no_mem_minus(handler, vm, lookup):
    """Positive 1-byte immediate (``c.addi sp, +0x10`` shape): neither
    ``MEM_MINUS`` nor ``value_negative`` may appear."""
    out = tokenize_operand_immediate_generic(
        addressing_control_flow_instructions=set(),
        arithmetic_instructions={"c.addi"},
        insn=_make_insn("c.addi"),
        lookup=lookup,
        op=_make_imm_op(0x10),
        func_max_addr=0x1000,
        func_min_addr=0x0,
        constant_handler=handler,
        vocab_manager=vm,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, f"MEM_MINUS leaked into positive immediate: {ids!r}"
    assert _value_negative_id(vm) not in ids, (
        f"value_negative leaked into positive immediate: {ids!r}"
    )
    # The magnitude byte 0x10 is still present.
    assert 0x10 in ids, f"magnitude byte 0x10 missing from {ids!r}"


# --------------------------------------------------------------------------
# Memory-disp path: bracket separator always MEM_PLUS; sign as postfix.
# --------------------------------------------------------------------------


def test_negative_memory_disp_emits_mem_plus_and_value_negative(handler, vm, lookup):
    """``ld r1, [sp + -0x10]`` shape: the bracket separator must be
    ``MEM_PLUS`` regardless of disp sign; the sign migrates to the
    postfix ``value_negative`` metatoken on the inner
    ``valued_const_v2``."""
    out = tokenize_operand_memory_base_disp(
        insn=_make_insn(),
        lookup=lookup,
        op=_make_mem_op(base_name="sp", base_id=0, disp=-0x10),
        text_end=0x10000,
        text_start=0x100,
        func_max_addr=0x2000,
        func_min_addr=0x1000,
        vocab_manager=vm,
        constant_handler=handler,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, (
        f"MEM_MINUS leaked into memory disp token stream; got {ids!r}"
    )
    assert _mem_plus_id(vm) in ids, (
        f"MEM_PLUS must always be the bracket separator; got {ids!r}"
    )
    assert _value_negative_id(vm) in ids, (
        f"value_negative postfix missing from negative disp; got {ids!r}"
    )

    # Structural shape inside the brackets: MEM_PLUS precedes
    # valued_const_v2, magnitude bytes precede value_negative.
    plus_at = ids.index(_mem_plus_id(vm))
    vneg_at = ids.index(_value_negative_id(vm))
    vct_at = ids.index(_valued_const_v2_type_id(vm))
    assert plus_at < vct_at < vneg_at, (
        f"expected MEM_PLUS < valued_const_v2 < value_negative in {ids!r}"
    )


def test_positive_memory_disp_emits_mem_plus_and_no_value_negative(handler, vm, lookup):
    """``ld r1, [sp + 0x10]`` shape: positive small disp emits the
    ``MEM_PLUS`` separator and no postfix ``value_negative``. Mirror
    case to the negative test above to catch any accidental
    sign-loss on the positive branch."""
    out = tokenize_operand_memory_base_disp(
        insn=_make_insn(),
        lookup=lookup,
        op=_make_mem_op(base_name="sp", base_id=0, disp=0x10),
        text_end=0x10000,
        text_start=0x100,
        func_max_addr=0x2000,
        func_min_addr=0x1000,
        vocab_manager=vm,
        constant_handler=handler,
    )

    ids = _ids(out)
    assert _mem_minus_id(vm) not in ids, f"MEM_MINUS leaked: {ids!r}"
    assert _mem_plus_id(vm) in ids, f"MEM_PLUS missing: {ids!r}"
    assert _value_negative_id(vm) not in ids, (
        f"value_negative leaked into positive disp: {ids!r}"
    )


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
    range-based). Consumers receive the same view either way; the v2
    precedence walk distinguishes entry vs interior via
    ``_is_function_entry(meta, value)``.
    """

    def __init__(self, start_addr: int) -> None:
        self._view = SimpleNamespace(
            kind=AddressKind.LOCAL_FUNCTION,
            start_addr=start_addr,
        )

    def lookup(self, addr: int):
        return self._view


def test_intra_function_jump_target_emits_block_v2(handler, vm):
    """Control-flow target strictly inside the calling function's body
    must reach the v2 precedence walk's step 4 (``block_v2``), not be
    short-circuited to step 11 (``valued_const_v2``).

    Regression: the legacy ``is_arithmetic=True`` pre-classification at
    this call site contradicted ``precedence.md`` step 4 ("Address falls
    strictly inside a function body -> block"). The fix collapses the
    pre-classification, handing both intra-function and cross-function
    targets to the constant_handler with ``is_arithmetic=False`` so the
    predicate dispatch decides.
    """
    func_start = 0x1000
    func_end = 0x2000
    intra_target = 0x1100  # strictly inside [func_start, func_end), != entry

    out = tokenize_operand_immediate_generic(
        addressing_control_flow_instructions={"j"},
        arithmetic_instructions=set(),
        insn=_make_insn("j"),
        lookup=_LocalFunctionLookup(start_addr=func_start),
        op=_make_imm_op(intra_target),
        func_max_addr=func_end,
        func_min_addr=func_start,
        constant_handler=handler,
        vocab_manager=vm,
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
