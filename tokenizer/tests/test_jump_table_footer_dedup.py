"""Tests for canonical-yield coalescing in ``_emit_jump_table_footer``.

Concern: ``iter_switch_tables`` may yield the same ``base_addr`` more
than once when a function has multiple computed-jump dispatch
instructions reading the same switch table (Ghidra's ref-graph
back-walk recovers the table from each dispatch site independently,
possibly with a different resolved-target subset at each site). The
JUMP_TABLE identity cache returns the same ``jt_id`` on every
``get_identity`` call, so a naive per-yield emit produces N separate
``block_def jump_table <jt_id> ...`` blocks for the SAME identity
(visible in the inspector as two collapsible "Jump table: 0" entries
with different target lists).

The fix coalesces yields per base_addr with a target-set UNION before
emitting one footer. These tests pin that invariant with a mocked
``iter_switch_tables`` (no JVM / no real disassembler required).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from tokenizer import fill_constant_candidates as fcc
from tokenizer.fill_constant_candidates import _emit_jump_table_footer
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category, TokenResolver


# ---------------------------------------------------------------------------
# Mock disassembler — drives only ``iter_switch_tables``.
#
# ``_emit_jump_table_footer`` reads NOTHING else off the provider on the
# canonical path; the slot-fallback path reads ``resolver.id_maps`` (not
# the provider). Keeping the mock surface this narrow ensures the test
# stays focused on the coalescing concern.
# ---------------------------------------------------------------------------


class _MockDisasmProvider:
    """Yields a scripted sequence of ``(table_addr, target_addrs)``.

    The footer pass treats ``func`` as opaque (passed straight to the
    provider) so the mock ignores it and returns the scripted yields
    regardless. Each test constructs a fresh instance.
    """

    def __init__(self, yields: list[tuple[int, list[int]]]) -> None:
        self._yields = yields

    def iter_switch_tables(self, _func) -> Iterable[tuple[int, list[int]]]:
        return iter(self._yields)


# A sentinel stand-in for ``FunctionView``. ``_emit_jump_table_footer``
# only forwards it to the provider's ``iter_switch_tables`` (which the
# mock ignores), so anything truthy works.
_FUNC_SENTINEL = object()


def _fresh_state() -> tuple[VocabularyManager, TokenResolver, FunctionTokenList]:
    """Build the minimum state ``_emit_jump_table_footer`` writes into.

    A v2 vocab is mandatory: the function early-returns on v1. The
    function-token-list is sized for a single block by default; the
    helper extends it as footers are appended.
    """
    vm = VocabularyManager(platform="x86_64", format_version=2)
    resolver = TokenResolver()
    func_tokens = FunctionTokenList(num_blocks=4, vocab_manager=vm)
    return vm, resolver, func_tokens


def _emitted_jump_table_footers(
    func_tokens: FunctionTokenList,
) -> list[tuple[int, list[int]]]:
    """Walk the emitted blocks and return the ``(jt_id, [block_ids])``
    of every jump-table footer block.

    A jump-table footer block is identified by its first token being a
    ``Block_Def`` followed by a ``Jump_Table(id)`` token; the remaining
    tokens in that synthetic instruction are the target ``Block_V2``
    tokens. This is the same structural shape the footer emitter
    produces — using it (vs. matching the human label) keeps the test
    decoupled from the debug ``insn_str``.
    """
    from tokenizer.tokens import BlockDefToken, JumpTableToken, BlockTokenV2

    footers: list[tuple[int, list[int]]] = []
    for block in func_tokens.iter_blocks(transient=True):
        tokens = list(block.iter_tokens())
        if len(tokens) < 2:
            continue
        if not isinstance(tokens[0], BlockDefToken):
            continue
        if not isinstance(tokens[1], JumpTableToken):
            continue
        jt_id = tokens[1].id
        target_ids = [t.id for t in tokens[2:] if isinstance(t, BlockTokenV2)]
        footers.append((jt_id, target_ids))
    return footers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_same_base_identical_targets_emits_one_footer():
    """Two yields of ``(0x1000, [0x100])`` -> ONE footer, one target.

    Pins the simplest coalescing case: identical target lists across
    duplicate yields collapse to a single emission. Without the fix
    this would produce TWO ``block_def jump_table 0 ...`` blocks with
    the same ``jt_id`` (cache hit on the second ``get_identity``).
    """
    vm, resolver, func_tokens = _fresh_state()
    provider = _MockDisasmProvider(
        [
            (0x1000, [0x100]),
            (0x1000, [0x100]),
        ]
    )

    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )

    footers = _emitted_jump_table_footers(func_tokens)
    assert len(footers) == 1, f"expected ONE footer for one base_addr; got {footers!r}"
    jt_id, target_ids = footers[0]
    assert jt_id == 0, f"single coalesced base_addr -> jt_id 0; got {jt_id}"
    # One unique target -> one Block_V2 token; identity allocator
    # assigns it id 0 (first BLOCK allocation in this function).
    assert len(target_ids) == 1, f"expected one target token; got {target_ids!r}"


def test_same_base_overlapping_targets_unions_in_first_seen_order():
    """``(0x1000, [0x100, 0x200])`` then ``(0x1000, [0x200, 0x300])`` ->
    ONE footer with targets ``[0x100, 0x200, 0x300]`` (first-seen
    order; 0x200 dedup'd against the first yield, 0x300 appended).

    Pins the union semantics specifically (the fix is not a silent
    skip-on-seen — the second yield can legitimately carry a target
    the first did not, and the test would fail if the implementation
    just kept the first yield's list).
    """
    vm, resolver, func_tokens = _fresh_state()
    provider = _MockDisasmProvider(
        [
            (0x1000, [0x100, 0x200]),
            (0x1000, [0x200, 0x300]),
        ]
    )

    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )

    footers = _emitted_jump_table_footers(func_tokens)
    assert len(footers) == 1, f"expected ONE footer for one base_addr; got {footers!r}"
    _, target_ids = footers[0]
    assert len(target_ids) == 3, (
        f"target union should have 3 distinct entries (0x100, 0x200, 0x300); "
        f"got {len(target_ids)} ids ({target_ids!r})"
    )

    # The JUMP_TABLE metadata recorded by ``get_identity`` must reflect
    # the UNION, in first-seen order. This is the load-bearing check
    # that the implementation moved the get_identity call to AFTER
    # accumulation (else only the first-yield meta would be recorded).
    jt_meta_list = resolver.metadata[Category.JUMP_TABLE]
    assert len(jt_meta_list) == 1, f"one identity allocated; got {jt_meta_list!r}"
    assert jt_meta_list[0]["target_block_addrs"] == [
        hex(0x100),
        hex(0x200),
        hex(0x300),
    ], (
        f"meta target_block_addrs must reflect first-seen union order; "
        f"got {jt_meta_list[0]['target_block_addrs']!r}"
    )


def test_different_bases_emit_separate_footers_in_yield_order_A():
    """``(0x1000, [0x100])`` then ``(0x2000, [0x200])`` -> TWO footers,
    one per base_addr. Regression guard: the coalescer must not over-
    merge across distinct base_addrs.
    """
    vm, resolver, func_tokens = _fresh_state()
    provider = _MockDisasmProvider(
        [
            (0x1000, [0x100]),
            (0x2000, [0x200]),
        ]
    )
    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )
    footers = _emitted_jump_table_footers(func_tokens)
    assert len(footers) == 2, f"expected TWO footers for two base_addrs; got {footers!r}"
    # First-seen base_addr -> jt_id 0; second -> jt_id 1.
    assert [jt_id for jt_id, _ in footers] == [0, 1], (
        f"expected jt_ids in yield order [0, 1]; got {[jt_id for jt_id, _ in footers]!r}"
    )


def test_different_bases_emit_separate_footers_in_yield_order_B():
    """Reverse-order pair: ``(0x2000, [0x200])`` then ``(0x1000,
    [0x100])`` still yields two footers, with jt_ids matching yield
    order (jt_id 0 for 0x2000, jt_id 1 for 0x1000). Order-independence
    of the coalescer-vs-different-bases path.
    """
    vm, resolver, func_tokens = _fresh_state()
    provider = _MockDisasmProvider(
        [
            (0x2000, [0x200]),
            (0x1000, [0x100]),
        ]
    )
    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )
    footers = _emitted_jump_table_footers(func_tokens)
    assert len(footers) == 2, f"expected TWO footers; got {footers!r}"
    assert [jt_id for jt_id, _ in footers] == [0, 1]


def test_emit_helper_called_exactly_once_with_unioned_targets(monkeypatch):
    """Direct mutation guard on ``_emit_jump_table_footer_for``: with
    two same-base yields carrying overlapping target subsets, the
    helper is invoked EXACTLY ONCE — and the ``target_addrs`` it
    receives is the unioned list in first-seen order.

    This is the load-bearing structural invariant: the bug surfaced as
    two helper invocations producing two synthetic blocks. Counting
    helper-calls (not just output blocks) makes the regression
    detection direct.
    """
    vm, resolver, func_tokens = _fresh_state()
    provider = _MockDisasmProvider(
        [
            (0x1000, [0x100, 0x200]),
            (0x1000, [0x200, 0x300]),
        ]
    )

    calls: list[dict] = []
    real_helper = fcc._emit_jump_table_footer_for

    def spy(**kwargs):
        # Capture the kwargs the helper receives, then forward to the
        # real implementation so the func_tokens stream stays valid for
        # any downstream assertion. The list-copy on target_addrs
        # snapshots the value at call time (the helper mutates nothing
        # but defensive copying keeps the test robust to future edits).
        calls.append({**kwargs, "target_addrs": list(kwargs["target_addrs"])})
        return real_helper(**kwargs)

    monkeypatch.setattr(fcc, "_emit_jump_table_footer_for", spy)

    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )

    assert len(calls) == 1, (
        f"expected _emit_jump_table_footer_for to be called ONCE per coalesced "
        f"base_addr; got {len(calls)} calls (target_addrs at each call: "
        f"{[c['target_addrs'] for c in calls]!r})"
    )
    assert calls[0]["base_addr"] == 0x1000
    assert calls[0]["target_addrs"] == [0x100, 0x200, 0x300], (
        f"helper must receive UNIONED target list in first-seen order; "
        f"got {calls[0]['target_addrs']!r}"
    )


def test_empty_target_yield_does_not_seed_canonical_entry():
    """A yield of ``(0x1000, [])`` is the "table-with-no-resolved-
    slots" case the canonical path intentionally skips (the slot-
    fallback emits a target-less declaration if the table was reached
    via slot classification, but the canonical path stays silent).

    If a later yield carries a target for the same base, the
    accumulator must record that target — but the empty-yield must
    NOT count as having "seen" the base for coalescing purposes
    (otherwise a same-base later yield with real targets would still
    succeed; the regression we're guarding against here is that an
    empty-first yield leaves an empty list in the accumulator and the
    union becomes ``[real_targets]`` -- which is actually the
    behaviour we want, so the test pins that exact behaviour).
    """
    vm, resolver, func_tokens = _fresh_state()
    provider = _MockDisasmProvider(
        [
            (0x1000, []),  # skipped
            (0x1000, [0x100]),  # accumulates
        ]
    )
    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )
    footers = _emitted_jump_table_footers(func_tokens)
    assert len(footers) == 1, f"expected ONE footer; got {footers!r}"
    _, target_ids = footers[0]
    assert len(target_ids) == 1, f"expected one target; got {target_ids!r}"


def test_v1_vocab_is_noop():
    """``format_version=1`` short-circuits the entire helper — no
    block is appended to ``func_tokens`` regardless of what the
    provider yields. Pins the existing v1 no-op invariant so the
    coalescing refactor cannot accidentally start emitting on v1.
    """
    vm = VocabularyManager(platform="x86_64", format_version=1)
    resolver = TokenResolver()
    func_tokens = FunctionTokenList(num_blocks=2, vocab_manager=vm)
    provider = _MockDisasmProvider([(0x1000, [0x100]), (0x1000, [0x200])])

    _emit_jump_table_footer(
        func=_FUNC_SENTINEL,
        func_tokens=func_tokens,
        disasm_provider=provider,
        resolver=resolver,
        vocab_manager=vm,
    )

    assert func_tokens.block_count == 0, (
        f"v1 vocab must short-circuit; got block_count={func_tokens.block_count}"
    )
