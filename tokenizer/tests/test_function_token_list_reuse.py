"""Grow-only buffer reuse regressions for ``FunctionTokenList``.

The phase-1 producer (``fill_constant_candidates``) reuses ONE
``FunctionTokenList`` across all functions of a binary: ``reset()`` on
entry, in-place ``view()`` blocks, arrays growing only when a larger
function than any seen before arrives. These tests pin the invariants
that make that safe:

1. Encoding a function through a REUSED buffer (after arbitrarily many
   reset/fill cycles, including smaller-then-larger functions) is
   identical to encoding it through a FRESH buffer — i.e. ``reset()``
   leaks no stale counters, lookup offsets, or token data.
2. ``reset()`` never shrinks the arrays (grow-only).
3. ``reset()`` releases an abandoned ``view()`` child (mid-function
   exception path).
4. ``snapshot()`` is independent of subsequent reset/fill cycles.
5. ``add_block`` releases an EMPTY view child's buffer lease.
"""

from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.main_loop import build_vocab_tokenize_and_index
from tokenizer.token_manager import VocabularyManager


def _make_vm() -> VocabularyManager:
    return VocabularyManager(platform="x86_64", format_version=2)


def _fill(ftl: FunctionTokenList, vm: VocabularyManager, func_spec: list[int]) -> None:
    """Producer-pattern fill: one ``view()`` child per block committed via
    ``add_block``, one ``append_as_insn`` per instruction — the exact
    write path ``fill_constant_candidates`` uses. ``func_spec`` gives the
    per-block instruction count; token ids vary per block/insn so stale
    data from a previous fill cannot accidentally match.
    """
    for block_i, insn_count in enumerate(func_spec):
        block = ftl.view()
        block.append_as_insn(
            insn_str=f"block {block_i}",
            tokens=[vm.Block_Def(), vm.BlockId(block_i)],
        )
        for insn_i in range(insn_count):
            block.append_as_insn(
                insn_str=f"insn {block_i}:{insn_i}",
                tokens=[vm.Block_Def(), vm.BlockId(block_i * 1000 + insn_i + 1)],
            )
        ftl.add_block(block, hex(0x1000 * (block_i + 1)))


def _encode(ftl: FunctionTokenList) -> tuple[str, str, str]:
    """The production encode: main_loop's array derivation + base64."""
    tokens, block_rl, insn_rl = build_vocab_tokenize_and_index(ftl)
    return (
        ndarray_to_base64(tokens),
        ndarray_to_base64(block_rl),
        ndarray_to_base64(insn_rl),
    )


def _fresh_encode(vm: VocabularyManager, func_spec: list[int]) -> tuple[str, str, str]:
    fresh = FunctionTokenList(num_blocks=len(func_spec), vocab_manager=vm)
    _fill(fresh, vm, func_spec)
    return _encode(fresh)


# Three synthetic functions: A large (forces growth from a tiny seed,
# including the chained view-child insn-axis growth), B strictly smaller
# (stale-state canary), C larger than A (cross-function grow event).
_FUNC_A = [3, 100, 5]
_FUNC_B = [1, 2]
_FUNC_C = [10, 150, 8, 4]


def test_reuse_across_functions_matches_fresh_encode():
    vm = _make_vm()
    reused = FunctionTokenList(num_blocks=1, vocab_manager=vm)

    for spec in (_FUNC_A, _FUNC_B, _FUNC_C):
        reused.reset()
        _fill(reused, vm, spec)
        assert _encode(reused) == _fresh_encode(vm, spec), f"reused-buffer encode diverged for spec {spec}"


def test_reset_is_grow_only():
    vm = _make_vm()
    reused = FunctionTokenList(num_blocks=1, vocab_manager=vm)

    sizes: list[tuple[int, ...]] = []
    for spec in (_FUNC_A, _FUNC_B, _FUNC_C):
        reused.reset()
        _fill(reused, vm, spec)
        sizes.append(
            (
                len(reused.token_ids),
                len(reused.metatoken_type_ids),
                len(reused.insn_metatoken_run_lengths),
                len(reused.block_insn_run_lengths),
            )
        )
    for prev, cur in zip(sizes, sizes[1:]):
        assert all(c >= p for p, c in zip(prev, cur)), f"buffer shrank across reset: {prev} -> {cur}"


def test_reset_clears_abandoned_view_child():
    vm = _make_vm()
    ftl = FunctionTokenList(num_blocks=2, vocab_manager=vm)
    ftl.view()  # abandoned, e.g. by a mid-function exception
    ftl.reset()
    ftl.view()  # must not raise "already has an active view child"


def test_add_block_releases_empty_view_child():
    vm = _make_vm()
    ftl = FunctionTokenList(num_blocks=2, vocab_manager=vm)
    empty = ftl.view()
    ftl.add_block(empty, "0x0")  # nothing written: early-return path
    ftl.view()  # lease must have been released


def test_snapshot_survives_reset_and_refill():
    vm = _make_vm()
    reused = FunctionTokenList(num_blocks=1, vocab_manager=vm)

    _fill(reused, vm, _FUNC_A)
    expected_a = _encode(reused)
    snap = reused.snapshot()
    assert _encode(snap) == expected_a

    reused.reset()
    _fill(reused, vm, _FUNC_B)
    assert _encode(snap) == expected_a, "snapshot must be independent of the buffer's next fill"
