"""Pass-1 matched-walker ``unique_called`` is category-grouped (LOCAL →
PLT → EXTERN) with intra-category encoder-allocation order preserved.

The matched-function walker unions called-function names across
surviving variants of the same function via the shared
``_typed_called_union.category_grouped_first_seen_union`` helper:
first-seen wins on duplicates, intra-category encounter order is
anchored on the FIRST surviving variant that introduced each name, and
the categories are stable-sorted into LOCAL → PLT → EXTERN blocks
(plan Decisions 20 + 21).

These tests construct :class:`ParsedRecord` instances directly so the
per-row ``called_funcs`` order is dictated by the test, NOT by any
helper-side alphabetization (the helper in
``test_passes_registry_wiring.py`` sorts for determinism in that test's
expected-value comparisons; encounter-order tests need the raw shape).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.parsed_record_iter import Matched, ParsedRecord
from tokenizer.memmap_builder._dedup import open_arm_dedup_state
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.passes import process_matched_function


@dataclass(frozen=True)
class _FakeVKey:
    label: str


def _make_record(
    func_name: str,
    called_funcs: "list[tuple[str, CallTargetType]]",
    *,
    token_fill: int,
) -> ParsedRecord:
    """ParsedRecord with caller-specified callee order preserved verbatim."""
    tokens = np.full(4, token_fill, dtype=np.uint16)
    block_runlength = np.array([1, 2], dtype=np.uint8)
    insn_runlength = np.array([3, 4], dtype=np.uint8)
    content_hash = int(tokens.tobytes().__hash__() & 0xFFFFFFFFFFFFFFFF)
    return ParsedRecord(
        func_name=func_name,
        insn_runlength=insn_runlength,
        block_runlength=block_runlength,
        tokens=tokens,
        called_funcs=list(called_funcs),
        extern_libraries={},
        content_hash=content_hash,
    )


def test_unique_called_anchors_on_first_variant_order(tmp_path):
    """Variant 0's callee order is ``[gamma, alpha, beta]`` (NOT
    alphabetical). Variant 1 shares ``alpha`` + ``beta`` in a DIFFERENT
    order and adds a novel ``delta``. The union preserves variant 0's
    order then appends variant 1's novel callee at the tail."""
    LOCAL = CallTargetType.LOCAL
    vkey0 = _FakeVKey("v0")
    vkey1 = _FakeVKey("v1")
    matched = Matched(
        func_name="caller_fn",
        records={
            0: _make_record(
                "caller_fn",
                [("gamma", LOCAL), ("alpha", LOCAL), ("beta", LOCAL)],
                token_fill=1,
            ),
            1: _make_record(
                "caller_fn",
                [("beta", LOCAL), ("alpha", LOCAL), ("delta", LOCAL)],
                token_fill=2,
            ),
        },
    )

    registry = FunctionNamesRegistry()
    state = open_arm_dedup_state(tmp_path / "matched_encounter.bin")
    try:
        entry = process_matched_function(matched, [vkey0, vkey1], state, registry)
    finally:
        state.writer.finalize()

    assert entry is not None
    assert entry["unique_called"] == [
        ("gamma", LOCAL),
        ("alpha", LOCAL),
        ("beta", LOCAL),
        ("delta", LOCAL),
    ]


def test_unique_called_shuffled_input_not_alphabetised(tmp_path):
    """Variant 0's callee list is NOT alphabetically sorted and variant 1
    contributes no new callees (a permutation-subset). The union echoes
    variant 0's order verbatim instead of alphabetising it.

    Two variants are required because the matched walker drops any
    function whose surviving versions all dedup to the same data_offset;
    variant 1's callee list is a permutation-subset of variant 0's so it
    contributes no novel entries and the test still pins the
    first-variant-anchored order property.
    """
    LOCAL = CallTargetType.LOCAL
    vkey0 = _FakeVKey("v0")
    vkey1 = _FakeVKey("v1")
    matched = Matched(
        func_name="shuffled_fn",
        records={
            0: _make_record(
                "shuffled_fn",
                [("zeta", LOCAL), ("alpha", LOCAL), ("mu", LOCAL)],
                token_fill=1,
            ),
            1: _make_record(
                "shuffled_fn",
                [("mu", LOCAL), ("alpha", LOCAL)],
                token_fill=2,
            ),
        },
    )

    registry = FunctionNamesRegistry()
    state = open_arm_dedup_state(tmp_path / "matched_shuffled.bin")
    try:
        entry = process_matched_function(matched, [vkey0, vkey1], state, registry)
    finally:
        state.writer.finalize()

    assert entry is not None
    assert entry["unique_called"] == [
        ("zeta", LOCAL),
        ("alpha", LOCAL),
        ("mu", LOCAL),
    ]


def test_unique_called_cross_category_is_local_plt_extern_grouped(tmp_path):
    """Variant 0's callees mix LOCAL/PLT/EXTERN in a non-grouped order
    (``[(a, LOCAL), (b, PLT), (c, EXTERN), (d, LOCAL)]``); variant 1
    contributes ``[(c, EXTERN), (b, PLT), (e, PLT)]``. The section-level
    union enforces LOCAL → PLT → EXTERN grouping
    (``Section.call_targets[]`` invariant asserted at
    ``loader/_session_helpers.py``); intra-category order preserves
    first-seen encounter order across variants (stable sort).

    Expected:
    - LOCAL block: ``a`` (v0), ``d`` (v0).
    - PLT block: ``b`` (v0), ``e`` (v1, novel).
    - EXTERN block: ``c`` (v0).
    """
    LOCAL = CallTargetType.LOCAL
    PLT = CallTargetType.PLT
    EXTERN = CallTargetType.EXTERN
    vkey0 = _FakeVKey("v0")
    vkey1 = _FakeVKey("v1")
    matched = Matched(
        func_name="mixed_fn",
        records={
            0: _make_record(
                "mixed_fn",
                [
                    ("a", LOCAL),
                    ("b", PLT),
                    ("c", EXTERN),
                    ("d", LOCAL),
                ],
                token_fill=1,
            ),
            1: _make_record(
                "mixed_fn",
                [("c", EXTERN), ("b", PLT), ("e", PLT)],
                token_fill=2,
            ),
        },
    )

    registry = FunctionNamesRegistry()
    state = open_arm_dedup_state(tmp_path / "matched_cross_category.bin")
    try:
        entry = process_matched_function(matched, [vkey0, vkey1], state, registry)
    finally:
        state.writer.finalize()

    assert entry is not None
    assert entry["unique_called"] == [
        ("a", LOCAL),
        ("d", LOCAL),
        ("b", PLT),
        ("e", PLT),
        ("c", EXTERN),
    ]
