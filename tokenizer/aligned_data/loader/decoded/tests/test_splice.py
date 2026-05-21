"""Tests for ``decoded.splice.splice_with_callees`` + its helpers.

Synthetic-only: no ``BinarySession``. ``decode_callee`` is a closure over
a tiny ``{offset: (DecodedFunction, _StubSection)}`` table; ``_StubSection``
+ ``_StubCallTarget`` expose just the duck-type attributes the walker
reads.

Coverage targets (per spawn prompt §TESTS):

* depth=0 returns root unchanged
* depth=1 single callee, single category — exact rebase arithmetic
* two callees, per-category running max accumulates
* sentinel-sticky rebase
* overflow clips to sentinel
* self-recursion cycle (A -> A)
* mutual recursion cycle (A -> B -> A)
* missing callee (``is_callee_present`` False)
* empty ``call_targets``
* all 8 categories rebased independently
* concat ordering of real_tokens / identities / numbers
* func_name + metadata propagate from root only
* DAG semantics — same callee called twice via two CTs splices twice
* multi-level chain (A -> B -> C -> D) at various depths
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.decoded_function import (
    DecodedFunction,
)
from tokenizer.aligned_data.loader.decoded.splice import (
    IDENTITY_SENTINEL,
    _concat_decoded,
    _max_non_sentinel,
    _rebase_identity_array,
    splice_with_callees,
)
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Test stubs / builders
# ---------------------------------------------------------------------------


def _make_decoded(
    *,
    func_name: str = "root",
    real_tokens=(),
    identities: Dict[Category, list] | None = None,
    numbers: Tuple[int, ...] = (),
    sign_exps: Tuple[int, ...] = (),
    metadata: Dict | None = None,
) -> DecodedFunction:
    """Build a ``DecodedFunction`` with sane defaults.

    ``identities`` accepts sparse ``{Category: list[int]}``; absent
    categories default to length-0 arrays so the dataclass invariant
    holds. ``numbers`` + ``sign_exps`` must be the same length.
    """
    if identities is None:
        identities = {}
    if metadata is None:
        metadata = {}
    full_identities = {
        c: np.array(identities.get(c, []), dtype=np.uint16)
        for c in Category
    }
    return DecodedFunction(
        real_tokens=np.array(real_tokens, dtype=np.uint16),
        identities=full_identities,
        numbers_significant=np.array(numbers, dtype=np.uint64),
        numbers_sign_exponent=np.array(sign_exps, dtype=np.uint32),
        func_name=func_name,
        metadata=metadata,
    )


class _StubCallTarget:
    """Duck-type for ``aligned_data.matched_sections_bin.CallTarget``.

    Only ``function_section_ptr`` is read by the walker; ``is_matched``
    is kept for parity with the real type even though the splicer routes
    presence/absence through the ``is_callee_present`` callback.
    """

    def __init__(self, function_section_ptr: int, is_matched: bool = True):
        self.function_section_ptr = function_section_ptr
        self.is_matched = is_matched


class _StubSection:
    """Duck-type for ``aligned_data.matched_sections_bin.Section``.

    Only ``call_targets`` is read.
    """

    def __init__(self, call_targets: List[_StubCallTarget]):
        self.call_targets = call_targets


def _make_table(
    *entries: Tuple[int, DecodedFunction, List[_StubCallTarget]],
) -> Tuple[
    callable, callable, Dict[int, Tuple[DecodedFunction, _StubSection]]
]:
    """Build a ``(decode_callee, is_callee_present, table)`` triple.

    ``entries`` is ``(section_offset, decoded, call_targets_list)``. The
    returned table maps section_offset -> (decoded, _StubSection). The
    arm is ignored by both closures — single-arm tests; mutual-recursion
    tests use the arm to switch tables externally.
    """
    table: Dict[int, Tuple[DecodedFunction, _StubSection]] = {
        offset: (decoded, _StubSection(call_targets))
        for offset, decoded, call_targets in entries
    }

    def decode_callee(offset: int, arm: str):
        return table[offset]

    def is_callee_present(offset: int, arm: str) -> bool:
        return offset in table

    return decode_callee, is_callee_present, table


# ---------------------------------------------------------------------------
# Identity-array helpers
# ---------------------------------------------------------------------------


class TestMaxNonSentinel:
    def test_empty(self):
        assert _max_non_sentinel(np.array([], dtype=np.uint16)) == -1

    def test_all_sentinel(self):
        arr = np.array([0xFFFF, 0xFFFF, 0xFFFF], dtype=np.uint16)
        assert _max_non_sentinel(arr) == -1

    def test_mixed_excludes_sentinel(self):
        arr = np.array([3, 0xFFFF, 7, 0xFFFF, 5], dtype=np.uint16)
        assert _max_non_sentinel(arr) == 7

    def test_zero_baseline(self):
        # Single non-sentinel == 0 → max is 0 (NOT -1), so offset logic
        # uses 0 + 1 = 1 next.
        arr = np.array([0, 0xFFFF], dtype=np.uint16)
        assert _max_non_sentinel(arr) == 0


class TestRebaseIdentityArray:
    def test_empty_preserves_dtype(self):
        out = _rebase_identity_array(np.array([], dtype=np.uint16), 5)
        assert out.dtype == np.uint16
        assert out.size == 0

    def test_simple_add(self):
        out = _rebase_identity_array(np.array([0, 1, 2], dtype=np.uint16), 3)
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, [3, 4, 5])

    def test_sentinel_sticks(self):
        out = _rebase_identity_array(
            np.array([0xFFFF, 5, 0xFFFF], dtype=np.uint16), 10
        )
        np.testing.assert_array_equal(out, [0xFFFF, 15, 0xFFFF])

    def test_overflow_clips_to_sentinel(self):
        # 0xFFFE + 5 > 0xFFFE → 0xFFFF
        out = _rebase_identity_array(
            np.array([0xFFFE], dtype=np.uint16), 5
        )
        np.testing.assert_array_equal(out, [0xFFFF])

    def test_overflow_boundary(self):
        # 0xFFFD + 1 == 0xFFFE → still legal (not sentinel).
        # 0xFFFD + 2 == 0xFFFF → must clip.
        out_legal = _rebase_identity_array(
            np.array([0xFFFD], dtype=np.uint16), 1
        )
        np.testing.assert_array_equal(out_legal, [0xFFFE])
        out_clip = _rebase_identity_array(
            np.array([0xFFFD], dtype=np.uint16), 2
        )
        np.testing.assert_array_equal(out_clip, [0xFFFF])

    def test_zero_offset_is_identity(self):
        arr = np.array([0, 5, 0xFFFE, 0xFFFF], dtype=np.uint16)
        out = _rebase_identity_array(arr, 0)
        np.testing.assert_array_equal(out, arr)

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            _rebase_identity_array(np.array([0], dtype=np.uint16), -1)


# ---------------------------------------------------------------------------
# Concat helper
# ---------------------------------------------------------------------------


class TestConcatDecoded:
    def test_propagates_root_func_name_and_metadata(self):
        root = _make_decoded(
            func_name="caller",
            real_tokens=[300, 301],
            metadata={"origin": "root"},
        )
        callee = _make_decoded(
            func_name="callee",
            real_tokens=[400],
            metadata={"origin": "callee"},
        )
        out = _concat_decoded(root, callee)
        assert out.func_name == "caller"
        assert out.metadata == {"origin": "root"}
        np.testing.assert_array_equal(out.real_tokens, [300, 301, 400])

    def test_concat_ordering(self):
        root = _make_decoded(
            real_tokens=[10, 11, 12],
            identities={Category.BLOCK: [1, 2, 3]},
            numbers=(7,),
            sign_exps=(70,),
        )
        callee = _make_decoded(
            real_tokens=[20, 21],
            identities={Category.BLOCK: [9, 8]},
            numbers=(99,),
            sign_exps=(990,),
        )
        out = _concat_decoded(root, callee)
        np.testing.assert_array_equal(out.real_tokens, [10, 11, 12, 20, 21])
        np.testing.assert_array_equal(
            out.identities[Category.BLOCK], [1, 2, 3, 9, 8]
        )
        np.testing.assert_array_equal(out.numbers_significant, [7, 99])
        np.testing.assert_array_equal(
            out.numbers_sign_exponent, [70, 990]
        )

    def test_root_only_is_clone_like(self):
        root = _make_decoded(
            real_tokens=[300],
            identities={Category.BLOCK: [5]},
        )
        out = _concat_decoded(root)
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.BLOCK],
            root.identities[Category.BLOCK],
        )


# ---------------------------------------------------------------------------
# splice_with_callees: leaf / no-op cases
# ---------------------------------------------------------------------------


class TestDepthZeroAndEmpty:
    def test_max_depth_zero_returns_root_unchanged(self):
        root = _make_decoded(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [0, 1, 2]},
            numbers=(42,),
            sign_exps=(420,),
        )
        callee = _make_decoded(
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [9]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=0,
        )

        # Shape-identical to the root.
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC],
            root.identities[Category.LOCAL_FUNC],
        )
        np.testing.assert_array_equal(
            out.numbers_significant, root.numbers_significant
        )
        np.testing.assert_array_equal(
            out.numbers_sign_exponent, root.numbers_sign_exponent
        )

    def test_empty_call_targets_returns_root_unchanged(self):
        root = _make_decoded(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [0, 1]},
        )
        decode_callee, is_callee_present, _ = _make_table()
        root_section = _StubSection([])  # no callees at all

        # Any max_depth is allowed; output stays root-shaped.
        for depth in (0, 1, 3):
            out = splice_with_callees(
                root_decoded=root,
                root_arm="matched",
                root_section=root_section,
                root_section_offset=100,
                decode_callee=decode_callee,
                is_callee_present=is_callee_present,
                max_depth=depth,
            )
            np.testing.assert_array_equal(
                out.real_tokens, root.real_tokens
            )

    def test_missing_callee_skipped(self):
        # Callee 200 NOT in table → is_callee_present returns False.
        root = _make_decoded(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [0, 1]},
        )
        decode_callee, is_callee_present, _ = _make_table()
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=2,
        )
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC],
            root.identities[Category.LOCAL_FUNC],
        )


# ---------------------------------------------------------------------------
# splice_with_callees: single + multi-callee rebase arithmetic
# ---------------------------------------------------------------------------


class TestSingleCalleeRebase:
    def test_one_category_simple(self):
        # root LOCAL_FUNC = [0, 1, 2]  -> max=2  -> offset=3
        # callee LOCAL_FUNC = [0, 5]    -> rebased [3, 8]
        # output = [0, 1, 2, 3, 8]
        root = _make_decoded(
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [0, 1, 2]},
        )
        callee = _make_decoded(
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [0, 5]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 2, 3, 8]
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])

    def test_root_no_identities_offset_starts_at_zero(self):
        # root has no identities → running_max = -1 → first callee uses
        # offset 0 (i.e. identities pass through unchanged).
        root = _make_decoded(real_tokens=[300])
        callee = _make_decoded(
            real_tokens=[400],
            identities={Category.BLOCK: [0, 1, 2]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        np.testing.assert_array_equal(
            out.identities[Category.BLOCK], [0, 1, 2]
        )


class TestTwoCalleesRunningMax:
    def test_per_category_running_max_accumulates(self):
        # root LOCAL_FUNC = [0, 1]      max=1
        # callee1 LOCAL_FUNC = [10]    offset=2 → [12]      new max=12
        # callee2 LOCAL_FUNC = [5]     offset=13 → [18]     final max=18
        # output identities[LOCAL_FUNC] = [0, 1, 12, 18]
        root = _make_decoded(
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [0, 1]},
        )
        c1 = _make_decoded(
            func_name="c1",
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [10]},
        )
        c2 = _make_decoded(
            func_name="c2",
            real_tokens=[401],
            identities={Category.LOCAL_FUNC: [5]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, c1, []),
            (201, c2, []),
        )
        root_section = _StubSection(
            [_StubCallTarget(200), _StubCallTarget(201)]
        )

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 12, 18]
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 401])


class TestSentinelOnSplice:
    def test_sentinel_sticky_on_rebase(self):
        # root LOCAL_FUNC = [10, 0xFFFF]   max=10 (sentinel excluded)
        # callee LOCAL_FUNC = [0, 0xFFFF, 5]   offset=11
        # rebased = [11, 0xFFFF, 16]
        # output = [10, 0xFFFF, 11, 0xFFFF, 16]
        root = _make_decoded(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [10, 0xFFFF]},
        )
        callee = _make_decoded(
            real_tokens=[400, 401, 402],
            identities={Category.LOCAL_FUNC: [0, 0xFFFF, 5]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC],
            [10, 0xFFFF, 11, 0xFFFF, 16],
        )

    def test_overflow_clips_to_sentinel_after_rebase(self):
        # callee LOCAL_FUNC = [0xFFFE]; offset = 5 -> overflow -> 0xFFFF
        # root has LOCAL_FUNC = [4] (max=4, offset=5).
        root = _make_decoded(
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [4]},
        )
        callee = _make_decoded(
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [0xFFFE]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [4, 0xFFFF]
        )


# ---------------------------------------------------------------------------
# All-Category independence
# ---------------------------------------------------------------------------


class TestAllCategoriesIndependent:
    def test_each_category_has_its_own_running_max(self):
        # Root has different max per category. Callee values must be
        # offset INDEPENDENTLY per category — no cross-pollination.
        root_ids = {
            Category.BLOCK: [3],         # max=3,  offset=4
            Category.LOCAL_FUNC: [10],   # max=10, offset=11
            Category.PLT_FUNC: [],       # max=-1, offset=0
            Category.EXT_FUNC: [0],      # max=0,  offset=1
            Category.RO_DATA_PTR: [50],  # max=50, offset=51
            Category.RW_DATA_PTR: [7],   # max=7,  offset=8
            Category.STRING_PTR: [],     # max=-1, offset=0
            Category.JUMP_TABLE: [1],    # max=1,  offset=2
        }
        callee_ids = {c: [0, 2] for c in Category}

        root = _make_decoded(
            real_tokens=[300],
            identities=root_ids,
        )
        callee = _make_decoded(
            real_tokens=[400],
            identities=callee_ids,
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )

        expected_offsets = {
            Category.BLOCK: 4,
            Category.LOCAL_FUNC: 11,
            Category.PLT_FUNC: 0,
            Category.EXT_FUNC: 1,
            Category.RO_DATA_PTR: 51,
            Category.RW_DATA_PTR: 8,
            Category.STRING_PTR: 0,
            Category.JUMP_TABLE: 2,
        }
        for c, off in expected_offsets.items():
            root_seq = list(root_ids[c])
            callee_seq = [off + 0, off + 2]
            np.testing.assert_array_equal(
                out.identities[c], root_seq + callee_seq
            ), c.name


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_self_recursion_skipped(self):
        # Root's call_target points back at the root's own section_offset.
        # Walker must skip it (cycle).
        root = _make_decoded(
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [0, 1]},
        )
        # Empty table: the cycle check fires before any decode happens,
        # so we don't need to register the root in decode_callee.
        decode_callee, _, _ = _make_table()

        # is_callee_present returns True for the cycle target so the
        # cycle check (not the presence check) is what filters it.
        def is_callee_present(offset: int, arm: str) -> bool:
            return offset == 100

        root_section = _StubSection([_StubCallTarget(100)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=3,
        )
        # Output is just the root — cycle target skipped.
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC],
            root.identities[Category.LOCAL_FUNC],
        )

    def test_mutual_recursion_terminates(self):
        # A (offset 100) -> B (offset 200) -> A
        # depth=2: A spliced (root), B spliced (one level down).
        #          B's call to A skipped (A in visited).
        # No infinite recursion; output is A's body + B's body.
        a = _make_decoded(
            func_name="A",
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [0, 1]},
        )
        b = _make_decoded(
            func_name="B",
            real_tokens=[400, 401],
            identities={Category.LOCAL_FUNC: [0]},
        )
        # B's section calls back into A.
        decode_callee, is_callee_present, _ = _make_table(
            (100, a, [_StubCallTarget(200)]),  # not actually invoked at root level
            (200, b, [_StubCallTarget(100)]),
        )
        # A's call_targets points to B.
        a_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=2,
        )

        # A's body [300, 301] + B's body [400, 401]
        np.testing.assert_array_equal(
            out.real_tokens, [300, 301, 400, 401]
        )
        # A's LOCAL_FUNC = [0, 1]; max=1; B's LOCAL_FUNC rebased = [2].
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 2]
        )

    def test_visited_discards_on_exit_allows_repeat_via_sibling(self):
        # Per DAG-active-path semantics (plan ## Algorithm pseudo-code:
        # visited.add / visited.discard), the same callee reached via two
        # sibling CTs at the same depth is spliced TWICE.
        root = _make_decoded(
            func_name="root",
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [0]},
        )
        # B is the shared callee.
        b = _make_decoded(
            func_name="B",
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [0]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, b, []),
        )
        # Root calls B twice (two CTs, same target).
        root_section = _StubSection(
            [_StubCallTarget(200), _StubCallTarget(200)]
        )

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        # Root real_tokens [300] + B [400] + B [400] = [300, 400, 400].
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 400])
        # Root LOCAL_FUNC=[0]; max=0; B1 offset=1 → [1]; max=1; B2 offset=2 → [2].
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 2]
        )


# ---------------------------------------------------------------------------
# Multi-level chains + depth budget
# ---------------------------------------------------------------------------


class TestDepthCapBudget:
    def _build_chain(self):
        """A -> B -> C -> D, single-call-target chain."""
        a = _make_decoded(func_name="A", real_tokens=[300])
        b = _make_decoded(func_name="B", real_tokens=[400])
        c = _make_decoded(func_name="C", real_tokens=[500])
        d = _make_decoded(func_name="D", real_tokens=[600])
        decode_callee, is_callee_present, _ = _make_table(
            (100, a, [_StubCallTarget(200)]),
            (200, b, [_StubCallTarget(300)]),
            (300, c, [_StubCallTarget(400)]),
            (400, d, []),
        )
        a_section = _StubSection([_StubCallTarget(200)])
        return a, a_section, decode_callee, is_callee_present

    def test_depth_3_full_chain(self):
        a, a_section, dc, icp = self._build_chain()
        out = splice_with_callees(
            root_decoded=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee=dc,
            is_callee_present=icp,
            max_depth=3,
        )
        np.testing.assert_array_equal(
            out.real_tokens, [300, 400, 500, 600]
        )

    def test_depth_2_skips_deepest(self):
        # A -> B -> C; D's body NOT spliced (depth runs out at C).
        a, a_section, dc, icp = self._build_chain()
        out = splice_with_callees(
            root_decoded=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee=dc,
            is_callee_present=icp,
            max_depth=2,
        )
        np.testing.assert_array_equal(
            out.real_tokens, [300, 400, 500]
        )

    def test_depth_1_only_one_level(self):
        a, a_section, dc, icp = self._build_chain()
        out = splice_with_callees(
            root_decoded=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee=dc,
            is_callee_present=icp,
            max_depth=1,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])


# ---------------------------------------------------------------------------
# Propagation of func_name / metadata
# ---------------------------------------------------------------------------


class TestRootMetadataPropagation:
    def test_func_name_and_metadata_from_root_only(self):
        root = _make_decoded(
            func_name="caller",
            real_tokens=[300],
            metadata={"origin": "caller"},
        )
        callee = _make_decoded(
            func_name="callee",
            real_tokens=[400],
            metadata={"origin": "callee"},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        assert out.func_name == "caller"
        assert out.metadata == {"origin": "caller"}


# ---------------------------------------------------------------------------
# Number side-array concatenation (uniformity check)
# ---------------------------------------------------------------------------


class TestNumberConcat:
    def test_numbers_concatenate_in_order(self):
        root = _make_decoded(
            real_tokens=[300],
            numbers=(1, 2),
            sign_exps=(10, 20),
        )
        callee = _make_decoded(
            real_tokens=[400],
            numbers=(99,),
            sign_exps=(990,),
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_decoded=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
        )
        np.testing.assert_array_equal(out.numbers_significant, [1, 2, 99])
        np.testing.assert_array_equal(
            out.numbers_sign_exponent, [10, 20, 990]
        )


# ---------------------------------------------------------------------------
# Public-API guard: negative max_depth rejected
# ---------------------------------------------------------------------------


class TestPublicGuards:
    def test_negative_max_depth_rejected(self):
        root = _make_decoded(real_tokens=[300])
        decode_callee, is_callee_present, _ = _make_table()
        root_section = _StubSection([])
        with pytest.raises(ValueError):
            splice_with_callees(
                root_decoded=root,
                root_arm="matched",
                root_section=root_section,
                root_section_offset=100,
                decode_callee=decode_callee,
                is_callee_present=is_callee_present,
                max_depth=-1,
            )


# ---------------------------------------------------------------------------
# Sentinel constant exposed for downstream importers
# ---------------------------------------------------------------------------


def test_identity_sentinel_constant_is_uint16_0xFFFF():
    assert int(IDENTITY_SENTINEL) == 0xFFFF
    assert np.uint16(IDENTITY_SENTINEL).dtype == np.uint16
