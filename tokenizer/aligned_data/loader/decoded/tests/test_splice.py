"""Tests for ``decoded.splice.splice_with_callees`` + its helpers.

Synthetic-only: no ``BinarySession``. The
``decode_callee_to_staging`` callback is closed over a tiny
``{offset: (_StagingDecoded, _StubSection)}`` table; ``_StubSection`` +
``_StubCallTarget`` expose just the duck-type attributes the walker
reads.

Coverage targets (post FID-resolution + compaction refactor):

* :func:`_compact_ids` — densification, alias merging, sentinel
  preservation, offset shift, overflow assertion, u16 + u32 input.
* :func:`_concat_staging` — verbatim per-Category concat, root
  func_name/metadata propagation.
* :func:`splice_with_callees`
  - ``depth=0`` returns root (compacted).
  - empty ``call_targets`` returns root.
  - missing callee skipped (``is_callee_present`` False).
  - single + multi-callee splice.
  - same FID across caller + callee aliases to one compacted id.
  - sentinel positions stay sentinels through compaction.
  - all 8 categories compact independently.
  - self-recursion + mutual-recursion cycle keys.
  - DAG semantics: same callee reached via two siblings splices twice.
  - multi-level chain at depth budgets 1/2/3.
  - func_name + metadata propagate from root only.
  - number side-arrays concatenate in order.
  - negative ``max_depth`` rejected.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.extract import _StagingDecoded
from tokenizer.aligned_data.loader.decoded.splice import (
    IDENTITY_SENTINEL,
    _compact_ids,
    _concat_staging,
    _input_sentinel_for,
    splice_with_callees,
)
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Test stubs / builders
# ---------------------------------------------------------------------------


def _make_staging(
    *,
    func_name: str = "root",
    real_tokens=(),
    identities: "Dict[Category, list] | None" = None,
    numbers: Tuple[int, ...] = (),
    sign_exps: Tuple[int, ...] = (),
    metadata: "Dict | None" = None,
    fid_keyed_dtype: np.dtype = np.dtype(np.uint16),
) -> _StagingDecoded:
    """Build a :class:`_StagingDecoded` with sane defaults.

    ``identities`` accepts sparse ``{Category: list[int]}``; absent
    categories default to length-0 arrays. ``fid_keyed_dtype`` selects
    the dtype for absent FID-keyed categories so the synthetic test can
    construct either u16 or u32 staging shapes; categories whose
    ``identities[c]`` list is supplied inherit dtype from that list
    via ``np.array(..., dtype=)`` -- the test passes lists of Python
    ints which numpy coerces freely. ``numbers`` + ``sign_exps`` must
    be the same length.
    """
    if identities is None:
        identities = {}
    if metadata is None:
        metadata = {}
    from tokenizer.aligned_data.loader.decoded.category_tokens import (
        FID_KEYED_CATEGORIES,
    )
    full_identities: Dict[Category, np.ndarray] = {}
    for c in Category:
        if c in identities:
            # Caller specified the dtype implicitly via the list contents;
            # default to u16 so the synthetic tests don't need to think
            # about per-Category staging dtypes when they don't care.
            full_identities[c] = np.array(identities[c], dtype=np.uint16)
        else:
            dtype = fid_keyed_dtype if c in FID_KEYED_CATEGORIES else np.uint16
            full_identities[c] = np.empty(0, dtype=dtype)
    return _StagingDecoded(
        real_tokens=np.array(real_tokens, dtype=np.uint16),
        identities=full_identities,
        numbers_significant=np.array(numbers, dtype=np.uint64),
        numbers_sign_exponent=np.array(sign_exps, dtype=np.uint32),
        func_name=func_name,
        metadata=metadata,
    )


class _StubCallTarget:
    """Duck-type for ``aligned_data.matched_sections_bin.CallTarget``."""

    def __init__(self, function_section_ptr: int, is_matched: bool = True):
        self.function_section_ptr = function_section_ptr
        self.is_matched = is_matched


class _StubVariant:
    """Duck-type for ``aligned_data.matched_sections_bin.VariantBlock``.

    Only the two attributes the walker reads are exposed:

    * ``per_call_entries: list[tuple[int, int]]`` -- (called_idx,
      section_variant_index) pairs.
    * ``variant_ref_offset: int`` -- the vkey (variant_ref_offset on
      the real type).
    """

    def __init__(
        self,
        per_call_entries: List[Tuple[int, int]],
        variant_ref_offset: int = 0,
    ):
        self.per_call_entries = per_call_entries
        self.variant_ref_offset = variant_ref_offset


class _StubSection:
    """Duck-type for ``aligned_data.matched_sections_bin.Section``.

    Reads ``call_targets`` + ``variants``. Default ``variants`` builds
    a single variant with ``vkey=0`` whose ``per_call_entries`` maps
    each call_target index to ``section_variant_index=0`` -- the
    legacy single-variant default that all pre-variants tests inherit.
    """

    def __init__(
        self,
        call_targets: List[_StubCallTarget],
        variants: "List[_StubVariant] | None" = None,
    ):
        self.call_targets = call_targets
        if variants is None:
            variants = [
                _StubVariant(
                    per_call_entries=[
                        (i, 0) for i in range(len(call_targets))
                    ],
                    variant_ref_offset=0,
                )
            ]
        self.variants = variants


def _make_table(
    *entries,
):
    """Build a ``(decode_callee_to_staging, is_callee_present, table)`` triple.

    ``entries`` is ``(section_offset, staging, call_targets_list)``.
    The callback accepts the new ``callee_variant_index`` arg per the
    walker contract and ignores it (the default stub section's per_call
    entries always map J=0).
    """
    table: Dict[int, Tuple[_StagingDecoded, _StubSection]] = {
        offset: (staging, _StubSection(call_targets))
        for offset, staging, call_targets in entries
    }

    def decode_callee_to_staging(
        offset: int, arm: str, callee_variant_index: int
    ):
        # Default stub: per_call_entries maps J=0 for every call_target,
        # so the walker should always request J=0 here. The assertion
        # is cheap and catches regressions in the J propagation path.
        assert callee_variant_index == 0, (
            f"default _make_table stub expects callee_variant_index=0; "
            f"got {callee_variant_index}"
        )
        return table[offset]

    def is_callee_present(offset: int, arm: str) -> bool:
        return offset in table

    return decode_callee_to_staging, is_callee_present, table


# Single-variant defaults: every legacy test instantiates the walker
# with these to match the pre-variants behavior under the new API.
_DEFAULT_PRIMARY_VARIANT_IDX = 0
_DEFAULT_SELECTION_VKEYS = frozenset({0})


# ---------------------------------------------------------------------------
# _compact_ids -- plan Decision 28 + 31
# ---------------------------------------------------------------------------


class TestCompactIds:
    def test_empty(self):
        out = _compact_ids(np.array([], dtype=np.uint16))
        assert out.dtype == np.uint16
        assert out.size == 0

    def test_all_unique_u16(self):
        out = _compact_ids(np.array([42, 99, 7], dtype=np.uint16))
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, [0, 1, 2])

    def test_alias_first_occurrence_wins(self):
        # 42 appears at positions 0 + 2; both share compacted id 0.
        # 99 at position 1 gets compacted id 1. 7 at position 3 gets 2.
        # 99 again at position 4 reuses id 1.
        out = _compact_ids(np.array([42, 99, 42, 7, 99], dtype=np.uint16))
        np.testing.assert_array_equal(out, [0, 1, 0, 2, 1])

    def test_sentinel_preserved_u16(self):
        sentinel = _input_sentinel_for(np.dtype(np.uint16))
        out = _compact_ids(
            np.array([42, sentinel, 99, sentinel], dtype=np.uint16)
        )
        np.testing.assert_array_equal(out, [0, 0xFFFF, 1, 0xFFFF])

    def test_all_sentinel_u16(self):
        sentinel = _input_sentinel_for(np.dtype(np.uint16))
        out = _compact_ids(np.array([sentinel] * 3, dtype=np.uint16))
        np.testing.assert_array_equal(out, [0xFFFF, 0xFFFF, 0xFFFF])

    def test_u32_input_compacts_to_u16(self):
        # FID-keyed staging uses u32 with 0xFFFFFFFF sentinel; compaction
        # downsizes to u16 and folds the sentinel.
        sentinel_u32 = _input_sentinel_for(np.dtype(np.uint32))
        out = _compact_ids(
            np.array([100_000, 200_000, 100_000, sentinel_u32], dtype=np.uint32)
        )
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, [0, 1, 0, 0xFFFF])

    def test_offset_shifts_compacted_range(self):
        out = _compact_ids(
            np.array([42, 99, 42], dtype=np.uint16), offset=5
        )
        np.testing.assert_array_equal(out, [5, 6, 5])

    def test_overflow_assertion(self):
        # 0xFFFF distinct values exceed the u16 - 1 cap.
        many = np.arange(0xFFFF, dtype=np.uint32)
        with pytest.raises(AssertionError):
            _compact_ids(many)

    def test_unsupported_dtype_rejected(self):
        with pytest.raises(ValueError):
            _compact_ids(np.array([1, 2, 3], dtype=np.uint8))


# ---------------------------------------------------------------------------
# _concat_staging
# ---------------------------------------------------------------------------


class TestConcatStaging:
    def test_propagates_root_func_name_and_metadata(self):
        root = _make_staging(
            func_name="caller",
            real_tokens=[300, 301],
            metadata={"origin": "root"},
        )
        callee = _make_staging(
            func_name="callee",
            real_tokens=[400],
            metadata={"origin": "callee"},
        )
        out = _concat_staging(root, callee)
        assert out.func_name == "caller"
        assert out.metadata == {"origin": "root"}
        np.testing.assert_array_equal(out.real_tokens, [300, 301, 400])

    def test_concat_ordering(self):
        root = _make_staging(
            real_tokens=[10, 11, 12],
            identities={Category.BLOCK: [1, 2, 3]},
            numbers=(7,),
            sign_exps=(70,),
        )
        callee = _make_staging(
            real_tokens=[20, 21],
            identities={Category.BLOCK: [9, 8]},
            numbers=(99,),
            sign_exps=(990,),
        )
        out = _concat_staging(root, callee)
        np.testing.assert_array_equal(out.real_tokens, [10, 11, 12, 20, 21])
        np.testing.assert_array_equal(
            out.identities[Category.BLOCK], [1, 2, 3, 9, 8]
        )
        np.testing.assert_array_equal(out.numbers_significant, [7, 99])
        np.testing.assert_array_equal(
            out.numbers_sign_exponent, [70, 990]
        )

    def test_root_only_is_clone_like(self):
        root = _make_staging(
            real_tokens=[300],
            identities={Category.BLOCK: [5]},
        )
        out = _concat_staging(root)
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.BLOCK],
            root.identities[Category.BLOCK],
        )


# ---------------------------------------------------------------------------
# splice_with_callees: leaf / no-op cases
# ---------------------------------------------------------------------------


class TestDepthZeroAndEmpty:
    def test_max_depth_zero_returns_root_compacted(self):
        root = _make_staging(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [42, 99, 42]},
            numbers=(7,),
            sign_exps=(70,),
        )
        callee = _make_staging(
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [9]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=0,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        # Root LOCAL_FUNC = [42, 99, 42] -> compacted [0, 1, 0].
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 0]
        )
        np.testing.assert_array_equal(
            out.numbers_significant, root.numbers_significant
        )
        np.testing.assert_array_equal(
            out.numbers_sign_exponent, root.numbers_sign_exponent
        )

    def test_empty_call_targets_returns_root_compacted(self):
        root = _make_staging(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [10, 20, 10]},
        )
        decode_callee, is_callee_present, _ = _make_table()
        root_section = _StubSection([])

        for depth in (0, 1, 3):
            out = splice_with_callees(
                root_staging=root,
                root_arm="matched",
                root_section=root_section,
                root_section_offset=100,
                decode_callee_to_staging=decode_callee,
                is_callee_present=is_callee_present,
                max_depth=depth,
                primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
                initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
            )
            np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
            np.testing.assert_array_equal(
                out.identities[Category.LOCAL_FUNC], [0, 1, 0]
            )

    def test_missing_callee_skipped(self):
        root = _make_staging(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [42, 99]},
        )
        decode_callee, is_callee_present, _ = _make_table()
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=2,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1]
        )


# ---------------------------------------------------------------------------
# splice_with_callees: single + multi-callee compaction arithmetic
# ---------------------------------------------------------------------------


class TestSingleCalleeCompaction:
    def test_one_category_simple(self):
        # Root LOCAL_FUNC = [42, 99, 7]; callee LOCAL_FUNC = [42, 5].
        # Verbatim concat: [42, 99, 7, 42, 5].
        # Compaction first-occurrence wins: [0, 1, 2, 0, 3].
        # The shared FID 42 at positions 0 + 3 aliases to compact id 0 --
        # FID unification across the splice.
        root = _make_staging(
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [42, 99, 7]},
        )
        callee = _make_staging(
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [42, 5]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 2, 0, 3]
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])

    def test_root_no_identities_callee_compacts_from_zero(self):
        root = _make_staging(real_tokens=[300])
        callee = _make_staging(
            real_tokens=[400],
            identities={Category.BLOCK: [10, 20, 10]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        # Empty + [10, 20, 10] -> compact [0, 1, 0].
        np.testing.assert_array_equal(
            out.identities[Category.BLOCK], [0, 1, 0]
        )


class TestTwoCalleesShareFid:
    def test_shared_fid_across_callees_aliases(self):
        # Root has no LOCAL_FUNC tokens; the two callees both emit the
        # SAME FID 77 -- compaction must alias to one compact id.
        root = _make_staging(real_tokens=[300])
        c1 = _make_staging(
            func_name="c1",
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [77]},
        )
        c2 = _make_staging(
            func_name="c2",
            real_tokens=[401],
            identities={Category.LOCAL_FUNC: [77]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, c1, []),
            (201, c2, []),
        )
        root_section = _StubSection(
            [_StubCallTarget(200), _StubCallTarget(201)]
        )

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 0]
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 401])


class TestSentinelOnSplice:
    def test_sentinel_sticks_through_compaction(self):
        sentinel = _input_sentinel_for(np.dtype(np.uint16))
        root = _make_staging(
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [10, sentinel]},
        )
        callee = _make_staging(
            real_tokens=[400, 401, 402],
            identities={Category.LOCAL_FUNC: [99, sentinel, 5]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        # Concat: [10, sentinel, 99, sentinel, 5]
        # Compact: [0, 0xFFFF, 1, 0xFFFF, 2]
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC],
            [0, 0xFFFF, 1, 0xFFFF, 2],
        )


# ---------------------------------------------------------------------------
# All-Category independence
# ---------------------------------------------------------------------------


class TestAllCategoriesIndependent:
    def test_each_category_compacts_independently(self):
        root_ids = {
            Category.BLOCK: [3],
            Category.LOCAL_FUNC: [10],
            Category.PLT_FUNC: [],
            Category.EXT_FUNC: [0],
            Category.RO_DATA_PTR: [50],
            Category.RW_DATA_PTR: [7],
            Category.STRING_PTR: [],
            Category.JUMP_TABLE: [1],
        }
        callee_ids = {c: [0, 2] for c in Category}

        root = _make_staging(real_tokens=[300], identities=root_ids)
        callee = _make_staging(real_tokens=[400], identities=callee_ids)
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )

        # Per Category, concat is (root_ids + [0, 2]); compaction gives
        # first-occurrence-wins ids in encounter order. Cross-category
        # values are independent -- no shared id space.
        for c, root_seq in root_ids.items():
            concat = root_seq + [0, 2]
            mapping: Dict[int, int] = {}
            expected = []
            for v in concat:
                if v not in mapping:
                    mapping[v] = len(mapping)
                expected.append(mapping[v])
            np.testing.assert_array_equal(
                out.identities[c], expected, err_msg=f"Category {c.name}"
            )


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_self_recursion_skipped(self):
        root = _make_staging(
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [42, 99]},
        )
        decode_callee, _, _ = _make_table()

        def is_callee_present(offset: int, arm: str) -> bool:
            return offset == 100

        root_section = _StubSection([_StubCallTarget(100)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=3,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(out.real_tokens, root.real_tokens)
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1]
        )

    def test_mutual_recursion_terminates(self):
        # A (offset 100) -> B (offset 200) -> A. The visited set blocks
        # the cycle on the second leg.
        a = _make_staging(
            func_name="A",
            real_tokens=[300, 301],
            identities={Category.LOCAL_FUNC: [42, 99]},
        )
        b = _make_staging(
            func_name="B",
            real_tokens=[400, 401],
            identities={Category.LOCAL_FUNC: [42]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (100, a, [_StubCallTarget(200)]),
            (200, b, [_StubCallTarget(100)]),
        )
        a_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=2,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )

        np.testing.assert_array_equal(
            out.real_tokens, [300, 301, 400, 401]
        )
        # Concat: [42, 99] + [42] = [42, 99, 42]; compact -> [0, 1, 0].
        # The shared FID 42 aliases across A + B.
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 0]
        )

    def test_visited_discards_on_exit_allows_repeat_via_sibling(self):
        # DAG-active-path semantics: the same callee reached via two
        # sibling CTs at the same depth is spliced TWICE.
        root = _make_staging(
            func_name="root",
            real_tokens=[300],
            identities={Category.LOCAL_FUNC: [42]},
        )
        b = _make_staging(
            func_name="B",
            real_tokens=[400],
            identities={Category.LOCAL_FUNC: [99]},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, b, []),
        )
        root_section = _StubSection(
            [_StubCallTarget(200), _StubCallTarget(200)]
        )

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 400])
        # Concat: [42, 99, 99]; compact [0, 1, 1] -- B's body spliced
        # twice but the shared FID 99 aliases to one compact id.
        np.testing.assert_array_equal(
            out.identities[Category.LOCAL_FUNC], [0, 1, 1]
        )


# ---------------------------------------------------------------------------
# Multi-level chains + depth budget
# ---------------------------------------------------------------------------


class TestDepthCapBudget:
    def _build_chain(self):
        a = _make_staging(func_name="A", real_tokens=[300])
        b = _make_staging(func_name="B", real_tokens=[400])
        c = _make_staging(func_name="C", real_tokens=[500])
        d = _make_staging(func_name="D", real_tokens=[600])
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
            root_staging=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee_to_staging=dc,
            is_callee_present=icp,
            max_depth=3,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(
            out.real_tokens, [300, 400, 500, 600]
        )

    def test_depth_2_skips_deepest(self):
        a, a_section, dc, icp = self._build_chain()
        out = splice_with_callees(
            root_staging=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee_to_staging=dc,
            is_callee_present=icp,
            max_depth=2,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 500])

    def test_depth_1_only_one_level(self):
        a, a_section, dc, icp = self._build_chain()
        out = splice_with_callees(
            root_staging=a,
            root_arm="matched",
            root_section=a_section,
            root_section_offset=100,
            decode_callee_to_staging=dc,
            is_callee_present=icp,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])


# ---------------------------------------------------------------------------
# Propagation of func_name / metadata
# ---------------------------------------------------------------------------


class TestRootMetadataPropagation:
    def test_func_name_and_metadata_from_root_only(self):
        root = _make_staging(
            func_name="caller",
            real_tokens=[300],
            metadata={"origin": "caller"},
        )
        callee = _make_staging(
            func_name="callee",
            real_tokens=[400],
            metadata={"origin": "callee"},
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
        )
        assert out.func_name == "caller"
        assert out.metadata == {"origin": "caller"}


# ---------------------------------------------------------------------------
# Number side-array concatenation (uniformity check)
# ---------------------------------------------------------------------------


class TestNumberConcat:
    def test_numbers_concatenate_in_order(self):
        root = _make_staging(
            real_tokens=[300],
            numbers=(1, 2),
            sign_exps=(10, 20),
        )
        callee = _make_staging(
            real_tokens=[400],
            numbers=(99,),
            sign_exps=(990,),
        )
        decode_callee, is_callee_present, _ = _make_table(
            (200, callee, []),
        )
        root_section = _StubSection([_StubCallTarget(200)])

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode_callee,
            is_callee_present=is_callee_present,
            max_depth=1,
            primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
            initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
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
        root = _make_staging(real_tokens=[300])
        decode_callee, is_callee_present, _ = _make_table()
        root_section = _StubSection([])
        with pytest.raises(ValueError):
            splice_with_callees(
                root_staging=root,
                root_arm="matched",
                root_section=root_section,
                root_section_offset=100,
                decode_callee_to_staging=decode_callee,
                is_callee_present=is_callee_present,
                max_depth=-1,
                primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
                initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
            )


# ---------------------------------------------------------------------------
# Sentinel constant exposed for downstream importers
# ---------------------------------------------------------------------------


def test_identity_sentinel_constant_is_uint16_0xFFFF():
    assert int(IDENTITY_SENTINEL) == 0xFFFF
    assert np.uint16(IDENTITY_SENTINEL).dtype == np.uint16
