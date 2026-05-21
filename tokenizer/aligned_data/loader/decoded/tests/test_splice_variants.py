"""Walker tests for the variant-aware splice contract.

Covers the new walker behaviors layered on top of the legacy splice:

* Inlining-equivalence flag: a call_target K is spliced iff SOME
  variants in the current selection called K AND some did not.
* Callee variant follows ``per_call_entries``: the third arg of
  ``decode_callee_to_staging`` is the ``section_variant_index`` from
  the primary variant's per_call_entry for the call_target. Under the
  D6 fallback (flag ON + primary didn't call K), it is the lowest
  ``v_idx`` in ``called_by_in_selection``'s per_call_entry instead.
* Selection narrowing on recursion: ONLY under flag ON; flag OFF
  threads the original selection through unchanged.

Test stubs mirror the production ``Section`` API:

* ``_StubVariant`` carries ``per_call_entries: list[tuple[called_idx,
  section_variant_index]]`` + ``variant_ref_offset: int`` (vkey).
* ``_StubSection`` carries ``call_targets`` + ``variants`` (no default
  here -- every test in this file constructs a multi-variant section
  explicitly).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.loader.decoded.extract import _StagingDecoded
from tokenizer.aligned_data.loader.decoded.splice import splice_with_callees
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class _StubCallTarget:
    def __init__(self, function_section_ptr: int) -> None:
        self.function_section_ptr = function_section_ptr
        self.is_matched = True


class _StubVariant:
    def __init__(
        self,
        per_call_entries: List[Tuple[int, int]],
        variant_ref_offset: int,
    ) -> None:
        self.per_call_entries = per_call_entries
        self.variant_ref_offset = variant_ref_offset


class _StubSection:
    """Explicit multi-variant section for variant-specific walker tests."""

    def __init__(
        self,
        call_targets: List[_StubCallTarget],
        variants: List[_StubVariant],
    ) -> None:
        self.call_targets = call_targets
        self.variants = variants


def _make_staging(
    *,
    func_name: str = "f",
    real_tokens: Tuple[int, ...] = (),
    block_ids: Tuple[int, ...] = (),
) -> _StagingDecoded:
    identities: Dict[Category, np.ndarray] = {
        c: np.empty(0, dtype=np.uint16) for c in Category
    }
    if block_ids:
        identities[Category.BLOCK] = np.array(block_ids, dtype=np.uint16)
    return _StagingDecoded(
        real_tokens=np.array(real_tokens, dtype=np.uint16),
        identities=identities,
        numbers_significant=np.empty(0, dtype=np.uint64),
        numbers_sign_exponent=np.empty(0, dtype=np.uint32),
        func_name=func_name,
        metadata={},
    )


def _make_table_with_capture(
    table: Dict[int, Tuple[_StagingDecoded, _StubSection]],
    *,
    captures: "List[Tuple[int, int]] | None" = None,
) -> Tuple[Callable, Callable]:
    """Build ``decode_callee_to_staging`` + ``is_callee_present`` from
    a ``{section_offset: (staging, section)}`` table. When ``captures``
    is provided, every callback invocation appends ``(offset,
    callee_variant_index)`` to it -- enables assertions on the J the
    walker propagated.
    """

    def decode(offset: int, arm: str, callee_variant_index: int):
        if captures is not None:
            captures.append((offset, callee_variant_index))
        return table[offset]

    def present(offset: int, arm: str) -> bool:
        return offset in table

    return decode, present


# ---------------------------------------------------------------------------
# Inlining flag: per-callsite decision (D5)
# ---------------------------------------------------------------------------


class TestInliningFlagPerCallTargetDecision:
    """A 2-variant section where the two variants disagree on calling K.

    * vkey 0 (v_idx=0) calls K with J=0.
    * vkey 1 (v_idx=1) does NOT call K (empty per_call_entries).

    With BOTH variants in the selection:
    * Flag OFF (current behavior): K is spliced.
    * Flag ON: ``called_by_in_selection == {0}`` which is strictly
      shorter than ``selection_v_idxs_in_section == {0, 1}`` -- so
      "some called, some didn't"; K IS spliced.

    With ONLY v_idx=0 in selection:
    * Flag ON: called_by == selection == {0} -> K skipped (no
      inlining variation).

    With ONLY v_idx=1 in selection:
    * Flag ON: called_by is empty -> K skipped (no selected variant
      called it).
    """

    def _build(self):
        callee = _make_staging(func_name="K", real_tokens=[400])
        callee_section = _StubSection(
            call_targets=[],
            variants=[
                _StubVariant(per_call_entries=[], variant_ref_offset=0)
            ],
        )
        table = {200: (callee, callee_section)}
        decode, present = _make_table_with_capture(table)
        root = _make_staging(func_name="root", real_tokens=[300])
        root_section = _StubSection(
            call_targets=[_StubCallTarget(200)],
            variants=[
                _StubVariant(
                    per_call_entries=[(0, 0)], variant_ref_offset=0
                ),
                _StubVariant(
                    per_call_entries=[], variant_ref_offset=1
                ),
            ],
        )
        return root, root_section, decode, present

    def test_flag_off_splices_K(self):
        root, root_section, decode, present = self._build()
        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({0, 1}),
            inlined_equivalent_call_targets_only=False,
        )
        # K's body spliced after root.
        np.testing.assert_array_equal(out.real_tokens, [300, 400])

    def test_flag_on_both_variants_in_selection_splices_K(self):
        root, root_section, decode, present = self._build()
        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({0, 1}),
            inlined_equivalent_call_targets_only=True,
        )
        # called_by={0}, selection={0,1} -> not equal -> splice.
        np.testing.assert_array_equal(out.real_tokens, [300, 400])

    def test_flag_on_only_caller_in_selection_skips_K(self):
        root, root_section, decode, present = self._build()
        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({0}),
            inlined_equivalent_call_targets_only=True,
        )
        # called_by={0}, selection={0} -> equal -> K skipped.
        np.testing.assert_array_equal(out.real_tokens, [300])

    def test_flag_on_only_non_caller_in_selection_skips_K(self):
        root, root_section, decode, present = self._build()
        # Switch primary to v_idx=1 (the variant that didn't call K).
        # Without changing root_staging, we still feed v=1 as primary
        # to exercise the "called_by empty" branch.
        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=1,
            initial_selection_vkeys=frozenset({1}),
            inlined_equivalent_call_targets_only=True,
        )
        # called_by=∅ -> K skipped.
        np.testing.assert_array_equal(out.real_tokens, [300])


# ---------------------------------------------------------------------------
# Callee variant follows per_call_entries (D6 primary path)
# ---------------------------------------------------------------------------


class TestCalleeVariantFromPrimary:
    """Variant 0's per_call_entry for K carries section_variant_index=3.

    The walker must invoke decode_callee_to_staging with J=3, NOT 0
    (the legacy hand-picked default). The callee section has four
    variants so v=3 is valid.
    """

    def test_callback_receives_primarys_J(self):
        callee = _make_staging(func_name="K", real_tokens=[400])
        callee_section = _StubSection(
            call_targets=[],
            variants=[
                _StubVariant(per_call_entries=[], variant_ref_offset=v)
                for v in (10, 11, 12, 13)
            ],
        )
        table = {200: (callee, callee_section)}
        captures: List[Tuple[int, int]] = []
        decode, present = _make_table_with_capture(table, captures=captures)
        root = _make_staging(func_name="root", real_tokens=[300])
        root_section = _StubSection(
            call_targets=[_StubCallTarget(200)],
            variants=[
                _StubVariant(
                    per_call_entries=[(0, 3)], variant_ref_offset=0
                )
            ],
        )

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({0}),
            inlined_equivalent_call_targets_only=False,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])
        # Callback invoked exactly once with J=3.
        assert captures == [(200, 3)]


# ---------------------------------------------------------------------------
# Selection narrowing on recursion (D7)
# ---------------------------------------------------------------------------


class TestSelectionNarrowingDepth2:
    """Depth-2 chain: root R -> K -> L.

    Construction:

    * R has 3 variants (vkeys 100, 101, 102), all in the selection.
      v=0 calls K with J=0. v=1 calls K with J=1. v=2 does NOT call K.
      -> called_by_K = {0, 1} != selection_v_idxs = {0, 1, 2} -- K
      IS spliced under flag ON ("some called, some didn't").

    * K has 3 variants (vkeys 200, 201, 202). v=0 (vkey 200) calls L
      with J=0. v=1 (vkey 201) does NOT call L. v=2 (vkey 202) does
      NOT call L. The narrowed selection at K = {200, 201} (only the
      vkeys reached from R's variants that called K). When K's
      variants are filtered against {200, 201}, v_idxs = {0, 1}.

      called_by_L = {0} != {0, 1} -> L IS spliced under flag ON.

    * If narrowing were a no-op (flag-ON bug), K's selection would
      stay {100, 101, 102}. K's variants have vkeys 200/201/202; none
      of those are in {100, 101, 102} -> selection_v_idxs at K = ∅,
      called_by_L = ∅ -> L would be SKIPPED. That is the negative
      observation that pins narrowing as the actual cause of L being
      spliced.

    * Flag OFF: no narrowing, no inlining check -> L spliced via
      standard iteration regardless.
    """

    def _build(self):
        root = _make_staging(func_name="R", real_tokens=[300])
        k = _make_staging(func_name="K", real_tokens=[400])
        ell = _make_staging(func_name="L", real_tokens=[500])

        ell_section = _StubSection(
            call_targets=[],
            variants=[
                _StubVariant(per_call_entries=[], variant_ref_offset=300)
            ],
        )
        k_section = _StubSection(
            call_targets=[_StubCallTarget(500)],
            variants=[
                # K's v=0 (vkey 200) calls L with J=0.
                _StubVariant(
                    per_call_entries=[(0, 0)], variant_ref_offset=200
                ),
                # K's v=1 (vkey 201) does NOT call L.
                _StubVariant(
                    per_call_entries=[], variant_ref_offset=201
                ),
                # K's v=2 (vkey 202) does NOT call L (and isn't reached
                # by any of R's selected variants either).
                _StubVariant(
                    per_call_entries=[], variant_ref_offset=202
                ),
            ],
        )
        root_section = _StubSection(
            call_targets=[_StubCallTarget(400)],
            variants=[
                # R's v=0 (vkey 100) calls K with J=0 (K vkey 200).
                _StubVariant(
                    per_call_entries=[(0, 0)], variant_ref_offset=100
                ),
                # R's v=1 (vkey 101) calls K with J=1 (K vkey 201).
                _StubVariant(
                    per_call_entries=[(0, 1)], variant_ref_offset=101
                ),
                # R's v=2 (vkey 102) does NOT call K -- triggers the
                # "some called, some didn't" branch under flag ON.
                _StubVariant(
                    per_call_entries=[], variant_ref_offset=102
                ),
            ],
        )
        table = {
            400: (k, k_section),
            500: (ell, ell_section),
        }
        captures: List[Tuple[int, int]] = []
        decode, present = _make_table_with_capture(table, captures=captures)
        return root, root_section, decode, present, captures

    def test_flag_on_narrows_to_callee_vkeys_and_splices_L(self):
        root, root_section, decode, present, captures = self._build()
        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=2,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({100, 101, 102}),
            inlined_equivalent_call_targets_only=True,
        )
        # R -> K -> L all spliced (only because narrowing landed K's
        # selection on {200, 201}, which is a strict subset of K's
        # full variant set).
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 500])
        # K's body decoded once at J=0 (primary's J). L's body decoded
        # once at J=0 (K's primary's per_call_entry for L). The two
        # captures must be (400, 0) then (500, 0) in that order.
        assert captures == [(400, 0), (500, 0)]

    def test_flag_off_does_not_narrow_but_still_splices_L(self):
        """Flag OFF: no narrowing AND no inlining check. The walker
        falls through every layer of selection logic and splices L via
        standard iteration. This pins that flag OFF threads the
        ORIGINAL selection vkeys through unchanged (no narrowing) yet
        does not trip a skip.
        """
        root, root_section, decode, present, captures = self._build()
        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=2,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({100, 101, 102}),
            inlined_equivalent_call_targets_only=False,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400, 500])


# ---------------------------------------------------------------------------
# D6 fallback: flag ON + primary didn't call K, lowest v_idx in
# called_by_in_selection wins.
# ---------------------------------------------------------------------------


class TestD6FallbackLowestVIdx:
    """3-variant section where:
    * v=0 calls K with J=7.
    * v=1 calls K with J=8.
    * v=2 (primary) does NOT call K.

    All three vkeys are in the selection. Under flag ON, "some called,
    some didn't" so K is not skipped. Primary didn't call K -> D6
    fallback: lowest v_idx in called_by_in_selection = v=0 -> use
    v=0's J=7 for the callback.
    """

    def test_callback_receives_lowest_v_idx_J(self):
        callee = _make_staging(func_name="K", real_tokens=[400])
        callee_section = _StubSection(
            call_targets=[],
            variants=[
                _StubVariant(per_call_entries=[], variant_ref_offset=v)
                for v in range(10)
            ],
        )
        table = {200: (callee, callee_section)}
        captures: List[Tuple[int, int]] = []
        decode, present = _make_table_with_capture(table, captures=captures)
        root = _make_staging(func_name="root", real_tokens=[300])
        root_section = _StubSection(
            call_targets=[_StubCallTarget(200)],
            variants=[
                _StubVariant(
                    per_call_entries=[(0, 7)], variant_ref_offset=0
                ),
                _StubVariant(
                    per_call_entries=[(0, 8)], variant_ref_offset=1
                ),
                _StubVariant(
                    per_call_entries=[], variant_ref_offset=2
                ),
            ],
        )

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=2,
            initial_selection_vkeys=frozenset({0, 1, 2}),
            inlined_equivalent_call_targets_only=True,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])
        # Lowest v_idx in called_by ({0, 1}) is 0 -> its J=7.
        assert captures == [(200, 7)]


# ---------------------------------------------------------------------------
# Flag OFF + primary didn't call K: falls through to v_idx-wide
# fallback (level 3 of _choose_callee_variant). Selection is single-
# variant primary; called_by_in_selection ends up empty -> level 3.
# ---------------------------------------------------------------------------


class TestFlagOffFallbackToAllVariants:
    """A section where primary didn't call K but a non-selected
    variant did.

    * v=0 (primary, selected) does NOT call K.
    * v=1 (NOT selected) calls K with J=5.

    Selection = {primary's vkey only}. called_by_in_selection ends up
    empty (only primary is in selection, primary didn't call K).
    Under flag OFF the walker still iterates K (legacy behavior:
    iterate all call_targets). J is resolved via level-3 fallback:
    scan ALL variants in the section -> lowest v_idx that called K is
    v=1 -> J=5.
    """

    def test_level3_fallback_when_called_by_empty(self):
        callee = _make_staging(func_name="K", real_tokens=[400])
        callee_section = _StubSection(
            call_targets=[],
            variants=[
                _StubVariant(per_call_entries=[], variant_ref_offset=v)
                for v in range(10)
            ],
        )
        table = {200: (callee, callee_section)}
        captures: List[Tuple[int, int]] = []
        decode, present = _make_table_with_capture(table, captures=captures)
        root = _make_staging(func_name="root", real_tokens=[300])
        root_section = _StubSection(
            call_targets=[_StubCallTarget(200)],
            variants=[
                _StubVariant(
                    per_call_entries=[], variant_ref_offset=0
                ),
                _StubVariant(
                    per_call_entries=[(0, 5)], variant_ref_offset=1
                ),
            ],
        )

        out = splice_with_callees(
            root_staging=root,
            root_arm="matched",
            root_section=root_section,
            root_section_offset=100,
            decode_callee_to_staging=decode,
            is_callee_present=present,
            max_depth=1,
            primary_variant_idx=0,
            initial_selection_vkeys=frozenset({0}),
            inlined_equivalent_call_targets_only=False,
        )
        np.testing.assert_array_equal(out.real_tokens, [300, 400])
        # Level-3 fallback: only v=1 in the whole section called K, so
        # J=5.
        assert captures == [(200, 5)]
