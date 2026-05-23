"""Stage 1 integration tests -- :func:`walk_sections` end-to-end.

Single concern: assert that the three Phase-1 submodules (resolve,
layout, callee-walk) compose correctly into a :class:`Stage1Batch`,
covering each VariantPadding policy + the depth / cycle / inlining
filter knobs.

The fake session below combines the surfaces from
``test_resolve_pointers.py``'s 1a fake +
``test_callee_walk.py``'s 1b fake -- i.e. the union of the methods
1a + 1b + 1d touch on the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._batch_layout import UINT32_MAX
from tokenizer.aligned_data.loader.batch_decode._section_walk import (
    walk_sections,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    Stage1Batch,
    Stage1CallTarget,
    Stage1Section,
    Stage1Variant,
    VariantPadding,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.matched_function import MatchedFunction
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
    Section,
    VariantBlock,
)
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Fixture builders -- shared shapes for FunctionData / Section / call_targets
# ---------------------------------------------------------------------------


def _make_function_data(name: str) -> FunctionData:
    """A minimal valid :class:`FunctionData` -- one non-number token at
    position 0 satisfies :func:`build_inline_decode_state`'s
    ``run_lengths`` precondition (position 0 must not be a number-band
    token). The body shape is irrelevant for stage 1 -- the walker only
    cares about the section's call_targets + the body's load handles.
    """
    return FunctionData(
        func_name=name,
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.array([300], dtype=np.uint16),
        insn_runlength=np.array([1], dtype=np.uint32),
        block_runlength=np.array([1], dtype=np.uint32),
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _make_variant(
    *,
    vkey: int,
    per_call_entries: Optional[List[Tuple[int, int]]] = None,
) -> VariantBlock:
    return VariantBlock(
        variant_ref_offset=vkey,
        data_offset_shifted=0,
        per_call_entries=list(per_call_entries or []),
    )


def _make_section(
    *,
    section_offset: int,
    function_name_ptr: int,
    call_targets: Optional[List[CallTarget]] = None,
    variants: Optional[List[VariantBlock]] = None,
) -> Section:
    return Section(
        function_name_ptr=function_name_ptr,
        section_offset=section_offset,
        call_targets=list(call_targets or []),
        variants=list(variants or [_make_variant(vkey=0)]),
    )


def _ct_local(*, fid: int, target_offset: int) -> CallTarget:
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=target_offset,
        type=CallTargetType.LOCAL,
        is_matched=True,
    )


# ---------------------------------------------------------------------------
# Fake session combining 1a + 1b surfaces
# ---------------------------------------------------------------------------


@dataclass
class _FakeSession:
    """Combined fake session for the stage-1 wiring.

    Surfaces:

    * 1a path: ``_load_matched_section_and_variants(idx)``,
      ``_load_unmatched_record_and_section(idx)`` -- the variant-sampling
      + body-harvesting step returns the parsed section along with the
      per-arm body container (``MatchedFunction`` for matched, single
      ``FunctionData`` for unmatched), so the wiring does NOT re-issue
      the per-arm load to pick up the root variant body.
    * 1b path: ``_idx_for_section_offset(byte_offset, arm_str)``,
      ``_load_matched_for_splice(idx, variant_index)``,
      ``_load_unmatched_for_splice(idx)`` -- the DFS callee walk
      resolves each call_target row and loads the callee body via these
      helpers. The root body is no longer re-loaded here; 1d reads it
      from :attr:`ResolvedSection.function_data_per_sampled_variant`.

    The fake uses each section's own ``section_offset`` as its idx so
    ``_idx_for_section_offset`` is a trivial round-trip; the per-section
    FunctionData is keyed by ``(section_offset, variant_index)`` for
    matched and by ``section_offset`` for unmatched.
    """

    matched_sections: Dict[int, Section] = field(default_factory=dict)
    matched_function_data: Dict[Tuple[int, int], FunctionData] = field(
        default_factory=dict
    )
    unmatched_sections: Dict[int, Section] = field(default_factory=dict)
    unmatched_function_data: Dict[int, FunctionData] = field(
        default_factory=dict
    )

    # ---- registration -------------------------------------------------

    def add_matched(
        self,
        section: Section,
        variant_function_data: Dict[int, FunctionData],
    ) -> None:
        self.matched_sections[section.section_offset] = section
        for v_idx, fd in variant_function_data.items():
            self.matched_function_data[(section.section_offset, v_idx)] = fd

    def add_unmatched(self, section: Section, fd: FunctionData) -> None:
        self.unmatched_sections[section.section_offset] = section
        self.unmatched_function_data[section.section_offset] = fd

    # ---- 1a path ------------------------------------------------------

    def _load_matched_section_and_variants(
        self, idx: int
    ) -> Tuple[Section, int, MatchedFunction]:
        section = self.matched_sections[idx]
        # Build the variant list in the section's native variant-index
        # order, matching the real loader's contract
        # (``MatchedFunction.variants[v]`` is the v-th variant body).
        variants = [
            self.matched_function_data[(idx, v)]
            for v in range(len(section.variants))
        ]
        matched = MatchedFunction(func_name=f"m{idx}", variants=variants)
        return section, section.section_offset, matched

    def _load_unmatched_record_and_section(
        self, idx: int
    ) -> Tuple[Section, int, FunctionData]:
        section = self.unmatched_sections[idx]
        fd = self.unmatched_function_data[idx]
        return section, section.section_offset, fd

    # ---- 1b path ------------------------------------------------------

    def _idx_for_section_offset(
        self, section_offset: int, arm: str
    ) -> Optional[int]:
        if arm == "matched":
            return (
                section_offset
                if section_offset in self.matched_sections
                else None
            )
        if arm == "unmatched":
            return (
                section_offset
                if section_offset in self.unmatched_sections
                else None
            )
        raise ValueError(f"unknown arm: {arm!r}")

    def _load_matched_for_splice(
        self, idx: int, variant_index: int
    ) -> Tuple[FunctionData, Section, int]:
        section = self.matched_sections[idx]
        fd = self.matched_function_data[(idx, variant_index)]
        return fd, section, section.section_offset

    def _load_unmatched_for_splice(
        self, idx: int
    ) -> Tuple[FunctionData, Section, int]:
        section = self.unmatched_sections[idx]
        fd = self.unmatched_function_data[idx]
        return fd, section, section.section_offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0xC0FFEE) -> np.random.Generator:
    return np.random.default_rng(seed)


def _single_section_session_no_callees(
    *,
    section_offset: int = 100,
    function_name_ptr: int = 1,
    n_variants: int = 1,
) -> Tuple[_FakeSession, Section]:
    """Build a session containing one matched section with no call_targets
    + ``n_variants`` variants. Each variant gets a unique vkey + a
    distinct FunctionData."""
    section = _make_section(
        section_offset=section_offset,
        function_name_ptr=function_name_ptr,
        call_targets=[],
        variants=[
            _make_variant(vkey=10 + v) for v in range(n_variants)
        ],
    )
    session = _FakeSession()
    fds = {v: _make_function_data(f"root_v{v}") for v in range(n_variants)}
    session.add_matched(section, fds)
    return session, section


# ---------------------------------------------------------------------------
# 1. Single matched pointer, 1 variant, no callees
# ---------------------------------------------------------------------------


def test_single_matched_pointer_one_variant_no_callees() -> None:
    """End-to-end smoke: 1 section, 1 variant, no callees -> Stage1Batch
    with 1 section, 1 variant, 1 call_target (the root); mapping has 1
    row."""
    session, section = _single_section_session_no_callees(
        section_offset=64, function_name_ptr=7, n_variants=1
    )

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=64)],
        num_variants_per_section=1,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    assert isinstance(out, Stage1Batch)
    assert out.batch_size == 1
    assert out.batch_idx_to_section_variant.shape == (1, 2)
    np.testing.assert_array_equal(
        out.batch_idx_to_section_variant, np.array([[0, 0]], dtype=np.uint32)
    )

    assert len(out.sections) == 1
    st1_sec = out.sections[0]
    assert isinstance(st1_sec, Stage1Section)
    assert st1_sec.arm is SectionKind.MATCHED
    assert st1_sec.idx == 64
    assert st1_sec.section is section

    assert len(st1_sec.variants) == 1
    st1_var = st1_sec.variants[0]
    assert isinstance(st1_var, Stage1Variant)
    assert st1_var.variant_idx == 0
    assert st1_var.variant_ref_offset == 10  # vkey 10 from _make_variant
    assert st1_var.batch_idx == 0

    assert len(st1_var.call_targets) == 1
    root_ct = st1_var.call_targets[0]
    assert isinstance(root_ct, Stage1CallTarget)
    assert root_ct.encounter_category is Category.LOCAL_FUNC
    assert root_ct.parent_call_target_index is None
    assert root_ct.function_name_ptr == 7


# ---------------------------------------------------------------------------
# 2. Single matched pointer, multiple variants -> all variants present
# ---------------------------------------------------------------------------


def test_single_pointer_multiple_variants_all_present() -> None:
    """Sample every variant; each :class:`Stage1Variant` carries a
    correct, unique ``batch_idx`` in [0, nv)."""
    session, _section = _single_section_session_no_callees(
        section_offset=200, function_name_ptr=42, n_variants=3
    )

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=200)],
        num_variants_per_section=3,
        max_depth=1,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    assert out.batch_size == 3
    np.testing.assert_array_equal(
        out.batch_idx_to_section_variant,
        np.array([[0, 0], [0, 1], [0, 2]], dtype=np.uint32),
    )
    st1_sec = out.sections[0]
    assert len(st1_sec.variants) == 3
    batch_idxs = [v.batch_idx for v in st1_sec.variants]
    assert batch_idxs == [0, 1, 2]
    # vkeys 10, 11, 12 in encounter order.
    vkeys = [v.variant_ref_offset for v in st1_sec.variants]
    assert vkeys == [10, 11, 12]


# ---------------------------------------------------------------------------
# 3. Multiple pointers, mixed matched + unmatched -> ordering preserved
# ---------------------------------------------------------------------------


def test_multiple_pointers_mixed_arms_preserve_order() -> None:
    """``out.sections[i]`` corresponds to ``section_pointers[i]``."""
    matched_section = _make_section(
        section_offset=100, function_name_ptr=1, variants=[_make_variant(vkey=0)]
    )
    unmatched_section = _make_section(
        section_offset=200, function_name_ptr=2, variants=[_make_variant(vkey=0)]
    )
    session = _FakeSession()
    session.add_matched(matched_section, {0: _make_function_data("m")})
    session.add_unmatched(unmatched_section, _make_function_data("u"))

    pointers = [
        SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=200),
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=100),
    ]
    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=pointers,
        num_variants_per_section=1,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    assert [(s.arm, s.idx) for s in out.sections] == [
        (SectionKind.UNMATCHED, 200),
        (SectionKind.MATCHED, 100),
    ]
    # And the section objects line up.
    assert out.sections[0].section is unmatched_section
    assert out.sections[1].section is matched_section
    # Each variant's root call_target carries the matching FID.
    assert out.sections[0].variants[0].call_targets[0].function_name_ptr == 2
    assert out.sections[1].variants[0].call_targets[0].function_name_ptr == 1


# ---------------------------------------------------------------------------
# 4. Root with LOCAL callees at depth 1
# ---------------------------------------------------------------------------


def test_root_with_local_callees_depth_one() -> None:
    """Root + 2 LOCAL callees in encounter order -> variant.call_targets
    has length 3; indices 1+ point at the right FIDs."""
    callee_a = _make_section(section_offset=200, function_name_ptr=2)
    callee_b = _make_section(section_offset=300, function_name_ptr=3)
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[
            _ct_local(fid=2, target_offset=200),  # called_idx=0
            _ct_local(fid=3, target_offset=300),  # called_idx=1
        ],
        variants=[
            _make_variant(vkey=0, per_call_entries=[(0, 0), (1, 0)]),
        ],
    )
    session = _FakeSession()
    session.add_matched(root_section, {0: _make_function_data("root")})
    session.add_matched(callee_a, {0: _make_function_data("a")})
    session.add_matched(callee_b, {0: _make_function_data("b")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)],
        num_variants_per_section=1,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    cts = out.sections[0].variants[0].call_targets
    assert [c.function_name_ptr for c in cts] == [1, 2, 3]
    assert cts[0].parent_call_target_index is None
    assert cts[1].parent_call_target_index == 0
    assert cts[2].parent_call_target_index == 1


# ---------------------------------------------------------------------------
# 5. max_depth=0 -> no callees recursed
# ---------------------------------------------------------------------------


def test_max_depth_zero_no_callee_recursion() -> None:
    """``max_depth=0`` returns only the root, even when callees exist."""
    callee = _make_section(section_offset=200, function_name_ptr=2)
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=[_make_variant(vkey=0, per_call_entries=[(0, 0)])],
    )
    session = _FakeSession()
    session.add_matched(root_section, {0: _make_function_data("root")})
    session.add_matched(callee, {0: _make_function_data("callee")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)],
        num_variants_per_section=1,
        max_depth=0,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    cts = out.sections[0].variants[0].call_targets
    assert len(cts) == 1
    assert cts[0].function_name_ptr == 1


# ---------------------------------------------------------------------------
# 6. inlined_equivalent_call_targets_only=True -> filter applied
# ---------------------------------------------------------------------------


def test_inlined_equivalent_filter_is_threaded_through() -> None:
    """The filter is forwarded to the callee walker. We construct a
    setup where the filter prunes a callee that the unfiltered run
    would include, and assert the difference."""
    # Two callees; root has 2 variants:
    #   variant 0 calls callee 0 only
    #   variant 1 calls callees 0 and 1
    # => called_idx=0 (fid=2) by {0, 1} (ALL)  -> filter skips
    # => called_idx=1 (fid=3) by {1}     (SOME) -> filter keeps
    callee_a = _make_section(section_offset=200, function_name_ptr=2)
    callee_b = _make_section(section_offset=300, function_name_ptr=3)
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[
            _ct_local(fid=2, target_offset=200),
            _ct_local(fid=3, target_offset=300),
        ],
        variants=[
            _make_variant(vkey=0, per_call_entries=[(0, 0)]),
            _make_variant(vkey=1, per_call_entries=[(0, 0), (1, 0)]),
        ],
    )
    session = _FakeSession()
    session.add_matched(
        root_section,
        {0: _make_function_data("root_v0"), 1: _make_function_data("root_v1")},
    )
    session.add_matched(callee_a, {0: _make_function_data("a")})
    session.add_matched(callee_b, {0: _make_function_data("b")})

    # Unfiltered: variant 0's root has both call_targets visible.
    unfiltered = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)],
        num_variants_per_section=2,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )
    fids_v0_unfiltered = [
        c.function_name_ptr for c in unfiltered.sections[0].variants[0].call_targets
    ]
    assert fids_v0_unfiltered == [1, 2, 3]

    # Filtered: called_idx=0 dropped (ALL), called_idx=1 kept (SOME).
    filtered = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)],
        num_variants_per_section=2,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=True,
        rng=_rng(),
    )
    fids_v0_filtered = [
        c.function_name_ptr for c in filtered.sections[0].variants[0].call_targets
    ]
    # Root + callee 3 only; callee 2 dropped (ALL variants called it).
    assert fids_v0_filtered == [1, 3]


# ---------------------------------------------------------------------------
# 7. PAD_NULL -> short sections produce padding rows
# ---------------------------------------------------------------------------


def test_pad_null_short_section_padding_rows() -> None:
    """With a short section + PAD_NULL, the layout has sentinel rows but
    the variants list reflects ONLY the real sampled variants -- no
    None-padded entries on :class:`Stage1Section.variants`."""
    short_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        variants=[_make_variant(vkey=10)],  # only 1 variant
    )
    full_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        variants=[
            _make_variant(vkey=20),
            _make_variant(vkey=21),
        ],
    )
    session = _FakeSession()
    session.add_matched(short_section, {0: _make_function_data("short")})
    session.add_matched(
        full_section,
        {0: _make_function_data("full_v0"), 1: _make_function_data("full_v1")},
    )

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=100),
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=200),
        ],
        num_variants_per_section=2,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    # Mapping is 4 rows: row 1 is the padding sentinel for section 0.
    assert out.batch_size == 4
    expected_map = np.array(
        [
            [0, 0],
            [UINT32_MAX, UINT32_MAX],
            [1, 0],
            [1, 1],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(out.batch_idx_to_section_variant, expected_map)

    # Section 0 has only 1 real variant in its variants list (no None).
    assert len(out.sections[0].variants) == 1
    assert out.sections[0].variants[0].batch_idx == 0
    # Section 1 has 2 real variants.
    assert len(out.sections[1].variants) == 2
    assert out.sections[1].variants[0].batch_idx == 2
    assert out.sections[1].variants[1].batch_idx == 3


# ---------------------------------------------------------------------------
# 8. RAGGED -> batch_size matches total real variants
# ---------------------------------------------------------------------------


def test_ragged_dense_no_padding() -> None:
    """``batch_size == total_real_variants``; no sentinel rows."""
    section_a = _make_section(
        section_offset=100,
        function_name_ptr=1,
        variants=[_make_variant(vkey=v) for v in (10, 11)],
    )
    section_b = _make_section(
        section_offset=200,
        function_name_ptr=2,
        variants=[_make_variant(vkey=20)],
    )
    session = _FakeSession()
    session.add_matched(
        section_a,
        {0: _make_function_data("a0"), 1: _make_function_data("a1")},
    )
    session.add_matched(section_b, {0: _make_function_data("b")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=100),
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=200),
        ],
        num_variants_per_section=2,
        max_depth=1,
        variant_padding=VariantPadding.RAGGED,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    assert out.batch_size == 3
    assert not (out.batch_idx_to_section_variant == UINT32_MAX).any()
    # batch_idx assignment: section 0 -> rows 0+1; section 1 -> row 2.
    assert [v.batch_idx for v in out.sections[0].variants] == [0, 1]
    assert [v.batch_idx for v in out.sections[1].variants] == [2]


# ---------------------------------------------------------------------------
# 9. RNG reproducibility -- same seed yields same Stage1Batch shape
# ---------------------------------------------------------------------------


def test_rng_reproducibility_same_seed() -> None:
    """Two runs with freshly-seeded RNGs produce identical sampled
    variant_idx + identical batch_idx_to_section_variant mappings."""
    # Many variants to force undersampling -> RNG-driven choice.
    section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        variants=[_make_variant(vkey=10 + v) for v in range(10)],
    )
    fds = {v: _make_function_data(f"v{v}") for v in range(10)}
    session = _FakeSession()
    session.add_matched(section, fds)

    pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)]
    kw = dict(
        num_variants_per_section=4,
        max_depth=1,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
    )

    out_a = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=pointers,
        rng=_rng(seed=42),
        **kw,
    )
    out_b = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=pointers,
        rng=_rng(seed=42),
        **kw,
    )

    np.testing.assert_array_equal(
        out_a.batch_idx_to_section_variant, out_b.batch_idx_to_section_variant
    )
    assert out_a.batch_size == out_b.batch_size
    a_vidx = [v.variant_idx for v in out_a.sections[0].variants]
    b_vidx = [v.variant_idx for v in out_b.sections[0].variants]
    assert a_vidx == b_vidx
    # Sanity: it was actually a random sample, not the cover-all path.
    assert len(a_vidx) == 4


# ---------------------------------------------------------------------------
# 10. Cycle in callee tree -> handled, no infinite loop
# ---------------------------------------------------------------------------


def test_cycle_in_callee_tree_terminates() -> None:
    """Mutual A <-> B recursion terminates with the active-path cycle
    guard (verified by :func:`walk_callees`'s tests; this integration
    test pins the contract that :func:`walk_sections` does NOT defeat
    it)."""
    a_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=[_make_variant(vkey=0, per_call_entries=[(0, 0)])],
    )
    b_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        call_targets=[_ct_local(fid=1, target_offset=100)],
        variants=[_make_variant(vkey=0, per_call_entries=[(0, 0)])],
    )
    session = _FakeSession()
    session.add_matched(a_section, {0: _make_function_data("a")})
    session.add_matched(b_section, {0: _make_function_data("b")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)],
        num_variants_per_section=1,
        max_depth=10,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )
    fids = [
        c.function_name_ptr
        for c in out.sections[0].variants[0].call_targets
    ]
    assert fids == [1, 2]


# ---------------------------------------------------------------------------
# Additional invariants for the composition itself
# ---------------------------------------------------------------------------


def test_empty_section_pointers_yields_empty_batch() -> None:
    """No pointers -> empty mapping + empty sections list."""
    session = _FakeSession()
    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )
    assert out.batch_size == 0
    assert out.batch_idx_to_section_variant.shape == (0, 2)
    assert out.sections == []


def test_unmatched_root_uses_unmatched_loader() -> None:
    """An UNMATCHED pointer must end up routed through
    ``_load_unmatched_for_splice`` for the variant body re-load -- i.e.
    the unmatched FunctionData ends up on the root call_target."""
    section = _make_section(
        section_offset=100,
        function_name_ptr=9,
        variants=[_make_variant(vkey=0)],
    )
    unmatched_fd = _make_function_data("unmatched_root")
    session = _FakeSession()
    session.add_unmatched(section, unmatched_fd)

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[
            SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=100)
        ],
        num_variants_per_section=1,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )
    root_ct = out.sections[0].variants[0].call_targets[0]
    assert root_ct.function_data is unmatched_fd
    assert root_ct.function_name_ptr == 9


def test_resample_records_first_batch_row_for_multi_mapped_slot() -> None:
    """Per the module docstring: when RESAMPLE maps multiple batch rows
    to the same ``(section_idx, slot_v)``, ``Stage1Variant.batch_idx``
    holds the FIRST matching row. The downstream walker recovers the
    other rows by scanning ``batch_idx_to_section_variant`` directly.
    """
    section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        variants=[_make_variant(vkey=10)],
    )
    session = _FakeSession()
    session.add_matched(section, {0: _make_function_data("root")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=100)
        ],
        num_variants_per_section=3,
        max_depth=1,
        variant_padding=VariantPadding.RESAMPLE_WITHIN_SECTION,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    # The section has 1 real variant; nv=3 => 2 resampled slots all
    # collapsing to (0, 0).
    assert out.batch_size == 3
    np.testing.assert_array_equal(
        out.batch_idx_to_section_variant,
        np.array([[0, 0], [0, 0], [0, 0]], dtype=np.uint32),
    )
    # Only 1 real Stage1Variant (the 1 sampled variant), and its
    # batch_idx is the FIRST matching row -- row 0.
    st1_sec = out.sections[0]
    assert len(st1_sec.variants) == 1
    assert st1_sec.variants[0].batch_idx == 0
    # The OTHER matching rows (1, 2) are discoverable via the mapping
    # directly.
    other_rows = np.flatnonzero(
        (out.batch_idx_to_section_variant[:, 0] == 0)
        & (out.batch_idx_to_section_variant[:, 1] == 0)
    )
    assert list(other_rows) == [0, 1, 2]


def test_redistribute_policy_threaded_through() -> None:
    """REDISTRIBUTE policy is threaded through to the layout module --
    we observe the policy's specific shape (linear-by-section with
    deficits potentially filled by donors).

    Note: ``_select_variant_indices`` caps each section's sampled list
    at ``num_variants_per_section``, so 1d's flow can never produce
    surplus donors. With every section sampling at most ``nv``, the
    REDISTRIBUTE shape degenerates to PAD_NULL semantics: short sections
    keep their UINT32_MAX rows. The layout module is shape-preserving
    regardless; this test pins the integration's correct dispatch.
    """
    section_a = _make_section(
        section_offset=100,
        function_name_ptr=1,
        variants=[_make_variant(vkey=10 + v) for v in range(2)],
    )
    section_b = _make_section(
        section_offset=200,
        function_name_ptr=2,
        variants=[_make_variant(vkey=20)],
    )
    session = _FakeSession()
    session.add_matched(
        section_a,
        {v: _make_function_data(f"a_v{v}") for v in range(2)},
    )
    session.add_matched(section_b, {0: _make_function_data("b")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=100),
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=200),
        ],
        num_variants_per_section=2,
        max_depth=1,
        variant_padding=VariantPadding.REDISTRIBUTE,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )

    # 2 sections * nv=2 = 4 rows; section A fills (0,0)+(0,1), section B
    # has 1 real + 1 deficit. No donors -> deficit stays UINT32_MAX.
    assert out.batch_size == 4
    # Stage1Variants only enumerate the REAL sampled entries:
    assert [v.batch_idx for v in out.sections[0].variants] == [0, 1]
    assert [v.batch_idx for v in out.sections[1].variants] == [2]
    # Last row is the deficit sentinel (no donor available).
    assert int(out.batch_idx_to_section_variant[3, 0]) == int(UINT32_MAX)
    assert int(out.batch_idx_to_section_variant[3, 1]) == int(UINT32_MAX)


def test_root_function_name_ptr_taken_from_section_header() -> None:
    """The root call_target's ``function_name_ptr`` must come from
    :attr:`Section.function_name_ptr` -- the section's own FID -- not
    from a per-variant value."""
    section = _make_section(
        section_offset=300,
        function_name_ptr=999,
        variants=[_make_variant(vkey=0)],
    )
    session = _FakeSession()
    session.add_matched(section, {0: _make_function_data("the_root")})

    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=300)
        ],
        num_variants_per_section=1,
        max_depth=1,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )
    root_ct = out.sections[0].variants[0].call_targets[0]
    assert root_ct.function_name_ptr == 999


def test_stage1batch_dataclass_shape_matches_plan() -> None:
    """Defensive: the produced :class:`Stage1Batch` must populate every
    required field. Catches a regression where the wiring forgets to
    plumb e.g. ``batch_size`` or the mapping."""
    session, _ = _single_section_session_no_callees(
        section_offset=10, function_name_ptr=1, n_variants=1
    )
    out = walk_sections(
        session=session,  # type: ignore[arg-type]
        section_pointers=[SectionPointerSpec(arm=SectionKind.MATCHED, idx=10)],
        num_variants_per_section=1,
        max_depth=1,
        variant_padding=VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only=False,
        rng=_rng(),
    )
    assert hasattr(out, "sections")
    assert hasattr(out, "batch_idx_to_section_variant")
    assert hasattr(out, "batch_size")
    assert out.batch_size == out.batch_idx_to_section_variant.shape[0]
    assert out.batch_idx_to_section_variant.dtype == np.uint32


def test_negative_max_depth_rejected_via_callee_walker() -> None:
    """The wiring delegates ``max_depth`` enforcement to
    :func:`walk_callees`; a negative value must surface as the
    walker's :class:`ValueError`."""
    session, _ = _single_section_session_no_callees(
        section_offset=10, function_name_ptr=1, n_variants=1
    )
    with pytest.raises(ValueError, match="max_depth"):
        walk_sections(
            session=session,  # type: ignore[arg-type]
            section_pointers=[
                SectionPointerSpec(arm=SectionKind.MATCHED, idx=10)
            ],
            num_variants_per_section=1,
            max_depth=-1,
            variant_padding=VariantPadding.PAD_NULL,
            inlined_equivalent_call_targets_only=False,
            rng=_rng(),
        )
