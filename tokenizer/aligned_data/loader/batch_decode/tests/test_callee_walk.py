"""Unit tests for the stage-1 level-synchronous callee walker.

Single concern: validate the BFS encounter order + once-only dedup
+ all-variants-equivalence exclusion + max_depth cap on
:func:`tokenizer.aligned_data.loader.batch_decode._callee_walk.walk_callees`.

Every test builds a synthetic call graph via the :class:`_FakeSession`
fixture below: a section registry keyed by ``section_offset`` so the
walker's ``_idx_for_section_offset`` ->
``_load_matched_section_and_variants``
round-trip resolves to caller-supplied :class:`Section` +
:class:`FunctionData` pairs. No real binary memmap is touched.

Because the once-only mask spans a section's FULL variant set and
excludes any callee reached by EVERY variant, a single-variant section
splices nothing (FLAG-A). The resolution-mechanics tests therefore pair
each caller variant with a quiet sibling (:func:`_calling_variant`) so
variant 0's callees are "reached by some but not all" and get included;
``walk_callees`` returns variant 0's list.

These tests pin the walker's observable contract under the owner's
once-only + all-variants-equivalence spec (supersedes the legacy
active-path DAG semantics):

1. Single-function root (no callees) -- length 1, root is LOCAL_FUNC,
   parent_call_target_index None.
2. Root + 1 LOCAL callee -- length 2, callee LOCAL_FUNC.
3. Root + 1 PLT callee -- callee PLT_FUNC.
4. EXT callee -- skipped (EXT_FUNC bodies not inlined per plan D3).
5. max_depth=0 -- root only.
6. max_depth=1 -- root + direct callees; grandchildren skipped.
7. Cycle A -> B -> A -- output [A, B]; the recursive edge back to an
   already-included section is a once-only dedup no-op.
8. Diamond A -> {B, C}; B -> D; C -> D -- output [A, B, C, D] in BFS
   level order; D is deduped to ONE inclusion (the legacy active-path
   walk emitted it twice as [A, B, D, C, D]).
9. all-variants-equivalence -- a callee every variant of a section
   called is excluded; a variant includes only callees IT directly
   called (its per-call entries).
10. parent_call_target_index correctness -- non-root entries index
    into the PARENT's ``call_targets_section``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._callee_walk import (
    walk_callees,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
    Section,
    VariantBlock,
)
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Section + FunctionData fixture builders
# ---------------------------------------------------------------------------


def _make_function_data(
    name: str,
    *,
    variant_tokens: Optional[np.ndarray] = None,
) -> FunctionData:
    # ``build_inline_decode_state``'s ``run_lengths`` precondition
    # requires position 0 to be False (i.e. not a number-band token);
    # a single non-number token (>= 272 -- past the eager block in the
    # unified vocab) is the minimal valid stream that satisfies the
    # precondition without exercising any inline-byte / sign logic the
    # walker doesn't care about.
    if variant_tokens is None:
        variant_tokens = np.zeros(0, dtype=np.uint16)
    return FunctionData(
        func_name=name,
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.array([300], dtype=np.uint16),
        insn_runlength=np.array([1], dtype=np.uint32),
        block_runlength=np.array([1], dtype=np.uint32),
        variant_tokens=variant_tokens,
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


def _calling_variant(per_call_entries: List[Tuple[int, int]]) -> List[VariantBlock]:
    """A caller variant (variant 0) PLUS a quiet sibling (variant 1).

    The once-only / all-variants-equivalence walk excludes a callee
    reached by EVERY variant of the section. A single-variant section
    therefore splices nothing (FLAG-A). These resolution-mechanics tests
    want variant 0's calls to actually splice, so they pair the caller
    with a quiet sibling that calls nothing -- making variant 0's callees
    "reached by SOME but not all" and hence included. ``walk_callees``
    returns variant 0's list.
    """
    return [
        _make_variant(vkey=0, per_call_entries=per_call_entries),
        _make_variant(vkey=1, per_call_entries=[]),
    ]


def _ct_local(*, fid: int, target_offset: int) -> CallTarget:
    """LOCAL call_target row pointing at ``target_offset``."""
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=target_offset,
        type=CallTargetType.LOCAL,
        is_matched=True,
    )


def _ct_plt(*, fid: int, target_offset: int) -> CallTarget:
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=target_offset,
        type=CallTargetType.PLT,
        is_matched=True,
    )


def _ct_extern(*, fid: int, target_offset: int) -> CallTarget:
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=target_offset,
        type=CallTargetType.EXTERN,
        is_matched=False,
    )


# ---------------------------------------------------------------------------
# Fake BinarySession -- only the methods walk_callees actually calls.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeMatched:
    """Matched-load result stub: the walker indexes ``.variants`` with
    the table-size-validated variant choice."""

    variants: List[Optional[FunctionData]]


@dataclass
class _FakeSession:
    """Minimal session stub keyed by ``section_offset``.

    The walker reaches into the session via three methods:
    ``_idx_for_section_offset(byte_offset, arm_str)``,
    ``_load_matched_section_and_variants(idx)``, and
    ``_load_unmatched_for_splice(idx)`` (plus the ``_binary_name``
    identity attribute for demotion inventory logs). The fake uses the
    section's own ``section_offset`` as its idx so the round-trip is
    trivially invertible.

    Per-section ``FunctionData`` is keyed by ``(section_offset,
    variant_index)`` for matched and by ``section_offset`` for
    unmatched (which has exactly one variant by construction).
    """

    _binary_name: str = "fake-bin"
    matched_sections: Dict[int, Section] = field(default_factory=dict)
    matched_function_data: Dict[Tuple[int, int], FunctionData] = field(
        default_factory=dict
    )
    unmatched_sections: Dict[int, Section] = field(default_factory=dict)
    unmatched_function_data: Dict[int, FunctionData] = field(
        default_factory=dict
    )

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

    # --- session-API surface the walker invokes -------------------------

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

    def _load_matched_section_and_variants(
        self, idx: int
    ) -> Tuple[Section, int, "_FakeMatched"]:
        section = self.matched_sections[idx]
        variants = [
            self.matched_function_data.get((idx, v))
            for v in range(len(section.variants))
        ]
        return section, section.section_offset, _FakeMatched(variants)

    def _matched_section_meta(self, idx: int) -> Tuple[Section, int]:
        section = self.matched_sections[idx]
        return section, section.section_offset

    def _load_matched_variant_body(
        self, idx: int, variant_index: int
    ) -> FunctionData:
        return self.matched_function_data[(idx, variant_index)]

    def _unmatched_section_meta(self, idx: int) -> Tuple[Section, int]:
        section = self.unmatched_sections[idx]
        return section, section.section_offset

    def _load_unmatched_for_splice(
        self, idx: int
    ) -> Tuple[FunctionData, Section, int]:
        section = self.unmatched_sections[idx]
        fd = self.unmatched_function_data[idx]
        return fd, section, section.section_offset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_only_no_callees() -> None:
    """Single-function root with no call_targets -- output is length 1,
    LOCAL_FUNC, parent_call_target_index is None."""
    root_section = _make_section(section_offset=100, function_name_ptr=1)
    root_fd = _make_function_data("root")
    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 1
    assert out[0].encounter_category is Category.LOCAL_FUNC
    assert out[0].parent_call_target_index is None
    assert out[0].function_name_ptr == 1
    assert out[0].function_data is root_fd


def test_one_local_callee_at_depth_1() -> None:
    callee_section = _make_section(section_offset=200, function_name_ptr=2)
    callee_fd = _make_function_data("callee")
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")

    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(callee_section, {0: callee_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 2
    assert out[1].encounter_category is Category.LOCAL_FUNC
    assert out[1].parent_call_target_index == 0
    assert out[1].function_name_ptr == 2
    assert out[1].function_data is callee_fd


def test_one_plt_callee_yields_plt_func_category() -> None:
    callee_section = _make_section(section_offset=200, function_name_ptr=2)
    callee_fd = _make_function_data("plt_callee")
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_plt(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")

    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(callee_section, {0: callee_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 2
    assert out[1].encounter_category is Category.PLT_FUNC


def test_extern_callee_is_skipped() -> None:
    """EXT_FUNC bodies are not inlined per plan D3 -- extern call_targets
    must not appear in the walker output even if a target_offset is
    nominally available in the session."""
    # NOTE: extern call_targets carry function_section_ptr == 0 in
    # production (there's no body), but the walker's EXTERN check fires
    # BEFORE the resolved-pointer check, so a defensive non-zero offset
    # value still must NOT inline. We register a section at the same
    # offset to prove the walker's EXTERN guard is the gate that
    # blocks the splice.
    ghost_section = _make_section(section_offset=200, function_name_ptr=2)
    ghost_fd = _make_function_data("ghost_extern")
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_extern(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")

    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(ghost_section, {0: ghost_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 1
    assert out[0].function_name_ptr == 1


def test_max_depth_zero_returns_root_only() -> None:
    callee_section = _make_section(section_offset=200, function_name_ptr=2)
    callee_fd = _make_function_data("callee")
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")

    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(callee_section, {0: callee_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=0,
    )

    assert len(out) == 1
    assert out[0].function_name_ptr == 1


def test_max_depth_one_includes_direct_callees_only() -> None:
    grandchild_section = _make_section(
        section_offset=300, function_name_ptr=3
    )
    grandchild_fd = _make_function_data("gc")
    child_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        call_targets=[_ct_local(fid=3, target_offset=300)],
        variants=_calling_variant([(0, 0)]),
    )
    child_fd = _make_function_data("child")
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")

    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(child_section, {0: child_fd})
    session.add_matched(grandchild_section, {0: grandchild_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=1,
    )

    # Root + 1 direct callee; grandchild NOT included.
    assert len(out) == 2
    fids = [entry.function_name_ptr for entry in out]
    assert fids == [1, 2]


def test_cycle_a_b_a_deduped_by_once_only_mask() -> None:
    """A -> B -> A: the recursive edge back to A is a once-only dedup
    no-op -- A's section was seeded at the mask's column 0 (the root is
    always included once), so B's call back to A re-marks an
    already-True cell and includes nothing. Output [A, B]."""
    # Section A points at B; section B points back at A.
    a_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    b_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        call_targets=[_ct_local(fid=1, target_offset=100)],
        variants=_calling_variant([(0, 0)]),
    )

    a_fd = _make_function_data("a")
    b_fd = _make_function_data("b")
    session = _FakeSession()
    session.add_matched(a_section, {0: a_fd})
    session.add_matched(b_section, {0: b_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=a_section,
        root_variant_idx=0,
        root_function_data=a_fd,
        root_function_name_ptr=1,
        max_depth=10,
    )

    # [A, B]; B's call back to A is deduped (A already at mask col 0).
    assert [e.function_name_ptr for e in out] == [1, 2]


def test_dag_a_b_d_and_a_c_d_dedups_d_to_once() -> None:
    """Once-only semantics (owner's spec, supersedes plan-D3 DAG): the
    diamond A -> B -> D and A -> C -> D includes D ONCE per variant, not
    twice. BFS level order: [A(0), B(1), C(1), D(2)] -- D is deduped
    across the two branches (the legacy active-path walk emitted it
    twice in DFS order [A, B, D, C, D])."""
    d_section = _make_section(section_offset=400, function_name_ptr=4)
    d_fd = _make_function_data("d")
    b_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        call_targets=[_ct_local(fid=4, target_offset=400)],
        variants=_calling_variant([(0, 0)]),
    )
    c_section = _make_section(
        section_offset=300,
        function_name_ptr=3,
        call_targets=[_ct_local(fid=4, target_offset=400)],
        variants=_calling_variant([(0, 0)]),
    )
    a_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[
            _ct_local(fid=2, target_offset=200),
            _ct_local(fid=3, target_offset=300),
        ],
        variants=_calling_variant([(0, 0), (1, 0)]),
    )
    a_fd = _make_function_data("a")
    b_fd = _make_function_data("b")
    c_fd = _make_function_data("c")

    session = _FakeSession()
    session.add_matched(a_section, {0: a_fd})
    session.add_matched(b_section, {0: b_fd})
    session.add_matched(c_section, {0: c_fd})
    session.add_matched(d_section, {0: d_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=a_section,
        root_variant_idx=0,
        root_function_data=a_fd,
        root_function_name_ptr=1,
        max_depth=10,
    )

    # BFS level order, D deduped to one occurrence.
    assert [e.function_name_ptr for e in out] == [1, 2, 3, 4]
    assert out[3].function_data is d_fd


def test_path_depth_tracks_bfs_level() -> None:
    """``path_depth`` is the BFS level: root 0, callee +1 per level.

    The DAG ``A -> {B, C}; B -> D; C -> D`` flattens in BFS order
    ``[A, B, C, D]`` (D deduped). The matching path-depths are
    ``[0, 1, 1, 2]`` -- the root at 0, direct callees B/C at 1, and the
    once-included grandchild D at 2. This pins the property that makes
    one max-depth walk serve every shallower depth: a call_target
    belongs to the depth-``k`` expansion iff its ``path_depth <= k``."""
    d_section = _make_section(section_offset=400, function_name_ptr=4)
    d_fd = _make_function_data("d")
    b_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        call_targets=[_ct_local(fid=4, target_offset=400)],
        variants=_calling_variant([(0, 0)]),
    )
    c_section = _make_section(
        section_offset=300,
        function_name_ptr=3,
        call_targets=[_ct_local(fid=4, target_offset=400)],
        variants=_calling_variant([(0, 0)]),
    )
    a_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[
            _ct_local(fid=2, target_offset=200),
            _ct_local(fid=3, target_offset=300),
        ],
        variants=_calling_variant([(0, 0), (1, 0)]),
    )
    a_fd = _make_function_data("a")
    session = _FakeSession()
    session.add_matched(a_section, {0: a_fd})
    session.add_matched(b_section, {0: _make_function_data("b")})
    session.add_matched(c_section, {0: _make_function_data("c")})
    session.add_matched(d_section, {0: d_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=a_section,
        root_variant_idx=0,
        root_function_data=a_fd,
        root_function_name_ptr=1,
        max_depth=10,
    )

    assert [e.function_name_ptr for e in out] == [1, 2, 3, 4]
    assert [e.path_depth for e in out] == [0, 1, 1, 2]


def test_max_depth_one_caps_path_depth_at_one() -> None:
    """At ``max_depth=1`` every emitted call_target has ``path_depth <= 1``.

    The depth-cap prunes at ``current_depth >= max_depth``, so the
    depth-1 walk emits exactly the root (0) + direct callees (1) and no
    deeper rows -- the prefix property the multi-depth build relies on."""
    grandchild_section = _make_section(
        section_offset=300, function_name_ptr=3
    )
    child_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        call_targets=[_ct_local(fid=3, target_offset=300)],
        variants=_calling_variant([(0, 0)]),
    )
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")
    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(child_section, {0: _make_function_data("child")})
    session.add_matched(
        grandchild_section, {0: _make_function_data("gc")}
    )

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=1,
    )

    assert [e.path_depth for e in out] == [0, 1]
    assert max(e.path_depth for e in out) <= 1


def test_all_variants_equivalence_excludes_and_direct_calls_only() -> None:
    """Once-only + all-variants-equivalence, the default-and-only walk:

    A variant includes a callee iff it DIRECTLY called it (its own
    per-call entries) AND the callee was NOT reached by every variant
    (columnwise-ALL exclusion). The returned list is the requested
    variant's; ``walk_callees(root_variant_idx=0)`` returns variant 0's
    inclusions -- callees variant 0 itself called, minus the ones every
    variant called.
    """
    # 4 callees; parent section has 3 variants. Per-call-entry sets:
    #   variant 0 calls callees [0, 1]
    #   variant 1 calls callees [0]
    #   variant 2 calls callees [0, 1, 2]
    # variant-0 perspective (the returned list):
    # -> called_idx=0 by {0,1,2} (ALL)   -> EXCLUDED (equivalence)
    # -> called_idx=1 by {0, 2}          -> v0 called it, not all -> INCLUDE
    # -> called_idx=2 by {2} only        -> v0 did NOT call it -> absent
    # -> called_idx=3 by {}              -> nobody called it -> absent
    callee_sections = {
        offset: _make_section(section_offset=offset, function_name_ptr=fid)
        for offset, fid in [(200, 2), (300, 3), (400, 4), (500, 5)]
    }
    parent_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[
            _ct_local(fid=2, target_offset=200),  # called_idx=0
            _ct_local(fid=3, target_offset=300),  # called_idx=1
            _ct_local(fid=4, target_offset=400),  # called_idx=2
            _ct_local(fid=5, target_offset=500),  # called_idx=3
        ],
        variants=[
            _make_variant(vkey=10, per_call_entries=[(0, 0), (1, 0)]),
            _make_variant(vkey=11, per_call_entries=[(0, 0)]),
            _make_variant(
                vkey=12,
                per_call_entries=[(0, 0), (1, 0), (2, 0)],
            ),
        ],
    )
    parent_fd = _make_function_data("parent")

    session = _FakeSession()
    session.add_matched(parent_section, {0: parent_fd})
    for offset, section in callee_sections.items():
        session.add_matched(section, {0: _make_function_data(f"c@{offset}")})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=parent_section,
        root_variant_idx=0,
        root_function_data=parent_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    # Root + the single callee variant 0 directly called that isn't
    # all-variants-excluded: called_idx=1 (fid=3). called_idx=0 is
    # excluded (all called); called_idx=2/3 variant 0 never called.
    fids = [e.function_name_ptr for e in out]
    assert fids == [1, 3], (
        f"expected [1, 3], got {fids} -- variant 0 includes only its "
        "own direct call_idx=1 (some-not-all); call_idx=0 is excluded "
        "(all-variants), call_idx=2 variant 0 never called"
    )
    assert [e.parent_call_target_index for e in out[1:]] == [1]


def test_parent_call_target_index_indexes_into_parents_call_targets() -> None:
    """The non-root entry's ``parent_call_target_index`` must index into
    the PARENT's call_targets_section list. Specifically: if the
    parent's call_targets has 3 rows and only row index 2 resolves to
    a non-extern non-cycle callee, the child entry's
    parent_call_target_index must be 2 (NOT 0)."""
    child_section = _make_section(section_offset=400, function_name_ptr=4)
    child_fd = _make_function_data("child")

    # Parent has 3 call_targets:
    #   idx 0 -> EXTERN (skipped by EXTERN guard)
    #   idx 1 -> unresolved (function_section_ptr=0; skipped)
    #   idx 2 -> LOCAL @ 400 (the only resolvable row)
    parent_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[
            _ct_extern(fid=10, target_offset=999),
            _ct_local(fid=11, target_offset=0),
            _ct_local(fid=4, target_offset=400),
        ],
        variants=_calling_variant([(0, 0), (1, 0), (2, 0)]),
    )
    parent_fd = _make_function_data("parent")
    session = _FakeSession()
    session.add_matched(parent_section, {0: parent_fd})
    session.add_matched(child_section, {0: child_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=parent_section,
        root_variant_idx=0,
        root_function_data=parent_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 2
    # Root has no parent.
    assert out[0].parent_call_target_index is None
    # Child entry: its parent_call_target_index must be 2 -- the index
    # in PARENT.call_targets_section that pointed here.
    assert out[1].parent_call_target_index == 2
    # And it must actually index back to the row we expect.
    parent_row = out[0].call_targets_section[
        out[1].parent_call_target_index
    ]
    assert parent_row.function_section_ptr == 400
    assert parent_row.type is CallTargetType.LOCAL


# ---------------------------------------------------------------------------
# Additional invariants -- defensive coverage for the cleanest contract.
# ---------------------------------------------------------------------------


def test_root_encounter_category_is_always_local_func() -> None:
    """Per plan D3: the root function is always a LOCAL entity, even if
    it has no callees + the request never went through a PLT stub."""
    root_section = _make_section(section_offset=42, function_name_ptr=7)
    root_fd = _make_function_data("solo")
    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=7,
        max_depth=3,
    )

    assert out[0].encounter_category is Category.LOCAL_FUNC


def test_unresolved_pointer_is_skipped() -> None:
    """``function_section_ptr == 0`` is the BIN's "unresolved" sentinel;
    such rows must be skipped even before EXTERN screening."""
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=0)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")
    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 1


def test_callee_in_unknown_arm_is_skipped() -> None:
    """When ``_idx_for_section_offset`` returns ``None`` (callee not in
    the requested arm), the walker must skip the row instead of
    raising."""
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root")
    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    # Note: callee section is NOT registered -> lookup returns None.

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 1


def test_negative_max_depth_rejected() -> None:
    root_section = _make_section(section_offset=100, function_name_ptr=1)
    root_fd = _make_function_data("root")
    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})

    with pytest.raises(ValueError, match="max_depth"):
        walk_callees(
            session=session,
            root_arm=SectionKind.MATCHED,
            root_section=root_section,
            root_variant_idx=0,
            root_function_data=root_fd,
            root_function_name_ptr=1,
            max_depth=-1,
        )


def test_variant_tokens_prepended_only_at_root_not_at_callees() -> None:
    """Per-row variant-axis contract: variant_tokens are a ROW-level
    identity prefix carried on :class:`Stage1Variant`, NOT a per-
    call-target token stream concern. Every call_target (root + inlined
    callees) feeds ``function_data.tokens`` (body only) into
    ``state.raw_tokens`` -- no special-case root path.

    Each splice tree shares one compilation variant axis; the
    variant_tokens prefix is emitted once at row start by Stage 4
    (:func:`_token_assembly.assemble_tokens`) before any call_target
    body. Per-call-target token streams are body-only for every
    encounter category.

    Locking this in at the walker level guards against a regression of
    the legacy ``full_token_stream()`` special-case that fed the root's
    ``variant_tokens + body_tokens`` through ``InlineDecodeState`` --
    which mis-ordered the row (the LOCAL_FUNC self-token sat BEFORE
    the variant_tokens prefix when in reality it marks "root body
    starts here" and belongs AFTER the prefix).
    """
    axis_tokens = np.array([280, 281, 282], dtype=np.uint16)
    callee_section = _make_section(section_offset=200, function_name_ptr=2)
    callee_fd = _make_function_data("callee", variant_tokens=axis_tokens)
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root", variant_tokens=axis_tokens)

    session = _FakeSession()
    session.add_matched(root_section, {0: root_fd})
    session.add_matched(callee_section, {0: callee_fd})

    out = walk_callees(
        session=session,
        root_arm=SectionKind.MATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert len(out) == 2
    # Root: state.raw_tokens == function_data.tokens (body only) --
    # variant_tokens do NOT enter the per-call-target token stream.
    np.testing.assert_array_equal(out[0].state.raw_tokens, root_fd.tokens)
    assert int(out[0].state.raw_tokens.shape[0]) == int(
        root_fd.tokens.shape[0]
    )
    # Callee: state.raw_tokens == body_tokens only -- variant_tokens
    # are NOT prepended at the splice point either; the contract is
    # uniform across root and callees.
    np.testing.assert_array_equal(out[1].state.raw_tokens, callee_fd.tokens)
    assert int(out[1].state.raw_tokens.shape[0]) == int(
        callee_fd.tokens.shape[0]
    )


def test_unmatched_arm_uses_load_unmatched_for_splice() -> None:
    """The walker dispatches on ``root_arm``: matched -> _load_matched...
    unmatched -> _load_unmatched_for_splice (no variant_index)."""
    callee_section = _make_section(
        section_offset=200,
        function_name_ptr=2,
        variants=[_make_variant(vkey=0)],  # one variant per unmatched
                                            # section by construction
    )
    callee_fd = _make_function_data("callee_unmatched")
    root_section = _make_section(
        section_offset=100,
        function_name_ptr=1,
        call_targets=[_ct_local(fid=2, target_offset=200)],
        variants=_calling_variant([(0, 0)]),
    )
    root_fd = _make_function_data("root_unmatched")

    session = _FakeSession()
    session.add_unmatched(root_section, root_fd)
    session.add_unmatched(callee_section, callee_fd)

    out = walk_callees(
        session=session,
        root_arm=SectionKind.UNMATCHED,
        root_section=root_section,
        root_variant_idx=0,
        root_function_data=root_fd,
        root_function_name_ptr=1,
        max_depth=5,
    )

    assert [e.function_name_ptr for e in out] == [1, 2]
    assert out[1].function_data is callee_fd
