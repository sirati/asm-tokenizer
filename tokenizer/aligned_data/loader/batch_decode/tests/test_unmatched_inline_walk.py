"""Flag-ON unmatched-outline inlining over the batch_decode callee walk.

Mirrors the vector_batch inclusion-BFS surfacing test on the OTHER loader:
a synthetic single-arm catalog with matched callees behind unmatched
outlines (co-located in one arm so the single-arm walk reaches them),
asserting that with the flag ON the walk emits the matched callee behind
an unmatched outline IN PLACE OF the outline shell, recurses
unmatched->unmatched, and respects the depth cap. Proves the batch_decode
path implements the same surfacing the vector_batch path does (the
production cross-loader byte-identity gate covers their agreement where the
feature fires; this pins the surfacing semantics on a fixture that fires).

Reuses the ``_FakeSession`` + section/function fixtures from
``test_callee_walk`` (a section registry keyed by ``section_offset``; no
real memmap).
"""

from __future__ import annotations

from typing import List

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._callee_walk import (
    CalleeSectionMetaMemo,
    finalise_pending_call_targets,
    walk_section_callees_pending,
)
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import CallTarget
from tokenizer.aligned_data.splice_inclusion import OnceOnlyInclusion

from .test_callee_walk import (
    _FakeSession,
    _make_function_data,
    _make_section,
    _make_variant,
)


def _ct(*, fid: int, target_offset: int, is_matched: bool) -> CallTarget:
    """A LOCAL call_target with an explicit ``is_matched`` flag."""
    return CallTarget(
        function_name_ptr=fid,
        function_section_ptr=target_offset,
        type=CallTargetType.LOCAL,
        is_matched=is_matched,
    )


def _walk(session, root_section, *, unmatched_inline, depth=3) -> List[List]:
    """Drive the walk over root variant 0 + a quiet sibling, flag-gated.

    Two sampled rows (caller v0 + quiet v1) so a callee reached by only v0
    is included (FLAG-A would exclude an all-rows callee). Returns the
    per-slot finalised lists; slot 0 is v0's emitted list.
    """
    collector = BucketedRunLengthCollector()
    decider = OnceOnlyInclusion()
    root_fd = _make_function_data("root")
    per_variant = walk_section_callees_pending(
        session,
        arm=SectionKind.MATCHED,
        section=root_section,
        sampled_variant_indices=[0, 1],
        root_function_data_per_sampled=[root_fd, root_fd],
        root_function_name_ptr=int(root_section.function_name_ptr),
        max_depth=depth_to_bfs(depth),
        decider=decider,
        collector=collector,
        section_meta_memo=CalleeSectionMetaMemo(),
        unmatched_inline=unmatched_inline,
        unmatched_inline_depth=depth,
    )
    runlen = collector.flush()
    return [finalise_pending_call_targets(rows, runlen) for rows in per_variant]


def depth_to_bfs(_cap: int) -> int:
    """BFS max_depth (independent of the unmatched-hop cap) -- ample here."""
    return 8


def _emitted_fids(slot_rows) -> List[int]:
    """The non-root emitted function_name_ptr list (root at index 0 dropped)."""
    return [row.function_name_ptr for row in slot_rows[1:]]


def _build_session_chain(is_matched_chain: List[bool]):
    """root (matched, 2 var) -> a LOCAL chain sec1 -> sec2 -> ... per the
    ``is_matched_chain`` flags, terminating at a leaf.

    Section k (k>=1) is at offset ``0x10 * (k + 1)`` and calls section k+1
    (its slot 0) unless it is the last. ``is_matched_chain[k-1]`` is the
    ``is_matched`` flag of the root/parent edge INTO section k. All bodies
    live in the matched arm (co-located so the single-arm walk reaches
    them -- the production corpus splits them cross-arm, which is the whole
    finding).
    """
    session = _FakeSession()
    n = len(is_matched_chain)
    offsets = [0x10 * (k + 1) for k in range(n + 1)]  # root + n chain secs

    # root: 2 variants; v0 calls slot0 (-> sec1), v1 calls nothing.
    root = _make_section(
        section_offset=offsets[0],
        function_name_ptr=0,
        call_targets=[_ct(fid=1, target_offset=offsets[1], is_matched=is_matched_chain[0])],
        variants=[
            _make_variant(vkey=0, per_call_entries=[(0, 0)]),
            _make_variant(vkey=1, per_call_entries=[]),
        ],
    )
    session.add_matched(root, {0: _make_function_data("root"), 1: _make_function_data("root")})

    for k in range(1, n + 1):
        is_last = k == n
        cts = (
            []
            if is_last
            else [
                _ct(
                    fid=k + 1,
                    target_offset=offsets[k + 1],
                    is_matched=is_matched_chain[k],
                )
            ]
        )
        pce = [] if is_last else [(0, 0)]
        sec = _make_section(
            section_offset=offsets[k],
            function_name_ptr=k,
            call_targets=cts,
            variants=[_make_variant(vkey=0, per_call_entries=pce)],
        )
        session.add_matched(sec, {0: _make_function_data(f"sec{k}")})
    return session, root


def test_unmatched_to_matched_surfaces_under_flag():
    """root -unmatched-> sec1 -matched-> sec2(leaf).

    Flag OFF: v0 splices the unmatched shell sec1. Flag ON: sec1 is looked
    through; sec2 (the matched callee behind it) is emitted in its place.
    """
    # edge into sec1 is unmatched; edge sec1->sec2 is matched; sec2 leaf.
    session, root = _build_session_chain([False, True])
    off = _walk(session, root, unmatched_inline=False)
    on = _walk(session, root, unmatched_inline=True)
    # sec1 fid==1, sec2 fid==2.
    assert 1 in _emitted_fids(off[0])  # OFF splices the unmatched shell
    on_fids = _emitted_fids(on[0])
    assert 2 in on_fids  # ON surfaces the matched callee behind sec1
    assert 1 not in on_fids  # the shell sec1 is replaced


def test_unmatched_unmatched_matched_recurses():
    """root -U-> sec1 -U-> sec2 -matched-> sec3(leaf); flag ON surfaces sec3."""
    session, root = _build_session_chain([False, False, True])
    on = _walk(session, root, unmatched_inline=True, depth=3)
    assert 3 in _emitted_fids(on[0])  # sec3 surfaced 2 unmatched hops deep


def test_depth_cap_blocks_too_deep_matched():
    """root -U-> s1 -U-> s2 -U-> s3 -matched-> s4(leaf).

    s4 sits behind 3 unmatched hops; cap=2 must NOT surface it, cap=3 must.
    """
    session, root = _build_session_chain([False, False, False, True])
    on_cap2 = _walk(session, root, unmatched_inline=True, depth=2)
    on_cap3 = _walk(session, root, unmatched_inline=True, depth=3)
    assert 4 not in _emitted_fids(on_cap2[0])
    assert 4 in _emitted_fids(on_cap3[0])
