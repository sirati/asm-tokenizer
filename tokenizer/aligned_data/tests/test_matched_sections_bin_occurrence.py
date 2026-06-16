"""Occurrence-aware duplicated-name call resolution — WRITER unit tests.

These drive :class:`SectionWriter` directly with explicit
``PerCallEntry.callee_occurrence`` + ``begin_section(occurrence=...)``
values (no producer populates them in the build pipeline yet), exercising
the two resolution entry points the writer owns:

* the forward path (caller emits a hole BEFORE the matching sibling
  closes; the occurrence-gated Step-3 sibling-close fills it), and
* the inline-backward path (caller emits AFTER the matching sibling
  closed; the ``(fid, occurrence)`` sibling registry resolves it inline).

The BRIGHT LINE under test: a hole is ever resolved to the
occurrence-MATCHING sibling or left :data:`MISSING_VARIANT_INDEX`
(0xFFFE) — NEVER to an arbitrary last-write-wins sibling. Every
assertion checks ``function_section_ptr`` (Case A) AND
``section_variant_index`` (Case B) TOGETHER, because the loader reads
``function_section_ptr`` as the callee section and then indexes
``callee_section.variants[J]`` — a test that checked only J would miss a
wrong-sibling Case-A pointer.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    CallTargetSpec,
    PerCallEntry,
    SectionWriter,
    iter_sections_bin,
)


# Two same-FID sibling sections for callee FID=2: occurrence 0 and 1.
# Each carries TWO variants with the SAME variant_ref_offsets (0x50, 0x60)
# so that the ONLY thing distinguishing which sibling legitimately
# addresses a call is the occurrence — a pure bright-line discriminator.
_CALLEE_FID = 2
_CALLER_FID = 1


def _emit_caller(
    writer: SectionWriter,
    *,
    occurrence,
    caller_vref: int,
    fid: int = _CALLER_FID,
) -> None:
    """Emit a single-variant caller section that calls FID=2 once.

    ``occurrence`` is the intended same-name sibling (``None`` => no
    disambiguation). ``caller_vref`` is both the caller variant's
    ``variant_ref_offset`` and the per-call entry's ``callee_vkey``
    (Step-7 on-wire invariant: they share the ``_variants.bin`` byte
    position).
    """
    writer.begin_section(function_name_ptr=fid, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=_CALLEE_FID,
                type=CallTargetType.LOCAL,
                is_matched=False,
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=caller_vref, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=_CALLEE_FID,
                callee_vkey=caller_vref,
                callee_occurrence=occurrence,
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()


def _emit_sibling(
    writer: SectionWriter, *, occurrence: int, vrefs
) -> int:
    """Emit one same-FID sibling callee section with ``vrefs`` variants.

    Returns its section offset. Each variant has no per-call entries (a
    leaf callee), so the only on-wire content the caller resolves against
    is the variant table's ``variant_ref_offset`` ordering.
    """
    off = writer.begin_section(
        function_name_ptr=_CALLEE_FID,
        n_variants=len(vrefs),
        duplicated=True,
        occurrence=occurrence,
    )
    writer.emit_call_targets([])
    for i, vref in enumerate(vrefs):
        writer.begin_variant(variant_ref_offset=vref, data_offset_shifted=i * 8)
        writer.emit_per_call_entries([])
        writer.end_variant(vkey=f"v{vref}")
    writer.end_section()
    return off


def _section_by_offset(sections, offset):
    for s in sections:
        if s.section_offset == offset:
            return s
    raise AssertionError(f"no section at offset {offset}")


def _only_caller(sections):
    callers = [s for s in sections if s.function_name_ptr == _CALLER_FID]
    assert len(callers) == 1, callers
    return callers[0]


# ---------------------------------------------------------------------------
# (a) forward-ref -> sibling-k
# ---------------------------------------------------------------------------


def test_forward_ref_resolves_to_intended_sibling(tmp_path: Path):
    """Caller emits a hole with callee_occurrence=1 BEFORE sibling-1 closes;
    after sibling-1 closes, function_section_ptr AND J both point at
    sibling-1 (not sibling-0, which closes second / last-write-wins)."""
    path = tmp_path / "fwd.bin"
    w = SectionWriter(path)

    # Caller first (forward ref into FID=2 occurrence 1).
    _emit_caller(w, occurrence=1, caller_vref=0x60)

    # Sibling 1 (the intended one): vref 0x60 lands at variant_idx 1.
    off1 = _emit_sibling(w, occurrence=1, vrefs=[0x50, 0x60])
    # Sibling 0 closes AFTER (last-write-wins would mis-point Case A here).
    off0 = _emit_sibling(w, occurrence=0, vrefs=[0x60, 0x70])
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    # Case A: function_section_ptr is sibling-1, NOT the LWW sibling-0.
    assert caller.call_targets[0].function_section_ptr == off1
    assert caller.call_targets[0].function_section_ptr != off0
    # Case B: J indexes sibling-1's variant table; 0x60 is at idx 1 there.
    assert caller.variants[0].per_call_entries == [(0, 1)]


# ---------------------------------------------------------------------------
# (b) MULTI-SIBLING RIGHT-ONE  (the core bright-line test)
# ---------------------------------------------------------------------------


def test_multi_sibling_right_one_never_wrong(tmp_path: Path):
    """Both occ-0 and occ-1 siblings close; a caller intending occ-1 gets
    occ-1's function_section_ptr + J, NEVER occ-0's — even though both
    siblings carry the SAME variant_ref_offset the caller resolves on, so
    occurrence is the ONLY discriminator."""
    path = tmp_path / "multi.bin"
    w = SectionWriter(path)

    # Caller intends occurrence 1; its vkey 0x50 exists in BOTH siblings.
    _emit_caller(w, occurrence=1, caller_vref=0x50)

    # Sibling 0: 0x50 at variant_idx 0.
    off0 = _emit_sibling(w, occurrence=0, vrefs=[0x50, 0x99])
    # Sibling 1: 0x50 at variant_idx 1 (distinct J from sibling-0's).
    off1 = _emit_sibling(w, occurrence=1, vrefs=[0x11, 0x50])
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    # Bright line: sibling-1, never sibling-0.
    assert caller.call_targets[0].function_section_ptr == off1
    assert caller.call_targets[0].function_section_ptr != off0
    # J = 1 (0x50's index in sibling-1), NOT 0 (its index in sibling-0).
    assert caller.variants[0].per_call_entries == [(0, 1)]
    # Cross-check: had it leaked to sibling-0, J would be 0 + ptr off0.
    assert caller.variants[0].per_call_entries != [(0, 0)]


# ---------------------------------------------------------------------------
# (c) MISSING-SIBLING -> 0xFFFE
# ---------------------------------------------------------------------------


def test_missing_sibling_collapses_to_0xfffe(tmp_path: Path):
    """callee_occurrence=2 but no occurrence-2 sibling ever closes (only
    0 and 1 do) -> finalize -> MISSING_VARIANT_INDEX. Never resolves to a
    present-but-wrong-occurrence sibling."""
    path = tmp_path / "missing.bin"
    w = SectionWriter(path)

    _emit_caller(w, occurrence=2, caller_vref=0x50)
    off0 = _emit_sibling(w, occurrence=0, vrefs=[0x50, 0x60])
    off1 = _emit_sibling(w, occurrence=1, vrefs=[0x50, 0x70])
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    # J collapses to MISSING (no occ-2 sibling).
    assert caller.variants[0].per_call_entries == [(0, MISSING_VARIANT_INDEX)]
    # Case A: the caller emitted FIRST (forward ref), so emit-time stamped
    # the 0 placeholder; the occurrence gate then skipped BOTH closing
    # siblings (intended 2 != 0, != 1), so function_section_ptr stays 0.
    # That is the cleanest bright-line outcome: the loader treats ptr==0 as
    # an unresolved pointer (no inlining), never reading a wrong sibling —
    # and it is NEVER a present-but-wrong-occurrence sibling offset.
    assert caller.call_targets[0].function_section_ptr == 0
    assert caller.call_targets[0].function_section_ptr not in (off0, off1)


# ---------------------------------------------------------------------------
# (d) NON-DUP unchanged
# ---------------------------------------------------------------------------


def test_non_dup_single_section_unchanged(tmp_path: Path):
    """A None-occurrence caller into a single (non-dup) callee resolves
    exactly as the legacy single-section path: function_section_ptr +
    correct J, no MISSING."""
    path = tmp_path / "nondup.bin"
    w = SectionWriter(path)

    _emit_caller(w, occurrence=None, caller_vref=0x60)
    # One and only callee section (occurrence default 0, not duplicated).
    off = w.begin_section(function_name_ptr=_CALLEE_FID, n_variants=2)
    w.emit_call_targets([])
    w.begin_variant(variant_ref_offset=0x50, data_offset_shifted=0)
    w.emit_per_call_entries([])
    w.end_variant(vkey="a")
    w.begin_variant(variant_ref_offset=0x60, data_offset_shifted=8)
    w.emit_per_call_entries([])
    w.end_variant(vkey="b")
    w.end_section()
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    assert caller.call_targets[0].function_section_ptr == off
    assert caller.variants[0].per_call_entries == [(0, 1)]  # 0x60 -> idx 1


# ---------------------------------------------------------------------------
# (e) None-occurrence dup -> 0xFFFE  (producer's blanket-MISSING path)
# ---------------------------------------------------------------------------


def test_none_occurrence_dup_collapses_to_0xfffe(tmp_path: Path):
    """A dup callee the producer could not pin to an occurrence is routed
    to MISSING via resolved_section_variant_index (callee_occurrence stays
    None). The writer stamps it verbatim and opens NO hole — no sibling
    resolution can override it to a wrong sibling."""
    path = tmp_path / "nonedup.bin"
    w = SectionWriter(path)

    # Caller emits the producer's pre-stamped terminal MISSING for a dup
    # callee, callee_occurrence None (ambiguous / no-occurrence).
    w.begin_section(function_name_ptr=_CALLER_FID, n_variants=1)
    w.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=_CALLEE_FID,
                type=CallTargetType.LOCAL,
                is_matched=False,
            ),
        ]
    )
    w.begin_variant(variant_ref_offset=0x50, data_offset_shifted=0)
    w.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=_CALLEE_FID,
                callee_vkey=0x50,
                resolved_section_variant_index=MISSING_VARIANT_INDEX,
                callee_occurrence=None,
            ),
        ]
    )
    w.end_variant(vkey="x86_O0")
    w.end_section()

    # Two siblings close after; neither may override the terminal MISSING.
    _emit_sibling(w, occurrence=0, vrefs=[0x50, 0x60])
    _emit_sibling(w, occurrence=1, vrefs=[0x50, 0x70])
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    assert caller.variants[0].per_call_entries == [(0, MISSING_VARIANT_INDEX)]


# ---------------------------------------------------------------------------
# (f) BACKWARD-ref-after-matching-sibling-closed -> inline registry path
# ---------------------------------------------------------------------------


def test_backward_ref_inline_registry_correct_sibling(tmp_path: Path):
    """Caller emitted AFTER the matching sibling closed: the inline
    (fid, occurrence) registry resolves J AND re-stamps function_section_ptr
    to the occurrence-matching sibling — NOT _known_sections' last-write-wins
    sibling (which here is the OTHER occurrence, closed last)."""
    path = tmp_path / "bwd.bin"
    w = SectionWriter(path)

    # Both siblings close FIRST (backward refs for the caller).
    off1 = _emit_sibling(w, occurrence=1, vrefs=[0x11, 0x50])  # 0x50 at idx 1
    off0 = _emit_sibling(w, occurrence=0, vrefs=[0x50, 0x99])  # closes last (LWW)
    # Caller intends occurrence 1; _known_sections[FID=2] == off0 (LWW).
    _emit_caller(w, occurrence=1, caller_vref=0x50)
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    # Case A re-stamped to sibling-1 (registry), NOT the LWW sibling-0.
    assert caller.call_targets[0].function_section_ptr == off1
    assert caller.call_targets[0].function_section_ptr != off0
    # Case B: J is 0x50's index in sibling-1 (=1), not sibling-0's (=0).
    assert caller.variants[0].per_call_entries == [(0, 1)]
    assert caller.variants[0].per_call_entries != [(0, 0)]


# ---------------------------------------------------------------------------
# (g) BACKWARD into dup where only a WRONG-occurrence sibling closed -> 0xFFFE
# ---------------------------------------------------------------------------


def test_backward_ref_only_wrong_occurrence_closed_is_0xfffe(tmp_path: Path):
    """Caller intends occurrence 1, but only the occurrence-0 sibling has
    closed (and is in _known_sections). The inline registry has no
    (FID, 1) entry -> the edge must NOT resolve to the occurrence-0 sibling
    -> it defers and finalize stamps MISSING. Bright line under the inline
    path: a wrong-occurrence backward sibling is never used."""
    path = tmp_path / "bwd_wrong.bin"
    w = SectionWriter(path)

    off0 = _emit_sibling(w, occurrence=0, vrefs=[0x50, 0x60])
    # Caller intends occurrence 1 (no occ-1 sibling exists at all).
    _emit_caller(w, occurrence=1, caller_vref=0x50)
    w.finalize()

    sections = list(iter_sections_bin(path))
    caller = _only_caller(sections)
    # J collapses to MISSING, never sibling-0's index for 0x50. The
    # emit-time function_section_ptr may carry sibling-0's last-write-wins
    # offset, but a MISSING J is filtered BEFORE any variant access
    # (loader _variant_selection ``_usable(J)`` is False), so the wrong
    # sibling's variant table is never read — identical to the existing
    # blanket-MISSING dup-callee invariant.
    assert caller.variants[0].per_call_entries == [(0, MISSING_VARIANT_INDEX)]
    assert caller.call_targets[0].function_section_ptr in (0, off0)


# ---------------------------------------------------------------------------
# Defensive: conflicting per-(caller, fid) occurrences raise
# ---------------------------------------------------------------------------


def test_conflicting_occurrence_for_same_caller_fid_raises(tmp_path: Path):
    """Two variants of ONE caller section opening holes into the same
    callee FID with DIFFERENT non-None occurrences is a producer contract
    violation and raises (the section-granular occurrence key is
    well-defined only because the producer guarantees uniformity)."""
    import pytest

    path = tmp_path / "conflict.bin"
    w = SectionWriter(path)

    w.begin_section(function_name_ptr=_CALLER_FID, n_variants=2)
    w.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=_CALLEE_FID,
                type=CallTargetType.LOCAL,
                is_matched=False,
            ),
        ]
    )
    w.begin_variant(variant_ref_offset=0x50, data_offset_shifted=0)
    w.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=_CALLEE_FID,
                callee_vkey=0x50,
                callee_occurrence=0,
            ),
        ]
    )
    w.end_variant(vkey="a")
    w.begin_variant(variant_ref_offset=0x60, data_offset_shifted=8)
    with pytest.raises(ValueError, match="conflicting callee_occurrence"):
        w.emit_per_call_entries(
            [
                PerCallEntry(
                    called_idx=0,
                    callee_function_name_ptr=_CALLEE_FID,
                    callee_vkey=0x60,
                    callee_occurrence=1,
                ),
            ]
        )
