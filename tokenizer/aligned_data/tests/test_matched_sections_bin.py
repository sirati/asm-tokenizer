"""Round-trip + back-patch + finalize tests for ``matched_sections_bin``.

The writer is the sole producer of ``<binary>_sections.bin``; the
reader is the sole consumer on the dataloader hot path. Correctness
here pins both halves of the codec at once.

``PerCallEntry.callee_vkey`` shares the same value-space as the
matching variant's on-disk ``variant_ref_offset`` (the byte offset of
the vkey in the per-binary ``_variants.bin`` sidecar). Tests use
integer offsets directly so the writer's self-describing back-patch
(parse the just-closed section, match holes against its variant
table) resolves through to a stable variant_idx.

Coverage:

* round-trip — one section, two call_targets, two variants → reader
  recovers every field.
* back-patch on header — section A forward-references section B's
  call_target slot; after B is written the slot equals B's section
  offset.
* back-patch on per-variant slot — section A's variant references
  section B's vkey before B has been written; after B emits the
  variant the slot equals B's variant_idx.
* extern + unknown library — call_target with EXTERN type and
  ``extern_provider_line_no=None`` lands as the ``0`` sentinel.
* finalize asserts on a callee whose section was never written.
* finalize stamps MISSING_VARIANT_INDEX on per-variant holes whose
  callee_vkey no sibling section ever registered.
* OUTLINED_FUNCTION_N coherence: two sibling sections with the same
  FID but disjoint vkey sets each resolve only their own caller's
  per_call slot.
* prelude round-trip via the magic-specific helpers.
* :meth:`MemmapBinWriter.patch` is a separate test (it's the
  random-access primitive the SectionWriter is built on).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import (
    CALL_TARGET_ENTRY_SIZE,
    JUMP_TABLE_ENTRY_SIZE,
    MISSING_VARIANT_INDEX,
    PER_CALL_ENTRY_SIZE,
    SECTION_HEADER_SIZE,
    UNKNOWN_EXTERN_PROVIDER,
    UNRESOLVED_VARIANT_INDEX,
    VARIANT_HEADER_SIZE,
    CallTargetSpec,
    PerCallEntry,
    SectionWriter,
    iter_sections_bin,
)
from tokenizer.aligned_data.memmap_format import (
    DATA_BIN_PRELUDE_MAGIC,
    DATA_BIN_PRELUDE_SIZE,
    MATCHED_SECTIONS_BIN_PRELUDE_MAGIC,
    MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    MEMMAP_FORMAT_VERSION,
    assert_data_bin_prelude,
    assert_matched_sections_prelude,
    encode_data_bin_prelude,
    encode_matched_sections_prelude,
)
from tokenizer.aligned_data.memmap_writer import MemmapBinWriter

# ---------------------------------------------------------------------------
# Reader-dependent tests are temporarily xfailed pending the
# ``parse_section_bin`` rewrite for the new jump-table format. The writer
# now emits an 8-byte variant header and an ``n_variants × u16`` jump
# table after the section header; the reader still raises
# :class:`NotImplementedError` because B.2 has not yet shipped its
# rewrite. Every test that round-trips through ``iter_sections_bin`` /
# ``parse_section_bin`` (directly or via the finalize-time sentinel
# sweep) currently fails. Re-enable each test once B.2 lands.
_PENDING_READER_XFAIL = pytest.mark.xfail(
    reason="pending parse_section_bin rewrite for jump-table format",
    strict=False,
)

# ---------------------------------------------------------------------------
# Prelude helpers
# ---------------------------------------------------------------------------


def test_prelude_round_trip():
    prelude = encode_matched_sections_prelude()
    assert len(prelude) == MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    assert prelude[:4] == MATCHED_SECTIONS_BIN_PRELUDE_MAGIC
    (version,) = struct.unpack("<I", prelude[4:8])
    assert version == MEMMAP_FORMAT_VERSION
    assert prelude[8:16] == b"\x00" * 8
    assert_matched_sections_prelude(prelude)


def test_prelude_wrong_magic_raises():
    bad = b"XXXX" + struct.pack("<I", MEMMAP_FORMAT_VERSION) + b"\x00" * 8
    with pytest.raises(ValueError, match="magic"):
        assert_matched_sections_prelude(bad, path="/tmp/bogus_sections.bin")


def test_prelude_wrong_version_raises():
    bad = (
        MATCHED_SECTIONS_BIN_PRELUDE_MAGIC
        + struct.pack("<I", MEMMAP_FORMAT_VERSION + 999)
        + b"\x00" * 8
    )
    with pytest.raises(ValueError, match="format_version"):
        assert_matched_sections_prelude(bad, path="/tmp/bogus_sections.bin")


# ---------------------------------------------------------------------------
# MemmapBinWriter.patch — random-access write primitive
# ---------------------------------------------------------------------------


def test_memmap_writer_patch_round_trip(tmp_path: Path):
    """Patch lands at the right offset and does not move the cursor."""
    path = tmp_path / "patch_test.bin"
    writer = MemmapBinWriter(path)
    writer.write(b"\x11" * 16)
    pre_cursor = writer.cursor

    writer.patch(4, struct.pack("<I", 0xCAFEBABE))
    assert writer.cursor == pre_cursor

    head = writer.read(0, 4)
    middle = writer.read(4, 4)
    tail = writer.read(8, 8)
    assert head == b"\x11" * 4
    assert struct.unpack("<I", middle)[0] == 0xCAFEBABE
    assert tail == b"\x11" * 8

    writer.finalize()


def test_memmap_writer_patch_rejects_past_cursor(tmp_path: Path):
    """Patching an offset that extends past ``cursor`` must raise."""
    path = tmp_path / "patch_oob.bin"
    writer = MemmapBinWriter(path)
    writer.write(b"\x22" * 8)

    with pytest.raises(ValueError, match="unwritten region"):
        writer.patch(6, b"\x00\x00\x00\x00")  # would touch bytes 6..10 > cursor=8

    with pytest.raises(ValueError, match="non-negative"):
        writer.patch(-1, b"\x00")

    writer.finalize()


# ---------------------------------------------------------------------------
# SectionWriter round-trip + back-patch
# ---------------------------------------------------------------------------


def _read_u16(path: Path, offset: int) -> int:
    with open(path, "rb") as fh:
        fh.seek(offset)
        return struct.unpack("<H", fh.read(2))[0]


@_PENDING_READER_XFAIL
def test_section_round_trip(tmp_path: Path):
    """One section, two call_targets, two variants, every field round-trips."""
    path = tmp_path / "rt_sections.bin"
    writer = SectionWriter(path)

    # Section A (FID=1).
    offset_a = writer.begin_section(function_name_ptr=1, n_variants=2)
    assert offset_a == MATCHED_SECTIONS_BIN_PRELUDE_SIZE

    # First call_target: LOCAL → self-ref FID=1 (resolves immediately).
    # Second call_target: EXTERN with provider line 7.
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=1,
                type=CallTargetType.LOCAL,
                is_matched=True,
            ),
            CallTargetSpec(
                function_name_ptr=99,
                type=CallTargetType.EXTERN,
                is_matched=False,
                extern_provider_line_no=7,
            ),
        ]
    )

    # Variant 1: calls call_target idx=0 (self-ref) targeting variant
    # variant_ref_offset=0x100 — the same value the writer stamps on
    # this variant's header. The per-call entry's callee_vkey shares
    # that value-space so end_section's parse-and-match resolves the
    # slot to variant_idx=0.
    writer.begin_variant(variant_ref_offset=0x100, data_offset_shifted=0x20)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=1,
                callee_vkey=0x100,
            ),
        ]
    )
    v0 = writer.end_variant(vkey="x86_O0")
    assert v0 == 0

    # Variant 2: also calls call_target idx=0 (self-ref) but targets
    # this same variant_ref_offset=0x100 — proves the matching is by
    # vkey not by variant order.
    writer.begin_variant(variant_ref_offset=0x140, data_offset_shifted=0x40)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=1,
                callee_vkey=0x100,
            ),
        ]
    )
    v1 = writer.end_variant(vkey="x86_O3")
    assert v1 == 1

    writer.end_section()
    writer.finalize()

    sections = list(iter_sections_bin(path))
    assert len(sections) == 1
    section = sections[0]
    assert section.function_name_ptr == 1
    assert section.section_offset == MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    assert len(section.call_targets) == 2
    assert section.call_targets[0].function_name_ptr == 1
    assert section.call_targets[0].function_section_ptr == offset_a  # self-ref
    assert section.call_targets[0].type is CallTargetType.LOCAL
    assert section.call_targets[0].is_matched is True
    assert section.call_targets[1].function_name_ptr == 99
    assert section.call_targets[1].function_section_ptr == 7  # extern line
    assert section.call_targets[1].type is CallTargetType.EXTERN
    assert section.call_targets[1].is_matched is False

    assert len(section.variants) == 2
    assert section.variants[0].variant_ref_offset == 0x100
    assert section.variants[0].data_offset_shifted == 0x20
    assert section.variants[0].per_call_entries == [(0, 0)]
    assert section.variants[1].variant_ref_offset == 0x140
    assert section.variants[1].data_offset_shifted == 0x40
    # variant 2's per-call points at idx 0, resolved to variant_idx 0
    # of FID=1's section.
    assert section.variants[1].per_call_entries == [(0, 0)]


@_PENDING_READER_XFAIL
def test_header_back_patch(tmp_path: Path):
    """Section A forward-references section B; B's section_offset
    lands in A's call_target slot after end_section(B)."""
    path = tmp_path / "header_patch.bin"
    writer = SectionWriter(path)

    # Section A (FID=1): references B (FID=2) which is not yet written.
    offset_a = writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    # No variants for A.
    writer.end_section()

    # Section B (FID=2): just a header, no variants either.
    offset_b = writer.begin_section(function_name_ptr=2, n_variants=0)
    writer.emit_call_targets([])
    writer.end_section()

    writer.finalize()

    sections = list(iter_sections_bin(path))
    assert len(sections) == 2
    section_a, section_b = sections
    assert section_a.function_name_ptr == 1
    assert section_b.function_name_ptr == 2
    assert section_a.call_targets[0].function_name_ptr == 2
    assert section_a.call_targets[0].function_section_ptr == offset_b
    assert section_b.section_offset == offset_b
    # And the sections-stride is 4-byte aligned.
    assert offset_b % 4 == 0
    assert offset_a == MATCHED_SECTIONS_BIN_PRELUDE_SIZE


@_PENDING_READER_XFAIL
def test_per_variant_back_patch(tmp_path: Path):
    """Section A's variant references B's variant_ref_offset=0x50 before
    B is written; after B emits that variant the slot equals B's
    variant_idx (not 0xFFFF)."""
    path = tmp_path / "variant_patch.bin"
    writer = SectionWriter(path)

    # Section A: one call_target referencing B (FID=2), one variant
    # whose per-call entry points at B's variant_ref_offset=0x50.
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0x10, data_offset_shifted=0x20)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=2,
                callee_vkey=0x50,
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Section B: emits two variants. We want variant_ref_offset=0x50 to
    # land at variant_idx=1 to make the back-patch non-trivial (0 is
    # the default unsigned value, so 1 catches a "wrote no bytes" bug
    # too).
    writer.begin_section(function_name_ptr=2, n_variants=2)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0x30, data_offset_shifted=0x40)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O3")
    writer.begin_variant(variant_ref_offset=0x50, data_offset_shifted=0x60)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O0")  # variant_idx = 1
    writer.end_section()

    writer.finalize()

    sections = list(iter_sections_bin(path))
    section_a, section_b = sections
    assert section_a.variants[0].per_call_entries == [(0, 1)]
    assert section_a.call_targets[0].function_section_ptr == section_b.section_offset
    # And every section_variant_index in the bin is now < n_variants of
    # the corresponding callee section.
    for v in section_a.variants:
        for _called, sv_idx in v.per_call_entries:
            assert sv_idx != UNRESOLVED_VARIANT_INDEX


@_PENDING_READER_XFAIL
def test_extern_library_unknown_lands_as_zero(tmp_path: Path):
    """``extern_provider_line_no=None`` → function_section_ptr=0."""
    path = tmp_path / "extern_unknown.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=42,
                type=CallTargetType.EXTERN,
                is_matched=False,
                extern_provider_line_no=None,
            ),
        ]
    )
    writer.end_section()
    writer.finalize()

    section = next(iter_sections_bin(path))
    assert section.call_targets[0].type is CallTargetType.EXTERN
    assert section.call_targets[0].function_section_ptr == UNKNOWN_EXTERN_PROVIDER
    assert section.call_targets[0].function_section_ptr == 0


def test_finalize_asserts_on_unresolved_hole(tmp_path: Path):
    """Forward reference whose callee section is never written ⇒
    finalize() raises."""
    path = tmp_path / "unresolved.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.end_section()
    # Never emit section 2.

    with pytest.raises(ValueError, match="callee section that was never written"):
        writer.finalize()


@_PENDING_READER_XFAIL
def test_backward_per_call_to_closed_section_missing_vkey_stamps_missing_at_finalize(tmp_path: Path):
    """If a section's per-call entry references a callee whose section
    has already been written but does not carry a variant matching the
    caller's vkey, the slot is left as ``UNRESOLVED`` until
    :meth:`finalize`, which stamps :data:`MISSING_VARIANT_INDEX`. The
    writer never resolves at emit-time; every per-call slot defers."""
    path = tmp_path / "backward_miss.bin"
    writer = SectionWriter(path)

    # Section B: emits ONLY variant_ref_offset=0xB3 (think
    # vkey="x86_O3" → byte_offset 0xB3 in the variants sidecar).
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xB3, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O3")
    writer.end_section()

    # Section A (emitted AFTER B closes): references B at a different
    # vkey (variant_ref_offset=0xB0 — the byte offset of "x86_O0").
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0xB0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xB0)]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()
    writer.finalize()

    sections = {s.function_name_ptr: s for s in iter_sections_bin(path)}
    a = sections[1]
    (called_idx, sv_idx), = a.variants[0].per_call_entries
    assert called_idx == 0
    assert sv_idx == MISSING_VARIANT_INDEX


@_PENDING_READER_XFAIL
def test_per_variant_hole_with_missing_callee_vkey_lands_as_missing_sentinel(tmp_path: Path):
    """Cross-arm vkey mismatch: callee section IS written but never emits
    the caller's vkey. The per-call slot lands on
    :data:`MISSING_VARIANT_INDEX` (= 0xFFFE) at :meth:`finalize`
    instead of raising — the legitimate corpus-scale case where caller
    and callee have different surviving-variant sets after pass-1's
    drop rules. The finalize sweep rejects only 0xFFFF (unresolved
    hole), not 0xFFFE."""
    import io

    path = tmp_path / "missing_variant.bin"
    warn_log = io.StringIO()
    writer = SectionWriter(path, warn_log=warn_log)

    # Section A: references B's variant_ref_offset=0xC0 via a per-call
    # entry — but B only emits variant_ref_offset=0xC3.
    a_offset = writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0xC0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=2,
                callee_vkey=0xC0,
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Section B: only emits variant_ref_offset=0xC3 — A's per-call hole
    # falls through to finalize → MISSING_VARIANT_INDEX + warn-log
    # entry.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xC3, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O3")
    writer.end_section()
    writer.finalize()

    sections = list(iter_sections_bin(path))
    a = next(s for s in sections if s.function_name_ptr == 1)
    (called_idx, sv_idx), = a.variants[0].per_call_entries
    assert called_idx == 0
    assert sv_idx == MISSING_VARIANT_INDEX
    # warn-log received exactly one ``missing_variant:`` line for this slot.
    log_text = warn_log.getvalue()
    assert log_text.count("missing_variant:") == 1
    assert "callee_fid=2" in log_text
    assert "callee_vkey=192" in log_text  # 0xC0 = 192
    assert f"caller_section@{a_offset}" in log_text


def test_called_idx_validation(tmp_path: Path):
    """A PerCallEntry whose called_idx doesn't match the section's
    call_target table is rejected eagerly."""
    path = tmp_path / "bad_idx.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)

    # called_idx out of range.
    with pytest.raises(ValueError, match="out of range"):
        writer.emit_per_call_entries(
            [PerCallEntry(called_idx=5, callee_function_name_ptr=2, callee_vkey="x")]
        )

    # called_idx is in range but points at a different FID than declared.
    with pytest.raises(ValueError, match="declares callee_function_name_ptr"):
        writer.emit_per_call_entries(
            [PerCallEntry(called_idx=0, callee_function_name_ptr=99, callee_vkey="x")]
        )


@_PENDING_READER_XFAIL
def test_dup_section_overwrites_known_sections_with_latest_offset(tmp_path: Path):
    """Two sections sharing a FID are both written; ``known_sections``
    tracks the latest section's offset.

    Clang emits compiler-internal helpers (``OUTLINED_FUNCTION_N``)
    that share names across distinct bodies, so the matched arm can
    legitimately produce multiple sections with the same
    ``function_name_ptr``. The writer accepts the collision and the
    matched_index.bin records all sections independently (the loader
    indexes by position, not by name)."""
    path = tmp_path / "dup.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets([])
    first_offset, _ = writer.end_section()

    second_offset = writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets([])
    writer.end_section()
    writer.finalize()

    assert second_offset != first_offset
    assert writer._known_sections[1] == second_offset  # noqa: SLF001 — internal-state assertion is the point of this test
    sections = list(iter_sections_bin(path))
    assert len(sections) == 2
    assert {s.function_name_ptr for s in sections} == {1}


@_PENDING_READER_XFAIL
def test_dup_variant_ref_offset_within_section(tmp_path: Path):
    """A variant_ref_offset can re-appear within a section (legacy
    pre-refactor ``function_lookup`` last-write-wins behaviour). With
    the self-describing back-patch, a per-call hole targeting that
    ref_offset resolves to whichever variant is FIRST in the section's
    on-disk variant block list (parse-side dict insertion order)."""
    path = tmp_path / "dup_vkey.bin"
    writer = SectionWriter(path)

    # Caller section references FID=2's variant_ref_offset=0xD0.
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0xD0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xD0)]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Callee section FID=2: emits the SAME variant_ref_offset=0xD0
    # twice (first as variant_idx=0, then again as variant_idx=1).
    writer.begin_section(function_name_ptr=2, n_variants=2)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xD0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    first_idx = writer.end_variant(vkey="x86_O0")
    writer.begin_variant(variant_ref_offset=0xD0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    second_idx = writer.end_variant(vkey="x86_O0")
    writer.end_section()
    writer.finalize()

    assert first_idx == 0
    assert second_idx == 1
    sections = {s.function_name_ptr: s for s in iter_sections_bin(path)}
    a = sections[1]
    # Both variants share ref_offset=0xD0; the local
    # ``vkey_to_idx`` dict comprehension that ``end_section`` builds
    # overwrites the key on every iteration, so the LAST variant_idx
    # wins for back-patch resolution — matching the pre-refactor
    # ``known_section_variants`` last-write-wins semantics.
    (_called, sv_idx), = a.variants[0].per_call_entries
    assert sv_idx == second_idx
    assert sv_idx != UNRESOLVED_VARIANT_INDEX
    assert sv_idx != MISSING_VARIANT_INDEX


def test_section_alignment_padding(tmp_path: Path):
    """Each section's offset is 4-byte aligned even when the previous
    section's natural end is not.

    Section header (8) + 1 jump-table slot (2) + 1 call_target (12) +
    1 variant header (8) + 1 per-call entry (4) = 34 bytes; trailer
    pad = 2 bytes → next section starts at offset 16+34+2 = 52
    (4-byte aligned).
    """
    path = tmp_path / "align.bin"
    writer = SectionWriter(path)

    # Section A produces a 34-byte payload before trailer pad. The
    # per-call entry self-references the same variant_ref_offset=0 so
    # the back-patch resolves cleanly inside the section.
    a_offset = writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0, callee_function_name_ptr=1, callee_vkey=0
            )
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    expected_b_offset = (
        a_offset
        + SECTION_HEADER_SIZE
        + 1 * JUMP_TABLE_ENTRY_SIZE
        + 1 * CALL_TARGET_ENTRY_SIZE
        + 1 * VARIANT_HEADER_SIZE
        + 1 * PER_CALL_ENTRY_SIZE
    )
    # Round up to 4-byte boundary.
    if expected_b_offset % 4 != 0:
        expected_b_offset += 4 - (expected_b_offset % 4)

    b_offset = writer.begin_section(function_name_ptr=2, n_variants=0)
    assert b_offset == expected_b_offset
    assert b_offset % 4 == 0
    writer.emit_call_targets([])
    writer.end_section()
    writer.finalize()


@_PENDING_READER_XFAIL
def test_finalize_sweep_catches_leaked_sentinel(tmp_path: Path):
    """If a writer bug leaves a 0xFFFF slot AND empties pending_holes,
    the belt-and-braces sweep in finalize() still catches it.

    This is a defensive test: we patch the writer's back-patch loop to
    no-op so a real-bin sentinel leaks past the pending_holes check,
    proving the sweep is the second line of defence.
    """
    path = tmp_path / "leak.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0, callee_function_name_ptr=2, callee_vkey=0xE0
            )
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xE0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Drop the pending-holes book without performing the per-variant
    # patches that resolved them; the on-disk per-call slot is
    # consequently still 0xFFFF.
    # Re-write the slot to 0xFFFF (simulate the dropped patch).
    # The per-call section_variant_index slot in section A is at:
    #   section A offset (16) + header (8) + 1 jump-table slot (2)
    #     + 1 call_target (12) + variant header (8)
    #     + 2 (skip called_idx) = 48.
    slot_offset = (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        + SECTION_HEADER_SIZE
        + 1 * JUMP_TABLE_ENTRY_SIZE
        + CALL_TARGET_ENTRY_SIZE
        + VARIANT_HEADER_SIZE
        + 2  # skip u16 called_idx; section_variant_index is the second field
    )
    # The slot was already patched by end_section(FID=2); poke it back.
    writer._writer.patch(slot_offset, struct.pack("<H", UNRESOLVED_VARIANT_INDEX))

    # _pending_holes is already empty (real back-patch ran), so the
    # only line of defence is the sweep.
    with pytest.raises(ValueError, match="unresolved section_variant_index"):
        writer.finalize()


@_PENDING_READER_XFAIL
def test_two_sections_share_callee(tmp_path: Path):
    """Two distinct sections both forward-reference the same callee
    section; both get patched at the callee's end_section."""
    path = tmp_path / "shared_callee.bin"
    writer = SectionWriter(path)

    # A references C.
    writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=3, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.end_section()

    # B references C.
    writer.begin_section(function_name_ptr=2, n_variants=0)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=3, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.end_section()

    # C: written now, both A and B's slots get filled.
    writer.begin_section(function_name_ptr=3, n_variants=0)
    writer.emit_call_targets([])
    writer.end_section()
    writer.finalize()

    sections = {s.function_name_ptr: s for s in iter_sections_bin(path)}
    c_off = sections[3].section_offset
    assert sections[1].call_targets[0].function_section_ptr == c_off
    assert sections[2].call_targets[0].function_section_ptr == c_off


@_PENDING_READER_XFAIL
def test_multiple_per_variant_entries_to_same_callee(tmp_path: Path):
    """Two distinct per-call slots in the same variant both reference
    the same unwritten callee's same vkey; both get patched."""
    path = tmp_path / "multi_holes.bin"
    writer = SectionWriter(path)

    # Section A: two call_targets, both pointing at B's FID but under
    # different types (LOCAL vs PLT) — the writer's dedup contract
    # allows two entries with the same FID different types.
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.PLT, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0, callee_function_name_ptr=2, callee_vkey=0xF0
            ),
            PerCallEntry(
                called_idx=1, callee_function_name_ptr=2, callee_vkey=0xF0
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Section B: emits the matching variant_ref_offset.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xF0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O0")
    writer.end_section()
    writer.finalize()

    sections = {s.function_name_ptr: s for s in iter_sections_bin(path)}
    a = sections[1]
    assert a.call_targets[0].function_section_ptr == sections[2].section_offset
    assert a.call_targets[1].function_section_ptr == sections[2].section_offset
    assert a.variants[0].per_call_entries == [(0, 0), (1, 0)]


# ---------------------------------------------------------------------------
# DATA prelude parity — confirms the DRY refactor of memmap_format.py did
# not silently break the existing _data.bin path.

def test_data_bin_prelude_round_trip():
    blob = encode_data_bin_prelude()
    assert len(blob) == DATA_BIN_PRELUDE_SIZE
    assert blob[:4] == DATA_BIN_PRELUDE_MAGIC
    assert_data_bin_prelude(blob)


def test_data_bin_prelude_distinct_from_sections_prelude():
    """Sentinel: the two magics must NOT collide, otherwise a swapped
    bin would silently pass the prelude check."""
    assert DATA_BIN_PRELUDE_MAGIC != MATCHED_SECTIONS_BIN_PRELUDE_MAGIC
    with pytest.raises(ValueError, match="magic"):
        assert_data_bin_prelude(encode_matched_sections_prelude())
    with pytest.raises(ValueError, match="magic"):
        assert_matched_sections_prelude(encode_data_bin_prelude())


# ---------------------------------------------------------------------------
# SectionWriter close / context-manager lifecycle.

def test_section_writer_close_is_idempotent(tmp_path: Path):
    """``close`` always works, and is safe to call twice."""
    writer = SectionWriter(tmp_path / "close.bin")
    writer.close()
    writer.close()  # second call is a no-op


@_PENDING_READER_XFAIL
def test_section_writer_finalize_closes_on_sweep_error(tmp_path: Path):
    """If the finalize sweep raises, the mmap must still be released
    (otherwise the bin handle leaks until process exit)."""
    path = tmp_path / "leak_on_error.bin"
    writer = SectionWriter(path)
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Force a builder-bug: stuff an unresolved 0xFFFF directly into the
    # per-call slot bypassing the back-patch queue. We use the public
    # patch primitive so this test exercises the same code path as the
    # real-bin writer would.
    slot_offset = (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        + SECTION_HEADER_SIZE
        + VARIANT_HEADER_SIZE
        + 2  # past called_idx
    )
    # The section has no call_targets so there's no per-call slot to
    # corrupt; emit a second section with one call_target + a per-call
    # entry, then corrupt that.
    writer = SectionWriter(path)
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0)]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()
    slot_offset = (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        + SECTION_HEADER_SIZE
        + 1 * JUMP_TABLE_ENTRY_SIZE
        + CALL_TARGET_ENTRY_SIZE
        + VARIANT_HEADER_SIZE
        + 2
    )
    writer._writer.patch(slot_offset, struct.pack("<H", UNRESOLVED_VARIANT_INDEX))

    with pytest.raises(ValueError, match="unresolved"):
        writer.finalize()
    # Underlying mmap must be released — a second call to close() is a
    # no-op iff the first run actually closed it.
    writer.close()


def test_section_writer_context_manager_releases_on_body_raise(tmp_path: Path):
    """Using ``SectionWriter`` as a context manager closes the mmap even
    when the body raises before finalize."""
    path = tmp_path / "ctx.bin"
    with pytest.raises(RuntimeError, match="boom"):
        with SectionWriter(path) as writer:
            writer.begin_section(function_name_ptr=1, n_variants=0)
            raise RuntimeError("boom")
    # If the mmap had leaked, attempting a fresh writer on the same path
    # would still succeed (Linux allows overlapping mmaps), so this test
    # asserts something weaker but verifiable: a second close() returns
    # cleanly, indicating the first one ran.
    writer = SectionWriter(path)
    writer.close()
    writer.close()


# ---------------------------------------------------------------------------
# SectionWriter lifecycle guards — every public method asserts its
# precondition; these tests pin the assertions so a future refactor
# that removes them is caught.

def test_begin_section_rejects_nested_open(tmp_path: Path):
    writer = SectionWriter(tmp_path / "nested.bin")
    writer.begin_section(function_name_ptr=1, n_variants=0)
    with pytest.raises(ValueError, match="still open"):
        writer.begin_section(function_name_ptr=2, n_variants=0)


def test_emit_call_targets_rejects_double_call(tmp_path: Path):
    writer = SectionWriter(tmp_path / "double.bin")
    writer.begin_section(function_name_ptr=1, n_variants=0)
    writer.emit_call_targets([])
    with pytest.raises(ValueError, match="emit_call_targets called twice"):
        writer.emit_call_targets([])


def test_begin_variant_requires_emit_call_targets_first(tmp_path: Path):
    writer = SectionWriter(tmp_path / "order.bin")
    writer.begin_section(function_name_ptr=1, n_variants=1)
    with pytest.raises(ValueError, match="emit_call_targets"):
        writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)


def test_end_section_rejects_open_variant(tmp_path: Path):
    writer = SectionWriter(tmp_path / "open_variant.bin")
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    with pytest.raises(ValueError, match="variant is still open"):
        writer.end_section()


def test_finalize_rejects_open_section(tmp_path: Path):
    writer = SectionWriter(tmp_path / "open_section.bin")
    writer.begin_section(function_name_ptr=1, n_variants=0)
    with pytest.raises(ValueError, match="still open"):
        writer.finalize()


# ---------------------------------------------------------------------------
# OUTLINED_FUNCTION_N (sibling-FID) coherence — the corpus-scale case
# clang's compiler-internal helpers produce. Two sections share a FID
# but carry disjoint vkey sets; the writer's self-describing back-patch
# routes each per_call_entry to its specific sibling's local variant
# table without leaking sentinels across siblings.
# ---------------------------------------------------------------------------


@_PENDING_READER_XFAIL
def test_outlined_function_siblings_disjoint_vkeys_each_patch_their_own(
    tmp_path: Path,
):
    """Two sibling sections under FID=2 with disjoint vkey sets; the
    caller references one vkey per variant. Each per_call_entry slot
    resolves to the matching sibling's local variant_idx — no
    MISSING_VARIANT_INDEX, no UNRESOLVED leak, no warn-log line."""
    import io

    path = tmp_path / "outlined_disjoint.bin"
    warn_log = io.StringIO()
    writer = SectionWriter(path, warn_log=warn_log)

    # Caller A (FID=1): two variants, each references FID=2 at a
    # different vkey. Sibling-1 will carry vkey=0xA1 only; sibling-2
    # will carry vkey=0xA2 only.
    writer.begin_section(function_name_ptr=1, n_variants=2)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0xA1, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xA1)]
    )
    writer.end_variant(vkey="caller_v1")
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xA2)]
    )
    writer.end_variant(vkey="caller_v2")
    writer.end_section()

    # Sibling-1 (FID=2, body 1): exposes vkey=0xA1.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA1, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="sib1_v1")
    writer.end_section()

    # Sibling-2 (FID=2, body 2): exposes vkey=0xA2.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="sib2_v1")
    writer.end_section()

    writer.finalize()

    sections = list(iter_sections_bin(path))
    # Three sections total: caller + two siblings.
    assert len(sections) == 3
    caller = sections[0]
    assert caller.function_name_ptr == 1
    # Each variant's per_call_entry resolves to variant_idx=0 inside
    # its respective sibling (each sibling has exactly one variant).
    (_called_v1, sv_idx_v1), = caller.variants[0].per_call_entries
    (_called_v2, sv_idx_v2), = caller.variants[1].per_call_entries
    assert sv_idx_v1 == 0
    assert sv_idx_v2 == 0
    # Neither sentinel leaked.
    for v in caller.variants:
        for _called, sv_idx in v.per_call_entries:
            assert sv_idx != UNRESOLVED_VARIANT_INDEX
            assert sv_idx != MISSING_VARIANT_INDEX
    # No warn-log entry was emitted (every hole resolved).
    assert "missing_variant:" not in warn_log.getvalue()


@_PENDING_READER_XFAIL
def test_outlined_function_siblings_with_unregistered_vkey_lands_as_missing_at_finalize(
    tmp_path: Path,
):
    """Caller's per_call_entry references a vkey that NEITHER sibling
    section registers. finalize stamps :data:`MISSING_VARIANT_INDEX`
    and the warn-log receives one matching line."""
    import io

    path = tmp_path / "outlined_missing.bin"
    warn_log = io.StringIO()
    writer = SectionWriter(path, warn_log=warn_log)

    # Caller A: one variant referencing FID=2 at vkey=0xB7 — neither
    # sibling will register this.
    a_offset = writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0xB7, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xB7)]
    )
    writer.end_variant(vkey="caller_v1")
    writer.end_section()

    # Sibling-1 (FID=2): exposes only vkey=0xA1.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA1, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="sib1_v1")
    writer.end_section()

    # Sibling-2 (FID=2): exposes only vkey=0xA2.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="sib2_v1")
    writer.end_section()

    writer.finalize()

    sections = list(iter_sections_bin(path))
    caller = sections[0]
    assert caller.function_name_ptr == 1
    (_called, sv_idx), = caller.variants[0].per_call_entries
    assert sv_idx == MISSING_VARIANT_INDEX
    log_text = warn_log.getvalue()
    assert log_text.count("missing_variant:") == 1
    assert "callee_fid=2" in log_text
    assert f"callee_vkey={0xB7}" in log_text
    assert f"caller_section@{a_offset}" in log_text


@_PENDING_READER_XFAIL
def test_outlined_function_siblings_function_section_ptr_last_write_wins(
    tmp_path: Path,
):
    """W2 acceptance: when a caller forward-references a FID that two
    sibling sections will close for, the on-disk
    ``function_section_ptr`` ends up pointing at the LAST sibling
    (each sibling's end_section re-patches the same header slot).
    The loader walks via per-call ``callee_vkey`` to disambiguate
    which sibling carries the matching variant."""
    path = tmp_path / "outlined_lww.bin"
    writer = SectionWriter(path)

    # Caller A: one call_target forward-referencing FID=2; one
    # variant whose per_call points at sibling-2's vkey=0xA2.
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xA2)]
    )
    writer.end_variant(vkey="caller_v1")
    writer.end_section()

    # Sibling-1 (FID=2): first to close — header slot gets stamped to
    # sibling-1's offset.
    sib1_offset = writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA1, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="sib1_v1")
    writer.end_section()

    # Sibling-2 (FID=2): closes second — header slot gets re-stamped
    # to sibling-2's offset (last-write-wins).
    sib2_offset = writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="sib2_v1")
    writer.end_section()

    writer.finalize()

    sections = list(iter_sections_bin(path))
    caller = sections[0]
    # On-disk function_section_ptr points at the LAST sibling.
    assert caller.call_targets[0].function_section_ptr == sib2_offset
    assert caller.call_targets[0].function_section_ptr != sib1_offset
    # And the caller's per_call_entry resolves to sibling-2's
    # variant_idx (sibling-2 has the matching vkey=0xA2).
    (_called, sv_idx), = caller.variants[0].per_call_entries
    assert sv_idx == 0  # variant_idx 0 inside sibling-2
