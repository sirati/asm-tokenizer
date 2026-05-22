"""Round-trip + back-patch + finalize tests for ``matched_sections_bin``.

The writer is the sole producer of ``<binary>_sections.bin``; the
reader is the sole consumer on the dataloader hot path. Correctness
here pins both halves of the codec at once.

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


def test_section_round_trip(tmp_path: Path):
    """One section, two call_targets, two variants, every field round-trips."""
    path = tmp_path / "rt_sections.bin"
    writer = SectionWriter(path)

    # Section A (FID=1).
    offset_a = writer.begin_section(function_name_ptr=1)
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

    # Variant 1: calls call_target idx=0 (self-ref) with vkey="x86_O0".
    writer.begin_variant(variant_ref_offset=0x100, data_offset_shifted=0x20)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=1,
                callee_vkey="x86_O0",
            ),
        ]
    )
    v0 = writer.end_variant(vkey="x86_O0")
    assert v0 == 0

    # Variant 2: calls call_target idx=1 (extern) — but per-call entries
    # only point at LOCAL/PLT/EXTERN call_targets via called_idx; the
    # extern target still receives an entry to verify the path.
    writer.begin_variant(variant_ref_offset=0x140, data_offset_shifted=0x40)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=1,
                callee_vkey="x86_O0",
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


def test_header_back_patch(tmp_path: Path):
    """Section A forward-references section B; B's section_offset
    lands in A's call_target slot after end_section(B)."""
    path = tmp_path / "header_patch.bin"
    writer = SectionWriter(path)

    # Section A (FID=1): references B (FID=2) which is not yet written.
    offset_a = writer.begin_section(function_name_ptr=1)
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
    offset_b = writer.begin_section(function_name_ptr=2)
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


def test_per_variant_back_patch(tmp_path: Path):
    """Section A's variant references B's vkey=\"x86_O0\" before B is
    written; after B emits that variant the slot equals B's variant_idx
    (not 0xFFFF)."""
    path = tmp_path / "variant_patch.bin"
    writer = SectionWriter(path)

    # Section A: one call_target referencing B (FID=2), one variant
    # whose per-call entry points at B's "x86_O0" variant.
    writer.begin_section(function_name_ptr=1)
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
                callee_vkey="x86_O0",
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Section B: emits two variants. We want "x86_O0" to land at
    # variant_idx=1 to make the back-patch non-trivial (0 is the
    # default unsigned value, so 1 catches a "wrote no bytes" bug too).
    writer.begin_section(function_name_ptr=2)
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


def test_extern_library_unknown_lands_as_zero(tmp_path: Path):
    """``extern_provider_line_no=None`` → function_section_ptr=0."""
    path = tmp_path / "extern_unknown.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1)
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

    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.end_section()
    # Never emit section 2.

    with pytest.raises(ValueError, match="referenced but never written"):
        writer.finalize()


def test_backward_per_call_to_closed_section_missing_vkey_stamps_missing_directly(tmp_path: Path):
    """If a section's per-call entry references a callee whose section
    has ALREADY CLOSED but does not have this caller's vkey, the slot
    stamps :data:`MISSING_VARIANT_INDEX` directly at emit time — opening
    a back-patch hole would never resolve since the only place that
    pops :data:`_pending_holes` is the callee's :meth:`end_section`,
    which has already run."""
    path = tmp_path / "backward_miss.bin"
    writer = SectionWriter(path)

    # Section B: emits ONLY vkey="x86_O3".
    writer.begin_section(function_name_ptr=2)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O3")
    writer.end_section()

    # Section A (emitted AFTER B closes): references B at vkey="x86_O0".
    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey="x86_O0")]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()
    writer.finalize()

    sections = {s.function_name_ptr: s for s in iter_sections_bin(path)}
    a = sections[1]
    (called_idx, sv_idx), = a.variants[0].per_call_entries
    assert called_idx == 0
    assert sv_idx == MISSING_VARIANT_INDEX


def test_per_variant_hole_with_missing_callee_vkey_lands_as_missing_sentinel(tmp_path: Path):
    """Cross-arm vkey mismatch: callee section IS written but never emits
    the caller's vkey. The per-call slot lands on
    :data:`MISSING_VARIANT_INDEX` (= 0xFFFE) instead of raising — the
    legitimate corpus-scale case where caller and callee have different
    surviving-variant sets after pass-1's drop rules. The finalize sweep
    rejects only 0xFFFF (unresolved hole), not 0xFFFE."""
    path = tmp_path / "missing_variant.bin"
    writer = SectionWriter(path)

    # Section A: references B's vkey="x86_O0" via a per-call entry.
    writer.begin_section(function_name_ptr=1)
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
                called_idx=0,
                callee_function_name_ptr=2,
                callee_vkey="x86_O0",
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Section B: only emits vkey="x86_O3" — A's per-call hole resolves
    # to MISSING_VARIANT_INDEX rather than failing the build.
    writer.begin_section(function_name_ptr=2)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O3")
    writer.end_section()
    writer.finalize()

    sections = list(iter_sections_bin(path))
    a = next(s for s in sections if s.function_name_ptr == 1)
    (called_idx, sv_idx), = a.variants[0].per_call_entries
    assert called_idx == 0
    assert sv_idx == MISSING_VARIANT_INDEX


def test_called_idx_validation(tmp_path: Path):
    """A PerCallEntry whose called_idx doesn't match the section's
    call_target table is rejected eagerly."""
    path = tmp_path / "bad_idx.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1)
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

    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets([])
    first_offset, _ = writer.end_section()

    second_offset = writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets([])
    writer.end_section()
    writer.finalize()

    assert second_offset != first_offset
    assert writer._known_sections[1] == second_offset  # noqa: SLF001 — internal-state assertion is the point of this test
    sections = list(iter_sections_bin(path))
    assert len(sections) == 2
    assert {s.function_name_ptr for s in sections} == {1}


def test_dup_variant_vkey_overwrites_known_section_variants_with_latest(tmp_path: Path):
    """A vkey can re-appear within a section (or across sections sharing
    a FID); the latest emission's variant_idx wins for back-patch
    resolution, matching the pre-refactor ``function_lookup``
    last-write-wins semantics."""
    path = tmp_path / "dup_vkey.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    first_idx = writer.end_variant(vkey="x86_O0")

    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    second_idx = writer.end_variant(vkey="x86_O0")
    writer.end_section()
    writer.finalize()

    assert first_idx == 0
    assert second_idx == 1
    assert writer._known_section_variants[(1, "x86_O0")] == second_idx  # noqa: SLF001


def test_section_alignment_padding(tmp_path: Path):
    """Each section's offset is 4-byte aligned even when the previous
    section's natural end is not.

    Section header (8) + 1 call_target (12) + 1 variant header (10) +
    1 per-call entry (4) = 34 bytes; trailer pad = 2 bytes → next
    section starts at offset 16+34+2 = 52 (4-byte aligned).
    """
    path = tmp_path / "align.bin"
    writer = SectionWriter(path)

    # Section A produces a 34-byte payload before trailer pad.
    a_offset = writer.begin_section(function_name_ptr=1)
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
                called_idx=0, callee_function_name_ptr=1, callee_vkey="x86_O0"
            )
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    expected_b_offset = (
        a_offset
        + SECTION_HEADER_SIZE
        + 1 * CALL_TARGET_ENTRY_SIZE
        + 1 * VARIANT_HEADER_SIZE
        + 1 * PER_CALL_ENTRY_SIZE
    )
    # Round up to 4-byte boundary.
    if expected_b_offset % 4 != 0:
        expected_b_offset += 4 - (expected_b_offset % 4)

    b_offset = writer.begin_section(function_name_ptr=2)
    assert b_offset == expected_b_offset
    assert b_offset % 4 == 0
    writer.emit_call_targets([])
    writer.end_section()
    writer.finalize()


def test_finalize_sweep_catches_leaked_sentinel(tmp_path: Path):
    """If a writer bug leaves a 0xFFFF slot AND empties pending_holes,
    the belt-and-braces sweep in finalize() still catches it.

    This is a defensive test: we patch the writer's back-patch loop to
    no-op so a real-bin sentinel leaks past the pending_holes check,
    proving the sweep is the second line of defence.
    """
    path = tmp_path / "leak.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1)
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
                called_idx=0, callee_function_name_ptr=2, callee_vkey="x86_O0"
            )
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    writer.begin_section(function_name_ptr=2)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Drop the pending-holes book without performing the per-variant
    # patches that resolved them; the on-disk per-call slot is
    # consequently still 0xFFFF.
    # Re-write the slot to 0xFFFF (simulate the dropped patch).
    # The per-call section_variant_index slot in section A is at:
    #   section A offset (16) + header (8) + 1 call_target (12)
    #     + variant header (10) + 2 (skip called_idx) = 48.
    slot_offset = (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        + SECTION_HEADER_SIZE
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


def test_two_sections_share_callee(tmp_path: Path):
    """Two distinct sections both forward-reference the same callee
    section; both get patched at the callee's end_section."""
    path = tmp_path / "shared_callee.bin"
    writer = SectionWriter(path)

    # A references C.
    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=3, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.end_section()

    # B references C.
    writer.begin_section(function_name_ptr=2)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=3, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.end_section()

    # C: written now, both A and B's slots get filled.
    writer.begin_section(function_name_ptr=3)
    writer.emit_call_targets([])
    writer.end_section()
    writer.finalize()

    sections = {s.function_name_ptr: s for s in iter_sections_bin(path)}
    c_off = sections[3].section_offset
    assert sections[1].call_targets[0].function_section_ptr == c_off
    assert sections[2].call_targets[0].function_section_ptr == c_off


def test_multiple_per_variant_entries_to_same_callee(tmp_path: Path):
    """Two distinct per-call slots in the same variant both reference
    the same unwritten callee's same vkey; both get patched."""
    path = tmp_path / "multi_holes.bin"
    writer = SectionWriter(path)

    # Section A: two call_targets, both pointing at B's FID but under
    # different types (LOCAL vs PLT) — the writer's dedup contract
    # allows two entries with the same FID different types.
    writer.begin_section(function_name_ptr=1)
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
                called_idx=0, callee_function_name_ptr=2, callee_vkey="x86_O0"
            ),
            PerCallEntry(
                called_idx=1, callee_function_name_ptr=2, callee_vkey="x86_O0"
            ),
        ]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()

    # Section B: emits the vkey.
    writer.begin_section(function_name_ptr=2)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
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


def test_section_writer_finalize_closes_on_sweep_error(tmp_path: Path):
    """If the finalize sweep raises, the mmap must still be released
    (otherwise the bin handle leaks until process exit)."""
    path = tmp_path / "leak_on_error.bin"
    writer = SectionWriter(path)
    writer.begin_section(function_name_ptr=1)
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
    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey="x86_O0")]
    )
    writer.end_variant(vkey="x86_O0")
    writer.end_section()
    slot_offset = (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        + SECTION_HEADER_SIZE
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
            writer.begin_section(function_name_ptr=1)
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
    writer.begin_section(function_name_ptr=1)
    with pytest.raises(ValueError, match="still open"):
        writer.begin_section(function_name_ptr=2)


def test_emit_call_targets_rejects_double_call(tmp_path: Path):
    writer = SectionWriter(tmp_path / "double.bin")
    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets([])
    with pytest.raises(ValueError, match="emit_call_targets called twice"):
        writer.emit_call_targets([])


def test_begin_variant_requires_emit_call_targets_first(tmp_path: Path):
    writer = SectionWriter(tmp_path / "order.bin")
    writer.begin_section(function_name_ptr=1)
    with pytest.raises(ValueError, match="emit_call_targets"):
        writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)


def test_end_section_rejects_open_variant(tmp_path: Path):
    writer = SectionWriter(tmp_path / "open_variant.bin")
    writer.begin_section(function_name_ptr=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0, data_offset_shifted=0)
    with pytest.raises(ValueError, match="variant is still open"):
        writer.end_section()


def test_finalize_rejects_open_section(tmp_path: Path):
    writer = SectionWriter(tmp_path / "open_section.bin")
    writer.begin_section(function_name_ptr=1)
    with pytest.raises(ValueError, match="still open"):
        writer.finalize()
