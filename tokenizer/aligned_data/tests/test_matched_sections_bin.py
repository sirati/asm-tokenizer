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

import hashlib
import struct
from pathlib import Path

import numpy as np
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
    _padded_jump_table_bytes,
    _variant_block_offset,
    iter_sections_bin,
    parse_section_bin,
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

    # Variant 1: calls call_target idx=0 (self-ref). The per-call
    # entry inherits its owning caller variant's
    # ``variant_ref_offset`` as the callee_vkey (Step 7's on-wire
    # invariant), so the writer's parse-and-match resolves the slot
    # to the callee variant whose ``variant_ref_offset`` matches —
    # variant_idx=0 here, since this is the same section.
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

    # Variant 2: also calls call_target idx=0 (self-ref); its
    # ``variant_ref_offset`` is different from variant 1's, so the
    # per-call entry's callee_vkey matches a different variant_idx
    # — proves the matching is by vkey not by variant order.
    writer.begin_variant(variant_ref_offset=0x140, data_offset_shifted=0x40)
    writer.emit_per_call_entries(
        [
            PerCallEntry(
                called_idx=0,
                callee_function_name_ptr=1,
                callee_vkey=0x140,
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
    # variant 2's per-call points at idx 0, resolved to variant_idx 1
    # of FID=1's section (the variant whose vkey is 0x140).
    assert section.variants[1].per_call_entries == [(0, 1)]


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


def test_duplicated_marker_round_trips_and_keeps_clean_fid(tmp_path: Path):
    """The ``duplicated`` flag rides bit 31 of the header FID, round-trips
    through ``parse_section_bin`` as ``Section.is_duplicated``, and leaves
    the FID (and every emit-time identity compare) clean.

    Two same-FID sibling sections (FID=1, both duplicated) plus one
    unmarked section (FID=2) that LOCAL-references FID=1: the call_target
    row carries the CLEAN FID and resolves to a real section offset, proving
    the marker bit never leaks into call_target equality.
    """
    path = tmp_path / "dup_marker.bin"
    writer = SectionWriter(path)

    # Sibling A (FID=1, duplicated): one variant.
    writer.begin_section(function_name_ptr=1, n_variants=1, duplicated=True)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0x10, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey=0x10)
    writer.end_section()

    # Sibling B (FID=1, duplicated): a distinct body sharing the FID.
    writer.begin_section(function_name_ptr=1, n_variants=1, duplicated=True)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0x20, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey=0x20)
    writer.end_section()

    # Section C (FID=2, NOT duplicated): LOCAL-references FID=1.
    writer.begin_section(function_name_ptr=2, n_variants=0)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=False
            ),
        ]
    )
    writer.end_section()
    writer.finalize()

    sections = list(iter_sections_bin(path))
    assert [s.function_name_ptr for s in sections] == [1, 1, 2]
    assert [s.is_duplicated for s in sections] == [True, True, False]
    # The call_target row carries the CLEAN FID (1) and was resolved to a
    # real section offset (the last sibling, last-write-wins) -- the marker
    # bit never participated in the identity compare.
    section_c = sections[2]
    assert section_c.call_targets[0].function_name_ptr == 1
    assert section_c.call_targets[0].function_section_ptr == sections[1].section_offset


def test_begin_section_rejects_fid_using_reserved_marker_bit(tmp_path: Path):
    """A FID that would collide with the duplicated-marker bit is rejected
    up front rather than silently corrupting the marker."""
    writer = SectionWriter(tmp_path / "fid_overflow.bin")
    try:
        with pytest.raises(ValueError, match="duplicated marker"):
            writer.begin_section(function_name_ptr=(1 << 31), n_variants=0)
    finally:
        writer.close()


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
    # Step 7's on-wire invariant: the per-call entry's callee_vkey
    # equals its owning caller variant's variant_ref_offset (since
    # the same _variants.bin byte position is shared when caller and
    # callee carry the same VersionKey).
    writer.begin_variant(variant_ref_offset=0x50, data_offset_shifted=0x20)
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
    assert writer._known_sections.get(1) == second_offset  # noqa: SLF001 — internal-state assertion is the point of this test
    sections = list(iter_sections_bin(path))
    assert len(sections) == 2
    assert {s.function_name_ptr for s in sections} == {1}


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

    Section header (8) + padded jump-table region (4 for n_variants=1)
    + 1 call_target (12) + 1 variant header (8) + 1 per-call entry (4)
    = 36 bytes; already u32-aligned so trailer pad = 0 → next section
    starts at offset 16+36 = 52.
    """
    path = tmp_path / "align.bin"
    writer = SectionWriter(path)

    # Section A produces a 36-byte payload (no trailer pad needed). The
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
        + _padded_jump_table_bytes(1)
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


def test_finalize_sweep_catches_leaked_sentinel(tmp_path: Path):
    """If a writer bug leaves a 0xFFFF slot AND empties pending_holes,
    the belt-and-braces sweep in finalize() still catches it.

    This is a defensive test: we simulate a buggy writer that
    forgot to mark a caller as waiting on a callee (the
    ``pending_holes`` book got cleared) but ALSO failed to back-patch
    the per-call slot, so an UNRESOLVED sentinel leaks past every
    structural check. The sweep is the second line of defence.
    """
    path = tmp_path / "leak.bin"
    writer = SectionWriter(path)

    # A normal forward-ref setup: caller A references callee B at
    # vkey 0xE0; B exposes that vkey, so the sibling-close path
    # legitimately resolves the per-call slot.
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0xE0, data_offset_shifted=0)
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

    # Simulate a writer bug: pending_holes got wiped before finalize
    # could run its MISSING-stamp sweep, AND the per-call slot was
    # corrupted back to UNRESOLVED (e.g. a bad partial patch
    # restored the placeholder). The sweep is the only thing that
    # can catch this combination.
    slot_offset = (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        + SECTION_HEADER_SIZE
        + _padded_jump_table_bytes(1)
        + CALL_TARGET_ENTRY_SIZE
        + VARIANT_HEADER_SIZE
        + 2  # skip u16 called_idx; section_variant_index is the second field
    )
    writer._writer.patch(slot_offset, struct.pack("<H", UNRESOLVED_VARIANT_INDEX))
    writer._pending_holes.clear()  # noqa: SLF001 — simulating writer-bug

    with pytest.raises(ValueError, match="unresolved section_variant_index"):
        writer.finalize()


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
    writer.begin_variant(variant_ref_offset=0xF0, data_offset_shifted=0)
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
        + _padded_jump_table_bytes(1)
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


# ---------------------------------------------------------------------------
# On-disk variant ordering — variants are flushed in
# ``variant_ref_offset``-ascending order regardless of caller emit order.
# ---------------------------------------------------------------------------


def test_variants_flushed_in_vref_sorted_order(tmp_path: Path):
    """The writer buffers variants between begin_section and end_section
    and flushes them in ``variant_ref_offset``-ascending order at
    :meth:`end_section`. ``data_offset_shifted`` is paired with each
    header so the (header, per_call) pairings must survive the reorder.
    Stable sort: equal vrefs keep their declared sub-order.

    Asserts both halves: on-disk vrefs are sorted AND ``data_offset_shifted``
    travels with the matching variant_ref_offset across the reorder.
    """
    path = tmp_path / "sort_order.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=5)
    writer.emit_call_targets([])
    # Declared-emit order is intentionally not sorted; ``data_offset_shifted``
    # is ``vref << 1`` so the pairing is recoverable after reorder.
    declared = [0x50, 0x10, 0x30, 0x40, 0x20]
    for vref in declared:
        writer.begin_variant(variant_ref_offset=vref, data_offset_shifted=vref << 1)
        writer.emit_per_call_entries([])
        writer.end_variant(vkey=vref)
    writer.end_section()
    writer.finalize()

    section = next(iter_sections_bin(path))
    on_disk_vrefs = [v.variant_ref_offset for v in section.variants]
    on_disk_data = [v.data_offset_shifted for v in section.variants]
    assert on_disk_vrefs == sorted(declared)
    # Each header's data_offset_shifted travels with its vref under the reorder.
    assert on_disk_data == [vref << 1 for vref in on_disk_vrefs]


def test_variants_sort_is_stable_on_equal_vrefs(tmp_path: Path):
    """Two variants with the same ``variant_ref_offset`` keep their
    declared sub-order on disk — the writer's sibling-close
    back-patch uses ``searchsorted(side="right") - 1`` to pick the
    LAST equal entry, which only matches the legacy last-write-wins
    semantic when the on-disk equal-vref run is in declared order.
    """
    path = tmp_path / "stable_sort.bin"
    writer = SectionWriter(path)

    writer.begin_section(function_name_ptr=1, n_variants=4)
    writer.emit_call_targets([])
    # vrefs intentionally include a repeated key (0x20). ``data_offset_shifted``
    # is unique per declared variant so we can tell them apart after reorder.
    pairs = [(0x20, 0xAA), (0x10, 0xBB), (0x20, 0xCC), (0x30, 0xDD)]
    for vref, data in pairs:
        writer.begin_variant(variant_ref_offset=vref, data_offset_shifted=data)
        writer.emit_per_call_entries([])
        writer.end_variant(vkey=(vref, data))
    writer.end_section()
    writer.finalize()

    section = next(iter_sections_bin(path))
    on_disk = [(v.variant_ref_offset, v.data_offset_shifted) for v in section.variants]
    # Sort key = vref ascending; ties broken by declared order.
    assert on_disk == [(0x10, 0xBB), (0x20, 0xAA), (0x20, 0xCC), (0x30, 0xDD)]


def test_per_call_entries_follow_their_variant_under_sort(tmp_path: Path):
    """Per-call entries are flushed paired with their variant header,
    so when the writer reorders variants by ``variant_ref_offset``,
    each variant's per-call entries travel with it (not with the
    original declared-emit position).
    """
    path = tmp_path / "per_call_with_variant.bin"
    writer = SectionWriter(path)

    # Self-call setup: the section's own call_target points at FID=1,
    # and each variant's per-call entry references its OWN
    # ``variant_ref_offset`` so the self-resolve sweep at end_section
    # stamps each slot with the (post-sort) variant_idx of the
    # matching local variant. After sort: vrefs are [0x10, 0x20,
    # 0x30, 0x40], so each per-call slot stamps the SORTED index
    # of its own vref.
    writer.begin_section(function_name_ptr=1, n_variants=4)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    declared = [0x30, 0x10, 0x40, 0x20]
    for vref in declared:
        writer.begin_variant(variant_ref_offset=vref, data_offset_shifted=vref << 1)
        writer.emit_per_call_entries(
            [
                PerCallEntry(
                    called_idx=0, callee_function_name_ptr=1, callee_vkey=vref
                )
            ]
        )
        writer.end_variant(vkey=vref)
    writer.end_section()
    writer.finalize()

    section = next(iter_sections_bin(path))
    on_disk_vrefs = [v.variant_ref_offset for v in section.variants]
    assert on_disk_vrefs == sorted(declared)
    # The self-call per-call slot for variant @ sorted_idx i references
    # its own vref, which lives at sorted_idx i — so every slot stamps i.
    for sorted_idx, variant in enumerate(section.variants):
        (called_idx, sv_idx), = variant.per_call_entries
        assert called_idx == 0
        assert sv_idx == sorted_idx, (
            f"per-call entry at sorted_idx={sorted_idx} should point at the "
            f"same variant (self-call to own vref), got sv_idx={sv_idx}"
        )


# ---------------------------------------------------------------------------
# np.view() write-through proof
# ---------------------------------------------------------------------------


def test_numpy_view_writes_persist_to_bin(tmp_path: Path):
    """In-place numpy writes through a ``.view(uint32)`` go through to
    the mmap-backed bin.

    Downstream vectorised hole-fills reinterpret the already-written
    bin as ``uint32`` and assign into it. For that to be a real
    in-place write (and not a silent copy that gets discarded), the
    numpy array must be backed by the mmap region directly. This test
    pins the property: open a writer, emit one section with an odd
    ``n_variants=3`` (so jump-table padding is exercised), patch a
    known u32-aligned offset via a ``.view(uint32)`` slice, finalize,
    re-read the file, and assert the patched bytes survive. The
    padding bytes themselves are also asserted to be zero on disk.
    """
    path = tmp_path / "view_writethrough.bin"
    writer = SectionWriter(path)

    # Section with n_variants=3 — jump-table region is 6 bytes raw, 8
    # bytes once padded. The trailing u16 (bytes 6..7 of the table)
    # must be zero on disk; the 8 bytes are guaranteed u32-aligned.
    section_offset = writer.begin_section(function_name_ptr=1, n_variants=3)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    for vkey in (0x10, 0x20, 0x30):
        writer.begin_variant(variant_ref_offset=vkey, data_offset_shifted=0)
        writer.emit_per_call_entries(
            [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=vkey)]
        )
        writer.end_variant(vkey=vkey)
    writer.end_section()

    # Build a writable u8 numpy array over the already-written region
    # of the underlying mmap. ``np.frombuffer`` on a memoryview slice
    # shares storage with the mmap; ``setflags(write=True)`` is the
    # explicit opt-in for numpy's safety check.
    cursor = writer._writer.cursor  # noqa: SLF001 — exercising the mmap
    bin_u8 = np.frombuffer(writer._writer._mm, dtype=np.uint8, count=cursor)  # noqa: SLF001
    bin_u8.setflags(write=True)

    # Reinterpret as u32. The view shares storage with bin_u8, which
    # shares storage with the mmap. An in-place assignment here MUST
    # land in the file.
    bin_u32 = bin_u8.view(np.uint32)

    # Pick a u32-aligned slot that we can verify post-finalize: the
    # call_target row's ``function_section_ptr`` u32 is at
    # section_offset + SECTION_HEADER_SIZE + padded_jump_table + 4
    # (the second u32 of the row). All operands are multiples of 4 so
    # the byte offset divides by 4 and the u32 index is well-defined.
    ptr_byte_offset = (
        section_offset
        + SECTION_HEADER_SIZE
        + _padded_jump_table_bytes(3)
        + 4  # skip the first u32 of the call_target row (function_name_ptr)
    )
    assert ptr_byte_offset % 4 == 0, "test invariant: u32-aligned patch site"
    u32_index = ptr_byte_offset // 4

    sentinel = 0xDEADBEEF
    bin_u32[u32_index] = sentinel

    # Drop both numpy refs before finalize: they hold buffer pointers
    # into the mmap and ``mmap.close`` (called from
    # :meth:`MemmapBinWriter.finalize`) refuses while exported
    # pointers exist. This mirrors what real downstream code paths
    # must do — the .view() handle is short-lived around the
    # vectorised hole-fill, not held across finalize.
    del bin_u32
    del bin_u8

    # Finalize (truncates + closes); reopen from disk and verify the
    # sentinel survived as little-endian bytes.
    writer.finalize()
    raw = path.read_bytes()
    on_disk = struct.unpack_from("<I", raw, ptr_byte_offset)[0]
    assert on_disk == sentinel, (
        f"in-place .view(uint32) write at byte offset {ptr_byte_offset} "
        f"did not persist: expected 0x{sentinel:08X}, got 0x{on_disk:08X}"
    )

    # Jump-table padding (the trailing u16 for n_variants=3) must be a
    # deterministic zero on disk — the reader skips it but a non-zero
    # value would tell us garbage leaked through the reservation.
    pad_offset = (
        section_offset
        + SECTION_HEADER_SIZE
        + 3 * JUMP_TABLE_ENTRY_SIZE  # raw table width (the three real slots)
    )
    table_end = section_offset + SECTION_HEADER_SIZE + _padded_jump_table_bytes(3)
    assert pad_offset + 2 == table_end, (
        "test invariant: the padding u16 occupies the final 2 bytes of "
        "the padded jump-table region"
    )
    pad_value = struct.unpack_from("<H", raw, pad_offset)[0]
    assert pad_value == 0, (
        f"jump-table padding u16 at byte offset {pad_offset} should be "
        f"zero, got 0x{pad_value:04X}"
    )


# ---------------------------------------------------------------------------
# Vectorized hole-fill byte-equivalence proof
# ---------------------------------------------------------------------------


def _reference_resolve_caller_section(
    self,
    *,
    caller_section_offset: int,
    callee_fid: int,
    callee_section_offset: int,
    callee_sorted_vrefs: "np.ndarray",
    callee_sort_order: "np.ndarray",
    context: str = "Step3-sibling-close",
) -> None:
    """Python-loop reference implementation of ``_resolve_caller_section``.

    Re-derives the same (vkey -> variant_idx) dict the pre-vectorized
    code consulted from the (sorted_vrefs, sort_order) pair, then walks
    the caller section row-by-row exactly as the legacy code did:
    Case A patches ``function_section_ptr`` per call_target via
    :meth:`MemmapBinWriter.patch`; Case B re-walks the per-call entries
    and patches each ``UNRESOLVED`` slot whose owning variant's vkey is
    in the callee's table. Used as the oracle for the byte-equivalence
    test below.
    """
    # Recover vkey_to_idx with last-write-wins among equal vrefs —
    # iterating in sorted order means the last appearance of an equal
    # vref overwrites earlier ones, which matches the legacy dict
    # comprehension's behaviour on a stably-sorted input.
    callee_vkey_to_idx: dict[int, int] = {}
    for sorted_pos in range(callee_sorted_vrefs.shape[0]):
        callee_vkey_to_idx[int(callee_sorted_vrefs[sorted_pos])] = int(
            callee_sort_order[sorted_pos]
        )

    blob = self._writer.view()
    try:
        caller, _end = parse_section_bin(blob, caller_section_offset)
    finally:
        blob.release()

    call_targets_start = (
        caller_section_offset
        + SECTION_HEADER_SIZE
        + _padded_jump_table_bytes(len(caller.variants))
    )
    target_called_idxs: set[int] = set()
    packed_section_offset = struct.pack("<I", callee_section_offset)
    for i, ct in enumerate(caller.call_targets):
        if ct.function_name_ptr != callee_fid:
            continue
        target_called_idxs.add(i)
        row_offset = call_targets_start + i * CALL_TARGET_ENTRY_SIZE
        self._writer.patch(row_offset + 4, packed_section_offset)

    if not target_called_idxs:
        return

    for v_idx, variant in enumerate(caller.variants):
        resolved_idx = callee_vkey_to_idx.get(variant.variant_ref_offset)
        if resolved_idx is None:
            continue
        variant_offset = _variant_block_offset(caller, v_idx)
        entry_offset = variant_offset + VARIANT_HEADER_SIZE
        packed_resolved = struct.pack("<H", resolved_idx)
        for called_idx, sv_idx in variant.per_call_entries:
            slot_offset = entry_offset + 2
            entry_offset += PER_CALL_ENTRY_SIZE
            if called_idx not in target_called_idxs:
                continue
            if sv_idx != UNRESOLVED_VARIANT_INDEX:
                continue
            self._writer.patch(slot_offset, packed_resolved)


def _build_byte_equivalence_fixture(path: Path) -> None:
    """Drive a writer through a multi-section scenario that exercises
    every hole-fill case: sibling FIDs, a self-call, a caller variant
    whose vkey no callee sibling carries (MISSING-stamp at finalize),
    and an odd ``n_variants`` so the padded jump-table region is also
    on the path. Used by the vectorized vs. reference oracle test.
    """
    writer = SectionWriter(path)

    # Caller section (FID=1) — forward-refs FID=2 (LOCAL) and itself
    # (LOCAL, exercising the self-call hole-fill path). ``n_variants=3``
    # so the jump-table padding bytes are part of the captured bytes.
    writer.begin_section(function_name_ptr=1, n_variants=3)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=2, type=CallTargetType.LOCAL, is_matched=True
            ),
            CallTargetSpec(
                function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    # Variant 0: vkey=0xA1 (matched by sibling-1 of FID=2 below).
    writer.begin_variant(variant_ref_offset=0xA1, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xA1),
            PerCallEntry(called_idx=1, callee_function_name_ptr=1, callee_vkey=0xA1),
        ]
    )
    writer.end_variant(vkey=0xA1)
    # Variant 1: vkey=0xA2 (matched by sibling-2 of FID=2 below).
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xA2),
        ]
    )
    writer.end_variant(vkey=0xA2)
    # Variant 2: vkey=0xFE — no sibling carries this, MISSING-stamped
    # at finalize. Exercises both the per-call hole + the MISSING path.
    writer.begin_variant(variant_ref_offset=0xFE, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [
            PerCallEntry(called_idx=0, callee_function_name_ptr=2, callee_vkey=0xFE),
        ]
    )
    writer.end_variant(vkey=0xFE)
    writer.end_section()

    # Sibling-1 of FID=2 — carries vkey=0xA1.
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA1, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey=0xA1)
    writer.end_section()

    # Sibling-2 of FID=2 — carries vkey=0xA2 (disjoint set; last
    # sibling wins in ``_known_sections`` for the FID).
    writer.begin_section(function_name_ptr=2, n_variants=1)
    writer.emit_call_targets([])
    writer.begin_variant(variant_ref_offset=0xA2, data_offset_shifted=0)
    writer.emit_per_call_entries([])
    writer.end_variant(vkey=0xA2)
    writer.end_section()

    writer.finalize()


def test_vectorized_hole_fill_byte_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vectorized ``_resolve_caller_section`` produces the same
    on-disk bytes as a correctness-preserving Python-loop oracle.

    Runs the same multi-section fixture twice — once with the live
    vectorized writer, once with a monkey-patched reference impl that
    re-implements the legacy dict-lookup loop on the new signature —
    and compares the raw bytes byte-for-byte. The fixture covers
    sibling FIDs (sibling close patches), a self-call, a caller-missing
    vkey (finalize-time MISSING stamp), and an odd ``n_variants`` so
    jump-table padding is also in the captured region.
    """
    vec_path = tmp_path / "vectorized.bin"
    _build_byte_equivalence_fixture(vec_path)
    vectorized_bytes = vec_path.read_bytes()

    ref_path = tmp_path / "reference.bin"
    monkeypatch.setattr(
        SectionWriter,
        "_resolve_caller_section",
        _reference_resolve_caller_section,
    )
    _build_byte_equivalence_fixture(ref_path)
    reference_bytes = ref_path.read_bytes()

    assert len(vectorized_bytes) == len(reference_bytes), (
        "fixture lengths diverge: vectorized="
        f"{len(vectorized_bytes)} reference={len(reference_bytes)}"
    )
    if vectorized_bytes != reference_bytes:
        # Surface the first differing byte to make a real failure
        # diagnosable instead of just "bytes differ".
        for i, (a, b) in enumerate(zip(vectorized_bytes, reference_bytes)):
            if a != b:
                raise AssertionError(
                    f"vectorized bytes diverge from reference at offset {i}: "
                    f"vectorized=0x{a:02X} reference=0x{b:02X}"
                )


def _build_forward_ref_chain(path: Path, n: int) -> None:
    """Long forward-reference chain + self/back edges, every section
    carrying multiple variants — stresses the jump-table cumsum
    back-patch addressing across many sections at once.

    Section ``i`` declares LOCAL call_targets to ``i+1`` (forward ref),
    ``i`` (self-call), and ``i-1`` (backward ref), then emits three
    variants each per-calling all of them.
    """
    writer = SectionWriter(path)
    for i in range(n):
        callees: list[CallTargetSpec] = []
        seen: list[int] = []
        for fid in (i + 1, i, max(i - 1, 0)):
            if fid not in seen and 0 <= fid < n:
                seen.append(fid)
                callees.append(
                    CallTargetSpec(
                        function_name_ptr=fid,
                        type=CallTargetType.LOCAL,
                        is_matched=True,
                    )
                )
        writer.begin_section(function_name_ptr=i, n_variants=3)
        writer.emit_call_targets(callees)
        for v in range(3):
            vref = 0x1000 + v
            writer.begin_variant(variant_ref_offset=vref, data_offset_shifted=v)
            writer.emit_per_call_entries(
                [
                    PerCallEntry(
                        called_idx=ci,
                        callee_function_name_ptr=ct.function_name_ptr,
                        callee_vkey=vref,
                    )
                    for ci, ct in enumerate(callees)
                ]
            )
            writer.end_variant(vkey=vref)
        writer.end_section()
    writer.finalize()


# Frozen sha256 digests captured on BASE e468e3c (the pre-refactor
# parse_section_bin writer) and re-verified on the jump-table-native
# tip — the back-patch refactor is addressing-only, so these scenarios
# must hash identically forever. A digest mismatch means the on-disk
# section bytes changed; that is a semantic regression, never a benign
# diff. See the writer's _SectionLayoutView docstring.
_GATE2_FROZEN_DIGESTS = {
    "dense": (
        "1f8f9e65ae152418a623055c1ffe46deafdbf76de293094125a3e5e161d9dec2"
    ),
    "chain50": (
        "22a3607c94fe5b3874e036fd2baaed67276a4c6afd197c466d34a64399f46bcc"
    ),
    "chain400": (
        "bc8e92cb201637f36dc53ca1d4c025144005036fb279090c1f52e01c1c0582e8"
    ),
}


@pytest.mark.parametrize("scenario", sorted(_GATE2_FROZEN_DIGESTS))
def test_back_patch_output_is_byte_identical_to_base(
    tmp_path: Path, scenario: str
) -> None:
    """Pin the jump-table-native back-patch's on-disk output byte-for-byte.

    The refactor that replaced the full-section ``parse_section_bin``
    re-parse in the resolve path with jump-table cumsum addressing
    changed only HOW slot byte-offsets are found, never WHAT is written.
    These sha256 digests were captured on BASE e468e3c (the re-parse
    writer) and must reproduce on the refactored writer; a mismatch is a
    correctness regression. ``dense`` exercises sibling FIDs, a
    self-call, a finalize MISSING-stamp, and an odd ``n_variants``;
    ``chain50`` / ``chain400`` exercise long forward-ref chains with
    multi-variant sections so the cumsum addressing is stressed at scale.
    """
    path = tmp_path / f"{scenario}.bin"
    if scenario == "dense":
        _build_byte_equivalence_fixture(path)
    elif scenario == "chain50":
        _build_forward_ref_chain(path, 50)
    elif scenario == "chain400":
        _build_forward_ref_chain(path, 400)
    else:  # pragma: no cover - parametrize guard
        raise AssertionError(f"unknown scenario {scenario!r}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == _GATE2_FROZEN_DIGESTS[scenario], (
        f"{scenario}: section bytes changed vs BASE e468e3c — "
        f"got {digest}, expected {_GATE2_FROZEN_DIGESTS[scenario]}; "
        "the back-patch refactor must be addressing-only"
    )
