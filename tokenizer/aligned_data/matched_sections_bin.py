"""Binary codec for ``<binary>_sections.bin`` — matched + unmatched section catalog.

Single concern: encode/decode the on-disk layout of the section catalog
that the dataloader reads in lieu of the legacy ``<binary>_sections.csv``.
Every section records a function's header (function_name_ptr,
n_call_targets, n_variants), the typed call_target table
(``(function_name_ptr, function_section_ptr, type, is_matched)`` per
entry), and per-variant blocks each holding a sparse list of
``(called_idx, section_variant_index)`` pairs into the section's
call_target table.

The writer back-patches forward references in two places:

* ``function_section_ptr`` on a call_target — set to ``0`` when emitting
  the call_target if the callee section hasn't been written yet;
  patched once the callee section opens (its ``section_offset`` is
  registered the moment :meth:`SectionWriter.begin_section` runs, so
  every call_target referencing it from earlier-emitted sections gets
  filled in when those *referencing* sections' back-patch queue runs).
* ``section_variant_index`` inside a per-call entry — set to
  ``0xFFFF`` when the callee's variant index isn't known yet (callee
  section unwritten OR not yet emitted that variant); patched at the
  callee section's :meth:`SectionWriter.end_section`.

A finalize-time sweep asserts no unresolved holes remain and no
``0xFFFF`` slot leaked through. Both are builder-bug detectors — a
correct caller writes a section for every callee FID that any
call_target references AND a variant for every ``vkey`` any per-call
entry references.

The wire format is documented in detail in
``polished-greeting-moler.md`` (Approach → A. Binary section file
layout). All multi-byte integers are little-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Iterator, Optional

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.memmap_format import (
    MATCHED_SECTIONS_BIN_PRELUDE_MAGIC,
    MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    assert_matched_sections_prelude,
    encode_matched_sections_prelude,
)
from tokenizer.aligned_data.memmap_writer import MemmapBinWriter

# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

#: Bytes per section header (``u32 func_line_no | u16 n_call_targets | u16 n_variants``).
SECTION_HEADER_SIZE: int = 8

#: Bytes per call_target table entry
#: (``u32 function_name_ptr | u32 function_section_ptr | u16 flags | u16 reserved``).
CALL_TARGET_ENTRY_SIZE: int = 12

#: Bytes per variant header
#: (``u32 variant_ref_offset | u32 data_offset_shifted | u16 n_calls``).
VARIANT_HEADER_SIZE: int = 10

#: Bytes per per-call entry (``u16 called_idx | u16 section_variant_index``).
PER_CALL_ENTRY_SIZE: int = 4

#: Sections are 4-byte aligned in the bin so the ``matched_index.bin``
#: ``>> 2`` shift stays correct.
SECTION_ALIGNMENT: int = 4

#: Sentinel value for ``section_variant_index`` placeholder slots; replaced
#: with the resolved variant index at back-patch time.
UNRESOLVED_VARIANT_INDEX: int = 0xFFFF

#: Sentinel value for ``function_section_ptr`` on extern call_targets
#: whose provider library is unknown.
UNKNOWN_EXTERN_PROVIDER: int = 0

# Bit packing for the call_target ``flags`` field.
_FLAG_IS_MATCHED_BIT: int = 0
_FLAG_TYPE_SHIFT: int = 1
_FLAG_TYPE_MASK: int = 0b11  # two bits → fits CallTargetType {0,1,2}


def _pack_flags(call_type: CallTargetType, is_matched: bool) -> int:
    """Pack the ``u16 flags`` field for a call_target entry."""
    value = (int(call_type) & _FLAG_TYPE_MASK) << _FLAG_TYPE_SHIFT
    if is_matched:
        value |= 1 << _FLAG_IS_MATCHED_BIT
    return value


def _unpack_flags(flags: int) -> tuple[CallTargetType, bool]:
    """Inverse of :func:`_pack_flags`. Returns ``(call_type, is_matched)``."""
    is_matched = bool((flags >> _FLAG_IS_MATCHED_BIT) & 1)
    call_type = CallTargetType((flags >> _FLAG_TYPE_SHIFT) & _FLAG_TYPE_MASK)
    return call_type, is_matched


# ---------------------------------------------------------------------------
# Writer-side input dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallTargetSpec:
    """One row in a section's call_target table (writer input).

    The caller deduplicates by ``(function_name_ptr, type)`` per the
    plan's correctness fix; :class:`SectionWriter` does not re-dedupe.

    ``extern_provider_line_no`` is the 1-indexed line into
    ``<binary>_extern_providers.txt`` when ``type == EXTERN`` and the
    provider library is known; ``None`` maps to the
    :data:`UNKNOWN_EXTERN_PROVIDER` sentinel (``0``) on the wire.
    Ignored for LOCAL / PLT types — those resolve their
    ``function_section_ptr`` via the writer's
    ``known_sections`` map (back-patched if forward-referenced).
    """

    function_name_ptr: int
    type: CallTargetType
    is_matched: bool
    extern_provider_line_no: Optional[int] = None


@dataclass(frozen=True)
class PerCallEntry:
    """One per-call slot inside a variant block (writer input).

    ``called_idx`` is the index into the CURRENT section's call_target
    table. ``callee_function_name_ptr`` + ``callee_vkey`` together look
    up the resolved variant index in the callee section's variant block
    list via ``known_section_variants[(callee_FID, callee_vkey)]``.
    """

    called_idx: int
    callee_function_name_ptr: int
    callee_vkey: Hashable


# ---------------------------------------------------------------------------
# Reader-side parsed dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallTarget:
    """Parsed call_target row."""

    function_name_ptr: int
    function_section_ptr: int
    type: CallTargetType
    is_matched: bool


@dataclass(frozen=True)
class VariantBlock:
    """Parsed variant block."""

    variant_ref_offset: int
    data_offset_shifted: int
    per_call_entries: list[tuple[int, int]]
    """List of ``(called_idx, section_variant_index)`` pairs."""


@dataclass(frozen=True)
class Section:
    """Parsed section (one entry in the catalog)."""

    function_name_ptr: int
    section_offset: int
    call_targets: list[CallTarget]
    variants: list[VariantBlock]


# ---------------------------------------------------------------------------
# Writer-side back-patch bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _HoleRecord:
    """Tracks one referencing-section's back-patch needs for a callee.

    Single owner: :class:`SectionWriter`. Stored in
    ``_pending_holes[callee_FID]``; resolved at the moment that callee
    section closes.

    ``section_offset`` is the offset of the REFERENCING section (the
    section that emitted the slots needing a patch); used for diagnostic
    messages when the finalizer trips and to deduplicate the per-section
    record in ``_current_section_holes_by_callee``.

    ``header_hole_offsets`` is the list of byte offsets of ``u32
    function_section_ptr`` slots in the referencing section that need
    the callee's section_offset written into them. A section may have
    MULTIPLE call_target rows referencing the same callee FID under
    different ``type`` discriminators (LOCAL vs PLT vs EXTERN) — the
    plan's correctness fix deduplicates by ``(FID, type)``, not by FID
    — so each forward-referenced row contributes one entry here.
    An empty list means no header slot needs patching, which happens
    when the callee's header was resolved at emit time but a per-variant
    slot still references an unwritten variant of this callee.

    ``per_variant_holes`` is the list of ``(file_offset_of_u16_slot,
    callee_vkey)`` tuples — each one identifies a ``u16
    section_variant_index`` slot in the referencing section that needs
    to be patched with the variant index that the callee section
    assigns to that ``vkey``.
    """

    section_offset: int
    header_hole_offsets: list[int] = field(default_factory=list)
    per_variant_holes: list[tuple[int, Hashable]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class SectionWriter:
    """Memmap-backed writer for ``<binary>_sections.bin`` with back-patching.

    Lifecycle per section, called in order:

    1. :meth:`begin_section` — aligns + records the section's offset.
    2. :meth:`emit_call_targets` — stamps the section header (with
       ``n_variants = 0`` placeholder) + the call_target table.
    3. Per variant:
        a. :meth:`begin_variant` — stamps the variant header (with
           ``n_calls = 0`` placeholder).
        b. :meth:`emit_per_call_entries` — writes the per-call slots
           and back-patches the variant's ``n_calls`` field.
        c. :meth:`end_variant` — registers the variant's ``vkey`` →
           ``variant_idx`` mapping.
    4. :meth:`end_section` — back-patches ``n_variants``, pads to a
       4-byte boundary, and resolves any pending holes whose
       callee_FID equals THIS section's function_name_ptr.
    5. After all sections: :meth:`finalize` — asserts no holes remain
       and closes the underlying memmap.
    """

    def __init__(self, path: Path) -> None:
        # Prelude is the natural identity of the bin: stamped at open
        # so the first section starts at byte 16. The writer keeps a
        # direct reference to MemmapBinWriter; the public API never
        # exposes the underlying mapping to callers.
        self._writer = MemmapBinWriter(
            Path(path), prelude_bytes=encode_matched_sections_prelude()
        )

        # Cross-section state.
        self._known_sections: dict[int, int] = {}
        self._known_section_variants: dict[tuple[int, Hashable], int] = {}
        self._pending_holes: dict[int, list[_HoleRecord]] = {}

        # Per-section state (cleared on every begin_section).
        self._current_fid: Optional[int] = None
        self._current_section_offset: Optional[int] = None
        # Offset of THIS section's u16 n_variants slot (patched in end_section).
        self._n_variants_slot: Optional[int] = None
        # Specs of the section's call_targets, kept around so
        # emit_per_call_entries can validate that a PerCallEntry's
        # called_idx points at the call_target whose FID the entry
        # declares.
        self._current_call_targets: list[CallTargetSpec] = []
        # variant_idx → variant_count assigned so far in this section.
        self._current_variant_count: int = 0
        # For the currently-open variant: file offset of its u16 n_calls
        # slot. None when no variant is open.
        self._current_variant_n_calls_slot: Optional[int] = None
        # Holes opened by THIS section, keyed by callee_FID — lets
        # emit_per_call_entries reuse a HoleRecord (one per
        # (referencing-section, callee) pair) rather than duplicating.
        self._current_section_holes_by_callee: dict[int, _HoleRecord] = {}

    # ------------------------------------------------------------------
    # Section lifecycle
    # ------------------------------------------------------------------

    def begin_section(self, function_name_ptr: int) -> int:
        """Open a new section for ``function_name_ptr``.

        Aligns the cursor up to a 4-byte boundary (pads the gap with
        zero bytes), records the section's start offset in
        ``known_sections``, and returns it. Forward references to this
        FID emitted by EARLIER sections remain in ``_pending_holes`` —
        they're resolved when this section closes via
        :meth:`end_section`.
        """
        self._assert_no_open_section()
        self._pad_to_alignment()
        section_offset = self._writer.cursor

        if function_name_ptr in self._known_sections:
            raise ValueError(
                f"section for function_name_ptr={function_name_ptr} "
                f"already written at offset "
                f"{self._known_sections[function_name_ptr]}"
            )
        self._known_sections[function_name_ptr] = section_offset

        self._current_fid = function_name_ptr
        self._current_section_offset = section_offset
        self._n_variants_slot = None
        self._current_call_targets = []
        self._current_variant_count = 0
        self._current_variant_n_calls_slot = None
        self._current_section_holes_by_callee = {}
        return section_offset

    def emit_call_targets(self, call_targets: list[CallTargetSpec]) -> None:
        """Stamp the section header + the call_target table.

        ``n_variants`` is stamped at ``0`` and patched in
        :meth:`end_section`. The caller is responsible for having
        deduplicated ``call_targets`` by ``(function_name_ptr, type)``;
        SectionWriter does not check.
        """
        self._assert_section_open()
        if self._n_variants_slot is not None:
            raise ValueError("emit_call_targets called twice for the same section")

        n_call_targets = len(call_targets)
        # Section header: func_line_no | n_call_targets | n_variants (placeholder).
        header = struct.pack(
            "<IHH",
            self._current_fid,
            n_call_targets,
            0,  # n_variants — patched at end_section
        )
        header_offset = self._writer.write(header)
        # n_variants is the second u16 → header_offset + 4 (u32) + 2 (u16).
        self._n_variants_slot = header_offset + 4 + 2

        for spec in call_targets:
            row_offset = self._writer.cursor
            function_section_ptr = self._resolve_function_section_ptr(
                spec, row_offset
            )
            flags = _pack_flags(spec.type, spec.is_matched)
            row = struct.pack(
                "<IIHH",
                spec.function_name_ptr,
                function_section_ptr,
                flags,
                0,  # reserved
            )
            self._writer.write(row)

        self._current_call_targets = list(call_targets)

    def begin_variant(
        self, variant_ref_offset: int, data_offset_shifted: int
    ) -> None:
        """Stamp the variant header (``n_calls`` set to 0 placeholder).

        Cursor is left at the start of the per-call entries; the next
        :meth:`emit_per_call_entries` call writes them and patches
        ``n_calls``.
        """
        self._assert_section_open()
        if self._n_variants_slot is None:
            raise ValueError(
                "begin_variant called before emit_call_targets; the "
                "section header must be stamped first"
            )
        if self._current_variant_n_calls_slot is not None:
            raise ValueError(
                "begin_variant called while a previous variant is still "
                "open; call emit_per_call_entries + end_variant first"
            )

        variant_header = struct.pack(
            "<IIH",
            variant_ref_offset,
            data_offset_shifted,
            0,  # n_calls — patched in emit_per_call_entries
        )
        header_offset = self._writer.write(variant_header)
        # n_calls is the trailing u16 → header_offset + 4 + 4.
        self._current_variant_n_calls_slot = header_offset + 4 + 4

    def emit_per_call_entries(self, entries: list[PerCallEntry]) -> None:
        """Write the variant's per-call entries + patch ``n_calls``.

        For each entry, the writer looks up
        ``known_section_variants[(callee_FID, callee_vkey)]``; a hit
        stamps the u16 directly, a miss stamps
        :data:`UNRESOLVED_VARIANT_INDEX` and records a back-patch
        target. The HoleRecord is shared across multiple per-call
        slots referencing the same callee from this section — the
        ``_current_section_holes_by_callee`` map carries the
        deduplication so we never duplicate header_hole_offset.
        """
        self._assert_variant_open()

        for entry in entries:
            self._assert_called_idx_matches(entry)
            entry_offset = self._writer.cursor
            # The u16 section_variant_index slot is the second field in
            # the entry (after the u16 called_idx) → +2 from entry start.
            slot_offset = entry_offset + 2
            resolved = self._known_section_variants.get(
                (entry.callee_function_name_ptr, entry.callee_vkey)
            )
            if resolved is None:
                section_variant_index = UNRESOLVED_VARIANT_INDEX
                self._record_per_variant_hole(
                    callee_fid=entry.callee_function_name_ptr,
                    slot_offset=slot_offset,
                    callee_vkey=entry.callee_vkey,
                )
            else:
                section_variant_index = resolved
            self._writer.write(
                struct.pack("<HH", entry.called_idx, section_variant_index)
            )

        # Patch the variant header's n_calls (u16 LE).
        self._writer.patch(
            self._current_variant_n_calls_slot,
            struct.pack("<H", len(entries)),
        )

    def end_variant(self, vkey: Hashable) -> int:
        """Finalise the currently-open variant.

        Computes the variant's 0-based index in the section's variant
        block list and records the
        ``known_section_variants[(current_FID, vkey)] = variant_idx``
        mapping. Future emissions referencing this ``(FID, vkey)``
        will resolve directly without back-patching.
        """
        self._assert_variant_open()
        variant_idx = self._current_variant_count
        if variant_idx > UNRESOLVED_VARIANT_INDEX - 1:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} has "
                f"{variant_idx + 1} variants; max is "
                f"{UNRESOLVED_VARIANT_INDEX} per section (u16 slot reserves "
                f"0xFFFF as the unresolved-hole sentinel)"
            )
        key = (self._current_fid, vkey)
        if key in self._known_section_variants:
            raise ValueError(
                f"vkey {vkey!r} already registered for "
                f"function_name_ptr={self._current_fid} "
                f"at variant_idx={self._known_section_variants[key]}"
            )
        self._known_section_variants[key] = variant_idx
        self._current_variant_count += 1
        self._current_variant_n_calls_slot = None
        return variant_idx

    def end_section(self) -> tuple[int, int]:
        """Close the current section.

        Patches ``n_variants``, pads to a 4-byte boundary, then walks
        ``_pending_holes[current_FID]`` resolving every header slot
        and per-variant slot that referenced this section. Returns
        ``(section_offset, section_length)`` -- the start byte the
        section was opened at and the trailer-aligned byte width the
        section occupies in the bin. The length is what the per-binary
        ``matched_index.bin`` u24 stores; both are 4-byte aligned (the
        section trailer pad enforced above guarantees the length is a
        multiple of :data:`SECTION_ALIGNMENT`).
        """
        self._assert_section_open()
        if self._current_variant_n_calls_slot is not None:
            raise ValueError(
                "end_section called while a variant is still open; "
                "call end_variant first"
            )

        # Patch n_variants.
        self._writer.patch(
            self._n_variants_slot,
            struct.pack("<H", self._current_variant_count),
        )
        # Align section trailer.
        self._pad_to_alignment()

        # Resolve back-patches whose callee == THIS section.
        section_offset = self._current_section_offset
        section_length = self._writer.cursor - section_offset
        fid = self._current_fid
        holes = self._pending_holes.pop(fid, [])
        for hole in holes:
            packed_offset = struct.pack("<I", section_offset)
            for header_slot in hole.header_hole_offsets:
                self._writer.patch(header_slot, packed_offset)
            for slot_offset, callee_vkey in hole.per_variant_holes:
                variant_idx = self._known_section_variants.get((fid, callee_vkey))
                if variant_idx is None:
                    # Caller declared a per-call entry referencing
                    # (callee=fid, vkey=callee_vkey) but the section
                    # never emitted that variant. Builder bug — surface
                    # eagerly with the referencing section offset.
                    raise ValueError(
                        f"per-call back-patch unresolved: callee section "
                        f"function_name_ptr={fid} did not emit a variant "
                        f"for vkey={callee_vkey!r} (referenced from "
                        f"section at offset {hole.section_offset})"
                    )
                self._writer.patch(slot_offset, struct.pack("<H", variant_idx))

        # Clear per-section state.
        self._current_fid = None
        self._current_section_offset = None
        self._n_variants_slot = None
        self._current_call_targets = []
        self._current_variant_count = 0
        self._current_variant_n_calls_slot = None
        self._current_section_holes_by_callee = {}

        return section_offset, section_length

    def finalize(self) -> None:
        """Close the underlying memmap; assert no holes leaked.

        First a structural assertion: ``_pending_holes`` must be empty
        — any non-empty key means a call_target whose callee section
        was never written, which is a builder bug.

        Belt-and-braces: scan the entire written bin for any remaining
        :data:`UNRESOLVED_VARIANT_INDEX` slot. The pre-section
        back-patch queue should have caught everything, but a writer
        bug (e.g. forgetting to register a per-variant slot in the
        hole record) would leak through; this sweep surfaces it
        before the bin is sealed.

        The memmap is closed unconditionally in a ``finally``: if the
        sweep or any of the structural checks raises, the underlying
        bin still gets unmapped + truncated rather than leaking until
        process exit.
        """
        try:
            if self._current_fid is not None:
                raise ValueError(
                    "finalize called while section "
                    f"function_name_ptr={self._current_fid} is still open"
                )
            if self._pending_holes:
                unresolved = sorted(self._pending_holes.keys())
                raise ValueError(
                    f"finalize: {len(unresolved)} callee section(s) were "
                    f"referenced but never written: function_name_ptrs="
                    f"{unresolved!r}"
                )
            self._sweep_for_unresolved_sentinels()
        finally:
            self.close()

    def close(self) -> None:
        """Flush + unmap the underlying bin without running checks.

        Idempotent. The happy-path entry is :meth:`finalize`, which
        runs the structural assertions first; ``close`` exists as the
        always-runs cleanup so an error mid-finalize still releases
        the mmap. Safe to call from a ``try``/``finally`` or
        ``__exit__``.
        """
        self._writer.finalize()

    def __enter__(self) -> "SectionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If the body raised, callers haven't reached finalize; close
        # the mmap so the exception path doesn't leak it. If the body
        # already finalized cleanly, ``close`` is a no-op
        # (``MemmapBinWriter.finalize`` is idempotent).
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_no_open_section(self) -> None:
        if self._current_fid is not None:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} is "
                "still open; call end_section before begin_section"
            )

    def _assert_section_open(self) -> None:
        if self._current_fid is None:
            raise ValueError("no section is currently open")

    def _assert_variant_open(self) -> None:
        self._assert_section_open()
        if self._current_variant_n_calls_slot is None:
            raise ValueError("no variant is currently open")

    def _assert_called_idx_matches(self, entry: PerCallEntry) -> None:
        """Defensive check: a per-call entry's ``called_idx`` is in range
        AND points at a call_target whose FID matches the entry's
        ``callee_function_name_ptr``. Catches caller bugs where the
        sparse per-variant index got rebased against the wrong section.
        """
        n = len(self._current_call_targets)
        if not (0 <= entry.called_idx < n):
            raise ValueError(
                f"called_idx={entry.called_idx} is out of range "
                f"(section has {n} call_targets)"
            )
        spec = self._current_call_targets[entry.called_idx]
        if spec.function_name_ptr != entry.callee_function_name_ptr:
            raise ValueError(
                f"called_idx={entry.called_idx} indexes call_target "
                f"function_name_ptr={spec.function_name_ptr} but entry "
                f"declares callee_function_name_ptr="
                f"{entry.callee_function_name_ptr}"
            )

    def _pad_to_alignment(self) -> None:
        """Pad the cursor up to the next :data:`SECTION_ALIGNMENT` boundary."""
        cursor = self._writer.cursor
        rem = cursor % SECTION_ALIGNMENT
        if rem == 0:
            return
        self._writer.write(b"\x00" * (SECTION_ALIGNMENT - rem))

    def _resolve_function_section_ptr(
        self, spec: CallTargetSpec, row_offset: int
    ) -> int:
        """Resolve a call_target's ``function_section_ptr`` at emit time.

        LOCAL / PLT: look up the callee's section_offset in
        ``known_sections``; miss ⇒ register a header hole and write 0
        as a placeholder.

        EXTERN: write ``extern_provider_line_no`` if provided, else
        :data:`UNKNOWN_EXTERN_PROVIDER` (= 0).
        """
        # The function_section_ptr field is the second u32 in the row,
        # i.e. row_offset + 4 (u32 function_name_ptr).
        header_slot_offset = row_offset + 4

        if spec.type is CallTargetType.EXTERN:
            if spec.extern_provider_line_no is None:
                return UNKNOWN_EXTERN_PROVIDER
            return spec.extern_provider_line_no

        # LOCAL / PLT.
        known = self._known_sections.get(spec.function_name_ptr)
        if known is not None:
            return known
        # Forward reference: open a hole.
        self._open_header_hole(
            callee_fid=spec.function_name_ptr,
            header_slot_offset=header_slot_offset,
        )
        return 0  # placeholder; patched in end_section of the callee.

    def _open_header_hole(self, *, callee_fid: int, header_slot_offset: int) -> None:
        """Append a header slot to the HoleRecord for a forward-referenced callee.

        At most one :class:`_HoleRecord` exists per
        ``(current_section, callee_fid)`` pair; if a hole was already
        opened by a previous slot in this section (e.g. another
        call_target row with the same FID-but-different-type, or an
        emit_per_call_entries miss that ran first under a different
        ordering), the existing record gains another header slot in
        its ``header_hole_offsets`` list.
        """
        record = self._get_or_create_section_hole(callee_fid)
        record.header_hole_offsets.append(header_slot_offset)

    def _record_per_variant_hole(
        self,
        *,
        callee_fid: int,
        slot_offset: int,
        callee_vkey: Hashable,
    ) -> None:
        """Append a per-variant slot to the callee's HoleRecord.

        Reuses the per-section record if one exists; otherwise creates
        a fresh record with no header slots (the callee's header was
        resolved at emit time, only per-variant slots need patching).
        """
        record = self._get_or_create_section_hole(callee_fid)
        record.per_variant_holes.append((slot_offset, callee_vkey))

    def _get_or_create_section_hole(self, callee_fid: int) -> _HoleRecord:
        """Return THIS section's HoleRecord for ``callee_fid``, creating it
        on first use.

        Centralises the deduplication invariant (one record per
        ``(current_section, callee_fid)``) so callers — both the
        header-hole opener and the per-variant-hole opener — share
        the same accumulator.
        """
        record = self._current_section_holes_by_callee.get(callee_fid)
        if record is None:
            record = _HoleRecord(section_offset=self._current_section_offset)
            self._current_section_holes_by_callee[callee_fid] = record
            self._pending_holes.setdefault(callee_fid, []).append(record)
        return record

    def _sweep_for_unresolved_sentinels(self) -> None:
        """Walk the bin sections; assert no ``0xFFFF`` slot leaked.

        Reuses the public parser so the sweep can't drift from the
        emit-side layout: any section the parser produces is also
        what the bin will look like to readers, and we check every
        per-call entry's ``section_variant_index``.

        Uses a zero-copy :meth:`MemmapBinWriter.view` over the
        already-written region instead of :meth:`MemmapBinWriter.read`
        — at corpus scale the bin is multi-GB and a ``bytes`` copy at
        finalize-time would more than double peak RAM. The memoryview
        is explicitly :meth:`memoryview.release`-d in a ``finally`` so
        the subsequent :meth:`MemmapBinWriter.finalize` (which calls
        ``mmap.close``) does not trip on an exported pointer being
        held alive by the traceback of an in-flight exception.
        """
        blob = self._writer.view()
        try:
            end = len(blob)
            offset = MATCHED_SECTIONS_BIN_PRELUDE_SIZE
            while offset < end:
                section, offset = parse_section_bin(blob, offset)
                for v_idx, variant in enumerate(section.variants):
                    for called_idx, sv_idx in variant.per_call_entries:
                        if sv_idx == UNRESOLVED_VARIANT_INDEX:
                            raise ValueError(
                                f"unresolved section_variant_index in section "
                                f"function_name_ptr={section.function_name_ptr} "
                                f"variant_idx={v_idx} called_idx={called_idx}"
                            )
        finally:
            blob.release()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def parse_section_bin(blob: memoryview, offset: int) -> tuple[Section, int]:
    """Parse one section starting at ``offset`` in ``blob``.

    Returns ``(Section, end_offset)`` where ``end_offset`` is past the
    section's trailing alignment padding (so the caller can call
    again with the new offset to read the next section).
    """
    cursor = offset

    # Section header.
    func_line_no, n_call_targets, n_variants = struct.unpack_from(
        "<IHH", blob, cursor
    )
    cursor += SECTION_HEADER_SIZE

    # Call target table.
    call_targets: list[CallTarget] = []
    for _ in range(n_call_targets):
        function_name_ptr, function_section_ptr, flags, _reserved = (
            struct.unpack_from("<IIHH", blob, cursor)
        )
        call_type, is_matched = _unpack_flags(flags)
        call_targets.append(
            CallTarget(
                function_name_ptr=function_name_ptr,
                function_section_ptr=function_section_ptr,
                type=call_type,
                is_matched=is_matched,
            )
        )
        cursor += CALL_TARGET_ENTRY_SIZE

    # Variant blocks.
    variants: list[VariantBlock] = []
    for _ in range(n_variants):
        variant_ref_offset, data_offset_shifted, n_calls = struct.unpack_from(
            "<IIH", blob, cursor
        )
        cursor += VARIANT_HEADER_SIZE
        per_call_entries: list[tuple[int, int]] = []
        for _call_idx in range(n_calls):
            called_idx, section_variant_index = struct.unpack_from(
                "<HH", blob, cursor
            )
            per_call_entries.append((called_idx, section_variant_index))
            cursor += PER_CALL_ENTRY_SIZE
        variants.append(
            VariantBlock(
                variant_ref_offset=variant_ref_offset,
                data_offset_shifted=data_offset_shifted,
                per_call_entries=per_call_entries,
            )
        )

    # Section trailer alignment.
    rem = cursor % SECTION_ALIGNMENT
    if rem != 0:
        cursor += SECTION_ALIGNMENT - rem

    return (
        Section(
            function_name_ptr=func_line_no,
            section_offset=offset,
            call_targets=call_targets,
            variants=variants,
        ),
        cursor,
    )


def iter_sections_bin(path: Path) -> Iterator[Section]:
    """Yield every section in ``path`` in file order.

    Validates the prelude via :func:`assert_matched_sections_prelude`;
    a mismatched magic / version raises before the first yield.
    """
    path = Path(path)
    raw = path.read_bytes()
    assert_matched_sections_prelude(raw, path=str(path))
    blob = memoryview(raw)
    offset = MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    end = len(raw)
    while offset < end:
        section, offset = parse_section_bin(blob, offset)
        yield section
