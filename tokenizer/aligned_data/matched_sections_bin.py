"""Binary codec for ``<binary>_sections.bin`` — matched + unmatched section catalog.

Single concern: encode/decode the on-disk layout of the section catalog
that the dataloader reads in lieu of the legacy ``<binary>_sections.csv``.
Every section records a function's header (function_name_ptr,
n_call_targets, n_variants), the typed call_target table
(``(function_name_ptr, function_section_ptr, type, is_matched)`` per
entry), and per-variant blocks each holding a sparse list of
``(called_idx, section_variant_index)`` pairs into the section's
call_target table.

The writer back-patches forward references in two places, and on every
emit it re-parses the relevant callee section's own bytes through
:func:`parse_section_bin` instead of carrying a cross-section variant
map in writer memory:

* ``function_section_ptr`` on a call_target — set to ``0`` when
  emitting the call_target if the callee section hasn't been written
  yet; patched when the callee section closes. Sibling sections that
  share a ``function_name_ptr`` (clang's ``OUTLINED_FUNCTION_N``)
  each stamp their own offset over the placeholder; the loader walks
  via per-call ``callee_vkey`` to disambiguate which sibling carries
  the matching variant.
* ``section_variant_index`` inside a per-call entry — backward
  references (callee section already closed) re-parse the section
  pointed at by ``_known_sections[callee_fid]`` (the LAST sibling
  closed, the same offset that ends up in the call_target's
  ``function_section_ptr``) and look up the entry's ``callee_vkey``;
  a hit stamps the resolved index directly, a miss defers via
  :data:`UNRESOLVED_VARIANT_INDEX` plus a back-patch hole. Forward
  references stamp the same sentinel + hole. At every
  :meth:`SectionWriter.end_section`, the writer re-parses the
  just-closed section and resolves any hole whose callee FID matches
  THIS section AND whose ``callee_vkey`` is in this section's
  variant table. Sibling sections with disjoint vkey sets each patch
  only their own matching holes; holes whose vkey is never
  registered by any sibling fall through to
  :meth:`SectionWriter.finalize`, which stamps
  :data:`MISSING_VARIANT_INDEX` and emits a one-line ``warn-log``
  entry (so the corpus rebuild can audit how often the cross-arm
  vkey mismatch fires).

Each section is self-describing: the variant table needed to resolve
back-patches is recoverable from the section's own bytes. No
cross-section variant map is kept in writer memory.

A finalize-time sweep asserts no ``0xFFFF`` (UNRESOLVED) sentinel
leaked through to the on-disk bytes. Any unresolved header hole at
finalize is a hard builder bug (callee section was never written) and
raises; per-variant holes get the :data:`MISSING_VARIANT_INDEX`
sentinel + warn-log line (legitimate cross-arm vkey mismatch).

The wire format is documented in detail in
``polished-greeting-moler.md`` (Approach → A. Binary section file
layout). All multi-byte integers are little-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Iterator, Optional, TextIO

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
#: with the resolved variant index at back-patch time. The finalize-time
#: sweep rejects any of these remaining — they signal a writer bug
#: (a hole was opened but never resolved).
UNRESOLVED_VARIANT_INDEX: int = 0xFFFF

#: Sentinel value for ``section_variant_index`` when the callee section
#: exists but does not have a variant matching the caller's vkey. This
#: happens at corpus scale when caller and callee have different
#: surviving-variant sets after pass-1's drop rules (encoder skip,
#: dedup-to-same-offset). The per-call entry still records "this call
#: existed", but the callee's variant is not directly addressable; the
#: loader treats the slot as "no inlined callee body for this vkey".
#: Distinct from :data:`UNRESOLVED_VARIANT_INDEX` so the finalize sweep
#: can tell a writer bug (`0xFFFF`) from a legitimate cross-arm vkey
#: mismatch (`0xFFFE`).
MISSING_VARIANT_INDEX: int = 0xFFFE

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
    table. ``callee_function_name_ptr`` + ``callee_vkey`` are used to
    look up the resolved variant index in the callee section: at the
    callee's :meth:`SectionWriter.end_section`, the writer parses
    that section's bytes back and matches each pending hole's
    ``callee_vkey`` against the on-disk ``variant_ref_offset`` of
    each variant in the section. Callers MUST therefore use the SAME
    Hashable value space for ``PerCallEntry.callee_vkey`` and the
    matching variant's ``begin_variant(variant_ref_offset=...)``
    argument — typically both are the integer byte offset of the
    vkey in the per-binary variants sidecar.
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
    ``_pending_holes[(callee_FID, referencing_section_offset)]``; the
    key shape is what carries the (referencing-section, callee)
    dedup. The referencing section's offset is therefore implicit in
    the map key — no field on the record itself.

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
    assigns to that ``vkey``. Resolved at the callee section's
    :meth:`SectionWriter.end_section` by parsing the just-closed
    section back from its own bytes and reading the variant table.
    """

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
        b. :meth:`emit_per_call_entries` — writes the per-call slots.
           Backward references re-parse the callee section pointed at
           by ``_known_sections[callee_fid]`` and stamp the resolved
           variant idx directly on a vkey hit; misses (and forward
           references) defer to a back-patch hole. Patches the
           variant's ``n_calls`` field afterwards.
        c. :meth:`end_variant` — increments the section's variant
           count.
    4. :meth:`end_section` — back-patches ``n_variants``, pads to a
       4-byte boundary, parses the just-closed section's bytes to
       recover its variant table, and resolves any pending holes
       whose callee_FID equals THIS section's function_name_ptr.
    5. After all sections: :meth:`finalize` — stamps
       :data:`MISSING_VARIANT_INDEX` on any per_variant_holes still
       open (with one warn-log line per stamp), raises on any
       remaining header_hole_offsets (callee section was never
       written — builder bug), and closes the underlying memmap.
    """

    def __init__(self, path: Path, warn_log: Optional[TextIO] = None) -> None:
        # Prelude is the natural identity of the bin: stamped at open
        # so the first section starts at byte 16. The writer keeps a
        # direct reference to MemmapBinWriter; the public API never
        # exposes the underlying mapping to callers.
        self._writer = MemmapBinWriter(
            Path(path), prelude_bytes=encode_matched_sections_prelude()
        )

        # Optional per-binary warn-log; receives one line per
        # :data:`MISSING_VARIANT_INDEX` stamped at finalize. ``None``
        # silences the writer (test-fixture default).
        self._warn_log: Optional[TextIO] = warn_log

        # Cross-section state.
        #
        # ``_known_sections``: emit-time O(1) ``callee FID -> section
        # offset`` lookup for resolving call_target ``function_section_ptr``
        # on the spot. Sibling sections that share a FID overwrite
        # (last-write-wins); the loader resolves the ambiguity by
        # walking via per-call ``callee_vkey``.
        self._known_sections: dict[int, int] = {}
        # ``_pending_holes`` keyed on ``(callee_FID,
        # referencing_section_offset)``: at most one record per
        # (referencing section, callee) pair. The key shape is what
        # carries the dedup that the legacy ``_current_section_holes_by_callee``
        # auxiliary used to enforce.
        self._pending_holes: dict[tuple[int, int], _HoleRecord] = {}

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

        Function names are not globally unique within a binary: clang
        emits ``OUTLINED_FUNCTION_N`` for compiler-internal helpers
        and these share names across distinct bodies. The matched arm
        therefore yields multiple entries with the same ``func_name``,
        producing multiple sections that share a
        ``function_name_ptr``. We accept the collision and overwrite
        ``known_sections`` with the latest section's offset; the
        ``function_section_ptr`` back-patch loop at every sibling's
        :meth:`end_section` re-stamps the same slot, so the on-disk
        slot ends up pointing at the LAST sibling. The loader walks
        sibling sections via the per-call ``callee_vkey`` to resolve
        which body actually carries the matching variant. The
        ``matched_index.bin`` locator records every section
        independently so the loader can still address all of them by
        index.
        """
        self._assert_no_open_section()
        self._pad_to_alignment()
        section_offset = self._writer.cursor

        self._known_sections[function_name_ptr] = section_offset

        self._current_fid = function_name_ptr
        self._current_section_offset = section_offset
        self._n_variants_slot = None
        self._current_call_targets = []
        self._current_variant_count = 0
        self._current_variant_n_calls_slot = None
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

        Backward references (``callee_fid in _known_sections``) re-parse
        the callee section pointed at by ``_known_sections[callee_fid]``
        — the LAST sibling closed, which is also the offset that ends
        up in the call_target's ``function_section_ptr`` after sibling
        last-write-wins. The just-parsed section's local
        ``variant_ref_offset -> variant_idx`` map is consulted for the
        entry's ``callee_vkey``: a hit stamps the resolved index
        directly. A miss — or a forward reference whose callee section
        has not been opened yet — defers via a back-patch hole, written
        as :data:`UNRESOLVED_VARIANT_INDEX`. At each callee
        :meth:`end_section`, the writer parses the just-closed
        section's bytes and resolves any pending hole whose
        ``callee_vkey`` matches THIS section's variant table.
        Anything still unresolved at :meth:`finalize` (cross-arm vkey
        mismatch) gets :data:`MISSING_VARIANT_INDEX` + a warn-log
        line.

        The :class:`_HoleRecord` is shared across multiple per-call
        slots referencing the same callee from this section: the
        ``_pending_holes`` map is keyed on ``(callee_FID,
        current_section_offset)`` so a setdefault returns the same
        record for both the first call_target row's header-hole and
        the variant's per-call-slot accumulation.
        """
        self._assert_variant_open()

        for entry in entries:
            self._assert_called_idx_matches(entry)
            entry_offset = self._writer.cursor
            # The u16 section_variant_index slot is the second field in
            # the entry (after the u16 called_idx) → +2 from entry start.
            slot_offset = entry_offset + 2
            section_variant_index = self._resolve_backward_variant_index(
                callee_fid=entry.callee_function_name_ptr,
                callee_vkey=entry.callee_vkey,
            )
            if section_variant_index is None:
                # Forward reference, OR backward reference whose vkey
                # is not in the callee section's local variant table —
                # defer via back-patch hole. Stamp the unresolved
                # sentinel; a future end_section (sibling close) or
                # finalize will patch.
                section_variant_index = UNRESOLVED_VARIANT_INDEX
                self._record_per_variant_hole(
                    callee_fid=entry.callee_function_name_ptr,
                    slot_offset=slot_offset,
                    callee_vkey=entry.callee_vkey,
                )
            self._writer.write(
                struct.pack("<HH", entry.called_idx, section_variant_index)
            )

        # Patch the variant header's n_calls (u16 LE).
        self._writer.patch(
            self._current_variant_n_calls_slot,
            struct.pack("<H", len(entries)),
        )

    def _resolve_backward_variant_index(
        self, *, callee_fid: int, callee_vkey: Hashable
    ) -> Optional[int]:
        """Resolve a backward-reference variant idx by re-parsing the
        callee section pointed at by ``_known_sections[callee_fid]``.

        ``None`` when the callee FID has no closed section yet
        (forward reference) OR the closed section's local variant
        table does not carry ``callee_vkey`` (e.g. the loader will
        observe the call but the inlined callee body is not directly
        addressable — handled later by the hole path).

        The section we re-parse is the SAME one whose offset will end
        up in the call_target row's ``function_section_ptr`` after the
        sibling last-write-wins, so the loader can never observe a
        ``(section_offset, variant_idx)`` pair that points into the
        wrong sibling's variant table.

        ``parse_section_bin`` is given a bounded memoryview of the
        already-written region; the view is released in a ``finally``
        to keep the mmap unmappable on later finalize.
        """
        section_offset = self._known_sections.get(callee_fid)
        if section_offset is None:
            return None
        blob = self._writer.view()
        try:
            parsed, _end = parse_section_bin(blob, section_offset)
            for i, variant in enumerate(parsed.variants):
                if variant.variant_ref_offset == callee_vkey:
                    return i
        finally:
            blob.release()
        return None

    def end_variant(self, vkey: Hashable) -> int:
        """Finalise the currently-open variant.

        Returns the variant's 0-based index in the section's variant
        block list. The vkey itself was already stamped into the
        variant header's ``variant_ref_offset`` field (via
        :meth:`begin_variant`'s caller-supplied byte offset), so the
        writer does NOT need a cross-section map of
        ``(FID, vkey) → variant_idx``: :meth:`end_section` recovers
        it by parsing the just-closed section back from its own
        bytes.

        Multiple sections sharing a ``function_name_ptr`` (see the
        :meth:`begin_section` docstring) can emit overlapping vkeys
        — each sibling resolves only the per-call holes whose
        ``callee_vkey`` matches its own local variant table.
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
        self._current_variant_count += 1
        self._current_variant_n_calls_slot = None
        return variant_idx

    def end_section(self) -> tuple[int, int]:
        """Close the current section.

        Patches ``n_variants``, pads to a 4-byte boundary, then parses
        the just-closed section back from its own bytes to recover
        the variant table. Walks ``_pending_holes`` for every entry
        whose callee FID equals THIS section: stamps every
        ``header_hole_offsets`` slot with THIS section's offset
        (sibling sections that share a FID re-stamp last-write-wins —
        the loader disambiguates via the per-call ``callee_vkey``)
        and stamps every ``per_variant_holes`` whose ``callee_vkey``
        is in THIS section's local table. Holes whose vkey is not in
        this section's table stay in ``_pending_holes`` for a later
        sibling (or for :meth:`finalize` to stamp as
        :data:`MISSING_VARIANT_INDEX`).

        Returns ``(section_offset, section_length)`` — the start byte
        the section was opened at and the trailer-aligned byte width
        the section occupies in the bin. The length is what the
        per-binary ``matched_index.bin`` u24 stores; both are 4-byte
        aligned (the section trailer pad enforced above guarantees
        the length is a multiple of :data:`SECTION_ALIGNMENT`).
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

        section_offset = self._current_section_offset
        section_length = self._writer.cursor - section_offset
        fid = self._current_fid

        # Recover THIS section's variant table from its own bytes —
        # each section is self-describing. The memoryview is bounded
        # to [0, cursor); the section we just closed lies fully
        # inside that range. Released in a finally so the view does
        # not outlive the mmap.
        blob = self._writer.view()
        try:
            parsed, _end = parse_section_bin(blob, section_offset)
            vkey_to_idx = {
                v.variant_ref_offset: i for i, v in enumerate(parsed.variants)
            }
        finally:
            blob.release()

        # Resolve back-patches whose callee FID == THIS section's FID.
        # Header slots: stamp this section's offset every time a
        # sibling closes — the on-disk slot ends up pointing at the
        # LAST sibling to close (W2 last-write-wins). The accumulator
        # is intentionally NOT cleared so a subsequent sibling
        # re-patches the same slot. The slot's validity is implicit
        # in ``fid in _known_sections``, which :meth:`finalize` checks
        # before deciding a non-empty ``header_hole_offsets`` is a
        # builder bug.
        # Per-variant slots: stamp ONLY those whose ``callee_vkey``
        # is in THIS section's local variant table. Removed once
        # matched (no other sibling carries the same vkey by
        # construction — each variant_ref_offset is unique to one
        # sibling's variant table). Non-matching ones stay for a
        # later sibling (or :meth:`finalize`) to handle.
        packed_section_offset = struct.pack("<I", section_offset)
        for (hole_fid, _ref_offset), record in self._pending_holes.items():
            if hole_fid != fid:
                continue
            for header_slot in record.header_hole_offsets:
                self._writer.patch(header_slot, packed_section_offset)
            remaining: list[tuple[int, Hashable]] = []
            for slot_offset, callee_vkey in record.per_variant_holes:
                variant_idx = vkey_to_idx.get(callee_vkey)
                if variant_idx is None:
                    remaining.append((slot_offset, callee_vkey))
                    continue
                self._writer.patch(slot_offset, struct.pack("<H", variant_idx))
            record.per_variant_holes = remaining

        # Clear per-section state.
        self._current_fid = None
        self._current_section_offset = None
        self._n_variants_slot = None
        self._current_call_targets = []
        self._current_variant_count = 0
        self._current_variant_n_calls_slot = None

        return section_offset, section_length

    def finalize(self) -> None:
        """Close the underlying memmap; resolve or assert on remaining holes.

        Any remaining ``header_hole_offsets`` are a HARD ERROR: the
        callee section was never written (builder bug — a call_target
        rows references a FID that no section opens). Raises with
        the unresolved FIDs.

        Any remaining ``per_variant_holes`` are a legitimate
        corpus-scale outcome (cross-arm/cross-section vkey mismatch:
        caller and callee survived pass-1 with disjoint vkey sets).
        The slot is patched with :data:`MISSING_VARIANT_INDEX` and
        — if a warn-log was supplied — one line is appended naming
        the callee FID, the unresolved vkey, and the referencing
        section's offset. The loader treats the sentinel as "no
        inlined callee body available for this vkey".

        Belt-and-braces: after the pending-holes sweep, scan the
        entire written bin for any remaining
        :data:`UNRESOLVED_VARIANT_INDEX` slot. The pre-section
        back-patch path plus the finalize-time MISSING stamp should
        have eliminated every ``0xFFFF`` — a leak indicates a writer
        bug (e.g. forgetting to register a per-variant slot in the
        hole record); this sweep surfaces it before the bin is
        sealed.

        The memmap is closed unconditionally in a ``finally``: if any
        check raises, the underlying bin still gets unmapped +
        truncated rather than leaking until process exit.
        """
        try:
            if self._current_fid is not None:
                raise ValueError(
                    "finalize called while section "
                    f"function_name_ptr={self._current_fid} is still open"
                )
            self._resolve_or_stamp_remaining_holes()
            self._sweep_for_unresolved_sentinels()
        finally:
            self.close()

    def _resolve_or_stamp_remaining_holes(self) -> None:
        """Walk ``_pending_holes`` at finalize.

        A record's ``header_hole_offsets`` is allowed to be non-empty
        IFF ``fid in _known_sections`` — the slots have been patched
        (possibly repeatedly by sibling closes) and the accumulator
        is kept around purely so :meth:`end_section` can re-patch on
        the next sibling close. A record with non-empty
        ``header_hole_offsets`` AND ``fid not in _known_sections`` is
        a HARD ERROR: a call_target referenced a FID that no section
        ever opened (builder bug).

        Per-variant holes left at finalize get
        :data:`MISSING_VARIANT_INDEX` stamped, with one warn-log line
        each (silently stamped when no warn-log was supplied).
        """
        unresolved: list[tuple[int, int]] = [
            (fid, ref_offset)
            for (fid, ref_offset), record in self._pending_holes.items()
            if record.header_hole_offsets and fid not in self._known_sections
        ]
        if unresolved:
            sorted_unresolved = sorted(unresolved)
            raise ValueError(
                f"finalize: {len(sorted_unresolved)} call_target row(s) "
                "reference a callee section that was never written: "
                f"(callee_fid, referencing_section_offset)={sorted_unresolved!r}"
            )

        # Header slots are all resolved. Stamp MISSING on every
        # remaining per-variant hole and (optionally) log a line each.
        for (fid, ref_offset), record in self._pending_holes.items():
            for slot_offset, callee_vkey in record.per_variant_holes:
                self._writer.patch(
                    slot_offset, struct.pack("<H", MISSING_VARIANT_INDEX)
                )
                if self._warn_log is not None:
                    self._warn_log.write(
                        f"missing_variant: callee_fid={fid} "
                        f"callee_vkey={callee_vkey!r} "
                        f"caller_section@{ref_offset}\n"
                    )
        self._pending_holes.clear()

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
        ``(callee_fid, current_section_offset)`` pair — the
        ``_pending_holes`` key shape carries the dedup. If a hole was
        already opened by a previous slot in this section (e.g.
        another call_target row with the same FID-but-different-type,
        or an emit_per_call_entries miss that ran first under a
        different ordering), the existing record gains another header
        slot in its ``header_hole_offsets`` list.
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
        a fresh record. ``callee_vkey`` is the value that will appear
        as ``variant_ref_offset`` on the matching variant in the
        callee section — see :class:`PerCallEntry`.
        """
        record = self._get_or_create_section_hole(callee_fid)
        record.per_variant_holes.append((slot_offset, callee_vkey))

    def _get_or_create_section_hole(self, callee_fid: int) -> _HoleRecord:
        """Return THIS section's HoleRecord for ``callee_fid``, creating it
        on first use.

        Centralises the deduplication invariant (one record per
        ``(callee_fid, current_section_offset)``) so callers — both
        the header-hole opener and the per-variant-hole opener —
        share the same accumulator.
        """
        key = (callee_fid, self._current_section_offset)
        record = self._pending_holes.get(key)
        if record is None:
            record = _HoleRecord()
            self._pending_holes[key] = record
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
