"""Binary codec for ``<binary>_sections.bin`` — matched + unmatched section catalog.

Single concern: encode/decode the on-disk layout of the section catalog
that the dataloader reads in lieu of the legacy ``<binary>_sections.csv``.
Every section records a function's header (function_name_ptr,
n_call_targets, n_variants), a per-section ``n_variants × u16`` jump
table that carries each variant's per-call entry count (used by the
reader to address variant_i in O(1)), the typed call_target table
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
  a hit stamps the resolved index directly, a miss defers as
  :data:`UNRESOLVED_VARIANT_INDEX` plus a "this section is waiting on
  ``callee_fid``" marker in :attr:`SectionWriter._pending_holes`.
  Forward references stamp the same sentinel + marker. At every
  :meth:`SectionWriter.end_section`, the writer re-parses the
  just-closed section AND every caller section the marker points
  at, derives slot positions from the bin's self-describing bytes,
  and resolves any per-call slot whose owning caller variant's
  ``variant_ref_offset`` is in THIS section's variant table.
  Sibling sections with disjoint vkey sets each patch only their
  own matching slots; slots whose vkey is never registered by any
  sibling fall through to :meth:`SectionWriter.finalize`, which
  stamps :data:`MISSING_VARIANT_INDEX` and emits a one-line
  ``warn-log`` entry (so the corpus rebuild can audit how often the
  cross-arm vkey mismatch fires).

Each section is self-describing: both the variant table needed to
resolve back-patches AND the slot byte positions inside per-call
entries are recoverable from the section's own bytes. No slot
byte-offset cache or cross-section variant map is kept in writer
memory; :attr:`SectionWriter._pending_holes` only records WHICH
caller sections are waiting on a given callee FID.

A finalize-time sweep asserts no ``0xFFFF`` (UNRESOLVED) sentinel
leaked through to the on-disk bytes. Any FID still in
:attr:`SectionWriter._pending_holes` whose callee section was never
written is a hard builder bug (raises); FIDs whose section IS in
:attr:`SectionWriter._known_sections` get a per-caller-section
re-scan that stamps :data:`MISSING_VARIANT_INDEX` on every remaining
``0xFFFF`` per-call slot pointing at that FID + one warn-log line
each (legitimate cross-arm vkey mismatch).

The wire format is documented in detail in
``polished-greeting-moler.md`` (Approach → A. Binary section file
layout). All multi-byte integers are little-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterator, Optional, TextIO

from dedup_hashmap import HashMapU32U32

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

#: Bytes per per-section jump-table entry. The jump table sits immediately
#: after the section header and carries ``n_variants × u16`` lengths,
#: encoded as ``(variant_i_total_bytes - VARIANT_HEADER_SIZE) >> 2``. With
#: :data:`VARIANT_HEADER_SIZE` = 8 and :data:`PER_CALL_ENTRY_SIZE` = 4 this
#: evaluates to ``n_calls_for_variant_i``, so the reader can address
#: variant_i in O(1) via ``cumsum(jump_table) * 4 + arange(...) * 8``
#: rather than the variable-length variant walk it used to need.
JUMP_TABLE_ENTRY_SIZE: int = 2

#: Bytes per call_target table entry
#: (``u32 function_name_ptr | u32 function_section_ptr | u16 flags | u16 reserved``).
CALL_TARGET_ENTRY_SIZE: int = 12

#: Bytes per variant header
#: (``u32 variant_ref_offset | u32 data_offset_shifted``). ``n_calls`` lives
#: in the section's jump table (one ``u16`` per variant) so the reader does
#: not need to walk every prior variant to address variant_i.
VARIANT_HEADER_SIZE: int = 8

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
# Writer
# ---------------------------------------------------------------------------


class SectionWriter:
    """Memmap-backed writer for ``<binary>_sections.bin`` with back-patching.

    Lifecycle per section, called in order:

    1. :meth:`begin_section` — aligns + records the section's offset,
       and stashes the caller-declared ``n_variants`` so
       :meth:`emit_call_targets` can size the per-section jump-table
       reservation that immediately follows the 8-byte header.
    2. :meth:`emit_call_targets` — stamps the section header (with
       ``n_variants = 0`` placeholder), reserves ``n_variants × u16``
       zero bytes for the jump table, then writes the call_target
       table. The header's ``n_variants`` is patched at
       :meth:`end_section`.
    3. Per variant:
        a. :meth:`begin_variant` — stamps the 8-byte variant header
           (``u32 variant_ref_offset | u32 data_offset_shifted``);
           ``n_calls`` lives in the section's jump table, not the
           variant header.
        b. :meth:`emit_per_call_entries` — writes the per-call slots.
           Backward references re-parse the callee section pointed at
           by ``_known_sections[callee_fid]`` and stamp the resolved
           variant idx directly on a vkey hit; misses (and forward
           references) defer to a back-patch hole. Stamps this
           variant's jump-table slot afterwards.
        c. :meth:`end_variant` — increments the section's variant
           count.
    4. :meth:`end_section` — back-patches ``n_variants``, pads to a
       4-byte boundary, parses the just-closed section's bytes to
       recover its variant table, resolves any self-references the
       just-closed section made into itself (the "callee_fid ==
       my_fid" case that ``emit_per_call_entries`` deferred), then
       walks every caller section in ``_pending_holes[my_fid]`` and
       re-parses each caller's own bytes to stamp the call_target's
       ``function_section_ptr`` AND any per-call entry whose owning
       caller variant's ``variant_ref_offset`` matches a variant in
       THIS section's local table.
    5. After all sections: :meth:`finalize` — for every FID still in
       ``_pending_holes``: if the callee section was never written
       it raises (builder bug); otherwise the per-caller-section
       scan stamps :data:`MISSING_VARIANT_INDEX` on any remaining
       ``UNRESOLVED`` slot pointing at that FID (cross-arm vkey
       mismatch — one warn-log line per stamp). Then runs a
       belt-and-braces sweep for any leaked ``0xFFFF`` and closes
       the underlying memmap.
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
        # walking via per-call ``callee_vkey``. Both key (u32
        # function_name_ptr) and value (u32 section offset, by the
        # bin's wire layout) fit in u32, so the map is backed by the
        # ``dedup_hashmap`` crate's :class:`HashMapU32U32` for a
        # halved per-entry footprint at corpus scale.
        self._known_sections = HashMapU32U32()
        # ``_pending_holes`` records which caller sections have at
        # least one unresolved slot pointing at a given callee FID.
        # NO slot byte-offsets cached, NO callee_vkeys stashed —
        # both are re-derived from the bin's self-describing bytes at
        # fill time (caller's call_targets + variants region carry
        # ``name_ptr`` + ``variant_ref_offset`` natively). Membership
        # of ``caller_section_offset`` in the set is the only signal
        # the writer needs to revisit a caller when a callee closes.
        self._pending_holes: dict[int, set[int]] = {}

        # Per-section state (cleared on every begin_section).
        self._current_fid: Optional[int] = None
        self._current_section_offset: Optional[int] = None
        # Offset of THIS section's u16 n_variants slot (patched in end_section).
        self._n_variants_slot: Optional[int] = None
        # Caller-declared variant count for THIS section. Used to size the
        # jump-table reservation in :meth:`emit_call_targets` and asserted
        # against the observed count at :meth:`end_section`.
        self._current_n_variants_declared: Optional[int] = None
        # File offset of THIS section's jump table (first u16 slot). The
        # table is ``n_variants × u16`` immediately after the 8-byte
        # section header, so this is ``section_offset + SECTION_HEADER_SIZE``.
        self._current_jump_table_offset: Optional[int] = None
        # Specs of the section's call_targets, kept around so
        # emit_per_call_entries can validate that a PerCallEntry's
        # called_idx points at the call_target whose FID the entry
        # declares.
        self._current_call_targets: list[CallTargetSpec] = []
        # variant_idx → variant_count assigned so far in this section.
        self._current_variant_count: int = 0
        # Whether a variant is currently open (between :meth:`begin_variant`
        # and the corresponding :meth:`end_variant`). Per-call counts now
        # live in the section's jump table, not the variant header, so the
        # writer no longer tracks a per-variant slot offset.
        self._current_variant_open: bool = False

    # ------------------------------------------------------------------
    # Section lifecycle
    # ------------------------------------------------------------------

    def begin_section(self, function_name_ptr: int, n_variants: int) -> int:
        """Open a new section for ``function_name_ptr``.

        Aligns the cursor up to a 4-byte boundary (pads the gap with
        zero bytes), records the section's start offset in
        ``known_sections``, and returns it. Forward references to this
        FID emitted by EARLIER sections remain in ``_pending_holes`` —
        they're resolved when this section closes via
        :meth:`end_section`.

        ``n_variants`` is the exact number of variants the caller is
        about to emit. It is used to reserve ``n_variants × u16`` bytes
        for the per-section jump table immediately after the section
        header (see :data:`JUMP_TABLE_ENTRY_SIZE`). The reader uses that
        table to address variant_i in O(1). Declaring a count that
        differs from the actual number of :meth:`end_variant` calls is
        rejected at :meth:`end_section`; the writer cannot recover from
        the mismatch because the call_targets block sits at a fixed
        offset past the table.

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
        if n_variants < 0:
            raise ValueError(
                f"n_variants must be non-negative, got {n_variants}"
            )
        if n_variants > UNRESOLVED_VARIANT_INDEX - 1:
            raise ValueError(
                f"section for function_name_ptr={function_name_ptr} declares "
                f"n_variants={n_variants}; max is {UNRESOLVED_VARIANT_INDEX} "
                f"per section (u16 slot reserves 0xFFFF as the "
                f"unresolved-hole sentinel)"
            )
        self._pad_to_alignment()
        section_offset = self._writer.cursor

        self._known_sections.set(function_name_ptr, section_offset)

        self._current_fid = function_name_ptr
        self._current_section_offset = section_offset
        self._n_variants_slot = None
        self._current_n_variants_declared = n_variants
        # The jump table starts immediately after the section header. It is
        # written by emit_call_targets (which knows its own header_offset),
        # but the offset is deterministic so we cache it here for
        # emit_per_call_entries to stamp into.
        self._current_jump_table_offset = section_offset + SECTION_HEADER_SIZE
        self._current_call_targets = []
        self._current_variant_count = 0
        self._current_variant_open = False
        return section_offset

    def emit_call_targets(self, call_targets: list[CallTargetSpec]) -> None:
        """Stamp the section header + jump-table reservation + call_target table.

        ``n_variants`` is stamped at ``0`` in the header and patched in
        :meth:`end_section` from the observed variant count. The jump
        table sits between the header and the call_target table; it is
        reserved with zero bytes here and each entry is stamped by
        :meth:`emit_per_call_entries` for its owning variant. The caller
        is responsible for having deduplicated ``call_targets`` by
        ``(function_name_ptr, type)``; SectionWriter does not check.
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

        # Reserve the per-section jump table immediately after the header.
        # The reservation is zero-initialised; each entry is patched by
        # :meth:`emit_per_call_entries` when the owning variant's per-call
        # entries are written. Sized from the caller-declared n_variants
        # so the call_target table that follows starts at a deterministic
        # offset.
        jump_table_bytes = self._current_n_variants_declared * JUMP_TABLE_ENTRY_SIZE
        if jump_table_bytes:
            self._writer.write(b"\x00" * jump_table_bytes)

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
        """Stamp the variant header.

        The variant header is an 8-byte
        ``u32 variant_ref_offset | u32 data_offset_shifted``; the
        per-variant ``n_calls`` lives in the section's jump table and is
        stamped by :meth:`emit_per_call_entries` once the entries are
        written. Cursor is left at the start of the per-call entries.
        """
        self._assert_section_open()
        if self._n_variants_slot is None:
            raise ValueError(
                "begin_variant called before emit_call_targets; the "
                "section header must be stamped first"
            )
        if self._current_variant_open:
            raise ValueError(
                "begin_variant called while a previous variant is still "
                "open; call emit_per_call_entries + end_variant first"
            )
        if self._current_variant_count >= self._current_n_variants_declared:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} declared "
                f"n_variants={self._current_n_variants_declared} at "
                f"begin_section but begin_variant was called a "
                f"{self._current_variant_count + 1}-th time; the jump-table "
                f"reservation has no slot for this variant"
            )

        variant_header = struct.pack(
            "<II",
            variant_ref_offset,
            data_offset_shifted,
        )
        self._writer.write(variant_header)
        self._current_variant_open = True

    def emit_per_call_entries(self, entries: list[PerCallEntry]) -> None:
        """Write the variant's per-call entries + stamp its jump-table slot.

        Backward references (``callee_fid in _known_sections``) re-parse
        the callee section pointed at by ``_known_sections[callee_fid]``
        — the LAST sibling closed, which is also the offset that ends
        up in the call_target's ``function_section_ptr`` after sibling
        last-write-wins. The just-parsed section's local
        ``variant_ref_offset -> variant_idx`` map is consulted for the
        entry's ``callee_vkey``: a hit stamps the resolved index
        directly. A miss — or a forward reference whose callee section
        has not been opened yet, or a self-reference whose section is
        still in flight — stamps :data:`UNRESOLVED_VARIANT_INDEX` and
        records THIS section's offset in
        ``_pending_holes[callee_fid]``. The marker is the only thing
        the writer needs: at the callee's :meth:`end_section` the
        writer re-parses both that section AND every caller it has
        marked, and re-derives the slot byte offsets from the bin's
        self-describing bytes (jump table + variants region).
        Self-references skip the marker: the slot is resolved by
        :meth:`end_section`'s "step 2" self-resolve pass on the
        just-closed section's own bytes, keeping the self-call path
        disjoint from the sibling-close path so the two never
        double-patch the same slot.

        Anything still unresolved at :meth:`finalize` (cross-arm vkey
        mismatch) gets :data:`MISSING_VARIANT_INDEX` + a warn-log
        line.

        After the entries are written the section's jump table receives
        ``jump_table[current_variant_idx] = len(entries)`` so the reader
        can address variant_i in O(1).
        """
        self._assert_variant_open()

        for entry in entries:
            self._assert_called_idx_matches(entry)
            section_variant_index = self._resolve_backward_variant_index(
                callee_fid=entry.callee_function_name_ptr,
                callee_vkey=entry.callee_vkey,
            )
            if section_variant_index is None:
                # Forward reference, backward reference whose vkey
                # is not in the callee section's local variant table,
                # or self-reference (callee_fid == my_fid, section
                # still in flight). Stamp the unresolved sentinel; a
                # future end_section (sibling close, or our own close
                # for self-refs) or finalize will patch.
                section_variant_index = UNRESOLVED_VARIANT_INDEX
                if entry.callee_function_name_ptr != self._current_fid:
                    # Sibling-close path: mark THIS section as waiting
                    # on callee_fid. Self-call (callee_fid == my_fid)
                    # is deliberately excluded — :meth:`end_section`'s
                    # step-2 self-resolve pass handles it from our
                    # own variant table, so keeping the two paths
                    # disjoint avoids double-patching.
                    self._pending_holes.setdefault(
                        entry.callee_function_name_ptr, set()
                    ).add(self._current_section_offset)
            self._writer.write(
                struct.pack("<HH", entry.called_idx, section_variant_index)
            )

        # Stamp the jump-table slot for THIS variant. ``_current_variant_count``
        # is the 0-based index of the currently-open variant (incremented at
        # :meth:`end_variant`), which is exactly the slot we want.
        n_calls = len(entries)
        if n_calls > 0xFFFF:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} "
                f"variant_idx={self._current_variant_count} has {n_calls} "
                f"per-call entries; max is {0xFFFF} (u16 jump-table slot)"
            )
        jump_table_slot = (
            self._current_jump_table_offset
            + self._current_variant_count * JUMP_TABLE_ENTRY_SIZE
        )
        self._writer.patch(jump_table_slot, struct.pack("<H", n_calls))

    def _resolve_backward_variant_index(
        self, *, callee_fid: int, callee_vkey: Hashable
    ) -> Optional[int]:
        """Resolve a backward-reference variant idx by re-parsing the
        callee section pointed at by ``_known_sections[callee_fid]``.

        ``None`` when the callee FID has no closed section yet
        (forward reference) OR the callee FID is THIS section (the
        in-flight section's header carries ``n_variants=0`` placeholder
        until :meth:`end_section`, so its bytes are not yet
        parser-readable — :meth:`end_section`'s step-2 self-resolve
        pass handles it once the section's own bytes describe its
        variant table) OR the closed section's local variant table
        does not carry ``callee_vkey`` (legitimate cross-arm vkey
        mismatch — a sibling close or :meth:`finalize` will sweep
        the slot to :data:`MISSING_VARIANT_INDEX`).

        The section we re-parse is the SAME one whose offset will end
        up in the call_target row's ``function_section_ptr`` after the
        sibling last-write-wins, so the loader can never observe a
        ``(section_offset, variant_idx)`` pair that points into the
        wrong sibling's variant table.

        ``parse_section_bin`` is given a bounded memoryview of the
        already-written region; the view is released in a ``finally``
        to keep the mmap unmappable on later finalize.
        """
        if callee_fid == self._current_fid:
            return None
        section_offset = self._known_sections.get(callee_fid)
        if section_offset is None:
            return None
        blob = self._writer.view()
        try:
            parsed, _end = parse_section_bin(blob, section_offset)
        finally:
            blob.release()
        for i, variant in enumerate(parsed.variants):
            if variant.variant_ref_offset == callee_vkey:
                return i
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
        self._current_variant_open = False
        return variant_idx

    def end_section(self) -> tuple[int, int]:
        """Close the current section.

        Patches ``n_variants``, pads to a 4-byte boundary, then parses
        the just-closed section back from its own bytes to recover
        the variant table. Resolves back-patches in two disjoint
        sweeps, each parsing on-wire bytes (no writer-side slot-offset
        cache):

        * **Step 2 — self-resolve.** Iterate the just-closed
          section's own call_targets; any row whose ``name_ptr`` is
          in ``_known_sections`` (in single-threaded writing this is
          exactly the self-call case — non-self backward refs were
          stamped inline at ``emit_call_targets`` time because the
          callee was already in ``_known_sections`` then) gets its
          ``function_section_ptr`` re-stamped (idempotent for the
          non-self case) AND every per-call entry whose ``called_idx``
          points at that row gets its ``section_variant_index``
          resolved from the callee section's variant table. For the
          self-call, the callee section IS this section, and its
          variant table is exactly the one we just parsed.

        * **Step 3 — sibling-close patches.** For each
          ``caller_section_offset`` in
          ``_pending_holes.get(my_fid, ())``: re-parse that caller
          section's bytes and (Case A) re-stamp every
          ``function_section_ptr`` slot whose ``name_ptr == my_fid``
          to THIS section's offset (W2 last-write-wins on sibling
          collisions; the loader disambiguates via per-call
          ``callee_vkey``), and (Case B) for every per-call entry
          whose owning caller variant's ``variant_ref_offset`` is in
          THIS section's local table AND whose slot is still
          ``UNRESOLVED``, stamp the resolved index. The marker set
          is NOT popped — a later sibling's close re-walks the same
          callers (overwriting Case A, filling more of Case B's
          slots), and :meth:`finalize` consults the same map for
          the MISSING-stamping sweep.

        Returns ``(section_offset, section_length)`` — the start byte
        the section was opened at and the trailer-aligned byte width
        the section occupies in the bin. The length is what the
        per-binary ``matched_index.bin`` u24 stores; both are 4-byte
        aligned (the section trailer pad enforced above guarantees
        the length is a multiple of :data:`SECTION_ALIGNMENT`).
        """
        self._assert_section_open()
        if self._current_variant_open:
            raise ValueError(
                "end_section called while a variant is still open; "
                "call end_variant first"
            )
        if self._current_variant_count != self._current_n_variants_declared:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} declared "
                f"n_variants={self._current_n_variants_declared} at "
                f"begin_section but emitted {self._current_variant_count}; "
                f"the jump-table reservation cannot be retroactively resized "
                f"because the call_targets block sits at a fixed offset past it"
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
        # each section is self-describing. The vkey_to_idx map is
        # used by both the step-2 self-resolve pass (for self-call
        # per-call entries) and step-3 sibling-close Case B (for
        # caller per-call entries).
        blob = self._writer.view()
        try:
            parsed_self, _end = parse_section_bin(blob, section_offset)
        finally:
            blob.release()
        my_vkey_to_idx: dict[Hashable, int] = {
            v.variant_ref_offset: i for i, v in enumerate(parsed_self.variants)
        }

        # Step 2: self-resolve. Re-stamp call_target rows + resolve
        # per-call entries for any row whose ``name_ptr`` is in
        # ``_known_sections``. For non-self refs this is idempotent
        # (the slot was stamped inline at emit time); the load-bearing
        # case is the self-call, whose per-call slots are still
        # ``UNRESOLVED`` because ``_resolve_backward_variant_index``
        # bails out when ``callee_fid == self._current_fid``.
        self._resolve_caller_section(
            caller_section_offset=section_offset,
            callee_fid=fid,
            callee_section_offset=section_offset,
            callee_vkey_to_idx=my_vkey_to_idx,
        )

        # Step 3: sibling-close patches. Walk every caller section
        # that is waiting on my_fid; re-derive slot positions from
        # their own bytes. Do NOT pop — siblings still to close and
        # :meth:`finalize` both re-walk the same set.
        for caller_section_offset in self._pending_holes.get(fid, ()):
            self._resolve_caller_section(
                caller_section_offset=caller_section_offset,
                callee_fid=fid,
                callee_section_offset=section_offset,
                callee_vkey_to_idx=my_vkey_to_idx,
            )

        # Clear per-section state.
        self._current_fid = None
        self._current_section_offset = None
        self._n_variants_slot = None
        self._current_n_variants_declared = None
        self._current_jump_table_offset = None
        self._current_call_targets = []
        self._current_variant_count = 0
        self._current_variant_open = False

        return section_offset, section_length

    def finalize(self) -> None:
        """Close the underlying memmap; resolve or assert on remaining holes.

        Any callee FID in ``_pending_holes`` that is NOT in
        ``_known_sections`` is a HARD ERROR: at least one caller
        section references a FID whose section was never written
        (builder bug). Raises with the offending FIDs +
        ``caller_section_offset`` pairs.

        Any callee FID in ``_pending_holes`` that IS in
        ``_known_sections`` may still carry caller per-call slots
        that no sibling registered (cross-arm/cross-section vkey
        mismatch: caller and callee survived pass-1 with disjoint
        vkey sets). For each such caller, re-parse the caller's
        bytes, find every per-call slot pointing at the FID that is
        still ``UNRESOLVED``, stamp :data:`MISSING_VARIANT_INDEX`,
        and — if a warn-log was supplied — append one line naming
        the callee FID, the caller variant's vkey, and the caller
        section's offset.

        Belt-and-braces: after the pending-holes sweep, scan the
        entire written bin for any remaining
        :data:`UNRESOLVED_VARIANT_INDEX` slot. The sibling-close
        patches plus the finalize-time MISSING stamp should have
        eliminated every ``0xFFFF`` — a leak indicates a writer bug
        (e.g. forgetting to mark a caller as waiting on the
        callee_fid); this sweep surfaces it before the bin is sealed.

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

        A FID whose section never opened (``fid not in
        _known_sections``) is a HARD ERROR — at least one caller's
        ``function_section_ptr`` placeholder still points at zero
        and the loader has no way to recover. Raises with the
        ``(callee_fid, caller_section_offset)`` pairs that triggered
        the failure.

        For every other FID, re-parse each caller section in the
        set and stamp :data:`MISSING_VARIANT_INDEX` on each per-call
        slot whose ``called_idx`` points at a call_target row
        referencing this FID AND is still ``UNRESOLVED`` (a sibling
        close patched the ones it could). One warn-log line is
        appended per stamp; the caller variant's ``variant_ref_offset``
        carries the unresolved vkey (Step 7 invariant: the per-call
        entry's intended callee_vkey is exactly its owning caller
        variant's vkey).
        """
        unresolved: list[tuple[int, int]] = [
            (fid, caller_offset)
            for fid, caller_offsets in self._pending_holes.items()
            for caller_offset in caller_offsets
            if fid not in self._known_sections
        ]
        if unresolved:
            sorted_unresolved = sorted(unresolved)
            raise ValueError(
                f"finalize: {len(sorted_unresolved)} call_target row(s) "
                "reference a callee section that was never written: "
                f"(callee_fid, referencing_section_offset)={sorted_unresolved!r}"
            )

        # Every remaining FID is known; what's left is per-call slots
        # whose callee_vkey never appeared in any sibling's variant
        # table. Stamp MISSING + warn-log per slot.
        for fid, caller_offsets in self._pending_holes.items():
            for caller_offset in caller_offsets:
                self._stamp_missing_in_caller(
                    caller_section_offset=caller_offset,
                    callee_fid=fid,
                )
        self._pending_holes.clear()

    def _stamp_missing_in_caller(
        self, *, caller_section_offset: int, callee_fid: int
    ) -> None:
        """Walk one caller section's bytes; stamp MISSING on every
        per-call slot still pointing at ``callee_fid`` as unresolved.

        Slot positions are re-derived from the caller's own bytes
        (call_targets + variants region): no writer-side
        slot-offset cache. One warn-log line per stamp.
        """
        blob = self._writer.view()
        try:
            caller, _end = parse_section_bin(blob, caller_section_offset)
        finally:
            blob.release()
        target_called_idxs = {
            i
            for i, ct in enumerate(caller.call_targets)
            if ct.function_name_ptr == callee_fid
        }
        if not target_called_idxs:
            return
        for v_idx, variant in enumerate(caller.variants):
            variant_offset = _variant_block_offset(caller, v_idx)
            entry_offset = variant_offset + VARIANT_HEADER_SIZE
            for called_idx, sv_idx in variant.per_call_entries:
                slot_offset = entry_offset + 2
                entry_offset += PER_CALL_ENTRY_SIZE
                if called_idx not in target_called_idxs:
                    continue
                if sv_idx != UNRESOLVED_VARIANT_INDEX:
                    continue
                self._writer.patch(
                    slot_offset, struct.pack("<H", MISSING_VARIANT_INDEX)
                )
                if self._warn_log is not None:
                    self._warn_log.write(
                        f"missing_variant: callee_fid={callee_fid} "
                        f"callee_vkey={variant.variant_ref_offset!r} "
                        f"caller_section@{caller_section_offset}\n"
                    )

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
        if not self._current_variant_open:
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
        ``known_sections``; miss ⇒ mark THIS section as waiting on
        ``callee_fid`` (so the callee's :meth:`end_section` re-visits
        us via Case A) and write ``0`` as a placeholder.

        EXTERN: write ``extern_provider_line_no`` if provided, else
        :data:`UNKNOWN_EXTERN_PROVIDER` (= 0).
        """
        if spec.type is CallTargetType.EXTERN:
            if spec.extern_provider_line_no is None:
                return UNKNOWN_EXTERN_PROVIDER
            return spec.extern_provider_line_no

        # LOCAL / PLT.
        known = self._known_sections.get(spec.function_name_ptr)
        if known is not None:
            return known
        # Forward reference: mark this caller as waiting on the
        # callee FID. Slot byte-offset is re-derived from the
        # caller's call_targets bytes at the callee's
        # :meth:`end_section`, so no slot-offset is cached here.
        self._pending_holes.setdefault(
            spec.function_name_ptr, set()
        ).add(self._current_section_offset)
        return 0  # placeholder; patched in end_section of the callee.

    def _resolve_caller_section(
        self,
        *,
        caller_section_offset: int,
        callee_fid: int,
        callee_section_offset: int,
        callee_vkey_to_idx: dict[Hashable, int],
    ) -> None:
        """Re-parse one caller section's bytes; patch Cases A + B.

        * Case A (``function_section_ptr``): for every call_target
          row in the caller whose ``function_name_ptr == callee_fid``,
          stamp ``callee_section_offset`` into the row's
          ``function_section_ptr`` slot (W2 last-write-wins on
          sibling collisions).

        * Case B (``section_variant_index``): for every per-call
          entry in the caller whose ``called_idx`` points at one of
          those rows AND whose slot is still
          :data:`UNRESOLVED_VARIANT_INDEX`, look up the OWNING caller
          variant's ``variant_ref_offset`` (== the entry's intended
          ``callee_vkey`` by Step 7's on-wire invariant) in
          ``callee_vkey_to_idx``; on a hit, stamp the resolved
          index. A miss leaves the slot ``UNRESOLVED`` for a later
          sibling close (or :meth:`finalize`'s MISSING-stamp pass) to
          handle.

        Used by both the step-2 self-resolve pass
        (``caller_section_offset == callee_section_offset``) and the
        step-3 sibling-close pass. The two paths are disjoint by
        ``emit_per_call_entries``'s rule that self-call slots never
        appear in ``_pending_holes[my_fid]``, so the same call_target
        + per-call slot is never patched twice.
        """
        blob = self._writer.view()
        try:
            caller, _end = parse_section_bin(blob, caller_section_offset)
        finally:
            blob.release()

        # Case A: function_section_ptr re-stamps.
        call_targets_start = (
            caller_section_offset
            + SECTION_HEADER_SIZE
            + len(caller.variants) * JUMP_TABLE_ENTRY_SIZE
        )
        target_called_idxs: set[int] = set()
        packed_section_offset = struct.pack("<I", callee_section_offset)
        for i, ct in enumerate(caller.call_targets):
            if ct.function_name_ptr != callee_fid:
                continue
            target_called_idxs.add(i)
            # function_section_ptr is the second u32 of the row.
            row_offset = call_targets_start + i * CALL_TARGET_ENTRY_SIZE
            self._writer.patch(row_offset + 4, packed_section_offset)

        if not target_called_idxs:
            return

        # Case B: per-call entry section_variant_index resolves.
        for v_idx, variant in enumerate(caller.variants):
            resolved_idx = callee_vkey_to_idx.get(variant.variant_ref_offset)
            if resolved_idx is None:
                # Caller variant's vkey not in the callee's local
                # variant table; leave its slots alone — a sibling
                # close or :meth:`finalize` may resolve / MISSING-
                # stamp them.
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
                    # Already patched by a prior sibling close (or
                    # by step-2 self-resolve for the self-call). Do
                    # not overwrite — the existing value carries a
                    # different sibling's variant index that the
                    # loader will follow via ``function_section_ptr``
                    # = LAST sibling, so the slot is intentionally
                    # last-write-wins on the FIRST resolver that
                    # carries this vkey.
                    continue
                self._writer.patch(slot_offset, packed_resolved)

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


def _variant_block_offset(section: Section, variant_idx: int) -> int:
    """Return the file-offset of variant ``variant_idx``'s header.

    Derives the position directly from the section layout (section
    header + jump table + call_targets table + sum of preceding
    variant block sizes). The writer uses this to address per-call
    slot byte offsets when back-patching; the caller is responsible
    for ensuring ``variant_idx`` is a valid index into
    ``section.variants``.
    """
    variants_region_start = (
        section.section_offset
        + SECTION_HEADER_SIZE
        + len(section.variants) * JUMP_TABLE_ENTRY_SIZE
        + len(section.call_targets) * CALL_TARGET_ENTRY_SIZE
    )
    preceding_entries = sum(
        len(section.variants[i].per_call_entries) for i in range(variant_idx)
    )
    return (
        variants_region_start
        + variant_idx * VARIANT_HEADER_SIZE
        + preceding_entries * PER_CALL_ENTRY_SIZE
    )


def parse_section_bin(blob: memoryview, offset: int) -> tuple[Section, int]:
    """Parse one section starting at ``offset`` in ``blob``.

    Returns ``(Section, end_offset)`` where ``end_offset`` is past the
    section's trailing alignment padding (so the caller can call
    again with the new offset to read the next section).

    Wire format from ``offset``:

    1. 8 B section header — ``<IHH`` →
       ``function_name_ptr | n_call_targets | n_variants``.
    2. ``n_variants × u16`` jump table — slot ``i`` holds the number of
       per-call entries that variant ``i``'s block carries. ``cumsum``
       of the jump table gives variant-start offsets within the
       variants region, so the reader can address variant_i in O(1).
    3. ``n_call_targets × 12 B`` call_target rows — ``<IIHH`` →
       ``function_name_ptr | function_section_ptr | flags | reserved``.
    4. Variant blocks (one per declared variant). Each block is
       8 B header (``<II`` → ``variant_ref_offset | data_offset_shifted``)
       followed by ``jump_table[i] × 4 B`` per-call entries
       (``<HH`` → ``called_idx | section_variant_index``).
    5. Trailer pad up to :data:`SECTION_ALIGNMENT`.

    All multi-byte integers are little-endian.
    """
    section_offset = offset

    (
        function_name_ptr,
        n_call_targets,
        n_variants,
    ) = struct.unpack_from("<IHH", blob, offset)
    offset += SECTION_HEADER_SIZE

    jump_table: list[int] = list(
        struct.unpack_from(f"<{n_variants}H", blob, offset)
    )
    offset += n_variants * JUMP_TABLE_ENTRY_SIZE

    call_targets: list[CallTarget] = []
    for _ in range(n_call_targets):
        (
            ct_function_name_ptr,
            function_section_ptr,
            flags,
            _reserved,
        ) = struct.unpack_from("<IIHH", blob, offset)
        offset += CALL_TARGET_ENTRY_SIZE
        call_type, is_matched = _unpack_flags(flags)
        call_targets.append(
            CallTarget(
                function_name_ptr=ct_function_name_ptr,
                function_section_ptr=function_section_ptr,
                type=call_type,
                is_matched=is_matched,
            )
        )

    variants: list[VariantBlock] = []
    for n_calls in jump_table:
        variant_ref_offset, data_offset_shifted = struct.unpack_from(
            "<II", blob, offset
        )
        offset += VARIANT_HEADER_SIZE
        if n_calls:
            raw = struct.unpack_from(f"<{2 * n_calls}H", blob, offset)
            per_call_entries = list(zip(raw[0::2], raw[1::2]))
        else:
            per_call_entries = []
        offset += n_calls * PER_CALL_ENTRY_SIZE
        variants.append(
            VariantBlock(
                variant_ref_offset=variant_ref_offset,
                data_offset_shifted=data_offset_shifted,
                per_call_entries=per_call_entries,
            )
        )

    # Trailer pad — round up to SECTION_ALIGNMENT so the next section
    # starts on a 4-byte boundary (the writer pre-pays this pad at
    # :meth:`SectionWriter.end_section`).
    rem = offset % SECTION_ALIGNMENT
    if rem:
        offset += SECTION_ALIGNMENT - rem

    section = Section(
        function_name_ptr=function_name_ptr,
        section_offset=section_offset,
        call_targets=call_targets,
        variants=variants,
    )
    return section, offset


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
