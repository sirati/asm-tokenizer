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

The writer back-patches forward references in two places. Every
back-patch slot read/write addresses the relevant section's bytes
through the per-section jump table — the format is itself a jump
table designed for O(1) random access (``jump_table[i] =
n_calls_for_variant_i``, one ``u16`` each immediately after the
header), so a single ``np.cumsum`` over the jump-table bytes yields
every variant's byte offset. :class:`_SectionLayoutView` is the one
helper that turns a section's raw header + jump table into the
``uint16`` / ``uint32`` slot positions all back-patch paths need; NO
``CallTarget`` / ``VariantBlock`` Python object is materialised in the
resolve path, and NO cross-section variant map is carried in writer
memory:

* ``function_section_ptr`` on a call_target — set to ``0`` when
  emitting the call_target if the callee section hasn't been written
  yet; patched when the callee section closes. Sibling sections that
  share a ``function_name_ptr`` (clang's ``OUTLINED_FUNCTION_N``)
  each stamp their own offset over the placeholder; the loader walks
  via per-call ``callee_vkey`` to disambiguate which sibling carries
  the matching variant.
* ``section_variant_index`` inside a per-call entry — backward
  references (callee section already closed) read the section pointed
  at by ``_known_sections[callee_fid]`` (the LAST sibling closed, the
  same offset that ends up in the call_target's
  ``function_section_ptr``) via its jump-table layout and scan the
  on-disk ``variant_ref_offset`` array for the entry's ``callee_vkey``;
  a hit stamps the resolved index directly, a miss defers as
  :data:`UNRESOLVED_VARIANT_INDEX` plus a "this section is waiting on
  ``callee_fid``" marker in :attr:`SectionWriter._pending_holes`.
  Forward references stamp the same sentinel + marker. At every
  :meth:`SectionWriter.end_section`, the writer reads the just-closed
  section's jump-table layout AND every caller section the marker
  points at, derives slot positions from the jump-table cumsum, and
  resolves any per-call slot whose owning caller variant's
  ``variant_ref_offset`` is in THIS section's variant table.
  Sibling sections with disjoint vkey sets each patch only their
  own matching slots; slots whose vkey is never registered by any
  sibling fall through to :meth:`SectionWriter.finalize`, which
  stamps :data:`MISSING_VARIANT_INDEX` and emits a one-line
  ``warn-log`` entry (so the corpus rebuild can audit how often the
  cross-arm vkey mismatch fires).

Each section is self-describing: both the variant table needed to
resolve back-patches AND the slot byte positions inside per-call
entries are recoverable from the section's own jump table on demand
(O(n_variants) numpy cumsum per touch — NOT a memoized parsed cache,
so the no-parallel-indexing rule holds). No slot byte-offset cache or
cross-section variant map is kept in writer memory;
:attr:`SectionWriter._pending_holes` only records WHICH caller
sections are waiting on a given callee FID.

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

import numpy as np

from dedup_hashmap import HashMapU32U32

from tokenizer.aligned_data._matched_sections_variant_buffer import VariantBuffer
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


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    rem = value % alignment
    return value if rem == 0 else value + (alignment - rem)


def _padded_jump_table_bytes(n_variants: int) -> int:
    """Reserved byte width of the jump-table region for ``n_variants``.

    The table itself is ``n_variants × u16``, but when ``n_variants`` is
    odd that is a non-multiple of 4. Downstream callers reinterpret the
    bin as ``uint32`` to vectorise hole-fills, which requires the bytes
    that FOLLOW the jump table (call_targets, variants region) to start
    on a u32 boundary. Padding the table out to the next multiple of 4
    keeps the post-table offsets aligned without changing any per-slot
    semantics — the trailing u16 (when present) is a deterministic
    zero that the reader skips over.
    """
    return ((n_variants + 1) // 2) * 4

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

#: Section-header ``function_name_ptr`` is a u32 1-indexed line number
#: into ``<binary>_function_names.txt`` — always far below 2**31 (it is
#: bounded by the count of distinct names in a single binary). Bit 31 is
#: therefore free and carries the per-section ``duplicated`` marker: set
#: when every body of this canonical name was routed to the unmatched arm
#: because the name maps to several distinct functions in the binary
#: (calls into it stamp :data:`MISSING_VARIANT_INDEX`). The bit rides ONLY
#: on the section HEADER's name_ptr; call_target rows carry the clean FID,
#: so emit-time identity compares (``_known_sections``,
#: :meth:`_resolve_caller_section` Case A) are unaffected. Both section
#: parsers (:func:`parse_section_bin` and the columnar reader) mask it off
#: and surface it as a separate boolean, so every downstream FID consumer
#: still sees the clean line number.
_SECTION_DUPLICATED_BIT: int = 1 << 31
_SECTION_FID_MASK: int = _SECTION_DUPLICATED_BIT - 1

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

    ``resolved_section_variant_index`` overrides the entire resolution
    machinery: when not ``None`` the writer stamps it verbatim and opens
    NO back-patch hole, used when the caller already knows the slot is
    unresolvable (e.g. an edge into a callee whose name is duplicated in
    the binary, so no single variant table legitimately addresses the
    call — the caller passes :data:`MISSING_VARIANT_INDEX`).
    ``callee_vkey`` is then never consulted for this entry, but
    ``called_idx`` / ``callee_function_name_ptr`` still describe the
    real call edge.

    ``callee_occurrence`` is the single intended same-name sibling this
    call edge targets: when the callee FID has several same-FID sibling
    sections (clang ``OUTLINED_FUNCTION_N`` / per-TU static collisions,
    one section per ``(name, occurrence)`` body), this names WHICH
    sibling's variant table legitimately addresses the call. ``None``
    means "no disambiguation" — a non-duplicated callee (one section),
    OR a duplicated callee the producer could not pin to a single
    occurrence (ambiguous / no-occurrence), which it instead routes to
    :data:`MISSING_VARIANT_INDEX` via ``resolved_section_variant_index``.
    The writer carries an occurrence-bearing hole until the sibling whose
    own ``begin_section(occurrence=...)`` matches closes, then resolves
    BOTH the call_target ``function_section_ptr`` AND the per-call
    ``section_variant_index`` to that one sibling — never an arbitrary
    last-write-wins sibling. A ``None`` occurrence leaves today's
    single-section resolution path unchanged. Build-time only; NOT
    serialized to the wire (the on-disk format is occurrence-blind, the
    loader reads the already-disambiguated ``(function_section_ptr, J)``
    pair).
    """

    called_idx: int
    callee_function_name_ptr: int
    callee_vkey: Hashable
    resolved_section_variant_index: Optional[int] = None
    callee_occurrence: Optional[int] = None


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
    """Parsed section (one entry in the catalog).

    ``is_duplicated`` is the decoded section-header marker (see
    :data:`_SECTION_DUPLICATED_BIT`): ``True`` when this section is one
    of several same-name sibling bodies routed to the unmatched arm
    because the canonical name maps to several distinct functions in the
    binary. ``function_name_ptr`` is always the CLEAN line number (the
    marker bit is masked off here), so FID consumers are unaffected.
    """

    function_name_ptr: int
    section_offset: int
    call_targets: list[CallTarget]
    variants: list[VariantBlock]
    is_duplicated: bool = False


# ---------------------------------------------------------------------------
# Raw-bytes section layout view (back-patch addressing, parse-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SectionLayoutView:
    """Cumsum-derived slot addressing for ONE section, read from raw bytes.

    Single concern: turn a section's self-describing header + jump table
    (read directly from the writer's live ``uint8`` mmap view) into the
    byte/u16/u32 slot positions the back-patch paths need, WITHOUT
    materialising any ``CallTarget`` / ``VariantBlock`` Python object.
    The section format is itself a jump table designed for O(1) random
    access: ``jump_table[i] = n_calls_for_variant_i`` (one ``u16`` each,
    immediately after the 8-byte header), so a single ``np.cumsum`` over
    the jump-table bytes yields every variant's byte offset. This is
    computed on demand from the bin bytes each time a section is touched
    (O(n_variants) numpy), NOT a memoized parsed cache — the
    no-parallel-indexing rule still holds.

    All position arrays are expressed as element indices into the
    matching-dtype reinterpretation of the writer's buffer:

    * ``caller_vrefs`` — ``uint32`` ``variant_ref_offset`` per variant,
      in on-disk (``variant_ref_offset``-ascending) order.
    * ``called_idx_pos`` / ``sv_idx_pos`` — ``uint16`` index, relative
      to ``variants_region_start // 2``, of each per-call entry's
      ``called_idx`` / ``section_variant_index`` slot, in global entry
      order (variant ascending, entry ascending).
    * ``entry_to_variant`` — owning variant index per global per-call
      entry (same order as the ``*_pos`` arrays).

    Empty (``n_variants == 0`` or no per-call entries) sections yield
    zero-length arrays; callers branch on ``n_call_targets == 0`` /
    ``total_entries == 0`` exactly as the inline code did.
    """

    fid: int
    n_call_targets: int
    n_variants: int
    jump_table_offset: int
    call_targets_start: int
    variants_region_start: int
    section_end: int
    jump_slots: "np.ndarray"
    total_entries: int
    entry_to_variant: "np.ndarray"
    called_idx_pos: "np.ndarray"
    sv_idx_pos: "np.ndarray"
    caller_vrefs: "np.ndarray"

    @classmethod
    def from_bytes(
        cls, bin_u8: "np.ndarray", section_offset: int
    ) -> "_SectionLayoutView":
        """Build the layout view for the section at ``section_offset``.

        ``bin_u8`` is the writer's zero-copy ``uint8`` view of the
        already-written region. The ``uint16`` / ``uint32``
        reinterpretations alias the same buffer, so the position arrays
        index straight into ``bin_u8.view(np.uint16)`` /
        ``.view(np.uint32)`` with no copy. The duplicated bit (bit 31 of
        the header FID) is masked off so ``fid`` is the clean line number
        every identity compare expects.
        """
        bin_u16 = bin_u8.view(np.uint16)
        bin_u32 = bin_u8.view(np.uint32)

        raw_fid, n_call_targets, n_variants = struct.unpack_from(
            "<IHH", bin_u8, section_offset
        )
        fid = raw_fid & _SECTION_FID_MASK

        jump_table_offset = section_offset + SECTION_HEADER_SIZE
        call_targets_start = jump_table_offset + _padded_jump_table_bytes(
            n_variants
        )
        variants_region_start = (
            call_targets_start + n_call_targets * CALL_TARGET_ENTRY_SIZE
        )

        empty_i64 = np.empty(0, dtype=np.int64)
        if n_variants == 0:
            return cls(
                fid=fid,
                n_call_targets=n_call_targets,
                n_variants=n_variants,
                jump_table_offset=jump_table_offset,
                call_targets_start=call_targets_start,
                variants_region_start=variants_region_start,
                section_end=_align_up(variants_region_start, SECTION_ALIGNMENT),
                jump_slots=np.empty(0, dtype=np.uint16),
                total_entries=0,
                entry_to_variant=empty_i64,
                called_idx_pos=empty_i64,
                sv_idx_pos=empty_i64,
                caller_vrefs=np.empty(0, dtype=np.uint32),
            )

        # Jump-table decode — slot = (variant_byte_len - 8) >> 2 = n_calls.
        jump_slots = bin_u16[
            jump_table_offset // 2 : jump_table_offset // 2 + n_variants
        ]
        total_entries = int(jump_slots.astype(np.uint32).sum())

        # Per-call entry slot positions inside the variants region (u16
        # units, relative to variants_region_start // 2). Derivation:
        #   variant v's header u16-offset = 4*v + 2*cumsum_entries[v]
        #   entry j of variant v at u16-offset header + 4 + 2*j
        #   For global entry k where j = k - cumsum_entries[v]:
        #     called_idx_u16[k] = 4*v + 4 + 2*k   (cumsum_entries cancels)
        #     sv_idx_u16[k]     = 4*v + 5 + 2*k
        entry_to_variant = np.repeat(
            np.arange(n_variants, dtype=np.int64),
            jump_slots.astype(np.int64),
        )
        k = np.arange(total_entries, dtype=np.int64)
        sv_idx_pos = 4 * entry_to_variant + 5 + 2 * k
        called_idx_pos = sv_idx_pos - 1

        # Caller variant_ref_offsets — first u32 of each variant header,
        # in on-disk order. cumsum of per-variant entry counts gives each
        # header's u32 position.
        cumsum = np.concatenate(
            ([np.int64(0)], np.cumsum(jump_slots.astype(np.int64)))
        )
        hdr_u32_pos = 2 * np.arange(n_variants, dtype=np.int64) + cumsum[:-1]
        variants_u32 = bin_u32[variants_region_start // 4 :]
        caller_vrefs = variants_u32[hdr_u32_pos]

        variants_region_bytes = (
            n_variants * VARIANT_HEADER_SIZE
            + total_entries * PER_CALL_ENTRY_SIZE
        )
        section_end = _align_up(
            variants_region_start + variants_region_bytes, SECTION_ALIGNMENT
        )

        return cls(
            fid=fid,
            n_call_targets=n_call_targets,
            n_variants=n_variants,
            jump_table_offset=jump_table_offset,
            call_targets_start=call_targets_start,
            variants_region_start=variants_region_start,
            section_end=section_end,
            jump_slots=jump_slots,
            total_entries=total_entries,
            entry_to_variant=entry_to_variant,
            called_idx_pos=called_idx_pos,
            sv_idx_pos=sv_idx_pos,
            caller_vrefs=caller_vrefs,
        )


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
    3. Per variant — buffered into :class:`VariantBuffer` so the
       section's variant blocks land on disk in
       ``variant_ref_offset``-ascending order rather than the caller's
       declared-emit order. The reader is agnostic to this ordering;
       sibling-close back-patches re-derive variant indices from the
       on-disk variant table, so sorting at write time keeps the
       per-section variant block list scannable in
       ``variant_ref_offset`` order without changing the wire format.
        a. :meth:`begin_variant` — packs the 8-byte variant header
           (``u32 variant_ref_offset | u32 data_offset_shifted``) into
           the buffer; ``n_calls`` lives in the section's jump table,
           not the variant header.
        b. :meth:`emit_per_call_entries` — appends per-call slots to
           the buffer. Backward references read the callee section's
           jump-table layout (``_known_sections[callee_fid]``) and stamp
           the resolved variant idx directly on a vkey hit (the callee's
           on-disk variants are already sorted, so the resolved idx is
           the post-sort idx the loader will read); misses (and
           forward references) defer to a back-patch hole.
        c. :meth:`end_variant` — increments the section's variant
           count.
    4. :meth:`end_section` — flushes the variant buffer in
       ``variant_ref_offset``-ascending order, stamps the jump table
       from the sorted variants' ``n_calls``, back-patches
       ``n_variants``, pads to a 4-byte boundary, reads the
       just-closed section's jump-table layout to recover its variant
       table, resolves any self-references the just-closed section made
       into itself (the "callee_fid == my_fid" case that
       ``emit_per_call_entries`` deferred), then walks every caller
       section in ``_pending_holes[my_fid]`` and re-derives each
       caller's slot positions from its own jump table to stamp the
       call_target's ``function_section_ptr`` AND any per-call entry
       whose owning caller variant's ``variant_ref_offset`` matches a
       variant in THIS section's local table.
    5. After all sections: :meth:`finalize` — for every FID still in
       ``_pending_holes``: if the callee section was never written
       it raises (builder bug); otherwise the per-caller-section
       scan stamps :data:`MISSING_VARIANT_INDEX` on any remaining
       ``UNRESOLVED`` slot pointing at that FID (cross-arm vkey
       mismatch — one warn-log line per stamp). Then runs a
       belt-and-braces sweep for any leaked ``0xFFFF`` and closes
       the underlying memmap.

    Debug flag ``verify_holes_unfilled``: when ``True``, every
    hole-fill operation reads the target slot's current u16 BEFORE
    writing and asserts it equals :data:`UNRESOLVED_VARIANT_INDEX`
    (``0xFFFF``); on mismatch, raises ``AssertionError`` with a full
    byte-offset + value diagnostic. Use to catch double-write or
    wrong-byte-target writer bugs. Default ``False``; zero overhead
    when ``False`` (a single not-taken branch at each hole-fill site).
    """

    def __init__(
        self,
        path: Path,
        warn_log: Optional[TextIO] = None,
        *,
        verify_holes_unfilled: bool = False,
    ) -> None:
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

        # Diagnostic flag: when True, every hole-fill site reads the
        # target slot's current u16 before writing and asserts it
        # equals :data:`UNRESOLVED_VARIANT_INDEX`. Off by default;
        # gated branch at each site has zero work-cost when False.
        self.verify_holes_unfilled: bool = verify_holes_unfilled

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
        # Caller-declared ``duplicated`` marker for THIS section. Stamped
        # into bit 31 of the header ``function_name_ptr`` by
        # :meth:`emit_call_targets` (the clean FID is what
        # ``_known_sections`` / back-patch compares use).
        self._current_duplicated: bool = False
        # File offset of THIS section's jump table (first u16 slot). The
        # table is ``n_variants × u16`` immediately after the 8-byte
        # section header, so this is ``section_offset + SECTION_HEADER_SIZE``.
        self._current_jump_table_offset: Optional[int] = None
        # Specs of the section's call_targets, kept around so
        # emit_per_call_entries can validate that a PerCallEntry's
        # called_idx points at the call_target whose FID the entry
        # declares.
        self._current_call_targets: list[CallTargetSpec] = []
        # Per-section variant accumulator. Variants are buffered between
        # :meth:`begin_section` and :meth:`end_section` so the flush at
        # :meth:`end_section` can write them in
        # ``variant_ref_offset``-ascending order regardless of caller
        # emit order. ``None`` outside a section; the buffer itself
        # answers ``variant_open`` / ``n_variants`` while a section is
        # live.
        self._variant_buffer: Optional[VariantBuffer] = None

    # ------------------------------------------------------------------
    # Section lifecycle
    # ------------------------------------------------------------------

    def begin_section(
        self,
        function_name_ptr: int,
        n_variants: int,
        *,
        duplicated: bool = False,
    ) -> int:
        """Open a new section for ``function_name_ptr``.

        Aligns the cursor up to a 4-byte boundary (pads the gap with
        zero bytes), records the section's start offset in
        ``known_sections``, and returns it. Forward references to this
        FID emitted by EARLIER sections remain in ``_pending_holes`` —
        they're resolved when this section closes via
        :meth:`end_section`.

        ``duplicated`` is a generic per-section marker the writer stamps
        into bit 31 of the header ``function_name_ptr`` (see
        :data:`_SECTION_DUPLICATED_BIT`). The writer is domain-agnostic
        about its meaning; the caller (the unmatched arm) sets it for the
        same-name sibling sections of a duplicated function. The clean
        ``function_name_ptr`` is what ``_known_sections`` and the
        back-patch compares use — only the on-disk header bytes carry the
        bit, so equality with call_target FIDs is preserved.

        ``n_variants`` is the exact number of variants the caller is
        about to emit. It is used to reserve ``n_variants × u16`` bytes
        (rounded up to a multiple of 4 — see
        :func:`_padded_jump_table_bytes`) for the per-section jump table
        immediately after the section header (see
        :data:`JUMP_TABLE_ENTRY_SIZE`). The reader uses that table to
        address variant_i in O(1). Declaring a count that
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
        if function_name_ptr < 0 or function_name_ptr > _SECTION_FID_MASK:
            raise ValueError(
                f"function_name_ptr={function_name_ptr} does not fit the "
                f"31-bit section-header FID field (max {_SECTION_FID_MASK}); "
                f"bit 31 is reserved for the duplicated marker"
            )
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
        self._current_duplicated = duplicated
        # The jump table starts immediately after the section header. It is
        # written by emit_call_targets (which knows its own header_offset),
        # but the offset is deterministic so we cache it here for
        # :meth:`end_section` to stamp into once the variant buffer is
        # flushed in sorted order.
        self._current_jump_table_offset = section_offset + SECTION_HEADER_SIZE
        self._current_call_targets = []
        self._variant_buffer = VariantBuffer()
        return section_offset

    def emit_call_targets(self, call_targets: list[CallTargetSpec]) -> None:
        """Stamp the section header + jump-table reservation + call_target table.

        ``n_variants`` is stamped at ``0`` in the header and patched in
        :meth:`end_section` from the observed variant count. The jump
        table sits between the header and the call_target table; it is
        reserved with zero bytes here (rounded up to a multiple of 4 so
        the call_targets table is u32-aligned) and each entry is stamped
        by :meth:`emit_per_call_entries` for its owning variant. The
        caller is responsible for having deduplicated ``call_targets``
        by ``(function_name_ptr, type)``; SectionWriter does not check.
        """
        self._assert_section_open()
        if self._n_variants_slot is not None:
            raise ValueError("emit_call_targets called twice for the same section")

        n_call_targets = len(call_targets)
        # Section header: func_line_no | n_call_targets | n_variants (placeholder).
        # Bit 31 of the func_line_no field carries the per-section
        # ``duplicated`` marker; the clean FID is kept in ``_current_fid``
        # for every emit-time identity compare, so only these header bytes
        # see the bit. Both section parsers mask it back off.
        header_fid = self._current_fid
        if self._current_duplicated:
            header_fid |= _SECTION_DUPLICATED_BIT
        header = struct.pack(
            "<IHH",
            header_fid,
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
        # AND padded out to a multiple of 4 (see
        # :func:`_padded_jump_table_bytes`) so the call_target table that
        # follows starts on a u32 boundary — downstream vectorised
        # hole-fills reinterpret the bin as ``uint32``.
        jump_table_bytes = _padded_jump_table_bytes(
            self._current_n_variants_declared
        )
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
        """Pack the variant header into the section's variant buffer.

        The variant header is an 8-byte
        ``u32 variant_ref_offset | u32 data_offset_shifted``; the
        per-variant ``n_calls`` lives in the section's jump table and
        is stamped by :meth:`end_section` when the buffer is flushed in
        ``variant_ref_offset``-ascending order. The bytes are NOT
        written through to the underlying writer yet — they are
        accumulated in :attr:`_variant_buffer` and reordered at flush.
        """
        self._assert_section_open()
        if self._n_variants_slot is None:
            raise ValueError(
                "begin_variant called before emit_call_targets; the "
                "section header must be stamped first"
            )
        if self._variant_buffer.variant_open:
            raise ValueError(
                "begin_variant called while a previous variant is still "
                "open; call emit_per_call_entries + end_variant first"
            )
        if self._variant_buffer.n_variants >= self._current_n_variants_declared:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} declared "
                f"n_variants={self._current_n_variants_declared} at "
                f"begin_section but begin_variant was called a "
                f"{self._variant_buffer.n_variants + 1}-th time; the "
                f"jump-table reservation has no slot for this variant"
            )

        variant_header = struct.pack(
            "<II",
            variant_ref_offset,
            data_offset_shifted,
        )
        self._variant_buffer.begin_variant(variant_ref_offset, variant_header)

    def emit_per_call_entries(self, entries: list[PerCallEntry]) -> None:
        """Append the variant's per-call entries to the variant buffer.

        Backward references (``callee_fid in _known_sections``) read the
        jump-table layout of the callee section pointed at by
        ``_known_sections[callee_fid]`` — the LAST sibling closed, which
        is also the offset that ends up in the call_target's
        ``function_section_ptr`` after sibling last-write-wins. The
        section's on-disk ``variant_ref_offset`` array is scanned for the
        entry's ``callee_vkey``: a hit stamps the resolved index
        directly. The callee's on-disk variants are already
        ``variant_ref_offset``-sorted (flushed by :meth:`end_section`
        when the callee closed), so the resolved index IS the post-sort
        idx the loader will read.

        A miss — or a forward reference whose callee section has not
        been opened yet, or a self-reference whose section is still in
        flight — stamps :data:`UNRESOLVED_VARIANT_INDEX` and records
        THIS section's offset in ``_pending_holes[callee_fid]``. The
        marker is the only thing the writer needs: at the callee's
        :meth:`end_section` the writer reads the jump-table layout of
        both that section AND every caller it has marked, and re-derives
        the slot byte offsets from the bin's self-describing bytes (jump
        table + variants region). Self-references skip the marker: the slot is
        resolved by :meth:`end_section`'s "step 2" self-resolve pass
        on the just-closed section's own bytes, keeping the self-call
        path disjoint from the sibling-close path so the two never
        double-patch the same slot.

        Anything still unresolved at :meth:`finalize` (cross-arm vkey
        mismatch) gets :data:`MISSING_VARIANT_INDEX` + a warn-log
        line.

        The buffered per-call bytes are not yet on disk — they are
        flushed in sorted order by :meth:`end_section`, which also
        stamps the per-section jump table from the sorted variants'
        per-call counts.
        """
        self._assert_variant_open()

        n_calls = len(entries)
        if n_calls > 0xFFFF:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} "
                f"variant_idx={self._variant_buffer.n_variants} has "
                f"{n_calls} per-call entries; max is {0xFFFF} (u16 "
                f"jump-table slot)"
            )

        for entry in entries:
            self._assert_called_idx_matches(entry)
            if entry.resolved_section_variant_index is not None:
                # Caller-declared terminal index: stamp verbatim, open no
                # hole. The slot is known-unresolvable (e.g. a duplicated
                # callee) so the resolution machinery is skipped entirely.
                self._variant_buffer.append_per_call_entry(
                    struct.pack(
                        "<HH",
                        entry.called_idx,
                        entry.resolved_section_variant_index,
                    )
                )
                continue
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
            self._variant_buffer.append_per_call_entry(
                struct.pack("<HH", entry.called_idx, section_variant_index)
            )

    def _resolve_backward_variant_index(
        self, *, callee_fid: int, callee_vkey: Hashable
    ) -> Optional[int]:
        """Resolve a backward-reference variant idx via the callee
        section's jump-table layout (``_known_sections[callee_fid]``).

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

        The section we address is the SAME one whose offset will end
        up in the call_target row's ``function_section_ptr`` after the
        sibling last-write-wins, so the loader can never observe a
        ``(section_offset, variant_idx)`` pair that points into the
        wrong sibling's variant table.

        Reads the callee section's header + jump table directly from the
        live ``uint8`` mmap view (``_SectionLayoutView``); the on-disk
        ``variant_ref_offset`` per variant (``caller_vrefs``, in
        on-disk/ascending order) is scanned for ``callee_vkey`` and the
        FIRST matching on-disk index is returned — byte-for-byte the same
        index the old per-variant ``for`` loop returned, with no
        ``CallTarget`` / ``VariantBlock`` object materialised.
        """
        if callee_fid == self._current_fid:
            return None
        section_offset = self._known_sections.get(callee_fid)
        if section_offset is None:
            return None
        bin_u8 = self._writer.writable_u8_view()
        layout = _SectionLayoutView.from_bytes(bin_u8, section_offset)
        # callee_vkey lives in the u32 variant_ref_offset field; a value
        # outside the u32 range can never equal an on-disk vref, matching
        # the old loop's int==int compare (it would never fire either).
        if not (0 <= int(callee_vkey) <= 0xFFFFFFFF):
            return None
        hits = np.nonzero(layout.caller_vrefs == np.uint32(callee_vkey))[0]
        if hits.size == 0:
            return None
        return int(hits[0])

    def end_variant(self, vkey: Hashable) -> int:
        """Finalise the currently-open variant in the variant buffer.

        Returns the variant's 0-based **declared-emit-order** index.
        Variants are reordered by ``variant_ref_offset`` ascending at
        :meth:`end_section`-flush time, so the returned index does NOT
        in general match the variant's on-disk position; the value is
        documentary (callers that need the on-disk index recover it by
        parsing the closed section). The vkey itself was already stamped
        into the variant header's ``variant_ref_offset`` field (via
        :meth:`begin_variant`'s caller-supplied byte offset), so the
        writer does NOT need a cross-section map of
        ``(FID, vkey) → variant_idx``: :meth:`end_section` recovers it
        by parsing the just-closed section back from its own bytes.

        Multiple sections sharing a ``function_name_ptr`` (see the
        :meth:`begin_section` docstring) can emit overlapping vkeys
        — each sibling resolves only the per-call holes whose
        ``callee_vkey`` matches its own local variant table.
        """
        self._assert_variant_open()
        variant_idx = self._variant_buffer.n_variants
        if variant_idx > UNRESOLVED_VARIANT_INDEX - 1:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} has "
                f"{variant_idx + 1} variants; max is "
                f"{UNRESOLVED_VARIANT_INDEX} per section (u16 slot reserves "
                f"0xFFFF as the unresolved-hole sentinel)"
            )
        return self._variant_buffer.end_variant()

    def end_section(self) -> tuple[int, int]:
        """Close the current section.

        Flushes the variant buffer in ``variant_ref_offset``-ascending
        order (stable: equal vrefs keep their declared sub-order so the
        ``searchsorted(side="right") - 1`` last-write-wins tie-break
        still reproduces the legacy semantic). Each flushed variant
        contributes its 8-byte header + per-call entry bytes to the
        underlying writer at section-trailer-time, and its ``n_calls``
        slots into the section's pre-reserved jump table at the
        post-sort position. Then patches ``n_variants``, pads to a
        4-byte boundary, reads the just-closed section's jump-table
        layout to recover the variant table. Resolves back-patches in
        two disjoint sweeps, each addressing on-wire bytes through the
        jump-table cumsum (no writer-side slot-offset cache, no
        per-variant object materialisation):

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
          variant table is exactly the one we just read.

        * **Step 3 — sibling-close patches.** For each
          ``caller_section_offset`` in
          ``_pending_holes.get(my_fid, ())``: read that caller
          section's jump-table layout and (Case A) re-stamp every
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
        if self._variant_buffer.variant_open:
            raise ValueError(
                "end_section called while a variant is still open; "
                "call end_variant first"
            )
        n_variants = self._variant_buffer.n_variants
        if n_variants != self._current_n_variants_declared:
            raise ValueError(
                f"section for function_name_ptr={self._current_fid} declared "
                f"n_variants={self._current_n_variants_declared} at "
                f"begin_section but emitted {n_variants}; the jump-table "
                f"reservation cannot be retroactively resized because the "
                f"call_targets block sits at a fixed offset past it"
            )

        # Flush the variant buffer in ``variant_ref_offset``-ascending
        # order. Each variant contributes its header + per-call bytes
        # at the current cursor; the jump-table slot at the variant's
        # post-sort position is stamped from its ``n_calls``. The order
        # in which slots are stamped is the sort order so slot ``i`` of
        # the jump table aligns with the ``i``-th flushed variant —
        # exactly what the reader scans.
        for sort_idx, (header, per_call_bytes, n_calls) in enumerate(
            self._variant_buffer.flush_sorted()
        ):
            self._writer.write(header)
            if per_call_bytes:
                self._writer.write(per_call_bytes)
            jump_table_slot = (
                self._current_jump_table_offset
                + sort_idx * JUMP_TABLE_ENTRY_SIZE
            )
            self._writer.patch(jump_table_slot, struct.pack("<H", n_calls))

        # Patch n_variants.
        self._writer.patch(
            self._n_variants_slot,
            struct.pack("<H", n_variants),
        )
        # Align section trailer.
        self._pad_to_alignment()

        section_offset = self._current_section_offset
        section_length = self._writer.cursor - section_offset
        fid = self._current_fid

        # Recover THIS section's variant table from its own bytes —
        # each section is self-describing. Two numpy arrays drive both
        # the step-2 self-resolve pass (self-call per-call entries) and
        # step-3 sibling-close Case B (caller per-call entries):
        # ``my_sorted_vrefs`` is the section's variant_ref_offsets in
        # ascending order for a single ``np.searchsorted``;
        # ``my_sort_order`` maps each sorted position back to the
        # variant's on-wire ``section_variant_index``.
        #
        # The variant buffer flushed variants in
        # ``variant_ref_offset``-ascending order, so ``my_vrefs`` is
        # already sorted and ``argsort`` collapses to ``arange(n)``.
        # The two-array dance is preserved verbatim because (a) it is
        # the same shape ``_resolve_caller_section`` expects from the
        # step-3 sibling-close path (where caller order is unrelated
        # to callee order) and (b) the ``kind="stable"`` argsort
        # together with the buffer's stable flush keeps the
        # ``searchsorted(side="right") - 1`` last-write-wins tie-break
        # consistent across both step-2 and step-3 paths when a
        # section legitimately repeats the same ``variant_ref_offset``.
        #
        # ``my_vrefs`` is read straight off the just-flushed jump-table
        # layout (``caller_vrefs``, on-disk order) — no per-variant
        # object materialisation. The buffer wrote variants already
        # sorted, so this is identical to the ascending vref sequence the
        # old ``parse_section_bin`` loop produced.
        bin_u8 = self._writer.writable_u8_view()
        my_vrefs = np.ascontiguousarray(
            _SectionLayoutView.from_bytes(bin_u8, section_offset).caller_vrefs
        )
        my_sort_order = np.argsort(my_vrefs, kind="stable").astype(np.int64)
        my_sorted_vrefs = my_vrefs[my_sort_order]

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
            callee_sorted_vrefs=my_sorted_vrefs,
            callee_sort_order=my_sort_order,
            context="Step2-self-resolve",
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
                callee_sorted_vrefs=my_sorted_vrefs,
                callee_sort_order=my_sort_order,
                context="Step3-sibling-close",
            )

        # Clear per-section state.
        self._current_fid = None
        self._current_section_offset = None
        self._n_variants_slot = None
        self._current_n_variants_declared = None
        self._current_duplicated = False
        self._current_jump_table_offset = None
        self._current_call_targets = []
        self._variant_buffer = None

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
        vkey sets). For each such caller, read the caller's jump-table
        layout, find every per-call slot pointing at the FID that is
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

        For every other FID, read each caller section's jump-table
        layout and stamp :data:`MISSING_VARIANT_INDEX` on each per-call
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
        """Stamp MISSING on every per-call slot in one caller section
        still pointing at ``callee_fid`` as unresolved.

        Slot positions are re-derived from the caller's own jump-table
        layout (``_SectionLayoutView``) read straight off the live mmap:
        no per-variant object materialisation, no writer-side
        slot-offset cache. The matching per-call slots are visited in
        global entry order (variant ascending, entry ascending), so the
        MISSING stamps AND the one-warn-log-line-per-stamp output land
        byte-for-byte in the same order the old per-variant /
        per-entry ``for`` loop produced.
        """
        bin_u8 = self._writer.writable_u8_view()
        bin_u16 = bin_u8.view(np.uint16)
        bin_u32 = bin_u8.view(np.uint32)
        layout = _SectionLayoutView.from_bytes(bin_u8, caller_section_offset)
        if layout.n_call_targets == 0 or layout.total_entries == 0:
            return

        # call_target name_ptr column — first u32 of each 3*u32 row.
        ct_name_ptr = bin_u32[
            layout.call_targets_start // 4 : layout.variants_region_start // 4
        ].reshape(layout.n_call_targets, 3)[:, 0]
        target_called_idxs = np.nonzero(
            ct_name_ptr == np.uint32(callee_fid)
        )[0].astype(np.uint16)
        if target_called_idxs.size == 0:
            return

        variants_u16 = bin_u16[layout.variants_region_start // 2 :]
        hole_mask = (
            np.isin(variants_u16[layout.called_idx_pos], target_called_idxs)
            & (
                variants_u16[layout.sv_idx_pos]
                == np.uint16(UNRESOLVED_VARIANT_INDEX)
            )
        )
        hole_entries = np.nonzero(hole_mask)[0]
        if hole_entries.size == 0:
            return

        target_slots = layout.sv_idx_pos[hole_entries]
        if self.verify_holes_unfilled:
            self._assert_slots_unresolved_vec(
                variants_u16,
                target_slots,
                np.full(target_slots.shape, MISSING_VARIANT_INDEX, dtype=np.uint16),
                context="finalize-MISSING",
                caller_section_offset=caller_section_offset,
                callee_fid=callee_fid,
                variants_region_start=layout.variants_region_start,
            )
        variants_u16[target_slots] = np.uint16(MISSING_VARIANT_INDEX)

        if self._warn_log is not None:
            # One line per stamp, in global entry order — owning variant's
            # variant_ref_offset cast to a plain int so its ``!r`` repr
            # matches the struct-derived int the old loop logged.
            hole_vrefs = layout.caller_vrefs[
                layout.entry_to_variant[hole_entries]
            ]
            for vref in hole_vrefs:
                self._warn_log.write(
                    f"missing_variant: callee_fid={callee_fid} "
                    f"callee_vkey={int(vref)!r} "
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
        if not self._variant_buffer.variant_open:
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

    def _assert_slots_unresolved_vec(
        self, variants_u16, slot_positions, intended_vec, *,
        context, caller_section_offset, callee_fid, variants_region_start,
    ) -> None:
        """Vectorised pre-write check: every ``slot_positions`` entry
        (u16 index into ``variants_u16``) must hold
        :data:`UNRESOLVED_VARIANT_INDEX`. Reads u16, compares to
        ``0xFFFF``; diagnostic context passed in.
        """
        actual = variants_u16[slot_positions]
        bad = actual != np.uint16(UNRESOLVED_VARIANT_INDEX)
        if not bad.any():
            return
        i = int(np.nonzero(bad)[0][0])
        raise AssertionError(
            f"SectionWriter.verify_holes_unfilled[{context}]: caller_section_offset="
            f"{caller_section_offset} callee_fid={callee_fid} byte_offset="
            f"{variants_region_start + int(slot_positions[i]) * 2} current=0x"
            f"{int(actual[i]):04x} intended=0x{int(intended_vec[i]):04x} "
            f"bad_total={int(bad.sum())}"
        )

    def _assert_slot_unresolved(
        self, byte_offset: int, intended_value: int, *,
        context: str, caller_section_offset: int, callee_fid: int,
    ) -> None:
        """Scalar pre-write check for one ``_writer.patch`` u16."""
        blob = self._writer.view()
        try:
            actual = int.from_bytes(
                bytes(blob[byte_offset : byte_offset + 2]), "little"
            )
        finally:
            blob.release()
        if actual == UNRESOLVED_VARIANT_INDEX:
            return
        raise AssertionError(
            f"SectionWriter.verify_holes_unfilled[{context}]: caller_section_offset="
            f"{caller_section_offset} callee_fid={callee_fid} byte_offset="
            f"{byte_offset} current=0x{actual:04x} intended=0x{intended_value:04x}"
        )

    def _resolve_caller_section(
        self,
        *,
        caller_section_offset: int,
        callee_fid: int,
        callee_section_offset: int,
        callee_sorted_vrefs: "np.ndarray",
        callee_sort_order: "np.ndarray",
        context: str = "Step3-sibling-close",
    ) -> None:
        """Re-derive caller's slot positions; write Cases A + B in place
        through the live mmap.

        Each dtype gets exactly one numpy view of the writer's bytes
        (uint8 / uint16 / uint32 of the same buffer); slot positions
        inside the variants region are derived from the section's own
        jump table by ``(slot << 2) + 8 → byte length → cumsum → byte
        offsets``.

        * Case A (``function_section_ptr``): every call_target row
          whose ``name_ptr == callee_fid`` gets its ``section_ptr``
          column stamped (last-write-wins on sibling collisions).

        * Case B (``section_variant_index``): every per-call entry
          whose ``called_idx`` points at one of those rows AND whose
          slot is still :data:`UNRESOLVED_VARIANT_INDEX` gets
          resolved against the caller variant's
          ``variant_ref_offset`` (== intended callee_vkey by Step 7's
          on-wire invariant) via ``np.searchsorted`` on the
          pre-built sort index. First-resolver-wins; misses leave
          the slot ``UNRESOLVED`` for a sibling close or
          :meth:`finalize` to mark MISSING.

        Used by both the step-2 self-resolve pass
        (``caller_section_offset == callee_section_offset``) and the
        step-3 sibling-close pass. The two paths are disjoint by
        ``emit_per_call_entries``'s rule that self-call slots never
        appear in ``_pending_holes[my_fid]``, so the same call_target
        + per-call slot is never patched twice.
        """
        bin_u8 = self._writer.writable_u8_view()
        bin_u16 = bin_u8.view(np.uint16)
        bin_u32 = bin_u8.view(np.uint32)

        # Shared jump-table layout addressing (header + cumsum-derived
        # slot positions) — same helper the self-resolve / backward-ref /
        # finalize paths use, so the layout arithmetic lives in exactly
        # one place.
        layout = _SectionLayoutView.from_bytes(bin_u8, caller_section_offset)
        if layout.n_call_targets == 0:
            return

        # Case A: stamp function_section_ptr. 3*u32 view of call_targets
        # (name_ptr | section_ptr | flags_packed).
        ct = bin_u32[
            layout.call_targets_start // 4 : layout.variants_region_start // 4
        ].reshape(layout.n_call_targets, 3)
        ct_mask = ct[:, 0] == np.uint32(callee_fid)
        if not ct_mask.any():
            return
        ct[ct_mask, 1] = callee_section_offset

        if layout.n_variants == 0:
            return
        if layout.total_entries == 0:
            return

        entry_to_variant = layout.entry_to_variant
        sv_idx_pos = layout.sv_idx_pos
        called_idx_pos = layout.called_idx_pos
        variants_region_start = layout.variants_region_start

        variants_u16 = bin_u16[variants_region_start // 2 :]

        target_called_idxs = np.nonzero(ct_mask)[0].astype(np.uint16)
        hole_mask = (
            np.isin(variants_u16[called_idx_pos], target_called_idxs)
            & (variants_u16[sv_idx_pos] == np.uint16(UNRESOLVED_VARIANT_INDEX))
        )
        if not hole_mask.any():
            return

        # Caller variant_ref_offsets — first u32 of each variant header.
        caller_vrefs = layout.caller_vrefs

        # Resolve via pre-built sort index — single searchsorted, no
        # Python dict. ``side="right"`` paired with ``- 1`` picks the
        # LAST equal entry (the callee's argsort is stable, so this
        # reproduces the legacy ``vkey_to_idx`` dict's last-write-wins
        # tie-break on a repeated ``variant_ref_offset``).
        hole_indices = np.nonzero(hole_mask)[0]
        hole_vrefs = caller_vrefs[entry_to_variant[hole_indices]]
        ss = np.searchsorted(callee_sorted_vrefs, hole_vrefs, side="right") - 1
        in_bounds = ss >= 0
        matches = np.zeros_like(hole_indices, dtype=bool)
        matches[in_bounds] = (
            callee_sorted_vrefs[ss[in_bounds]] == hole_vrefs[in_bounds]
        )
        if not matches.any():
            return

        # In-place writes through the live mmap.
        target_slots = sv_idx_pos[hole_indices[matches]]
        target_values = callee_sort_order[ss[matches]].astype(np.uint16)
        if self.verify_holes_unfilled:
            self._assert_slots_unresolved_vec(
                variants_u16,
                target_slots,
                target_values,
                context=context,
                caller_section_offset=caller_section_offset,
                callee_fid=callee_fid,
                variants_region_start=variants_region_start,
            )
        variants_u16[target_slots] = target_values

    def _sweep_for_unresolved_sentinels(self) -> None:
        """Walk the bin sections; assert no ``0xFFFF`` slot leaked.

        Jump-table-native, parse-free: each section's
        :class:`_SectionLayoutView` gives the exact ``section_variant_index``
        u16 slot positions and the section's end offset (to advance to the
        next section), so the sweep reads only the per-call ``sv_idx``
        slots — no ``CallTarget`` / ``VariantBlock`` object is
        materialised. A single linear pass over the bin.

        Reads through the writer's zero-copy ``uint8`` mmap view — at
        corpus scale the bin is multi-GB and a ``bytes`` copy at
        finalize-time would more than double peak RAM. The reported
        ``variant_idx`` / ``called_idx`` of the first leak are recovered
        from the layout's ``entry_to_variant`` and the on-disk
        ``called_idx`` slot, byte-identical to the old per-entry walk's
        diagnostic.
        """
        # Collect the first leak's diagnostic fields (if any) inside a
        # nested scope so EVERY numpy view aliasing the mmap (the
        # ``uint8`` base, its ``uint16`` reinterpretation, per-section
        # slices, and the layout view's live ``jump_slots`` slice) is
        # dropped when the scan returns. Raising from a frame that still
        # holds such a view would leave an exported pointer alive in the
        # traceback and block :meth:`MemmapBinWriter.finalize`'s
        # ``mmap.close`` with "cannot close exported pointers", so the
        # raise is hoisted out, after every view is out of scope. The
        # scan returns only plain ``int`` diagnostics — no mmap alias
        # escapes it.
        leak_diag = self._first_unresolved_slot()
        if leak_diag is not None:
            fid, v_idx, called_idx = leak_diag
            raise ValueError(
                f"unresolved section_variant_index in section "
                f"function_name_ptr={fid} "
                f"variant_idx={v_idx} called_idx={called_idx}"
            )

    def _first_unresolved_slot(self) -> Optional[tuple[int, int, int]]:
        """Return ``(fid, variant_idx, called_idx)`` of the first
        ``UNRESOLVED`` per-call slot in the bin, or ``None`` if clean.

        All numpy views aliasing the writer's mmap live only in this
        frame; returning plain ints lets the caller raise without any
        exported pointer outliving the scan (so the finalize
        ``mmap.close`` cannot trip on a live export).
        """
        bin_u8 = self._writer.writable_u8_view()
        bin_u16 = bin_u8.view(np.uint16)
        end = bin_u8.shape[0]
        offset = MATCHED_SECTIONS_BIN_PRELUDE_SIZE
        while offset < end:
            layout = _SectionLayoutView.from_bytes(bin_u8, offset)
            if layout.total_entries:
                variants_u16 = bin_u16[layout.variants_region_start // 2 :]
                leak = np.nonzero(
                    variants_u16[layout.sv_idx_pos]
                    == np.uint16(UNRESOLVED_VARIANT_INDEX)
                )[0]
                if leak.size:
                    k = int(leak[0])
                    return (
                        layout.fid,
                        int(layout.entry_to_variant[k]),
                        int(variants_u16[layout.called_idx_pos[k]]),
                    )
            offset = layout.section_end
        return None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _variant_block_offset(section: Section, variant_idx: int) -> int:
    """Return the file-offset of variant ``variant_idx``'s header.

    Derives the position directly from the section layout (section
    header + jump table + call_targets table + sum of preceding
    variant block sizes). This is the simple O(variant_idx) scalar
    addressing; the writer's back-patch paths instead use the
    cumsum-vectorised :class:`_SectionLayoutView` (which is O(1) per
    slot). This helper survives as the test-side reference oracle's
    independent addressing path, used to cross-check that the
    vectorised writer lands every slot on the same byte. The caller is
    responsible for ensuring ``variant_idx`` is a valid index into
    ``section.variants``.
    """
    variants_region_start = (
        section.section_offset
        + SECTION_HEADER_SIZE
        + _padded_jump_table_bytes(len(section.variants))
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
       The region is padded up to a multiple of 4 (one trailing zero
       ``u16`` when ``n_variants`` is odd) so the call_targets table
       that follows is u32-aligned for vectorised downstream readers.
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
        raw_function_name_ptr,
        n_call_targets,
        n_variants,
    ) = struct.unpack_from("<IHH", blob, offset)
    # Bit 31 of the header FID is the per-section duplicated marker; mask
    # it off so ``function_name_ptr`` is the clean line number every FID
    # consumer expects, and surface the bit separately.
    is_duplicated = bool(raw_function_name_ptr & _SECTION_DUPLICATED_BIT)
    function_name_ptr = raw_function_name_ptr & _SECTION_FID_MASK
    offset += SECTION_HEADER_SIZE

    jump_table: list[int] = list(
        struct.unpack_from(f"<{n_variants}H", blob, offset)
    )
    # Skip the table AND the trailing u16 of zero padding that the
    # writer reserves when n_variants is odd, so the call_targets table
    # that follows is u32-aligned.
    offset += _padded_jump_table_bytes(n_variants)

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
        is_duplicated=is_duplicated,
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
