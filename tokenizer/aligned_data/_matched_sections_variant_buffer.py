"""Per-section accumulator for variant blocks emitted by :class:`SectionWriter`.

Single concern: hold each section's per-variant bytes (8-byte header +
per-call entries) between :meth:`SectionWriter.begin_section` and
:meth:`SectionWriter.end_section` so the variants can be flushed to
disk in ``variant_ref_offset``-ascending order rather than in the
caller's declared-emit order.

The writer itself owns the structural concerns (section header layout,
jump-table sizing, sibling/back-patch resolution); this module knows
nothing about call_targets, ``_known_sections``, or ``_pending_holes``.
It only sees:

* the variant header bytes (already packed by the writer),
* the per-call entry bytes (already packed, already sv_idx-resolved
  where possible) appended one variant at a time,
* and the variant_ref_offset that determines the flush order.

On :meth:`flush_sorted`, the buffer yields one
``(header_bytes, per_call_bytes, n_calls)`` tuple per variant in
``variant_ref_offset``-ascending order (stable: variants with equal
``variant_ref_offset`` keep their declared sub-order, so the writer's
``searchsorted(side="right") - 1`` last-write-wins tie-break still
reproduces the legacy semantic on a repeated vkey).
"""

from __future__ import annotations

from typing import Iterator


class VariantBuffer:
    """Hold one section's variant blocks until :meth:`flush_sorted`.

    The writer drives this buffer through three calls per variant:

    1. :meth:`begin_variant` — stage the 8-byte variant header AND the
       sort key (``variant_ref_offset``). Calling this while a variant
       is already open is rejected — the writer asserts that lifecycle
       at a higher level.
    2. :meth:`append_per_call_entry` — append one 4-byte per-call entry.
       Called once per per-call slot.
    3. :meth:`end_variant` — close the currently-open variant. Returns
       its 0-based declared-emit-order index (used by
       :meth:`SectionWriter.end_variant`'s public API; documentary only).

    The buffer enforces a strict ``begin_variant`` → ``append`` (zero or
    more) → ``end_variant`` cycle; calling out of order raises.
    """

    def __init__(self) -> None:
        # Parallel arrays indexed by declared-emit order. Each variant
        # contributes one entry to all four lists at :meth:`end_variant`
        # time. ``_open_*`` fields hold the in-flight variant's state
        # between :meth:`begin_variant` and :meth:`end_variant`.
        self._sort_keys: list[int] = []
        self._headers: list[bytes] = []
        self._per_call_bytes: list[bytes] = []
        self._per_call_counts: list[int] = []

        self._open_sort_key: int | None = None
        self._open_header: bytes | None = None
        self._open_per_call_chunks: list[bytes] = []
        self._open_per_call_count: int = 0

    # ------------------------------------------------------------------
    # Per-variant lifecycle
    # ------------------------------------------------------------------

    def begin_variant(self, variant_ref_offset: int, header_bytes: bytes) -> None:
        """Open a new variant. Header is already packed by the caller."""
        if self._open_header is not None:
            raise ValueError(
                "VariantBuffer.begin_variant called while a previous "
                "variant is still open"
            )
        self._open_sort_key = variant_ref_offset
        self._open_header = header_bytes
        self._open_per_call_chunks = []
        self._open_per_call_count = 0

    def append_per_call_entry(self, entry_bytes: bytes) -> None:
        """Append one per-call entry's packed bytes to the open variant."""
        if self._open_header is None:
            raise ValueError(
                "VariantBuffer.append_per_call_entry called without an "
                "open variant"
            )
        self._open_per_call_chunks.append(entry_bytes)
        self._open_per_call_count += 1

    def end_variant(self) -> int:
        """Close the currently-open variant; return its declared-order index."""
        if self._open_header is None:
            raise ValueError(
                "VariantBuffer.end_variant called without an open variant"
            )
        declared_idx = len(self._sort_keys)
        self._sort_keys.append(self._open_sort_key)
        self._headers.append(self._open_header)
        self._per_call_bytes.append(b"".join(self._open_per_call_chunks))
        self._per_call_counts.append(self._open_per_call_count)
        self._open_sort_key = None
        self._open_header = None
        self._open_per_call_chunks = []
        self._open_per_call_count = 0
        return declared_idx

    # ------------------------------------------------------------------
    # Section flush
    # ------------------------------------------------------------------

    @property
    def n_variants(self) -> int:
        """Number of variants ``end_variant``-d so far in this section."""
        return len(self._sort_keys)

    @property
    def variant_open(self) -> bool:
        """Whether a :meth:`begin_variant` is currently unmatched."""
        return self._open_header is not None

    def flush_sorted(self) -> Iterator[tuple[bytes, bytes, int]]:
        """Yield ``(header_bytes, per_call_bytes, n_calls)`` in sorted order.

        Sort key is ``variant_ref_offset`` ascending; ties are broken
        by declared-emit order (stable sort). The writer's sibling-close
        resolver relies on the stable tie-break to reproduce its
        ``searchsorted(side="right") - 1`` last-write-wins semantic on
        a repeated vkey.

        Counting per-call entries from ``len(per_call_bytes) //
        PER_CALL_ENTRY_SIZE`` would couple this module to the wire
        format; ``n_calls`` is tracked at append time and returned
        explicitly so the buffer stays format-agnostic — the writer
        slots it into the jump table.
        """
        # ``sorted`` on a (sort_key, declared_idx) tuple is stable by
        # declared_idx because declared_idx is the second tuple field.
        # Returning a generator keeps memory bounded — the writer
        # consumes one variant at a time during flush.
        order = sorted(
            range(len(self._sort_keys)),
            key=lambda i: (self._sort_keys[i], i),
        )
        for i in order:
            yield self._headers[i], self._per_call_bytes[i], self._per_call_counts[i]
