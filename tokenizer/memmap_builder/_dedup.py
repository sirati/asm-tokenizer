"""Content-addressed dedup for ``_data.bin`` records.

One :class:`ArmDedupState` per arm (matched, unmatched) owns the bin
writer and the two-tier hashmap that backs the dedup:

* **primary** (Rust-side ``HashMapU64U32`` from the
  ``dedup_hashmap`` wheel): ``content_hash -> offset >> 4``. Hot path.
* **collision** (Python ``dict[int, list[int]]``): only populated when
  two distinct record bodies share the same 64-bit hash. Cold path —
  the birthday bound on xxh3-64 keeps this map at low tens of entries
  per arm at corpus scale.

``SENTINEL = 0`` in the primary marks "this hash now has multiple
candidates, consult the collision map". The bin's 16-byte file-level
prelude (see :mod:`tokenizer.aligned_data.memmap_format`) ensures
offset 0 is never a real record, so the sentinel value can never alias
a legitimate primary entry.

The single public helper :func:`dedup_and_write` runs the algorithm
the plan specifies: primary miss -> write + remember; primary hit
non-sentinel -> readback + byte-compare -> emit-or-promote-to-collision;
primary sentinel -> walk collision list, emit on match or append.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dedup_hashmap import HashMapU64U32

from tokenizer.aligned_data.binary_format import (
    ENTRY_IDX_SIZE,
    MAX_HEADER_BYTES,
    parse_binary_header,
    prefix_bytes_for_header,
    record_total_size,
)
from tokenizer.aligned_data.memmap_writer import MemmapBinWriter

# Sentinel value in the primary map. Offset 0 is reserved by the bin's
# file-level prelude (16 bytes, see ``memmap_format``), so no legitimate
# shifted-offset can equal 0. The sentinel can therefore be a plain
# integer compared via ``==`` rather than a special class instance.
_PRIMARY_SENTINEL: int = 0

# Records on disk are 16-byte aligned; the primary map stores
# ``offset >> 4`` so an offset up to 64 GiB fits in u32.
_OFFSET_SHIFT: int = 4


@dataclass
class ArmDedupState:
    """Per-arm dedup state: writer + primary + collision maps.

    Constructed once per ``build_memmap_files`` arm; lifetime spans the
    whole pass-1 walk of that arm. ``finalize`` on the writer is the
    caller's responsibility (orchestrated via ``contextlib.ExitStack``
    in ``builder.py``).

    ``_n_entries_emitted`` is the encounter-order counter feeding
    each record's on-wire ``entry_idx`` field. Only the actual-write
    branches of :func:`dedup_and_write` increment it (a dedup hit
    re-uses the existing record's idx); the writer's per-arm trailer
    consumes the final count via :attr:`n_entries_emitted`.
    """

    writer: MemmapBinWriter
    primary: HashMapU64U32 = field(default_factory=HashMapU64U32)
    collision: Dict[int, List[int]] = field(default_factory=dict)
    _n_entries_emitted: int = 0

    @property
    def n_entries_emitted(self) -> int:
        """How many records this arm has appended to its ``_data.bin``.

        Equals the next pending record's ``entry_idx`` (e.g. 0 before
        the first write). At ``finalize`` time this is the file's
        ``total_entries`` trailer value.
        """
        return self._n_entries_emitted


def dedup_and_write(
    state: ArmDedupState,
    record_bytes: bytes,
    content_hash: int,
) -> Tuple[int, int]:
    """Run the dedup algorithm; return ``(offset, total_record_bytes)``.

    On a true content match (either primary hit confirmed by readback,
    or collision-list hit confirmed by readback) the existing record's
    offset is returned and no new bytes are written. Otherwise the new
    bytes are written and the offset of the fresh record is returned.

    ``record_bytes`` is the fully-assembled record (header + body +
    padding), as produced by
    :func:`tokenizer.aligned_data._writers.assemble_function_record`.
    ``content_hash`` is the 64-bit xxh3 of the per-function body bytes
    (insn || block || tokens), computed once in the per-CSV iterator.
    """
    new_len = len(record_bytes)
    # Header geometry of the candidate record. The idx-mask slice
    # below is constant per record (it only depends on the header
    # form's prefix layout), so we precompute it here once.
    candidate_header, _ = parse_binary_header(record_bytes[:MAX_HEADER_BYTES])
    idx_slice = _entry_idx_slice(candidate_header)

    if content_hash not in state.primary:
        offset = state.writer.write(record_bytes)
        state._n_entries_emitted += 1
        state.primary.set(content_hash, offset >> _OFFSET_SHIFT)
        return (offset, new_len)

    cached_shifted = state.primary.get(content_hash)
    if cached_shifted != _PRIMARY_SENTINEL:
        cached_offset = cached_shifted << _OFFSET_SHIFT
        existing = _read_existing_record(state.writer, cached_offset)
        if _bytes_equal_modulo_idx(existing, record_bytes, idx_slice):
            return (cached_offset, new_len)
        # Genuine hash collision — promote this hash to collision mode.
        # The cached entry's shifted form goes into the collision list
        # first so its order in the list matches the order records were
        # written to disk; downstream readers don't depend on this
        # ordering but it keeps debug traces sane.
        state.collision[content_hash] = [cached_shifted]
        offset = state.writer.write(record_bytes)
        state._n_entries_emitted += 1
        state.collision[content_hash].append(offset >> _OFFSET_SHIFT)
        state.primary.set(content_hash, _PRIMARY_SENTINEL)
        return (offset, new_len)

    # Primary said "collision mode"; walk the per-hash list.
    candidates = state.collision[content_hash]
    for candidate_shifted in candidates:
        candidate_offset = candidate_shifted << _OFFSET_SHIFT
        existing = _read_existing_record(state.writer, candidate_offset)
        if _bytes_equal_modulo_idx(existing, record_bytes, idx_slice):
            return (candidate_offset, new_len)
    # No match in the collision list either; new record.
    offset = state.writer.write(record_bytes)
    state._n_entries_emitted += 1
    candidates.append(offset >> _OFFSET_SHIFT)
    return (offset, new_len)


def _read_existing_record(writer: MemmapBinWriter, offset: int) -> bytes:
    """Return the full bytes of the existing record starting at ``offset``.

    Uses the self-describing record header to learn the record's length
    before reading the body, so byte-compare against a candidate of
    different geometry is exact rather than potentially overlapping into
    the next record.
    """
    prefix = writer.read(offset, MAX_HEADER_BYTES)
    header, _prefix_bytes = parse_binary_header(prefix)
    total = record_total_size(header)
    return writer.read(offset, total)


def _entry_idx_slice(header) -> tuple:
    """Return ``(start, end)`` byte-offsets of ``entry_idx`` within a record.

    ``entry_idx`` sits at the tail of the header (last
    :data:`ENTRY_IDX_SIZE` bytes), regardless of whether the header
    used the ULTRASHORT or NORMAL form. The dedup byte-compare must
    mask this slice so two records whose body bytes are identical but
    whose tentative-vs-existing ``entry_idx`` differ still hit the
    dedup. ``prefix_bytes_for_header(header)`` is the single source
    of truth for the header's on-disk width.
    """
    prefix = prefix_bytes_for_header(header)
    return (prefix - ENTRY_IDX_SIZE, prefix)


def _bytes_equal_modulo_idx(
    existing: bytes, candidate: bytes, idx_slice: tuple
) -> bool:
    """Compare two record-byte buffers ignoring the ``entry_idx`` field.

    ``idx_slice`` is the ``(start, end)`` tuple from
    :func:`_entry_idx_slice` for the candidate header (the candidate's
    header form drives the slice; the existing record by construction
    has the same field layout for the same body content, since the
    encoder's form choice is body-only-driven).
    """
    if len(existing) != len(candidate):
        return False
    start, end = idx_slice
    return existing[:start] == candidate[:start] and existing[end:] == candidate[end:]


def open_arm_dedup_state(path) -> ArmDedupState:
    """Construct an :class:`ArmDedupState` with a fresh memmap'd bin.

    Stamps the file-level prelude (16 bytes) into the bin so SENTINEL=0
    is safe at the dedup-map level. The caller owns the returned
    state's writer lifecycle — call :func:`finalize_arm_dedup_state`
    when done (or stash it in an ``ExitStack`` callback) so the
    ``total_entries`` trailer is stamped before the writer closes.
    """
    from tokenizer.aligned_data.memmap_format import encode_data_bin_prelude

    writer = MemmapBinWriter(path, prelude_bytes=encode_data_bin_prelude())
    return ArmDedupState(writer=writer)


def finalize_arm_dedup_state(state: ArmDedupState) -> None:
    """Stamp the ``total_entries`` trailer + finalize the underlying writer.

    Single chokepoint that owns the ``_data.bin``-specific trailer:
    the writer's generic ``finalize`` only flushes + closes; this
    wrapper appends the u32-aligned ``total_entries`` trailer
    (geometry owned by :func:`encode_data_bin_trailer`) first. Re-runs
    safely thanks to the writer's idempotent ``finalize`` (a second
    call would no-op the trailer append because the fd is closed).
    """
    from tokenizer.aligned_data.memmap_format import encode_data_bin_trailer

    writer = state.writer
    # ``finalize`` is idempotent; guard the trailer write so a double
    # call doesn't append a second trailer to a stale (closed) writer.
    if writer.is_finalized:
        return
    trailer = encode_data_bin_trailer(
        state.n_entries_emitted, cursor=writer.cursor
    )
    writer.write(trailer)
    writer.finalize()
