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
    MAX_HEADER_BYTES,
    parse_binary_header,
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
    """

    writer: MemmapBinWriter
    primary: HashMapU64U32 = field(default_factory=HashMapU64U32)
    collision: Dict[int, List[int]] = field(default_factory=dict)


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

    if content_hash not in state.primary:
        offset = state.writer.write(record_bytes)
        state.primary.set(content_hash, offset >> _OFFSET_SHIFT)
        return (offset, new_len)

    cached_shifted = state.primary.get(content_hash)
    if cached_shifted != _PRIMARY_SENTINEL:
        cached_offset = cached_shifted << _OFFSET_SHIFT
        existing = _read_existing_record(state.writer, cached_offset)
        if existing == record_bytes:
            return (cached_offset, new_len)
        # Genuine hash collision — promote this hash to collision mode.
        # The cached entry's shifted form goes into the collision list
        # first so its order in the list matches the order records were
        # written to disk; downstream readers don't depend on this
        # ordering but it keeps debug traces sane.
        state.collision[content_hash] = [cached_shifted]
        offset = state.writer.write(record_bytes)
        state.collision[content_hash].append(offset >> _OFFSET_SHIFT)
        state.primary.set(content_hash, _PRIMARY_SENTINEL)
        return (offset, new_len)

    # Primary said "collision mode"; walk the per-hash list.
    candidates = state.collision[content_hash]
    for candidate_shifted in candidates:
        candidate_offset = candidate_shifted << _OFFSET_SHIFT
        existing = _read_existing_record(state.writer, candidate_offset)
        if existing == record_bytes:
            return (candidate_offset, new_len)
    # No match in the collision list either; new record.
    offset = state.writer.write(record_bytes)
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


def open_arm_dedup_state(path) -> ArmDedupState:
    """Construct an :class:`ArmDedupState` with a fresh memmap'd bin.

    Stamps the file-level prelude (16 bytes) into the bin so SENTINEL=0
    is safe at the dedup-map level. The caller owns the returned
    state's writer lifecycle — call ``state.writer.finalize()`` when
    done (or stash it in an ``ExitStack`` callback).
    """
    from tokenizer.aligned_data.memmap_format import encode_data_bin_prelude

    writer = MemmapBinWriter(path, prelude_bytes=encode_data_bin_prelude())
    return ArmDedupState(writer=writer)
