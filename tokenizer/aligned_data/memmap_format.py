"""Single source of truth for the memmap output-chain format version.

This version covers every artifact produced by the memmap-output chain:
the unified vocab CSV, per-binary sections CSV preludes, the slim variants
CSV prelude, and the ``_index.bin`` prelude. Bumping the constant here is
intended to cascade: writers import it for the version they stamp, readers
import it for the version they assert, and a bump is a one-line edit plus a
full cascade-rebuild migration of any persisted artifacts.

Also owns the 16-byte file-level prelude stamped on every
``<binary>_data.bin`` / ``<binary>_unmatched_data.bin`` written by the
memmap builder. The prelude reserves bytes 0..15 of every bin file so
that record offset 0 is never legal — a property the content-addressed
dedup primary map relies on (SENTINEL = 0 means "this hash has multiple
candidates, consult the collision map"). The first record always starts
at offset 16, preserving the existing 16-byte record alignment.

The same 16-byte prelude shape (magic + version + zero-pad) is also
stamped on ``<binary>_sections.bin`` (magic ``b"MSEC"``) — the per-bin
section codec for the matched/unmatched section catalog. The generic
``encode_bin_prelude`` / ``assert_bin_prelude`` helpers below own the
shared layout; magic-specific wrappers (``encode_data_bin_prelude``,
``encode_matched_sections_prelude``, …) are thin parameterisations so
no caller hand-rolls the magic+version dance.
"""

import struct

MEMMAP_FORMAT_VERSION: int = 1

# Shared file-level prelude geometry: every memmap bin starts with a
# 4-byte magic + ``u32`` ``MEMMAP_FORMAT_VERSION`` (little-endian) +
# 8 zero bytes of reserved space (16 bytes total). The magic differs
# per bin type; everything else is identical.
_PRELUDE_MAGIC_SIZE: int = 4
_PRELUDE_VERSION_SIZE: int = 4
_PRELUDE_RESERVED_SIZE: int = 8
_PRELUDE_SIZE: int = (
    _PRELUDE_MAGIC_SIZE + _PRELUDE_VERSION_SIZE + _PRELUDE_RESERVED_SIZE
)


def encode_bin_prelude(
    magic: bytes, reserved: bytes = b"\x00" * _PRELUDE_RESERVED_SIZE
) -> bytes:
    """Return the 16-byte file-level prelude for a bin with the given magic.

    ``magic`` must be exactly 4 bytes. The prelude is the little-endian
    ``u32`` ``MEMMAP_FORMAT_VERSION`` followed by ``reserved`` (exactly 8
    bytes). ``reserved`` defaults to all-zero (the historic shape); callers
    that carry an identity fingerprint (e.g. the data-bin's
    vocab-fingerprint, :func:`encode_data_bin_prelude`) pass it here. All-
    zero reserved means "no fingerprint stamped" — readers treat it as a
    soft no-op, preserving forward/backward compatibility with bins written
    before the fingerprint existed.
    """
    if len(magic) != _PRELUDE_MAGIC_SIZE:
        raise ValueError(
            f"prelude magic must be {_PRELUDE_MAGIC_SIZE} bytes, "
            f"got {len(magic)} ({magic!r})"
        )
    if len(reserved) != _PRELUDE_RESERVED_SIZE:
        raise ValueError(
            f"prelude reserved must be {_PRELUDE_RESERVED_SIZE} bytes, "
            f"got {len(reserved)}"
        )
    return bytes(magic) + struct.pack("<I", MEMMAP_FORMAT_VERSION) + bytes(reserved)


def read_bin_prelude_reserved(prelude: bytes) -> bytes:
    """Return the 8 reserved bytes of a bin prelude (the fingerprint slot).

    All-zero means "no fingerprint stamped". Callers compare a non-zero
    value against the expected identity (see the data-bin vocab-fingerprint
    check in :class:`~tokenizer.aligned_data.loader.session.BinarySession`).
    """
    if len(prelude) < _PRELUDE_SIZE:
        raise ValueError(
            f"bin prelude too short for reserved field: got {len(prelude)} "
            f"bytes, expected >= {_PRELUDE_SIZE}"
        )
    start = _PRELUDE_MAGIC_SIZE + _PRELUDE_VERSION_SIZE
    return bytes(prelude[start : start + _PRELUDE_RESERVED_SIZE])


#: Sentinel reserved value meaning "no vocab fingerprint stamped" (bins
#: written before #27, or non-data bins). Readers soft-skip the check.
NO_FINGERPRINT: bytes = b"\x00" * _PRELUDE_RESERVED_SIZE


def assert_bin_prelude(
    prelude: bytes, *, expected_magic: bytes, path: str = ""
) -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid bin prelude.

    Checks length, magic, and format version. ``expected_magic`` must be
    exactly 4 bytes (asserted). ``path`` is included in error messages
    when supplied so loaders can route to the offending file without
    catching + re-raising with context.
    """
    if len(expected_magic) != _PRELUDE_MAGIC_SIZE:
        raise ValueError(
            f"expected_magic must be {_PRELUDE_MAGIC_SIZE} bytes, "
            f"got {len(expected_magic)} ({expected_magic!r})"
        )
    if len(prelude) < _PRELUDE_SIZE:
        raise ValueError(
            f"bin prelude at {path or '<unknown>'} is short: "
            f"got {len(prelude)} bytes, expected {_PRELUDE_SIZE}"
        )
    magic = bytes(prelude[:_PRELUDE_MAGIC_SIZE])
    if magic != bytes(expected_magic):
        raise ValueError(
            f"bin prelude at {path or '<unknown>'} has unexpected "
            f"magic {magic!r}; expected {bytes(expected_magic)!r}"
        )
    (version,) = struct.unpack(
        "<I",
        bytes(
            prelude[
                _PRELUDE_MAGIC_SIZE : _PRELUDE_MAGIC_SIZE + _PRELUDE_VERSION_SIZE
            ]
        ),
    )
    if version != MEMMAP_FORMAT_VERSION:
        raise ValueError(
            f"bin prelude at {path or '<unknown>'} reports "
            f"format_version={version}; v{MEMMAP_FORMAT_VERSION} required"
        )


# 16-byte file-level prelude written at the start of every `_data.bin` /
# `_unmatched_data.bin`. Layout (little-endian):
#
#     bytes 0..3   : magic = b"DATA"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
DATA_BIN_PRELUDE_MAGIC: bytes = b"DATA"
DATA_BIN_PRELUDE_SIZE: int = _PRELUDE_SIZE


def encode_data_bin_prelude(
    vocab_fingerprint: bytes = NO_FINGERPRINT,
) -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``_data.bin``.

    ``vocab_fingerprint`` (8 bytes) records the identity of the unified
    vocab this catalog was built against (see
    :func:`tokenizer.aligned_data.loader.unified_vocab_gate.compute_vocab_fingerprint`).
    The loader compares it against the loaded vocab and HARD-FAILS on
    mismatch — catching the case where a catalog is decoded with the wrong
    (same-format-version) vocab. ``_data.bin`` stores unified-vocab token ids
    for the whole stream, so a wrong vocab silently mis-decodes EVERY token
    (not just the variant axes). Defaults to :data:`NO_FINGERPRINT` (soft
    no-op) so fixtures + bins predating the fingerprint stay valid.
    """
    return encode_bin_prelude(DATA_BIN_PRELUDE_MAGIC, reserved=vocab_fingerprint)


def assert_data_bin_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid ``_data.bin`` prelude.

    ``path`` is included in the error message when supplied. Single
    chokepoint for the validation so loaders can call it without
    duplicating the magic / version check inline.
    """
    assert_bin_prelude(prelude, expected_magic=DATA_BIN_PRELUDE_MAGIC, path=path)


# Trailing ``total_entries`` field stamped at the end of every
# ``_data.bin`` / ``_unmatched_data.bin``. Holds the count of entries
# (records) that the file contains; paired with each record's
# per-header ``entry_idx`` (see
# :mod:`tokenizer.aligned_data.binary_format._header`) it gives the
# loader a cross-check (``entry_idx < total_entries`` per lookup,
# ``entry_idx == i`` on the load-time per-arm sweep).
#
# Written u32-aligned: writers pad the prior content's tail up to a
# 4-byte boundary with zero bytes before stamping the trailer. The
# pad is at most 3 bytes (record bodies are already 16-byte-aligned
# in the production writer; the manual-fixture path however appends
# bytes without record alignment).
DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE: int = 4


def _trailer_pad_for_cursor(cursor: int) -> int:
    """Return the zero-pad byte count needed to u32-align ``cursor``."""
    return (-cursor) % DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE


def encode_data_bin_trailer(total_entries: int, *, cursor: int) -> bytes:
    """Return the trailer bytes (zero-pad + ``u32 total_entries``).

    ``cursor`` is the writer's current byte position immediately
    before the trailer is written; we use it to derive a 0..3-byte
    zero pad so the ``u32 total_entries`` lands u32-aligned. Callers
    pass the trailer to the writer's append-at-cursor primitive.
    """
    if total_entries < 0:
        raise ValueError(
            f"total_entries must be non-negative, got {total_entries}"
        )
    pad = _trailer_pad_for_cursor(cursor)
    return b"\x00" * pad + struct.pack("<I", total_entries)


def read_data_bin_trailer(data_mmap) -> int:
    """Read the trailing ``total_entries`` u32 from a ``_data.bin`` mmap.

    The trailer is always the last 4 bytes of the file (the pad that
    u32-aligns it is BEFORE the trailer, so the trailer itself ends
    on the file's last byte). Returns the ``total_entries`` count.
    Raises :class:`ValueError` when the file is too short to hold a
    trailer (i.e. shorter than 4 bytes past the prelude).
    """
    file_size = len(data_mmap)
    if file_size < DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE:
        raise ValueError(
            f"_data.bin too short to hold a {DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE}"
            f"-byte total_entries trailer: file_size={file_size}"
        )
    trailer_start = file_size - DATA_BIN_TRAILER_TOTAL_ENTRIES_SIZE
    (total_entries,) = struct.unpack(
        "<I", bytes(data_mmap[trailer_start:file_size])
    )
    return total_entries


# 16-byte file-level prelude written at the start of every
# ``<binary>_sections.bin``. Layout (little-endian):
#
#     bytes 0..3   : magic = b"MSEC"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
MATCHED_SECTIONS_BIN_PRELUDE_MAGIC: bytes = b"MSEC"
MATCHED_SECTIONS_BIN_PRELUDE_SIZE: int = _PRELUDE_SIZE


def encode_matched_sections_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``<binary>_sections.bin``."""
    return encode_bin_prelude(MATCHED_SECTIONS_BIN_PRELUDE_MAGIC)


def assert_matched_sections_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid sections.bin prelude.

    Same single-chokepoint policy as :func:`assert_data_bin_prelude`.
    """
    assert_bin_prelude(
        prelude, expected_magic=MATCHED_SECTIONS_BIN_PRELUDE_MAGIC, path=path
    )


# 16-byte file-level prelude written at the start of every
# ``<binary>_lengths.bin`` / ``<binary>_unmatched_lengths.bin`` (the
# realized-token-length sidecars). Layout (little-endian):
#
#     bytes 0..3   : magic = b"RLEN"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
#
# Body after the prelude: a flat ``u32`` array of realized record-body
# lengths (one per (section, variant), section-major, variants in
# catalog order). See :mod:`.realized_lengths._format` for the body /
# CSR jump-table dtype contract.
REALIZED_LENGTHS_BIN_PRELUDE_MAGIC: bytes = b"RLEN"
REALIZED_LENGTHS_BIN_PRELUDE_SIZE: int = _PRELUDE_SIZE


def encode_realized_lengths_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``<binary>_lengths.bin``."""
    return encode_bin_prelude(REALIZED_LENGTHS_BIN_PRELUDE_MAGIC)


def assert_realized_lengths_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid lengths.bin prelude.

    Same single-chokepoint policy as :func:`assert_data_bin_prelude`.
    """
    assert_bin_prelude(
        prelude, expected_magic=REALIZED_LENGTHS_BIN_PRELUDE_MAGIC, path=path
    )


# 16-byte file-level prelude written at the start of every
# ``<binary>_lengths_index.bin`` / ``<binary>_unmatched_lengths_index.bin``
# (the per-section CSR jump table for the realized-length sidecars).
# Layout (little-endian):
#
#     bytes 0..3   : magic = b"RLIX"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
#
# Body after the prelude: ``n_sections + 1`` ``u32`` CSR entries (element
# offsets into the paired ``_lengths.bin`` body). See
# :mod:`.realized_lengths._format`.
REALIZED_LENGTHS_INDEX_BIN_PRELUDE_MAGIC: bytes = b"RLIX"
REALIZED_LENGTHS_INDEX_BIN_PRELUDE_SIZE: int = _PRELUDE_SIZE


def encode_realized_lengths_index_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``<binary>_lengths_index.bin``."""
    return encode_bin_prelude(REALIZED_LENGTHS_INDEX_BIN_PRELUDE_MAGIC)


def assert_realized_lengths_index_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid lengths_index.bin prelude.

    Same single-chokepoint policy as :func:`assert_data_bin_prelude`.
    """
    assert_bin_prelude(
        prelude,
        expected_magic=REALIZED_LENGTHS_INDEX_BIN_PRELUDE_MAGIC,
        path=path,
    )


# 16-byte file-level prelude written at the start of every
# ``<binary>_realized.bin`` / ``<binary>_unmatched_realized.bin`` (the
# realized-GEOMETRY sidecars — a superset of the realized-length body).
# Layout (little-endian):
#
#     bytes 0..3   : magic = b"RLG3"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
#
# Body after the prelude: THREE contiguous ``u32`` blocks each of length
# N = total (section, variant) count, section-major, in axis order
# (body_len, id_count, value_count). The three blocks are parallel and
# share the single ``RGIX`` CSR jump table. See
# :mod:`.realized_lengths._geometry_format` for the body / CSR contract.
REALIZED_GEOMETRY_BIN_PRELUDE_MAGIC: bytes = b"RLG3"
REALIZED_GEOMETRY_BIN_PRELUDE_SIZE: int = _PRELUDE_SIZE


def encode_realized_geometry_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``<binary>_realized.bin``."""
    return encode_bin_prelude(REALIZED_GEOMETRY_BIN_PRELUDE_MAGIC)


def assert_realized_geometry_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid realized.bin prelude.

    Same single-chokepoint policy as :func:`assert_data_bin_prelude`.
    """
    assert_bin_prelude(
        prelude, expected_magic=REALIZED_GEOMETRY_BIN_PRELUDE_MAGIC, path=path
    )


# 16-byte file-level prelude written at the start of every
# ``<binary>_realized_index.bin`` / ``<binary>_unmatched_realized_index.bin``
# (the per-section CSR jump table SHARED by the three geometry blocks).
# Layout (little-endian):
#
#     bytes 0..3   : magic = b"RGIX"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
#
# Body after the prelude: ``n_sections + 1`` ``u32`` CSR entries (element
# offsets into EACH of the three parallel geometry blocks). See
# :mod:`.realized_lengths._geometry_format`.
REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_MAGIC: bytes = b"RGIX"
REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_SIZE: int = _PRELUDE_SIZE


def encode_realized_geometry_index_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``<binary>_realized_index.bin``."""
    return encode_bin_prelude(REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_MAGIC)


def assert_realized_geometry_index_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid realized_index.bin prelude.

    Same single-chokepoint policy as :func:`assert_data_bin_prelude`.
    """
    assert_bin_prelude(
        prelude,
        expected_magic=REALIZED_GEOMETRY_INDEX_BIN_PRELUDE_MAGIC,
        path=path,
    )
