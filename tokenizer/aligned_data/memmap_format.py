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


def encode_bin_prelude(magic: bytes) -> bytes:
    """Return the 16-byte file-level prelude for a bin with the given magic.

    ``magic`` must be exactly 4 bytes. The rest of the prelude is the
    little-endian ``u32`` ``MEMMAP_FORMAT_VERSION`` followed by 8 zero
    bytes of reserved space (currently unused; future format revisions
    may carve fields out of it without bumping the prelude size).
    """
    if len(magic) != _PRELUDE_MAGIC_SIZE:
        raise ValueError(
            f"prelude magic must be {_PRELUDE_MAGIC_SIZE} bytes, "
            f"got {len(magic)} ({magic!r})"
        )
    return (
        bytes(magic)
        + struct.pack("<I", MEMMAP_FORMAT_VERSION)
        + b"\x00" * _PRELUDE_RESERVED_SIZE
    )


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


def encode_data_bin_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``_data.bin``."""
    return encode_bin_prelude(DATA_BIN_PRELUDE_MAGIC)


def assert_data_bin_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid ``_data.bin`` prelude.

    ``path`` is included in the error message when supplied. Single
    chokepoint for the validation so loaders can call it without
    duplicating the magic / version check inline.
    """
    assert_bin_prelude(prelude, expected_magic=DATA_BIN_PRELUDE_MAGIC, path=path)


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
