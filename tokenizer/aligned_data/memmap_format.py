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
"""

import struct

MEMMAP_FORMAT_VERSION: int = 1

# 16-byte file-level prelude written at the start of every `_data.bin` /
# `_unmatched_data.bin`. Layout (little-endian):
#
#     bytes 0..3   : magic = b"DATA"
#     bytes 4..7   : u32 MEMMAP_FORMAT_VERSION
#     bytes 8..15  : reserved (zero)
DATA_BIN_PRELUDE_MAGIC: bytes = b"DATA"
DATA_BIN_PRELUDE_SIZE: int = 16


def encode_data_bin_prelude() -> bytes:
    """Return the 16-byte prelude bytes for a fresh ``_data.bin``."""
    return (
        DATA_BIN_PRELUDE_MAGIC
        + struct.pack("<I", MEMMAP_FORMAT_VERSION)
        + b"\x00" * 8
    )


def assert_data_bin_prelude(prelude: bytes, *, path: str = "") -> None:
    """Raise ``ValueError`` if ``prelude`` is not a valid ``_data.bin`` prelude.

    ``path`` is included in the error message when supplied. Single
    chokepoint for the validation so loaders can call it without
    duplicating the magic / version check inline.
    """
    if len(prelude) < DATA_BIN_PRELUDE_SIZE:
        raise ValueError(
            f"_data.bin prelude at {path or '<unknown>'} is short: "
            f"got {len(prelude)} bytes, expected {DATA_BIN_PRELUDE_SIZE}"
        )
    magic = bytes(prelude[: len(DATA_BIN_PRELUDE_MAGIC)])
    if magic != DATA_BIN_PRELUDE_MAGIC:
        raise ValueError(
            f"_data.bin prelude at {path or '<unknown>'} has unexpected "
            f"magic {magic!r}; expected {DATA_BIN_PRELUDE_MAGIC!r}"
        )
    version_offset = len(DATA_BIN_PRELUDE_MAGIC)
    (version,) = struct.unpack(
        "<I", bytes(prelude[version_offset : version_offset + 4])
    )
    if version != MEMMAP_FORMAT_VERSION:
        raise ValueError(
            f"_data.bin prelude at {path or '<unknown>'} reports "
            f"format_version={version}; v{MEMMAP_FORMAT_VERSION} required"
        )
