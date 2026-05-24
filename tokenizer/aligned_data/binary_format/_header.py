"""On-wire header for ``_data.bin`` records: dataclass + parse + encode.

Two on-wire forms live here and the format dispatch happens in ONE
place (the ``parse_binary_header`` / ``encode_binary_header`` pair):

* **Ultrashort** (7 bytes total). Triggered iff every field fits in a
  small range AND the block runlength is ``u8``::

      byte 0  bits 0-1   = 0           (format tag = ultrashort)
              bits 2-7   = #insn       (u6, cap 63)
      byte 1             = #block_word (u8, cap 255 -- block_enc implicit u8)
      byte 2             = #tokens     (u8, cap 255)
      bytes 3-6 (u32 LE) = entry_idx   (in-file ordinal, 0-based)

* **Normal** (11-14 bytes). The packed byte's low 2 bits double as
  block-encoding selector and the next 2 bits select the ``#tokens``
  width tag; the field's high 4 bits live in byte 0 and the remaining
  low bytes follow::

      byte 0  bits 0-1   = format ∈ {1,2,3}   (= block_enc + 1 -> u8/u16/u32)
              bits 2-3   = tokens width tag   (0->u12, 1->u20, 2->u28, 3->u36)
              bits 4-7   = high 4 bits of #tokens
      next 1-4 bytes     = low bytes of #tokens (per width tag)
      next u24 LE        = #insn        (byte count, cap 16 MiB)
      next u16 LE        = #block_word  (word count, cap 65535)
      next u32 LE        = entry_idx    (in-file ordinal, 0-based)

The ``entry_idx`` field is the record's encounter-order position in
its containing ``_data.bin`` file: the first record written has
``entry_idx == 0``, the second has ``1``, ... the Nth has ``N-1``.
Paired with the file-level ``total_entries`` trailer (see
:mod:`tokenizer.aligned_data.memmap_format`) it lets the loader assert
``entry_idx < total_entries`` per lookup and ``entry_idx == i`` over
the arm's known per-record starts at session open.

The byte layout has NO reserved bits and NO magic guard -- by user
direction the trade-off is accepted in exchange for the compact header.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Public constants -- caps + alignment.
# ---------------------------------------------------------------------------

RECORD_ALIGNMENT = 16

# Ultrashort field caps (exclusive upper bounds match the plan's "< X" form).
ULTRASHORT_INSN_CAP = 1 << 6      # u6 -> 0..63
ULTRASHORT_BLOCK_CAP = 1 << 8     # u8 -> 0..255
ULTRASHORT_TOKENS_CAP = 1 << 8    # u8 -> 0..255

# Normal field caps.
NORMAL_INSN_CAP = 1 << 24         # u24 -> 0..16 777 215 bytes
NORMAL_BLOCK_WORD_CAP = 1 << 16   # u16 -> 0..65 535 words

# Token-width-tag -> cap on #tokens (4 + tag*8 bits of total token width).
NORMAL_TOKEN_CAPS: Tuple[int, int, int, int] = (
    1 << 12,   # tag 0  ->  4095
    1 << 20,   # tag 1  ->  1 048 575
    1 << 28,   # tag 2  ->  268 435 455
    1 << 36,   # tag 3  ->  68 719 476 735
)

# Width of the trailing ``entry_idx`` field present on every header
# (both ULTRASHORT and NORMAL forms).
ENTRY_IDX_SIZE = 4

# Maximum header byte counts, indexed by token-width tag for the normal
# form (1 packed byte + 1..4 low bytes + 3 byte #insn + 2 byte #block
# + 4 byte entry_idx).
NORMAL_PREFIX_BYTES: Tuple[int, int, int, int] = (
    7 + ENTRY_IDX_SIZE,
    8 + ENTRY_IDX_SIZE,
    9 + ENTRY_IDX_SIZE,
    10 + ENTRY_IDX_SIZE,
)
ULTRASHORT_PREFIX_BYTES = 3 + ENTRY_IDX_SIZE
MAX_HEADER_BYTES = NORMAL_PREFIX_BYTES[-1]  # 14

# block_enc index -> sizeof(block word) in bytes.
BLOCK_WORD_SIZE: Tuple[int, int, int] = (1, 2, 4)


# ---------------------------------------------------------------------------
# IndexEntrySkip: shared skip exception (used by encoders project-wide).
# ---------------------------------------------------------------------------


class IndexEntrySkip(Exception):
    """Raised by encoders when a per-section field overflows its cap.

    Callers catch and translate into an ``error.log`` entry plus a
    skipped index entry; the build continues. ``reason`` names the
    overflowing field; ``value`` is the offending integer so the log
    line can record what triggered the skip.
    """

    def __init__(self, reason: str, value: int) -> None:
        super().__init__(f"{reason} (value={value})")
        self.reason = reason
        self.value = value


# ---------------------------------------------------------------------------
# Header dataclass + format enum.
# ---------------------------------------------------------------------------


class BinaryHeaderFormat(enum.IntEnum):
    """On-wire format tag carried in the control byte's low 2 bits."""

    UltraShort = 0
    Normal = 1


@dataclass(frozen=True)
class BinaryHeader:
    """Decoded record header.

    ``block_enc`` is always one of ``{0, 1, 2}`` denoting block-word
    width ``u8 / u16 / u32``; for ultrashort records it is fixed to 0
    (block words are implicit ``u8``). ``block_word_count`` is a WORD
    count -- the byte count is ``block_word_count * sizeof(block_enc)``.
    """

    format: BinaryHeaderFormat
    block_enc: int
    insn_len: int
    block_word_count: int
    token_count: int
    entry_idx: int


# ---------------------------------------------------------------------------
# Block-encoding helper (writer-side input mapping).
# ---------------------------------------------------------------------------


def determine_block_encoding(block_runlength: np.ndarray) -> int:
    """Return the block_enc index (0=u8, 1=u16, 2=u32) for an array."""
    if block_runlength.dtype == np.uint8:
        return 0
    if block_runlength.dtype == np.uint16:
        return 1
    if block_runlength.dtype == np.uint32:
        return 2
    raise ValueError(
        f"unsupported block_runlength dtype {block_runlength.dtype!r}; "
        "expected one of uint8 / uint16 / uint32"
    )


# ---------------------------------------------------------------------------
# Internal predicates: ultrashort eligibility + token-width-tag selection.
# ---------------------------------------------------------------------------


def _ultrashort_eligible(
    block_enc: int, insn_len: int, block_word_count: int, token_count: int
) -> bool:
    """The strict ultrashort predicate (one source of truth).

    Used by both ``encode_binary_header`` (to pick the form) and
    ``prefix_bytes_for_header`` (to recover prefix width from a
    parsed header); the validator's pad-consistency check therefore
    re-derives the same form the writer would have chosen.
    """
    return (
        block_enc == 0
        and insn_len < ULTRASHORT_INSN_CAP
        and block_word_count < ULTRASHORT_BLOCK_CAP
        and token_count < ULTRASHORT_TOKENS_CAP
    )


def _select_token_width_tag(token_count: int) -> int:
    """Pick the smallest normal-form width tag whose cap holds ``token_count``.

    Raises :class:`IndexEntrySkip` (``token_count_overflow``) when the
    value exceeds even the largest cap (u36).
    """
    for tag, cap in enumerate(NORMAL_TOKEN_CAPS):
        if token_count < cap:
            return tag
    raise IndexEntrySkip("token_count_overflow", token_count)


def prefix_bytes_for_header(header: BinaryHeader) -> int:
    """How many bytes ``header`` occupies on disk.

    Mirrors the encoder's dispatch so callers that only have a parsed
    header can recover the prefix width without re-encoding. Public so
    geometry helpers in sibling modules can consume it.
    """
    if header.format is BinaryHeaderFormat.UltraShort:
        return ULTRASHORT_PREFIX_BYTES
    width_tag = _select_token_width_tag(header.token_count)
    return NORMAL_PREFIX_BYTES[width_tag]


# ---------------------------------------------------------------------------
# Parse / encode -- format dispatch lives here, in ONE pair of functions.
# ---------------------------------------------------------------------------


def _byte_at(data, idx: int) -> int:
    """Return one byte of ``data`` as a plain ``int`` (zero-copy)."""
    return int(data[idx])


def _slice_to_int(data, start: int, end: int) -> int:
    """Read bytes ``[start:end]`` as a little-endian unsigned int."""
    return int.from_bytes(bytes(data[start:end]), "little")


def parse_binary_header(
    data: Union[bytes, bytearray, memoryview, np.ndarray],
) -> Tuple[BinaryHeader, int]:
    """Decode the record header from ``data`` (starting at byte 0).

    Returns ``(header, prefix_bytes)`` where ``prefix_bytes`` is the
    number of bytes the header occupies on disk. Only the prefix bytes
    of ``data`` are touched -- for a memmap slice that means just
    those bytes are paged in, never the full record body.
    """
    packed = _byte_at(data, 0)
    fmt_bits = packed & 0b11

    if fmt_bits == BinaryHeaderFormat.UltraShort:
        insn_len = (packed >> 2) & 0b111111  # u6
        block_word_count = _byte_at(data, 1)
        token_count = _byte_at(data, 2)
        entry_idx = _slice_to_int(data, 3, 3 + ENTRY_IDX_SIZE)
        header = BinaryHeader(
            format=BinaryHeaderFormat.UltraShort,
            block_enc=0,
            insn_len=insn_len,
            block_word_count=block_word_count,
            token_count=token_count,
            entry_idx=entry_idx,
        )
        return header, ULTRASHORT_PREFIX_BYTES

    # Normal: fmt_bits is block_enc + 1 -> block_enc ∈ {0,1,2}.
    block_enc = fmt_bits - 1
    width_tag = (packed >> 2) & 0b11
    token_hi4 = (packed >> 4) & 0b1111
    low_byte_count = width_tag + 1  # 1..4
    token_low = _slice_to_int(data, 1, 1 + low_byte_count)
    token_count = (token_hi4 << (low_byte_count * 8)) | token_low

    cursor = 1 + low_byte_count
    insn_len = _slice_to_int(data, cursor, cursor + 3)
    cursor += 3
    block_word_count = _slice_to_int(data, cursor, cursor + 2)
    cursor += 2
    entry_idx = _slice_to_int(data, cursor, cursor + ENTRY_IDX_SIZE)
    cursor += ENTRY_IDX_SIZE

    header = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=block_enc,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
        entry_idx=entry_idx,
    )
    return header, cursor


def encode_binary_header(header: BinaryHeader) -> bytes:
    """Encode ``header`` into 3 bytes (ultrashort) or 7-10 bytes (normal).

    The ``header.format`` field is informational; the encoder
    re-derives the form from the strict ultrashort predicate so a
    handwritten ``BinaryHeader(format=Normal, ...)`` that *would* fit
    ultrashort is still emitted in the canonical (shortest) form. That
    keeps the on-wire layout deterministic for any given field tuple
    -- the validator's pad-consistency check relies on re-deriving
    the form from the parsed fields alone.

    Raises :class:`IndexEntrySkip` (``insn_len_overflow``,
    ``block_word_count_overflow``, or ``token_count_overflow``) when a
    field exceeds its on-wire cap; raises :class:`ValueError` for
    programmer-error inputs (negative fields, out-of-range
    ``block_enc``).
    """
    if header.block_enc not in (0, 1, 2):
        raise ValueError(
            f"block_enc must be 0, 1, or 2; got {header.block_enc}"
        )
    if (
        header.insn_len < 0
        or header.block_word_count < 0
        or header.token_count < 0
        or header.entry_idx < 0
    ):
        raise ValueError(
            "header fields must be non-negative; got "
            f"insn_len={header.insn_len}, "
            f"block_word_count={header.block_word_count}, "
            f"token_count={header.token_count}, "
            f"entry_idx={header.entry_idx}"
        )

    entry_idx_bytes = struct.pack("<I", header.entry_idx)

    if _ultrashort_eligible(
        header.block_enc,
        header.insn_len,
        header.block_word_count,
        header.token_count,
    ):
        packed = (
            BinaryHeaderFormat.UltraShort
            | ((header.insn_len & 0b111111) << 2)
        )
        return bytes(
            (packed, header.block_word_count, header.token_count)
        ) + entry_idx_bytes

    # Normal form.
    if header.insn_len >= NORMAL_INSN_CAP:
        raise IndexEntrySkip("insn_len_overflow", header.insn_len)
    if header.block_word_count >= NORMAL_BLOCK_WORD_CAP:
        raise IndexEntrySkip("block_word_count_overflow", header.block_word_count)
    width_tag = _select_token_width_tag(header.token_count)
    low_byte_count = width_tag + 1
    fmt_value = header.block_enc + 1  # {1,2,3}

    low_bits = low_byte_count * 8
    token_hi4 = (header.token_count >> low_bits) & 0b1111
    token_low_mask = (1 << low_bits) - 1
    token_low = header.token_count & token_low_mask

    packed = fmt_value | (width_tag << 2) | (token_hi4 << 4)
    out = bytearray()
    out.append(packed)
    out.extend(token_low.to_bytes(low_byte_count, "little"))
    out.extend(struct.pack("<I", header.insn_len)[0:3])
    out.extend(struct.pack("<H", header.block_word_count))
    out.extend(entry_idx_bytes)
    return bytes(out)
