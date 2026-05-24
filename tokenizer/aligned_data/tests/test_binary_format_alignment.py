"""Tests for ``tokenizer.aligned_data.binary_format`` (variable-width header).

Covers:
* Ultrashort round-trip + cap edges (insn u6, block_word u8, tokens u8).
* Normal round-trip across all 4 ``#tokens`` width tags (u12/u20/u28/u36)
  and every ``block_enc`` (0/1/2).
* Encoder canonicalisation: a ``BinaryHeader(format=Normal, ...)`` that
  *fits* ultrashort is still emitted in ultrashort form.
* Pad placement covers (block_enc × insn_len mod 4 × token_count) and
  exercises the ``B > P`` fallback (all-pad-pre-block).
* ``record_total_size(header) % 16 == 0`` on random inputs.
* Overflow guards: ``IndexEntrySkip`` for insn_len / block_word_count /
  token_count beyond their normal-form caps.
* ``record_token_count_from_memmap`` reads the header without touching
  the body.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    BLOCK_WORD_SIZE,
    MAX_HEADER_BYTES,
    NORMAL_BLOCK_WORD_CAP,
    NORMAL_INSN_CAP,
    NORMAL_PREFIX_BYTES,
    NORMAL_TOKEN_CAPS,
    RECORD_ALIGNMENT,
    ULTRASHORT_BLOCK_CAP,
    ULTRASHORT_INSN_CAP,
    ULTRASHORT_PREFIX_BYTES,
    ULTRASHORT_TOKENS_CAP,
    BinaryHeader,
    BinaryHeaderFormat,
    IndexEntrySkip,
    derive_pad_placement,
    determine_block_encoding,
    encode_binary_header,
    extract_arrays_from_data,
    parse_binary_header,
    prefix_bytes_for_header,
    record_token_count,
    record_token_count_from_memmap,
    record_total_size,
)



# ---------------------------------------------------------------------------
# Header round-trip: ultrashort.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("insn_len", [0, 1, 7, 31, ULTRASHORT_INSN_CAP - 1])
@pytest.mark.parametrize("block_word_count", [0, 1, 17, 200, ULTRASHORT_BLOCK_CAP - 1])
@pytest.mark.parametrize("token_count", [0, 1, 50, 200, ULTRASHORT_TOKENS_CAP - 1])
def test_ultrashort_round_trip(insn_len, block_word_count, token_count):
    h = BinaryHeader(
        format=BinaryHeaderFormat.UltraShort,
        block_enc=0,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
        entry_idx=0,
    )
    enc = encode_binary_header(h)
    assert len(enc) == ULTRASHORT_PREFIX_BYTES == 7
    assert enc[0] & 0b11 == BinaryHeaderFormat.UltraShort

    parsed, prefix = parse_binary_header(enc)
    assert prefix == 7
    assert parsed == h


def test_ultrashort_control_byte_wire_layout():
    """Wire layout: byte0[bits 0-1]=0, byte0[bits 2-7]=insn, byte1=block,
    byte2=tokens, bytes 3..6=entry_idx (u32 LE)."""
    h = BinaryHeader(
        format=BinaryHeaderFormat.UltraShort,
        block_enc=0,
        insn_len=0b101010,
        block_word_count=0xA5,
        token_count=0x5A,
        entry_idx=0x12345678,
    )
    enc = encode_binary_header(h)
    # bits 0-1 = 00 (ultrashort), bits 2-7 = 0b101010
    assert enc[0] == (0b101010 << 2)
    assert enc[1] == 0xA5
    assert enc[2] == 0x5A
    assert enc[3:7] == b"\x78\x56\x34\x12"


def test_encoder_canonicalises_normal_to_ultrashort_when_eligible():
    """Encoder MUST emit ultrashort whenever the predicate holds.

    A handwritten ``BinaryHeader(format=Normal, ...)`` whose fields all
    fit ultrashort still serialises in 3 bytes. Otherwise the format
    field would let callers pick a non-canonical on-wire layout and the
    validator's pad-consistency rule (which re-derives the form from
    fields alone) would disagree with the writer.
    """
    h_normal = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=0,
        insn_len=10,
        block_word_count=5,
        token_count=7,
        entry_idx=0,
    )
    enc = encode_binary_header(h_normal)
    assert len(enc) == ULTRASHORT_PREFIX_BYTES
    parsed, _ = parse_binary_header(enc)
    assert parsed.format is BinaryHeaderFormat.UltraShort
    assert parsed.block_enc == 0
    assert parsed.insn_len == 10
    assert parsed.block_word_count == 5
    assert parsed.token_count == 7


# ---------------------------------------------------------------------------
# Header round-trip: normal, across all (width_tag, block_enc) combinations.
# ---------------------------------------------------------------------------


# token_count values: just below the cap for each width tag so the tag is
# forced. The encoder picks the smallest tag whose cap holds the value.
_NORMAL_TOKEN_AT_TAG = (
    NORMAL_TOKEN_CAPS[0] - 1,  # u12 max -> 4095
    NORMAL_TOKEN_CAPS[1] - 1,  # u20 max
    NORMAL_TOKEN_CAPS[2] - 1,  # u28 max
    NORMAL_TOKEN_CAPS[3] - 1,  # u36 max
)


@pytest.mark.parametrize("width_tag", [0, 1, 2, 3])
@pytest.mark.parametrize("block_enc", [0, 1, 2])
def test_normal_round_trip_per_width_tag(width_tag, block_enc):
    token_count = _NORMAL_TOKEN_AT_TAG[width_tag]
    # Force "normal" by exceeding ultrashort caps even when block_enc==0.
    insn_len = ULTRASHORT_INSN_CAP + 5  # >= 64
    block_word_count = ULTRASHORT_BLOCK_CAP + 1  # >= 256
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=block_enc,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
        entry_idx=0,
    )
    enc = encode_binary_header(h)
    expected_prefix = NORMAL_PREFIX_BYTES[width_tag]
    assert len(enc) == expected_prefix, (width_tag, enc.hex())
    assert enc[0] & 0b11 == block_enc + 1  # normal format value

    parsed, prefix = parse_binary_header(enc)
    assert prefix == expected_prefix
    assert parsed == h


def test_normal_format_dispatch_picks_smallest_width_tag():
    """The encoder picks the *smallest* width tag whose cap holds tokens.

    Cross-checked by inspecting the on-wire prefix length per tag.
    """
    # Just below each cap -> selects exactly that tag.
    for expected_tag, n_tokens in enumerate(_NORMAL_TOKEN_AT_TAG):
        h = BinaryHeader(
            format=BinaryHeaderFormat.Normal,
            block_enc=1,
            insn_len=NORMAL_INSN_CAP - 1,
            block_word_count=NORMAL_BLOCK_WORD_CAP - 1,
            token_count=n_tokens,
            entry_idx=0,
        )
        enc = encode_binary_header(h)
        assert len(enc) == NORMAL_PREFIX_BYTES[expected_tag], (
            n_tokens, expected_tag, len(enc)
        )
        assert (enc[0] >> 2) & 0b11 == expected_tag


def test_normal_wire_layout_byte_for_byte():
    """Spot-check: a u20 (tag=1) header serialises to exactly the bytes
    we expect, byte by byte."""
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=2,                     # -> fmt_value 3
        insn_len=0x010203,               # u24 = bytes 03 02 01
        block_word_count=0x0405,         # u16 = bytes 05 04
        token_count=0x000A_BCDE,         # 20 bits set: hi4 = 0xA, low16 = 0xBCDE
        entry_idx=0xDEADBEEF,            # u32 LE -> EF BE AD DE
    )
    enc = encode_binary_header(h)
    # tag = 1 (u20) -> prefix = 8 + 4 (entry_idx) = 12 bytes
    assert len(enc) == 12
    # byte 0: fmt=3, tag=1, hi4=0xA  ->  0b1010_01_11 = 0xA7
    assert enc[0] == 0xA7
    # bytes 1-2: low16 of token_count (LE)
    assert enc[1:3] == b"\xDE\xBC"
    # bytes 3-5: u24 insn_len (LE)
    assert enc[3:6] == b"\x03\x02\x01"
    # bytes 6-7: u16 block_word_count (LE)
    assert enc[6:8] == b"\x05\x04"
    # bytes 8-11: u32 entry_idx (LE)
    assert enc[8:12] == b"\xEF\xBE\xAD\xDE"


# ---------------------------------------------------------------------------
# Pad placement: 16-byte record alignment + (B vs P) branch coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [0, 1, 2])
@pytest.mark.parametrize("insn_len_mod", [0, 1, 2, 3])
@pytest.mark.parametrize("token_count", [0, 1, 7, 16, 50])
def test_record_total_size_is_aligned(block_enc, insn_len_mod, token_count):
    insn_len = 100 + insn_len_mod  # span all residues mod 4
    block_word_count = 5
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=block_enc,
        insn_len=insn_len,
        block_word_count=block_word_count,
        token_count=token_count,
        entry_idx=0,
    )
    total = record_total_size(h)
    assert total % RECORD_ALIGNMENT == 0
    pre, post = derive_pad_placement(h)
    assert 0 <= pre <= 15 and 0 <= post <= 15
    # Block start must be aligned to block_word_size whenever possible
    # (i.e. when the writer chose pre = B, the all-pad-pre fallback aside).


def test_pad_placement_block_aligned_when_b_le_p():
    """In the common B <= P branch, block_start is aligned to block_word_size."""
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=2,        # u32 -> block_align = 4
        insn_len=65,        # forces normal (>= 64)
        block_word_count=1, # block_bytes = 4
        token_count=1,
        entry_idx=0,
    )
    pre, post = derive_pad_placement(h)
    prefix = prefix_bytes_for_header(h)
    block_start = prefix + h.insn_len + pre
    assert block_start % BLOCK_WORD_SIZE[h.block_enc] == 0
    # Sanity: total still 16-aligned.
    assert record_total_size(h) % RECORD_ALIGNMENT == 0


def test_pad_placement_fallback_when_b_gt_p():
    """The B > P branch: ``derive_pad_placement`` returns (P, 0).

    We construct a record where the natural total pad P is *smaller*
    than the alignment-padding B that the block would need; the rule
    falls back to all-pre-block, leaving the block unaligned this
    record.

    block_enc=2 -> block_align=4; ultrashort eligibility is escaped via
    block_enc != 0 so the format is normal. We want
    B = (-(prefix + insn_len)) % 4 > P = (-U) % 16.
    """
    # Iterate small ints and pick the first hit so the branch is real.
    found = None
    for insn_len in range(64, 200):
        for block_word_count in range(1, 20):
            for token_count in range(0, 20):
                h = BinaryHeader(
                    format=BinaryHeaderFormat.Normal,
                    block_enc=2,
                    insn_len=insn_len,
                    block_word_count=block_word_count,
                    token_count=token_count,
                    entry_idx=0,
                )
                prefix = prefix_bytes_for_header(h)
                block_bytes = block_word_count * BLOCK_WORD_SIZE[2]
                unpadded = prefix + insn_len + block_bytes + 2 * token_count
                P = (-unpadded) % RECORD_ALIGNMENT
                B = (-(prefix + insn_len)) % BLOCK_WORD_SIZE[2]
                if B > P:
                    found = (h, P, B)
                    break
            if found:
                break
        if found:
            break
    assert found is not None, "no (insn, block, tokens) triple triggers B > P branch"
    h, P, B = found
    pre, post = derive_pad_placement(h)
    assert pre == P
    assert post == 0
    # Total still 16-aligned.
    assert record_total_size(h) % RECORD_ALIGNMENT == 0


def test_record_total_size_random_inputs_aligned():
    """Random sweep: 500 random (block_enc, fields) triples must all
    produce 16-byte-aligned record totals."""
    rng = random.Random(20260520)
    for _ in range(500):
        block_enc = rng.randrange(3)
        insn_len = rng.randrange(0, 4096)
        block_word_count = rng.randrange(0, 1024)
        token_count = rng.randrange(0, 10_000)
        h = BinaryHeader(
            format=BinaryHeaderFormat.Normal,
            block_enc=block_enc,
            insn_len=insn_len,
            block_word_count=block_word_count,
            token_count=token_count,
            entry_idx=0,
        )
        sz = record_total_size(h)
        assert sz % RECORD_ALIGNMENT == 0, (h, sz)


# ---------------------------------------------------------------------------
# Overflow guards: IndexEntrySkip on field-cap violation (normal form).
# ---------------------------------------------------------------------------


def test_insn_len_overflow_raises_index_entry_skip():
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=0,
        insn_len=NORMAL_INSN_CAP,
        block_word_count=ULTRASHORT_BLOCK_CAP + 1,  # force normal
        token_count=ULTRASHORT_TOKENS_CAP + 1,
        entry_idx=0,
    )
    with pytest.raises(IndexEntrySkip) as excinfo:
        encode_binary_header(h)
    assert excinfo.value.reason == "insn_len_overflow"
    assert excinfo.value.value == NORMAL_INSN_CAP


def test_block_word_count_overflow_raises_index_entry_skip():
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=1,
        insn_len=100,
        block_word_count=NORMAL_BLOCK_WORD_CAP,
        token_count=10,
        entry_idx=0,
    )
    with pytest.raises(IndexEntrySkip) as excinfo:
        encode_binary_header(h)
    assert excinfo.value.reason == "block_word_count_overflow"
    assert excinfo.value.value == NORMAL_BLOCK_WORD_CAP


def test_token_count_overflow_raises_index_entry_skip():
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=0,
        insn_len=100,
        block_word_count=ULTRASHORT_BLOCK_CAP + 1,  # force normal
        token_count=NORMAL_TOKEN_CAPS[-1],          # = u36 cap
        entry_idx=0,
    )
    with pytest.raises(IndexEntrySkip) as excinfo:
        encode_binary_header(h)
    assert excinfo.value.reason == "token_count_overflow"
    assert excinfo.value.value == NORMAL_TOKEN_CAPS[-1]


# ---------------------------------------------------------------------------
# Programmer-error guards: ValueError on bad block_enc / negative fields.
# ---------------------------------------------------------------------------


def test_bad_block_enc_raises_value_error():
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=3,
        insn_len=0,
        block_word_count=0,
        token_count=0,
        entry_idx=0,
    )
    with pytest.raises(ValueError):
        encode_binary_header(h)


def test_negative_field_raises_value_error():
    h = BinaryHeader(
        format=BinaryHeaderFormat.Normal,
        block_enc=0,
        insn_len=-1,
        block_word_count=0,
        token_count=0,
        entry_idx=0,
    )
    with pytest.raises(ValueError):
        encode_binary_header(h)


# ---------------------------------------------------------------------------
# Block encoding determination helper.
# ---------------------------------------------------------------------------


def test_determine_block_encoding_maps_dtype():
    assert determine_block_encoding(np.zeros(0, dtype=np.uint8)) == 0
    assert determine_block_encoding(np.zeros(0, dtype=np.uint16)) == 1
    assert determine_block_encoding(np.zeros(0, dtype=np.uint32)) == 2


def test_determine_block_encoding_rejects_other_dtypes():
    with pytest.raises(ValueError):
        determine_block_encoding(np.zeros(0, dtype=np.int8))


# ---------------------------------------------------------------------------
# record_token_count + record_token_count_from_memmap.
# ---------------------------------------------------------------------------


def test_record_token_count_returns_header_field():
    h = BinaryHeader(
        format=BinaryHeaderFormat.UltraShort,
        block_enc=0,
        insn_len=10,
        block_word_count=5,
        token_count=42,
        entry_idx=0,
    )
    assert record_token_count(h) == 42


def test_record_token_count_from_memmap_reads_only_header(tmp_path):
    """Pack three records of different forms into a memmap and confirm
    ``record_token_count_from_memmap`` returns the right count for each
    at the right offset."""
    headers = [
        BinaryHeader(
            format=BinaryHeaderFormat.UltraShort,
            block_enc=0,
            insn_len=3,
            block_word_count=2,
            token_count=11,
            entry_idx=0,
        ),
        BinaryHeader(
            format=BinaryHeaderFormat.Normal,
            block_enc=1,
            insn_len=200,
            block_word_count=40,
            token_count=3000,                  # u12 tag
            entry_idx=1,
        ),
        BinaryHeader(
            format=BinaryHeaderFormat.Normal,
            block_enc=2,
            insn_len=500,
            block_word_count=80,
            token_count=NORMAL_TOKEN_CAPS[2] - 1,  # u28 tag
            entry_idx=2,
        ),
    ]

    bin_path = tmp_path / "synthetic_data.bin"
    offsets = []
    with open(bin_path, "wb") as fh:
        for h in headers:
            offsets.append(fh.tell())
            enc = encode_binary_header(h)
            prefix = len(enc)
            # Fabricate a body that round-trips: insn + pre_pad + block + post_pad + tokens.
            pre, post = derive_pad_placement(h)
            block_bytes = h.block_word_count * BLOCK_WORD_SIZE[h.block_enc]
            body = (
                b"\xAA" * h.insn_len
                + b"\x00" * pre
                + b"\xBB" * block_bytes
                + b"\x00" * post
                + b"\x00" * (2 * h.token_count)
            )
            fh.write(enc)
            fh.write(body)

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    for h, off in zip(headers, offsets):
        assert record_token_count_from_memmap(mmap, off) == h.token_count


def test_record_token_count_from_memmap_reads_at_most_max_header_bytes(tmp_path):
    """``record_token_count_from_memmap`` must not require more than
    :data:`MAX_HEADER_BYTES` bytes after ``offset`` to be valid."""
    h = BinaryHeader(
        format=BinaryHeaderFormat.UltraShort,
        block_enc=0,
        insn_len=2,
        block_word_count=1,
        token_count=99,
        entry_idx=0,
    )
    enc = encode_binary_header(h)  # 7 bytes
    bin_path = tmp_path / "tiny.bin"
    # Write the header followed by ONLY zeroes -- if the function ever
    # accessed bytes deeper than the actual header end it would return
    # garbage. (Ultrashort prefix = 3, function reads min(3+max, len).)
    with open(bin_path, "wb") as fh:
        fh.write(enc)
        # Append a single zero byte so the file isn't zero-tail-trimmed.
        fh.write(b"\x00")
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    assert record_token_count_from_memmap(mmap, 0) == 99
    # MAX_HEADER_BYTES is the encoder's upper bound (= 14 after the
    # 4-byte entry_idx field was appended to every header form).
    assert MAX_HEADER_BYTES == 14
