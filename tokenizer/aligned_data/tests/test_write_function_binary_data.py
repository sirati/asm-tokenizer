"""Round-trip + cap-overflow tests for ``write_function_binary_data``.

The writer is the sole producer of ``_data.bin`` records; correctness
here is the foundation that the reader-side tests consume. The new
self-describing record header carries ``insn_len``,
``block_word_count`` and ``token_count`` -- callers no longer hand the
writer a precomputed length and the writer no longer surfaces one (the
return value is just the data-bin offset). Records always align to 16
bytes; pad placement is fully derivable from the header via
:func:`derive_pad_placement`.

We assert round-trip fidelity by reading back via
:func:`extract_arrays_from_data` (the read-side helper covered by 0A's
own tests, so this test is testing the writer-as-producer, not
re-testing the reader). Both the ultrashort form (small records, block
words implicit u8) and the normal form (all 4 token-width tags +
``block_enc`` ∈ {0,1,2}) are exercised.
"""

from __future__ import annotations

import io as stdio
import random

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import (
    BLOCK_WORD_SIZE,
    NORMAL_TOKEN_CAPS,
    RECORD_ALIGNMENT,
    ULTRASHORT_BLOCK_CAP,
    ULTRASHORT_INSN_CAP,
    ULTRASHORT_TOKENS_CAP,
    BinaryHeaderFormat,
    IndexEntrySkip,
    derive_pad_placement,
    determine_block_encoding,
    extract_arrays_from_data,
    parse_binary_header,
    record_total_size,
)
from tokenizer.aligned_data._writers import write_function_binary_data


def _make_inputs(
    insn_len: int,
    block_enc: int,
    block_count: int,
    token_count: int,
    rng: random.Random,
):
    """Build the three ndarrays the writer expects."""
    insn = np.frombuffer(rng.randbytes(insn_len), dtype=np.uint8).copy()
    block_dtype = (np.uint8, np.uint16, np.uint32)[block_enc]
    nbytes = block_count * BLOCK_WORD_SIZE[block_enc]
    block = np.frombuffer(rng.randbytes(nbytes), dtype=block_dtype).copy()
    tokens = np.frombuffer(rng.randbytes(token_count * 2), dtype=np.uint16).copy()
    return insn, block, tokens


def _round_trip(insn, block, tokens):
    """Drive the writer once and slice the bytes back via the read path.

    Returns the parsed header plus the three ndarrays that fall out of
    :func:`extract_arrays_from_data` so the test can compare element-wise.
    """
    buf = stdio.BytesIO()
    offset, total = write_function_binary_data(buf, tokens, block, insn, entry_idx=0)
    assert offset == 0  # first write lands at offset 0
    raw = buf.getvalue()
    assert total == len(raw)
    assert len(raw) % RECORD_ALIGNMENT == 0
    header, prefix_bytes = parse_binary_header(raw)
    insn_out, block_out, tokens_out = extract_arrays_from_data(
        raw, header, prefix_bytes
    )
    return header, insn_out, block_out, tokens_out, raw


# ---------------------------------------------------------------------------
# Ultrashort form (block_enc=0, small fields)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("insn_len", [0, 1, 7, ULTRASHORT_INSN_CAP - 1])
@pytest.mark.parametrize("block_count", [0, 1, 64, ULTRASHORT_BLOCK_CAP - 1])
@pytest.mark.parametrize("token_count", [0, 1, 31, ULTRASHORT_TOKENS_CAP - 1])
def test_ultrashort_round_trip(insn_len: int, block_count: int, token_count: int):
    """Every combination in the ultrashort regime round-trips bit-exactly."""
    rng = random.Random((insn_len, block_count, token_count).__hash__())
    insn, block, tokens = _make_inputs(insn_len, 0, block_count, token_count, rng)
    header, insn_out, block_out, tokens_out, _ = _round_trip(insn, block, tokens)

    assert header.format is BinaryHeaderFormat.UltraShort
    assert header.block_enc == 0
    assert header.insn_len == insn_len
    assert header.block_word_count == block_count
    assert header.token_count == token_count
    assert insn_out.tobytes() == insn.tobytes()
    assert block_out.tobytes() == block.tobytes()
    assert tokens_out.tobytes() == tokens.tobytes()


# ---------------------------------------------------------------------------
# Normal form, every block_enc + token-width tag combination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_enc", [0, 1, 2])
def test_normal_form_via_block_enc(block_enc: int):
    """``block_enc`` ∈ {1, 2} forces the normal form because ultrashort
    requires block_enc=0; ``block_enc=0`` here uses block_count past the
    ultrashort cap so the encoder still picks normal."""
    rng = random.Random(0xC0DE + block_enc)
    insn_len = 10
    block_count = ULTRASHORT_BLOCK_CAP if block_enc == 0 else 3
    token_count = 5
    insn, block, tokens = _make_inputs(
        insn_len, block_enc, block_count, token_count, rng
    )
    header, insn_out, block_out, tokens_out, _ = _round_trip(insn, block, tokens)

    assert header.format is BinaryHeaderFormat.Normal
    assert header.block_enc == block_enc
    assert header.insn_len == insn_len
    assert header.block_word_count == block_count
    assert header.token_count == token_count
    assert insn_out.tobytes() == insn.tobytes()
    assert block_out.tobytes() == block.tobytes()
    assert tokens_out.tobytes() == tokens.tobytes()


@pytest.mark.parametrize("width_tag", [0, 1, 2])
def test_normal_form_every_token_width_tag(width_tag: int):
    """Drive the writer at a token_count that selects each width tag's
    cap regime. (Tag 3 reaches into u36 territory which would allocate
    several GiB of token memory; covered by encoder unit tests in 0A.)
    """
    rng = random.Random(0xD00D + width_tag)
    # Pick a token_count that exceeds the prior width's cap so the
    # encoder must use width_tag (or larger) to encode it.
    lower_cap = NORMAL_TOKEN_CAPS[width_tag - 1] if width_tag > 0 else 0
    token_count = max(lower_cap, ULTRASHORT_TOKENS_CAP)  # past ultrashort too
    insn_len = 3
    block_count = 4  # past ultrashort_block cap via token_count? no, via tag
    insn, block, tokens = _make_inputs(insn_len, 0, block_count, token_count, rng)
    # block_enc=0 but token_count past ultrashort → normal form.
    header, _, _, tokens_out, _ = _round_trip(insn, block, tokens)
    assert header.format is BinaryHeaderFormat.Normal
    assert header.token_count == token_count
    assert tokens_out.size == token_count


# ---------------------------------------------------------------------------
# Pad placement + 16-byte alignment invariant on random shapes
# ---------------------------------------------------------------------------


def _shape_iter(rng, n):
    """Yield ``n`` random (insn_len, block_enc, block_count, token_count) shapes.

    Distribution covers ultrashort + normal + several pad residues mod 16.
    """
    for _ in range(n):
        insn_len = rng.randint(0, 200)
        block_enc = rng.randint(0, 2)
        block_count = rng.randint(0, 80)
        token_count = rng.randint(0, 80)
        yield insn_len, block_enc, block_count, token_count


def test_round_trip_200_random_shapes():
    rng = random.Random(0xA5A5)
    seen_ultrashort = 0
    seen_normal = 0
    for insn_len, block_enc, block_count, token_count in _shape_iter(rng, 200):
        insn, block, tokens = _make_inputs(
            insn_len, block_enc, block_count, token_count, rng
        )
        header, insn_out, block_out, tokens_out, raw = _round_trip(
            insn, block, tokens
        )

        assert header.insn_len == insn_len
        assert header.block_word_count == block_count
        assert header.token_count == token_count
        assert header.block_enc == determine_block_encoding(block)
        assert insn_out.tobytes() == insn.tobytes()
        assert block_out.tobytes() == block.tobytes()
        assert tokens_out.tobytes() == tokens.tobytes()

        # Geometry-rule invariants.
        assert len(raw) == record_total_size(header)
        assert len(raw) % RECORD_ALIGNMENT == 0
        pre_pad, post_pad = derive_pad_placement(header)
        # Pad bytes must be zero where the writer claims it laid them.
        # Reconstruct slice boundaries from the same rule the reader uses.
        from tokenizer.aligned_data.binary_format import prefix_bytes_for_header

        prefix = prefix_bytes_for_header(header)
        insn_end = prefix + header.insn_len
        block_start = insn_end + pre_pad
        block_bytes = header.block_word_count * BLOCK_WORD_SIZE[header.block_enc]
        block_end = block_start + block_bytes
        tokens_start = block_end + post_pad
        assert raw[insn_end:block_start] == b"\x00" * pre_pad
        assert raw[block_end:tokens_start] == b"\x00" * post_pad

        if header.format is BinaryHeaderFormat.UltraShort:
            seen_ultrashort += 1
        else:
            seen_normal += 1
    assert seen_ultrashort > 0, "shape distribution missed ultrashort form"
    assert seen_normal > 0, "shape distribution missed normal form"


def test_writer_appends_one_record_per_call():
    """Sequential writes append, each leaving the file 16-byte aligned."""
    rng = random.Random(0xBEEF)
    buf = stdio.BytesIO()
    offsets = []
    for i, (insn_len, block_enc, block_count, token_count) in enumerate(_shape_iter(rng, 25)):
        insn, block, tokens = _make_inputs(
            insn_len, block_enc, block_count, token_count, rng
        )
        before = buf.tell()
        offset, total = write_function_binary_data(
            buf, tokens, block, insn, entry_idx=i
        )
        assert offset == before
        assert buf.tell() == before + total
        assert buf.tell() % RECORD_ALIGNMENT == 0
        offsets.append((offset, buf.tell()))
    # Each record's span is positive and the records do not overlap.
    for (off, end_pos), (next_off, _) in zip(offsets, offsets[1:]):
        assert end_pos == next_off


# ---------------------------------------------------------------------------
# Cap-overflow propagation + error_log + truncate-on-skip
# ---------------------------------------------------------------------------


def test_insn_len_cap_logs_and_truncates():
    """``insn_len >= NORMAL_INSN_CAP`` (= 1<<24) is the smallest insn-len
    overflow; encoder raises ``insn_len_overflow`` and the writer logs +
    truncates back to the pre-call file position."""
    insn = np.zeros(1 << 24, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    buf = stdio.BytesIO()
    buf.write(b"\x11" * 4)  # pre-existing bytes; truncate must preserve
    pre_offset = buf.tell()
    log = stdio.StringIO()

    result = write_function_binary_data(
        buf, tokens, block, insn, entry_idx=0, func_name="big_insn", error_log=log
    )
    assert result is None
    assert buf.tell() == pre_offset
    assert len(buf.getvalue()) == pre_offset

    log_text = log.getvalue()
    assert "insn_len_overflow" in log_text
    assert "big_insn" in log_text


def test_block_word_count_cap_logs_and_truncates():
    """``block_word_count >= NORMAL_BLOCK_WORD_CAP`` (= 65536) is the
    smallest block-word overflow."""
    insn = np.zeros(0, dtype=np.uint8)
    block = np.zeros(1 << 16, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    buf = stdio.BytesIO()
    pre_offset = buf.tell()
    log = stdio.StringIO()

    result = write_function_binary_data(
        buf, tokens, block, insn, entry_idx=0, func_name="big_block", error_log=log
    )
    assert result is None
    assert buf.tell() == pre_offset
    assert "block_word_count_overflow" in log.getvalue()


def test_no_error_log_propagates_skip():
    """Without ``error_log`` the cap-overflow exception bubbles up and
    the file is truncated back to the pre-call position."""
    insn = np.zeros(1 << 24, dtype=np.uint8)
    block = np.zeros(0, dtype=np.uint8)
    tokens = np.zeros(0, dtype=np.uint16)
    buf = stdio.BytesIO()
    buf.write(b"\xaa" * 8)
    pre_offset = buf.tell()
    with pytest.raises(IndexEntrySkip) as info:
        write_function_binary_data(buf, tokens, block, insn, entry_idx=0)
    assert info.value.reason == "insn_len_overflow"
    assert buf.tell() == pre_offset
    assert len(buf.getvalue()) == pre_offset


# ---------------------------------------------------------------------------
# Ultrashort vs normal dispatch happens in encode_binary_header, not in
# the writer. Pin the canonical-form invariant: a tiny record always
# uses ultrashort even though the writer hands the encoder ``Normal``.
# ---------------------------------------------------------------------------


def test_tiny_record_picks_ultrashort_form():
    insn = np.zeros(2, dtype=np.uint8)
    block = np.zeros(3, dtype=np.uint8)  # block_enc=0
    tokens = np.zeros(4, dtype=np.uint16)
    buf = stdio.BytesIO()
    write_function_binary_data(buf, tokens, block, insn, entry_idx=0)
    header, _ = parse_binary_header(buf.getvalue())
    assert header.format is BinaryHeaderFormat.UltraShort


def test_block_enc_one_forces_normal_form():
    """``block_enc=1`` (u16 block words) is ineligible for ultrashort
    regardless of how small the other fields are."""
    insn = np.zeros(2, dtype=np.uint8)
    block = np.zeros(3, dtype=np.uint16)  # block_enc=1
    tokens = np.zeros(4, dtype=np.uint16)
    buf = stdio.BytesIO()
    write_function_binary_data(buf, tokens, block, insn, entry_idx=0)
    header, _ = parse_binary_header(buf.getvalue())
    assert header.format is BinaryHeaderFormat.Normal
    assert header.block_enc == 1
