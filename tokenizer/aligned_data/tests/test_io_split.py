"""Byte-identical equivalence between the path-based and handle-based
function-record readers in ``tokenizer.aligned_data.io``.

The split exists so a session can mmap ``_data.bin`` once and slice
many records out of it (avoiding per-call ``np.memmap`` syscalls on
the dataloader hot path). The wrapper preserves the legacy single-shot
API. This test pins down that the split is behaviour-preserving:
every byte of every output array must match between the two forms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.io import (
    parse_function_data_memmap,
    read_function_data_memmap,
    write_function_binary_data,
)

def _make_record(rng: np.random.Generator, n_tokens: int, n_blocks: int, n_insns: int):
    """Build one synthetic function record's three arrays.

    Sizes vary across records so the test exercises distinct record
    geometries (token counts spanning every header width tag) and
    distinct block-encoding widths (the encoder picks uint8/uint16/uint32
    based on the largest block runlength).
    """
    tokens = rng.integers(0, 2**16, size=n_tokens, dtype=np.uint16)
    block_runlength = rng.integers(1, 200, size=n_blocks, dtype=np.uint8)
    insn_runlength = rng.integers(1, 200, size=n_insns, dtype=np.uint8)
    return tokens, block_runlength, insn_runlength


def _write_synthetic_data_bin(path: Path):
    """Append several records to a fresh ``_data.bin`` and return the
    list of ``(offset, tokens, block_rl, insn_rl)`` tuples for each
    record so the test can read them back and compare. Records are
    self-describing on disk -- the reader derives the total byte
    count from the parsed header, so callers do not track length.
    """
    rng = np.random.default_rng(seed=0xA51CED)
    record_specs = [
        # n_tokens, n_blocks, n_insns — varied so output sizes differ.
        (8, 3, 5),
        (64, 12, 20),
        (1, 1, 1),
        (200, 50, 75),
    ]
    written = []
    with open(path, "wb") as f:
        for i, (n_tok, n_blk, n_ins) in enumerate(record_specs):
            tokens, block_rl, insn_rl = _make_record(rng, n_tok, n_blk, n_ins)
            off, _total = write_function_binary_data(
                f, tokens, block_rl, insn_rl, entry_idx=i
            )
            written.append((off, tokens, block_rl, insn_rl))
    return written


def test_parse_matches_read_byte_for_byte(tmp_path):
    """For every record in a multi-record bin, both readers must
    produce arrays whose dtype, shape, and bytes are identical."""
    data_bin = tmp_path / "synthetic_data.bin"
    records = _write_synthetic_data_bin(data_bin)

    # Open one memmap of the whole file — exactly what
    # ``BinarySession`` will do once per session.
    whole_mmap = np.memmap(data_bin, dtype=np.uint8, mode="r")

    for offset, tokens_in, block_in, insn_in in records:
        path_form = read_function_data_memmap(str(data_bin), offset)
        handle_form = parse_function_data_memmap(whole_mmap, offset)

        insn_p, block_p, tok_p = path_form
        insn_h, block_h, tok_h = handle_form

        # dtype and shape parity per array.
        assert insn_p.dtype == insn_h.dtype == np.uint8
        assert block_p.dtype == block_h.dtype
        assert tok_p.dtype == tok_h.dtype == np.uint16
        assert insn_p.shape == insn_h.shape
        assert block_p.shape == block_h.shape
        assert tok_p.shape == tok_h.shape

        # Byte-for-byte parity — the load-bearing guarantee.
        assert insn_p.tobytes() == insn_h.tobytes()
        assert block_p.tobytes() == block_h.tobytes()
        assert tok_p.tobytes() == tok_h.tobytes()

        # And both must match what the writer fed in.
        assert insn_h.tobytes() == insn_in.tobytes()
        assert block_h.tobytes() == block_in.astype(block_h.dtype).tobytes()
        assert tok_h.tobytes() == tokens_in.tobytes()


def test_parse_with_ndarray_view(tmp_path):
    """``parse_function_data_memmap`` is documented to accept any
    1-D uint8 array view, not just ``np.memmap``. Confirm an in-memory
    ndarray of the same bytes gives identical results — this matters
    because tests and small in-memory paths shouldn't require a real
    file to exercise the parser.
    """
    data_bin = tmp_path / "synthetic_data.bin"
    records = _write_synthetic_data_bin(data_bin)

    in_memory = np.frombuffer(data_bin.read_bytes(), dtype=np.uint8)

    for offset, _, _, _ in records:
        whole_mmap = np.memmap(data_bin, dtype=np.uint8, mode="r")
        insn_m, block_m, tok_m = parse_function_data_memmap(whole_mmap, offset)
        insn_n, block_n, tok_n = parse_function_data_memmap(in_memory, offset)

        assert insn_m.tobytes() == insn_n.tobytes()
        assert block_m.tobytes() == block_n.tobytes()
        assert tok_m.tobytes() == tok_n.tobytes()
