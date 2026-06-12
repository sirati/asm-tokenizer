"""Shared-record dedup: each distinct record is measured exactly once.

Many variants share one identical tokenization -> one shared
``_data.bin`` record (write-time dedup in memmap_builder), so the same
record offset appears for multiple variants in the catalog. The
realized-length compute must measure each DISTINCT offset's body length
once (via the build-time hashmap), no matter how many variants reference
it.

Pinned at the compute seam (:func:`.._compute.realized_lengths_for_offsets`):
a repeated offset is exactly what a shared record produces, so feeding an
offsets array with repeats reproduces the shared-record case faithfully.
A counting wrapper around the bulk length engine asserts the engine is
invoked on each unique offset exactly once; an integration assertion
confirms every reference to a shared record gets the same length.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.memmap_format import (
    encode_data_bin_prelude,
    encode_data_bin_trailer,
)
from tokenizer.aligned_data._writers import write_function_binary_data
from tokenizer.aligned_data.realized_lengths import _compute
from tokenizer.aligned_data.realized_lengths._compute import (
    realized_lengths_for_offsets,
)


def _lay_data_bin(tmp_path, token_lists):
    """Write a real ``_data.bin`` from ``token_lists``; return (data_u8, offsets)."""
    path = tmp_path / "bin_data.bin"
    offsets = []
    with open(path, "wb") as fh:
        fh.write(encode_data_bin_prelude())
        for n, toks in enumerate(token_lists):
            tokens = np.asarray(toks, dtype=np.uint16)
            block_rl = np.array([tokens.size], dtype=np.uint8)
            insn_rl = np.array(
                [2, tokens.size - 2 if tokens.size > 2 else 1], dtype=np.uint8
            )
            offset, _length = write_function_binary_data(
                fh, tokens, block_rl, insn_rl, entry_idx=n
            )
            offsets.append(offset)
        fh.write(encode_data_bin_trailer(len(token_lists), cursor=fh.tell()))
    data_u8 = np.fromfile(path, dtype=np.uint8)
    return data_u8, offsets


def test_shared_record_measured_once(tmp_path, monkeypatch) -> None:
    # Three distinct records (token ids all > 256 so body == token count).
    data_u8, offsets = _lay_data_bin(
        tmp_path,
        [
            [300, 301, 302, 303],          # body 4
            [400, 401, 402, 403, 404, 405],  # body 6
            [500, 501, 502],               # body 3
        ],
    )
    o0, o1, o2 = offsets

    # Variant offset column: record 0 referenced by FIVE variants, record
    # 1 by THREE, record 2 by ONE -- nine variants, three unique records.
    var_offsets = np.array(
        [o0, o0, o1, o0, o2, o1, o0, o1, o0], dtype=np.int64
    )

    # Count the unique offsets handed to the bulk length engine.
    measured = {"total": 0}
    real_bulk = _compute.bulk_contributing_body_lengths

    def counting_bulk(data, starts, counts):
        measured["total"] += int(np.asarray(starts).size)
        return real_bulk(data, starts, counts)

    monkeypatch.setattr(_compute, "bulk_contributing_body_lengths", counting_bulk)

    lengths = realized_lengths_for_offsets(data_u8, var_offsets)

    # Each variant gets the right shared length.
    expected = np.array([4, 4, 6, 4, 3, 6, 4, 6, 4], dtype=np.uint32)
    np.testing.assert_array_equal(lengths, expected)

    # Exactly three records measured -- one per distinct offset, never
    # once per variant reference.
    assert measured["total"] == 3


def test_dedup_across_chunks_still_measures_once(
    tmp_path, monkeypatch
) -> None:
    # Shrink the chunk so a shared offset straddles chunk boundaries; the
    # hashmap (persisted across chunks) must still measure it once.
    data_u8, offsets = _lay_data_bin(
        tmp_path, [[300, 301, 302, 303, 304], [400, 401]]
    )
    o0, o1 = offsets
    var_offsets = np.array([o0, o1, o0, o0, o1, o0], dtype=np.int64)

    measured = {"total": 0}
    real_bulk = _compute.bulk_contributing_body_lengths

    def counting_bulk(data, starts, counts):
        measured["total"] += int(np.asarray(starts).size)
        return real_bulk(data, starts, counts)

    monkeypatch.setattr(_compute, "bulk_contributing_body_lengths", counting_bulk)
    monkeypatch.setattr(_compute, "_CHUNK_OFFSETS", 2)

    lengths = realized_lengths_for_offsets(data_u8, var_offsets)
    np.testing.assert_array_equal(
        lengths, np.array([5, 2, 5, 5, 2, 5], dtype=np.uint32)
    )
    assert measured["total"] == 2


def test_empty_offsets_returns_empty(tmp_path) -> None:
    data_u8, _offsets = _lay_data_bin(tmp_path, [[300, 301]])
    out = realized_lengths_for_offsets(data_u8, np.zeros(0, dtype=np.int64))
    assert out.dtype == np.uint32 and out.size == 0
