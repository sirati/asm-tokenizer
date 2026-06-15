"""Shared-record dedup for the geometry triple: measured once, gathered N.

The geometry compute reuses the SAME dedup loop the length compute does,
but accumulates the full ``(body_len, id_count, value_count)`` triple per
distinct record (the hashmap stores a row-index, not a scalar length).
This pins that a record referenced by many variants is scanned by the
bulk geometry engine exactly once and every reference gathers the same
triple, across chunk boundaries, with the overflow guard firing per axis.
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
    realized_geometry_for_offsets,
)


def _lay_data_bin(tmp_path, token_lists):
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


def _counting_geometry(monkeypatch):
    measured = {"total": 0}
    real_bulk = _compute.bulk_contributing_geometry

    def counting(data, starts, counts):
        measured["total"] += int(np.asarray(starts).size)
        return real_bulk(data, starts, counts)

    monkeypatch.setattr(_compute, "bulk_contributing_geometry", counting)
    return measured


def test_shared_record_triple_measured_once(tmp_path, monkeypatch) -> None:
    # Three distinct records (all ids > 256 -> body == token count, zero
    # identity/number carriers).
    data_u8, offsets = _lay_data_bin(
        tmp_path,
        [
            [300, 301, 302, 303],
            [400, 401, 402, 403, 404, 405],
            [500, 501, 502],
        ],
    )
    o0, o1, o2 = offsets
    var_offsets = np.array([o0, o0, o1, o0, o2, o1, o0, o1, o0], dtype=np.int64)
    measured = _counting_geometry(monkeypatch)

    geom = realized_geometry_for_offsets(data_u8, var_offsets)
    expected_body = np.array([4, 4, 6, 4, 3, 6, 4, 6, 4], dtype=np.uint32)
    np.testing.assert_array_equal(geom.body_len, expected_body)
    # These records carry no identity/number tokens.
    np.testing.assert_array_equal(geom.id_count, np.zeros(9, dtype=np.uint32))
    np.testing.assert_array_equal(geom.value_count, np.zeros(9, dtype=np.uint32))
    # Exactly three records scanned -- one per distinct offset.
    assert measured["total"] == 3


def test_dedup_across_chunks_still_measures_once(tmp_path, monkeypatch) -> None:
    data_u8, offsets = _lay_data_bin(
        tmp_path, [[300, 301, 302, 303, 304], [400, 401]]
    )
    o0, o1 = offsets
    var_offsets = np.array([o0, o1, o0, o0, o1, o0], dtype=np.int64)
    measured = _counting_geometry(monkeypatch)
    monkeypatch.setattr(_compute, "_CHUNK_OFFSETS", 2)

    geom = realized_geometry_for_offsets(data_u8, var_offsets)
    np.testing.assert_array_equal(
        geom.body_len, np.array([5, 2, 5, 5, 2, 5], dtype=np.uint32)
    )
    assert measured["total"] == 2


def test_new_uniques_in_multiple_chunks_keep_row_indices(
    tmp_path, monkeypatch
) -> None:
    # Each chunk introduces a DISTINCT new unique record, so the running
    # row-index base must advance by the unique COUNT, not the block count
    # -- pins the cross-chunk row-index accounting (a block-count base
    # would alias chunk 2's record back onto chunk 1's row).
    data_u8, offsets = _lay_data_bin(
        tmp_path,
        [
            [300, 301, 302, 303],          # body 4
            [400, 401, 402, 403, 404],     # body 5
            [500, 501, 502, 503, 504, 505],  # body 6
            [600, 601, 602],               # body 3
        ],
    )
    o0, o1, o2, o3 = offsets
    # Chunk size 2 -> 3 chunks, each FIRST referencing a brand-new record:
    # [o0,o1] | [o2,o0] | [o3,o2]. Every chunk has a miss.
    var_offsets = np.array([o0, o1, o2, o0, o3, o2], dtype=np.int64)
    _counting_geometry(monkeypatch)
    monkeypatch.setattr(_compute, "_CHUNK_OFFSETS", 2)

    geom = realized_geometry_for_offsets(data_u8, var_offsets)
    np.testing.assert_array_equal(
        geom.body_len, np.array([4, 5, 6, 4, 3, 6], dtype=np.uint32)
    )


def test_empty_offsets_returns_empty(tmp_path) -> None:
    data_u8, _offsets = _lay_data_bin(tmp_path, [[300, 301]])
    geom = realized_geometry_for_offsets(data_u8, np.zeros(0, dtype=np.int64))
    for axis in (geom.body_len, geom.id_count, geom.value_count):
        assert axis.dtype == np.uint32 and axis.size == 0


def _force_axis(monkeypatch, *, body=0, ids=0, values=0) -> None:
    """Make the bulk engine report fixed per-axis values for every record."""
    from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
        ContributingGeometry,
    )

    def fake_bulk(data, starts, counts):
        n = np.asarray(starts).size
        return ContributingGeometry(
            body_len=np.full(n, body, dtype=np.int64),
            id_count=np.full(n, ids, dtype=np.int64),
            value_chunk_count=np.full(n, values, dtype=np.int64),
        )

    monkeypatch.setattr(_compute, "bulk_contributing_geometry", fake_bulk)


def test_max_storable_value_accepted(tmp_path, monkeypatch) -> None:
    data_u8, offsets = _lay_data_bin(tmp_path, [[300, 301, 302]])
    _force_axis(monkeypatch, body=0xFFFFFFFE, ids=0xFFFFFFFE, values=0xFFFFFFFE)
    geom = realized_geometry_for_offsets(data_u8, np.array(offsets, dtype=np.int64))
    assert int(geom.body_len[0]) == 0xFFFFFFFE
    assert int(geom.id_count[0]) == 0xFFFFFFFE
    assert int(geom.value_count[0]) == 0xFFFFFFFE


@pytest.mark.parametrize("axis", ["body", "ids", "values"])
def test_sentinel_value_raises_overflow_per_axis(tmp_path, monkeypatch, axis) -> None:
    # The reserved miss sentinel (0xFFFFFFFF) must hard-error from ANY axis.
    data_u8, offsets = _lay_data_bin(tmp_path, [[300, 301, 302]])
    _force_axis(monkeypatch, **{axis: 0xFFFFFFFF})
    with pytest.raises(OverflowError):
        realized_geometry_for_offsets(data_u8, np.array(offsets, dtype=np.int64))
