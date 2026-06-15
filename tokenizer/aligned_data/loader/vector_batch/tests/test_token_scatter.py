"""Unit tests for the fused token scatter (plan C2) on synthetic geometry.

Drives :func:`...vector_batch._scatter._token_scatter.scatter_tokens` on
hand-built geometry + hand-built expanded streams and asserts the
scattered ``tokens[B, L]`` matches a REFERENCE concatenation built at the
SAME columns the geometry assigns -- the scalar twin of the row layout
(variant prefix, then each node's ``expanded[:own_length]`` in BFS order,
the straddler cut at ``partial_cut_length``). Covers the full-row, the
straddler-cut, and the multi-row cases plus the empty edges.

These exercise the novel vectorized scatter math in isolation (no session
/ _data.bin); the corpus-level byte-identity vs ``batch_decode`` lives in
the entry harness.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.vector_batch._scatter._expand import (
    ExpandedBatch,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._token_scatter import (
    scatter_tokens,
)
from tokenizer.aligned_data.loader.vector_batch._types import (
    BatchGeometry,
    BatchRowEmission,
    BatchTokenLayout,
    DenseReservation,
)


def _geometry(
    *,
    row_offsets,
    own_length,
    prefix_len,
    straddler_local_idx,
    partial_cut_length,
    seq_len,
):
    """Assemble a minimal :class:`BatchGeometry` for the scatter test.

    Only the fields the token scatter reads are populated meaningfully;
    the dense reservation is a zero stub (the token scatter ignores it).
    """
    row_offsets = np.asarray(row_offsets, dtype=np.int64)
    own = np.asarray(own_length, dtype=np.int64)
    n_emitted = own.size
    n_rows = row_offsets.size - 1
    total = np.zeros(n_rows, dtype=np.int64)
    emission = BatchRowEmission(
        row_offsets=row_offsets,
        node=np.arange(n_emitted, dtype=np.int64),
        edge_type=np.zeros(n_emitted, dtype=np.uint8),
        own_length=own,
        id_total=np.zeros(n_emitted, dtype=np.int64),
        value_total=np.zeros(n_emitted, dtype=np.int64),
    )
    layout = BatchTokenLayout(
        seq_len=seq_len,
        prefix_len=np.asarray(prefix_len, dtype=np.int64),
        straddler_local_idx=np.asarray(straddler_local_idx, dtype=np.int64),
        partial_cut_length=np.asarray(partial_cut_length, dtype=np.int64),
        total_length=total,
    )
    reservation = DenseReservation(
        id_reserved=np.zeros(n_rows, dtype=np.int64),
        id_offsets=np.zeros(n_rows + 1, dtype=np.int64),
        value_reserved=np.zeros(n_rows, dtype=np.int64),
        value_offsets=np.zeros(n_rows + 1, dtype=np.int64),
    )
    return BatchGeometry(
        n_rows=n_rows,
        emission=emission,
        layout=layout,
        reservation=reservation,
        excluded_pool=np.zeros(0, dtype=np.int64),
        excluded_pool_offsets=np.zeros(n_rows + 1, dtype=np.int64),
    )


def _expanded(per_node):
    """Pack a list of per-node id lists into an :class:`ExpandedBatch`."""
    offsets = np.zeros(len(per_node) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in per_node], out=offsets[1:])
    flat = (
        np.concatenate([np.asarray(p, dtype=np.uint16) for p in per_node])
        if per_node
        else np.zeros(0, dtype=np.uint16)
    )
    return ExpandedBatch(expanded=flat, node_offsets=offsets)


def _reference_row(prefix, nodes, cuts, seq_len):
    """Scalar reference: prefix + per-node ``expanded[:cut]`` capped at L."""
    row = np.zeros(seq_len, dtype=np.uint16)
    col = min(len(prefix), seq_len)
    row[:col] = np.asarray(prefix, dtype=np.uint16)[:col]
    for node, cut in zip(nodes, cuts):
        if col >= seq_len or cut <= 0:
            continue
        take = min(cut, seq_len - col)
        row[col : col + take] = np.asarray(node, dtype=np.uint16)[:take]
        col += take
    return row


def test_full_row_prefix_then_bodies():
    """One row, two whole nodes after a 2-wide prefix."""
    g = _geometry(
        row_offsets=[0, 2],
        own_length=[3, 4],
        prefix_len=[2],
        straddler_local_idx=[-1],
        partial_cut_length=[0],
        seq_len=20,
    )
    bodies = [[101, 102, 103], [201, 202, 203, 204]]
    exp = _expanded(bodies)
    tokens = scatter_tokens(g, exp, np.array([7, 8], np.uint16), np.array([0, 2]))
    ref = _reference_row([7, 8], bodies, [3, 4], 20)
    assert np.array_equal(tokens[0], ref)


def test_straddler_cut_drops_tail():
    """The straddler keeps only ``partial_cut_length`` columns; later
    nodes are dropped."""
    g = _geometry(
        row_offsets=[0, 3],
        own_length=[3, 4, 5],
        prefix_len=[2],
        straddler_local_idx=[1],
        partial_cut_length=[1],
        seq_len=6,
    )
    bodies = [[101, 102, 103], [201, 202, 203, 204], [301, 302, 303, 304, 305]]
    exp = _expanded(bodies)
    tokens = scatter_tokens(g, exp, np.array([7, 8], np.uint16), np.array([0, 2]))
    # prefix(2) + node0(3) + node1 cut to 1 = 6 columns exactly.
    assert tokens[0].tolist() == [7, 8, 101, 102, 103, 201]
    ref = _reference_row([7, 8], bodies, [3, 1, 0], 6)
    assert np.array_equal(tokens[0], ref)


def test_multi_row_independent_columns():
    """Two rows with different prefixes, node sets, and one straddler."""
    # prefix_len (layout body-start) and the prefix-id CSR widths agree:
    # both derive from the same _variants.bin read. Here both rows have a
    # width-1 prefix.
    g = _geometry(
        row_offsets=[0, 2, 3],
        own_length=[2, 3, 4],
        prefix_len=[1, 1],
        straddler_local_idx=[-1, 0],
        partial_cut_length=[0, 2],
        seq_len=8,
    )
    bodies = [[101, 102], [201, 202, 203], [301, 302, 303, 304]]
    exp = _expanded(bodies)
    prefix_tokens = np.array([9, 5], np.uint16)  # row0 width1, row1 width1
    prefix_offsets = np.array([0, 1, 2])
    tokens = scatter_tokens(g, exp, prefix_tokens, prefix_offsets)
    # row0: prefix[9] + node0(2) + node1(3) = 6 cols.
    assert tokens[0].tolist() == [9, 101, 102, 201, 202, 203, 0, 0]
    # row1: prefix[5] + node2 straddler cut to 2 = 3 cols.
    assert tokens[1].tolist() == [5, 301, 302, 0, 0, 0, 0, 0]


def test_prefix_values_match_width_twin_and_shift():
    """``variant_prefix_values`` (the scatter's prefix-id reader) returns
    CSR offsets matching the width twin (``variant_prefix_lengths``) and
    shifts ids by ``- 256`` -- on the synthetic ``_variants.bin`` with
    non-empty per-row prefixes."""
    from tokenizer.aligned_data.loader.vector_batch._prefix import (
        variant_prefix_lengths,
    )
    from tokenizer.aligned_data.loader.vector_batch._scatter._prefix_values import (
        variant_prefix_values,
    )
    from tokenizer.token_manager import VocabularyManager

    from ._synthetic import build_synthetic_corpus

    c = build_synthetic_corpus()
    nodes = np.array([0, 2, 3])  # root_v0 (prefix 1), A (prefix 2), B (prefix 0)
    widths = variant_prefix_lengths(c.variants_u8, c.cols, nodes=nodes)
    values, offsets = variant_prefix_values(
        c.variants_u8, c.cols, nodes=nodes
    )
    assert widths.tolist() == [1, 2, 0]
    assert np.array_equal(np.diff(offsets), widths)
    assert offsets[-1] == int(widths.sum())
    # The synthetic records store raw id 0 in the payload; the shift is
    # ``raw - 256`` (uint16 wrap), so every value is the shifted id 0.
    shift = int(VocabularyManager._V2_RESERVED_DIGIT_COUNT)
    expected = np.full(values.size, (0 - shift) & 0xFFFF, dtype=np.uint16)
    assert np.array_equal(values, expected)


def test_prefix_values_empty_variants_buffer():
    """An ABSENT/empty ``_variants.bin`` yields a zero-width prefix for
    every row (mirrors the session's None-variants behaviour)."""
    from tokenizer.aligned_data.loader.vector_batch._scatter._prefix_values import (
        variant_prefix_values,
    )

    from ._synthetic import build_synthetic_corpus

    c = build_synthetic_corpus()
    values, offsets = variant_prefix_values(
        np.zeros(0, dtype=np.uint8), c.cols, nodes=np.array([0, 2, 3])
    )
    assert values.size == 0
    assert offsets.tolist() == [0, 0, 0, 0]


def test_empty_batch_and_zero_seqlen():
    """Degenerate shapes return the right zero tensors without error."""
    g0 = _geometry(
        row_offsets=[0],
        own_length=[],
        prefix_len=[],
        straddler_local_idx=[],
        partial_cut_length=[],
        seq_len=4,
    )
    t0 = scatter_tokens(g0, _expanded([]), np.zeros(0, np.uint16), np.array([0]))
    assert t0.shape == (0, 4)

    g1 = _geometry(
        row_offsets=[0, 1],
        own_length=[3],
        prefix_len=[0],
        straddler_local_idx=[-1],
        partial_cut_length=[0],
        seq_len=0,
    )
    t1 = scatter_tokens(g1, _expanded([[1, 2, 3]]), np.zeros(0, np.uint16), np.array([0, 0]))
    assert t1.shape == (1, 0)
