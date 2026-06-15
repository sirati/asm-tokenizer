"""Batched single-gather load of every emitted node's raw token stream.

Single concern: given the per-node ``(token_start, token_count)`` spans
(:mod:`._locator`), gather EVERY emitted node's raw u16 token region out
of ``_data.bin`` in ONE vectorized pass into a flat ``raw`` array with a
CSR jump table -- replacing the current path's per-edge, per-body
re-read + re-parse with a single batched gather.

The gather is the bulk twin of
:func:`...batch_decode._bulk_expand_lengths._scan_chunk`'s token gather:
token regions are u16-LE at even (16-byte-aligned record tail) byte
offsets, so a single ``data_u8.view(uint16)`` index at ``(start >> 1) +
within`` reads every record's stream with no byte-pair OR. The per-token
``record_of`` / ``within`` index columns are built with the classic
cumulative-offset ``arange`` construction (no Python per-record loop).

This module reads bytes but does NOT decode them (no run-length, no
promotion, no band logic) -- it owns ONLY the gather. The expand twin
(:mod:`._expand`) consumes the flat ``raw`` + CSR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["GatheredBodies", "gather_node_bodies"]


@dataclass(frozen=True)
class GatheredBodies:
    """Every emitted node's raw u16 token stream, flattened + CSR-indexed.

    ``raw`` is the concatenation of each node's record token region in
    emission order; ``record_offsets`` is the CSR jump table tying each
    node to its slice (node ``i`` owns ``raw[record_offsets[i] :
    record_offsets[i + 1]]``). ``[0] == 0``, ``[-1] == raw.size``.
    """

    raw: np.ndarray  # uint16[total_tokens] -- gathered raw token streams
    record_offsets: np.ndarray  # int64[n_nodes + 1] -- CSR into ``raw``


def gather_node_bodies(
    data_u8: np.ndarray,
    token_starts: np.ndarray,
    token_counts: np.ndarray,
) -> GatheredBodies:
    """Gather all nodes' raw token streams in one vectorized pass.

    Parameters
    ----------
    data_u8:
        The arm's ``_data.bin`` as a 1-D ``uint8`` array (read-only
        memmap).
    token_starts / token_counts:
        Parallel ``int[n_nodes]`` arrays from :func:`._locator.
        node_token_spans` -- the even byte offset + u16 count of each
        node's record token region.

    Returns
    -------
    GatheredBodies
        The flat ``raw`` u16 stream + the CSR ``record_offsets``.
    """
    starts = np.asarray(token_starts, dtype=np.int64).reshape(-1)
    counts = np.asarray(token_counts, dtype=np.int64).reshape(-1)
    if starts.shape != counts.shape:
        raise ValueError(
            "token_starts and token_counts must be parallel; got "
            f"{starts.shape} vs {counts.shape}"
        )
    n_nodes = starts.size
    record_offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    if n_nodes == 0:
        return GatheredBodies(
            raw=np.zeros(0, dtype=np.uint16), record_offsets=record_offsets
        )

    # u16-aligned record-tail offsets: the same evenness invariant the
    # bulk geometry scan validates. An odd offset is a corrupt locator,
    # not a silent mis-gather.
    if bool((starts & 1).any()):
        raise ValueError(
            "gather_node_bodies: token_starts must be even (u16-aligned "
            "record-tail offsets); got an odd offset"
        )

    np.cumsum(counts, out=record_offsets[1:])
    total = int(record_offsets[-1])
    if total == 0:
        return GatheredBodies(
            raw=np.zeros(0, dtype=np.uint16), record_offsets=record_offsets
        )

    # Per-gathered-token record id + within-record offset, built with the
    # cumulative-offset arange (no Python per-record loop). ``record_of``
    # fits int32 (n_nodes <= total <= the gather budget); the absolute
    # word index stays int64 (multi-GB corpora exceed 2**31).
    rec_starts = record_offsets[:-1]
    record_of = np.repeat(np.arange(n_nodes, dtype=np.int64), counts)
    within = np.arange(total, dtype=np.int64) - rec_starts[record_of]

    data_u16 = data_u8.view(np.uint16)
    word_idx = (starts >> 1)[record_of] + within
    raw = data_u16[word_idx].astype(np.uint16, copy=True)

    return GatheredBodies(raw=raw, record_offsets=record_offsets)
