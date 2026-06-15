"""Per-node token-stream expansion over the batched body gather.

Single concern: turn each emitted node's gathered raw u16 token stream
(:mod:`._body_load`) + its edge :class:`CallTargetType` into the
post-promotion / post-strip / post-shift ``expanded_token_ids`` stream
the token scatter places into ``[B, L]`` -- flattened across the batch
into one CSR-indexed array.

REUSE, NOT RE-IMPLEMENTATION (byte-identity contract): the expansion
SEMANTICS (VC2 / F128 promotion, strip + shift, the prepended calling-
category self-token at ``expanded[0]``) are owned by
:func:`...batch_decode._expand_tokens.expand_tokens`; the per-stream
mask pre-compute by :func:`...decoded._inline_decode_state.
build_inline_decode_state`; the ``CallTargetType -> Category`` collapse
by :data:`...batch_decode._dedup_walk._constants.
_CALL_TARGET_TYPE_TO_CATEGORY`. This module ONLY drives those owned
kernels from the geometry's flat ``node`` / ``edge_type`` axes and
concatenates their outputs -- it re-implements none of the rules, so the
byte-identity gate cannot diverge on expansion logic.

Remaining hot-path loop: the per-node ``expand_tokens`` dispatch is still
a Python loop over emitted nodes (the kernel is per-stream). The body
LOAD is already batched (one gather); the per-node expand is the next
vectorization target (a batched promotion+strip twin over the flat CSR
``raw``). Until then the loop reuses the proven scalar semantics so
correctness is never traded for speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CALL_TARGET_TYPE_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    build_inline_decode_state,
)

from ._body_load import GatheredBodies


__all__ = ["ExpandedBatch", "expand_node_bodies"]


#: The unified vocab is the only layout the carrier-band pre-compute is
#: valid under (see ``build_inline_decode_state``); the geometry path is
#: unified-vocab-only by construction.
_UNIFIED_VOCAB_FORMAT_VERSION = 1


@dataclass(frozen=True)
class _ExpandShim:
    """Minimal ``expand_tokens`` input: only ``state`` + category are read.

    ``expand_tokens`` consumes a ``Stage1CallTarget`` but touches ONLY
    its :attr:`state` (the :class:`InlineDecodeState`) and
    :attr:`encounter_category`. This shim exposes exactly those two so
    the owned kernel runs unchanged on geometry-driven inputs -- no
    dependency on the full Stage-1 hierarchy.
    """

    state: object
    encounter_category: object


@dataclass(frozen=True)
class ExpandedBatch:
    """Every emitted node's expanded token stream, flattened + CSR.

    ``expanded`` is the concatenation, in emission order, of each node's
    ``expanded_token_ids`` (slot 0 = the calling-category self-token,
    slots 1+ = post-promotion body). ``node_offsets`` is the CSR jump
    table: node ``i`` owns ``expanded[node_offsets[i] : node_offsets[i +
    1]]``. The per-node length equals the geometry ``own_length`` (=
    ``1 + realized body_len``) -- the equivalence anchor the token
    scatter relies on.
    """

    expanded: np.ndarray  # uint16[total_expanded]
    node_offsets: np.ndarray  # int64[n_nodes + 1] -- CSR into ``expanded``


def expand_node_bodies(
    bodies: GatheredBodies,
    edge_types: np.ndarray,
) -> ExpandedBatch:
    """Expand every gathered node body, flattened across the batch.

    Parameters
    ----------
    bodies:
        The batched body gather (:class:`._body_load.GatheredBodies`):
        flat ``raw`` u16 streams + the per-node CSR ``record_offsets``.
    edge_types:
        ``uint8[n_nodes]`` the :class:`CallTargetType` of the edge that
        reached each emitted node (the geometry ``BatchRowEmission.
        edge_type`` axis), parallel to the nodes ``bodies`` was gathered
        for. Maps to the self-token calling category.

    Returns
    -------
    ExpandedBatch
        The flat ``expanded`` u16 stream + CSR ``node_offsets``.
    """
    raw = bodies.raw
    rec = np.asarray(bodies.record_offsets, dtype=np.int64).reshape(-1)
    types = np.asarray(edge_types, dtype=np.uint8).reshape(-1)
    n_nodes = rec.size - 1
    if types.size != n_nodes:
        raise ValueError(
            f"edge_types has {types.size} entries but the gather covers "
            f"{n_nodes} nodes"
        )

    node_offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    if n_nodes == 0:
        return ExpandedBatch(
            expanded=np.zeros(0, dtype=np.uint16), node_offsets=node_offsets
        )

    pieces: list[np.ndarray] = []
    lengths = np.empty(n_nodes, dtype=np.int64)
    for i in range(n_nodes):
        raw_tokens = raw[rec[i] : rec[i + 1]]
        state = build_inline_decode_state(
            raw_tokens, format_version=_UNIFIED_VOCAB_FORMAT_VERSION
        )
        category = _CALL_TARGET_TYPE_TO_CATEGORY[CallTargetType(int(types[i]))]
        expanded = expand_tokens(
            _ExpandShim(state=state, encounter_category=category)
        ).expanded_token_ids
        pieces.append(expanded)
        lengths[i] = expanded.shape[0]

    np.cumsum(lengths, out=node_offsets[1:])
    expanded_flat = (
        np.concatenate(pieces)
        if pieces
        else np.zeros(0, dtype=np.uint16)
    )
    return ExpandedBatch(
        expanded=expanded_flat.astype(np.uint16, copy=False),
        node_offsets=node_offsets,
    )
