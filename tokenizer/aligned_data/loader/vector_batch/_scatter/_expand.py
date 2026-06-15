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
_CALL_TARGET_TYPE_TO_CATEGORY`. The expansion + state MATH now runs as a
few vectorized passes over the WHOLE flat CSR body stream
(:mod:`._batched_expand`, the batched twin of
:func:`...batch_decode._bulk_expand_lengths.bulk_contributing_geometry`);
this module ONLY collapses the edge axis to per-node self-token ids and
SLICES the batched result into the per-node ``states`` / mask list
contract the dense pass reads. No rule is re-implemented here or in the
batched twin -- both reproduce the owned kernels' rules, so the
byte-identity gate cannot diverge on expansion logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CALL_TARGET_TYPE_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    InlineDecodeState,
)
from tokenizer.tokens import Category
from tokenizer.token_manager import VocabularyManager

from ._batched_expand import BatchedExpansion, batched_expand
from ._body_load import GatheredBodies


__all__ = ["ExpandedBatch", "expand_node_bodies"]


# Self-token shifted ids -- LOCAL_FUNC is identity-block index 1, PLT_FUNC
# index 2 (the SAME anchors ``_expand_tokens`` derives); the model-facing
# value the scalar writes at ``expanded[0]`` is ``id - 256``. Building the
# CallTargetType -> shifted-id lookup once here keeps the edge-axis
# collapse a single vectorized gather (no per-node category branch).
_LOCAL_FUNC_SHIFTED = (
    VocabularyManager._V2_IDENTITY_BLOCK_START
    + 1
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)
_PLT_FUNC_SHIFTED = (
    VocabularyManager._V2_IDENTITY_BLOCK_START
    + 2
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)

#: ``CallTargetType -> shifted self-token id`` lookup table. Indexed by
#: the integer ``CallTargetType`` value; entries map through the SAME
#: ``_CALL_TARGET_TYPE_TO_CATEGORY`` collapse + the scalar
#: ``_calling_category_shifted_id`` rule (LOCAL_FUNC -> shifted id, PLT_FUNC
#: -> shifted id). Categories the scalar rejects (anything but LOCAL/PLT)
#: stay at the sentinel so a stage-1 walker bug surfaces as a raised error,
#: matching ``expand_tokens``'s AssertionError.
_SELF_TOKEN_SENTINEL = np.iinfo(np.uint32).max
_CATEGORY_TO_SHIFTED_ID = {
    Category.LOCAL_FUNC: _LOCAL_FUNC_SHIFTED,
    Category.PLT_FUNC: _PLT_FUNC_SHIFTED,
}


def _build_self_token_lut() -> np.ndarray:
    """``uint32`` lookup: ``CallTargetType`` int value -> shifted id.

    Reuses the OWNED ``CallTargetType -> Category`` collapse + the scalar
    ``_calling_category_shifted_id`` rule so the edge-axis-to-self-token
    map cannot drift from ``expand_tokens``. Types whose category is not
    an inlined call category hold the sentinel.
    """
    max_type = max(int(t) for t in CallTargetType) + 1
    lut = np.full(max_type, _SELF_TOKEN_SENTINEL, dtype=np.uint32)
    for ct, category in _CALL_TARGET_TYPE_TO_CATEGORY.items():
        shifted = _CATEGORY_TO_SHIFTED_ID.get(category)
        if shifted is not None:
            lut[int(ct)] = shifted
    return lut


_SELF_TOKEN_LUT = _build_self_token_lut()


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

    states: list = field(default_factory=list)
    """Per-emitted-node :class:`~tokenizer.aligned_data.loader.decoded.
    _inline_decode_state.InlineDecodeState`, parallel to
    ``geometry.emission.node``. Built once here (over the gathered raw
    body); the dense pass (:mod:`._dense`) reads ``raw_tokens`` /
    ``runlen_number`` / ``digit_cumsum`` / ``real_mask`` /
    ``is_negative_per_position`` off it instead of re-parsing the body
    (no re-parse in the call chain). Defaults empty for the token-only
    test constructors that do not exercise the dense pass."""

    extra_value_v2_masks: list = field(default_factory=list)
    """Per-node VC2 promotion mask from ``expand_tokens`` (slot 0 =
    prepend, always False). The dense number-decode kernel reads it to
    skip painted VC2 continuation slots."""

    extra_f128_masks: list = field(default_factory=list)
    """Per-node F128 promotion mask from ``expand_tokens`` (slot 0 =
    prepend, always False). The dense number-decode kernel reads it to
    skip painted F128 continuation slots + detect finite F128 sources."""


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

    if n_nodes == 0:
        return ExpandedBatch(
            expanded=np.zeros(0, dtype=np.uint16),
            node_offsets=np.zeros(1, dtype=np.int64),
            states=[],
            extra_value_v2_masks=[],
            extra_f128_masks=[],
        )

    # Collapse the edge axis to per-node shifted self-token ids via the
    # owned ``CallTargetType -> Category -> shifted id`` lookup; a category
    # the scalar ``expand_tokens`` rejects (not LOCAL/PLT) stays at the
    # sentinel, surfacing a stage-1 walker bug exactly as the scalar's
    # AssertionError would.
    self_token_ids = _SELF_TOKEN_LUT[types]
    if bool((self_token_ids == _SELF_TOKEN_SENTINEL).any()):
        bad = int(np.nonzero(self_token_ids == _SELF_TOKEN_SENTINEL)[0][0])
        raise AssertionError(
            "expand received an edge whose CallTargetType maps to a "
            "non-inlined category (only LOCAL_FUNC and PLT_FUNC are "
            f"inlined per plan D3); offending node {bad}, "
            f"CallTargetType={CallTargetType(int(types[bad]))!r}."
        )

    batched = batched_expand(raw, rec, self_token_ids)
    states, extra_value_v2_masks, extra_f128_masks = _slice_per_node(
        batched, np.asarray(raw, dtype=np.uint16).reshape(-1), rec
    )
    return ExpandedBatch(
        expanded=batched.expanded,
        node_offsets=batched.node_offsets,
        states=states,
        extra_value_v2_masks=extra_value_v2_masks,
        extra_f128_masks=extra_f128_masks,
    )


def _slice_per_node(
    batched: BatchedExpansion, raw: np.ndarray, rec: np.ndarray
) -> tuple[list, list, list]:
    """Slice the batched arrays into the per-node list contract.

    The dense pass (:mod:`._dense_adapter`) reads per-node
    :class:`InlineDecodeState` objects + the per-node expanded-space
    promotion masks. Each is a contiguous VIEW into the batched arrays
    (no per-node ``run_lengths`` / cumsum dispatch -- the batched twin
    computed them once): the raw-space fields slice by the body CSR
    ``rec``; the masks slice by the expanded CSR ``node_offsets``; the
    ``digit_cumsum`` block for node ``i`` spans ``rec[i] + i`` ..
    ``rec[i + 1] + (i + 1)`` (the per-node ``N + 1`` layout the batched
    twin packs)."""
    n_nodes = rec.size - 1
    node_off = batched.node_offsets
    states: list = []
    extra_value_v2_masks: list = []
    extra_f128_masks: list = []
    for i in range(n_nodes):
        lo = int(rec[i])
        hi = int(rec[i + 1])
        dc_lo = lo + i
        dc_hi = hi + (i + 1)
        states.append(
            InlineDecodeState(
                raw_tokens=raw[lo:hi],
                real_mask=batched.real_mask[lo:hi],
                number_mask=batched.number_mask[lo:hi],
                runlen_number=batched.runlen_number[lo:hi],
                runlen_value=batched.runlen_value[lo:hi],
                carries_inline_mask=batched.carries_inline_mask[lo:hi],
                is_negative_per_position=batched.is_negative_per_position[
                    lo:hi
                ],
                digit_cumsum=batched.digit_cumsum[dc_lo:dc_hi],
            )
        )
        eo_lo = int(node_off[i])
        eo_hi = int(node_off[i + 1])
        extra_value_v2_masks.append(batched.extra_value_v2_mask[eo_lo:eo_hi])
        extra_f128_masks.append(batched.extra_f128_mask[eo_lo:eo_hi])
    return states, extra_value_v2_masks, extra_f128_masks
