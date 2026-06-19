"""Equivalence: the batched expand twin vs the per-node scalar expand.

THE batched-expand correctness pin. Drives
:func:`...vector_batch._scatter._batched_expand.batched_expand` AND the
old per-node path (``build_inline_decode_state`` + ``expand_tokens``) on
the SAME synthetic flat body stream and asserts byte-identity of every
output: the flat ``expanded`` stream + CSR ``node_offsets``, the
expanded-space promotion masks, AND every per-node
:class:`InlineDecodeState` field. The corpus byte-identity gate
(``test_byte_identity.py``) pins the integrated path; this pins the
kernel in isolation on adversarial geometry the corpus may not hit
(zero-payload VC2, finite + NaN/Inf F128, sign markers, empty survivors,
back-to-back carriers).
"""

from __future__ import annotations

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
from tokenizer.aligned_data.loader.vector_batch._scatter._batched_expand import (
    batched_expand,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._expand import (
    _SELF_TOKEN_LUT,
    expand_node_bodies,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._body_load import (
    GatheredBodies,
)
from tokenizer.token_manager import VocabularyManager


_VC2 = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_F128 = _VC2 + VocabularyManager._V2_NUMBER_BLOCK_COUNT - 1  # 263
_SIGN = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID  # 256
_IDENT = VocabularyManager._V2_IDENTITY_BLOCK_START  # 264 (BLOCK_V2)
_LOCAL_EDGE = int(CallTargetType.LOCAL)
_PLT_EDGE = int(CallTargetType.PLT)


class _Shim:
    def __init__(self, state, category):
        self.state = state
        self.encounter_category = category


def _scalar_per_node(raw, rec, edge_types):
    """Per-node scalar reference: states + masks + flat expanded + CSR."""
    n = rec.size - 1
    pieces, states, vc2_masks, f128_masks = [], [], [], []
    lengths = np.empty(n, dtype=np.int64)
    for i in range(n):
        body = raw[rec[i] : rec[i + 1]]
        state = build_inline_decode_state(body, format_version=1)
        cat = _CALL_TARGET_TYPE_TO_CATEGORY[CallTargetType(int(edge_types[i]))]
        res = expand_tokens(_Shim(state, cat))
        pieces.append(res.expanded_token_ids)
        states.append(state)
        vc2_masks.append(res.extra_value_v2_mask)
        f128_masks.append(res.extra_f128_mask)
        lengths[i] = res.expanded_token_ids.shape[0]
    node_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=node_off[1:])
    flat = (
        np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)
    )
    return flat, node_off, states, vc2_masks, f128_masks


def _f128_payload(high_u16):
    """16 inline-digit slots (big-endian bytes) for an F128 payload whose
    high u16 is ``high_u16``; the rest zero."""
    out = [0] * 16
    out[0] = (high_u16 >> 8) & 0xFF
    out[1] = high_u16 & 0xFF
    return out


def _fixture_bodies():
    """A list of per-node raw bodies covering the adversarial cases."""
    bodies = [
        # plain identity carriers, no payload
        [_IDENT, _IDENT + 1],
        # VC2 with a 10-digit payload (2 chunks -> 1 painted slot)
        [_VC2] + list(range(10)),
        # VC2 with zero-payload edge (carrier then a real token) -> 1 chunk
        [_VC2, _IDENT],
        # negative number: carrier, sign marker, digits
        [_VC2, _SIGN, 1, 2, 3],
        # finite F128 (high u16 != 0x7fff) -> 1 painted slot
        [_F128] + _f128_payload(0x3FFF),
        # NaN/Inf F128 (high u16 == 0x7fff) -> no painted slot
        [_F128] + _f128_payload(0x7FFF),
        # back-to-back carriers + a trailing identity
        [_VC2, 5, _F128] + _f128_payload(0x4000) + [_IDENT],
        # identity carrier then a negative VC2 (sign marker) + digits;
        # a valid body always opens with a real carrier (run_lengths
        # requires position 0 to be non-digit), so the leading _IDENT is
        # mandatory, not decorative.
        [_IDENT, _VC2, _SIGN, 7, 8, 9],
    ]
    return bodies


def _pack(bodies):
    counts = [len(b) for b in bodies]
    rec = np.zeros(len(bodies) + 1, dtype=np.int64)
    np.cumsum(counts, out=rec[1:])
    raw = np.concatenate(
        [np.asarray(b, dtype=np.uint16) for b in bodies]
    ).astype(np.uint16)
    return raw, rec


def test_batched_expand_matches_scalar_per_node():
    """Byte-identity of the kernel output vs the per-node scalar path."""
    bodies = _fixture_bodies()
    raw, rec = _pack(bodies)
    edge_types = np.array(
        [_LOCAL_EDGE, _PLT_EDGE, _LOCAL_EDGE, _PLT_EDGE,
         _LOCAL_EDGE, _PLT_EDGE, _LOCAL_EDGE, _PLT_EDGE],
        dtype=np.uint8,
    )
    self_ids = _SELF_TOKEN_LUT[edge_types].astype(np.uint16)

    batched = batched_expand(raw, rec, self_ids)
    (
        ref_flat,
        ref_off,
        ref_states,
        ref_vc2,
        ref_f128,
    ) = _scalar_per_node(raw, rec, edge_types)

    assert np.array_equal(batched.node_offsets, ref_off)
    assert np.array_equal(batched.expanded, ref_flat)

    # Per-node state fields + promotion masks via the integrated slicer.
    exp = expand_node_bodies(
        GatheredBodies(raw=raw, record_offsets=rec), edge_types
    )
    assert np.array_equal(exp.expanded, ref_flat)
    assert np.array_equal(exp.node_offsets, ref_off)
    # Bind the lazy per-node view lists ONCE (each property access re-slices
    # the batched arrays); the dense path never reads them, but this kernel-
    # equivalence pin still validates the slicer.
    got_states = exp.states
    got_vc2 = exp.extra_value_v2_masks
    got_f128 = exp.extra_f128_masks
    for i, st in enumerate(ref_states):
        got = got_states[i]
        assert np.array_equal(got.raw_tokens, st.raw_tokens), i
        assert np.array_equal(got.real_mask, st.real_mask), i
        assert np.array_equal(got.number_mask, st.number_mask), i
        assert np.array_equal(got.runlen_number, st.runlen_number), i
        assert np.array_equal(got.runlen_value, st.runlen_value), i
        assert np.array_equal(
            got.carries_inline_mask, st.carries_inline_mask
        ), i
        assert np.array_equal(
            got.is_negative_per_position, st.is_negative_per_position
        ), i
        assert np.array_equal(got.digit_cumsum, st.digit_cumsum), i
        assert np.array_equal(got_vc2[i], ref_vc2[i]), i
        assert np.array_equal(got_f128[i], ref_f128[i]), i


def test_batched_expand_empty_batch():
    """Zero nodes -> empty flat stream + a singleton CSR."""
    raw = np.zeros(0, dtype=np.uint16)
    rec = np.zeros(1, dtype=np.int64)
    batched = batched_expand(raw, rec, np.zeros(0, dtype=np.uint16))
    assert batched.expanded.size == 0
    assert batched.node_offsets.tolist() == [0]
    exp = expand_node_bodies(
        GatheredBodies(raw=raw, record_offsets=rec),
        np.zeros(0, dtype=np.uint8),
    )
    assert exp.expanded.size == 0
    assert exp.states == []


def test_batched_expand_empty_node_neighbors_stay_boundary_local():
    """An empty node adjacent to non-empty ones keeps run-lengths +
    cumsum boundary-local.

    The scalar ``build_inline_decode_state`` cannot process an empty body
    (so the corpus never emits one), but the batched twin must not let an
    empty node leak a neighbor's run across the boundary. The non-empty
    nodes' batched state is pinned against the scalar built per node; the
    empty nodes contribute a self-token-only (length-1) expanded slot."""
    bodies = [[], [_VC2] + list(range(8)), [], [_IDENT], []]
    raw, rec = _pack(bodies)
    edge_types = np.array([_LOCAL_EDGE] * len(bodies), dtype=np.uint8)
    self_ids = _SELF_TOKEN_LUT[edge_types].astype(np.uint16)

    exp = expand_node_bodies(
        GatheredBodies(raw=raw, record_offsets=rec), edge_types
    )
    # Empty nodes -> own_length 1 (just the self-token); non-empty nodes
    # keep their full survivor body. The CSR widths prove no cross-node
    # leak (an empty node would otherwise absorb a neighbor's run).
    own = np.diff(exp.node_offsets)
    assert own.tolist() == [1, 2, 1, 2, 1]  # node1: VC2+1 painted; node3: ident

    got_states = exp.states  # bind the lazy view list once
    for i, body in enumerate(bodies):
        if not body:
            continue
        st = build_inline_decode_state(
            np.asarray(body, dtype=np.uint16), format_version=1
        )
        assert np.array_equal(
            got_states[i].runlen_number, st.runlen_number
        ), i
        assert np.array_equal(got_states[i].digit_cumsum, st.digit_cumsum), i
        assert np.array_equal(got_states[i].real_mask, st.real_mask), i
