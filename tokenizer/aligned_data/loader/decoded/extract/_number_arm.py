"""Number arm of the v2 decode pass.

Single concern of this module: per ``TokenType`` (VALUED_CONST_V2 + 6
FLOAT* variants), collect number-source chunks in stream order, promote
multi-chunk inline slots back into ``working_tokens`` so the strip pass
preserves them, and flatten the per-source chunks into the shared side
arrays.  The per-TokenType dispatch table (``_FP_ENCODERS``) lives here
as the single source of truth for FP-width -> chunk encoder.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from tokenizer.tokens import TokenType

from .._inline_decode_state import InlineDecodeState
from ..custom_float import (
    from_bfloat16,
    from_float16,
    from_float32,
    from_float64,
    from_float80,
    from_float128,
    from_int,
)


# ---------------------------------------------------------------------------
# Per-TokenType dispatch -- single place that maps "number-carrying
# TokenType -> (chunk_count_from_inline_len, payload_decoder)".  No
# if-chain over TokenType anywhere else in this file.
# ---------------------------------------------------------------------------

# ``_FP_ENCODERS``: bits -> list[(sig, sign_exp)] per source FP width.
# Used by the number arm; nothing else.  The map is the single dispatch
# table for FP source widths -- callers index by ``TokenType``, not by
# string or width-in-bytes.
_FP_ENCODERS: Dict[TokenType, Callable[[int], List[Tuple[np.uint64, np.uint32]]]] = {
    TokenType.FLOAT16: from_float16,
    TokenType.BFLOAT16: from_bfloat16,
    TokenType.FLOAT32: from_float32,
    TokenType.FLOAT64: from_float64,
    TokenType.FLOAT80: from_float80,
    TokenType.FLOAT128: from_float128,
}

# Per-TokenType chunk_count rule (drives both the in-place inline-slot
# promotion and the side-array entry count):
#   VALUED_CONST_V2: ceil(inline_len / 8), clamped to >= 1.  Plan decision 5.
#   FLOAT16/BFLOAT16/FLOAT32/FLOAT64/FLOAT80: 1.
#   FLOAT128: 2 for finite values (plan decision 15) -- collapses to 1 for
#       NaN/Inf (plan decision 14).
# The chunk_count is always ``len(chunks)`` returned by the encoder; no
# separate rule lookup, no second source of truth.


def _decode_number_payload(
    payload: bytes,
    *,
    token_type: TokenType,
    sign: int = +1,
) -> List[Tuple[np.uint64, np.uint32]]:
    """Per-TokenType payload decoder.  Returns the ordered chunk list.

    Dispatch is by ``TokenType`` membership in the FP encoder table.
    ``VALUED_CONST_V2`` is the only int-typed branch and is special-cased
    here (it is the only TokenType in the number set that is NOT in
    ``_FP_ENCODERS``).  No if-chain over individual FP TokenTypes -- the
    encoder map fans those out.

    ``sign`` is the precomputed sign for this source's chunks, supplied
    by the caller after peeking the next real-token slot for a postfix
    ``value_negative`` metatoken (see :func:`_collect_number_sources`).
    Only ``VALUED_CONST_V2`` honors ``sign``; FP source widths carry
    their own sign bit in the IEEE-754 (or x87) bit pattern, so the
    caller MUST pass ``sign=+1`` for any FP token type. The assertion
    below pins that contract; passing ``sign=-1`` for an FP token is
    always a caller bug because there is no syntax in the v2 emitter
    for negating an FP literal via the postfix metatoken (FP sign rides
    in the source bit pattern, not in the postfix annotation).
    """
    if token_type is TokenType.VALUED_CONST_V2:
        # Plan decision 5: chunks come from ``_split_to_chunks`` (called
        # via from_int); ``sign`` comes from the caller after peeking
        # the next real-token slot for ``value_negative``.  ``b''`` ->
        # int 0 -> single signed-zero chunk.
        value = int.from_bytes(payload, byteorder="big", signed=False)
        return from_int(value, sign=sign)

    encoder = _FP_ENCODERS.get(token_type)
    if encoder is None:
        raise ValueError(
            f"_decode_number_payload called with non-number TokenType "
            f"{token_type.name}; expected VALUED_CONST_V2 or a FLOAT* "
            f"variant"
        )
    if sign != +1:
        raise AssertionError(
            f"_decode_number_payload received sign={sign} for FP token "
            f"{token_type.name}; FP sign is carried by the source bit "
            "pattern, not by a postfix value_negative metatoken. The "
            "stream walker must only attach negation to VALUED_CONST_V2."
        )
    # FP encoders take an integer bit-pattern.  Payload length must match
    # the source format's byte width; we trust the v2 encoder produced
    # that width and decode the bytes as a big-endian unsigned integer.
    bits = int.from_bytes(payload, byteorder="big", signed=False)
    return encoder(bits)


def _collect_number_sources(
    state: InlineDecodeState,
    number_token_ids: Dict[TokenType, int],
) -> List[Tuple[int, int, List[Tuple[np.uint64, np.uint32]]]]:
    """Collect (position, type_token_id, chunks) for every number-source.

    Returns the flat list IN STREAM-POSITION ORDER. ``np.nonzero``
    returns sorted ascending indices over the carrier-band positions
    so no explicit sort is needed -- stream-position order is what the
    final ``real_tokens`` stream sees once the strip pass runs
    (positions are absolute in ``raw_tokens``; the strip preserves
    their relative order). This guarantees the side-array entry order
    matches the order of number-token real-tokens in ``real_tokens``,
    which is the invariant Phase 3 splicing relies on.

    The per-source filtering, sign lookup, and inline-length read are
    fully vectorized via :class:`InlineDecodeState`; only the
    per-source ``_decode_number_payload`` dispatch (variable-length
    ``int.from_bytes`` -> chunks) runs in a Python loop. The sign
    array's ``is_negative_per_position`` is precomputed by the state
    builder via a runlength diff so FP types whose
    ``is_negative_per_position`` is False simply get ``sign=+1`` --
    the postfix-invariant check at the entry of
    :func:`_decode_to_staging` already rejects streams where a non-VC2
    carrier has a sign marker glued to its tail, so this loop never
    sees ``sign=-1`` on an FP source.
    """
    raw_tokens = state.raw_tokens
    n = int(raw_tokens.shape[0])
    if n == 0 or not number_token_ids:
        return []

    # Number-type ids as a u16 array -- the single source of truth
    # for "is this carrier a number-source we care about".
    number_type_id_array = np.array(
        list(number_token_ids.values()), dtype=np.uint16
    )
    source_mask = state.carries_inline_mask & np.isin(
        raw_tokens, number_type_id_array
    )
    positions = np.nonzero(source_mask)[0]
    if positions.size == 0:
        return []

    type_ids = raw_tokens[positions]
    # Inline length at p+1; positions == N-1 (carrier at the tail) have
    # no p+1 slot, so their length defaults to 0 (zero-pad). Build the
    # array bounds-aware: for positions < N-1, read runlen_number[p+1];
    # for the lone tail-position case, pad to 0.
    inline_lens = np.zeros(positions.shape[0], dtype=np.uint16)
    non_tail = positions < (n - 1)
    inline_lens[non_tail] = state.runlen_number[positions[non_tail] + 1]
    signs = np.where(state.is_negative_per_position[positions], -1, +1)

    # Reverse map: type_token_id -> TokenType, so the dispatch table
    # in ``_decode_number_payload`` does not need a per-source lookup
    # through ``number_token_ids``. Built once per call.
    id_to_type: Dict[int, TokenType] = {
        type_id: tt for tt, type_id in number_token_ids.items()
    }

    sources: List[Tuple[int, int, List[Tuple[np.uint64, np.uint32]]]] = []
    for position, type_id, inline_len, sign in zip(
        positions.tolist(),
        type_ids.tolist(),
        inline_lens.tolist(),
        signs.tolist(),
    ):
        payload_end = position + 1 + int(inline_len)
        payload = bytes(raw_tokens[position + 1 : payload_end].tolist())
        token_type = id_to_type[int(type_id)]
        chunks = _decode_number_payload(
            payload, token_type=token_type, sign=int(sign)
        )
        sources.append((position, int(type_id), chunks))

    return sources


def _promote_inline_slots(
    working_tokens: np.ndarray,
    sources: List[Tuple[int, int, List[Tuple[np.uint64, np.uint32]]]],
) -> None:
    """Overwrite ``chunk_count - 1`` inline slots after each multi-chunk source.

    For chunk_count == 1 the slice is empty -- no-op.  For chunk_count > 1
    we paint the same ``type_token_id`` into ``working_tokens[p+1 : p+C]``
    so the strip step preserves ``C`` consecutive real-tokens of the same
    type at the source's location.  Mutates ``working_tokens`` in-place.
    """
    for position, type_token_id, chunks in sources:
        chunk_count = len(chunks)
        if chunk_count <= 1:
            continue
        working_tokens[position + 1 : position + chunk_count] = type_token_id


def _flatten_number_chunks(
    sources: List[Tuple[int, int, List[Tuple[np.uint64, np.uint32]]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten the per-source chunk lists into the side-array pair.

    ``sources`` is already sorted by raw-stream position, so iterating
    over it and concatenating each source's chunk list produces the
    stream-order side-array.  An empty source set yields two length-0
    arrays of the correct dtype.
    """
    total = sum(len(chunks) for _, _, chunks in sources)
    significand = np.empty(total, dtype=np.uint64)
    sign_exponent = np.empty(total, dtype=np.uint32)
    write_idx = 0
    for _, _, chunks in sources:
        for sig, sign_exp in chunks:
            significand[write_idx] = sig
            sign_exponent[write_idx] = sign_exp
            write_idx += 1
    return significand, sign_exponent
