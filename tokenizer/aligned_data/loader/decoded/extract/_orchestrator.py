"""Single-function decode pass: raw u16 token stream -> ``DecodedFunction``.

Single concern of this module: take a v2 wire-format token stream (uint16
array containing real-tokens at id >= 256 interleaved with inline-digit
bytes at id < 256) and produce the out-of-band decoded view defined in
``decoded_function.py``.  No splicing happens here -- the recursive
walker in ``splice.py`` calls into this module once per function body
and stitches the per-function results.

Plan reference: ``## Algorithm -- decode_raw_tokens`` and ``## Locked-in
decisions`` items 1, 2, 3, 5, 6, 7, 14, 15 + items 22, 26 + 28-31 for
the FID resolution + staging shape.

Algorithm shape:

1. One run-length pass over ``raw_tokens`` (``run_lengths`` from
   ``decoded.run_lengths``); used by BOTH the identity arm and the
   number arm.
2. Identity arm: per category, mask + positions + per-position inline-byte
   decode -> identity array.  Pure read pass over ``raw_tokens``; does
   NOT mutate the working buffer.  For categories in
   :data:`FID_KEYED_CATEGORIES` the per-position decode produces a
   caller-local id; when a section-derived
   ``fids_per_category[c]`` lookup is provided the local id is then
   indexed into that array to resolve the callee's globally-unique
   function identity (FID).  The result is staged as a ``uint32`` array
   for the FID-keyed branches (FIDs cannot be safely clipped to ``uint16``)
   and as ``uint16`` for the other five categories.
3. Number arm: per number-type, mask + positions + per-position payload
   decode via a per-type dispatch table (no if-chain over TokenType); the
   resulting per-position ``(sig, sign_exp)`` chunks are sorted into
   raw-stream-position order so the final side-array entries match the
   number-token order of the post-strip real-token stream 1:1.
4. Multi-chunk promotion: positions whose source produced ``C > 1`` chunks
   overwrite ``working_tokens[p+1 : p+C]`` with their own type-id, so the
   strip step preserves ``C`` consecutive real-tokens at the source's
   location.
5. Strip pass: ``real_tokens = working_tokens[working_tokens >= 256]``.

Working buffer isolation: ``working_tokens = raw_tokens.copy()``.  The
caller's array is never mutated, which matters because the splicer
holds the raw stream across multiple decode calls and the identity-arm
decode reads from ``raw_tokens`` (not ``working_tokens``) so the order
in which we process arms cannot smear inline-bytes across arms.

Single-concern guard: the identity arm and the number arm never touch
each other's category / TokenType.  Identities are owned by ``Category``;
numbers are owned by ``TokenType``.  The two key-sets are disjoint by
construction (the resolvers in ``category_tokens.py`` are the single
source of truth).

Staging vs DecodedFunction: the FID-resolved identity branch produces
``uint32`` per-position arrays, which violates :class:`DecodedFunction`'s
``uint16`` invariant.  The split internal entry
:func:`_decode_to_staging` returns a :class:`_StagingDecoded` carrying
possibly-mixed dtypes; :func:`decode_raw_tokens` is the
backwards-compatible u16-only public wrapper used by the synthetic-test
and standalone single-function decode paths (no ``fids_per_category``
lookup; identities follow the legacy "inline payload IS the identity
value" contract).  Splice integration calls
:func:`_decode_to_staging` directly and runs compaction at the splice
top level (see :mod:`tokenizer.aligned_data.loader.decoded.splice`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tokenizer.tokens import Category, TokenType

from .._inline_decode_state import (
    _V2_VALUE_NEGATIVE_TOKEN_ID,
    build_inline_decode_state,
)
from ..category_tokens import FID_KEYED_CATEGORIES
from ..decoded_function import DecodedFunction
from ._identity_arm import _extract_identities
from ._number_arm import (
    _collect_number_sources,
    _flatten_number_chunks,
    _promote_inline_slots,
)
from ._staging import _StagingDecoded


__all__ = ["decode_raw_tokens"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _decode_to_staging(
    raw_tokens: np.ndarray,
    *,
    id_token_ids: Dict[Category, int],
    number_token_ids: Dict[TokenType, int],
    value_negative_token_id: int,
    format_version: int,
    fids_per_category: "Optional[Dict[Category, np.ndarray]]" = None,
    func_name: str = "decoded",
    metadata: Optional[Dict[str, Any]] = None,
) -> _StagingDecoded:
    """Decode one v2 raw-token stream into a :class:`_StagingDecoded`.

    Plan reference: ``polished-greeting-moler.md`` -- this is the entry
    the splice walker calls per-callee (passes a section-derived
    ``fids_per_category``); the public :func:`decode_raw_tokens` is
    the u16-only standalone wrapper that delegates here with
    ``fids_per_category=None``.

    See module docstring for the algorithm.  ``raw_tokens`` itself is
    never mutated -- a working copy is allocated internally and the
    multi-chunk promotion writes back into that copy only.

    The vectorized :class:`InlineDecodeState` is built once per stream
    here (after the empty-stream short-circuit + leading-real
    precondition) and threaded into both the identity arm and the
    number arm so neither recomputes its own runlength / carrier mask.
    The postfix-sign invariant check is inlined against that same
    state (no separate ``np.maximum.accumulate`` pass).

    ``value_negative_token_id``: the uint16 vocab id of the v2 postfix
    ``value_negative`` metatoken (resolved via
    :func:`resolve_value_negative_token_id`). REQUIRED -- the unified
    vocab pins it at id 256 and the strip-and-shift path expects it
    out of the post-strip ``real_tokens`` (D5/D6 in the plan). The
    ``value_negative`` token is STRIPPED from ``real_tokens`` because
    its meaning is already captured in ``numbers_sign_exponent``.

    ``format_version``: REQUIRED -- the vectorized carrier-band lookup
    is only valid under the unified vocab (``format_version=1``). The
    state builder asserts on any other value.
    """
    n = int(raw_tokens.shape[0])

    use_fid_lookup = fids_per_category is not None
    empty_identities: Dict[Category, np.ndarray] = {}
    for category in Category:
        if use_fid_lookup and category in FID_KEYED_CATEGORIES:
            empty_identities[category] = np.empty(0, dtype=np.uint32)
        else:
            empty_identities[category] = np.empty(0, dtype=np.uint16)

    # ---- Empty stream short-circuit ----
    if n == 0:
        return _StagingDecoded(
            real_tokens=np.empty(0, dtype=np.uint16),
            identities=empty_identities,
            numbers_significant=np.empty(0, dtype=np.uint64),
            numbers_sign_exponent=np.empty(0, dtype=np.uint32),
            func_name=func_name,
            metadata=dict(metadata) if metadata else {},
        )

    # ---- Codec precondition: stream must lead with a real-token carrier.
    # Strict ``> _V2_VALUE_NEGATIVE_TOKEN_ID`` (= 256) rejects BOTH an
    # inline-digit byte AND a leading ``value_negative`` (which has no
    # preceding source to sign-mark). ----
    if int(raw_tokens[0]) <= _V2_VALUE_NEGATIVE_TOKEN_ID:
        raise AssertionError(
            "raw_tokens[0] must be a carrier real-token (id > 256); "
            "inline-digit runs at position 0 and the value_negative "
            "sign marker at position 0 are both invalid v2 wire shapes "
            "-- a leading value_negative has no preceding source to "
            "bind to."
        )

    # ---- Vectorized per-stream pre-compute: every downstream consumer
    # (identity arm, number arm, postfix-invariant check, strip/shift)
    # reads from ``state``; nothing rebuilds a mask another consumer
    # already produced. ----
    state = build_inline_decode_state(raw_tokens, format_version=format_version)

    # ---- Postfix-sign invariant: any non-VC2 carrier whose
    # ``is_negative_per_position`` flag is True is a bug (an FP /
    # identity carrier has a sign marker glued to its tail). When the
    # vocab carries no VALUED_CONST_V2 id at all, ANY positive sign
    # flag is a bug because there is no magnitude to bind the sign to.
    # The runlength-form state already encodes "which carriers are
    # postfix-signed", so this reduces to a single boolean op. ----
    valued_const_v2_id = number_token_ids.get(TokenType.VALUED_CONST_V2)
    negative_positions = np.nonzero(state.is_negative_per_position)[0]
    if negative_positions.size > 0:
        if valued_const_v2_id is None:
            raise AssertionError(
                f"v2 stream contains {negative_positions.size} value_negative "
                f"postfix marker(s) at positions "
                f"{negative_positions.tolist()[:5]} but the vocab has no "
                "VALUED_CONST_V2 type-token to bind them to -- the encoder "
                "emitted a sign marker without a magnitude."
            )
        owner_ids = raw_tokens[negative_positions]
        bad = owner_ids != valued_const_v2_id
        if bool(bad.any()):
            bad_idx = int(np.nonzero(bad)[0][0])
            owner_position = int(negative_positions[bad_idx])
            owner_value = int(owner_ids[bad_idx])
            raise AssertionError(
                f"value_negative postfix marker is bound to real-token id "
                f"{owner_value} at position {owner_position} (expected "
                f"valued_const_v2 id {valued_const_v2_id}); the v2 emitter "
                "only emits value_negative as a postfix sign marker for "
                "valued_const_v2."
            )

    # ---- Identity arm: pure read pass over the ORIGINAL raw_tokens ----
    identities = _extract_identities(
        state,
        id_token_ids,
        fids_per_category=fids_per_category,
    )

    # ---- Number arm: collect chunks per source in stream order (the
    # vectorized filtering uses ``state``), then promote inline slots,
    # then flatten to the side-array pair. ----
    sources = _collect_number_sources(state, number_token_ids)
    working_tokens = raw_tokens.copy()
    _promote_inline_slots(working_tokens, sources)
    numbers_significant, numbers_sign_exponent = _flatten_number_chunks(sources)

    # ---- Strip + shift: drop value_negative (id 256) AND every inline
    # byte; shift surviving real-token ids down by 256 so the
    # model-facing vocab compacts. Slot 0 in the shifted layout is the
    # reserved "null-content" id — it is the position the stripped
    # value_negative sign marker would have collapsed to, and it never
    # appears in a valid post-strip stream. Downstream consumers (model
    # embedding tables, special-token handlers) MAY use id 0 as a pad /
    # null / mask slot because the decoder guarantees it is never
    # emitted from real source content. The mask is recomputed AFTER
    # promotion so promoted slots survive the strip. ----
    keep_mask = working_tokens > _V2_VALUE_NEGATIVE_TOKEN_ID
    real_tokens = (
        working_tokens[keep_mask].astype(np.int32) - _V2_VALUE_NEGATIVE_TOKEN_ID
    ).astype(np.uint16)

    # ---- Sanity invariant: the side-array length equals the count of
    # number-tokens in the final real-token stream.  Cheap O(K) check
    # over the number-type set; flags any internal bug in the
    # promotion / chunk-count arithmetic before the consumer trips on
    # it. The number-id array is shifted into the post-strip id space
    # (D6 in the plan) so the ``np.isin`` comparison matches the new
    # ``real_tokens`` layout. ----
    number_token_id_array = np.array(
        list(number_token_ids.values()), dtype=np.uint16
    )
    shifted_number_id_array = (
        number_token_id_array.astype(np.int32) - _V2_VALUE_NEGATIVE_TOKEN_ID
    ).astype(np.uint16)
    final_number_count = int(
        np.isin(real_tokens, shifted_number_id_array).sum()
    )
    if final_number_count != numbers_significant.shape[0]:
        raise AssertionError(
            "number side-array length mismatch: real_tokens contains "
            f"{final_number_count} number-tokens but side-arrays carry "
            f"{numbers_significant.shape[0]} entries -- a multi-chunk "
            "promotion or stream-order alignment bug"
        )

    return _StagingDecoded(
        real_tokens=real_tokens,
        identities=identities,
        numbers_significant=numbers_significant,
        numbers_sign_exponent=numbers_sign_exponent,
        func_name=func_name,
        metadata=dict(metadata) if metadata else {},
    )


def decode_raw_tokens(
    raw_tokens: np.ndarray,
    *,
    id_token_ids: Dict[Category, int],
    number_token_ids: Dict[TokenType, int],
    value_negative_token_id: int,
    format_version: int,
    func_name: str = "decoded",
    metadata: Optional[Dict[str, Any]] = None,
) -> DecodedFunction:
    """Decode one v2 raw-token stream into a ``DecodedFunction``.

    See module docstring for the algorithm.  ``raw_tokens`` itself is
    never mutated.  This standalone entry point uses the legacy
    "inline payload IS the identity value" decode -- the FID-resolved
    path lives on the internal :func:`_decode_to_staging` and is
    consumed by the splice walker, which can defer compaction to the
    top of the spliced view.  Single-function consumers (synthetic
    tests, ad-hoc decode of one body without a section) keep the
    backwards-compatible u16-identity shape.

    Args:
        raw_tokens: ``uint16[N]`` v2 wire-format stream.  Real tokens
            at id >= 256; inline-digit bytes at id < 256.  The first
            position MUST be a real token (precondition of
            ``run_lengths`` and of the v2 codec contract: inline data
            never leads the stream).
        id_token_ids: ``dict[Category, int]`` of size 8 from
            ``resolve_category_token_ids``.  Each value is the uint16
            vocab id of the v2 type-token whose occurrences own that
            ``Category``'s identity space.
        number_token_ids: ``dict[TokenType, int]`` of size 7 from
            ``resolve_number_token_ids``.  One entry per number-
            carrying TokenType (VALUED_CONST_V2 + 6 FLOAT* variants).
        value_negative_token_id: REQUIRED uint16 vocab id of the v2
            postfix ``value_negative`` metatoken (typically resolved
            via :func:`resolve_value_negative_token_id`; pinned at 256
            under the unified vocab). The stream walker detects the
            postfix after a ``valued_const_v2`` source and decodes the
            source's chunks with ``sign=-1``; the ``value_negative``
            token itself is STRIPPED from ``real_tokens`` (its meaning
            is captured in ``numbers_sign_exponent``).
        format_version: REQUIRED unified-vocab format version (must be
            ``1``). The vectorized carrier-band lookup that drives
            sign + number-arm decode is only valid under that layout.
        func_name: human-readable root function name.  Defaults to
            ``"decoded"``; callers that have a real name should pass it.
        metadata: optional free-form dict attached to the result.

    Returns:
        ``DecodedFunction`` with strip-and-promote ``real_tokens``, one
        identity array per ``Category`` (length-0 if no occurrences),
        and the shared ``(numbers_significant, numbers_sign_exponent)``
        pair carrying ordered chunks for every number-token occurrence
        in the final stream.

        ``real_tokens`` ids are SHIFTED — every surviving vocab id has
        ``_V2_VALUE_NEGATIVE_TOKEN_ID`` (= 256) subtracted, so the
        model-facing vocab starts at slot 1 (originally id 257 =
        ``valued_const_v2``). Slot 0 is reserved as a "null-content" id:
        it would have held the stripped ``value_negative`` sign marker
        if it survived, but the strip drops every such occurrence, so
        id 0 NEVER appears in a valid post-strip stream. Downstream
        consumers may treat id 0 as pad / null / mask.
    """
    staging = _decode_to_staging(
        raw_tokens,
        id_token_ids=id_token_ids,
        number_token_ids=number_token_ids,
        fids_per_category=None,
        value_negative_token_id=value_negative_token_id,
        format_version=format_version,
        func_name=func_name,
        metadata=metadata,
    )
    return DecodedFunction(
        real_tokens=staging.real_tokens,
        identities=staging.identities,
        numbers_significant=staging.numbers_significant,
        numbers_sign_exponent=staging.numbers_sign_exponent,
        func_name=staging.func_name,
        metadata=staging.metadata,
    )
