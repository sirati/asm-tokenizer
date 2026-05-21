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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from tokenizer.tokens import Category, TokenType

from .category_tokens import FID_KEYED_CATEGORIES
from .custom_float import (
    from_bfloat16,
    from_float16,
    from_float32,
    from_float64,
    from_float80,
    from_float128,
    from_int,
)
from .decoded_function import DecodedFunction
from .run_lengths import run_lengths

__all__ = ["decode_raw_tokens"]


# u32 sentinel for FID-keyed identity staging.  Compaction (in splice.py)
# folds these positions back to the public uint16 sentinel ``0xFFFF`` in
# the final :class:`DecodedFunction`.
_IDENTITY_SENTINEL_U32 = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class _StagingDecoded:
    """Private mid-pipeline view: same fields as :class:`DecodedFunction`
    but with NO dtype invariant on the per-Category identity arrays.

    The splice walker concatenates these verbatim and runs per-Category
    compaction at the top level; the public :class:`DecodedFunction`
    is constructed only after compaction so the u16 invariant holds for
    consumers.  Treat the arrays as read-only -- consumers MUST NOT
    mutate them (the splicer reuses the same arrays across the concat
    step).
    """

    real_tokens: np.ndarray
    identities: Dict[Category, np.ndarray]
    numbers_significant: np.ndarray
    numbers_sign_exponent: np.ndarray
    func_name: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Shared occurrence iterator -- single source of truth for the
# "mask -> positions -> per-position (position, payload_bytes)" walk used
# by BOTH the identity arm and the number arm.  Per-position payload
# decoding (identity-vs-number value semantics) stays in each arm; only
# the raw-stream traversal is shared.
# ---------------------------------------------------------------------------


def _iter_token_occurrences(
    raw_tokens: np.ndarray,
    runlen: np.ndarray,
    token_id: int,
) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(position, payload_bytes)`` for every occurrence of ``token_id``.

    ``payload_bytes`` is the contiguous run of inline-digit bytes
    immediately following each occurrence, materialised from the uint16
    stream as a big-endian byte sequence.  Tail-position occurrences
    (no room for a trailing inline run) yield ``b''``.

    Pure read pass over ``raw_tokens`` -- never mutates the caller's
    array.  ``np.nonzero`` returns sorted ascending indices, so the
    yielded ``position`` values are stream-ascending.
    """
    n = raw_tokens.shape[0]
    mask = raw_tokens == token_id
    positions = np.nonzero(mask)[0]
    for position in positions:
        position_int = int(position)
        if position_int + 1 < n:
            inline_len = int(runlen[position_int + 1])
            payload_end = position_int + 1 + inline_len
            payload = bytes(raw_tokens[position_int + 1 : payload_end].tolist())
        else:
            payload = b""
        yield position_int, payload


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
) -> List[Tuple[np.uint64, np.uint32]]:
    """Per-TokenType payload decoder.  Returns the ordered chunk list.

    Dispatch is by ``TokenType`` membership in the FP encoder table.
    ``VALUED_CONST_V2`` is the only int-typed branch and is special-cased
    here (it is the only TokenType in the number set that is NOT in
    ``_FP_ENCODERS``).  No if-chain over individual FP TokenTypes -- the
    encoder map fans those out.
    """
    if token_type is TokenType.VALUED_CONST_V2:
        # Plan decision 5: chunks come from ``_split_to_chunks`` (called
        # via from_int); sign defaults to +1 because the v2 stream has
        # no inline sign for valued-const today.  ``b''`` -> int 0 ->
        # single signed-zero chunk.
        value = int.from_bytes(payload, byteorder="big", signed=False)
        return from_int(value, sign=+1)

    encoder = _FP_ENCODERS.get(token_type)
    if encoder is None:
        raise ValueError(
            f"_decode_number_payload called with non-number TokenType "
            f"{token_type.name}; expected VALUED_CONST_V2 or a FLOAT* "
            f"variant"
        )
    # FP encoders take an integer bit-pattern.  Payload length must match
    # the source format's byte width; we trust the v2 encoder produced
    # that width and decode the bytes as a big-endian unsigned integer.
    bits = int.from_bytes(payload, byteorder="big", signed=False)
    return encoder(bits)


# ---------------------------------------------------------------------------
# Identity arm -- pure read pass over the original raw_tokens
# ---------------------------------------------------------------------------


_IDENTITY_SENTINEL = 0xFFFF
_IDENTITY_MAX_NON_SENTINEL = 0xFFFE


def _decode_identity_payload(payload: bytes) -> int:
    """Decode an identity-arm payload as a big-endian unsigned integer.

    Empty payload decodes as 0 (``int.from_bytes(b'', ...)`` returns 0).
    A decoded value exceeding ``0xFFFE`` clips to the sentinel ``0xFFFF``
    per plan decision 7.
    """
    value = int.from_bytes(payload, byteorder="big", signed=False)
    if value > _IDENTITY_MAX_NON_SENTINEL:
        return _IDENTITY_SENTINEL
    return value


def _resolve_fid_payload(
    payload: bytes,
    *,
    fid_lookup: np.ndarray,
) -> int:
    """Decode an FID-keyed identity payload to the looked-up callee FID.

    Empty payload decodes as caller-local id 0 (``int.from_bytes(b'',
    ...)``); a local id beyond ``len(fid_lookup)`` yields the u32
    sentinel which compaction will fold to the public u16 sentinel
    downstream.  Returning a Python ``int`` keeps the call-site free
    from numpy-dtype dispatch -- the per-category array build below
    casts to ``uint32`` once.
    """
    local_id = int.from_bytes(payload, byteorder="big", signed=False)
    if local_id >= len(fid_lookup):
        return int(_IDENTITY_SENTINEL_U32)
    return int(fid_lookup[local_id])


def _extract_identities(
    raw_tokens: np.ndarray,
    runlen: np.ndarray,
    id_token_ids: Dict[Category, int],
    *,
    fids_per_category: "Optional[Dict[Category, np.ndarray]]" = None,
) -> Dict[Category, np.ndarray]:
    """Build one identity array per ``Category``.

    Pre-fills every ``Category`` member with an empty array of the
    expected staging dtype (``uint32`` for FID-keyed categories when
    ``fids_per_category`` is provided, otherwise ``uint16``) so the
    returned dict always carries the full 8-key set regardless of
    which Categories the caller's ``id_token_ids`` map covers.  Any
    Category present in ``id_token_ids`` then overwrites its empty
    slot with the decoded occurrences.

    Pure read pass over ``raw_tokens``; never touches the number arm's
    working buffer.  Iteration order over positions is stream-ascending
    so the per-category array order matches the order of category-token
    occurrences in the final post-strip real-token stream.

    FID resolution branch (plan Decision 22): for each category in
    :data:`FID_KEYED_CATEGORIES` whose
    ``fids_per_category[c]`` is supplied, the inline-digit payload is
    decoded as a caller-local id, then indexed into the per-category
    FID array to produce the callee's globally-unique function
    identity (FID).  Out-of-range caller-local ids resolve to the u32
    sentinel; compaction downstream folds them to the public u16
    sentinel.  Categories NOT in ``FID_KEYED_CATEGORIES`` and the
    FID-keyed categories whose ``fids_per_category`` is absent fall
    back to the legacy "inline payload IS the identity value" decode
    (u16, sentinel 0xFFFF on overflow).
    """
    use_fid_lookup = fids_per_category is not None
    identities: Dict[Category, np.ndarray] = {}
    for category in Category:
        # Empty-array dtype matches the staging dtype for this category
        # so concat downstream stays dtype-consistent.
        if use_fid_lookup and category in FID_KEYED_CATEGORIES:
            identities[category] = np.empty(0, dtype=np.uint32)
        else:
            identities[category] = np.empty(0, dtype=np.uint16)

    for category, type_token_id in id_token_ids.items():
        fid_lookup: "Optional[np.ndarray]" = (
            fids_per_category[category]  # type: ignore[index]
            if use_fid_lookup and category in FID_KEYED_CATEGORIES
            else None
        )
        values_list: List[int] = []
        for _position, payload in _iter_token_occurrences(
            raw_tokens, runlen, type_token_id
        ):
            if fid_lookup is not None:
                values_list.append(
                    _resolve_fid_payload(payload, fid_lookup=fid_lookup)
                )
            else:
                values_list.append(_decode_identity_payload(payload))
        if fid_lookup is not None:
            identities[category] = np.array(values_list, dtype=np.uint32)
        else:
            identities[category] = np.array(values_list, dtype=np.uint16)
    return identities


# ---------------------------------------------------------------------------
# Number arm -- per-source chunk decode + stream-order alignment
# ---------------------------------------------------------------------------


def _collect_number_sources(
    raw_tokens: np.ndarray,
    runlen: np.ndarray,
    number_token_ids: Dict[TokenType, int],
) -> List[Tuple[int, int, List[Tuple[np.uint64, np.uint32]]]]:
    """Collect (position, type_token_id, chunks) for every number-source.

    Returns the flat list SORTED by raw-stream position.  Stream-position
    order is what the final ``real_tokens`` stream sees once the strip
    pass runs (positions are absolute in ``raw_tokens``; the strip
    preserves their relative order).  This guarantees the side-array
    entry order matches the order of number-token real-tokens in
    ``real_tokens``, which is the invariant Phase 3 splicing relies on.
    """
    sources: List[Tuple[int, int, List[Tuple[np.uint64, np.uint32]]]] = []
    for token_type, type_token_id in number_token_ids.items():
        for position, payload in _iter_token_occurrences(
            raw_tokens, runlen, type_token_id
        ):
            chunks = _decode_number_payload(payload, token_type=token_type)
            sources.append((position, type_token_id, chunks))

    sources.sort(key=lambda triple: triple[0])
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _decode_to_staging(
    raw_tokens: np.ndarray,
    *,
    id_token_ids: Dict[Category, int],
    number_token_ids: Dict[TokenType, int],
    fids_per_category: "Optional[Dict[Category, np.ndarray]]" = None,
    func_name: str = "decoded",
    metadata: Optional[Dict[str, Any]] = None,
) -> _StagingDecoded:
    """Decode one v2 raw-token stream into a :class:`_StagingDecoded`.

    Plan reference: ``## Algorithm changes`` -- this is the entry the
    splice walker calls per-callee (passes a section-derived
    ``fids_per_category``); the public :func:`decode_raw_tokens` is
    the u16-only standalone wrapper that delegates here with
    ``fids_per_category=None``.

    See module docstring for the algorithm.  ``raw_tokens`` itself is
    never mutated -- a working copy is allocated internally and the
    multi-chunk promotion writes back into that copy only.
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

    # ---- Codec precondition (also enforced by run_lengths) ----
    if int(raw_tokens[0]) < 256:
        raise AssertionError(
            "raw_tokens[0] must be a real-token (id >= 256); inline-data "
            "runs never lead the v2 wire stream"
        )

    # ---- Working buffer + run-length state (shared by both arms) ----
    working_tokens = raw_tokens.copy()
    real_mask = raw_tokens >= 256
    inline_mask = ~real_mask
    runlen = run_lengths(inline_mask)

    # ---- Identity arm: pure read pass over the ORIGINAL raw_tokens ----
    identities = _extract_identities(
        raw_tokens,
        runlen,
        id_token_ids,
        fids_per_category=fids_per_category,
    )

    # ---- Number arm: collect chunks per source in stream order, then
    # promote inline slots, then flatten to the side-array pair. ----
    sources = _collect_number_sources(raw_tokens, runlen, number_token_ids)
    _promote_inline_slots(working_tokens, sources)
    numbers_significant, numbers_sign_exponent = _flatten_number_chunks(sources)

    # ---- Strip pass: recompute the mask AFTER promotion so promoted
    # slots survive. ----
    keep_mask = working_tokens >= 256
    real_tokens = working_tokens[keep_mask]

    # ---- Sanity invariant: the side-array length equals the count of
    # number-tokens in the final real-token stream.  Cheap O(K) check
    # over the number-type set; flags any internal bug in the
    # promotion / chunk-count arithmetic before the consumer trips on
    # it. ----
    number_token_id_array = np.array(
        list(number_token_ids.values()), dtype=np.uint16
    )
    final_number_count = int(np.isin(real_tokens, number_token_id_array).sum())
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
        func_name: human-readable root function name.  Defaults to
            ``"decoded"``; callers that have a real name should pass it.
        metadata: optional free-form dict attached to the result.

    Returns:
        ``DecodedFunction`` with strip-and-promote ``real_tokens``, one
        identity array per ``Category`` (length-0 if no occurrences),
        and the shared ``(numbers_significant, numbers_sign_exponent)``
        pair carrying ordered chunks for every number-token occurrence
        in the final stream.
    """
    staging = _decode_to_staging(
        raw_tokens,
        id_token_ids=id_token_ids,
        number_token_ids=number_token_ids,
        fids_per_category=None,
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
