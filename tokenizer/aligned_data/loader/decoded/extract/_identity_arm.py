"""Identity arm of the v2 decode pass.

Single concern of this module: per ``Category``, decode every
identity-token occurrence in the raw stream into a typed identity array.
Pure read pass over ``raw_tokens`` -- this arm never mutates the
working buffer.  The FID-resolution branch (plan Decision 22) is the
only complication; it sits behind the same per-position walk.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from tokenizer.tokens import Category

from .._inline_decode_state import InlineDecodeState
from ..category_tokens import FID_KEYED_CATEGORIES
from ._occurrence_iter import _iter_token_occurrences
from ._staging import _IDENTITY_SENTINEL_U32


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
    state: InlineDecodeState,
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
            state.raw_tokens, state.runlen_number, type_token_id
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
