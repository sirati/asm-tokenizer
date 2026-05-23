"""Vocab introspection: TokenType -> uint16 vocab-id resolution.

Single source of truth for the "which vocab id corresponds to which v2
TokenType?" question. Callers consult these resolvers once per
session and from then on work purely in terms of integer ids — no
further string lookups, no hardcoded ids leaking into downstream
consumers.

The resolver consults ``vocab_manager.id_to_token_type`` — the int8
ndarray indexed by vocab id, holding ``TokenType`` values. A requested
TokenType that has **zero matches** is silently omitted from the
returned dict: real corpora may legitimately lack a TokenType whose
source data was never encountered (e.g. an unmatched-only corpus
without EXT_FUNC, or a no-bigfloat corpus without FLOAT80/FLOAT128).
Downstream consumers handle the absent-key case by treating the
identity / number side-array as empty for that TokenType.

Anything that IS in the vocab but in a malformed state — multiple
matches for the same TokenType, or a match in the reserved
``[0, _V2_RESERVED_TOKEN_COUNT)`` digit + ``value_negative`` prefix —
is a real vocab bug and still raises a typed ``ValueError``
immediately rather than silently producing a wrong decode.

Plan reference: `## Module layout` row "decoded/category_tokens.py" +
`## Locked-in decisions` item 10 (opt-in decode path, no hardcoded ids).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Static maps — the single place that knows "which TokenType backs each
# v2 Category" and "which TokenTypes carry numbers". Every other consumer
# goes through the resolvers below, so there is exactly one
# TokenType-literal site per logical mapping.
# ---------------------------------------------------------------------------

# Category -> TokenType. Seven map straight through to the same-named
# TokenType; BLOCK maps to BLOCK_V2 because the v2 wire form uses
# BLOCK_V2 (the legacy TokenType.BLOCK is v1-only). See plan
# `## Locked-in decisions` item 1.
_CATEGORY_TO_TOKEN_TYPE: Mapping[Category, TokenType] = {
    Category.BLOCK: TokenType.BLOCK_V2,
    Category.LOCAL_FUNC: TokenType.LOCAL_FUNC,
    Category.PLT_FUNC: TokenType.PLT_FUNC,
    Category.EXT_FUNC: TokenType.EXT_FUNC,
    Category.RO_DATA_PTR: TokenType.RO_DATA_PTR,
    Category.RW_DATA_PTR: TokenType.RW_DATA_PTR,
    Category.STRING_PTR: TokenType.STRING_PTR,
    Category.JUMP_TABLE: TokenType.JUMP_TABLE,
}

# Number-carrying TokenTypes (1 valued-const + 6 floats). Order is
# fixed for stable iteration but the public dict semantics are
# unordered-by-contract.
_NUMBER_TOKEN_TYPES: tuple[TokenType, ...] = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)

# ---------------------------------------------------------------------------
# Internal resolver — shared by both public functions so the lookup
# rule has exactly one implementation.
# ---------------------------------------------------------------------------


def _resolve_token_type_ids(
    vocab_manager: VocabularyManager,
    token_types: Iterable[TokenType],
) -> dict[TokenType, int]:
    """Map each present TokenType to its unique uint16 vocab id.

    The mapping is derived by scanning ``vocab_manager.id_to_token_type``
    once. A requested TokenType with **zero matches** is silently
    omitted from the returned dict (vocabs derived from corpora that
    never encountered that TokenType legitimately lack it). A TokenType
    with **multiple matches** or a match in the reserved
    ``[0, _V2_RESERVED_TOKEN_COUNT)`` digit + ``value_negative`` prefix
    is a malformed vocab and raises ``ValueError``.
    """
    type_array = np.asarray(vocab_manager.id_to_token_type)

    resolved: dict[TokenType, int] = {}
    for token_type in token_types:
        positions = np.flatnonzero(type_array == int(token_type))
        if positions.size == 0:
            # Absent TokenType: silently omitted. Consumers treat the
            # corresponding side-array as empty for this TokenType.
            continue
        if positions.size > 1:
            raise ValueError(
                f"vocab has {positions.size} entries tagged with "
                f"{token_type.name} (TokenType={int(token_type)}); "
                "exactly one type-token per v2 TokenType is expected. "
                f"Conflicting ids: {positions.tolist()}."
            )
        token_id = int(positions[0])
        if token_id < VocabularyManager._V2_RESERVED_TOKEN_COUNT:
            raise ValueError(
                f"vocab returned id {token_id} for {token_type.name} but "
                "v2 type-tokens must live above the reserved digit+marker "
                f"prefix (id >= {VocabularyManager._V2_RESERVED_TOKEN_COUNT}). "
                "The vocab is mis-tagged."
            )
        resolved[token_type] = token_id
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_category_token_ids(
    vocab_manager: VocabularyManager,
) -> dict[Category, int]:
    """Map every present v2 identity ``Category`` to its uint16 vocab id.

    Returns one entry per ``Category`` whose backing TokenType is
    present in the vocab. A Category whose TokenType has zero matches
    in ``vocab_manager.id_to_token_type`` is silently omitted from the
    returned dict — the result may be a proper subset of ``Category``.
    Raises ``ValueError`` only when a TokenType is malformed (duplicate
    ids, or an id in the reserved ``[0, _V2_RESERVED_TOKEN_COUNT)`` range).
    """
    type_to_id = _resolve_token_type_ids(
        vocab_manager, _CATEGORY_TO_TOKEN_TYPE.values()
    )
    return {
        category: type_to_id[token_type]
        for category, token_type in _CATEGORY_TO_TOKEN_TYPE.items()
        if token_type in type_to_id
    }


def resolve_number_token_ids(
    vocab_manager: VocabularyManager,
) -> dict[TokenType, int]:
    """Map every present number-carrying v2 ``TokenType`` to its uint16 vocab id.

    Returns one entry per number TokenType whose tag exists in the
    vocab. A TokenType with zero matches is silently omitted from the
    returned dict (so the result may have fewer than the 7 canonical
    keys). Same malformed-vocab raise contract as
    :func:`resolve_category_token_ids`.
    """
    return _resolve_token_type_ids(vocab_manager, _NUMBER_TOKEN_TYPES)


def resolve_value_negative_token_id(
    vocab_manager: VocabularyManager,
) -> int | None:
    """Resolve the uint16 vocab id of the ``value_negative`` postfix metatoken.

    The v2 emitter writes ``[Valued_Const_V2(|value|), Value_Negative()]``
    for negative integer immediates / displacements; the decoder needs
    the metatoken's vocab id to detect the postfix and flip the chunk
    sign for the preceding ``valued_const_v2`` source. For ``format_version``
    in (1, 2) the id is invariantly pinned at slot
    ``_V2_VALUE_NEGATIVE_TOKEN_ID`` (= 256) — both the constructor and
    ``from_vocab`` re-publish that slot via the ``value_negative_token_id``
    instance attribute. Returns ``None`` for vocabs whose
    ``format_version`` lies outside (1, 2); the decoder treats ``None``
    as "skip sign-handling" so legacy / non-v2 vocabs stay decodable.
    """
    return vocab_manager.value_negative_token_id
