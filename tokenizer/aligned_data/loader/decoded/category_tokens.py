"""Vocab introspection: TokenType -> uint16 vocab-id resolution.

Single source of truth for the "which vocab id corresponds to which v2
TokenType?" question. The decoded / splice pipeline consults these
resolvers once per session (the wiring layer caches the dicts on the
``BinarySession``) and from then on works purely in terms of integer
ids — no further string lookups, no hardcoded ids leaking into the
extract / splice passes.

The resolver consults ``vocab_manager.id_to_token_type`` — the int8
ndarray indexed by vocab id, holding ``TokenType`` values — and asserts
that each requested TokenType maps to exactly one vocab id. Anything
else (zero matches, multiple matches, or an id outside the valid
[256, vocab_size) range) is a bug in the vocab and raises a typed
``ValueError`` immediately rather than silently producing a wrong
decode.

Plan reference: `## Module layout` row "decoded/category_tokens.py" +
`## Locked-in decisions` item 10 (opt-in decode path, no hardcoded ids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

from tokenizer.tokens import Category, TokenType

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from tokenizer.token_manager import VocabularyManager


# ---------------------------------------------------------------------------
# Static maps — the single place that knows "which TokenType backs each
# v2 Category" and "which TokenTypes carry numbers". Every other consumer
# in the decoded / splice pipeline goes through the resolvers below, so
# there is exactly one TokenType-literal site per logical mapping.
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
    vocab_manager: "VocabularyManager",
    token_types: Iterable[TokenType],
) -> dict[TokenType, int]:
    """Map each TokenType to its unique uint16 vocab id.

    The mapping is derived by scanning ``vocab_manager.id_to_token_type``
    once. Each requested TokenType must appear at exactly one vocab id
    in the [256, vocab_size) range; otherwise the vocab is malformed
    for the decoded pipeline and the caller is told loudly.
    """
    type_array = np.asarray(vocab_manager.id_to_token_type)

    resolved: dict[TokenType, int] = {}
    for token_type in token_types:
        positions = np.flatnonzero(type_array == int(token_type))
        if positions.size == 0:
            raise ValueError(
                f"vocab is missing the v2 type-token for {token_type.name} "
                f"(TokenType={int(token_type)}); the decoded pipeline "
                "requires a v1 unified vocab that exposes every v2 "
                "category + number token."
            )
        if positions.size > 1:
            raise ValueError(
                f"vocab has {positions.size} entries tagged with "
                f"{token_type.name} (TokenType={int(token_type)}); "
                "exactly one type-token per v2 TokenType is expected. "
                f"Conflicting ids: {positions.tolist()}."
            )
        token_id = int(positions[0])
        if token_id < 256:
            raise ValueError(
                f"vocab returned id {token_id} for {token_type.name} but "
                "v2 type-tokens must live above the reserved digit-slot "
                "range (id >= 256). The vocab is mis-tagged."
            )
        resolved[token_type] = token_id
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_category_token_ids(
    vocab_manager: "VocabularyManager",
) -> dict[Category, int]:
    """Map every v2 identity ``Category`` to its uint16 vocab id.

    Returns one entry per ``Category`` member (exactly 8). Each value is
    the unique vocab id whose ``id_to_token_type`` slot holds the
    matching v2 TokenType. Raises ``ValueError`` if any required type
    is missing or duplicated in the vocab.
    """
    type_to_id = _resolve_token_type_ids(
        vocab_manager, _CATEGORY_TO_TOKEN_TYPE.values()
    )
    return {
        category: type_to_id[token_type]
        for category, token_type in _CATEGORY_TO_TOKEN_TYPE.items()
    }


def resolve_number_token_ids(
    vocab_manager: "VocabularyManager",
) -> dict[TokenType, int]:
    """Map every number-carrying v2 ``TokenType`` to its uint16 vocab id.

    Returns one entry per number TokenType (exactly 7: VALUED_CONST_V2
    plus the six FLOAT* variants). Same single-match invariant as
    :func:`resolve_category_token_ids`.
    """
    return _resolve_token_type_ids(vocab_manager, _NUMBER_TOKEN_TYPES)
