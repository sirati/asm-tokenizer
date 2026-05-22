"""Tests for the vocab-introspection helpers.

The resolvers under test are pure functions of the vocab's
``id_to_token_type`` ndarray, so the unit-test stub mirrors that
contract directly: a ``SimpleNamespace`` carrying a hand-built int8
array. No real ``VocabularyManager`` is touched — the unit is the
TokenType lookup rule, not the persistence layer below it.

Reference fixture shape: the integration tests in
``tokenizer/aligned_data/loader/tests/test_unified_vocab_gate_v1.py``
use the same ``SimpleNamespace`` duck-typing pattern.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.category_tokens import (
    resolve_category_token_ids,
    resolve_number_token_ids,
)
from tokenizer.tokens import Category, TokenType


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


# The 8 Category-backing TokenTypes (BLOCK is BLOCK_V2 on v2).
_CATEGORY_TOKEN_TYPES = (
    TokenType.BLOCK_V2,
    TokenType.LOCAL_FUNC,
    TokenType.PLT_FUNC,
    TokenType.EXT_FUNC,
    TokenType.RO_DATA_PTR,
    TokenType.RW_DATA_PTR,
    TokenType.STRING_PTR,
    TokenType.JUMP_TABLE,
)

_NUMBER_TOKEN_TYPES = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)


def _build_stub_vocab(
    extra_tagged_ids: dict[int, TokenType] | None = None,
    *,
    omit: TokenType | None = None,
    duplicate: TokenType | None = None,
    misplace_low: TokenType | None = None,
) -> SimpleNamespace:
    """Build a stub vocab carrying the 8 + 7 v2 type-tokens at 256+.

    ``omit`` removes one type-token entirely (its id stays UNRESOLVED).
    ``duplicate`` adds a second id with the same TokenType tag.
    ``misplace_low`` overwrites a reserved-digit slot (<256) with the
    requested TokenType — exercises the "id below 256" guard.
    """
    capacity = 512
    arr = np.full(capacity, TokenType.UNRESOLVED, dtype=np.int8)

    # Slots 0..255 are reserved-digit slots: TokenType.UNRESOLVED.
    # Place the 15 v2 type-tokens at sequential ids starting at 256.
    next_id = 256
    for token_type in (*_CATEGORY_TOKEN_TYPES, *_NUMBER_TOKEN_TYPES):
        if token_type == omit:
            continue
        arr[next_id] = int(token_type)
        next_id += 1

    if duplicate is not None:
        arr[next_id] = int(duplicate)
        next_id += 1

    if misplace_low is not None:
        # Plant the tag at id 100 — should be rejected by the
        # "id >= 256" check.
        arr[100] = int(misplace_low)

    if extra_tagged_ids:
        for token_id, token_type in extra_tagged_ids.items():
            arr[token_id] = int(token_type)

    return SimpleNamespace(id_to_token_type=arr)


# ---------------------------------------------------------------------------
# Happy-path coverage
# ---------------------------------------------------------------------------


def test_category_resolver_returns_eight_entries() -> None:
    """Returns exactly 8 entries, keyed by every ``Category`` member."""
    stub = _build_stub_vocab()
    result = resolve_category_token_ids(stub)

    assert set(result.keys()) == set(Category)
    assert len(result) == 8
    assert all(v >= 256 for v in result.values())


def test_category_resolver_block_maps_to_block_v2() -> None:
    """``Category.BLOCK`` resolves to the BLOCK_V2-tagged vocab id, not the
    legacy BLOCK id."""
    stub = _build_stub_vocab(
        extra_tagged_ids={300: TokenType.BLOCK}  # legacy BLOCK at a distinct id
    )
    result = resolve_category_token_ids(stub)

    block_id = result[Category.BLOCK]
    arr = stub.id_to_token_type
    assert arr[block_id] == int(TokenType.BLOCK_V2)
    # The legacy BLOCK id is NOT the one we picked up.
    assert block_id != 300


def test_number_resolver_returns_seven_entries() -> None:
    """Returns one entry per number-carrying TokenType (7)."""
    stub = _build_stub_vocab()
    result = resolve_number_token_ids(stub)

    expected_keys = {
        TokenType.VALUED_CONST_V2,
        TokenType.FLOAT16,
        TokenType.BFLOAT16,
        TokenType.FLOAT32,
        TokenType.FLOAT64,
        TokenType.FLOAT80,
        TokenType.FLOAT128,
    }
    assert set(result.keys()) == expected_keys
    assert len(result) == 7
    assert all(v >= 256 for v in result.values())


def test_resolvers_return_consistent_ids_with_vocab_layout() -> None:
    """Each returned id, looked up in ``id_to_token_type``, matches its
    requested TokenType — round-trips the lookup contract."""
    stub = _build_stub_vocab()
    arr = stub.id_to_token_type

    cat_ids = resolve_category_token_ids(stub)
    num_ids = resolve_number_token_ids(stub)

    category_to_token_type = {
        Category.BLOCK: TokenType.BLOCK_V2,
        Category.LOCAL_FUNC: TokenType.LOCAL_FUNC,
        Category.PLT_FUNC: TokenType.PLT_FUNC,
        Category.EXT_FUNC: TokenType.EXT_FUNC,
        Category.RO_DATA_PTR: TokenType.RO_DATA_PTR,
        Category.RW_DATA_PTR: TokenType.RW_DATA_PTR,
        Category.STRING_PTR: TokenType.STRING_PTR,
        Category.JUMP_TABLE: TokenType.JUMP_TABLE,
    }
    for category, token_type in category_to_token_type.items():
        assert arr[cat_ids[category]] == int(token_type)

    for token_type, vocab_id in num_ids.items():
        assert arr[vocab_id] == int(token_type)


def test_resolvers_are_idempotent() -> None:
    """Calling twice returns equal dicts (no hidden state, no mutation)."""
    stub = _build_stub_vocab()

    first_cat = resolve_category_token_ids(stub)
    second_cat = resolve_category_token_ids(stub)
    assert first_cat == second_cat

    first_num = resolve_number_token_ids(stub)
    second_num = resolve_number_token_ids(stub)
    assert first_num == second_num


# ---------------------------------------------------------------------------
# Failure-mode coverage
# ---------------------------------------------------------------------------


def test_missing_category_silently_omitted() -> None:
    """A vocab that lacks one Category's TokenType silently omits that
    Category from the returned dict.  No raise: corpora may legitimately
    lack TokenTypes whose source data was never encountered."""
    stub = _build_stub_vocab(omit=TokenType.JUMP_TABLE)

    result = resolve_category_token_ids(stub)

    assert Category.JUMP_TABLE not in result
    assert len(result) == 7
    # Every other Category is still present + at a valid id.
    for category in Category:
        if category is Category.JUMP_TABLE:
            continue
        assert category in result
        assert result[category] >= 256


def test_missing_number_type_silently_omitted() -> None:
    """A vocab that lacks one number TokenType silently omits that
    TokenType from the returned dict; the result has fewer than 7 keys
    and the other number types are still resolved."""
    stub = _build_stub_vocab(omit=TokenType.FLOAT128)

    result = resolve_number_token_ids(stub)

    assert TokenType.FLOAT128 not in result
    assert len(result) == 6
    for token_type in _NUMBER_TOKEN_TYPES:
        if token_type is TokenType.FLOAT128:
            continue
        assert token_type in result
        assert result[token_type] >= 256


def test_resolvers_return_empty_when_vocab_has_no_v2_type_tokens() -> None:
    """A vocab carrying ONLY digit tokens (no v2 type-token at >=256)
    returns an empty dict from both resolvers — every TokenType is
    absent, every TokenType is silently omitted, no raise."""
    capacity = 512
    arr = np.full(capacity, TokenType.UNRESOLVED, dtype=np.int8)
    stub = SimpleNamespace(id_to_token_type=arr)

    cat_result = resolve_category_token_ids(stub)
    num_result = resolve_number_token_ids(stub)

    assert cat_result == {}
    assert num_result == {}


def test_duplicate_category_type_raises() -> None:
    """A vocab that tags two distinct ids with the same TokenType raises."""
    stub = _build_stub_vocab(duplicate=TokenType.LOCAL_FUNC)

    with pytest.raises(ValueError) as excinfo:
        resolve_category_token_ids(stub)

    message = str(excinfo.value)
    assert "LOCAL_FUNC" in message
    assert "2 entries" in message or "Conflicting" in message


def test_low_id_tag_raises() -> None:
    """A type-token whose vocab id lives in the reserved digit range is rejected."""
    # Build a vocab where FLOAT32 ONLY appears at id 100 (a digit slot).
    stub = _build_stub_vocab(omit=TokenType.FLOAT32, misplace_low=TokenType.FLOAT32)

    with pytest.raises(ValueError) as excinfo:
        resolve_number_token_ids(stub)

    message = str(excinfo.value)
    assert "FLOAT32" in message
    assert "256" in message or "digit-slot" in message
