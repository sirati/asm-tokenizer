"""Invariant tests for the DecodedFunction dataclass.

Covers every check inside ``__post_init__`` plus frozenness + zero-length
edge cases. The read-only-array contract is documented but not enforced
in Python without an explicit ``writeable=False`` toggle, which would
require allocator-level cooperation; tests below pin the consumer rule
via comment, not assertion.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.decoded_function import DecodedFunction
from tokenizer.tokens import Category


def _empty_identities() -> dict:
    """All 8 Category members map to length-0 uint16 arrays."""
    return {c: np.empty(0, dtype=np.uint16) for c in Category}


def _make_valid_kwargs() -> dict:
    """Baseline-valid constructor kwargs for happy-path-derivative tests."""
    return dict(
        real_tokens=np.array([300, 301, 302], dtype=np.uint16),
        identities=_empty_identities(),
        numbers_significant=np.array([1, 2], dtype=np.uint64),
        numbers_sign_exponent=np.array([10, 20], dtype=np.uint32),
        func_name="root_fn",
        metadata={"origin": "test"},
    )


def test_happy_path_constructs_cleanly():
    identities = _empty_identities()
    identities[Category.BLOCK] = np.array([0, 1, 2], dtype=np.uint16)
    identities[Category.LOCAL_FUNC] = np.array([5], dtype=np.uint16)

    df = DecodedFunction(
        real_tokens=np.array([256, 300, 0xFFFE], dtype=np.uint16),
        identities=identities,
        numbers_significant=np.array([42, 7], dtype=np.uint64),
        numbers_sign_exponent=np.array([100, 200], dtype=np.uint32),
        func_name="happy_fn",
        metadata={"k": "v"},
    )

    assert df.func_name == "happy_fn"
    assert df.metadata == {"k": "v"}
    assert df.real_tokens.dtype == np.uint16
    # All 8 categories present.
    assert set(df.identities.keys()) == set(Category)


def test_zero_length_arrays_construct_cleanly():
    df = DecodedFunction(
        real_tokens=np.empty(0, dtype=np.uint16),
        identities=_empty_identities(),
        numbers_significant=np.empty(0, dtype=np.uint64),
        numbers_sign_exponent=np.empty(0, dtype=np.uint32),
        func_name="empty_fn",
        metadata={},
    )
    assert df.real_tokens.size == 0
    assert df.numbers_significant.size == 0
    assert all(arr.size == 0 for arr in df.identities.values())


def test_sentinel_0xffff_in_identity_array_accepted():
    """Per plan Locked-in decision 7: 0xFFFF is the legitimate sentinel."""
    identities = _empty_identities()
    identities[Category.STRING_PTR] = np.array(
        [0, 0xFFFF, 5, 0xFFFF], dtype=np.uint16
    )
    df = DecodedFunction(
        real_tokens=np.array([300], dtype=np.uint16),
        identities=identities,
        numbers_significant=np.empty(0, dtype=np.uint64),
        numbers_sign_exponent=np.empty(0, dtype=np.uint32),
        func_name="sentinel_fn",
        metadata={},
    )
    assert df.identities[Category.STRING_PTR][1] == 0xFFFF


def test_mismatched_numbers_array_lengths_raise():
    kw = _make_valid_kwargs()
    kw["numbers_significant"] = np.array([1, 2, 3], dtype=np.uint64)
    kw["numbers_sign_exponent"] = np.array([10, 20], dtype=np.uint32)
    with pytest.raises(ValueError, match="3.*2|2.*3"):
        DecodedFunction(**kw)


def test_wrong_real_tokens_dtype_raises():
    kw = _make_valid_kwargs()
    kw["real_tokens"] = np.array([1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="real_tokens.*uint16"):
        DecodedFunction(**kw)

    kw["real_tokens"] = np.array([1, 2, 3], dtype=np.uint32)
    with pytest.raises(ValueError, match="real_tokens.*uint16"):
        DecodedFunction(**kw)


def test_wrong_numbers_significant_dtype_raises():
    kw = _make_valid_kwargs()
    kw["numbers_significant"] = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="numbers_significant.*uint64"):
        DecodedFunction(**kw)


def test_wrong_numbers_sign_exponent_dtype_raises():
    kw = _make_valid_kwargs()
    kw["numbers_sign_exponent"] = np.array([1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="numbers_sign_exponent.*uint32"):
        DecodedFunction(**kw)


def test_wrong_identity_dtype_raises_with_category_name():
    kw = _make_valid_kwargs()
    kw["identities"][Category.JUMP_TABLE] = np.array([1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="JUMP_TABLE.*uint16"):
        DecodedFunction(**kw)


def test_missing_category_key_raises_naming_it():
    kw = _make_valid_kwargs()
    del kw["identities"][Category.EXT_FUNC]
    with pytest.raises(ValueError, match="EXT_FUNC"):
        DecodedFunction(**kw)


def test_extra_non_category_key_raises():
    kw = _make_valid_kwargs()
    kw["identities"]["bogus_string_key"] = np.empty(0, dtype=np.uint16)
    with pytest.raises(ValueError, match="not Category"):
        DecodedFunction(**kw)


def test_empty_func_name_raises():
    kw = _make_valid_kwargs()
    kw["func_name"] = ""
    with pytest.raises(ValueError, match="func_name"):
        DecodedFunction(**kw)


def test_non_string_func_name_raises():
    kw = _make_valid_kwargs()
    kw["func_name"] = 42  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="func_name"):
        DecodedFunction(**kw)


def test_non_dict_metadata_raises():
    kw = _make_valid_kwargs()
    kw["metadata"] = [("k", "v")]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata"):
        DecodedFunction(**kw)


def test_frozen_dataclass_blocks_attribute_assignment():
    df = DecodedFunction(**_make_valid_kwargs())
    with pytest.raises(FrozenInstanceError):
        df.func_name = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        df.real_tokens = np.empty(0, dtype=np.uint16)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Read-only consumer contract (documentation pin -- NOT runtime-enforced).
#
# DecodedFunction does NOT copy the arrays handed in by its producers
# (extract.py / splice.py): the dataclass holds the same buffers the
# splicer's identity-rebase + multi-chunk number-alignment passes operate
# on. Consumers that mutate any field in place corrupt that bookkeeping
# for any subsequent splice pass that re-uses the same view. The class
# docstring states this contract; this test exists only to keep the rule
# visible alongside the invariant suite.
# ---------------------------------------------------------------------------
