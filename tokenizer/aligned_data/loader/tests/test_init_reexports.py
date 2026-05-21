"""Verify that the loader package re-exports the decoded-view surface.

These tests guard the public boundary: consumers must be able to import
``DecodedFunction``, ``splice_with_callees``, ``decode_raw_tokens``,
``Category``, the custom-float constants, the category/number token-id
resolvers, and ``IDENTITY_SENTINEL`` straight from
``tokenizer.aligned_data.loader`` (and from the nested
``tokenizer.aligned_data.loader.decoded`` package) — without reaching into
``loader.decoded.<submodule>`` or ``tokenizer.tokens`` directly.

The import statements at the top of each test ARE the assertion: a missing
re-export raises ``ImportError`` at collection time.
"""

from __future__ import annotations

import tokenizer.aligned_data.loader as loader_pkg
import tokenizer.aligned_data.loader.decoded as decoded_pkg

# --- Imports under test -----------------------------------------------------
# These imports succeeding is itself the contract being tested. We pull them
# from BOTH surfaces (the loader top-level + the nested decoded subpackage)
# so that the test fails if either re-export list regresses.

from tokenizer.aligned_data.loader.decoded import (  # noqa: F401
    Category as DecodedCategory,
    DecodedFunction as DecodedDecodedFunction,
    FID_KEYED_CATEGORIES as DECODED_FID_KEYED_CATEGORIES,
    IDENTITY_SENTINEL as DECODED_IDENTITY_SENTINEL,
    INFNAN_EXPONENT_UNBIASED as DECODED_INFNAN_EXPONENT_UNBIASED,
    TARGET_EXPONENT_BIAS as DECODED_TARGET_EXPONENT_BIAS,
    TARGET_EXPONENT_BITS as DECODED_TARGET_EXPONENT_BITS,
    TARGET_SIGNIFICAND_BITS as DECODED_TARGET_SIGNIFICAND_BITS,
    decode_raw_tokens as decoded_decode_raw_tokens,
    resolve_category_token_ids as decoded_resolve_category_token_ids,
    resolve_number_token_ids as decoded_resolve_number_token_ids,
    splice_with_callees as decoded_splice_with_callees,
)

from tokenizer.aligned_data.loader import (  # noqa: F401
    AlignedDataLoader,
    BinaryDataset,
    Category,
    DecodedFunction,
    FID_KEYED_CATEGORIES,
    FunctionData,
    IDENTITY_SENTINEL,
    INFNAN_EXPONENT_UNBIASED,
    MatchedFunction,
    TARGET_EXPONENT_BIAS,
    TARGET_EXPONENT_BITS,
    TARGET_SIGNIFICAND_BITS,
    decode_raw_tokens,
    load_single_matched_function,
    resolve_category_token_ids,
    resolve_number_token_ids,
    splice_with_callees,
)


DECODED_EXPECTED_SYMBOLS = (
    "Category",
    "DecodedFunction",
    "FID_KEYED_CATEGORIES",
    "IDENTITY_SENTINEL",
    "INFNAN_EXPONENT_UNBIASED",
    "TARGET_EXPONENT_BIAS",
    "TARGET_EXPONENT_BITS",
    "TARGET_SIGNIFICAND_BITS",
    "decode_raw_tokens",
    "resolve_category_token_ids",
    "resolve_number_token_ids",
    "splice_with_callees",
)

LOADER_PREEXISTING_PUBLIC_SYMBOLS = (
    "AlignedDataLoader",
    "BinaryDataset",
    "FunctionData",
    "MatchedFunction",
    "load_single_matched_function",
)


def test_decoded_all_contains_expected_symbols() -> None:
    assert set(DECODED_EXPECTED_SYMBOLS).issubset(set(decoded_pkg.__all__)), (
        "decoded.__all__ is missing one or more expected re-exports: "
        f"missing={set(DECODED_EXPECTED_SYMBOLS) - set(decoded_pkg.__all__)}"
    )


def test_decoded_all_has_no_unexpected_symbols() -> None:
    # The decoded package is a tight, plan-defined surface — anything outside
    # the documented symbols would be drift. If a new symbol is added by
    # design, extend DECODED_EXPECTED_SYMBOLS in the same commit.
    assert set(decoded_pkg.__all__) == set(DECODED_EXPECTED_SYMBOLS), (
        "decoded.__all__ drifted from the documented Phase 4.2 surface: "
        f"unexpected={set(decoded_pkg.__all__) - set(DECODED_EXPECTED_SYMBOLS)}"
    )


def test_fid_keyed_categories_membership_pinned() -> None:
    # Single source of truth for "which categories carry FIDs"; the splice
    # + extract pipelines branch on membership here. A drift would mean
    # the resolve / compaction passes silently treat a different category
    # as FID-keyed.
    assert DECODED_FID_KEYED_CATEGORIES == frozenset(
        {
            DecodedCategory.LOCAL_FUNC,
            DecodedCategory.PLT_FUNC,
            DecodedCategory.EXT_FUNC,
        }
    )
    # Same object on both surfaces (loader vs decoded) — re-export, not copy.
    assert FID_KEYED_CATEGORIES is DECODED_FID_KEYED_CATEGORIES


def test_decoded_all_has_no_private_names() -> None:
    leaked = [name for name in decoded_pkg.__all__ if name.startswith("_")]
    assert not leaked, f"decoded.__all__ leaks private symbols: {leaked}"


def test_loader_all_contains_decoded_surface() -> None:
    assert set(DECODED_EXPECTED_SYMBOLS).issubset(set(loader_pkg.__all__)), (
        "loader.__all__ is missing decoded re-exports: "
        f"missing={set(DECODED_EXPECTED_SYMBOLS) - set(loader_pkg.__all__)}"
    )


def test_loader_all_preserves_preexisting_public_symbols() -> None:
    assert set(LOADER_PREEXISTING_PUBLIC_SYMBOLS).issubset(set(loader_pkg.__all__)), (
        "loader.__all__ dropped pre-existing public symbols: "
        f"missing={set(LOADER_PREEXISTING_PUBLIC_SYMBOLS) - set(loader_pkg.__all__)}"
    )


def test_loader_all_has_no_private_names() -> None:
    leaked = [name for name in loader_pkg.__all__ if name.startswith("_")]
    assert not leaked, f"loader.__all__ leaks private symbols: {leaked}"


def test_decoded_and_loader_share_object_identity() -> None:
    # Re-exports must be the SAME object on both surfaces, not copies / aliases
    # that could silently drift. Object identity is the strictest check.
    for name in DECODED_EXPECTED_SYMBOLS:
        from_decoded = getattr(decoded_pkg, name)
        from_loader = getattr(loader_pkg, name)
        assert from_decoded is from_loader, (
            f"loader.{name} is not the same object as decoded.{name}; "
            "the re-export must be a single source of truth"
        )


def test_category_reexport_matches_tokens_module() -> None:
    # Specifically verify that Category surfaced through loader IS the
    # tokenizer.tokens.Category enum — consumers must NOT import a shim.
    import tokenizer.tokens as tokens_mod

    assert Category is tokens_mod.Category
