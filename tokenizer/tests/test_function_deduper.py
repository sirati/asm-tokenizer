"""Tests for the per-binary semantic-merge gate.

Two concerns under test, separated cleanly:

1. ``FunctionDeduper`` (in :mod:`tokenizer.function_deduper`) - the
   gate itself. The three-way condition (same name + same identity_key
   + same content) and the ``identity_key=None`` short-circuit (legacy
   disambiguation passthrough).

2. ``FunctionDataManager.add_function_data`` (in
   :mod:`tokenizer.function_data_manager`) integration with the gate.
   Same-key duplicate folds into the existing FID and returns its
   final name; different identity_key or different content keeps the
   legacy ``_N``-suffix path.
"""

from __future__ import annotations

from tokenizer.function_data_manager import FunctionData, FunctionDataManager
from tokenizer.function_deduper import FunctionDeduper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fd(tokens_base64: str) -> FunctionData:
    """Build a minimal FunctionData carrying a controllable token body.

    The deduper only consults ``tokens_base64``; the other fields are
    inert for these tests so we pass ``None`` placeholders (the
    dataclass runtime is structural, not type-checked).
    """
    return FunctionData(
        tokens=None,  # type: ignore[arg-type]
        tokens_base64=tokens_base64,
        block_runlength_base64="",
        instruction_runlength_base64="",
        metadata_cell="",
    )


# ---------------------------------------------------------------------------
# FunctionDeduper - the gate
# ---------------------------------------------------------------------------


def test_deduper_first_call_with_identity_key_is_not_duplicate() -> None:
    """A first encounter with an identity_key registers + returns
    False (the gate has nothing to fold into yet)."""
    deduper = FunctionDeduper()
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is False


def test_deduper_same_name_same_key_same_body_is_duplicate() -> None:
    """Three-way match: the gate folds (returns True on the second
    call)."""
    deduper = FunctionDeduper()
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is False
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is True


def test_deduper_same_name_different_key_is_not_duplicate() -> None:
    """A second call with the same name but a different identity_key
    is a genuine name-collision (different external resolved-to);
    legacy disambiguation path."""
    deduper = FunctionDeduper()
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is False
    assert deduper.is_duplicate("strcmp", 0xBEEF, "AAAA") is False


def test_deduper_same_name_same_key_different_body_is_not_duplicate() -> None:
    """Three-way check: when content diverges, the fold is rejected
    (the gate is the AND of all three)."""
    deduper = FunctionDeduper()
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is False
    assert deduper.is_duplicate("strcmp", 0xDEAD, "BBBB") is False


def test_deduper_first_recorded_body_wins_after_divergence() -> None:
    """When a same-(name, key) pair surfaces with a divergent body
    (rejected fold), the gate keeps the FIRST body as the canonical
    one. A subsequent matching call against the first body still
    folds; a matching call against the second does not."""
    deduper = FunctionDeduper()
    deduper.is_duplicate("strcmp", 0xDEAD, "AAAA")  # records AAAA
    deduper.is_duplicate("strcmp", 0xDEAD, "BBBB")  # divergent, rejected
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is True
    assert deduper.is_duplicate("strcmp", 0xDEAD, "BBBB") is False


def test_deduper_none_identity_key_never_dedupes() -> None:
    """``identity_key=None`` is the provider's "no identity beyond
    name" signal. Such calls are never duplicates and never recorded;
    a downstream legacy disambiguator (occurrence-suffix) handles them."""
    deduper = FunctionDeduper()
    assert deduper.is_duplicate("strcmp", None, "AAAA") is False
    assert deduper.is_duplicate("strcmp", None, "AAAA") is False


def test_deduper_none_then_keyed_recorded_independently() -> None:
    """``None`` calls do not pollute the keyed map; a later
    keyed call with the same name is recorded fresh."""
    deduper = FunctionDeduper()
    deduper.is_duplicate("strcmp", None, "AAAA")
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is False
    assert deduper.is_duplicate("strcmp", 0xDEAD, "AAAA") is True


# ---------------------------------------------------------------------------
# FunctionDataManager - the gate integrated into the manager API
# ---------------------------------------------------------------------------


def test_fdm_same_name_same_key_same_body_folds_into_existing_fid() -> None:
    """The merge condition holds: the second call is folded (no new
    slot consumed, the first record's final name is returned)."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    final_b = mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("AAAA"), identity_key=0xDEAD
    )
    assert final_a == "strcmp"
    assert final_b == "strcmp"  # folded, same FID
    assert mgr.get_used_count() == 1


def test_fdm_same_name_different_key_keeps_n_suffix() -> None:
    """Different identity_key (genuine collision) preserves the
    legacy ``_N``-suffix disambiguation."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    final_b = mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("AAAA"), identity_key=0xBEEF
    )
    assert final_a == "strcmp"
    assert final_b == "strcmp_1"
    assert mgr.get_used_count() == 2


def test_fdm_same_name_same_key_different_body_keeps_n_suffix() -> None:
    """Same (name, identity_key) but different content: NOT folded
    (regression-guards the same-content condition; we don't merge
    superficially-similar thunks whose bodies disagree)."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    final_b = mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("BBBB"), identity_key=0xDEAD
    )
    assert final_a == "strcmp"
    assert final_b == "strcmp_1"
    assert mgr.get_used_count() == 2


def test_fdm_none_identity_key_preserves_legacy_behaviour() -> None:
    """``identity_key=None`` for both: legacy ``_N``-suffix path
    runs unchanged (no fold)."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "ctor", 0x1000, "disas_a", "tok_a", _fd("AAAA")
    )
    final_b = mgr.add_function_data(
        "ctor", 0x2000, "disas_b", "tok_b", _fd("AAAA")
    )
    assert final_a == "ctor"
    assert final_b == "ctor_1"
    assert mgr.get_used_count() == 2


def test_fdm_fold_returns_existing_address_through_lookup() -> None:
    """A folded second call leaves the first record's address as the
    canonical mapping for that name (the second PLT slot's address is
    intentionally dropped; both call sites that reference either
    address resolve to the same logical function)."""
    mgr = FunctionDataManager(total_functions=4)
    mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("AAAA"), identity_key=0xDEAD
    )
    assert mgr.get_function_addr("strcmp", 0) == 0x1000
