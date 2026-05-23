"""Stage 4b unit tests -- ALG-9 prepend slot writes.

Single concern: validate that
:func:`tokenizer.aligned_data.loader.batch_decode._prepend.write_prepend_slot`
emits exactly the right two writes (token id at ``tokens[row, column]``
plus dedup counter at ``identities_flat_caller_local[identity_slice_start]``)
per plan ALG-9, and rejects EXT_FUNC per plan D3.

The shifted vocab ids are derived from
:class:`tokenizer.token_manager.VocabularyManager` at import time -- one
test below re-derives them locally and pins the expected values (9 / 10)
so that a future canonical-block extension surfaces here too, not only
in the production module.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._prepend import (
    _LOCAL_FUNC_SHIFTED,
    _PLT_FUNC_SHIFTED,
    write_prepend_slot,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_tokens(batch_size: int = 4, context_len: int = 16) -> np.ndarray:
    """``u16[batch_size, context_len]`` initialized to id 0 (null-content
    per plan D5) -- mirrors the stage-4 starting tensor."""
    return np.zeros((batch_size, context_len), dtype=np.uint16)


def _fresh_identities(n: int = 32) -> np.ndarray:
    """``u16[n]`` initialized to id 0 -- mirrors a freshly-allocated
    caller-local sidecar before stage-4 writes."""
    return np.zeros((n,), dtype=np.uint16)


# ---------------------------------------------------------------------------
# Shifted vocab id sanity (pin the post-shift IDENTITY-block offsets)
# ---------------------------------------------------------------------------


def test_shifted_constants_pin_to_canonical_vocab_layout() -> None:
    """LOCAL_FUNC=265, PLT_FUNC=266 in the canonical vocab; post-shift
    (-256) they are 9 and 10 per the IDENTITY band layout (plan D5)."""

    # Re-derive locally from the anchor instead of hardcoding -- this
    # mirrors the production module's derivation and surfaces drift if
    # the canonical block layout ever moves.
    expected_local = (
        VocabularyManager._V2_IDENTITY_BLOCK_START
        + 1
        - VocabularyManager._V2_RESERVED_DIGIT_COUNT
    )
    expected_plt = (
        VocabularyManager._V2_IDENTITY_BLOCK_START
        + 2
        - VocabularyManager._V2_RESERVED_DIGIT_COUNT
    )

    assert _LOCAL_FUNC_SHIFTED == expected_local == 9
    assert _PLT_FUNC_SHIFTED == expected_plt == 10


# ---------------------------------------------------------------------------
# Happy path: per-Category writes
# ---------------------------------------------------------------------------


def test_local_func_writes_shifted_id_9_and_counter_at_slice_start() -> None:
    """LOCAL_FUNC encounter -> tokens[row, column] == 9; identity at
    ``identity_slice_start`` == self_counter."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()

    write_prepend_slot(
        tokens,
        identities,
        row=2,
        column=5,
        identity_slice_start=7,
        encounter_category=Category.LOCAL_FUNC,
        self_counter=3,
    )

    assert tokens[2, 5] == 9
    assert identities[7] == 3


def test_plt_func_writes_shifted_id_10() -> None:
    """PLT_FUNC encounter -> tokens[row, column] == 10."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()

    write_prepend_slot(
        tokens,
        identities,
        row=1,
        column=0,
        identity_slice_start=4,
        encounter_category=Category.PLT_FUNC,
        self_counter=2,
    )

    assert tokens[1, 0] == 10
    assert identities[4] == 2


# ---------------------------------------------------------------------------
# Rejected category
# ---------------------------------------------------------------------------


def test_ext_func_raises_assertion_error() -> None:
    """EXT_FUNC has no inlined body per plan D3 -- the write must
    refuse it (assertion failure, not silent miswrite)."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()

    with pytest.raises(AssertionError):
        write_prepend_slot(
            tokens,
            identities,
            row=0,
            column=0,
            identity_slice_start=0,
            encounter_category=Category.EXT_FUNC,
            self_counter=0,
        )

    # Confirm the failure aborted BEFORE any write reached either array
    # (no half-state on the failure path).
    assert tokens.sum() == 0
    assert identities.sum() == 0


@pytest.mark.parametrize(
    "non_function_category",
    [
        Category.BLOCK,
        Category.RO_DATA_PTR,
        Category.RW_DATA_PTR,
        Category.STRING_PTR,
        Category.JUMP_TABLE,
    ],
)
def test_non_function_categories_rejected(non_function_category: Category) -> None:
    """Only LOCAL_FUNC and PLT_FUNC have inlined bodies per plan D3 +
    ALG-9; any other Category must be refused."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()

    with pytest.raises(AssertionError):
        write_prepend_slot(
            tokens,
            identities,
            row=0,
            column=0,
            identity_slice_start=0,
            encounter_category=non_function_category,
            self_counter=0,
        )


# ---------------------------------------------------------------------------
# Counter values
# ---------------------------------------------------------------------------


def test_root_counter_zero_writes_zero_at_slice_start() -> None:
    """Root call_target's ``self_counter`` is 0 (LOCAL_FUNC's seeded
    self-reservation per plan D4). Initialised array stays 0 at the
    prepend slot only because the write put 0 there explicitly --
    verify by perturbing the slot first."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()
    # Pre-poison the slot to a non-zero value so we can detect that
    # the function actually performs a write (vs. relying on the
    # initialisation).
    identities[10] = 99

    write_prepend_slot(
        tokens,
        identities,
        row=0,
        column=0,
        identity_slice_start=10,
        encounter_category=Category.LOCAL_FUNC,
        self_counter=0,
    )

    assert identities[10] == 0
    assert tokens[0, 0] == 9


def test_callee_counter_greater_than_zero_writes_through() -> None:
    """An inlined callee's prepend uses the freshly-issued dedup
    counter from the parent's ALG-3 walk -- verify a non-trivial value
    propagates through."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()

    write_prepend_slot(
        tokens,
        identities,
        row=3,
        column=7,
        identity_slice_start=15,
        encounter_category=Category.LOCAL_FUNC,
        self_counter=42,
    )

    assert tokens[3, 7] == 9
    assert identities[15] == 42


# ---------------------------------------------------------------------------
# In-place + isolation invariants
# ---------------------------------------------------------------------------


def test_idempotent_under_repeated_identical_calls() -> None:
    """Calling with the same args twice yields the same final values --
    the writes are pure overwrites with no accumulating state."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()

    for _ in range(3):
        write_prepend_slot(
            tokens,
            identities,
            row=2,
            column=4,
            identity_slice_start=8,
            encounter_category=Category.PLT_FUNC,
            self_counter=5,
        )

    assert tokens[2, 4] == 10
    assert identities[8] == 5


def test_only_target_positions_are_touched() -> None:
    """The two writes must NOT bleed into neighbouring positions; any
    off-by-one would be caught by the surrounding-zero invariant."""

    tokens = _fresh_tokens(batch_size=3, context_len=8)
    identities = _fresh_identities(n=16)

    write_prepend_slot(
        tokens,
        identities,
        row=1,
        column=3,
        identity_slice_start=5,
        encounter_category=Category.LOCAL_FUNC,
        self_counter=7,
    )

    # Target writes
    assert tokens[1, 3] == 9
    assert identities[5] == 7

    # Everything else still zero. We sum and subtract the expected
    # touched-position contributions so a single ``assert ... == 0`` is
    # sufficient even though uint16 0 is the empty-content marker.
    assert int(tokens.sum()) - 9 == 0
    assert int(identities.sum()) - 7 == 0


def test_writes_mutate_in_place_not_copy() -> None:
    """The arrays passed in must be the SAME objects mutated -- the
    orchestrator (stage 4c) reads them back through its own references
    after the call. A non-trivial array identity check by ``id`` would
    be brittle; instead check that a post-call read through the
    original references sees the writes."""

    tokens = _fresh_tokens()
    identities = _fresh_identities()
    tokens_alias = tokens
    identities_alias = identities

    write_prepend_slot(
        tokens,
        identities,
        row=0,
        column=0,
        identity_slice_start=0,
        encounter_category=Category.PLT_FUNC,
        self_counter=11,
    )

    assert tokens_alias[0, 0] == 10
    assert identities_alias[0] == 11
