"""End-to-end tests for the BatchDecode INSTR_REP text rendering path.

Single concern: pin the R2b integration -- the row walker's
:func:`emit_instr_rep` composes :func:`substitute_mem_chars`
(W3-3 W4-amended; the shared MEM-symbol display table) with
:func:`strip_arch_prefix` (the BatchDecode-side arch elision
helper) so the resulting :class:`AsmLine` text mirrors the FTL
backend's ``to_asm_like`` form.

The fixtures stand in a fake :class:`VocabularyManager` whose
:meth:`get_token_str` returns hand-crafted vocab strings keyed by
the original (pre-strip) id; the per-id mapping pins both the
MEM-bracket substitution path (``MEM_OPEN_BRACKET`` -> ``[``) and
the arch-prefix elision path (``x64_push`` under arch ``x86_64``
-> ``push``).

Plan reference: ``inspector-followup.md`` §A.2 (W4-amended) +
§R2b.
"""

from __future__ import annotations

from typing import Mapping
from unittest.mock import MagicMock

import numpy as np

from tokenizer.inspector._render._batch_decode_backend._arch_prefix import (
    arch_prefix_tuple,
)
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._protocol import AsmLine
from tokenizer.token_manager import VocabularyManager

from ._row_walk_fixtures import (
    BLOCK_V2,
    EMPTY_FID_COUNTS,
    EMPTY_FID_SIDECAR,
    EMPTY_NUMBERS,
    NULL_CALLEE_RESOLVER,
    make_fid_table,
    make_result,
)


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT


def _vocab_from_mapping(per_id: Mapping[int, str]) -> MagicMock:
    """Vocab stub with a hand-mapped ``original_id -> str`` table.

    Falls back to ``f"id_{token_id}"`` for any id not in ``per_id`` so
    tests only need to populate the ids they exercise.
    """
    vm = MagicMock(spec=VocabularyManager)
    vm.get_token_str = lambda token_id: per_id.get(
        token_id, f"id_{token_id}"
    )
    return vm


def _walk_one_row(
    *,
    tokens_row: np.ndarray,
    vocab_manager: MagicMock,
    arch_prefixes: tuple[str, ...] = (),
):
    """Shorthand: ``render_row_blocks`` over a single-CT n_axis=0 row.

    Numbers, identities, runlengths default empty; the caller supplies
    only the tokens + vocab mapping needed to exercise INSTR_REP
    rendering.
    """
    numbers_sig, numbers_se = EMPTY_NUMBERS
    return render_row_blocks(
        result=make_result(
            tokens_row=tokens_row,
            identities=np.asarray([0], dtype=np.uint16),
            numbers_sig=numbers_sig,
            numbers_se=numbers_se,
        ),
        row=0, n_axis=0,
        partial_cut_lengths=[int(tokens_row.shape[0])],
        call_targets_per_ct=[[]],
        vocab_manager=vocab_manager,
        fid_table=make_fid_table(
            per_category_counts=EMPTY_FID_COUNTS,
            sidecar=EMPTY_FID_SIDECAR,
        ),
        line_to_name={}, line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
        arch_prefixes=arch_prefixes,
    )


def test_arch_prefixed_mnemonic_renders_without_prefix() -> None:
    """``x64_push`` with arch_prefixes ``("x64_", "x_", "unified_x86_")``
    renders as ``push`` -- the per-ISA prefix wins over the family /
    unified prefixes (first ``startswith`` match in
    :func:`strip_arch_prefix`).
    """
    push_shifted_id = 17  # arbitrary INSTR_REP-band id; >= IDENTITY_HI.
    original_id = push_shifted_id + _V2_RESERVED_DIGIT_COUNT
    vocab = _vocab_from_mapping({original_id: "x64_push"})
    blocks = _walk_one_row(
        tokens_row=np.asarray([BLOCK_V2, push_shifted_id, 0], dtype=np.uint16),
        vocab_manager=vocab,
        arch_prefixes=arch_prefix_tuple("x86_64"),
    )
    items = blocks[0].items
    assert len(items) == 1
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "push"


def test_unified_family_prefix_elision() -> None:
    """``unified_x86_mov`` under arch ``x86_64`` strips the unified
    family prefix (the last of the three prefix tiers, when neither
    per-ISA nor family matches).
    """
    mov_shifted_id = 18
    original_id = mov_shifted_id + _V2_RESERVED_DIGIT_COUNT
    vocab = _vocab_from_mapping({original_id: "unified_x86_mov"})
    blocks = _walk_one_row(
        tokens_row=np.asarray([BLOCK_V2, mov_shifted_id, 0], dtype=np.uint16),
        vocab_manager=vocab,
        arch_prefixes=arch_prefix_tuple("x86_64"),
    )
    items = blocks[0].items
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "mov"


def test_mem_open_bracket_substitutes_to_display_char() -> None:
    """``MEM_OPEN_BRACKET`` vocab string substitutes to ``"["`` via
    :func:`substitute_mem_chars`; the arch-prefix elision is a no-op
    because the substituted text does not start with any prefix
    (W3-3 W4-amended substitution covers all SIX MEM symbols).
    """
    open_bracket_id = 19
    original_id = open_bracket_id + _V2_RESERVED_DIGIT_COUNT
    vocab = _vocab_from_mapping({original_id: "MEM_OPEN_BRACKET"})
    blocks = _walk_one_row(
        tokens_row=np.asarray(
            [BLOCK_V2, open_bracket_id, 0], dtype=np.uint16,
        ),
        vocab_manager=vocab,
        arch_prefixes=arch_prefix_tuple("x86_64"),
    )
    items = blocks[0].items
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "["


def test_no_arch_prefix_when_arch_prefixes_empty() -> None:
    """Default ``arch_prefixes=()`` (e.g. legacy tests / fixtures that
    don't plumb the arch through) leaves the vocab text unchanged
    after the MEM substitution; no arch prefix is stripped.
    """
    raw_id = 20
    original_id = raw_id + _V2_RESERVED_DIGIT_COUNT
    vocab = _vocab_from_mapping({original_id: "x64_pop"})
    blocks = _walk_one_row(
        tokens_row=np.asarray([BLOCK_V2, raw_id, 0], dtype=np.uint16),
        vocab_manager=vocab,
    )
    items = blocks[0].items
    assert isinstance(items[0], AsmLine)
    # No arch elision -> raw vocab text passes through (MEM-substitute
    # is a no-op for non-MEM atoms).
    assert items[0].text == "x64_pop"


def test_mem_symbols_substitute_independent_of_arch_prefix() -> None:
    """Every emitted MEM symbol -- not just OPEN_BRACKET -- maps to its
    display char via the shared substitution table. Spot-check via
    CLOSE_BRACKET (the other bracket; +, -, *, ',' are identity).
    """
    close_bracket_id = 21
    original_id = close_bracket_id + _V2_RESERVED_DIGIT_COUNT
    vocab = _vocab_from_mapping({original_id: "MEM_CLOSE_BRACKET"})
    blocks = _walk_one_row(
        tokens_row=np.asarray(
            [BLOCK_V2, close_bracket_id, 0], dtype=np.uint16,
        ),
        vocab_manager=vocab,
        arch_prefixes=arch_prefix_tuple("aarch64"),
    )
    items = blocks[0].items
    assert isinstance(items[0], AsmLine)
    assert items[0].text == "]"
