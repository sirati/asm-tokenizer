"""Tests for the Float_Annotation Inner class + TokenType.FLOAT_ANNOTATION wiring.

`float_annotation` is a width-agnostic, value-less MODIFIER marker (same wire
shape as `thread_local` / `vtable` / `code_ptr_table`): exactly one vocab id,
no payload. It marks an FP-typed pointer load whose value could not be
captured. It is NOT a NUMBER-block valued token and NOT an IDENTITY-block
token, so it must never land in or shift the canonical NUMBER (257..263) or
IDENTITY (264..271) blocks; like the other modifiers it lands lazily,
first-seen, in the instruction-rep band (>=272 once the canonical blocks are
pre-registered, >=257 on a bare VM).

Covers:
  * dispatch via `get_token_class_for_type(TokenType.FLOAT_ANNOTATION)`
  * construction: single id, no payload, correctly type-tagged
  * round-trip via representative-iteration + dispatch with a stable id
  * the canonical NUMBER/IDENTITY block ids 257..271 are UNCHANGED after
    registering the new modifier
"""

from __future__ import annotations

import pytest

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import FloatAnnotationToken, ModifierToken, TokenType


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_V2_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272


def _make_v1_vm() -> VocabularyManager:
    return VocabularyManager(platform=None, format_version=1)


# --------------------------------------------------------------------------
# Dispatch + construction shape
# --------------------------------------------------------------------------


def test_get_token_class_for_type_returns_float_annotation_class():
    vm = _make_v1_vm()
    assert vm.get_token_class_for_type(TokenType.FLOAT_ANNOTATION) is vm.Float_Annotation


def test_float_annotation_is_a_modifier():
    """Subclasses the modifier protocol — it is a value-less marker, not a
    NUMBER/IDENTITY token, so the valued-token contract does not govern it."""
    vm = _make_v1_vm()
    assert issubclass(vm.Float_Annotation, ModifierToken)
    assert issubclass(vm.Float_Annotation, FloatAnnotationToken)
    assert vm.Float_Annotation.token_type == TokenType.FLOAT_ANNOTATION


def test_float_annotation_constructs_single_value_less_id():
    """`vm.Float_Annotation()` takes no payload and emits exactly one id."""
    vm = _make_v1_vm()
    fa = vm.Float_Annotation()
    ids = fa.get_token_ids().tolist()
    assert len(ids) == 1
    # On a bare v1 VM the only prior registration is the pinned
    # value_negative marker at 256, so the first caller token lands at 257.
    assert ids[0] >= _V2_RESERVED_DIGIT_COUNT
    assert vm.id_to_token_type[ids[0]] == TokenType.FLOAT_ANNOTATION
    assert fa.to_string() == "float_annotation"
    assert fa.to_asm_like() == "float_annotation"


def test_float_annotation_registration_is_idempotent():
    """Repeat construction reuses the same vocab id (name short-circuit)."""
    vm = _make_v1_vm()
    [first] = vm.Float_Annotation().get_token_ids().tolist()
    [second] = vm.Float_Annotation().get_token_ids().tolist()
    assert first == second


def test_float_annotation_lands_in_instruction_rep_band_after_canonical_blocks():
    """With the canonical NUMBER+IDENTITY blocks pre-registered (unified VM),
    the modifier lands lazily at the first instruction-rep slot (>=272)."""
    vm = _make_v1_vm()
    vm._register_v2_canonical_blocks()
    [fa_id] = vm.Float_Annotation().get_token_ids().tolist()
    assert fa_id >= _V2_EAGER_BLOCK_END  # 272


# --------------------------------------------------------------------------
# Round-trip via representative iteration + dispatch
# --------------------------------------------------------------------------


def test_float_annotation_round_trips_with_stable_id():
    """The modifier surfaces as exactly one representative whose dispatched
    reconstruction re-emits the same single vocab id."""
    vm = _make_v1_vm()
    vm._register_v2_canonical_blocks()
    [fa_id] = vm.Float_Annotation().get_token_ids().tolist()

    reps = [t for t in vm.iter_representative_tokens()
            if t.token_type == TokenType.FLOAT_ANNOTATION]
    assert len(reps) == 1
    assert reps[0].get_token_ids().tolist() == [fa_id]

    # And the type-id dispatch reconstructs the same modifier shape.
    rebuilt = vm._make_v2_representative(TokenType.FLOAT_ANNOTATION)
    assert rebuilt.token_type == TokenType.FLOAT_ANNOTATION
    assert len(rebuilt.get_token_ids().tolist()) == 1


# --------------------------------------------------------------------------
# Canonical NUMBER/IDENTITY block invariant — 257..271 must NOT shift
# --------------------------------------------------------------------------


_CANONICAL_BLOCK_IDS = {
    # NUMBER block (257..263)
    "valued_const_v2": 257,
    "float16": 258,
    "bfloat16": 259,
    "float32": 260,
    "float64": 261,
    "float80": 262,
    "float128": 263,
    # IDENTITY block (264..271)
    "block_v2": 264,
    "local_func": 265,
    "plt_func": 266,
    "ext_func": 267,
    "string_ptr": 268,
    "jump_table": 269,
    "ro_data_ptr": 270,
    "rw_data_ptr": 271,
}


def test_float_annotation_does_not_shift_canonical_blocks():
    """Registering `float_annotation` after the canonical blocks leaves every
    canonical NUMBER/IDENTITY id (valued_const_v2==257 .. rw_data_ptr==271)
    untouched — the modifier is additive in the instruction-rep band."""
    vm = _make_v1_vm()
    vm._register_v2_canonical_blocks()

    # Pre-state: canonical ids as pinned by the helper.
    for name, expected in _CANONICAL_BLOCK_IDS.items():
        assert vm.get_token_id(name) == expected

    vm.Float_Annotation()

    # Post-state: identical — no canonical id moved.
    for name, expected in _CANONICAL_BLOCK_IDS.items():
        assert vm.get_token_id(name) == expected
    assert vm.number_block_range == (257, 264)
    assert vm.identity_block_range == (264, 272)


def test_canonical_block_helper_unaffected_by_new_modifier():
    """The eager canonical-block helper still terminates at exactly 272 — the
    new modifier is NOT part of the eager block and does not extend it."""
    vm = _make_v1_vm()
    vm._register_v2_canonical_blocks()
    assert len(vm.id_to_token) == _V2_EAGER_BLOCK_END  # 272
