"""Tests for ``VocabularyManager._register_v2_canonical_blocks``.

The unifier pre-registers the canonical number- and identity-carrying
type-marker tokens at fixed slots 257..271 on the unified VM. Per-binary
VMs (constructed by the tokenize worker) do NOT eager-register these
blocks — number/identity tokens land lazily at first-seen positions like
any other token there.

These tests pin:

* The fixed-slot layout produced by the helper.
* The unified-VM-only precondition (``platform is None`` assertion).
* The fresh-VM precondition (size-check refuses repeated calls and any
  vocab whose state is past the protocol-reserved prefix).
* The block-range properties returning the canonical intervals on a VM
  that has been pre-registered, and empty intervals everywhere else.
"""

from __future__ import annotations

import pytest

from tokenizer.token_manager import VocabularyManager


# --------------------------------------------------------------------------
# Layout pin
# --------------------------------------------------------------------------


def test_register_v2_canonical_blocks_layout():
    """After the helper runs on a fresh unified VM the canonical number+
    identity blocks live at the fixed slots 257..271 and the block-range
    properties report the matching half-open intervals."""
    vm = VocabularyManager(platform=None, format_version=1)
    vm._register_v2_canonical_blocks()

    # Number block — source-declaration order.
    assert vm.get_token_id("valued_const_v2") == 257
    assert vm.get_token_id("float16") == 258
    assert vm.get_token_id("bfloat16") == 259
    assert vm.get_token_id("float32") == 260
    assert vm.get_token_id("float64") == 261
    assert vm.get_token_id("float80") == 262
    assert vm.get_token_id("float128") == 263

    # Identity block — first 5 user-canonical, then alphabetical.
    assert vm.get_token_id("block_v2") == 264
    assert vm.get_token_id("local_func") == 265
    assert vm.get_token_id("plt_func") == 266
    assert vm.get_token_id("ext_func") == 267
    assert vm.get_token_id("string_ptr") == 268
    assert vm.get_token_id("jump_table") == 269
    assert vm.get_token_id("ro_data_ptr") == 270
    assert vm.get_token_id("rw_data_ptr") == 271

    assert len(vm.id_to_token) == 272
    assert vm.number_block_range == (257, 264)
    assert vm.identity_block_range == (264, 272)


# --------------------------------------------------------------------------
# Per-binary VMs do NOT eager-register and reject the helper
# --------------------------------------------------------------------------


def test_per_binary_vm_does_not_eager_register():
    """A per-binary VM (``platform`` set) carries no canonical-block
    tokens at construction — ``get_token_id`` reports the not-found
    sentinel — and the block-range properties report empty intervals at
    the current VM size."""
    vm = VocabularyManager(platform="x64", format_version=2)
    assert vm.get_token_id("valued_const_v2") == -1
    assert vm.number_block_range == (vm.size, vm.size)
    assert vm.identity_block_range == (vm.size, vm.size)


def test_helper_rejects_per_binary_vm():
    """The helper is only meaningful on the unified VM (``platform is
    None``); calling it on a per-binary VM trips the platform assert."""
    vm = VocabularyManager(platform="x64", format_version=2)
    with pytest.raises(AssertionError):
        vm._register_v2_canonical_blocks()


# --------------------------------------------------------------------------
# Helper requires a FRESH VM — size strictly equal to the reserved prefix
# --------------------------------------------------------------------------


def test_helper_rejects_non_fresh_vm_after_first_call():
    """A second call would extend the VM past ``_V2_EAGER_BLOCK_END``,
    breaking the canonical layout; the fresh-VM size assert traps that."""
    vm = VocabularyManager(platform=None, format_version=1)
    vm._register_v2_canonical_blocks()
    with pytest.raises(AssertionError):
        vm._register_v2_canonical_blocks()


def test_helper_rejects_vm_with_caller_registered_tokens():
    """Any caller-driven registration before the helper invalidates the
    fresh-VM precondition — the size assert fires."""
    vm = VocabularyManager(platform=None, format_version=1)
    # One arbitrary registration shifts size past _V2_RESERVED_TOKEN_COUNT.
    vm.Variant_Axis("arch:x64")
    with pytest.raises(AssertionError):
        vm._register_v2_canonical_blocks()


# --------------------------------------------------------------------------
# Block-range property semantics on a fresh (un-registered) unified VM
# --------------------------------------------------------------------------


def test_unified_vm_block_ranges_empty_before_helper():
    """On a fresh unified VM (just digits + value_negative pinned) the
    block-range properties report empty intervals at the current size."""
    vm = VocabularyManager(platform=None, format_version=1)
    assert vm.number_block_range == (vm.size, vm.size)
    assert vm.identity_block_range == (vm.size, vm.size)
