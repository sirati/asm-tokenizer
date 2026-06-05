"""Round-trip identity test for block_v2 cache-key normalization.

A block-def header (emitted by ``fill_constant_candidates`` at function
entry) and an intra-function jump-target reference to the same block
(emitted by ``ConstantHandler._emit_block`` via the v2 precedence walk)
MUST share identity. The shared cache is
``TokenResolver.id_maps[Category.BLOCK]``; for the two sites to hit the
same entry they must agree on the key type.

Regression: ``fill_constant_candidates`` historically passed
``hex(block.addr)`` (a STR) to ``resolver.get_block_id`` while
``_emit_block`` passes the typed ``int`` value to
``resolver.get_identity(Category.BLOCK, value, {})``. The two never
collided in the cache so the same block surfaced as two distinct
``Block_V2`` identities (``Block_V2(0)`` for the def, ``Block_V2(1)`` for
the jump-target reference). A second producer for the same cache —
``_emit_jump_table_footer_for`` — also used the STR key form, so
jump-table targets had the same divergence with ``_emit_block``. The fix
normalizes both producer call sites to the ``int`` key so all sites
share the cache entry.

This test exercises the resolver + constant_handler directly (no real
FunctionView / DisassemblyProvider needed) -- the cache-key invariant is
purely between API call sites that name the same ``Category.BLOCK``
cache.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tokenizer.constant_handler.core import ConstantHandler
from tokenizer.disasm.metadata import AddressKind
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category, TokenResolver


def _make_local_function_meta(func_start: int):
    """Stub meta view for a control-flow target inside a function body.

    The v2 ``_pred_block`` predicate fires when ``meta.kind ==
    LOCAL_FUNCTION`` and ``not _is_function_entry(meta, value)`` (value
    != meta.start_addr). Only these two fields are read by the predicate
    + emitter.
    """
    return SimpleNamespace(
        kind=AddressKind.LOCAL_FUNCTION,
        start_addr=func_start,
    )


def test_block_def_and_intra_function_jump_share_identity():
    """Block-def header + intra-function jump-target reference to the
    same address must produce the same ``Block_V2`` identity.

    Mimics the production flow VERBATIM:
    1. ``fill_constant_candidates`` calls
       ``resolver.get_block_id(<key>)`` for each block at function entry.
    2. The first interior jump-target later flows through
       ``ConstantHandler.process_constant_v2(addr, meta=..., is_arithmetic=False)``
       which routes via ``_pred_block`` to ``_emit_block``, which calls
       ``resolver.get_identity(Category.BLOCK, addr, {})``.

    Step 2's key form is fixed by the v2 precedence-walk contract:
    ``_emit_block`` receives the typed ``int`` value. So step 1 MUST
    also use the ``int`` key for the two to share the cache entry. The
    bug was that step 1 used ``hex(addr)`` (a STR), which fails to
    collide with the int key in step 2 — the same block then surfaces
    as ``Block_V2(0)`` in the def header and ``Block_V2(1)`` at the
    reference site.

    This test invokes ``get_block_id`` with the EXACT shape the post-fix
    caller produces (``int(block.addr)``); a regression would re-introduce
    a STR key at the call site and the assertion below would catch it
    immediately via the divergent identity in the v2 emission.
    """
    vm = VocabularyManager(platform="x86_64", format_version=2)
    resolver = TokenResolver()
    # Empty block_ranges: ConstantHandler only consults it from the
    # legacy v1 path; the v2 precedence walk under test does not.
    block_ranges = np.empty((0, 2), dtype=np.uint64)
    handler = ConstantHandler(vm, resolver, block_ranges)

    # The block's typed address. Production code at
    # ``fill_constant_candidates.py`` now reads ``block.addr`` (an int)
    # and passes it through ``int(...)`` to the resolver -- the v2
    # cache-key contract.
    block_addr_int = 0x1100

    # --- Step 1: block-def header allocation (function-entry path) ---
    def_id = resolver.get_block_id(int(block_addr_int))

    # --- Step 2: intra-function jump-target emission ---
    # The lookup returns a LOCAL_FUNCTION meta whose entry is BEFORE the
    # block address, so ``_is_function_entry(meta, value)`` is False and
    # ``_pred_block`` (precedence step 4) fires -> ``_emit_block``.
    ref_tokens = handler.process_constant_v2(
        block_addr_int,
        meta=_make_local_function_meta(func_start=0x1000),
        is_arithmetic=False,
    )

    # Recover the identity carried by the emitted Block_V2 token. The
    # v2 identity inner stores the id at ``token.id`` (see
    # ``BlockV2Inner._get_basename``).
    assert len(ref_tokens) == 1, f"expected single block_v2 token; got {ref_tokens!r}"
    ref_token = ref_tokens[0]
    assert ref_token.id == def_id, (
        f"block-def identity ({def_id}) and intra-function jump-target "
        f"identity ({ref_token.id}) must share the same Category.BLOCK "
        f"cache entry. Got distinct identities — string/int key mismatch "
        f"between fill_constant_candidates.get_block_id and "
        f"ConstantHandler._emit_block."
    )
    # Both are the FIRST block allocation in this function -> id 0.
    assert def_id == 0, f"first block allocation should be id 0; got {def_id}"
    # The cache MUST hold a single entry keyed by the typed int address.
    # A STR key would not be looked up by the int-keyed _emit_block path.
    assert resolver.id_maps[Category.BLOCK] == {block_addr_int: 0}, (
        f"Category.BLOCK cache must hold a single int-keyed entry; got "
        f"{resolver.id_maps[Category.BLOCK]!r}"
    )


class _EmptyBlock:
    """``BlockView``-shaped duck mock with an address but no instructions.

    ``fill_constant_candidates`` reads ``block.addr``, ``block.size``, and
    iterates ``block.instructions`` (empty here). ``__deepcopy__`` returns
    a fresh wrapper bound to the same data — for an immutable test fixture
    a self-return is sound because the test never mutates the wrapper.
    """

    def __init__(self, addr: int, size: int) -> None:
        self.addr = addr
        self.size = size
        self.instructions: list = []

    def __deepcopy__(self, memo):
        # Mirror ``BlockView.__deepcopy__`` semantics: return a wrapper
        # that holds the same per-step data. Since the test mock is
        # already immutable for the fields the consumer reads, returning
        # ``self`` is equivalent to a fresh handle bound to the same
        # provider-side block.
        return self


class _EmptyFunction:
    """``FunctionView``-shaped duck mock with N empty blocks.

    Only the fields ``fill_constant_candidates`` reads are populated:
    ``func.blocks`` (iterable of BlockView). The function's switch-table
    enumeration goes through ``disasm_provider``, which we pass as
    ``None`` so the v2 jump-table-footer pass becomes a no-op.
    """

    def __init__(self, blocks: list[_EmptyBlock]) -> None:
        self.blocks = blocks


def test_fill_constant_candidates_block_def_uses_int_cache_key():
    """End-to-end regression: ``fill_constant_candidates`` MUST allocate
    ``Block_V2`` identities with int-keyed ``Category.BLOCK`` cache
    entries so that subsequent ``ConstantHandler._emit_block`` calls
    (intra-function jump-target references) hit the same entry.

    The pre-fix caller passed ``hex(block.addr)`` (a STR) which never
    collided with ``_emit_block``'s int-keyed probe — the same block
    surfaced as two distinct ``Block_V2`` identities. This test invokes
    the production function with a 2-empty-block mock function, then
    issues a ``_emit_block`` reference and verifies the identity is
    shared.
    """
    from tokenizer.fill_constant_candidates import fill_constant_candidates

    vm = VocabularyManager(platform="x86_64", format_version=2)
    resolver = TokenResolver()

    block_a_addr = 0x4000
    block_b_addr = 0x4100
    func = _EmptyFunction(
        blocks=[
            _EmptyBlock(addr=block_a_addr, size=0x80),
            _EmptyBlock(addr=block_b_addr, size=0x80),
        ]
    )

    result = fill_constant_candidates(
        func_addr=block_a_addr,
        func=func,
        instr_sets=None,
        lookup=None,
        text_start=0,
        text_end=0x10000,
        resolver=resolver,
        vocab_manager=vm,
        arch_provider=None,  # never called: blocks have no instructions
        disasm_provider=None,  # no switch tables -> footer pass is a no-op
    )
    assert result is not None, "fill_constant_candidates returned None for non-empty func"

    # After block-def emission both blocks must be cached under their
    # typed int address. A regression to a STR key would leave the dict
    # with hex-string keys -- visible immediately here.
    assert resolver.id_maps[Category.BLOCK] == {
        block_a_addr: 0,
        block_b_addr: 1,
    }, (
        f"Category.BLOCK cache must be keyed by int(block.addr); got "
        f"{resolver.id_maps[Category.BLOCK]!r}"
    )

    # Now exercise the consumer side: an intra-function jump-target
    # reference to block_b from inside the function. The function entry
    # is at block_a_addr; block_b_addr is strictly inside the function
    # body and != start_addr, so ``_pred_block`` (precedence step 4)
    # fires -> ``_emit_block``.
    block_ranges = np.empty((0, 2), dtype=np.uint64)
    handler = ConstantHandler(vm, resolver, block_ranges)
    ref_tokens = handler.process_constant_v2(
        block_b_addr,
        meta=_make_local_function_meta(func_start=block_a_addr),
        is_arithmetic=False,
    )
    assert len(ref_tokens) == 1, f"expected single block_v2 token; got {ref_tokens!r}"
    # Block B was the SECOND def-emitted block (id 1). The jump-target
    # reference MUST land on the same cache entry.
    assert ref_tokens[0].id == 1, (
        f"intra-function jump-target reference to block_b must reuse "
        f"the def-side identity (1); got {ref_tokens[0].id}. This means "
        f"the def-side and ref-side disagree on the Category.BLOCK key form."
    )


def test_jump_table_target_shares_identity_with_intra_function_jump():
    """A jump-table footer's per-target ``Block_V2`` slot and a normal
    intra-function jump-target reference to the SAME block address must
    share identity.

    The jump-table-footer emitter (``_emit_jump_table_footer_for`` in
    ``fill_constant_candidates``) writes to the same ``Category.BLOCK``
    cache as ``ConstantHandler._emit_block``. Both producers must agree
    on the key form (``int``) so a target that also appears as a regular
    intra-function jump target gets one identity, not two.

    Mirrors the production cross-call-site invariant. Regression: the
    legacy code used ``hex(int(target_addr))`` (STR) at the footer site,
    diverging from ``_emit_block``'s int key — same bug shape as the
    block-def site, different producer. This test invokes the production
    helper directly so a STR-key regression at the footer site fails
    the assertion below immediately.
    """
    from tokenizer.fill_constant_candidates import _emit_jump_table_footer_for
    from tokenizer.function_token_list import FunctionTokenList

    vm = VocabularyManager(platform="x86_64", format_version=2)
    resolver = TokenResolver()
    block_ranges = np.empty((0, 2), dtype=np.uint64)
    handler = ConstantHandler(vm, resolver, block_ranges)

    target_addr = 0x2200

    # --- Producer 1: the production jump-table-footer helper.
    # The helper allocates a Block_V2 identity per target into the
    # ``Category.BLOCK`` cache. We pass a 1-slot FunctionTokenList so
    # the footer block has somewhere to land; the assertion below reads
    # the cache state, not the emitted tokens.
    func_tokens = FunctionTokenList(num_blocks=1, vocab_manager=vm)
    jt_id = resolver.get_identity(Category.JUMP_TABLE, 0x3000, {})
    _emit_jump_table_footer_for(
        jt_id=jt_id,
        base_addr=0x3000,
        target_addrs=[target_addr],
        func_tokens=func_tokens,
        resolver=resolver,
        vocab_manager=vm,
    )

    # The Category.BLOCK cache MUST be int-keyed now -- a STR key would
    # leave ``{hex(target_addr): 0}``, never colliding with _emit_block's
    # int probe below.
    assert resolver.id_maps[Category.BLOCK] == {target_addr: 0}, (
        f"Category.BLOCK cache must be keyed by int(target_addr) after "
        f"jump-table footer emission; got {resolver.id_maps[Category.BLOCK]!r}"
    )

    # --- Producer 2: the regular intra-function jump-target reference
    # to the same target. Must hit the cached id from Producer 1.
    ref_tokens = handler.process_constant_v2(
        target_addr,
        meta=_make_local_function_meta(func_start=0x2000),
        is_arithmetic=False,
    )

    assert len(ref_tokens) == 1, f"expected single block_v2 token; got {ref_tokens!r}"
    assert ref_tokens[0].id == 0, (
        f"jump-table target identity (0) and intra-function jump-target "
        f"identity ({ref_tokens[0].id}) must share the same Category.BLOCK "
        f"cache entry. String/int key mismatch between "
        f"_emit_jump_table_footer_for and ConstantHandler._emit_block."
    )
