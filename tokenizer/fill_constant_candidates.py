import copy
from typing import Optional

import numpy as np

from tokenizer.arch.provider import ArchitectureProvider
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm import DisassemblyProvider, FunctionView, MetadataLookup
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_lists import BlockTokenList
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import BlockToken, Category, TokenResolver, Tokens

VERIFICATION: bool = False


def _emit_jump_table_footer_for(
    jt_id: int,
    base_addr: int,
    target_addrs: list[int],
    func_tokens: FunctionTokenList,
    resolver: TokenResolver,
    vocab_manager: VocabularyManager,
) -> None:
    """Single emission helper used by BOTH the iter_switch_tables path and
    the slot-classification fallback. Produces one synthetic footer block:
    ``[Block_Def, Jump_Table(jt_id), Block_V2(t0), Block_V2(t1), ...]``.

    The caller has already allocated ``jt_id`` via
    ``resolver.get_identity(Category.JUMP_TABLE, base_addr, ...)`` and is
    responsible for dedup (the helper does not check
    ``id_maps[Category.JUMP_TABLE]``).
    """
    jt_token = vocab_manager.Jump_Table(jt_id)
    target_tokens: list[Tokens] = []
    for target_addr in target_addrs:
        target_addr_int = int(target_addr)
        # Mirror the per-block emission's key shape so that targets which
        # are already known blocks of this function share identity via
        # the existing cache entry — ``get_identity`` returns the cached
        # id and ignores ``meta`` on cache hits, so the
        # "comes_from_jump_table" annotation only attaches to brand-new
        # targets (slots that were never emitted as a regular block).
        # The ``Category.BLOCK`` cache is keyed by the typed ``int``
        # address (matching ``ConstantHandler._emit_block`` and the v2
        # block-def emission in ``fill_constant_candidates``); the hex
        # form is kept in ``target_meta["addr"]`` for human-readable
        # metadata only.
        target_meta = {"addr": hex(target_addr_int), "comes_from_jump_table": jt_id}
        target_id = resolver.get_identity(Category.BLOCK, target_addr_int, target_meta)
        target_tokens.append(vocab_manager.Block_V2(target_id))

    footer_tokens: list[Tokens] = [vocab_manager.Block_Def(), jt_token, *target_tokens]
    # One synthetic instruction holding the whole footer: keeps the
    # run-length arithmetic consistent (BlockTokenList expects at least
    # one instruction per block). The insn-str is a debug label only.
    base_addr_hex = hex(int(base_addr))
    footer_block = BlockTokenList(num_insns=1, vocab_manager=vocab_manager)
    footer_block.append_as_insn(
        insn_str=f"jump_table {base_addr_hex}",
        tokens=footer_tokens,
    )
    func_tokens.add_block(footer_block, base_addr_hex)


def _emit_jump_table_footer(
    func: FunctionView,
    func_tokens: FunctionTokenList,
    disasm_provider: Optional[DisassemblyProvider],
    resolver: TokenResolver,
    vocab_manager: VocabularyManager,
) -> None:
    """Append one ``block_def jump_table <id> block_v2 <t0> ...`` block per
    switch table known for ``func`` — UNIONED across two sources:

    1. Provider-recovered tables via ``disasm_provider.iter_switch_tables``
       (canonical path; carries resolved target addresses).
    2. Slot-classification fallback: any ``Category.JUMP_TABLE`` identity
       that was registered during instruction tokenization (via
       ``ConstantHandler._emit_jump_table_slot``) but whose base addr was
       NOT yielded by ``iter_switch_tables`` — the dispatch instruction
       wasn't recovered but a constant pointed into the table, so the
       footer pass still emits a (target-less) declaration so the slot
       reference resolves.

    Single-concern: this function owns the v2 jump-table-footer concern.
    Identity allocation goes through ``TokenResolver.get_identity`` so the
    per-function metadata accumulator records both the table address and
    each target's address; identities for already-known targets are shared
    with the per-block emission earlier in ``fill_constant_candidates``
    (same ``Category.BLOCK`` cache, same typed ``int`` address key —
    matching ``ConstantHandler._emit_block``'s key form).

    Dedup: footer-emit must be deterministic. When BOTH the provider
    yields a table and a slot-classification call has registered the
    same base addr, exactly ONE footer emits — the provider's (carries
    targets). The fallback iterates ``id_maps[Category.JUMP_TABLE]`` and
    skips any addr already emitted via the iter_switch_tables loop.

    No-ops when the vocab is v1 (the v2 ``Jump_Table`` / ``Block_V2``
    Inner classes assert ``format_version == 2``) or when neither source
    has any tables for ``func``.
    """
    if getattr(vocab_manager, "format_version", 1) != 2:
        return

    emitted_base_addrs: set[int] = set()

    if disasm_provider is not None:
        for table_addr, target_addrs in disasm_provider.iter_switch_tables(func):
            base_addr = int(table_addr)
            if not target_addrs:
                # Provider gave us a table with no resolved slots — nothing to
                # emit (would produce a `block_def jump_table` with zero slots,
                # which is structurally a free-floating identity declaration).
                # Skipping here is the canonical-path policy; the slot-fallback
                # below intentionally DOES emit a target-less declaration when
                # the table is known only via slot classification.
                continue

            jt_meta = {
                "jump_table_addr": hex(base_addr),
                "target_block_addrs": [hex(int(a)) for a in target_addrs],
            }
            jt_id = resolver.get_identity(Category.JUMP_TABLE, base_addr, jt_meta)
            _emit_jump_table_footer_for(
                jt_id=jt_id,
                base_addr=base_addr,
                target_addrs=list(target_addrs),
                func_tokens=func_tokens,
                resolver=resolver,
                vocab_manager=vocab_manager,
            )
            emitted_base_addrs.add(base_addr)

    # Slot-classification fallback: pick up any JUMP_TABLE identity that
    # was registered by ``ConstantHandler._emit_jump_table_slot`` but
    # never yielded by the provider's iter_switch_tables. The resolver's
    # per-function reset (called at function boundaries) means the map
    # only contains entries this function recorded.
    for base_addr, jt_id in list(resolver.id_maps[Category.JUMP_TABLE].items()):
        if base_addr in emitted_base_addrs:
            continue
        _emit_jump_table_footer_for(
            jt_id=jt_id,
            base_addr=base_addr,
            target_addrs=[],
            func_tokens=func_tokens,
            resolver=resolver,
            vocab_manager=vocab_manager,
        )
        emitted_base_addrs.add(base_addr)


def fill_constant_candidates(
    func_addr: int,
    func: FunctionView,
    instr_sets: InstructionSets,
    constant_dict: dict[str, list[str]],
    lookup: MetadataLookup,
    text_start: int,
    text_end: int,
    resolver: TokenResolver,
    vocab_manager: VocabularyManager,
    arch_provider: ArchitectureProvider,
    disasm_provider: Optional[DisassemblyProvider] = None,
) -> Optional[
    tuple[
        list[tuple[str, list[list[Tokens]]]],
        list[dict[BlockToken, tuple[str, str]]],
        dict[str, BlockToken],
        ConstantHandler,
        FunctionTokenList,
    ]
]:
    func_min_addr: int = int(func_addr)
    blocks: set = set()

    # `func.blocks` yields the SAME reused `BlockView` instance per step
    # (see ``tokenizer/disasm/types.py`` lifecycle docstring). To hold
    # multiple block references across iteration advances we deepcopy each
    # cursor — `BlockView.__deepcopy__` returns a fresh wrapper bound to
    # the same provider-side block handle; nested instruction iteration
    # stays lazy on each clone.
    block_snapshots: list = [copy.deepcopy(block) for block in func.blocks]
    num_blocks = len(block_snapshots)
    block_ranges: np.ndarray = np.empty((num_blocks, 2), dtype=np.uint64)

    for i, block in enumerate(block_snapshots):
        block_ranges[i, 0] = block.addr
        block_ranges[i, 1] = block.addr + block.size

    func_max_addr = int(block_ranges.max())
    constant_handler = ConstantHandler(vocab_manager, resolver, constant_dict, block_ranges)
    temp_bbs: list[tuple[str, list[list[Tokens]]]] = []
    block_list: list[dict[BlockToken, tuple[int, int]]] = []
    block_dict: dict[str, BlockToken] = {}

    if num_blocks == 1 and len(block_snapshots[0].instructions) == 0:
        return None

    func_tokens = FunctionTokenList(num_blocks, vocab_manager=vocab_manager)
    ordered_blocks = sorted(block_snapshots, key=lambda b: b.addr)
    for block in ordered_blocks:
        func_max_addr = max(block.addr, block.addr + block.size)

        # ``block_addr`` is the human-readable label used by ``block_dict``,
        # ``blocks`` set, ``temp_bbs``, etc. The resolver's BLOCK cache, in
        # contrast, is keyed by the typed ``int`` address (matching the
        # int key used by ``ConstantHandler._emit_block`` for intra-function
        # jump-target references). Hand the int to ``get_block_id`` so block
        # definitions and references to the same block share identity.
        block_addr = hex(block.addr)
        block_id = resolver.get_block_id(int(block.addr))
        block_token = vocab_manager.BlockId(block_id)
        block_list.append(
            {
                block_token: (
                    block.addr,
                    block.addr + block.size,
                )
            }
        )
        blocks.add(block_addr)

        block_dict[block_addr] = block_token

        block_def = [vocab_manager.Block_Def(), block_token]

        disassembly_list = BlockTokenList(len(block.instructions) + 1, vocab_manager=vocab_manager)
        disassembly_list.append_as_insn(insn_str=f"block {block_addr}", tokens=block_def)

        disassembly_list2 = [block_def]

        # `block.instructions` yields the SAME reused `InstructionView`
        # per step; the per-iteration body emits tokens immediately and
        # never stashes the wrapper across advances, satisfying the
        # owned-view reuse contract.
        for insn in block.instructions:
            insn_tokens = disassembly_list.view(insn_str=f"{insn.mnemonic} {insn.op_str}")

            insn_tokens = arch_provider.parse_instruction(
                instr_sets,
                constant_handler,
                func_max_addr,
                func_min_addr,
                insn,
                lookup,
                text_end,
                text_start,
                vocab_manager,
                insn_tokens,
            )
            disassembly_list.add_insn(insn_tokens)
            if VERIFICATION:
                disassembly_list2.append(list(insn_tokens))

        if VERIFICATION:
            for x, y in zip(
                [token for insn in disassembly_list2 for token in insn],
                disassembly_list.iter_raw_tokens(),
            ):
                if x != y:
                    print(f"Token mismatch: {x} != {y}")
                    raise ValueError("Token mismatch in disassembly list")

        if VERIFICATION:
            temp_bbs.append((block_addr, disassembly_list2))
        func_tokens.add_block(disassembly_list, block_addr)

    # v2 jump-table footer: one synthetic block per switch table the
    # provider could recover for ``func``. Guarded inside the helper
    # against v1 vocabs and against providers without switch-table
    # recovery (default returns empty iter — angr is a no-op).
    _emit_jump_table_footer(
        func=func,
        func_tokens=func_tokens,
        disasm_provider=disasm_provider,
        resolver=resolver,
        vocab_manager=vocab_manager,
    )

    return (
        temp_bbs,
        block_list,
        block_dict,
        constant_handler,
        func_tokens,
    )
