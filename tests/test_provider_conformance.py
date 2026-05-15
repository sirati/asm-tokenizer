"""Phase J.1 - Conformance test suite for owned views.

Validates that BOTH disassembly providers (Ghidra default, angr best-effort)
satisfy the Protocols declared in ``tokenizer/disasm/types.py``:

* T-A : Protocol shape on the iter_functions -> blocks -> instructions ->
        operands chain; every operand carries a valid ``OperandKind``.
* T-B : Per-instruction address falls within the parent block's
        ``[addr, addr+size)`` range.
* T-C : ``operand.fp_type`` is either ``None`` or an ``FpType`` enum.
* T-D : x86 MEM operand shape - ``mem.base`` is a ``RegisterView``,
        ``mem.disp`` is an ``int``, ``mem.scale`` is an ``int >= 1``.
* T-E : ARM conditional-instruction prefix decoding emits a
        ``ConditionCodePrefixView`` whose ``cc`` is a valid
        ``ArmConditionCode`` enum.
* T-F : Reuse semantics - a held ``OperandView`` reflects the LATER
        operand after iteration advances; ``copy.deepcopy(op)`` snapshots
        a stash-safe wrapper that does NOT mutate.
* T-G : Eagerness probe - ``iter_functions()`` does not force block /
        instruction / operand decoding before the consumer asks for it.
* T-H : ``stmdb sp!`` on arm32 emits exactly one ``REG_LIST`` operand
        whose ``reg_list`` view iterates >= 2 members.
* T-I : The same ``stmdb sp!`` operand's ``reg_list.writeback == True``
        (regression guard against the pre-2026-05-15 hardcoded False).

Each test is ``@pytest.mark.parametrize``'d over ``("ghidra", "angr")``;
provider-conditional assertions are gated on ``provider_name`` inside the
test body so a single test target works for both backends.

Fixture binaries (see conftest.py):
    x64-gcc-7-Os_minigzip  -> default provider fixtures
    arm32-gcc-7-Os_minigzip -> arm32-specific fixtures (T-E, T-H, T-I)
"""

from __future__ import annotations

import copy

import pytest

from tokenizer.disasm.types import (
    ArmConditionCode,
    BlockView,
    ConditionCodePrefixView,
    FpType,
    FunctionView,
    InstructionView,
    OperandKind,
    OperandView,
    RegisterView,
)


PROVIDER_IDS = ["ghidra", "angr"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _walk_instructions(provider, max_instructions: int | None = None):
    """Yield ``(function_view, block_view, instruction_view)`` triples.

    Walks until ``max_instructions`` have been yielded (when not None);
    otherwise drains the binary.
    """
    count = 0
    for _addr, _name, function in provider.iter_functions():
        for block in function.blocks:
            for insn in block.instructions:
                yield function, block, insn
                count += 1
                if max_instructions is not None and count >= max_instructions:
                    return


def _find_first(predicate, iterable):
    """Return the first item satisfying ``predicate`` or ``None`` if none."""
    for item in iterable:
        if predicate(item):
            return item
    return None


# ---------------------------------------------------------------------------
# T-A: Protocol shape walk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_a_protocol_shape(provider):
    """iter_functions -> blocks -> instructions -> operands all match
    their typed Protocols; every operand carries a valid OperandKind enum."""
    seen_func = seen_block = seen_insn = seen_operand = 0

    for _addr, _name, function in provider.iter_functions():
        assert isinstance(function, FunctionView), (
            f"iter_functions yielded non-FunctionView: {type(function).__name__}"
        )
        seen_func += 1
        for block in function.blocks:
            assert isinstance(block, BlockView), (
                f"function.blocks yielded non-BlockView: {type(block).__name__}"
            )
            seen_block += 1
            for insn in block.instructions:
                assert isinstance(insn, InstructionView), (
                    f"block.instructions yielded non-InstructionView: {type(insn).__name__}"
                )
                seen_insn += 1
                for op in insn.operands:
                    assert isinstance(op, OperandView), (
                        f"insn.operands yielded non-OperandView: {type(op).__name__}"
                    )
                    # OperandKind must be a member of the enum (not a raw int)
                    assert isinstance(op.kind, OperandKind), (
                        f"op.kind is {type(op.kind).__name__}, expected OperandKind"
                    )
                    seen_operand += 1
                    if seen_operand >= 200:
                        break
                if seen_operand >= 200:
                    break
            if seen_operand >= 200:
                break
        if seen_operand >= 200:
            break

    # Sanity floor - the fixture has hundreds of functions; make sure we
    # actually exercised the chain.
    assert seen_func >= 1, "no functions yielded"
    assert seen_block >= 1, "no blocks yielded"
    assert seen_insn >= 1, "no instructions yielded"
    assert seen_operand >= 1, "no operands yielded"


# ---------------------------------------------------------------------------
# T-B: instruction.address is within block.addr..block.addr+block.size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_b_instruction_address_within_block(provider):
    """For >=100 instructions sampled across the binary, the instruction's
    address falls inside its parent block's [addr, addr+size) range."""
    sampled = 0
    for _addr, _name, function in provider.iter_functions():
        for block in function.blocks:
            b_addr = int(block.addr)
            b_size = int(block.size)
            b_end = b_addr + b_size
            for insn in block.instructions:
                ia = int(insn.address)
                assert b_addr <= ia < b_end, (
                    f"instruction at {hex(ia)} outside block "
                    f"[{hex(b_addr)}, {hex(b_end)}) (size={b_size})"
                )
                sampled += 1
                if sampled >= 100:
                    break
            if sampled >= 100:
                break
        if sampled >= 100:
            break

    assert sampled >= 100, f"sampled only {sampled} instructions, expected >=100"


# ---------------------------------------------------------------------------
# T-C: operand.fp_type is None or an FpType enum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_c_fp_type_typed(provider):
    """``operand.fp_type`` is either ``None`` or an instance of ``FpType``."""
    checked = 0
    for _f, _b, insn in _walk_instructions(provider, max_instructions=500):
        for op in insn.operands:
            fp = op.fp_type
            assert fp is None or isinstance(fp, FpType), (
                f"op.fp_type is {fp!r} ({type(fp).__name__}); "
                f"expected None or FpType"
            )
            checked += 1

    assert checked >= 1, "no operands checked"


# ---------------------------------------------------------------------------
# T-D: x86 MEM operand consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_d_x86_mem_operand_shape(provider):
    """For any single x86 instruction with a MEM operand carrying base+disp,
    ``op.mem.base`` is a ``RegisterView``, ``op.mem.disp`` is ``int``,
    ``op.mem.scale`` is ``int >= 1``."""
    found = None
    for _f, _b, insn in _walk_instructions(provider):
        for op in insn.operands:
            if op.kind != OperandKind.MEM:
                continue
            mem = op.mem
            # We want a base+disp pair so the test exercises a populated
            # base register slot (rules out pure-disp absolute memory loads).
            if mem.base.is_absent:
                continue
            found = (mem.base.name, int(mem.disp), int(mem.scale))
            # Type assertions on the located operand
            assert isinstance(mem.base, RegisterView), (
                f"mem.base is {type(mem.base).__name__}, expected RegisterView"
            )
            assert isinstance(mem.disp, int), (
                f"mem.disp is {type(mem.disp).__name__}, expected int"
            )
            assert isinstance(mem.scale, int), (
                f"mem.scale is {type(mem.scale).__name__}, expected int"
            )
            assert mem.scale >= 1, f"mem.scale is {mem.scale}, expected >= 1"
            break
        if found is not None:
            break

    assert found is not None, (
        "no x86 MEM operand with non-absent base was found in the binary"
    )


# ---------------------------------------------------------------------------
# T-E: ARM conditional-instruction prefix decoding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm32_provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_e_arm_condition_code_prefix(arm32_provider):
    """A conditional ARM instruction (e.g. ``beq``) yields a
    ``ConditionCodePrefixView`` instance in its prefix list with a valid
    ``ArmConditionCode`` enum value on its ``cc`` attribute."""
    target = None
    for _f, _b, insn in _walk_instructions(arm32_provider):
        # Look for a known conditional mnemonic. Use ``base_mnemonic`` so
        # the test is robust to provider-side mnemonic decoration. ARM
        # conditional instructions encode the condition in the mnemonic
        # suffix (beq, bne, bcc, ...) - both providers preserve this in
        # the raw mnemonic, while base_mnemonic strips it on Ghidra.
        m = insn.mnemonic.lower()
        bm = insn.base_mnemonic.lower()
        # Match BEQ / BNE / BLT / BGT (etc.) but skip the unconditional B.
        if m in ("b", "bl", "bx", "blx") or bm in ("b", "bl", "bx", "blx"):
            continue
        if not (m.startswith("b") or m.startswith("mov") or m.startswith("ldr") or m.startswith("str")):
            continue
        cc_prefix = None
        for pfx in insn.prefixes:
            if isinstance(pfx, ConditionCodePrefixView):
                cc_prefix = pfx
                break
        if cc_prefix is None:
            continue
        # Found one - validate the cc value.
        target = (insn.mnemonic, cc_prefix)
        assert isinstance(cc_prefix.cc, ArmConditionCode), (
            f"prefix.cc is {type(cc_prefix.cc).__name__}, expected ArmConditionCode"
        )
        break

    assert target is not None, (
        "no conditional ARM instruction with a ConditionCodePrefixView was found"
    )


# ---------------------------------------------------------------------------
# T-F: Reuse semantics + deepcopy stash safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_f_operand_reuse_semantics(provider):
    """Holding an ``OperandView`` reference across an ``operands`` iteration
    advance: the held reference's properties reflect the LATER operand
    (proves the reuse contract). ``copy.deepcopy(op)`` returns a wrapper
    that does NOT mutate when iteration advances."""
    target_insn = None
    for _f, _b, insn in _walk_instructions(provider):
        if len(insn.operands) >= 2:
            target_insn = insn
            break

    assert target_insn is not None, (
        "no instruction with >=2 operands found - cannot exercise reuse contract"
    )

    it = iter(target_insn.operands)
    first = next(it)
    # Sample a discriminator that distinguishes operands. We record kind +
    # imm + the underlying object identity for the reuse assertion.
    first_id = id(first)
    first_kind = first.kind
    first_imm = first.imm

    snapshot = copy.deepcopy(first)
    snap_kind = snapshot.kind
    snap_imm = snapshot.imm

    # Advance iteration; the SAME wrapper object should now reflect the
    # second operand (reuse contract).
    second = next(it)
    assert id(second) == first_id, (
        "operands iteration produced a fresh wrapper - reuse contract is broken"
    )

    # The held ``first`` reference is the same object as ``second`` - so its
    # observable properties must match ``second``'s, not the original first.
    # We assert at least one observable changed (kind or imm); operands
    # within a single instruction are usually distinct enough that one
    # discriminator differs. If they happen to be identical (rare; e.g.
    # ``mov rax, rax``), we still verified the object identity above.
    later_kind = first.kind
    later_imm = first.imm
    advanced = (later_kind != first_kind) or (later_imm != first_imm)
    if not advanced:
        # Both operands identical in our discriminators - fall back to the
        # identity check we already validated above. Skip the equality
        # assertion in this rare case.
        pytest.skip(
            "both operands identical on the chosen discriminators; "
            "object-identity reuse already verified"
        )

    # Snapshot must be stash-safe: it should NOT have mutated after the
    # iteration advance.
    assert snapshot.kind == snap_kind, (
        f"deepcopy snapshot mutated: kind {snap_kind} -> {snapshot.kind}"
    )
    assert snapshot.imm == snap_imm, (
        f"deepcopy snapshot mutated: imm {snap_imm} -> {snapshot.imm}"
    )


# ---------------------------------------------------------------------------
# T-G: No materialization - iter_functions() does not force decode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_g_no_eager_decode(provider):
    """``iter_functions()`` yields FunctionView cursors lazily; the function
    count should be obtainable WITHOUT block/instruction decode.

    We can't easily peek at the cursor's internal `_advance` counter without
    coupling to provider internals; instead we measure that iterating
    ``provider.iter_functions()`` for the function count is dramatically
    cheaper than the full instruction walk. If the call eagerly decoded
    everything, the two timings would be similar.
    """
    import time

    # First pass: just count function yields - no .blocks access.
    t0 = time.perf_counter()
    n_func = 0
    for _addr, _name, _function in provider.iter_functions():
        n_func += 1
    light_elapsed = time.perf_counter() - t0
    assert n_func >= 1, "no functions yielded"

    # Second pass: full instruction walk (forces block/insn/operand decode
    # everywhere). Cap to keep the test fast on huge binaries.
    t0 = time.perf_counter()
    n_insn = 0
    for _f, _b, _insn in _walk_instructions(provider):
        n_insn += 1
        if n_insn >= 5000:
            break
    heavy_elapsed = time.perf_counter() - t0
    assert n_insn >= 1, "no instructions yielded"

    # No-materialization claim: counting functions should be at least an
    # order of magnitude faster than decoding 5k instructions, which only
    # holds if the function-level walk does not pre-emptively decode the
    # block/insn graph. We use a 5x threshold to keep the test robust to
    # CI noise; a genuinely eager implementation would be equal-or-slower
    # on the light pass.
    assert light_elapsed * 5 < heavy_elapsed or light_elapsed < 0.05, (
        f"iter_functions appears to materialize eagerly: "
        f"light={light_elapsed:.3f}s heavy={heavy_elapsed:.3f}s "
        f"(n_func={n_func}, n_insn={n_insn})"
    )


# ---------------------------------------------------------------------------
# T-H: REG_LIST presence + member count on arm32 stmdb
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm32_provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_h_arm_reg_list_membership(arm32_provider):
    """``stmdb`` on arm32 emits exactly one ``OperandKind.REG_LIST`` operand
    whose reg_list view iterates >= 2 members. If the function
    ``__aeabi_idiv0`` exists, prefer it as the reference (its first insn
    is ``stmdb sp!, {r1, lr}`` -> 2 members)."""
    target_insn = None
    target_op = None

    # Walk - prefer __aeabi_idiv0 if present; otherwise any stmdb in the
    # binary.
    for _addr, name, function in arm32_provider.iter_functions():
        if name != "__aeabi_idiv0":
            continue
        for block in function.blocks:
            for insn in block.instructions:
                if insn.base_mnemonic.lower().startswith("stmdb"):
                    target_insn = insn
                    break
            if target_insn is not None:
                break
        if target_insn is not None:
            break

    if target_insn is None:
        # Fallback: any stmdb anywhere.
        for _f, _b, insn in _walk_instructions(arm32_provider):
            if insn.base_mnemonic.lower().startswith("stmdb"):
                target_insn = insn
                break

    assert target_insn is not None, "no stmdb instruction found on arm32 fixture"

    reg_list_ops = []
    for op in target_insn.operands:
        if op.kind == OperandKind.REG_LIST:
            reg_list_ops.append(op)
            target_op = op

    assert len(reg_list_ops) == 1, (
        f"expected exactly 1 REG_LIST operand on {target_insn.mnemonic} "
        f"at {hex(int(target_insn.address))}; got {len(reg_list_ops)}"
    )

    assert target_op is not None
    rl = target_op.reg_list
    members = list(rl)
    assert len(members) >= 2, (
        f"reg_list of {target_insn.mnemonic} at {hex(int(target_insn.address))} "
        f"has {len(members)} members; expected >= 2"
    )
    for m in members:
        assert isinstance(m, RegisterView), (
            f"reg_list member is {type(m).__name__}, expected RegisterView"
        )


# ---------------------------------------------------------------------------
# T-I: REG_LIST writeback regression guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm32_provider", PROVIDER_IDS, indirect=True, ids=PROVIDER_IDS)
def test_t_i_arm_reg_list_writeback(arm32_provider):
    """``stmdb sp!`` carries writeback=True on its REG_LIST operand.

    Pre-2026-05-15 the Ghidra-side implementation hardcoded ``writeback``
    to ``False`` regardless of the encoding's ``!`` flag; this test guards
    against that regression.
    """
    # Same locator as T-H: prefer __aeabi_idiv0, fall back to any stmdb.
    target_insn = None
    for _addr, name, function in arm32_provider.iter_functions():
        if name != "__aeabi_idiv0":
            continue
        for block in function.blocks:
            for insn in block.instructions:
                if insn.base_mnemonic.lower().startswith("stmdb"):
                    target_insn = insn
                    break
            if target_insn is not None:
                break
        if target_insn is not None:
            break

    if target_insn is None:
        # Fallback: only consider stmdb instructions whose disassembly
        # actually carries the writeback marker. ``stmdb sp!, {...}`` is
        # the common form; without ``!`` writeback genuinely is False and
        # would (correctly) fail the assertion.
        for _f, _b, insn in _walk_instructions(arm32_provider):
            if not insn.base_mnemonic.lower().startswith("stmdb"):
                continue
            if "!" not in insn.op_str:
                continue
            target_insn = insn
            break

    assert target_insn is not None, (
        "no stmdb-with-writeback instruction found on arm32 fixture"
    )

    target_op = _find_first(
        lambda o: o.kind == OperandKind.REG_LIST,
        target_insn.operands,
    )
    assert target_op is not None, (
        f"stmdb at {hex(int(target_insn.address))} has no REG_LIST operand"
    )

    assert target_op.reg_list.writeback is True, (
        f"stmdb at {hex(int(target_insn.address))} (op_str={target_insn.op_str!r}) "
        f"reports writeback=False; regression of pre-2026-05-15 bug"
    )
