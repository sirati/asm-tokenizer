"""Regression: x86 ``tokenize_operand_memory`` must not emit the
displacement constant twice on the ``force_opaque`` branch.

Before the fix the displacement-emission block looked like::

    if force_opaque:
        tokens.extend(process_constant_v2(...))  # first emission
    meta = lookup.lookup(classified_value)
    if (in_text or out_of_func):                 # also true when force_opaque
        tokens.extend(process_constant_v2(...))  # second emission

so any ``force_opaque`` displacement whose ``classified_value`` fell
outside ``[func_min_addr, func_max_addr]`` (i.e. nearly every real
string-ptr / rw_data_ptr target) got the same token pair emitted
twice. The user-visible signature was renderings like::

    lea eax qword_ptr mem[ eax + string_ptr:0 string_ptr:0 ]mem

After the fix the second block is ``elif`` on ``force_opaque``, so the
three displacement paths (opaque, in-text-or-out-of-func, local
constant) are mutually exclusive.

The four unit tests below drive ``tokenize_operand_memory`` directly
with light fakes and assert ``process_constant_v2`` is invoked **once**
per memory operand on every force_opaque shape (resolved-target,
absolute-addressed disp without base register, large disp without
resolved target) AND on the non-force-opaque local-constant baseline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

from tokenizer.arch.x86.operands import tokenize_operand_memory


def _make_reg(absent: bool, name: str = "", reg_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(is_absent=absent, name=name, id=reg_id)


def _make_op(
    *,
    base_absent: bool,
    disp: int,
    resolved_target: Optional[int],
    fp_type=None,
    size: int = 8,
) -> SimpleNamespace:
    mem = SimpleNamespace(
        base=_make_reg(base_absent, "eax", 1),
        index=_make_reg(True),
        segment=_make_reg(True),
        scale=1,
        disp=disp,
        resolved_target=resolved_target,
    )
    return SimpleNamespace(mem=mem, size=size, fp_type=fp_type)


def _make_lookup(meta_value=SimpleNamespace(kind="STRING")) -> MagicMock:
    lookup = MagicMock()
    lookup.lookup = MagicMock(return_value=meta_value)
    return lookup


def _make_vocab() -> MagicMock:
    vm = MagicMock()
    # The displacement-emission path emits MEM brackets, PLUS, plus
    # whatever ``constant_handler.process_constant_v2`` returns. We do
    # not assert on the bracket/PLUS tokens here — only on the
    # process_constant_v2 invocation count. Every vocab method returns
    # a distinct sentinel so the assembled token list survives
    # ``tokens.append`` / ``tokens.extend`` without exercising any vocab
    # internals.
    return vm


def _make_constant_handler() -> MagicMock:
    ch = MagicMock()
    # Each invocation returns a fresh 1-element token list so we can
    # also verify the resulting token stream's token-from-constant-handler
    # count if needed (each emission is one entry).
    ch.process_constant_v2 = MagicMock(side_effect=lambda *a, **kw: [f"ct:{a}:{kw}"])
    return ch


def _make_insn() -> SimpleNamespace:
    return SimpleNamespace(mnemonic="lea", op_str="eax, [eax + 0x1234]")


# ---- 1. Repro / regression: force_opaque via has_resolved (PIC lea) ----


def test_force_opaque_resolved_target_emits_constant_once() -> None:
    """``lea eax, [eax + str_offset]`` after Ghidra has resolved the
    target to ``0x40_2000`` (a ``.rodata`` string outside the function
    body). Pre-fix: two ``process_constant_v2`` calls (force_opaque
    branch + unconditional out-of-func fallthrough). Post-fix: one.
    """
    ch = _make_constant_handler()
    tokenize_operand_memory(
        insn=_make_insn(),
        lookup=_make_lookup(),
        op=_make_op(base_absent=False, disp=0x100, resolved_target=0x40_2000),
        text_end=0x40_0000,
        text_start=0x10_0000,
        # Function body sits well below the rodata target.
        func_max_addr=0x20_0000,
        func_min_addr=0x10_0000,
        vocab_manager=_make_vocab(),
        constant_handler=ch,
    )
    assert ch.process_constant_v2.call_count == 1, (
        f"expected single process_constant_v2 emission for resolved "
        f"string target; got {ch.process_constant_v2.call_count} calls. "
        f"This is the double-emit signature that produced "
        f"``lea ... [ ... string_ptr:0 string_ptr:0 ]`` renderings."
    )


# ---- 2. force_opaque via has_disp && !has_reg (abs-addressed mem) ------


def test_force_opaque_absolute_addressed_emits_constant_once() -> None:
    """``mov eax, dword_ptr [0x401234]`` — no base register, disp is
    the absolute target. ``force_opaque`` fires via the
    ``has_disp and not has_reg`` branch. The post-fix mutual-exclusion
    keeps the emission count at one.
    """
    ch = _make_constant_handler()
    tokenize_operand_memory(
        insn=_make_insn(),
        lookup=_make_lookup(),
        op=_make_op(base_absent=True, disp=0x40_1234, resolved_target=None),
        text_end=0x40_0000,
        text_start=0x10_0000,
        func_max_addr=0x20_0000,
        func_min_addr=0x10_0000,
        vocab_manager=_make_vocab(),
        constant_handler=ch,
    )
    assert ch.process_constant_v2.call_count == 1


# ---- 3. force_opaque via large-disp without resolved_target -----------


def test_force_opaque_large_disp_emits_constant_once() -> None:
    """A ``classified_value > 1 << 18`` (no resolved_target, has_reg).
    The ``elif classified_value > (1 << 18)`` branch sets force_opaque.
    The follow-on out-of-func test would have also fired pre-fix
    (large disps are typically outside the func body), producing the
    double-emit. Post-fix: one emission only.
    """
    ch = _make_constant_handler()
    tokenize_operand_memory(
        insn=_make_insn(),
        lookup=_make_lookup(),
        # 0x50_0000 > 1<<18 (=0x40000); also > func_max_addr below, so
        # pre-fix BOTH the force_opaque branch and the out-of-func
        # fallthrough fired (call_count would be 2).
        op=_make_op(base_absent=False, disp=0x50_0000, resolved_target=None),
        text_end=0x40_0000,
        text_start=0x10_0000,
        func_max_addr=0x20_0000,
        func_min_addr=0x10_0000,
        vocab_manager=_make_vocab(),
        constant_handler=ch,
    )
    assert ch.process_constant_v2.call_count == 1


# ---- 4. Non-force-opaque local-constant baseline -----------------------


def test_local_constant_disp_emits_constant_once() -> None:
    """Non-force-opaque baseline: a disp that is in-function (so the
    out-of-func fallthrough does NOT fire), has base register (so
    ``has_disp and not has_reg`` does NOT fire), is ``<= 1 << 18`` (so
    the large-disp branch does NOT fire), and ``> 0xFF`` (so the
    ``<= 0xFF`` early branch does NOT fire). Lands on the final
    ``else`` arm (line 275, the local-constant arithmetic emitter).

    The bug never affected this path — it pre-fix already had one
    emission. This test guards against accidental fallthrough into a
    second emission if the if/elif structure ever regresses.
    """
    ch = _make_constant_handler()
    tokenize_operand_memory(
        insn=_make_insn(),
        lookup=_make_lookup(),
        # disp 0x150: > 0xFF, < 1<<18, inside [func_min, func_max].
        op=_make_op(base_absent=False, disp=0x150, resolved_target=None),
        text_end=0x40_0000,
        text_start=0x10_0000,
        func_max_addr=0x20_0000,
        func_min_addr=0x100,
        vocab_manager=_make_vocab(),
        constant_handler=ch,
    )
    assert ch.process_constant_v2.call_count == 1
    # Local-constant path uses the arithmetic short-circuit (no meta).
    call = ch.process_constant_v2.call_args
    assert call.kwargs.get("is_arithmetic") is True, (
        f"local-constant disp must use is_arithmetic=True; got kwargs={call.kwargs}"
    )


# ---- 5. Resolved + tiny value (precedence step 7 still wins) -----------


def test_resolved_tiny_value_takes_force_opaque_branch() -> None:
    """A resolved target whose numeric value is <= 0xFF (e.g. an early
    ``.rodata`` slot at address 0x42). The ``classified_value <= 0xFF
    and not has_resolved`` guard at line 217 is keyed on
    ``not has_resolved``, so this case falls THROUGH to the
    force_opaque branch and stays metadata-aware (string_ptr instead
    of bare valued_const). Verifies the fix did not regress the
    precedence-step-7 path.
    """
    ch = _make_constant_handler()
    tokenize_operand_memory(
        insn=_make_insn(),
        lookup=_make_lookup(),
        op=_make_op(base_absent=False, disp=0, resolved_target=0x42),
        text_end=0x40_0000,
        text_start=0x10_0000,
        func_max_addr=0x10_0000,  # 0x42 < func_min_addr, so out-of-func
        func_min_addr=0x100,
        vocab_manager=_make_vocab(),
        constant_handler=ch,
    )
    assert ch.process_constant_v2.call_count == 1
    # The single call must be the metadata-aware shape (kwargs include
    # ``meta=`` and ``is_arithmetic=False``), not the bare-arithmetic
    # short-circuit — that distinction is precedence-step-7's contract.
    call = ch.process_constant_v2.call_args
    assert call.kwargs.get("is_arithmetic") is False, (
        f"resolved-target tiny-value must emit via the metadata-aware "
        f"path (precedence step 7 / string_ptr), not the arithmetic "
        f"short-circuit; got kwargs={call.kwargs}"
    )
    assert "meta" in call.kwargs, (
        f"resolved-target emission must pass meta=; got kwargs={call.kwargs}"
    )
