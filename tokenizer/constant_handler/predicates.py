"""v2 precedence-list step predicates.

Each predicate is a total function ``(meta, value, ctx) -> bool``; no
side effects. The precedence-list registration in
``tokenizer.constant_handler.core`` pairs predicates with emitter method
names; first match wins.

Per ``precedence.md`` the address steps 2-10 are gated on
``not ctx.is_arithmetic`` (the value is a pure arithmetic operand). Step
1 (FP immediate) and step 11 (valued_const) still fire regardless of
``is_arithmetic`` per the plan's is_arithmetic short-circuit.
"""

from __future__ import annotations

from typing import Optional

from tokenizer.constant_handler.ctx import _Ctx
from tokenizer.disasm.metadata import AddressKind, AddressMetadataView


def _is_function_entry(meta: AddressMetadataView, value: int) -> bool:
    """True iff ``value`` is exactly the entry point of a function range.

    The provider reports a function range as ``[start_addr, end_addr)``
    with ``start_addr == entry``. Any value strictly inside the range is
    a ``block`` (step 4), not the entry.
    """
    return meta.start_addr is not None and value == meta.start_addr


# ---- Step predicates -----------------------------------------------------

# Step 1 fires whenever the caller signals "this value is an FP
# immediate". Per ``precedence.md``: top precedence, fires regardless of
# ``is_arithmetic`` ("an arithmetic FP immediate is a value, but it is a
# floatXX value, not an integer valued_const"). The task pseudocode
# included ``not is_arithmetic`` here -- deviation documented in the
# commit message because precedence.md is canonical.
def _pred_fp_immediate(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return ctx.fp_immediate_type is not None


# Steps 2-10 are gated on ``not is_arithmetic`` per the plan's
# is_arithmetic short-circuit ("steps 2-10 are skipped" for arithmetic
# operands; step 11 catches them). They also short-circuit when ``meta``
# is None (no lookup performed); only step 11 fires in that case.
def _pred_plt_func(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return not ctx.is_arithmetic and meta is not None and meta.kind == AddressKind.PLT_FUNCTION


def _pred_local_func(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 3: address EQUALS function entry in main object.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.LOCAL_FUNCTION
        and _is_function_entry(meta, value)
    )


def _pred_block(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 4: address STRICTLY INSIDE a function in main object.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.LOCAL_FUNCTION
        and not _is_function_entry(meta, value)
    )


def _pred_ext_func_real(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 5: extern function, NOT a synthetic stub.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.EXT_FUNCTION_REAL
    )


def _pred_ext_func_synthetic(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 6: extern function, IS a synthetic stub.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.EXT_FUNCTION_SYNTHETIC
    )


def _pred_string_ptr(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return not ctx.is_arithmetic and meta is not None and meta.kind == AddressKind.STRING


def _pred_jump_table_slot(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 8a: jump-table slot -- distinct emission shape from vtable /
    # code_ptr_table slots (no modifier; emits a Jump_Table identity that
    # bridges to the function-level footer via Category.JUMP_TABLE).
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.JUMP_TABLE_SLOT
    )


def _pred_slot_with_modifier(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    if ctx.is_arithmetic or meta is None:
        return False
    # Vtables surface as CODE_PTR_TABLE_SLOT (precedence above the bare
    # rodata/data kinds). ``is_vtable`` is also exposed as a separate
    # boolean for the modifier-selection step in the emitter.
    # JUMP_TABLE_SLOT is handled by ``_pred_jump_table_slot`` (placed
    # above this predicate in the precedence list) and never reaches here.
    return meta.kind == AddressKind.CODE_PTR_TABLE_SLOT or meta.is_vtable


def _pred_ro_data_ptr(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return not ctx.is_arithmetic and meta is not None and meta.kind == AddressKind.RODATA


def _pred_rw_data_ptr(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind in {AddressKind.DATA, AddressKind.BSS, AddressKind.THREAD_LOCAL_DATA}
    )


def _pred_fallback(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return True
