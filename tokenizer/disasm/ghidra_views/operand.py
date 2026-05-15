"""Operand cursor + sub-views.

Owns:
- ``_GhidraMemoryOperandView``: memory operand sub-view (base/index/scale/disp/segment).
- ``_GhidraShiftModifierView``: ARM shift kind + amount sub-view.
- ``_GhidraCrxFieldView``: PPC condition-register field sub-view.
- ``_GhidraOperandView``: reusable operand cursor.

The ``_GhidraOperandsView`` container view lives next to its parent
instruction view in ``instruction.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from tokenizer.disasm.ghidra_views.reg_list import _GhidraRegisterListView
from tokenizer.disasm.ghidra_views.register import (
    _REG_ID_ABSENT,
    _REG_NAME_ABSENT,
    _GhidraRegisterView,
)
from tokenizer.disasm.types import (
    Architecture,
    CrxFieldView,
    FpType,
    MemoryOperandView,
    OperandKind,
    RegisterListView,
    RegisterView,
    ShiftKind,
    ShiftModifierView,
)


# ---------------------------------------------------------------------------
# Memory operand sub-view
# ---------------------------------------------------------------------------
class _GhidraMemoryOperandView:
    """Sub-view bound to the parent operand's cursor.

    Holds the decomposed (base, index, scale, disp, segment) for the
    current memory operand. The parent ``_GhidraOperandView`` populates
    this once on first access (via the provider's per-ISA decomposition
    helpers, ``_compute_x86_memory_components`` /
    ``_compute_arm_memory_components`` / ``_compute_base_disp_memory_components``)
    and re-populates whenever the parent advances.

    The four ``RegisterView`` sub-slots (``base`` / ``index`` / ``segment``)
    are reusable child ``_GhidraRegisterView`` wrappers owned by this
    memory view. ``__iter__`` of OperandsView never reaches this layer
    so a bare slot read returns the same wrapper across all reads of the
    same operand cursor.
    """

    __slots__ = (
        "_arch",
        "_base",
        "_index",
        "_segment",
        "_scale",
        "_disp",
        "_writeback",
        "_pre_indexed",
        "_post_indexed",
        "_index_shift",
        "_resolved_target",
    )

    def __init__(self, arch: Architecture) -> None:
        self._arch: Architecture = arch
        self._base = _GhidraRegisterView(arch)
        self._index = _GhidraRegisterView(arch)
        self._segment = _GhidraRegisterView(arch)
        self._scale: int = 1
        self._disp: int = 0
        self._writeback: bool = False
        self._pre_indexed: bool = False
        self._post_indexed: bool = False
        # Index-shift sub-view is REUSED across operands (no per-operand
        # allocation); ``_populate`` repoints the underlying kind+amount
        # in-place each call so consumers re-reading ``mem.index_shift``
        # after the parent operand advances see the new cursor's state.
        self._index_shift = _GhidraShiftModifierView()
        self._resolved_target: Optional[int] = None

    def _populate(
        self,
        *,
        base_name: str,
        base_id: int,
        index_name: str,
        index_id: int,
        segment_name: str,
        segment_id: int,
        scale: int,
        disp: int,
        writeback: bool = False,
        pre_indexed: bool = False,
        post_indexed: bool = False,
        index_shift_kind: ShiftKind = ShiftKind.NONE,
        index_shift_amount: int = 0,
        resolved_target: Optional[int] = None,
    ) -> None:
        if base_id != _REG_ID_ABSENT or base_name != _REG_NAME_ABSENT:
            self._base._advance(base_name, base_id)
        else:
            self._base._set_absent()
        if index_id != _REG_ID_ABSENT or index_name != _REG_NAME_ABSENT:
            self._index._advance(index_name, index_id)
        else:
            self._index._set_absent()
        if segment_id != _REG_ID_ABSENT or segment_name != _REG_NAME_ABSENT:
            self._segment._advance(segment_name, segment_id)
        else:
            self._segment._set_absent()
        self._scale = scale
        self._disp = disp
        self._writeback = writeback
        self._pre_indexed = pre_indexed
        self._post_indexed = post_indexed
        self._index_shift._populate(index_shift_kind, index_shift_amount)
        self._resolved_target = resolved_target

    @property
    def base(self) -> RegisterView:
        return self._base

    @property
    def index(self) -> RegisterView:
        return self._index

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def disp(self) -> int:
        return self._disp

    @property
    def segment(self) -> RegisterView:
        return self._segment

    @property
    def writeback(self) -> bool:
        return self._writeback

    @property
    def pre_indexed(self) -> bool:
        return self._pre_indexed

    @property
    def post_indexed(self) -> bool:
        return self._post_indexed

    @property
    def index_shift(self) -> ShiftModifierView:
        return self._index_shift

    @property
    def resolved_target(self) -> Optional[int]:
        return self._resolved_target

    def __deepcopy__(self, memo) -> "_GhidraMemoryOperandView":
        clone = _GhidraMemoryOperandView(self._arch)
        clone._base = copy.deepcopy(self._base, memo)
        clone._index = copy.deepcopy(self._index, memo)
        clone._segment = copy.deepcopy(self._segment, memo)
        clone._scale = self._scale
        clone._disp = self._disp
        clone._writeback = self._writeback
        clone._pre_indexed = self._pre_indexed
        clone._post_indexed = self._post_indexed
        clone._index_shift = copy.deepcopy(self._index_shift, memo)
        clone._resolved_target = self._resolved_target
        return clone


# ---------------------------------------------------------------------------
# Shift modifier sub-view (ARM only)
# ---------------------------------------------------------------------------
class _GhidraShiftModifierView:
    """Sub-view bound to the parent operand's cursor.

    Holds the ARM shift kind + amount. ARM shifts come from SLEIGH-decoded
    operand text in Ghidra; the v1 _CapShift fields are mirrored here.
    Non-ARM operands keep the default ``ShiftKind.NONE`` + amount=0.
    """

    __slots__ = ("_kind", "_amount")

    def __init__(self) -> None:
        self._kind: ShiftKind = ShiftKind.NONE
        self._amount: int = 0

    def _populate(self, kind: ShiftKind, amount: int) -> None:
        self._kind = kind
        self._amount = amount

    @property
    def kind(self) -> ShiftKind:
        return self._kind

    @property
    def amount(self) -> int:
        return self._amount

    def __deepcopy__(self, memo) -> "_GhidraShiftModifierView":
        clone = _GhidraShiftModifierView()
        clone._kind = self._kind
        clone._amount = self._amount
        return clone


# ---------------------------------------------------------------------------
# CRX field sub-view (PPC only)
# ---------------------------------------------------------------------------
class _GhidraCrxFieldView:
    """Sub-view bound to the parent operand's cursor.

    Holds the PPC condition-register field reference. The reused
    ``_GhidraRegisterView`` exposes the cr0/cr1/... register name+id;
    non-PPC operands leave it absent.
    """

    __slots__ = ("_arch", "_reg")

    def __init__(self, arch: Architecture) -> None:
        self._arch = arch
        self._reg = _GhidraRegisterView(arch)

    def _populate(self, reg_name: str, reg_id: int) -> None:
        if reg_id != _REG_ID_ABSENT or reg_name != _REG_NAME_ABSENT:
            self._reg._advance(reg_name, reg_id)
        else:
            self._reg._set_absent()

    @property
    def reg(self) -> RegisterView:
        return self._reg

    def __deepcopy__(self, memo) -> "_GhidraCrxFieldView":
        clone = _GhidraCrxFieldView(self._arch)
        clone._reg = copy.deepcopy(self._reg, memo)
        return clone


# ---------------------------------------------------------------------------
# Operand
# ---------------------------------------------------------------------------
class _GhidraOperandView:
    """Reusable operand wrapper.

    The wrapper holds a "source spec" populated by the parent
    ``_GhidraInstructionView`` per-operand (kind, immediate value,
    register slot, FP type, size, type_int). MEM operands additionally
    carry a "decompose" callback the wrapper invokes lazily on first
    ``mem`` access; the callback uses the provider's per-ISA helpers to
    populate the bound ``_GhidraMemoryOperandView`` slots.

    Non-MEM operands leave ``mem`` at default-empty (all-absent regs +
    scale=1 + disp=0); non-ARM operands leave ``shift`` at default
    (ShiftKind.NONE); non-PPC operands leave ``crx`` at default
    (absent reg).

    Per-cursor cache: the lazy mem-decomposition is computed once per
    operand cursor position. ``_advance`` resets ``_mem_decomposed``
    so the next ``mem`` read recomputes against the new operand.
    """

    __slots__ = (
        "_arch",
        "_kind",
        "_reg",
        "_imm",
        "_mem",
        "_shift",
        "_crx",
        "_reg_list",
        "_size",
        "_fp_type",
        "_type_int",
        "_decompose_mem",
        "_mem_decomposed",
        "_decompose_reg_list",
        "_reg_list_decomposed",
    )

    def __init__(self, arch: Architecture, reg_map: Any = None) -> None:
        self._arch: Architecture = arch
        self._kind: OperandKind = OperandKind.INVALID
        self._reg = _GhidraRegisterView(arch)
        self._imm: int = 0
        self._mem = _GhidraMemoryOperandView(arch)
        self._shift = _GhidraShiftModifierView()
        self._crx = _GhidraCrxFieldView(arch)
        self._reg_list = _GhidraRegisterListView(arch, reg_map)
        self._size: int = 0
        self._fp_type: Optional[FpType] = None
        self._type_int: int = 0
        self._decompose_mem: Optional[Any] = None
        self._mem_decomposed: bool = False
        self._decompose_reg_list: Optional[Any] = None
        self._reg_list_decomposed: bool = False

    def _advance(
        self,
        *,
        kind: OperandKind,
        reg_name: str,
        reg_id: int,
        imm: int,
        size: int,
        fp_type: Optional[FpType],
        type_int: int,
        decompose_mem: Optional[Any],
        shift_kind: ShiftKind,
        shift_amount: int,
        crx_reg_name: str,
        crx_reg_id: int,
        decompose_reg_list: Optional[Any] = None,
    ) -> None:
        """Repoint the operand wrapper at the next operand.

        Resets sub-view state (mem-decomposition cache, shift, crx,
        reg-list decomposition cache) so bound sub-views reflect the new
        operand on next access.

        ``decompose_mem`` is a zero-argument callable that mutates
        ``self._mem`` via its ``_populate`` method. Provided ONLY when
        ``kind == OperandKind.MEM``; ``None`` otherwise (no MEM access
        is expected on non-MEM kinds). Mirror semantics apply to
        ``decompose_reg_list`` for ``kind == OperandKind.REG_LIST``.
        """
        self._kind = kind
        if reg_id != _REG_ID_ABSENT or reg_name != _REG_NAME_ABSENT:
            self._reg._advance(reg_name, reg_id)
        else:
            self._reg._set_absent()
        self._imm = imm
        self._size = size
        self._fp_type = fp_type
        self._type_int = type_int
        self._decompose_mem = decompose_mem
        self._mem_decomposed = False
        self._decompose_reg_list = decompose_reg_list
        self._reg_list_decomposed = False
        self._shift._populate(shift_kind, shift_amount)
        self._crx._populate(crx_reg_name, crx_reg_id)

    @property
    def kind(self) -> OperandKind:
        return self._kind

    @property
    def reg(self) -> RegisterView:
        return self._reg

    @property
    def imm(self) -> int:
        return self._imm

    @property
    def mem(self) -> MemoryOperandView:
        if not self._mem_decomposed and self._decompose_mem is not None:
            # Decompose lazily on first access; per-cursor cache so
            # repeated reads of ``mem.base`` etc. don't redo the work.
            self._decompose_mem(self._mem)
            self._mem_decomposed = True
        return self._mem

    @property
    def crx(self) -> CrxFieldView:
        return self._crx

    @property
    def shift(self) -> ShiftModifierView:
        return self._shift

    @property
    def reg_list(self) -> RegisterListView:
        if not self._reg_list_decomposed and self._decompose_reg_list is not None:
            # Decompose lazily on first access; per-cursor cache so
            # repeated reads of ``reg_list`` etc. don't redo the work.
            self._decompose_reg_list(self._reg_list)
            self._reg_list_decomposed = True
        return self._reg_list

    @property
    def size(self) -> int:
        return self._size

    @property
    def fp_type(self) -> Optional[FpType]:
        return self._fp_type

    @property
    def type_int(self) -> int:
        return self._type_int

    def __deepcopy__(self, memo) -> "_GhidraOperandView":
        clone = _GhidraOperandView(self._arch, self._reg_list._reg_map)
        clone._kind = self._kind
        clone._reg = copy.deepcopy(self._reg, memo)
        clone._imm = self._imm
        # Force materialization of mem decomposition before snapshotting
        # so the deepcopy's mem reflects the current cursor's data.
        if not self._mem_decomposed and self._decompose_mem is not None:
            self._decompose_mem(self._mem)
            self._mem_decomposed = True
        clone._mem = copy.deepcopy(self._mem, memo)
        clone._shift = copy.deepcopy(self._shift, memo)
        clone._crx = copy.deepcopy(self._crx, memo)
        # Mirror MEM materialization for REG_LIST so the clone's reg_list
        # reflects the current cursor's data.
        if not self._reg_list_decomposed and self._decompose_reg_list is not None:
            self._decompose_reg_list(self._reg_list)
            self._reg_list_decomposed = True
        clone._reg_list = copy.deepcopy(self._reg_list, memo)
        clone._size = self._size
        clone._fp_type = self._fp_type
        clone._type_int = self._type_int
        # The decompose callback closes over the original cursor's Java
        # handles; the clone freezes the decomposition (already
        # materialized above) so we don't need to carry it.
        clone._decompose_mem = None
        clone._mem_decomposed = True
        clone._decompose_reg_list = None
        clone._reg_list_decomposed = True
        return clone
