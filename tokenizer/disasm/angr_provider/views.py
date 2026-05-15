"""Owned-view implementations (lazy + reusable wrappers around angr/Capstone).

These concrete classes implement the Protocols in ``tokenizer/disasm/types.py``.
They follow the lifecycle contract documented at the top of that module:
one wrapper instance per kind, mutated in-place as iteration advances; sub-
views are bound to the parent's current cursor and read live; properties
compute on access; ``__deepcopy__`` returns a fresh wrapper bound to the
same backing object so the snapshot becomes stash-safe across iteration
advances on the original.

Backing-object lineage:
    _AngrFunctionView      -> angr ``Function``
    _AngrBlockView         -> angr ``Block`` (yielded by ``Function.blocks``)
    _AngrInstructionView   -> angr ``CapstoneInsn`` (from ``Block.capstone.insns``)
    _AngrOperandView       -> Capstone operand (e.g. ``X86Op``, ``ArmOp``, ``PpcOp``)
    _AngrRegisterView      -> stable ``RegisterView`` cursor reading reg id +
                               ``cs_insn.reg_name(id)`` against the current
                               instruction's Capstone reg-name table.

Architecture detection: provider-level ``_arch`` (resolved once from
``project.arch.name``) is passed into every wrapper at construction. We do
NOT recompute per-instruction -- angr loads one binary per provider, so the
architecture is process-stable.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from tokenizer.disasm.angr_provider.op_classify import (
    _CAPSTONE_ARM_SHIFT_TO_KIND,
    _capstone_op_type_to_operand_kind,
)
from tokenizer.disasm.angr_provider.prefixes import _build_prefixes
from tokenizer.disasm.types import (
    Architecture,
    BlockView,
    BlocksView,
    CrxFieldView,
    FpType,
    InstructionsView,
    InstructionView,
    MemoryOperandView,
    OperandKind,
    OperandsView,
    OperandView,
    PrefixesView,
    RegisterListView,
    RegisterView,
    ShiftKind,
    ShiftModifierView,
)


# ---- RegisterView ---------------------------------------------------------
class _AngrRegisterView:
    """Reusable rich register view bound to a parent ``cs_insn``.

    The Capstone register id (an ``int``) is stored on the cursor; the
    canonical asm name resolves through ``cs_insn.reg_name(id)``. The
    cursor is advanced by mutating ``_set(...)``; ``is_absent`` is True
    when the underlying operand slot is empty (Capstone reports id ``0``
    for an unused base / index / segment slot).
    """

    __slots__ = ("_cs_insn", "_reg_id", "_arch")

    def __init__(self, arch: Architecture) -> None:
        self._cs_insn: Any = None
        self._reg_id: int = 0
        self._arch: Architecture = arch

    def _set(self, cs_insn: Any, reg_id: int) -> None:
        self._cs_insn = cs_insn
        self._reg_id = reg_id

    @property
    def name(self) -> str:
        if self._reg_id == 0 or self._cs_insn is None:
            return ""
        return self._cs_insn.reg_name(self._reg_id) or ""

    @property
    def id(self) -> int:
        return self._reg_id

    @property
    def arch(self) -> Architecture:
        return self._arch

    @property
    def is_absent(self) -> bool:
        return self._reg_id == 0

    def __deepcopy__(self, memo) -> "_AngrRegisterView":
        clone = _AngrRegisterView(self._arch)
        clone._cs_insn = self._cs_insn
        clone._reg_id = self._reg_id
        return clone


# ---- Sub-views (bound to parent OperandView) ------------------------------
class _AngrMemoryOperandView:
    """Reusable memory-operand sub-view; reads through to the parent operand's
    Capstone ``op.mem`` substructure.

    ``base`` / ``index`` / ``segment`` are returned as ``RegisterView``s; we
    own three nested ``_AngrRegisterView`` cursors (one per slot) so each
    typed read is self-contained even though the wrappers are reused. The
    ``_segment_supported`` flag captures whether the active ISA's Capstone
    operand carries a segment field (only x86 does); on other ISAs the
    segment register view is permanently absent.
    """

    __slots__ = (
        "_op",
        "_cs_insn",
        "_arch",
        "_base",
        "_index",
        "_segment",
        "_segment_supported",
        "_index_shift",
    )

    def __init__(self, arch: Architecture) -> None:
        self._op: Any = None
        self._cs_insn: Any = None
        self._arch: Architecture = arch
        self._base = _AngrRegisterView(arch)
        self._index = _AngrRegisterView(arch)
        self._segment = _AngrRegisterView(arch)
        self._segment_supported: bool = arch == Architecture.X86
        # Index-shift sentinel: Capstone's mem sub-view doesn't carry an
        # independent shift-on-index field (the ``op.shift`` on an ARM
        # operand applies to the whole operand, which on x86/etc. is
        # unrelated to a mem-index shift). Sentinel-NONE keeps the
        # Protocol shape clean without engineering parity for the angr
        # path (see ``angr_limitations.md``).
        self._index_shift = _AngrShiftModifierView()

    def _set(self, cs_insn: Any, op: Any) -> None:
        self._cs_insn = cs_insn
        self._op = op

    @property
    def base(self) -> RegisterView:
        mem = self._op.mem if self._op is not None else None
        base_id = int(mem.base) if mem is not None else 0
        self._base._set(self._cs_insn, base_id)
        return self._base

    @property
    def index(self) -> RegisterView:
        mem = self._op.mem if self._op is not None else None
        # MIPS / PPC / RISC-V Capstone operands have no ``index`` field.
        index_id = int(getattr(mem, "index", 0)) if mem is not None else 0
        self._index._set(self._cs_insn, index_id)
        return self._index

    @property
    def scale(self) -> int:
        mem = self._op.mem if self._op is not None else None
        return int(getattr(mem, "scale", 1)) if mem is not None else 1

    @property
    def disp(self) -> int:
        mem = self._op.mem if self._op is not None else None
        return int(mem.disp) if mem is not None else 0

    @property
    def segment(self) -> RegisterView:
        if not self._segment_supported:
            self._segment._set(self._cs_insn, 0)
            return self._segment
        mem = self._op.mem if self._op is not None else None
        seg_id = int(getattr(mem, "segment", 0)) if mem is not None else 0
        self._segment._set(self._cs_insn, seg_id)
        return self._segment

    # ARM writeback / pre-/post-indexed addressing-mode flags. Capstone
    # encodes writeback as an instruction-level flag (``ARM_OP_INVALID``
    # placeholder + ``writeback`` on the instruction struct), not on the
    # per-operand mem sub-view; the consumer side currently treats the
    # angr/Capstone path as the best-effort backend and the v2 emitter
    # falls back to plain offset rendering when these flags are False
    # (see ``angr_limitations.md``). Sentinel-False keeps the Protocol
    # shape clean without engineering parity for the angr path.
    @property
    def writeback(self) -> bool:
        return False

    @property
    def pre_indexed(self) -> bool:
        return False

    @property
    def post_indexed(self) -> bool:
        return False

    @property
    def index_shift(self) -> ShiftModifierView:
        # Sentinel-NONE shift; ``_op`` is not bound so the underlying
        # reader returns ``ShiftKind.NONE`` + amount=0 unconditionally.
        return self._index_shift

    def __deepcopy__(self, memo) -> "_AngrMemoryOperandView":
        clone = _AngrMemoryOperandView(self._arch)
        clone._op = self._op
        clone._cs_insn = self._cs_insn
        return clone


class _AngrShiftModifierView:
    """Reusable ARM shift-modifier sub-view (``op.shift.{type, value}``).

    On non-ARM ISAs, Capstone operands have no ``shift`` field; the cursor
    keeps the empty default (``ShiftKind.NONE`` / amount=0).
    """

    __slots__ = ("_op",)

    def __init__(self) -> None:
        self._op: Any = None

    def _set(self, op: Any) -> None:
        self._op = op

    @property
    def kind(self) -> ShiftKind:
        if self._op is None:
            return ShiftKind.NONE
        shift = getattr(self._op, "shift", None)
        if shift is None:
            return ShiftKind.NONE
        return _CAPSTONE_ARM_SHIFT_TO_KIND.get(int(shift.type), ShiftKind.NONE)

    @property
    def amount(self) -> int:
        if self._op is None:
            return 0
        shift = getattr(self._op, "shift", None)
        if shift is None:
            return 0
        return int(shift.value)

    def __deepcopy__(self, memo) -> "_AngrShiftModifierView":
        clone = _AngrShiftModifierView()
        clone._op = self._op
        return clone


class _AngrCrxFieldView:
    """Reusable PPC CRX-operand sub-view (``op.crx.reg``).

    On non-PPC ISAs (or non-CRX operands), the backing ``op.crx`` is
    absent; ``reg`` returns an absent ``RegisterView``.
    """

    __slots__ = ("_op", "_cs_insn", "_arch", "_reg")

    def __init__(self, arch: Architecture) -> None:
        self._op: Any = None
        self._cs_insn: Any = None
        self._arch: Architecture = arch
        self._reg = _AngrRegisterView(arch)

    def _set(self, cs_insn: Any, op: Any) -> None:
        self._cs_insn = cs_insn
        self._op = op

    @property
    def reg(self) -> RegisterView:
        crx = getattr(self._op, "crx", None) if self._op is not None else None
        reg_id = int(getattr(crx, "reg", 0)) if crx is not None else 0
        self._reg._set(self._cs_insn, reg_id)
        return self._reg

    def __deepcopy__(self, memo) -> "_AngrCrxFieldView":
        clone = _AngrCrxFieldView(self._arch)
        clone._op = self._op
        clone._cs_insn = self._cs_insn
        return clone


class _AngrRegisterListView:
    """Sentinel reg-list sub-view for the angr/Capstone path.

    Capstone splits ARM stm/ldm reg-list members into independent REG
    operands (see ``angr_limitations.md``); the angr-backed
    ``_AngrOperandView`` therefore never reports
    ``OperandKind.REG_LIST``. This sentinel exists only to satisfy the
    runtime-checkable ``OperandView`` / ``RegisterListView`` Protocol
    isinstance check: ``base`` is permanently ``is_absent``,
    ``writeback`` is False, the container is empty.

    Consumers gate reg-list reads on ``op.kind == OperandKind.REG_LIST``
    (see ``tokenizer/arch/arm32/provider.py``) so this property is
    unreachable in normal flow on the angr path.
    """

    __slots__ = ("_base",)

    def __init__(self, arch: Architecture) -> None:
        # _AngrRegisterView defaults to _reg_id = 0 -> is_absent True.
        self._base = _AngrRegisterView(arch)

    @property
    def base(self) -> RegisterView:
        return self._base

    @property
    def writeback(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[RegisterView]:
        return iter(())

    def __getitem__(self, idx: int) -> RegisterView:
        raise IndexError(idx)

    def __deepcopy__(self, memo) -> "_AngrRegisterListView":
        return _AngrRegisterListView(self._base._arch)


# ---- OperandView ----------------------------------------------------------
class _AngrOperandView:
    """Reusable per-operand cursor over a Capstone operand object.

    Sub-views (``mem`` / ``shift`` / ``crx``) own their own cursors and
    return self-contained ``RegisterView`` instances. ``reg`` is its own
    cursor so callers can read ``op.reg`` directly (REG operand kind);
    ``imm`` and ``size`` are direct property reads.

    ``fp_type`` reads ``op.fp_type`` against the Capstone operand class
    default (stamped to ``None`` at module load via ``_stamp_fp_type_default``,
    see top of file + ``angr_limitations.md`` §1).
    """

    __slots__ = ("_cs_insn", "_op", "_arch", "_reg", "_mem", "_shift", "_crx", "_reg_list")

    def __init__(self, arch: Architecture) -> None:
        self._cs_insn: Any = None
        self._op: Any = None
        self._arch: Architecture = arch
        self._reg = _AngrRegisterView(arch)
        self._mem = _AngrMemoryOperandView(arch)
        self._shift = _AngrShiftModifierView()
        self._crx = _AngrCrxFieldView(arch)
        self._reg_list = _AngrRegisterListView(arch)

    def _set(self, cs_insn: Any, op: Any) -> None:
        self._cs_insn = cs_insn
        self._op = op

    @property
    def kind(self) -> OperandKind:
        if self._op is None:
            return OperandKind.INVALID
        return _capstone_op_type_to_operand_kind(int(self._op.type))

    @property
    def reg(self) -> RegisterView:
        # REG operand: ``op.reg`` is the register id. For non-REG operands
        # the Capstone union still exposes the field but it's meaningless;
        # consumers gate by ``kind`` so we just plumb the value.
        reg_id = int(getattr(self._op, "reg", 0)) if self._op is not None else 0
        self._reg._set(self._cs_insn, reg_id)
        return self._reg

    @property
    def imm(self) -> int:
        return int(self._op.imm) if self._op is not None else 0

    @property
    def mem(self) -> MemoryOperandView:
        self._mem._set(self._cs_insn, self._op)
        return self._mem

    @property
    def crx(self) -> CrxFieldView:
        self._crx._set(self._cs_insn, self._op)
        return self._crx

    @property
    def reg_list(self) -> RegisterListView:
        # Sentinel: Capstone never emits OperandKind.REG_LIST on the angr
        # path (see op_classify.py + angr_limitations.md). Property is
        # Protocol-satisfying and unreachable in normal flow.
        return self._reg_list

    @property
    def shift(self) -> ShiftModifierView:
        self._shift._set(self._op)
        return self._shift

    @property
    def size(self) -> int:
        return int(getattr(self._op, "size", 0)) if self._op is not None else 0

    @property
    def fp_type(self) -> Optional[FpType]:
        # Class-default ``fp_type = None`` stamped at module load
        # (see ``_stamp_fp_type_default`` + ``angr_limitations.md`` §1);
        # angr never populates this — Ghidra is the only provider that
        # produces a non-``None`` value.
        if self._op is None:
            return None
        value = getattr(self._op, "fp_type", None)
        return value if isinstance(value, FpType) else None

    @property
    def type_int(self) -> int:
        return int(self._op.type) if self._op is not None else 0

    def __deepcopy__(self, memo) -> "_AngrOperandView":
        clone = _AngrOperandView(self._arch)
        clone._cs_insn = self._cs_insn
        clone._op = self._op
        return clone


# ---- Container views ------------------------------------------------------
# Each container is a thin lazy wrapper: ``__len__`` is the only eager
# probe; ``__iter__`` walks the backing sequence yielding the SAME reused
# child cursor mutated in place.
class _AngrOperandsView:
    """Container over ``cs_insn.operands`` yielding the reused operand cursor."""

    __slots__ = ("_cs_insn", "_operand_cursor")

    def __init__(self, operand_cursor: "_AngrOperandView") -> None:
        self._cs_insn: Any = None
        self._operand_cursor = operand_cursor

    def _set(self, cs_insn: Any) -> None:
        self._cs_insn = cs_insn

    @property
    def _ops_seq(self) -> Any:
        if self._cs_insn is None:
            return ()
        # angr's ``CapstoneInsn`` proxies ``operands`` through ``__getattr__``
        # to the raw ``cs_insn.operands`` list.
        return self._cs_insn.operands or ()

    def __len__(self) -> int:
        return len(self._ops_seq)

    def __iter__(self) -> Iterator[OperandView]:
        for op in self._ops_seq:
            self._operand_cursor._set(self._cs_insn, op)
            yield self._operand_cursor


class _AngrPrefixesView:
    """Container over a precomputed list of typed prefix instances.

    Prefix count per instruction is small (0-3 typical), so unlike the
    operand hot path we materialise a plain list here. ``__getitem__`` IS
    supported per the Protocol (prefix instances are typed-distinct, not
    a single reused view).
    """

    __slots__ = ("_prefixes",)

    def __init__(self) -> None:
        self._prefixes: list = []

    def _set(self, prefixes: list) -> None:
        self._prefixes = prefixes

    def __len__(self) -> int:
        return len(self._prefixes)

    def __iter__(self):
        return iter(self._prefixes)

    def __getitem__(self, idx: int):
        return self._prefixes[idx]


# ---- InstructionView ------------------------------------------------------
class _AngrInstructionView:
    """Reusable instruction cursor over a ``CapstoneInsn``.

    ``mnemonic`` returns the Capstone ``mnemonic`` field verbatim (which on
    x86 INCLUDES the leading prefix word, e.g. ``"repe cmpsb"``);
    ``base_mnemonic`` is the Capstone ``insn_name()`` (the mnemonic without
    the prefix word, e.g. ``"cmpsb"``). The two surfaces match the
    contract in ``tokenizer/disasm/types.py``.

    Prefix construction is per-architecture (see ``_build_prefixes``); the
    cursor caches the materialised list per advance.
    """

    __slots__ = ("_cs_insn", "_arch", "_operands", "_prefixes_view")

    def __init__(self, arch: Architecture) -> None:
        self._cs_insn: Any = None
        self._arch: Architecture = arch
        self._operands = _AngrOperandsView(_AngrOperandView(arch))
        self._prefixes_view = _AngrPrefixesView()

    def _set(self, cs_insn: Any) -> None:
        self._cs_insn = cs_insn
        self._operands._set(cs_insn)
        # _build_prefixes lives below in commit G.2.b; until that commit
        # lands, prefixes are an empty list.
        self._prefixes_view._set(_build_prefixes(cs_insn, self._arch) if cs_insn is not None else [])

    @property
    def address(self) -> int:
        return int(self._cs_insn.address) if self._cs_insn is not None else 0

    @property
    def mnemonic(self) -> str:
        return str(self._cs_insn.mnemonic) if self._cs_insn is not None else ""

    @property
    def base_mnemonic(self) -> str:
        if self._cs_insn is None:
            return ""
        # ``CapstoneInsn`` proxies ``insn_name`` through ``__getattr__`` to
        # the raw ``cs_insn.insn_name()``.
        return str(self._cs_insn.insn_name()) or ""

    @property
    def op_str(self) -> str:
        return str(self._cs_insn.op_str) if self._cs_insn is not None else ""

    @property
    def operands(self) -> OperandsView:
        return self._operands

    @property
    def prefixes(self) -> PrefixesView:
        return self._prefixes_view

    def __deepcopy__(self, memo) -> "_AngrInstructionView":
        clone = _AngrInstructionView(self._arch)
        clone._set(self._cs_insn)
        return clone


class _AngrInstructionsView:
    """Container over the ``CapstoneInsn`` list of an angr ``Block``.

    Reads through ``block.capstone.insns`` (Capstone-decoded list); each
    iteration advances the reused instruction cursor. ``__len__`` is the
    list length; the underlying ``capstone`` property on angr's ``Block``
    caches its decoded value, so re-reading is cheap.
    """

    __slots__ = ("_block", "_insn_cursor")

    def __init__(self, insn_cursor: "_AngrInstructionView") -> None:
        self._block: Any = None
        self._insn_cursor = insn_cursor

    def _set(self, block: Any) -> None:
        self._block = block

    @property
    def _insn_list(self) -> list:
        if self._block is None:
            return []
        return self._block.capstone.insns or []

    def __len__(self) -> int:
        return len(self._insn_list)

    def __iter__(self) -> Iterator[InstructionView]:
        for cs_insn in self._insn_list:
            self._insn_cursor._set(cs_insn)
            yield self._insn_cursor


# ---- BlockView ------------------------------------------------------------
class _AngrBlockView:
    """Reusable block cursor over an angr ``Block``."""

    __slots__ = ("_block", "_arch", "_instructions")

    def __init__(self, arch: Architecture) -> None:
        self._block: Any = None
        self._arch: Architecture = arch
        self._instructions = _AngrInstructionsView(_AngrInstructionView(arch))

    def _set(self, block: Any) -> None:
        self._block = block
        self._instructions._set(block)

    @property
    def addr(self) -> int:
        return int(self._block.addr) if self._block is not None else 0

    @property
    def size(self) -> int:
        return int(self._block.size) if self._block is not None else 0

    @property
    def instructions(self) -> InstructionsView:
        return self._instructions

    def __deepcopy__(self, memo) -> "_AngrBlockView":
        clone = _AngrBlockView(self._arch)
        clone._set(self._block)
        return clone


class _AngrBlocksView:
    """Container over an angr ``Function.blocks`` generator.

    angr's ``Function.blocks`` is a generator (see
    ``angr/knowledge_plugins/functions/function.py:271``); to satisfy the
    Protocol's ``__len__`` we read ``len(func._local_blocks)`` (the
    underlying dict the generator iterates). ``__iter__`` re-invokes the
    generator on every call, so the container is reusable across multiple
    ``for block in func.blocks`` passes within the same function.
    """

    __slots__ = ("_func", "_block_cursor")

    def __init__(self, block_cursor: "_AngrBlockView") -> None:
        self._func: Any = None
        self._block_cursor = block_cursor

    def _set(self, func: Any) -> None:
        self._func = func

    def __len__(self) -> int:
        if self._func is None:
            return 0
        # ``Function._local_blocks`` is the dict the ``blocks`` generator
        # iterates (see ``angr/knowledge_plugins/functions/function.py``).
        local_blocks = getattr(self._func, "_local_blocks", None)
        if local_blocks is not None:
            return len(local_blocks)
        # Fallback: drain the generator into a counter. Slower; only used
        # if angr renames ``_local_blocks`` in a future release.
        return sum(1 for _ in self._func.blocks)

    def __iter__(self) -> Iterator[BlockView]:
        if self._func is None:
            return
        for block in self._func.blocks:
            self._block_cursor._set(block)
            yield self._block_cursor


# ---- FunctionView ---------------------------------------------------------
class _AngrFunctionView:
    """Reusable function cursor over an angr ``Function``.

    The provider mutates the same instance for every ``iter_functions``
    step (see lifecycle docstring at the top of
    ``tokenizer/disasm/types.py``).
    """

    __slots__ = ("_func", "_arch", "_blocks")

    def __init__(self, arch: Architecture) -> None:
        self._func: Any = None
        self._arch: Architecture = arch
        self._blocks = _AngrBlocksView(_AngrBlockView(arch))

    def _set(self, func: Any) -> None:
        self._func = func
        self._blocks._set(func)

    @property
    def entry(self) -> int:
        return int(self._func.addr) if self._func is not None else 0

    @property
    def name(self) -> str:
        return str(self._func.name) if self._func is not None else ""

    @property
    def blocks(self) -> BlocksView:
        return self._blocks

    def __deepcopy__(self, memo) -> "_AngrFunctionView":
        clone = _AngrFunctionView(self._arch)
        clone._set(self._func)
        return clone
