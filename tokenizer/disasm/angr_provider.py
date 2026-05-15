import copy
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import angr

from tokenizer.disasm import DisassemblyProvider, MetadataLookup
from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
)
from tokenizer.disasm.types import (
    AddressSizePrefixView,
    Architecture,
    ArmConditionCode,
    BlockView,
    BlocksView,
    BranchHintPrefixView,
    ConditionCodePrefixView,
    CrxFieldView,
    FpType,
    FunctionView,
    InstructionPrefixView,
    InstructionView,
    InstructionsView,
    LockPrefixView,
    MemoryOperandView,
    OperandKind,
    OperandSizePrefixView,
    OperandView,
    OperandsView,
    PpcBranchConditionPrefixView,
    PpcUpdateCr0PrefixView,
    PrefixesView,
    RegisterView,
    RepPrefixView,
    SegmentOverridePrefixView,
    ShiftKind,
    ShiftModifierView,
    UpdateFlagsPrefixView,
    WritebackPrefixView,
    X86BranchHint,
    X86Segment,
)


# ---------------------------------------------------------------------------
# Capstone-operand uniform ``fp_type`` default
# ---------------------------------------------------------------------------
# The angr path delivers raw Capstone CsOpnd objects (X86Op, ArmOp, ...) to
# consumer code. Capstone never populates an FP-precision signal on these,
# so the angr-side ``op.fp_type`` is uniformly ``None`` (matches the typed
# ``Optional[FpType]`` shape exposed by the Ghidra path's ``_CapOperand``;
# see ``tokenizer/disasm/types.py``). Stamping the default at module load
# (rather than per-instance per-instruction) keeps the consumer API uniform
# across providers — ``op.fp_type`` is a direct typed read with no
# ``getattr`` soft-probe — and avoids touching the Capstone object on the
# hot path. ``angr_limitations.md`` §1 documents why this field stays
# ``None`` on the angr side.
def _stamp_fp_type_default() -> None:
    """Attach class-level ``fp_type = None`` defaults to every Capstone
    operand class the angr-backed providers deliver to consumers.

    Only the classes we actually traverse are stamped; per-ISA imports are
    wrapped so an ISA whose Capstone bindings are unavailable in the active
    install (e.g. a stripped Capstone build) is silently skipped.
    """
    for module_name, class_name in (
        ("capstone.x86", "X86Op"),
        ("capstone.arm", "ArmOp"),
        ("capstone.arm64", "Arm64Op"),
        ("capstone.mips", "MipsOp"),
        ("capstone.ppc", "PpcOp"),
        ("capstone.riscv", "RiscvOp"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        # Skip if a value is already present (e.g. a future Capstone release
        # exposes the field natively or another module already stamped it).
        if "fp_type" not in cls.__dict__:
            cls.fp_type = None


_stamp_fp_type_default()


# ---------------------------------------------------------------------------
# Typed view for the angr-side MetadataLookup
# ---------------------------------------------------------------------------
class _AngrAddressMetadataView:
    """Concrete typed view returned by ``AngrMetadataLookup.lookup()``.

    Pure storage + read-only typed properties. ``AngrMetadataLookup`` calls
    ``_populate(...)`` to populate every slot at lookup time; consumers
    read typed properties exclusively. angr cannot resolve slot targets
    (``angr_limitations.md`` sections 2-3), so ``slot_target`` /
    ``jump_table_base_addr`` / ``jump_table_offset`` always return
    ``None``; ``string_encoding`` is always ASCII or UNKNOWN
    (``angr_limitations.md`` section 4).

    LIFECYCLE: instance is REUSED across ``lookup()`` calls. Use
    ``copy.deepcopy(view)`` to stash across lookups.
    """

    __slots__ = (
        "_kind",
        "_section_kind",
        "_section_name",
        "_string_encoding",
        "_string_bytes",
        "_name",
        "_start_addr",
        "_end_addr",
        "_size",
        "_library",
        "_is_vtable",
        "_tls",
    )

    def __init__(self) -> None:
        self._kind: AddressKind = AddressKind.NONE
        self._section_kind: SectionKind = SectionKind.UNKNOWN
        self._section_name: Optional[str] = None
        self._string_encoding: Encoding = Encoding.UNKNOWN
        self._string_bytes: Optional[bytes] = None
        self._name: Optional[str] = None
        self._start_addr: Optional[int] = None
        self._end_addr: Optional[int] = None
        self._size: Optional[int] = None
        self._library: Optional[str] = None
        self._is_vtable: bool = False
        self._tls: bool = False

    def _populate(
        self,
        *,
        kind: AddressKind,
        section_kind: SectionKind,
        section_name: Optional[str],
        string_encoding: Encoding,
        string_bytes: Optional[bytes],
        name: Optional[str],
        start_addr: Optional[int],
        end_addr: Optional[int],
        size: Optional[int],
        library: Optional[str],
        is_vtable: bool,
        tls: bool,
    ) -> None:
        """Replace all slot state in one call. Used by the lookup at
        the start of every ``lookup()`` so the consumer sees a consistent
        view bound to the current address.
        """
        self._kind = kind
        self._section_kind = section_kind
        self._section_name = section_name
        self._string_encoding = string_encoding
        self._string_bytes = string_bytes
        self._name = name
        self._start_addr = start_addr
        self._end_addr = end_addr
        self._size = size
        self._library = library
        self._is_vtable = is_vtable
        self._tls = tls

    # -- Typed property surface (AddressMetadataView Protocol) --------------
    @property
    def kind(self) -> AddressKind:
        return self._kind

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def section_kind(self) -> SectionKind:
        return self._section_kind

    @property
    def section_name(self) -> Optional[str]:
        return self._section_name

    @property
    def start_addr(self) -> Optional[int]:
        return self._start_addr

    @property
    def end_addr(self) -> Optional[int]:
        return self._end_addr

    @property
    def size(self) -> Optional[int]:
        return self._size

    @property
    def library(self) -> Optional[str]:
        return self._library

    @property
    def string_encoding(self) -> Encoding:
        return self._string_encoding

    @property
    def string_bytes(self) -> Optional[bytes]:
        return self._string_bytes

    @property
    def is_vtable(self) -> bool:
        return self._is_vtable

    @property
    def tls(self) -> bool:
        return self._tls

    @property
    def slot_target(self) -> Optional[AddressMetadataView]:
        # angr cannot resolve slot targets (angr_limitations.md sections 2-3).
        return None

    @property
    def jump_table_base_addr(self) -> Optional[int]:
        return None

    @property
    def jump_table_offset(self) -> Optional[int]:
        return None

    def __deepcopy__(self, memo) -> "_AngrAddressMetadataView":
        clone = _AngrAddressMetadataView()
        clone._kind = self._kind
        clone._section_kind = self._section_kind
        clone._section_name = self._section_name
        clone._string_encoding = self._string_encoding
        # bytes is immutable; safe to share
        clone._string_bytes = self._string_bytes
        clone._name = self._name
        clone._start_addr = self._start_addr
        clone._end_addr = self._end_addr
        clone._size = self._size
        clone._library = self._library
        clone._is_vtable = self._is_vtable
        clone._tls = self._tls
        return clone


# ---------------------------------------------------------------------------
# Owned-view implementations (lazy + reusable wrappers around angr/Capstone)
# ---------------------------------------------------------------------------
# These concrete classes implement the Protocols in ``tokenizer/disasm/types.py``.
# They follow the lifecycle contract documented at the top of that module:
# one wrapper instance per kind, mutated in-place as iteration advances; sub-
# views are bound to the parent's current cursor and read live; properties
# compute on access; ``__deepcopy__`` returns a fresh wrapper bound to the
# same backing object so the snapshot becomes stash-safe across iteration
# advances on the original.
#
# Backing-object lineage:
#     _AngrFunctionView      -> angr ``Function``
#     _AngrBlockView         -> angr ``Block`` (yielded by ``Function.blocks``)
#     _AngrInstructionView   -> angr ``CapstoneInsn`` (from ``Block.capstone.insns``)
#     _AngrOperandView       -> Capstone operand (e.g. ``X86Op``, ``ArmOp``, ``PpcOp``)
#     _AngrRegisterView      -> stable ``RegisterView`` cursor reading reg id +
#                                ``cs_insn.reg_name(id)`` against the current
#                                instruction's Capstone reg-name table.
#
# Architecture detection: provider-level ``_arch`` (resolved once from
# ``project.arch.name``) is passed into every wrapper at construction. We do
# NOT recompute per-instruction — angr loads one binary per provider, so the
# architecture is process-stable.

# archinfo arch.name -> owned ``Architecture`` enum. Centralised here so each
# wrapper does not need to know about archinfo internals.
_ARCHINFO_NAME_TO_ARCHITECTURE: dict[str, Architecture] = {
    "X86": Architecture.X86,
    "AMD64": Architecture.X86,
    "ARMEL": Architecture.ARM32,
    "ARMHF": Architecture.ARM32,
    "ARMCortexM": Architecture.ARM32,
    "AARCH64": Architecture.AARCH64,
    "MIPS32": Architecture.MIPS,
    "MIPS64": Architecture.MIPS,
    "PPC32": Architecture.PPC,
    "PPC64": Architecture.PPC,
    "RISCV64": Architecture.RISCV,
}


def _resolve_architecture(angr_arch: Any) -> Architecture:
    """Map an archinfo ``Arch`` instance onto our ``Architecture`` enum."""
    name = getattr(angr_arch, "name", None)
    if isinstance(name, str):
        mapped = _ARCHINFO_NAME_TO_ARCHITECTURE.get(name)
        if mapped is not None:
            return mapped
    return Architecture.UNKNOWN


# Capstone operand-type integer (REG=1/IMM=2/MEM=3, PPC CRX=64, ARM extras
# 64..67, ...) -> owned ``OperandKind``. ``OperandKind.OTHER`` covers any
# non-REG/IMM/MEM/CRX value (FP, CIMM, PIMM, SETEND, SYSREG, ...). The raw
# integer is preserved on ``OperandView.type_int`` so consumers that need
# the precise discriminator (e.g. emitting ``op_<n>`` platform tokens for
# ARM extras, see ``tokenizer/arch/arm32/provider.py``) can read it without
# losing the typed kind dispatch.
_CAPSTONE_OP_TYPE_REG = 1
_CAPSTONE_OP_TYPE_IMM = 2
_CAPSTONE_OP_TYPE_MEM = 3
_CAPSTONE_OP_TYPE_PPC_CRX = 64


def _capstone_op_type_to_operand_kind(op_type: int) -> OperandKind:
    if op_type == _CAPSTONE_OP_TYPE_REG:
        return OperandKind.REG
    if op_type == _CAPSTONE_OP_TYPE_IMM:
        return OperandKind.IMM
    if op_type == _CAPSTONE_OP_TYPE_MEM:
        return OperandKind.MEM
    if op_type == _CAPSTONE_OP_TYPE_PPC_CRX:
        return OperandKind.CRX
    if op_type == 0:
        return OperandKind.INVALID
    return OperandKind.OTHER


# Capstone ARM shift type -> owned ``ShiftKind``. Capstone uses the encoding
# ASR=1, LSL=2, LSR=3, ROR=4, RRX=5 (see ``capstone/arm_const.py``). The
# ``_REG`` variants (6..10) carry the same semantic kind with a register-
# valued amount; we collapse them onto the same enum entries (the consumer
# distinguishes register vs immediate amount via the operand structure on
# the parent, not via the shift kind).
_CAPSTONE_ARM_SHIFT_TO_KIND: dict[int, ShiftKind] = {
    0: ShiftKind.NONE,
    1: ShiftKind.ASR,
    2: ShiftKind.LSL,
    3: ShiftKind.LSR,
    4: ShiftKind.ROR,
    5: ShiftKind.RRX,
    6: ShiftKind.ASR,  # ASR_REG
    7: ShiftKind.LSL,  # LSL_REG
    8: ShiftKind.LSR,  # LSR_REG
    9: ShiftKind.ROR,  # ROR_REG
    10: ShiftKind.RRX,  # RRX_REG
}


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

    __slots__ = ("_op", "_cs_insn", "_arch", "_base", "_index", "_segment", "_segment_supported")

    def __init__(self, arch: Architecture) -> None:
        self._op: Any = None
        self._cs_insn: Any = None
        self._arch: Architecture = arch
        self._base = _AngrRegisterView(arch)
        self._index = _AngrRegisterView(arch)
        self._segment = _AngrRegisterView(arch)
        self._segment_supported: bool = arch == Architecture.X86

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

    __slots__ = ("_cs_insn", "_op", "_arch", "_reg", "_mem", "_shift", "_crx")

    def __init__(self, arch: Architecture) -> None:
        self._cs_insn: Any = None
        self._op: Any = None
        self._arch: Architecture = arch
        self._reg = _AngrRegisterView(arch)
        self._mem = _AngrMemoryOperandView(arch)
        self._shift = _AngrShiftModifierView()
        self._crx = _AngrCrxFieldView(arch)

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


# ---------------------------------------------------------------------------
# Per-platform instruction-prefix builders
# ---------------------------------------------------------------------------
# ``_build_prefixes(cs_insn, arch)`` is the single dispatcher that yields a
# typed ``list[InstructionPrefixView]`` for a Capstone instruction. Per-arch
# helpers live alongside it; consumers see only the dispatcher.
#
# Concrete subclasses for the prefix Protocols that carry data:
#     RepPrefixView, SegmentOverridePrefixView, BranchHintPrefixView,
#     ConditionCodePrefixView, PpcBranchConditionPrefixView
# The data-less prefix types (LockPrefixView, OperandSizePrefixView,
# AddressSizePrefixView, UpdateFlagsPrefixView, WritebackPrefixView,
# PpcUpdateCr0PrefixView) are concrete on the Protocol side and instantiated
# directly.


class _RepPrefix(RepPrefixView):
    __slots__ = ("_repeat_until_zero",)

    def __init__(self, repeat_until_zero: bool) -> None:
        self._repeat_until_zero = repeat_until_zero

    @property
    def repeat_until_zero(self) -> bool:
        return self._repeat_until_zero


class _SegmentOverridePrefix(SegmentOverridePrefixView):
    __slots__ = ("_segment",)

    def __init__(self, segment: X86Segment) -> None:
        self._segment = segment

    @property
    def segment(self) -> X86Segment:
        return self._segment


class _BranchHintPrefix(BranchHintPrefixView):
    __slots__ = ("_hint",)

    def __init__(self, hint: X86BranchHint) -> None:
        self._hint = hint

    @property
    def hint(self) -> X86BranchHint:
        return self._hint


class _ConditionCodePrefix(ConditionCodePrefixView):
    __slots__ = ("_cc",)

    def __init__(self, cc: ArmConditionCode) -> None:
        self._cc = cc

    @property
    def cc(self) -> ArmConditionCode:
        return self._cc


class _PpcBranchConditionPrefix(PpcBranchConditionPrefixView):
    __slots__ = ("_bc",)

    def __init__(self, bc: int) -> None:
        self._bc = bc

    @property
    def bc(self) -> int:
        return self._bc


# x86 prefix-byte tables. Lookup-by-byte avoids per-byte if-ladders inside
# the per-instruction hot path; the ``_X86_REP_PREFIXES`` and
# ``_X86_SEGMENT_PREFIXES`` tables are exhaustive for the bytes Capstone
# fills into ``cs_insn.prefix``.
_X86_LOCK_BYTE = 0xF0
_X86_REP_PREFIXES: dict[int, bool] = {
    # repeat_until_zero=False -> REPNE/REPNZ
    0xF2: False,
    # repeat_until_zero=True  -> REP / REPE / REPZ
    0xF3: True,
}
_X86_SEGMENT_PREFIXES: dict[int, X86Segment] = {
    0x2E: X86Segment.CS,
    0x36: X86Segment.SS,
    0x3E: X86Segment.DS,
    0x26: X86Segment.ES,
    0x64: X86Segment.FS,
    0x65: X86Segment.GS,
}
_X86_OPERAND_SIZE_BYTE = 0x66
_X86_ADDRESS_SIZE_BYTE = 0x67
_X86_BRANCH_HINTS: dict[int, X86BranchHint] = {
    # In branch context, 0x2E = "branch hint not taken" and 0x3E =
    # "branch hint taken". Outside branch context they are CS / DS
    # segment overrides — disambiguated in ``_build_prefixes_x86``.
    0x2E: X86BranchHint.NOT_TAKEN,
    0x3E: X86BranchHint.TAKEN,
}
# Capstone x86 group ids used to detect branch context for the
# CS / DS hint disambiguation. Stable values from
# ``capstone/x86_const.py`` (X86_GRP_JUMP=1, X86_GRP_BRANCH_RELATIVE=7).
_X86_BRANCH_GROUP_IDS: frozenset[int] = frozenset({1, 7})


def _x86_is_branch_instruction(cs_insn: Any) -> bool:
    """True if ``cs_insn`` belongs to an x86 branch group.

    Used to disambiguate the dual meaning of 0x2E / 0x3E (segment override
    vs branch hint). Capstone's ``cs_insn.groups`` is a tuple of integer
    group ids; we match against the jump / branch-relative groups.
    """
    groups = getattr(cs_insn, "groups", None) or ()
    for g in groups:
        if int(g) in _X86_BRANCH_GROUP_IDS:
            return True
    return False


def _build_prefixes_x86(cs_insn: Any) -> list[InstructionPrefixView]:
    """Decode ``cs_insn.prefix`` (a 4-byte array) into typed prefix views.

    Capstone fills the 4-slot ``cs_insn.prefix`` array with up to four
    legacy prefix bytes (lock, rep, segment, operand-size, address-size,
    branch-hint). Slots not in use are zero. We walk the array once and
    map each non-zero byte through the lookup tables above.
    """
    prefix_bytes = getattr(cs_insn, "prefix", None)
    if prefix_bytes is None:
        return []

    is_branch = _x86_is_branch_instruction(cs_insn)
    out: list[InstructionPrefixView] = []
    for byte in prefix_bytes:
        b = int(byte)
        if b == 0:
            continue
        if b == _X86_LOCK_BYTE:
            out.append(LockPrefixView())
            continue
        rep = _X86_REP_PREFIXES.get(b)
        if rep is not None:
            out.append(_RepPrefix(repeat_until_zero=rep))
            continue
        if b == _X86_OPERAND_SIZE_BYTE:
            out.append(OperandSizePrefixView())
            continue
        if b == _X86_ADDRESS_SIZE_BYTE:
            out.append(AddressSizePrefixView())
            continue
        if is_branch and b in _X86_BRANCH_HINTS:
            out.append(_BranchHintPrefix(hint=_X86_BRANCH_HINTS[b]))
            continue
        seg = _X86_SEGMENT_PREFIXES.get(b)
        if seg is not None:
            out.append(_SegmentOverridePrefix(segment=seg))
            continue
        # Unknown byte — silently skip; Capstone occasionally surfaces
        # REX (0x40-0x4F) or VEX/EVEX bytes here on long-mode encodings,
        # which carry no semantic prefix for our token surface.
    return out


# ARM Capstone condition-code id -> typed enum. Capstone uses 1..14 for
# EQ..LE; 0 = invalid, 15 = AL (always). The "always" condition is
# omitted (it's the implicit default and the consumer surface skips it).
_ARM_CC_TO_ENUM: dict[int, ArmConditionCode] = {
    1: ArmConditionCode.EQ,
    2: ArmConditionCode.NE,
    3: ArmConditionCode.CS,
    4: ArmConditionCode.CC,
    5: ArmConditionCode.MI,
    6: ArmConditionCode.PL,
    7: ArmConditionCode.VS,
    8: ArmConditionCode.VC,
    9: ArmConditionCode.HI,
    10: ArmConditionCode.LS,
    11: ArmConditionCode.GE,
    12: ArmConditionCode.LT,
    13: ArmConditionCode.GT,
    14: ArmConditionCode.LE,
}


def _build_prefixes_arm(cs_insn: Any) -> list[InstructionPrefixView]:
    """Extract typed ARM prefixes from a Capstone instruction.

    ARM has no byte-level prefixes; the per-instruction modifiers Capstone
    surfaces are condition-code (``cs_insn.cc``), update-flags S-suffix
    (``cs_insn.update_flags``), and writeback ! suffix
    (``cs_insn.writeback``). Capstone's CapstoneInsn proxy exposes these
    via ``__getattr__`` straight from the underlying ``cs_insn``.
    """
    out: list[InstructionPrefixView] = []
    cc_id = int(getattr(cs_insn, "cc", 0) or 0)
    cc_enum = _ARM_CC_TO_ENUM.get(cc_id)
    if cc_enum is not None:
        out.append(_ConditionCodePrefix(cc=cc_enum))
    if bool(getattr(cs_insn, "update_flags", False)):
        out.append(UpdateFlagsPrefixView())
    if bool(getattr(cs_insn, "writeback", False)):
        out.append(WritebackPrefixView())
    return out


def _build_prefixes_ppc(cs_insn: Any) -> list[InstructionPrefixView]:
    """Extract typed PPC prefixes from a Capstone instruction.

    PPC's per-instruction modifiers are the branch-condition field
    (``cs_insn.bc``, integer) and the CR0-update Rc bit
    (``cs_insn.update_cr0``).
    """
    out: list[InstructionPrefixView] = []
    bc = int(getattr(cs_insn, "bc", 0) or 0)
    if bc != 0:
        out.append(_PpcBranchConditionPrefix(bc=bc))
    if bool(getattr(cs_insn, "update_cr0", False)):
        out.append(PpcUpdateCr0PrefixView())
    return out


# Dispatch table: ``arch -> per-arch builder`` keeps ``_build_prefixes``
# free of if-ladders. ARM32 and AARCH64 share the ARM extractor (Capstone's
# ARM and ARM64 operands carry the same ``cc`` / ``update_flags`` /
# ``writeback`` fields). MIPS / RISC-V have no per-instruction prefix
# signal; they fall through to the empty-list default.
_PREFIX_BUILDERS: dict[Architecture, Any] = {
    Architecture.X86: _build_prefixes_x86,
    Architecture.ARM32: _build_prefixes_arm,
    Architecture.AARCH64: _build_prefixes_arm,
    Architecture.PPC: _build_prefixes_ppc,
}


def _build_prefixes(cs_insn: Any, arch: Architecture) -> list[InstructionPrefixView]:
    """Single dispatcher onto per-platform prefix builders.

    Returns the empty list for architectures whose Capstone bindings
    surface no per-instruction prefix signal (MIPS, RISC-V, UNKNOWN).
    """
    if cs_insn is None:
        return []
    builder = _PREFIX_BUILDERS.get(arch)
    if builder is None:
        return []
    return builder(cs_insn)


# Re-exports of the angr-side MetadataLookup. The lookup class itself is
# defined in ``tokenizer/address_meta_data_lookup.py`` (its original home)
# to avoid a hard import cycle with this provider file. The names are
# re-exported here so callers (and the task-validation step) can import
# both lookup + view via the provider module.
def _import_lookup_classes() -> tuple[type, type]:
    """Lazy import of ``AngrMetadataLookup`` / ``AddressMetaDataLookup``.

    Deferred to a function so the module-load cycle
    (``angr_provider`` <-> ``address_meta_data_lookup``) is broken.
    """
    from tokenizer.address_meta_data_lookup import AddressMetaDataLookup, AngrMetadataLookup
    return AngrMetadataLookup, AddressMetaDataLookup


def __getattr__(name: str):
    """Module-level ``__getattr__`` for re-export laziness.

    ``from tokenizer.disasm.angr_provider import AngrMetadataLookup``
    triggers this on first access; the import resolves at that point so
    we don't pay the ``address_meta_data_lookup`` cost at module load.
    """
    if name in {"AngrMetadataLookup", "AddressMetaDataLookup"}:
        AngrMetadataLookup, AddressMetaDataLookup = _import_lookup_classes()
        return {"AngrMetadataLookup": AngrMetadataLookup, "AddressMetaDataLookup": AddressMetaDataLookup}[name]
    raise AttributeError(f"module 'tokenizer.disasm.angr_provider' has no attribute {name!r}")


# ===========================================================================
# LEGACY COMPAT PROXY -- REMOVE IN PHASE H.1
# ===========================================================================
# Bridges the OWNED ``*View`` classes back to the legacy ``_Cap*`` shape that
# pre-G.3 consumers in ``tokenizer/arch/*`` and
# ``tokenizer/fill_constant_candidates.py`` still read against. Each proxy
# instance wraps a STASH-SAFE owned-view snapshot (see the lifecycle
# docstring at the top of ``tokenizer/disasm/types.py`` -- ``copy.deepcopy``
# binds a fresh wrapper to the same backing object), so iterating over
# ``func.blocks`` then ``block.capstone.insns`` then ``insn.operands``
# produces independent objects per iteration step the way consumers expect.
#
# Slated for removal once G.3 migrates ``tokenizer/arch/*`` +
# ``fill_constant_candidates.py`` to read the owned views directly.
# Auditable scope: every legacy attribute the audit
# ``grep -rn 'block\.capstone\.insns | insn\.insn\.insn_name | op\.type ==
#   | op\.mem\. | op\.reg | insn\.reg_name | \.prefix\b | \.cc\b
#   | \.update_flags | \.writeback | \.crx' tokenizer/arch
#   tokenizer/fill_constant_candidates.py``
# surfaces is bridged here; nothing more.


class _LegacyCompatMem:
    """Legacy ``op.mem`` shape -- direct integer reads off the owned
    ``MemoryOperandView``. Registers are exposed as raw ids (``RegisterView.id``)
    to match the legacy ``op.mem.base = 0``-means-absent convention.
    """

    __slots__ = ("base", "index", "scale", "disp", "segment")

    def __init__(self, mem_view: MemoryOperandView) -> None:
        self.base: int = mem_view.base.id
        self.index: int = mem_view.index.id
        self.scale: int = mem_view.scale
        self.disp: int = mem_view.disp
        self.segment: int = mem_view.segment.id


class _LegacyCompatShift:
    """Legacy ``op.shift`` shape -- carries Capstone's raw ``type`` integer
    (NOT the typed ``ShiftKind`` enum). Pre-G.3 consumer code in
    ``tokenizer/arch/arm32/operands.py`` indexes the integer through its
    own ``_ARM_SFT_NAMES`` table.

    The owned ``ShiftModifierView`` has already mapped Capstone's encoding
    onto ``ShiftKind``; to preserve the legacy expectation we reach back to
    the Capstone-native ``type`` integer via the deep-copied operand handle
    that the parent ``_LegacyCompatOperand`` carries.
    """

    __slots__ = ("type", "value")

    def __init__(self, type_: int, value: int) -> None:
        self.type: int = type_
        self.value: int = value


class _LegacyCompatCrx:
    """Legacy ``op.crx`` shape -- exposes ``reg`` as a raw integer."""

    __slots__ = ("reg",)

    def __init__(self, reg_id: int) -> None:
        self.reg: int = reg_id


class _LegacyCompatOperand:
    """Legacy ``_CapOperand`` shape, populated from an owned ``OperandView``.

    Materialised eagerly per operand so the consumer can hold the reference
    across iteration steps the way the pre-G.3 ``_CapOperand`` (a dataclass
    instance per operand) allowed.
    """

    __slots__ = ("type", "reg", "imm", "mem", "size", "shift", "crx", "fp_type")

    def __init__(self, op_view: OperandView, raw_op: Any) -> None:
        self.type: int = op_view.type_int
        self.reg: int = op_view.reg.id
        self.imm: int = op_view.imm
        self.mem: _LegacyCompatMem = _LegacyCompatMem(op_view.mem)
        self.size: int = op_view.size
        # Capstone shift type is the raw int the legacy consumers index
        # through (`tokenizer/arch/arm32/operands.py::_ARM_SFT_NAMES`); we
        # read it off the underlying Capstone op directly.
        raw_shift = getattr(raw_op, "shift", None) if raw_op is not None else None
        if raw_shift is None:
            self.shift = _LegacyCompatShift(0, 0)
        else:
            self.shift = _LegacyCompatShift(int(raw_shift.type), int(raw_shift.value))
        self.crx: _LegacyCompatCrx = _LegacyCompatCrx(op_view.crx.reg.id)
        self.fp_type: Optional[FpType] = op_view.fp_type


class _LegacyCompatInsnInner:
    """Stands in for ``insn.insn`` in the legacy shape.

    Pre-G.3 consumers in ``tokenizer/arch/{arm32,ppc,riscv,mips,x86}``
    access ``insn.insn.insn_name()`` for the base mnemonic and the ARM
    (``cc`` / ``update_flags`` / ``writeback``) and PPC (``bc`` /
    ``update_cr0``) modifier fields. We reconstruct each modifier by
    walking the typed ``PrefixesView`` once at construction time so the
    legacy attribute read is a flat integer / bool lookup.
    """

    __slots__ = ("_insn_name", "cc", "update_flags", "writeback", "bc", "update_cr0")

    def __init__(self, insn_view: InstructionView, raw_cs_insn: Any) -> None:
        self._insn_name: str = insn_view.base_mnemonic
        # Pull the raw integer ARM cc / PPC bc / bool flags directly off
        # the underlying ``cs_insn`` so the ARM "AL = 15" sentinel and any
        # other Capstone-native value the legacy consumer compares against
        # are preserved verbatim. ``getattr`` is safe -- the field is
        # absent on ISAs where it does not apply.
        self.cc: int = int(getattr(raw_cs_insn, "cc", 0) or 0) if raw_cs_insn is not None else 0
        self.update_flags: bool = bool(getattr(raw_cs_insn, "update_flags", False)) if raw_cs_insn is not None else False
        self.writeback: bool = bool(getattr(raw_cs_insn, "writeback", False)) if raw_cs_insn is not None else False
        self.bc: int = int(getattr(raw_cs_insn, "bc", 0) or 0) if raw_cs_insn is not None else 0
        self.update_cr0: bool = bool(getattr(raw_cs_insn, "update_cr0", False)) if raw_cs_insn is not None else False

    def insn_name(self) -> str:
        return self._insn_name


class _LegacyCompatInstruction:
    """Legacy ``_CapInstruction`` shape, populated from an owned
    ``InstructionView``.

    Carries an eagerly materialised ``operands`` list and an ``insn``
    inner-object that bundles the ARM/PPC modifier fields. ``prefix`` is
    the underlying Capstone ``cs_insn.prefix`` 4-byte array verbatim --
    pre-G.3 x86 consumers iterate it directly to match prefix opcode
    bytes against ``instr_sets.prefixes``. ``reg_name(reg_id)`` proxies
    to Capstone's per-instruction reg-name table on the same ``cs_insn``.
    """

    __slots__ = ("mnemonic", "op_str", "operands", "prefix", "insn", "_cs_insn")

    def __init__(self, insn_view: InstructionView) -> None:
        # Pull the raw Capstone instruction off the view so the legacy
        # ``reg_name`` and ``prefix`` reads bypass the owned-view layer
        # (we cannot expose Capstone-native reg-name lookup from the
        # owned ``RegisterView`` -- the consumer still passes raw int reg
        # ids around, and resolves the name per-call). This dependency
        # vanishes with the legacy proxy in Phase H.1.
        cs_insn = getattr(insn_view, "_cs_insn", None)
        self._cs_insn = cs_insn
        self.mnemonic: str = insn_view.mnemonic
        self.op_str: str = insn_view.op_str
        self.operands: list[_LegacyCompatOperand] = self._materialise_operands(insn_view, cs_insn)
        # Raw 4-byte prefix array from Capstone -- consumers iterate it
        # as ``for byte in insn.prefix``. ``b""`` for ISAs without
        # prefix bytes (ARM / PPC / MIPS / RISC-V).
        self.prefix = getattr(cs_insn, "prefix", b"") if cs_insn is not None else b""
        self.insn: _LegacyCompatInsnInner = _LegacyCompatInsnInner(insn_view, cs_insn)

    @staticmethod
    def _materialise_operands(insn_view: InstructionView, cs_insn: Any) -> list[_LegacyCompatOperand]:
        # The owned ``OperandsView`` reuses one cursor across iteration --
        # we have to materialise per element to match the legacy semantics
        # (operand list is a real list of independent dataclasses).
        out: list[_LegacyCompatOperand] = []
        raw_ops = getattr(cs_insn, "operands", None) or () if cs_insn is not None else ()
        ops_iter = iter(insn_view.operands)
        for raw_op in raw_ops:
            try:
                op_view = next(ops_iter)
            except StopIteration:
                break
            out.append(_LegacyCompatOperand(op_view, raw_op))
        return out

    def reg_name(self, reg_id: int) -> str:
        if self._cs_insn is None:
            return ""
        return self._cs_insn.reg_name(reg_id) or ""


class _LegacyCompatBlock:
    """Legacy ``_CapBlock`` shape -- ``addr``, ``size``, and
    ``capstone.insns`` (a list of ``_LegacyCompatInstruction``).

    ``capstone`` mirrors the angr ``Block.capstone`` -> ``CapstoneBlock``
    indirection the legacy ``_CapBlock`` already aped via its inner
    ``_CapstoneHolder``.
    """

    class _CapstoneHolder:
        __slots__ = ("insns",)

        def __init__(self, insns: list[_LegacyCompatInstruction]) -> None:
            self.insns = insns

    __slots__ = ("addr", "size", "capstone")

    def __init__(self, block_view: BlockView) -> None:
        self.addr: int = block_view.addr
        self.size: int = block_view.size
        # Materialise every instruction eagerly: the OWNED ``InstructionsView``
        # reuses one cursor, so we deep-copy each cursor mid-iteration to
        # bind the legacy wrapper to a stable backing ``cs_insn``.
        insns: list[_LegacyCompatInstruction] = []
        for insn_view in block_view.instructions:
            insns.append(_LegacyCompatInstruction(copy.deepcopy(insn_view)))
        self.capstone = self._CapstoneHolder(insns)


class _LegacyCompatFunction:
    """Legacy ``_CapFunction`` shape -- ``blocks`` is an eagerly-materialised
    list of ``_LegacyCompatBlock``.

    ``fill_constant_candidates`` calls ``list(func.blocks)`` then iterates
    multiple times; the legacy contract requires per-block independent
    objects, which we secure by deep-copying each reused ``BlockView``
    cursor as iteration advances.
    """

    __slots__ = ("blocks",)

    def __init__(self, func_view: FunctionView) -> None:
        blocks: list[_LegacyCompatBlock] = []
        for block_view in func_view.blocks:
            blocks.append(_LegacyCompatBlock(copy.deepcopy(block_view)))
        self.blocks = blocks


# ===========================================================================
# END LEGACY COMPAT PROXY
# ===========================================================================


__all__ = [  # noqa: F822 - "AngrMetadataLookup" / "AddressMetaDataLookup" resolved by __getattr__
    "AddressMetaDataLookup",
    "AngrDisassemblyProvider",
    "AngrMetadataLookup",
    "_AngrAddressMetadataView",
    "_LegacyCompatBlock",
    "_LegacyCompatFunction",
    "_LegacyCompatInstruction",
]


class AngrDisassemblyProvider(DisassemblyProvider):
    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path
        self.project: angr.Project = angr.Project(binary_path, auto_load_libs=False)
        self.cfg: angr.analyses.cfg.cfg_fast.CFGFast | None = None
        self._arch: Architecture = _resolve_architecture(self.project.arch)
        # Single reusable owned ``FunctionView`` cursor mutated by every
        # ``iter_functions`` step. Per the lifecycle docstring at the top
        # of ``tokenizer/disasm/types.py`` the wrapper is REUSED across
        # iteration; consumers that need to stash a snapshot must
        # ``copy.deepcopy(view)``.
        self._function_cursor: _AngrFunctionView = _AngrFunctionView(self._arch)

    def build_cfg(self) -> None:
        self.cfg = self.project.analyses.CFGFast(normalize=True)

    def get_text_section_bounds(self) -> tuple[int, int]:
        for section in self.project.loader.main_object.sections:
            if section.name == ".text":
                return section.vaddr, section.vaddr + section.memsize
        return 0, 0

    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]:
        if sections is None:
            sections = [".rodata"]

        all_entries = []
        addr_dict: dict[str, list[str]] = {}

        for sec in self.project.loader.main_object.sections:
            if sec.name not in sections:
                continue
            if sec.name == ".rodata" and sec.is_readable and sec.memsize > 0:
                data = self.project.loader.memory.load(sec.vaddr, sec.memsize)
                for match in re.finditer(b"[\x20-\x7e]{4,}\x00", data):
                    s = match.group().rstrip(b"\x00").decode("utf-8", errors="ignore")
                    start = sec.vaddr + match.start()
                    entry = {
                        "section": ".rodata",
                        "start": hex(start),
                        "end": hex(start + len(s) + 1),
                        "value": f'"{s}"',
                    }
                    all_entries.append(entry)
                    addr_dict[entry["start"]] = [entry["end"], entry["section"], entry["value"]]

        if output_csv_path:
            csv_path = Path(output_csv_path)
            consts_path = csv_path.parent / f"{csv_path.stem.replace('_output', '')}_consts.txt"
        else:
            consts_path = Path("parsed_constants.txt")

        with open(consts_path, "w") as f:
            for e in all_entries:
                f.write(f"{e['start']} - {e['end']}: {e['section']}: {e['value']}\n")

        print(f"Parsed {len(all_entries)} .rodata constants with exact addresses into {consts_path}")
        return addr_dict

    def create_metadata_lookup(self) -> MetadataLookup:
        # Lazy import keeps the ``angr_provider`` <-> ``address_meta_data_lookup``
        # circular-import broken at module load (see ``_import_lookup_classes``).
        _, AddressMetaDataLookupCls = _import_lookup_classes()
        return AddressMetaDataLookupCls(self.binary_path)

    def function_count(self) -> int:
        assert self.cfg is not None, "CFG not built yet — call build_cfg() first"
        return len(self.cfg.functions)

    def iter_functions(self) -> Iterable[tuple[int, str, FunctionView]]:
        """Yield ``(addr, name, function_view)`` per surviving CFG function.

        The third tuple element is the SAME reused ``_AngrFunctionView``
        cursor mutated to point at each function in turn (see lifecycle
        docstring at the top of ``tokenizer/disasm/types.py``). Consumers
        that need to hold a function reference across an advance must
        ``copy.deepcopy(view)`` the snapshot.

        Pre-G.3 consumers in ``tokenizer/arch/*`` and
        ``tokenizer/fill_constant_candidates.py`` still read the legacy
        ``_Cap*`` shape; they wrap the yielded view through
        ``_LegacyCompatFunction(view)`` (slated for removal in Phase H.1
        once G.3 migrates them to the owned-view interface).
        """
        assert self.cfg is not None, "CFG not built yet — call build_cfg() first"
        for func_addr, func in sorted(self.cfg.functions.items(), key=lambda item: item[1].name):
            func_name = func.name
            if func_name in ("UnresolvableCallTarget", "UnresolvableJumpTarget"):
                continue
            self._function_cursor._set(func)
            yield func_addr, func_name, self._function_cursor
