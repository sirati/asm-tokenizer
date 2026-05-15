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
    Architecture,
    BlockView,
    BlocksView,
    CrxFieldView,
    FpType,
    FunctionView,
    InstructionView,
    InstructionsView,
    MemoryOperandView,
    OperandKind,
    OperandView,
    OperandsView,
    PrefixesView,
    RegisterView,
    ShiftKind,
    ShiftModifierView,
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


# Forward declaration: ``_build_prefixes`` is defined in commit G.2.b. Until
# that commit lands, prefix construction returns the empty list.
def _build_prefixes(cs_insn: Any, arch: Architecture) -> list:
    return []


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


__all__ = [  # noqa: F822 - "AngrMetadataLookup" / "AddressMetaDataLookup" resolved by __getattr__
    "AddressMetaDataLookup",
    "AngrDisassemblyProvider",
    "AngrMetadataLookup",
    "_AngrAddressMetadataView",
]


class AngrDisassemblyProvider(DisassemblyProvider):
    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path
        self.project: angr.Project = angr.Project(binary_path, auto_load_libs=False)
        self.cfg: angr.analyses.cfg.cfg_fast.CFGFast | None = None

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

    def iter_functions(self) -> Iterable[tuple[int, str, Any]]:
        assert self.cfg is not None, "CFG not built yet — call build_cfg() first"
        for func_addr, func in sorted(self.cfg.functions.items(), key=lambda item: item[1].name):
            func_name = func.name
            if func_name in ("UnresolvableCallTarget", "UnresolvableJumpTarget"):
                continue
            yield func_addr, func_name, func
