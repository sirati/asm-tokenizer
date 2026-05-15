"""Owned disassembly domain types - lazy views with object reuse.

Both `DisassemblyProvider` implementations (Ghidra default, angr best-effort)
populate these view types. Consumer code (`tokenizer/arch/*`,
`tokenizer/constant_handler.py`, `tokenizer/fill_constant_candidates.py`)
reads ONLY these types - never provider-native objects.

LIFECYCLE - read carefully before consuming any view object
-----------------------------------------------------------
Every `*View` class in this module is a LAZY view. It does NOT hold its own
copy of the underlying data; properties compute on access by reading through
to the provider's native object.

The view instances are REUSED across iteration:
    - `provider.iter_functions()` yields the SAME `FunctionView` instance,
      mutated to point at each function in turn.
    - `function.blocks` returns a `BlocksView` whose `__iter__` yields the
      SAME `BlockView` instance per block.
    - `block.instructions`, `instruction.operands` behave identically with
      reused `InstructionView` / `OperandView`.
    - `op.mem`, `op.shift`, `op.crx` are sub-views bound to the parent
      operand; they implicitly invalidate when the parent operand advances.

The wrapper objects are VALID ONLY for the current iteration step. Holding a
reference to a view across an iteration advance yields stale data - by design,
to avoid GC pressure on the hot path.

To stash a view across iteration advances: `copy.deepcopy(view)`. This returns
a FRESH wrapper bound to the SAME underlying provider object (the provider's
native handle is stable and shareable - it is NOT duplicated). The fresh
wrapper has its own independent cursor. Inside the fresh wrapper, child
iteration (`copy.blocks`, `copy.instructions`, ...) is STILL lazy and
reusable - only the wrapper level you deepcopy becomes stash-safe.

The `View` suffix in every class name is the visual reminder of this contract:
saving the reference does NOT preserve a snapshot of the data.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Iterator, Optional, Protocol, runtime_checkable


# ---- ENUMS ----
class OperandKind(IntEnum):
    INVALID = 0
    REG = 1
    IMM = 2
    MEM = 3
    CRX = 4         # PPC condition-register field
    REG_LIST = 5    # ARM stm/ldm/push/pop/vpush/vpop/vstm/vldm register list
    OTHER = 99      # ARM FP/CIMM/PIMM/SETEND/SYSREG passthrough


class FpType(IntEnum):
    FLOAT16 = 16
    BFLOAT16 = -16  # distinct from Float16 at width=2 (BFloat16-mnemonic detection)
    FLOAT32 = 32
    FLOAT64 = 64
    FLOAT80 = 80
    FLOAT128 = 128


class ShiftKind(IntEnum):
    NONE = 0
    LSL = 1
    LSR = 2
    ASR = 3
    ROR = 4
    RRX = 5


class Architecture(IntEnum):
    UNKNOWN = 0
    X86 = 1        # covers x86-32 + x86-64
    ARM32 = 2
    AARCH64 = 3
    MIPS = 4
    PPC = 5
    RISCV = 6


class ArmConditionCode(IntEnum):
    EQ = 1
    NE = 2
    CS = 3
    CC = 4
    MI = 5
    PL = 6
    VS = 7
    VC = 8
    HI = 9
    LS = 10
    GE = 11
    LT = 12
    GT = 13
    LE = 14


class X86Segment(IntEnum):
    CS = 0
    SS = 1
    DS = 2
    ES = 3
    FS = 4
    GS = 5


class X86BranchHint(IntEnum):
    TAKEN = 0
    NOT_TAKEN = 1


# ---- REGISTER (rich type - replaces raw int register IDs) ----
@runtime_checkable
class RegisterView(Protocol):
    @property
    def name(self) -> str: ...        # canonical asm form, lowercase

    @property
    def id(self) -> int: ...          # provider-internal id; not consumed by arch walkers

    @property
    def arch(self) -> Architecture: ...

    @property
    def is_absent(self) -> bool: ...  # True iff operand slot is empty

    def __deepcopy__(self, memo) -> "RegisterView": ...


# ---- SUB-VIEWS (bound to parent OperandView) ----
@runtime_checkable
class MemoryOperandView(Protocol):
    """Decomposed memory-operand view.

    Carries the (base, index, scale, disp, segment) classical addressing
    fields plus the ARM-specific addressing-mode flags
    (``writeback``/``pre_indexed``/``post_indexed``) surfaced from the
    provider's representation list.

    Writeback semantics (ARM/AArch64 only):
      ``writeback`` is True iff the base register is auto-updated by the
      displacement as part of the memory access; the asm marker is the
      trailing ``!`` in ``[base, #imm]!``.
      ``pre_indexed`` is True iff the base is updated BEFORE the memory
      access (this implies ``writeback`` is also True).
      ``post_indexed`` is True iff the base is updated AFTER the memory
      access (writeback is implicit; the asm form is ``[base], #imm``).
      Plain offset-only addressing (``[base, #imm]``) leaves all three
      flags False; the three flags are mutually exclusive otherwise.
      Non-ARM ISAs always report all three as False.
    """

    @property
    def base(self) -> RegisterView: ...

    @property
    def index(self) -> RegisterView: ...

    @property
    def scale(self) -> int: ...

    @property
    def disp(self) -> int: ...        # signed

    @property
    def segment(self) -> RegisterView: ...  # x86 segment override; is_absent on others

    @property
    def writeback(self) -> bool: ...        # ARM `!`; base auto-updated by disp

    @property
    def pre_indexed(self) -> bool: ...      # ARM [base, #imm]! (base updated BEFORE access)

    @property
    def post_indexed(self) -> bool: ...     # ARM [base], #imm (base updated AFTER access)

    @property
    def index_shift(self) -> "ShiftModifierView": ...
    # ARM [base, index, lsl #N] shift on the index register. ``kind ==
    # ShiftKind.NONE`` on every non-shifted-index addressing mode and
    # on non-ARM ISAs.

    def __deepcopy__(self, memo) -> "MemoryOperandView": ...


@runtime_checkable
class ShiftModifierView(Protocol):
    @property
    def kind(self) -> ShiftKind: ...

    @property
    def amount(self) -> int: ...

    def __deepcopy__(self, memo) -> "ShiftModifierView": ...


@runtime_checkable
class CrxFieldView(Protocol):
    @property
    def reg(self) -> RegisterView: ...

    def __deepcopy__(self, memo) -> "CrxFieldView": ...


@runtime_checkable
class RegisterListView(Protocol):
    """Sub-view bound to the parent OperandView for ARM register-list operands.

    Modelled after the real ARM asm shape ``stmdb sp!, {r4, lr}``:
    ``base`` is the writeback target (the register *outside* the braces; e.g.
    ``sp`` in ``stmdb``), ``writeback`` is the ``!`` flag, and iterating the
    view yields one ``RegisterView`` per list member (the registers *inside*
    the braces). ``base.is_absent`` when the encoding does not carry a
    separate base slot (e.g. Ghidra may report a standalone reg-list operand
    where the base is a separate sibling operand).

    Like the other sub-views, this object is REUSABLE - bound to its parent
    ``OperandView``'s cursor; member-register sub-views are also reused
    across ``__iter__`` (see lifecycle docstring at top of this module).
    ``deepcopy`` snapshots the current member spec into a fresh wrapper.
    """

    @property
    def base(self) -> RegisterView: ...   # writeback target; is_absent if absent

    @property
    def writeback(self) -> bool: ...      # the `!` flag

    def __len__(self) -> int: ...         # list-member count (excluding base)

    def __iter__(self) -> Iterator[RegisterView]: ...  # iterates members

    def __getitem__(self, idx: int) -> RegisterView: ...  # small finite count

    def __deepcopy__(self, memo) -> "RegisterListView": ...


# ---- INSTRUCTION PREFIXES (general concept; typed list per instruction) ----
class InstructionPrefixView(ABC):
    """Base for typed instruction prefixes. Count per instruction is small
    (0-3 typical); fresh-per-iteration allocation acceptable here unlike the
    operand hot path."""
    pass


class LockPrefixView(InstructionPrefixView):
    pass


class RepPrefixView(InstructionPrefixView):
    @property
    @abstractmethod
    def repeat_until_zero(self) -> bool: ...  # True=REPE, False=REPNE


class SegmentOverridePrefixView(InstructionPrefixView):
    @property
    @abstractmethod
    def segment(self) -> X86Segment: ...


class OperandSizePrefixView(InstructionPrefixView):
    pass


class AddressSizePrefixView(InstructionPrefixView):
    pass


class BranchHintPrefixView(InstructionPrefixView):
    @property
    @abstractmethod
    def hint(self) -> X86BranchHint: ...


class ConditionCodePrefixView(InstructionPrefixView):
    @property
    @abstractmethod
    def cc(self) -> ArmConditionCode: ...


class UpdateFlagsPrefixView(InstructionPrefixView):
    pass


class WritebackPrefixView(InstructionPrefixView):
    pass


class PpcBranchConditionPrefixView(InstructionPrefixView):
    @property
    @abstractmethod
    def bc(self) -> int: ...


class PpcUpdateCr0PrefixView(InstructionPrefixView):
    pass


# ---- OPERAND ----
@runtime_checkable
class OperandView(Protocol):
    @property
    def kind(self) -> OperandKind: ...

    @property
    def reg(self) -> RegisterView: ...        # rich type

    @property
    def imm(self) -> int: ...                 # signed number

    @property
    def mem(self) -> MemoryOperandView: ...

    @property
    def crx(self) -> CrxFieldView: ...

    @property
    def shift(self) -> ShiftModifierView: ...

    @property
    def reg_list(self) -> RegisterListView: ...  # valid when kind == OperandKind.REG_LIST

    @property
    def size(self) -> int: ...                # bytes

    @property
    def fp_type(self) -> Optional[FpType]: ...

    @property
    def type_int(self) -> int: ...            # passthrough for OperandKind.OTHER

    def __deepcopy__(self, memo) -> "OperandView": ...


@runtime_checkable
class OperandsView(Protocol):
    """Container view over an instruction's operands. NOT a list. `__iter__`
    yields the same reused OperandView; `__getitem__` is intentionally absent
    (adjacent reads would invalidate each other under the reuse contract)."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[OperandView]: ...


@runtime_checkable
class PrefixesView(Protocol):
    """Container view over an instruction's prefixes. `__getitem__` IS
    supported (prefix instances are typed-distinct, not a single reused view).
    Prefix count per instruction is small."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[InstructionPrefixView]: ...

    def __getitem__(self, idx: int) -> InstructionPrefixView: ...


# ---- INSTRUCTION ----
@runtime_checkable
class InstructionView(Protocol):
    @property
    def address(self) -> int: ...

    @property
    def mnemonic(self) -> str: ...        # WITH x86 prefix word ("repe cmpsb")

    @property
    def base_mnemonic(self) -> str: ...   # WITHOUT prefix ("cmpsb")

    @property
    def op_str(self) -> str: ...

    @property
    def operands(self) -> OperandsView: ...

    @property
    def prefixes(self) -> PrefixesView: ...

    def __deepcopy__(self, memo) -> "InstructionView": ...


@runtime_checkable
class InstructionsView(Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[InstructionView]: ...


# ---- BLOCK ----
@runtime_checkable
class BlockView(Protocol):
    @property
    def addr(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def instructions(self) -> InstructionsView: ...

    def __deepcopy__(self, memo) -> "BlockView": ...


@runtime_checkable
class BlocksView(Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[BlockView]: ...


# ---- FUNCTION ----
@runtime_checkable
class FunctionView(Protocol):
    @property
    def entry(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def blocks(self) -> BlocksView: ...

    def __deepcopy__(self, memo) -> "FunctionView": ...


# ---- JUMP TABLE ----
@runtime_checkable
class JumpTableView(Protocol):
    @property
    def base_addr(self) -> int: ...

    @property
    def targets(self) -> Iterator[int]: ...

    def __deepcopy__(self, memo) -> "JumpTableView": ...
