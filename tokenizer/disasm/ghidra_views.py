"""Concrete owned-view implementations bound to Ghidra Java handles.

These classes implement the Protocols declared in ``tokenizer/disasm/types.py``
(``FunctionView``, ``BlockView``, ``InstructionView``, ``OperandView``,
``MemoryOperandView``, ``ShiftModifierView``, ``CrxFieldView``,
``RegisterView``, container views, typed prefix subclasses).

LIFECYCLE - reuse + lazy
------------------------
Every wrapper here is REUSABLE. ``_advance(...)`` repoints the wrapper at a
new Ghidra Java backing handle and resets per-cursor caches. Container views
(``_GhidraBlocksView`` / ``_GhidraInstructionsView`` / ``_GhidraOperandsView``)
are NOT lists; they iterate via the parent's reusable child wrapper. Sub-views
(``mem`` / ``shift`` / ``crx`` on ``OperandView``) are bound to the parent's
cursor and read through to its current backing.

``__deepcopy__`` returns a fresh wrapper bound to the SAME Ghidra handle. The
Ghidra Java handles are stable references managed by the JVM; deepcopying the
wrapper does NOT duplicate them. The fresh wrapper has its own independent
cursor and is safe to stash across iteration advances; child iteration inside
the deepcopied wrapper remains lazy and reusable.

The decomposition / prefix-builder helpers used by these views live in
``tokenizer.disasm.ghidra_provider`` (G.1.b).

These classes are constructed with PROVIDER-LEVEL state (the ``_RegisterMap``
the provider built once for the program, and the program handle for ISA
detection). They do not hold any per-iteration Java references at construction
time; ``_advance`` provides those.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator, Optional

from tokenizer.disasm.types import (
    AddressSizePrefixView,
    Architecture,
    ArmConditionCode,
    BlocksView,
    BranchHintPrefixView,
    ConditionCodePrefixView,
    CrxFieldView,
    FpType,
    InstructionPrefixView,
    InstructionsView,
    JumpTableView,
    LockPrefixView,
    MemoryOperandView,
    OperandKind,
    OperandSizePrefixView,
    OperandView,
    OperandsView,
    PpcBranchConditionPrefixView,
    PpcUpdateCr0PrefixView,
    PrefixesView,
    RegisterListView,
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
# Sentinel: REGISTER ABSENT
# ---------------------------------------------------------------------------
# Many sub-views (mem.base / mem.index / mem.segment / crx.reg) are slots
# that may be EMPTY for a given operand. Rather than returning ``None``
# (which would force every consumer into ``getattr(...) or None`` shapes),
# we return a sentinel ``_AbsentRegisterView`` whose ``is_absent == True``.
# Per the RegisterView protocol contract.

_REG_ID_ABSENT = 0
_REG_NAME_ABSENT = ""


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
class _GhidraRegisterView:
    """Reusable register wrapper.

    The wrapper holds the current register name + provider-internal id; the
    provider's ``_RegisterMap`` translates between Ghidra register names
    (``Register.getName()``) and provider-internal ids. Architecture is
    set once at construction and stable.

    A single ``_GhidraRegisterView`` is REUSED across operand iteration -
    the operand wrapper resets it per-operand. Stashing requires
    ``copy.deepcopy(reg_view)`` (returns a fresh wrapper with the current
    name/id snapshot).
    """

    __slots__ = ("_name", "_id", "_arch")

    def __init__(self, arch: Architecture) -> None:
        self._name: str = _REG_NAME_ABSENT
        self._id: int = _REG_ID_ABSENT
        self._arch: Architecture = arch

    def _advance(self, name: str, reg_id: int) -> None:
        """Repoint at ``(name, reg_id)``. ``name`` should be normalized
        lowercase by the caller; ``reg_id`` is the provider-internal id.
        Empty name + zero id mark the slot as absent."""
        self._name = name
        self._id = reg_id

    def _set_absent(self) -> None:
        self._name = _REG_NAME_ABSENT
        self._id = _REG_ID_ABSENT

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    @property
    def arch(self) -> Architecture:
        return self._arch

    @property
    def is_absent(self) -> bool:
        return self._id == _REG_ID_ABSENT and self._name == _REG_NAME_ABSENT

    def __deepcopy__(self, memo) -> "_GhidraRegisterView":
        clone = _GhidraRegisterView(self._arch)
        clone._name = self._name
        clone._id = self._id
        return clone


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
    )

    def __init__(self, arch: Architecture) -> None:
        self._arch: Architecture = arch
        self._base = _GhidraRegisterView(arch)
        self._index = _GhidraRegisterView(arch)
        self._segment = _GhidraRegisterView(arch)
        self._scale: int = 1
        self._disp: int = 0

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

    def __deepcopy__(self, memo) -> "_GhidraMemoryOperandView":
        clone = _GhidraMemoryOperandView(self._arch)
        clone._base = copy.deepcopy(self._base, memo)
        clone._index = copy.deepcopy(self._index, memo)
        clone._segment = copy.deepcopy(self._segment, memo)
        clone._scale = self._scale
        clone._disp = self._disp
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
# Register-list sub-view (ARM stm/ldm-family)
# ---------------------------------------------------------------------------
class _GhidraRegisterListView:
    """Sub-view bound to the parent operand's cursor for ARM reg-list operands.

    Holds the decomposed (base, writeback, [members]) for the current
    reg-list operand. The parent ``_GhidraOperandView`` populates this
    once on first ``op.reg_list`` access (via the provider's reg-list
    decomposition callback) and re-populates whenever the parent
    advances.

    Member registers are exposed as reusable ``_GhidraRegisterView``
    cursors. ``__iter__`` mutates ``_active_member`` to point at
    consecutive members and yields the SAME cursor instance per member;
    ``__getitem__`` returns a per-slot wrapper (small finite count, the
    member-view pool grows lazily on demand).
    """

    __slots__ = (
        "_arch",
        "_reg_map",
        "_base_view",
        "_writeback",
        "_member_views",
        "_member_specs",
        "_active_member",
    )

    def __init__(self, arch: Architecture, reg_map: Any) -> None:
        self._arch: Architecture = arch
        self._reg_map = reg_map
        self._base_view = _GhidraRegisterView(arch)
        self._writeback: bool = False
        # Per-member reusable register cursors; grown lazily on demand.
        self._member_views: list[_GhidraRegisterView] = []
        # Snapshot of (name, id) tuples for the current cursor's members.
        # `__iter__` walks this list and repoints `_member_views[i]` per slot.
        self._member_specs: list[tuple[str, int]] = []
        self._active_member: int = -1

    def _advance(
        self,
        *,
        base_name: str,
        base_id: int,
        writeback: bool,
        member_specs: list[tuple[str, int]],
    ) -> None:
        """Repoint at the next reg-list operand.

        ``base_name``/``base_id`` describe the writeback target outside
        the braces (may be absent when Ghidra reports the base as a
        separate sibling operand); ``member_specs`` is the list of
        (name, id) tuples for each register inside the braces.
        """
        if base_id != _REG_ID_ABSENT or base_name != _REG_NAME_ABSENT:
            self._base_view._advance(base_name, base_id)
        else:
            self._base_view._set_absent()
        self._writeback = writeback
        self._member_specs = member_specs
        # Ensure we have enough reusable register cursors for this list.
        while len(self._member_views) < len(member_specs):
            self._member_views.append(_GhidraRegisterView(self._arch))
        self._active_member = -1

    @property
    def base(self) -> RegisterView:
        return self._base_view

    @property
    def writeback(self) -> bool:
        return self._writeback

    def __len__(self) -> int:
        return len(self._member_specs)

    def __iter__(self) -> Iterator[RegisterView]:
        for i, (name, rid) in enumerate(self._member_specs):
            self._active_member = i
            view = self._member_views[i]
            view._advance(name, rid)
            yield view

    def __getitem__(self, idx: int) -> RegisterView:
        if idx < 0:
            idx += len(self._member_specs)
        if not (0 <= idx < len(self._member_specs)):
            raise IndexError(idx)
        name, rid = self._member_specs[idx]
        # Member views are small/finite; reuse the slot's cursor.
        while len(self._member_views) <= idx:
            self._member_views.append(_GhidraRegisterView(self._arch))
        view = self._member_views[idx]
        view._advance(name, rid)
        return view

    def __deepcopy__(self, memo) -> "_GhidraRegisterListView":
        clone = _GhidraRegisterListView(self._arch, self._reg_map)
        clone._base_view = copy.deepcopy(self._base_view, memo)
        clone._writeback = self._writeback
        # Snapshot the member-spec list (tuples are immutable).
        clone._member_specs = list(self._member_specs)
        # Pre-allocate matching cursors so the clone's iteration works
        # without re-checking growth.
        clone._member_views = [_GhidraRegisterView(self._arch) for _ in clone._member_specs]
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


# ---------------------------------------------------------------------------
# Container views (NOT lists - iterate, reuse child wrapper)
# ---------------------------------------------------------------------------
class _GhidraOperandsView:
    """Container view over an instruction's operands.

    NOT a list. ``__iter__`` advances the parent ``_GhidraInstructionView``'s
    reusable ``_GhidraOperandView`` per operand and yields the same wrapper
    instance each time. ``__getitem__`` is intentionally absent (random
    access would invalidate adjacent reads).
    """

    __slots__ = ("_instruction",)

    def __init__(self, instruction: "_GhidraInstructionView") -> None:
        self._instruction = instruction

    def __len__(self) -> int:
        return self._instruction._operand_count

    def __iter__(self) -> Iterator[OperandView]:
        return self._instruction._iter_operands()


class _GhidraPrefixesView:
    """Container view over an instruction's prefixes.

    Per the protocol: prefix instances are typed-distinct and few
    (0-3 typical), so ``__getitem__`` is supported and the underlying
    storage is just a list. The instruction wrapper rebuilds this list
    when its cursor advances.
    """

    __slots__ = ("_prefixes",)

    def __init__(self) -> None:
        self._prefixes: list[InstructionPrefixView] = []

    def _populate(self, prefixes: list[InstructionPrefixView]) -> None:
        self._prefixes = prefixes

    def __len__(self) -> int:
        return len(self._prefixes)

    def __iter__(self) -> Iterator[InstructionPrefixView]:
        return iter(self._prefixes)

    def __getitem__(self, idx: int) -> InstructionPrefixView:
        return self._prefixes[idx]


class _GhidraInstructionsView:
    """Container view over a block's instructions.

    NOT a list. ``__iter__`` walks the block's Ghidra instruction iterator
    and yields the same reused ``_GhidraInstructionView`` per instruction.
    """

    __slots__ = ("_block",)

    def __init__(self, block: "_GhidraBlockView") -> None:
        self._block = block

    def __len__(self) -> int:
        return self._block._instruction_count

    def __iter__(self) -> Iterator["_GhidraInstructionView"]:
        return self._block._iter_instructions()


class _GhidraBlocksView:
    """Container view over a function's blocks.

    NOT a list. ``__iter__`` walks the function's Ghidra block iterator
    and yields the same reused ``_GhidraBlockView`` per block.
    """

    __slots__ = ("_function",)

    def __init__(self, function: "_GhidraFunctionView") -> None:
        self._function = function

    def __len__(self) -> int:
        return self._function._block_count

    def __iter__(self) -> Iterator["_GhidraBlockView"]:
        return self._function._iter_blocks()


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------
class _GhidraInstructionView:
    """Reusable instruction wrapper.

    Holds a single backing ``Ghidra Instruction`` reference at any time.
    ``_advance(ghidra_insn)`` repoints at the next instruction and resets
    per-cursor caches (mnemonic split, prefixes list, ISA, operand spec
    lazily computed on iter_operands).

    The wrapper depends on a ``decode_helper`` injected by the provider;
    that helper exposes:
      - ``split_mnemonic(raw)`` -> (base, suffix_prefix_name, suffix_prefix_byte)
      - ``alias_mnemonic(base)`` -> canonical base
      - ``architecture(program)`` -> Architecture (cached on the view)
      - ``compute_fp_type(insn, op_idx, arch, base_mnemonic)`` -> Optional[FpType]
      - ``build_prefixes(insn, arch)`` -> list[InstructionPrefixView]
      - ``decompose_x86_memory(insn, op_idx, reg_map)`` -> callback that
        populates a passed-in _GhidraMemoryOperandView
      - ``decompose_arm_memory(insn, op_idx, reg_map)`` -> callback
      - ``decompose_base_disp_memory(insn, op_idx, reg_map)`` -> callback
      - ``operand_spec(insn, op_idx, arch, base_mnemonic, reg_map)``
        -> dict ready to pass as kwargs to ``_GhidraOperandView._advance``
    """

    __slots__ = (
        "_arch",
        "_program",
        "_reg_map",
        "_decode",
        "_ghidra_insn",
        "_address",
        "_mnemonic",
        "_base_mnemonic",
        "_op_str",
        "_operand_count",
        "_prefixes",
        "_operand_view",
        "_operands_view",
    )

    def __init__(
        self,
        arch: Architecture,
        program: Any,
        reg_map: Any,
        decode: Any,
    ) -> None:
        self._arch: Architecture = arch
        self._program = program
        self._reg_map = reg_map
        self._decode = decode
        self._ghidra_insn: Optional[Any] = None
        self._address: int = 0
        self._mnemonic: str = ""
        self._base_mnemonic: str = ""
        self._op_str: str = ""
        self._operand_count: int = 0
        self._prefixes = _GhidraPrefixesView()
        self._operand_view = _GhidraOperandView(arch, reg_map)
        self._operands_view = _GhidraOperandsView(self)

    def _advance(self, ghidra_insn: Any) -> None:
        """Repoint at the next Ghidra Instruction.

        Resets per-cursor state (mnemonic split, op_str, operand count,
        prefixes list). Operand decomposition is deferred to ``__iter__``
        on the operands view.
        """
        self._ghidra_insn = ghidra_insn
        self._address = int(ghidra_insn.getAddress().getOffset())

        raw_mnemonic = str(ghidra_insn.getMnemonicString())
        base_mnemonic, suffix_prefix_name, _suffix_prefix_byte = self._decode.split_mnemonic(raw_mnemonic)
        base_mnemonic = self._decode.alias_mnemonic(base_mnemonic)
        self._base_mnemonic = base_mnemonic
        if suffix_prefix_name is not None:
            self._mnemonic = f"{suffix_prefix_name} {base_mnemonic}"
        else:
            self._mnemonic = base_mnemonic

        # op_str: comma-joined per-operand default representation
        try:
            num_ops = int(ghidra_insn.getNumOperands())
        except Exception:
            num_ops = 0
        self._operand_count = num_ops
        op_strs: list[str] = []
        for i in range(num_ops):
            try:
                op_strs.append(str(ghidra_insn.getDefaultOperandRepresentation(i)))
            except Exception:
                op_strs.append("")
        self._op_str = ", ".join(op_strs)

        # Prefixes (typed list) - rebuilt fresh per instruction; they are
        # typed-distinct, low-count instances so the small allocation is
        # acceptable per the protocol contract.
        prefixes = self._decode.build_prefixes(ghidra_insn, self._arch)
        self._prefixes._populate(prefixes)

    def _iter_operands(self) -> Iterator[OperandView]:
        """Yield the reusable ``_GhidraOperandView`` for each operand of
        the current instruction. The same wrapper instance is yielded
        each time, mutated to point at the next operand."""
        if self._ghidra_insn is None:
            return
        op_view = self._operand_view
        for i in range(self._operand_count):
            spec = self._decode.operand_spec(
                self._ghidra_insn,
                i,
                self._arch,
                self._base_mnemonic,
                self._reg_map,
            )
            op_view._advance(**spec)
            yield op_view

    @property
    def address(self) -> int:
        return self._address

    @property
    def mnemonic(self) -> str:
        return self._mnemonic

    @property
    def base_mnemonic(self) -> str:
        return self._base_mnemonic

    @property
    def op_str(self) -> str:
        return self._op_str

    @property
    def operands(self) -> OperandsView:
        return self._operands_view

    @property
    def prefixes(self) -> PrefixesView:
        return self._prefixes

    def __deepcopy__(self, memo) -> "_GhidraInstructionView":
        clone = _GhidraInstructionView(self._arch, self._program, self._reg_map, self._decode)
        clone._ghidra_insn = self._ghidra_insn
        clone._address = self._address
        clone._mnemonic = self._mnemonic
        clone._base_mnemonic = self._base_mnemonic
        clone._op_str = self._op_str
        clone._operand_count = self._operand_count
        # Snapshot the prefix list (each prefix instance is itself
        # immutable per typed protocol).
        clone._prefixes._populate(list(self._prefixes._prefixes))
        # The operand_view + operands_view are fresh empty wrappers;
        # iterating them re-decodes lazily against the snapshotted
        # ghidra_insn (still a stable Java handle).
        return clone


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class _GhidraBlockView:
    """Reusable block wrapper.

    Holds a single backing ``Ghidra CodeBlock`` reference. The block's
    iteration depends on the program's ``Listing`` object (provided by
    the parent function wrapper) and the function's body filter (so
    instructions outside the function body are skipped, matching the
    legacy ``_CapBlock`` shape).
    """

    __slots__ = (
        "_arch",
        "_program",
        "_listing",
        "_reg_map",
        "_decode",
        "_function",
        "_ghidra_block",
        "_addr",
        "_size",
        "_instruction_count",
        "_instruction_view",
        "_instructions_view",
    )

    def __init__(
        self,
        arch: Architecture,
        program: Any,
        listing: Any,
        reg_map: Any,
        decode: Any,
        function: "_GhidraFunctionView",
    ) -> None:
        self._arch = arch
        self._program = program
        self._listing = listing
        self._reg_map = reg_map
        self._decode = decode
        self._function = function
        self._ghidra_block: Optional[Any] = None
        self._addr: int = 0
        self._size: int = 0
        self._instruction_count: int = 0
        self._instruction_view = _GhidraInstructionView(arch, program, reg_map, decode)
        self._instructions_view = _GhidraInstructionsView(self)

    def _advance(self, ghidra_block: Any, instruction_count: int) -> None:
        """Repoint at the next CodeBlock.

        ``instruction_count`` is precomputed by the parent function
        wrapper at iter_blocks time (single Listing walk so __len__ on
        InstructionsView is O(1) without re-iterating).
        """
        self._ghidra_block = ghidra_block
        self._addr = int(ghidra_block.getMinAddress().getOffset())
        self._size = int(ghidra_block.getMaxAddress().getOffset()) - self._addr + 1
        self._instruction_count = instruction_count

    def _iter_instructions(self) -> Iterator["_GhidraInstructionView"]:
        if self._ghidra_block is None:
            return
        body = self._function._ghidra_function.getBody() if self._function._ghidra_function is not None else None
        insn_iter = self._listing.getInstructions(self._ghidra_block, True)
        view = self._instruction_view
        while insn_iter.hasNext():
            ghidra_insn = insn_iter.next()
            if body is not None and not body.contains(ghidra_insn.getAddress()):
                continue
            view._advance(ghidra_insn)
            yield view

    @property
    def addr(self) -> int:
        return self._addr

    @property
    def size(self) -> int:
        return self._size

    @property
    def instructions(self) -> InstructionsView:
        return self._instructions_view

    def __deepcopy__(self, memo) -> "_GhidraBlockView":
        clone = _GhidraBlockView(
            self._arch, self._program, self._listing, self._reg_map, self._decode, self._function
        )
        clone._ghidra_block = self._ghidra_block
        clone._addr = self._addr
        clone._size = self._size
        clone._instruction_count = self._instruction_count
        return clone


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------
class _GhidraFunctionView:
    """Reusable function wrapper.

    Holds a single backing ``Ghidra Function`` reference. The block model
    is provider-supplied (constructed once per provider, reused).
    """

    __slots__ = (
        "_arch",
        "_program",
        "_listing",
        "_reg_map",
        "_decode",
        "_block_model",
        "_monitor",
        "_ghidra_function",
        "_entry",
        "_name",
        "_block_count",
        "_block_view",
        "_blocks_view",
    )

    def __init__(
        self,
        arch: Architecture,
        program: Any,
        listing: Any,
        reg_map: Any,
        decode: Any,
        block_model: Any,
        monitor: Any,
    ) -> None:
        self._arch = arch
        self._program = program
        self._listing = listing
        self._reg_map = reg_map
        self._decode = decode
        self._block_model = block_model
        self._monitor = monitor
        self._ghidra_function: Optional[Any] = None
        self._entry: int = 0
        self._name: str = ""
        self._block_count: int = 0
        self._block_view = _GhidraBlockView(arch, program, listing, reg_map, decode, self)
        self._blocks_view = _GhidraBlocksView(self)

    def _advance(self, ghidra_function: Any, block_count: int) -> None:
        """Repoint at the next Ghidra Function.

        ``block_count`` is precomputed by the provider's iter_functions
        loop (so ``len(blocks_view)`` is O(1)).
        """
        self._ghidra_function = ghidra_function
        self._entry = int(ghidra_function.getEntryPoint().getOffset())
        self._name = str(ghidra_function.getName())
        self._block_count = block_count

    def _iter_blocks(self) -> Iterator[_GhidraBlockView]:
        if self._ghidra_function is None:
            return
        body = self._ghidra_function.getBody()
        block_iter = self._block_model.getCodeBlocksContaining(body, self._monitor)
        view = self._block_view
        while block_iter.hasNext():
            gblock = block_iter.next()
            # Precount instructions for O(1) __len__ on InstructionsView.
            insn_iter = self._listing.getInstructions(gblock, True)
            count = 0
            while insn_iter.hasNext():
                ghidra_insn = insn_iter.next()
                if not body.contains(ghidra_insn.getAddress()):
                    continue
                count += 1
            view._advance(gblock, count)
            yield view

    @property
    def entry(self) -> int:
        return self._entry

    @property
    def name(self) -> str:
        return self._name

    @property
    def blocks(self) -> BlocksView:
        return self._blocks_view

    def __deepcopy__(self, memo) -> "_GhidraFunctionView":
        clone = _GhidraFunctionView(
            self._arch,
            self._program,
            self._listing,
            self._reg_map,
            self._decode,
            self._block_model,
            self._monitor,
        )
        clone._ghidra_function = self._ghidra_function
        clone._entry = self._entry
        clone._name = self._name
        clone._block_count = self._block_count
        return clone


# ---------------------------------------------------------------------------
# Jump table
# ---------------------------------------------------------------------------
class _GhidraJumpTableView:
    """Reusable jump-table wrapper.

    Reused by ``GhidraDisassemblyProvider.iter_switch_tables`` per yielded
    table. Holds the table's base address and the list of resolved target
    addresses (snapshotted per advance - the underlying Ghidra reference
    list is walked once at advance time).
    """

    __slots__ = ("_base_addr", "_targets")

    def __init__(self) -> None:
        self._base_addr: int = 0
        self._targets: list[int] = []

    def _advance(self, base_addr: int, targets: list[int]) -> None:
        self._base_addr = base_addr
        self._targets = targets

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def targets(self) -> Iterator[int]:
        return iter(self._targets)

    def __deepcopy__(self, memo) -> "_GhidraJumpTableView":
        clone = _GhidraJumpTableView()
        clone._base_addr = self._base_addr
        clone._targets = list(self._targets)
        return clone


# ---------------------------------------------------------------------------
# Typed instruction prefixes (Ghidra-side concrete subclasses)
# ---------------------------------------------------------------------------
# Per the protocol contract: prefix instances are typed-distinct, low count
# per instruction. They are constructed fresh per instruction by the
# provider's per-ISA prefix builders (G.1.b).

class _LockPrefix(LockPrefixView):
    pass


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


class _OperandSizePrefix(OperandSizePrefixView):
    pass


class _AddressSizePrefix(AddressSizePrefixView):
    pass


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


class _UpdateFlagsPrefix(UpdateFlagsPrefixView):
    pass


class _WritebackPrefix(WritebackPrefixView):
    pass


class _PpcBranchConditionPrefix(PpcBranchConditionPrefixView):
    __slots__ = ("_bc",)

    def __init__(self, bc: int) -> None:
        self._bc = bc

    @property
    def bc(self) -> int:
        return self._bc


class _PpcUpdateCr0Prefix(PpcUpdateCr0PrefixView):
    pass
