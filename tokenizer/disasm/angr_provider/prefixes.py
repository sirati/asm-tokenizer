"""Per-platform instruction-prefix builders + concrete prefix subclasses.

``_build_prefixes(cs_insn, arch)`` is the single dispatcher that yields a
typed ``list[InstructionPrefixView]`` for a Capstone instruction. Per-arch
helpers live alongside it; consumers see only the dispatcher.

Concrete subclasses for the prefix Protocols that carry data:
    RepPrefixView, SegmentOverridePrefixView, BranchHintPrefixView,
    ConditionCodePrefixView, PpcBranchConditionPrefixView
The data-less prefix types (LockPrefixView, OperandSizePrefixView,
AddressSizePrefixView, UpdateFlagsPrefixView, WritebackPrefixView,
PpcUpdateCr0PrefixView) are concrete on the Protocol side and instantiated
directly.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.types import (
    AddressSizePrefixView,
    Architecture,
    ArmConditionCode,
    BranchHintPrefixView,
    ConditionCodePrefixView,
    InstructionPrefixView,
    LockPrefixView,
    OperandSizePrefixView,
    PpcBranchConditionPrefixView,
    PpcUpdateCr0PrefixView,
    RepPrefixView,
    SegmentOverridePrefixView,
    UpdateFlagsPrefixView,
    WritebackPrefixView,
    X86BranchHint,
    X86Segment,
)


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
    # segment overrides -- disambiguated in ``_build_prefixes_x86``.
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
        # Unknown byte -- silently skip; Capstone occasionally surfaces
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


# Capstone's ``ppc_bc`` enum is version-specific: Capstone 4.x numbered the
# conditions 1..10, but 5.x renumbered them (``PPC_BC_LT=12``,
# ``PPC_BC_EQ=76``, ...). ``PpcBranchConditionPrefixView.bc`` carries the
# STABLE contract value (1..10) that the PPC provider's ``_PPC_BC_NAMES``
# table renders and the Ghidra prefix builder also emits, so the raw
# Capstone value must be translated to that contract -- keyed on the
# symbolic ``PPC_BC_*`` constants so the mapping survives Capstone enum
# renumbering. ``capstone`` is imported lazily (the angr PPC path requires
# it; the non-angr import paths that pull in this module must not).
_PPC_BC_TO_CONTRACT: dict[int, int] | None = None


def _ppc_bc_contract(bc_raw: int) -> int | None:
    """Translate a Capstone ``ppc_bc`` value to the stable bc contract.

    Returns the 1..10 contract value (``_PPC_BC_NAMES`` key) for a
    recognized condition, else ``None`` (``PPC_BC_INVALID`` / unconditional).
    """
    global _PPC_BC_TO_CONTRACT
    if _PPC_BC_TO_CONTRACT is None:
        from capstone import ppc_const as _c

        _PPC_BC_TO_CONTRACT = {
            _c.PPC_BC_LT: 1, _c.PPC_BC_LE: 2, _c.PPC_BC_EQ: 3,
            _c.PPC_BC_GE: 4, _c.PPC_BC_GT: 5, _c.PPC_BC_NE: 6,
            _c.PPC_BC_UN: 7, _c.PPC_BC_NU: 8, _c.PPC_BC_SO: 9,
            _c.PPC_BC_NS: 10,
        }
    return _PPC_BC_TO_CONTRACT.get(bc_raw)


def _build_prefixes_ppc(cs_insn: Any) -> list[InstructionPrefixView]:
    """Extract typed PPC prefixes from a Capstone instruction.

    PPC's per-instruction modifiers are the branch-condition field
    (``cs_insn.bc``, a version-specific Capstone ``ppc_bc`` enum value
    translated to the stable bc contract via ``_ppc_bc_contract``) and the
    CR0-update Rc bit (``cs_insn.update_cr0``).
    """
    out: list[InstructionPrefixView] = []
    bc = _ppc_bc_contract(int(getattr(cs_insn, "bc", 0) or 0))
    if bc is not None:
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
