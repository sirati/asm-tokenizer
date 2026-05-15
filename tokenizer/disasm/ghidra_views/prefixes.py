"""Typed instruction prefixes (Ghidra-side concrete subclasses).

Per the protocol contract: prefix instances are typed-distinct, low count
per instruction. They are constructed fresh per instruction by the
provider's per-ISA prefix builders (G.1.b).
"""

from __future__ import annotations

from tokenizer.disasm.types import (
    AddressSizePrefixView,
    ArmConditionCode,
    BranchHintPrefixView,
    ConditionCodePrefixView,
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
