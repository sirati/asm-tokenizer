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

Submodule layout (single-concern split, task #63):
- ``register``: ``_GhidraRegisterView`` + absent-slot sentinels.
- ``reg_list``: ``_GhidraRegisterListView`` (ARM stm/ldm-family).
- ``operand``: ``_GhidraOperandView`` + sub-views (mem / shift / crx).
- ``instruction``: ``_GhidraInstructionView`` + ``_GhidraOperandsView``
  + ``_GhidraPrefixesView``.
- ``block``: ``_GhidraBlockView`` + ``_GhidraInstructionsView``.
- ``function``: ``_GhidraFunctionView`` + ``_GhidraBlocksView``.
- ``jump_table``: ``_GhidraJumpTableView``.
- ``prefixes``: typed prefix concrete subclasses.
"""

from tokenizer.disasm.ghidra_views.block import (
    _GhidraBlockView,
    _GhidraInstructionsView,
)
from tokenizer.disasm.ghidra_views.function import (
    _GhidraBlocksView,
    _GhidraFunctionView,
)
from tokenizer.disasm.ghidra_views.instruction import (
    _GhidraInstructionView,
    _GhidraOperandsView,
    _GhidraPrefixesView,
)
from tokenizer.disasm.ghidra_views.jump_table import _GhidraJumpTableView
from tokenizer.disasm.ghidra_views.operand import (
    _GhidraCrxFieldView,
    _GhidraMemoryOperandView,
    _GhidraOperandView,
    _GhidraShiftModifierView,
)
from tokenizer.disasm.ghidra_views.prefixes import (
    _AddressSizePrefix,
    _BranchHintPrefix,
    _ConditionCodePrefix,
    _LockPrefix,
    _OperandSizePrefix,
    _PpcBranchConditionPrefix,
    _PpcUpdateCr0Prefix,
    _RepPrefix,
    _SegmentOverridePrefix,
    _UpdateFlagsPrefix,
    _WritebackPrefix,
)
from tokenizer.disasm.ghidra_views.reg_list import _GhidraRegisterListView
from tokenizer.disasm.ghidra_views.register import (
    _REG_ID_ABSENT,
    _REG_NAME_ABSENT,
    _GhidraRegisterView,
)

__all__ = [
    "_REG_ID_ABSENT",
    "_REG_NAME_ABSENT",
    "_GhidraRegisterView",
    "_GhidraRegisterListView",
    "_GhidraMemoryOperandView",
    "_GhidraShiftModifierView",
    "_GhidraCrxFieldView",
    "_GhidraOperandView",
    "_GhidraOperandsView",
    "_GhidraPrefixesView",
    "_GhidraInstructionView",
    "_GhidraInstructionsView",
    "_GhidraBlockView",
    "_GhidraBlocksView",
    "_GhidraFunctionView",
    "_GhidraJumpTableView",
    "_LockPrefix",
    "_RepPrefix",
    "_SegmentOverridePrefix",
    "_OperandSizePrefix",
    "_AddressSizePrefix",
    "_BranchHintPrefix",
    "_ConditionCodePrefix",
    "_UpdateFlagsPrefix",
    "_WritebackPrefix",
    "_PpcBranchConditionPrefix",
    "_PpcUpdateCr0Prefix",
]
