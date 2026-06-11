"""Block cursor + its instructions container view.

Owns:
- ``_GhidraBlockView``: reusable block wrapper.
- ``_GhidraInstructionsView``: container view over a block's instructions.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, TYPE_CHECKING

from tokenizer.disasm.ghidra_views.instruction import _GhidraInstructionView
from tokenizer.disasm.types import Architecture, InstructionsView

if TYPE_CHECKING:
    from tokenizer.disasm.ghidra_views.function import _GhidraFunctionView


# ---------------------------------------------------------------------------
# Container views (NOT lists - iterate, reuse child wrapper)
# ---------------------------------------------------------------------------
class _GhidraInstructionsView:
    """Container view over a block's instructions.

    NOT a list. ``__iter__`` walks the block's Ghidra instruction iterator
    and yields the same reused ``_GhidraInstructionView`` per instruction.
    ``__len__`` computes lazily (one Listing walk, cached per cursor
    position) — the hot path iterates instead of asking for the length,
    so the per-block instruction pre-count walk the function wrapper
    used to pay is gone.
    """

    __slots__ = ("_block",)

    def __init__(self, block: "_GhidraBlockView") -> None:
        self._block = block

    def __len__(self) -> int:
        return self._block._ensure_instruction_count()

    def __iter__(self) -> Iterator["_GhidraInstructionView"]:
        return self._block._iter_instructions()


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
        # Lazily-counted (``None`` = not counted for the current cursor
        # position; see ``_ensure_instruction_count``).
        self._instruction_count: Optional[int] = None
        self._instruction_view = _GhidraInstructionView(arch, program, reg_map, decode)
        self._instructions_view = _GhidraInstructionsView(self)

    def _advance(self, ghidra_block: Any) -> None:
        """Repoint at the next CodeBlock.

        The instruction count is NOT precomputed: ``len(instructions)``
        computes lazily via ``_ensure_instruction_count`` on first ask
        (cached until the next ``_advance``). The hot path materialises
        token rows by iterating, never by asking for the length, so the
        former per-block full-Listing pre-walk is gone.
        """
        self._ghidra_block = ghidra_block
        self._addr = int(ghidra_block.getMinAddress().getOffset())
        self._size = int(ghidra_block.getMaxAddress().getOffset()) - self._addr + 1
        self._instruction_count = None

    def _ensure_instruction_count(self) -> int:
        """Count the block's body-contained instructions lazily (cached
        per cursor position). Same walk + ``body.contains`` filter as
        ``_iter_instructions`` minus the per-instruction decode."""
        if self._instruction_count is None:
            count = 0
            if self._ghidra_block is not None:
                body = (
                    self._function._ghidra_function.getBody()
                    if self._function._ghidra_function is not None
                    else None
                )
                insn_iter = self._listing.getInstructions(self._ghidra_block, True)
                while insn_iter.hasNext():
                    ghidra_insn = insn_iter.next()
                    if body is not None and not body.contains(ghidra_insn.getAddress()):
                        continue
                    count += 1
            self._instruction_count = count
        return self._instruction_count

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
