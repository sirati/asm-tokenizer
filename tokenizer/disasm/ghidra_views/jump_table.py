"""Reusable jump-table view.

Owns ``_GhidraJumpTableView``: reused by
``GhidraDisassemblyProvider.iter_switch_tables`` per yielded table.
Holds the table's base address and the list of resolved target
addresses (snapshotted per advance - the underlying Ghidra reference
list is walked once at advance time).
"""

from __future__ import annotations

from typing import Iterator


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
