"""ARM stm/ldm-family register-list sub-view.

Owns ``_GhidraRegisterListView``: bound to the parent operand's cursor,
holds the decomposed ``(base, writeback, [members])`` for the current
reg-list operand. The provider's reg-list decomposition callback
populates this on first ``op.reg_list`` access.

Member registers are exposed as reusable ``_GhidraRegisterView`` cursors.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator

from tokenizer.disasm.ghidra_views.register import (
    _REG_ID_ABSENT,
    _REG_NAME_ABSENT,
    _GhidraRegisterView,
)
from tokenizer.disasm.types import Architecture, RegisterView


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
