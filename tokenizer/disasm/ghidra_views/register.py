"""Reusable register wrapper for Ghidra-backed views.

Owns:
- ``_GhidraRegisterView``: per-operand register cursor (reused).
- ``_REG_ID_ABSENT`` / ``_REG_NAME_ABSENT``: sentinel constants signalling
  an empty register slot (``RegisterView.is_absent``).

See module ``tokenizer.disasm.ghidra_views`` docstring for lifecycle rules.
"""

from __future__ import annotations

from tokenizer.disasm.types import Architecture


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
