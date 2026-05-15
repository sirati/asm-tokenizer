"""Mnemonic split + alias canonicalization + x86 prefix-byte extraction.

Owns:
- ``_RegisterMap``: bidirectional register name <-> small integer ID.
- ``_SEGMENT_REGISTERS``: x86 segment-register name set.
- ``_X86_PREFIX_BYTES``: x86 legacy prefix-byte set.
- ``_GHIDRA_SUFFIX_TO_PREFIX``: Ghidra ``.REPE`` / ``.LOCK`` suffix table.
- ``_GHIDRA_MNEMONIC_ALIASES``: Ghidra form -> Capstone canonical form.
- ``_split_ghidra_mnemonic``: factor Ghidra's suffix-encoded prefix off
  a raw mnemonic.
- ``_extract_x86_prefixes``: raw-byte scanner for x86 legacy prefixes.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Instruction translation helpers
# ---------------------------------------------------------------------------


class _RegisterMap:
    """Bidirectional register name <-> small integer ID map.

    Ghidra doesn't use integer register IDs like Capstone.  We assign
    sequential small ints so they work as cache indices in VocabularyManager.
    """

    def __init__(self, program: Any) -> None:
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        language = program.getLanguage()
        for reg in language.getRegisters():
            name = str(reg.getName()).lower()
            if name not in self._name_to_id:
                rid = len(self._name_to_id)
                self._name_to_id[name] = rid
                self._id_to_name[rid] = name

    def get_id(self, reg_name: str) -> int:
        """Get (or create) a small integer ID for a register name."""
        name = reg_name.lower()
        if name not in self._name_to_id:
            rid = len(self._name_to_id)
            self._name_to_id[name] = rid
            self._id_to_name[rid] = name
        return self._name_to_id[name]

    def get_name(self, reg_id: int) -> str:
        return self._id_to_name.get(reg_id, f"reg{reg_id}")


# ---------------------------------------------------------------------------
# x86/x64 instruction translation constants
# ---------------------------------------------------------------------------

_SEGMENT_REGISTERS = frozenset({"fs", "gs", "cs", "ds", "es", "ss"})

_X86_PREFIX_BYTES = frozenset(
    {
        0xF0,  # LOCK
        0xF2,  # REPNE/REPNZ
        0xF3,  # REP/REPE/REPZ
        0x26,  # ES segment override
        0x2E,  # CS segment override
        0x36,  # SS segment override
        0x3E,  # DS segment override
        0x64,  # FS segment override
        0x65,  # GS segment override
        0x66,  # Operand size override
        0x67,  # Address size override
    }
)

_GHIDRA_SUFFIX_TO_PREFIX: dict[str, tuple[int, str]] = {
    "repe": (0xF3, "repe"),
    "repz": (0xF3, "repz"),
    "rep": (0xF3, "rep"),
    "repne": (0xF2, "repne"),
    "repnz": (0xF2, "repnz"),
    "lock": (0xF0, "lock"),
}

_GHIDRA_MNEMONIC_ALIASES: dict[str, str] = {
    # Conditional jumps -- Ghidra form -> Capstone canonical
    "jz": "je",
    "jnz": "jne",
    "jnbe": "ja",
    "jnae": "jb",
    "jna": "jbe",
    "jnb": "jae",
    "jnge": "jl",
    "jnle": "jg",
    "jnl": "jge",
    "jng": "jle",
    "jpe": "jp",
    "jpo": "jnp",
    # Conditional moves
    "cmovz": "cmove",
    "cmovnz": "cmovne",
    "cmovnbe": "cmova",
    "cmovnae": "cmovb",
    "cmovna": "cmovbe",
    "cmovnb": "cmovae",
    "cmovnge": "cmovl",
    "cmovnle": "cmovg",
    "cmovnl": "cmovge",
    "cmovng": "cmovle",
    # Conditional sets
    "setz": "sete",
    "setnz": "setne",
    "setna": "setbe",
    "setnae": "setb",
    "setnb": "setae",
    "setnbe": "seta",
    "setng": "setle",
    "setnge": "setl",
    "setnl": "setge",
    "setnle": "setg",
    # Misc
    "retn": "ret",
}


def _split_ghidra_mnemonic(raw_mnemonic: str) -> tuple[str, str | None, int | None]:
    """Split Ghidra's suffix-encoded prefix from a mnemonic.

    Ghidra encodes rep/lock as a dot-suffix: ``CMPSB.REPE``, ``ADD.LOCK``.
    Returns ``(base_mnemonic, prefix_name, prefix_byte)`` or
    ``(mnemonic, None, None)`` when there is no suffix.
    """
    lower = raw_mnemonic.lower()
    if "." in lower:
        base, suffix = lower.rsplit(".", 1)
        if suffix in _GHIDRA_SUFFIX_TO_PREFIX:
            prefix_byte, prefix_name = _GHIDRA_SUFFIX_TO_PREFIX[suffix]
            return base, prefix_name, prefix_byte
    return lower, None, None


def _extract_x86_prefixes(ghidra_insn: Any) -> set[int]:
    """Extract x86 legacy prefix bytes from the raw instruction encoding."""
    raw = ghidra_insn.getBytes()
    prefixes: set[int] = set()
    for b in raw:
        unsigned = int(b) & 0xFF
        if unsigned in _X86_PREFIX_BYTES:
            prefixes.add(unsigned)
        else:
            break  # first non-prefix byte = opcode start
    return prefixes
