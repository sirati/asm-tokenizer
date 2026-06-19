"""Mnemonic split + alias canonicalization + x86 prefix-byte extraction.

Owns:
- ``_RegisterMap``: bidirectional register name <-> small integer ID.
- ``_SEGMENT_REGISTERS``: x86 segment-register name set.
- ``_X86_PREFIX_BYTES``: x86 legacy prefix-byte set.
- ``_GHIDRA_SUFFIX_TO_PREFIX``: Ghidra ``.REPE`` / ``.LOCK`` suffix table.
- ``_GHIDRA_MNEMONIC_ALIASES``: Ghidra form -> Capstone canonical form.
- ``_ARM_CC_LITERAL_TO_ENUM`` / ``_ARM_COND_MNEMONIC_ALLOWLIST`` /
  ``_strip_arm_cc_suffix``: detect Ghidra SLEIGH's ``^COND`` mnemonic
  concatenation (``bne``, ``beq``, ``mvneq``, ``b.eq``, ...) and split
  off the condition code.
- ``_split_ghidra_mnemonic``: factor Ghidra's suffix-encoded prefix off
  a raw mnemonic.
- ``_extract_x86_prefixes``: raw-byte scanner for x86 legacy prefixes.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.types import ArmConditionCode


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


# ---------------------------------------------------------------------------
# ARM / AArch64 condition-code mnemonic-suffix detection
# ---------------------------------------------------------------------------
# Ghidra's ARM SLEIGH spec encodes condition codes as a mnemonic SUFFIX
# via ``^COND`` (and AArch64 via ``^"."^BranchCondOp``): ``bne``, ``beq``,
# ``mvneq``, ``streq``, ``ldmiaeq``, ``cmpne``, ``b.eq``, ``bc.eq``, ...
# So ``getMnemonicString()`` returns the cc-suffixed form, and the
# ``ConditionCodePrefixView`` signal must be recovered by splitting that
# string. The split needs an allow-list of base stems because a blind
# 2-char suffix check false-positives on ``teq`` (looks like ``t``+``eq``)
# and ``tst``-style bases that happen to end in cc-like letters.

_ARM_CC_LITERAL_TO_ENUM: dict[str, ArmConditionCode] = {
    "eq": ArmConditionCode.EQ, "ne": ArmConditionCode.NE,
    "cs": ArmConditionCode.CS, "cc": ArmConditionCode.CC,
    "mi": ArmConditionCode.MI, "pl": ArmConditionCode.PL,
    "vs": ArmConditionCode.VS, "vc": ArmConditionCode.VC,
    "hi": ArmConditionCode.HI, "ls": ArmConditionCode.LS,
    "ge": ArmConditionCode.GE, "lt": ArmConditionCode.LT,
    "gt": ArmConditionCode.GT, "le": ArmConditionCode.LE,
}

# Base mnemonics known to take ``^COND`` per Ghidra's ARM SLEIGH spec.
# Derived by grepping
#   ``Ghidra/Processors/ARM/data/languages/*.sinc`` (and AArch64) for
#   ``^:<mnemonic>\^(COND|cc|thcc|part2thcc|ItCond|".\"^BranchCondOp)``
# (see ARM SLEIGH ``COND`` macro family). The allow-list is essential:
# without it, a blind 2-char suffix check would false-positively strip
# ``eq`` off ``teq`` (yielding stem ``t``) and so on.
_ARM_COND_MNEMONIC_ALLOWLIST: frozenset[str] = frozenset({
    "adc", "add", "addw", "adr", "and", "b", "bfc", "bfi", "bic", "bl",
    "blx", "bx", "bxj", "cbnz", "cbz", "cdp", "cdp2", "chka", "clrex",
    "clz", "cmn", "cmp", "cpsid", "cpsie", "cpy", "dbg", "dmb", "dsb",
    "enterx", "eor", "fldmdbx", "fldmiax", "fstmdbx", "fstmiax", "hb",
    "isb", "lda", "ldab", "ldaex", "ldaexb", "ldaexd", "ldaexh", "ldah",
    "ldc", "ldc2", "ldc2l", "ldcl", "ldm", "ldmdb", "ldmia", "ldr",
    "ldrb", "ldrbt", "ldrd", "ldrex", "ldrexb", "ldrexd", "ldrexh",
    "ldrh", "ldrht", "ldrsb", "ldrsbt", "ldrsh", "ldrsht", "ldrt",
    "leavex", "mcr", "mcr2", "mcrr", "mla", "mls", "mov", "movt",
    "movw", "mrc", "mrc2", "mrrc", "mrrc2", "mrs", "msr", "mul", "mvn",
    "nop", "orr", "pkhbt", "pkhtb", "pld", "pldw", "pli", "pop", "push",
    "qadd", "qadd16", "qadd8", "qasx", "qdadd", "qdsub", "qsax", "qsub",
    "qsub16", "qsub8", "rbit", "rev", "rev16", "revsh", "rsb", "rsc",
    "sadd16", "sadd8", "sasx", "sbfx", "sdiv", "sel", "setend", "sev",
    "sevl", "shadd16", "shadd8", "shasx", "shsax", "shsub16", "shsub8",
    "smc", "smlad", "smladx", "smlal", "smlald", "smlaldx", "smlsd",
    "smlsdx", "smlsld", "smlsldx", "smmla", "smmlar", "smmls", "smmlsr",
    "smmul", "smmulr", "smuad", "smuadx", "smulbb", "smulbt", "smull",
    "smultb", "smultt", "smusd", "smusdx", "srsdb", "srsia", "srsib",
    "ssat", "ssat16", "ssax", "ssub16", "ssub8", "stc", "stc2", "stc2l",
    "stcl", "stl", "stlb", "stlex", "stlexb", "stlexd", "stlexh", "stlh",
    "stm", "stmdb", "stmia", "str", "strb", "strbt", "strd", "strex",
    "strexb", "strexd", "strexh", "strh", "strht", "strt", "sub",
    "subw", "svc", "swi", "swp", "swpb", "sxtab", "sxtab16", "sxtah",
    "sxtb", "sxtb16", "sxth", "tbb", "tbh", "teq", "tst", "tt", "tta",
    "ttat", "ttt", "uadd16", "uadd8", "uasx", "ubfx", "udf", "udiv",
    "uhadd16", "uhadd8", "uhasx", "uhsax", "uhsub16", "uhsub8", "umaal",
    "umlal", "umull", "uqadd16", "uqadd8", "uqasx", "uqsax", "uqsub16",
    "uqsub8", "usad8", "usada8", "usat", "usat16", "usax", "usub16",
    "usub8", "uxtab", "uxtab16", "uxtah", "uxtb", "uxtb16", "uxth",
    "vabs", "vadd", "vcvt", "vcvtb", "vcvtr", "vcvtt", "vdiv", "vdup",
    "vfma", "vfms", "vfnma", "vfnms", "vldmdb", "vldmia", "vldr", "vmla",
    "vmls", "vmov", "vmrs", "vmsr", "vmul", "vneg", "vnmla", "vnmls",
    "vnmul", "vpop", "vpush", "vsqrt", "vstmdb", "vstmia", "vstr",
    "vsub", "wfe", "wfi", "yield",
    # AArch64 conditional branches: ``b.eq`` / ``bc.eq`` use a literal
    # dot separator. The 3-char tail path covers the dot form; the
    # bare stems are already in the ARM list above (``b``, plus we add
    # ``bc`` for AArch64-only).
    "bc",
})


def _strip_arm_cc_suffix(
    mnemonic: str,
) -> tuple[str, ArmConditionCode | None]:
    """Split a Ghidra ARM/AArch64 mnemonic into ``(stem, cc_or_None)``.

    Recognized forms:

    - 2-char suffix (ARM ``^COND``): ``bne`` -> (``b``, NE).
    - 3-char suffix with literal dot (AArch64 ``b^"."^BranchCondOp``):
      ``b.eq`` -> (``b``, EQ).

    The allow-list guards against false-positive strips on bases whose
    last two characters happen to be cc-like (``teq``, ``tst``, ``tt``,
    ``adr``, ...). When the stem is not in the allow-list, the input is
    returned unchanged with ``cc = None``.
    """
    for tail_len in (3, 2):
        if tail_len > len(mnemonic):
            continue
        if tail_len == 3 and mnemonic[-3] != ".":
            continue
        cc_chars = mnemonic[-2:]
        cc_enum = _ARM_CC_LITERAL_TO_ENUM.get(cc_chars)
        if cc_enum is None:
            continue
        stem = mnemonic[:-tail_len]
        if stem in _ARM_COND_MNEMONIC_ALLOWLIST:
            return (stem, cc_enum)
    return (mnemonic, None)


# ---------------------------------------------------------------------------
# PowerPC branch-condition + CR0-update (Rc) mnemonic-suffix detection
# ---------------------------------------------------------------------------
# Ghidra's PowerPC SLEIGH spec folds the branch condition into the
# displayed mnemonic via the ``CC`` subtable (``ppc_common.sinc``:
# ``CC: "lt"|"le"|"eq"|"ge"|"gt"|"ne"|"so"|"ns"``) concatenated as
# ``b^CC^...``. So ``getMnemonicString()`` returns forms such as ``beq``,
# ``bne``, ``blt``, plus the link / absolute / register variants
# ``beql`` / ``beqa`` / ``beqla`` / ``beqlr`` / ``beqlrl`` / ``beqctr`` /
# ``beqctrl``. The Rc bit (CR0 update) is rendered as a literal trailing
# ``.`` baked directly into the mnemonic (``add.``, ``or.``, ``rlwinm.``).
#
# Both signals are recovered by inspecting the mnemonic string -- the same
# idiom ARM uses (``_strip_arm_cc_suffix``) -- because Ghidra exposes
# neither the BO/BI fields nor the Rc bit as a typed operand on the
# instruction: the condition is folded into the mnemonic via the CC
# subtable and the Rc bit into the mnemonic spelling.
#
# The returned ``bc`` integer matches the Capstone-4.x ``ppc_bc`` small
# enum (lt=1..ns=10) that the PPC architecture provider's ``_PPC_BC_NAMES``
# table consumes -- the typed ``PpcBranchConditionPrefixView.bc`` contract
# value, NOT a Ghidra-internal code -- so the Ghidra and angr paths feed
# the same downstream rendering table.
#
# Ghidra renders only ``so`` / ``ns`` (never the ``un`` / ``nu`` unordered
# aliases), so those two contract values (7/8) are unreachable from this
# path -- as they are from the Ghidra mnemonic itself.
_PPC_CC_LITERAL_TO_BC: dict[str, int] = {
    "lt": 1, "le": 2, "eq": 3, "ge": 4, "gt": 5,
    "ne": 6, "so": 9, "ns": 10,
}

# Suffixes the ``b^CC^...`` constructs append after the two-char condition:
# "" (relative), "a" (absolute), "l"/"la" (link), and the register-branch
# tails "lr"/"lrl"/"ctr"/"ctrl". Enumerated explicitly so the recognized
# conditional-branch mnemonic set is exact -- a blind ``b``-prefix +
# cc-substring strip would false-positive on unconditional ``b`` / ``bl`` /
# ``blr`` / ``bctr`` / CTR-decrement ``bdnz`` (whose condition, on the
# ``bdnzt`` / ``bdnzf`` forms, is a separate operand Ghidra does NOT fold
# into the mnemonic and which this path therefore does not surface).
_PPC_BRANCH_COND_SUFFIXES: tuple[str, ...] = (
    "", "a", "l", "la", "lr", "lrl", "ctr", "ctrl",
)

# Exact recognized conditional-branch mnemonic -> bc contract value.
# Built as the cross product {cc} x {suffix}; membership is the sole
# discriminator, so there is no fragile substring stripping.
_PPC_BRANCH_MNEMONIC_TO_BC: dict[str, int] = {
    f"b{cc}{suffix}": bc
    for cc, bc in _PPC_CC_LITERAL_TO_BC.items()
    for suffix in _PPC_BRANCH_COND_SUFFIXES
}


def _ppc_branch_condition(mnemonic: str) -> int | None:
    """Return the ``bc`` contract value for a Ghidra PPC conditional branch.

    ``mnemonic`` is ``getMnemonicString()`` lowercased. Returns the
    Capstone-4.x ``ppc_bc`` small-enum integer (lt=1..ns=10) consumed by
    the PPC provider's ``_PPC_BC_NAMES`` table when ``mnemonic`` is one of
    the recognized ``b^CC^...`` conditional-branch forms, else ``None``
    (unconditional ``b`` / ``bl`` / ``blr`` / ``bctr`` and the
    CTR-decrement ``bdnz`` family, none of which fold a condition into the
    mnemonic).
    """
    return _PPC_BRANCH_MNEMONIC_TO_BC.get(mnemonic)


def _ppc_has_cr0_update(mnemonic: str) -> bool:
    """True iff the Ghidra PPC mnemonic carries the Rc-bit ``.`` suffix.

    Ghidra bakes the record-condition (Rc) bit directly into the displayed
    mnemonic as a trailing ``.`` (``add.``, ``or.``, ``rlwinm.``), the same
    bit Capstone surfaces as ``cs_insn.update_cr0``. The non-Rc CR writers
    (``cmpw`` and friends) carry no ``.`` and so are correctly excluded.
    """
    return mnemonic.endswith(".")
