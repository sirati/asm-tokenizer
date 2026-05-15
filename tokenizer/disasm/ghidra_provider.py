"""Ghidra-based disassembly provider using pyghidra (headless, no IPC).

pyghidra gives direct access to the Ghidra Java API from CPython via JPype.
This provider translates Ghidra's Instruction/Register/Scalar objects into
Capstone-compatible adapter objects so existing ArchitectureProviders work
unchanged.

Requirements:
    pip install pyghidra
    GHIDRA_INSTALL_DIR env var or pass install_dir to pyghidra.start()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from tokenizer.disasm import DisassemblyProvider, MetadataLookup

# ---------------------------------------------------------------------------
# Capstone-compatible adapter objects
# ---------------------------------------------------------------------------
# The existing ArchitectureProviders access deeply ISA-specific Capstone
# attributes.  Rather than abstracting those away (which would create a
# bloated union type), we produce thin wrappers that expose the same
# attribute surface as Capstone objects.
#
# Capstone operand type constants (mirrored here so we don't need capstone
# installed when only Ghidra is used):
_OP_REG = 1
_OP_IMM = 2
_OP_MEM = 3


@dataclass
class _CapMemOperand:
    """Capstone-compatible memory operand fields."""

    base: int = 0
    index: int = 0
    scale: int = 1
    disp: int = 0
    segment: int = 0


@dataclass
class _GhidraMemRawData:
    """Raw Ghidra data for native memory operand tokenization.

    Carried on _CapOperand so the Ghidra arch provider can tokenize
    MEM operands directly from Ghidra API objects without string parsing.
    """

    ghidra_insn: Any  # Ghidra Instruction object (for size inference)
    op_objects: list[Any]  # getOpObjects() result
    reg_map: Any  # _RegisterMap instance (for name -> id conversion)


@dataclass
class _CapShift:
    """Capstone-compatible shift descriptor (ARM32)."""

    type: int = 0
    value: int = 0


@dataclass
class _CapOperand:
    """Capstone-compatible operand object.

    Fields are set depending on ``type``:
        1 (REG)  -> ``reg``
        2 (IMM)  -> ``imm``
        3 (MEM)  -> ``mem``
       64 (CRX, PPC only) -> ``crx``

    ``fp_width_bytes`` is the per-operand FP width derived from Ghidra's
    ``OperandType.FLOAT`` bitmask (one of {2, 4, 8, 10, 16}) or ``None``
    when the operand is not FP-typed. Populated at decode time by
    ``_ghidra_insn_to_cap``; angr-path operands (raw Capstone CsOpnd) do
    not carry this attribute, so call sites read it via
    ``getattr(op, "fp_width_bytes", None)``. See ``angr_limitations.md``
    §1 for why the angr path stays None.
    """

    type: int = 0
    reg: int = 0
    imm: int = 0
    mem: _CapMemOperand = field(default_factory=_CapMemOperand)
    size: int = 0  # x86 operand size in bytes
    shift: _CapShift = field(default_factory=_CapShift)
    ghidra_raw_data: _GhidraMemRawData | None = None
    fp_width_bytes: Optional[int] = None

    @dataclass
    class _CRX:
        reg: int = 0

    crx: _CRX = field(default_factory=_CRX)


@dataclass
class _CapInsnInner:
    """Stands in for ``insn.insn`` — the raw Capstone CsInsn that
    ArchitectureProviders access for ISA-specific fields."""

    _insn_name: str = ""
    cc: int = 0  # ARM32 condition code
    update_flags: bool = False  # ARM32 S suffix
    writeback: bool = False  # ARM32 ! suffix
    bc: int = 0  # PPC branch condition
    update_cr0: bool = False  # PPC Rc bit

    def insn_name(self) -> str:
        return self._insn_name


class _CapInstruction:
    """Capstone-compatible instruction adapter wrapping a Ghidra Instruction.

    Attributes consumed by existing code:
        mnemonic, op_str, operands, prefix, insn (._CapInsnInner),
        reg_name(reg_id)
    """

    __slots__ = ("mnemonic", "op_str", "operands", "prefix", "insn", "_reg_map")

    def __init__(
        self,
        mnemonic: str,
        op_str: str,
        operands: list[_CapOperand],
        prefix: bytes,
        insn_inner: _CapInsnInner,
        reg_map: dict[int, str],
    ):
        self.mnemonic = mnemonic
        self.op_str = op_str
        self.operands = operands
        self.prefix = prefix
        self.insn = insn_inner
        self._reg_map = reg_map

    def reg_name(self, reg_id: int) -> str:
        return self._reg_map.get(reg_id, f"reg{reg_id}")


@dataclass
class _CapBlock:
    """Capstone-compatible block adapter.

    ``fill_constant_candidates`` accesses:
        block.addr, block.size, block.capstone.insns
    """

    addr: int
    size: int
    capstone: Any = None  # set after construction

    @dataclass
    class _CapstoneHolder:
        insns: list[_CapInstruction]

    def set_insns(self, insns: list[_CapInstruction]) -> None:
        self.capstone = self._CapstoneHolder(insns)


@dataclass
class _CapFunction:
    """Capstone-compatible function adapter.

    ``fill_constant_candidates`` accesses:
        func.blocks  (iterable, used multiple times — must be a list)

    ``_raw`` carries the underlying Ghidra ``Function`` object for
    Ghidra-side provider consumers (e.g. ``iter_switch_tables``) that
    need to reach back into the Ghidra API. Defaults to ``None`` so the
    angr-side provider, which constructs the wrapper differently, is
    unaffected.
    """

    _blocks: list[_CapBlock]
    _raw: Any = None

    @property
    def blocks(self) -> list[_CapBlock]:
        return self._blocks


# ---------------------------------------------------------------------------
# Section-name -> v2 section-type mapping
# ---------------------------------------------------------------------------
# Used by both the symbol-type fix (replacing bare ``"symbol"`` with the
# containing-section's type) and the memory-block fallback branch in
# ``GhidraMetadataLookup.lookup``. ELF section names are the source of truth;
# Ghidra preserves them verbatim as ``MemoryBlock.getName()``.
_SECTION_TYPE_BY_NAME: dict[str, str] = {
    # executable / code-bearing
    ".text": "code",
    ".init": "code",
    ".fini": "code",
    ".plt": "code",
    ".plt.got": "code",
    ".plt.sec": "code",
    # read-only data
    ".rodata": "rodata",
    ".rodata1": "rodata",
    ".data.rel.ro": "rodata",
    ".data.rel.ro.local": "rodata",
    ".eh_frame": "rodata",
    ".eh_frame_hdr": "rodata",
    ".gcc_except_table": "rodata",
    # code-pointer arrays (rodata-flavored but tagged for the classifier)
    ".init_array": "code_ptr_table",
    ".fini_array": "code_ptr_table",
    ".preinit_array": "code_ptr_table",
    ".ctors": "code_ptr_table",
    ".dtors": "code_ptr_table",
    # writable data
    ".data": "data",
    ".data1": "data",
    ".bss": "data",
    ".sbss": "data",
    # TLS
    ".tdata": "thread_local_data",
    ".tbss": "thread_local_data",
}


def _section_type_from_block(block: Any) -> str:
    """Map a Ghidra ``MemoryBlock`` to a v2 section-type string.

    Looks up the block's name in ``_SECTION_TYPE_BY_NAME`` first; falls
    back to permission-bit inference so unknown / vendor-specific section
    names still get a sensible type. ``block`` must be non-None.
    """
    name = str(block.getName())
    if name in _SECTION_TYPE_BY_NAME:
        return _SECTION_TYPE_BY_NAME[name]
    # Permission-bit fallback for unrecognized section names. Matches the
    # angr-side ``_get_section_type`` logic in ``address_meta_data_lookup.py``.
    if block.isExecute():
        return "code"
    if block.isWrite():
        return "data"
    return "rodata"


def _is_plt_block_name(name: str) -> bool:
    """Return True if ``name`` is any of the PLT-variant section names."""
    return name == ".plt" or name.startswith(".plt.")


def _is_code_ptr_table_block_name(name: str) -> bool:
    """Return True if ``name`` is a known function-pointer-array section."""
    return name in {".init_array", ".fini_array", ".preinit_array", ".ctors", ".dtors"}


def _is_tls_block_name(name: str) -> bool:
    """Return True if ``name`` is a TLS section."""
    return name in {".tdata", ".tbss"}


def _is_rodata_block_name(name: str) -> bool:
    """Return True if ``name`` is a rodata-flavored section (vtables live here)."""
    return name in {".rodata", ".rodata1", ".data.rel.ro", ".data.rel.ro.local"}


# ---------------------------------------------------------------------------
# String DataType detection
# ---------------------------------------------------------------------------
# Ghidra's string analyzers create ``Data`` objects whose ``DataType`` is a
# subclass of ``AbstractStringDataType`` (or, in older Ghidra, individual
# concrete classes like ``StringDataType`` / ``UnicodeDataType``). The pyghidra
# interface exposes ``data.hasStringValue()`` as the authoritative cross-version
# check; the data type's name then yields the encoding.
_STRING_TYPE_NAME_TO_ENCODING: dict[str, str] = {
    "string": "ascii",
    "string-utf8": "utf-8",
    "TerminatedCString": "ascii",
    "TerminatedAsciiString": "ascii",
    "TerminatedUTF8": "utf-8",
    "unicode": "utf-16-le",
    "unicode32": "utf-32-le",
    "TerminatedUnicode": "utf-16-le",
    "TerminatedUnicode32": "utf-32-le",
    "PascalString": "ascii",
    "PascalUnicode": "utf-16-le",
    "PascalString255": "ascii",
    "MBCString": "mbcs",
}


def _encoding_from_string_datatype(dt: Any) -> str:
    """Map a Ghidra string ``DataType`` to a Python codec name.

    Falls back to ``"ascii"`` when the type's name isn't recognized;
    callers can downgrade gracefully (the byte payload is preserved
    regardless of encoding).
    """
    if dt is None:
        return "ascii"
    name = str(dt.getName())
    # Direct hit
    if name in _STRING_TYPE_NAME_TO_ENCODING:
        return _STRING_TYPE_NAME_TO_ENCODING[name]
    # Substring heuristics for vendor-spelling variations
    lname = name.lower()
    if "unicode32" in lname or "utf32" in lname or "utf-32" in lname:
        return "utf-32-le"
    if "unicode" in lname or "utf16" in lname or "utf-16" in lname:
        return "utf-16-le"
    if "utf8" in lname or "utf-8" in lname:
        return "utf-8"
    return "ascii"


def _java_bytes_to_python(raw: Any) -> bytes:
    """Convert a Java ``byte[]`` (JPype-wrapped) into a Python ``bytes``.

    Java bytes are signed; mask each element to the unsigned 0–255 range
    before assembling. ``raw`` may be None when the Data object hasn't
    materialized its byte payload — in that case return ``b""``.
    """
    if raw is None:
        return b""
    return bytes(int(b) & 0xFF for b in raw)


# ---------------------------------------------------------------------------
# Vtable detection
# ---------------------------------------------------------------------------
# Ghidra's GCC-RTTI analyzer (``GccRttiAnalyzer``) creates ``Data`` objects
# for vftables with a symbol whose name includes ``vftable`` (case-insensitive
# variants exist: ``ClassName::vftable``, ``Vftable``, ``_ZTV...``). The data
# type's own name often contains ``Vftable`` as well. We check both
# signals because Ghidra versions vary in which one they populate.
def _looks_like_vtable_symbol_name(name: str) -> bool:
    lower = name.lower()
    return "vftable" in lower or "vtable" in lower


def _looks_like_vtable_datatype_name(dt_name: str) -> bool:
    lower = dt_name.lower()
    return "vftable" in lower or "vtable" in lower


# ---------------------------------------------------------------------------
# Ghidra metadata lookup
# ---------------------------------------------------------------------------
class GhidraMetadataLookup:
    """Address metadata lookup built from Ghidra's analysis results.

    Conforms to the ``MetadataLookup`` protocol (``lookup(addr) -> (dict, str)``).

    The returned ``dict`` carries the legacy keys (``name``, ``type``,
    ``size``, ``start_addr``, ``end_addr``, ``source``, ``library``) plus
    the v2-classifier enrichments below. All enrichment keys are always
    present; conservative defaults (``False`` / ``None``) are used when the
    Ghidra analyzers did not flag the address.

    v2 enrichment keys:
        is_plt              -- addr inside a ``.plt`` / PLT-thunk region
        is_extern_synthetic -- addr in Ghidra's external/synthetic-extern
                               namespace (Ghidra's analogue of CLE's synthetic
                               extern object)
        is_vtable           -- addr is a C++ vtable slot (RTTI-analyzer output)
        is_string           -- addr is the start of a typed string Data
        string_encoding     -- Python codec name for the string ("ascii",
                               "utf-8", "utf-16-le", ...) or None
        string_bytes        -- the raw string bytes or None
        is_jump_table_slot  -- addr is inside a Ghidra-recovered switch-table
                               (AddressTable / pointer-array referenced by a
                               computed-jump)
        is_code_ptr_table_slot
                            -- addr is inside ``.init_array`` / ``.fini_array``
                               / ``.dtors`` / ``.ctors`` / ``.preinit_array``
        tls                 -- addr is in ``.tdata`` / ``.tbss``

    Cross-provider parity: matches the keys the angr-side lookup will
    populate (Phase 1.B.2). When Ghidra has no signal the field is False/None
    just as on the angr side; the v2 classifier then routes to a lower
    precedence step (see ``tokenizer/disasm/precedence.md`` and
    ``tokenizer/disasm/angr_limitations.md``).
    """

    def __init__(self, program: Any, function_manager: Any) -> None:
        self._program = program
        self._fm = function_manager
        self._memory = program.getMemory()
        self._symbol_table = program.getSymbolTable()
        self._listing = program.getListing()

    # -- Enrichment helpers (one concern each) ------------------------------
    def _block_at(self, addr_obj: Any) -> Any:
        """Return the ``MemoryBlock`` containing ``addr_obj`` or None."""
        try:
            return self._memory.getBlock(addr_obj)
        except Exception:
            return None

    def _data_at(self, addr_obj: Any) -> Any:
        """Return the ``Data`` defined at ``addr_obj`` (exact start) or None.

        ``getDataAt`` returns None when the address is not the start of a
        Data unit; ``getDataContaining`` returns the Data whose body covers
        the address. We try exact first (matches v2's "addr is a slot start"
        semantics for jump tables / vtables); callers needing containment
        can ask separately.
        """
        try:
            return self._listing.getDataAt(addr_obj)
        except Exception:
            return None

    def _data_containing(self, addr_obj: Any) -> Any:
        """Return the ``Data`` whose body covers ``addr_obj`` or None."""
        try:
            return self._listing.getDataContaining(addr_obj)
        except Exception:
            return None

    def _classify_string(self, addr_obj: Any) -> tuple[bool, Optional[str], Optional[bytes]]:
        """Decide whether ``addr_obj`` starts a typed string Data.

        Returns ``(is_string, encoding, bytes)``. When the address is in
        the middle of a string Data (substring access) we still report
        ``is_string=True`` and return the *containing* string's bytes;
        the classifier consumes ``start_addr`` from the Data's min-address
        if it needs the offset.
        """
        data = self._data_at(addr_obj) or self._data_containing(addr_obj)
        if data is None:
            return False, None, None
        try:
            if not data.hasStringValue():
                return False, None, None
        except Exception:
            return False, None, None
        try:
            dt = data.getDataType()
        except Exception:
            dt = None
        encoding = _encoding_from_string_datatype(dt)
        try:
            raw = data.getBytes()
        except Exception:
            raw = None
        return True, encoding, _java_bytes_to_python(raw)

    def _is_vtable(self, addr_obj: Any, block: Any) -> bool:
        """Decide whether ``addr_obj`` is a C++ vtable slot.

        Two signals (either triggers): a symbol at the address whose
        name contains ``vftable``/``vtable``, OR a Data object whose
        DataType name contains the same substring. Only meaningful for
        rodata-flavored sections (``.rodata`` / ``.data.rel.ro``).
        """
        if block is None:
            return False
        if not _is_rodata_block_name(str(block.getName())):
            return False
        # Symbol-name signal — exact match at this address.
        try:
            symbols = self._symbol_table.getSymbols(addr_obj)
        except Exception:
            symbols = ()
        for sym in symbols or ():
            try:
                if _looks_like_vtable_symbol_name(str(sym.getName())):
                    return True
            except Exception:
                continue
        # DataType-name signal — Ghidra's GccRttiAnalyzer stamps the Data.
        data = self._data_containing(addr_obj)
        if data is not None:
            try:
                dt = data.getDataType()
                if dt is not None and _looks_like_vtable_datatype_name(str(dt.getName())):
                    return True
            except Exception:
                pass
        return False

    def _is_jump_table_slot(self, addr_obj: Any, block: Any) -> bool:
        """Decide whether ``addr_obj`` is a slot in a Ghidra-recovered switch table.

        First-pass implementation: check that the Data at/around the
        address is a pointer-array element AND the containing block
        is rodata-flavored. A precise cross-check against computed-jump
        inbound references is feasible but expensive (per-address ref
        walk); the function-level ``iter_switch_tables`` consumer below
        does that more cheaply by walking from the dispatch site.
        """
        if block is None:
            return False
        if not _is_rodata_block_name(str(block.getName())):
            return False
        data = self._data_at(addr_obj) or self._data_containing(addr_obj)
        if data is None:
            return False
        try:
            dt = data.getDataType()
        except Exception:
            return False
        if dt is None:
            return False
        # Pointer or Array-of-Pointer is the canonical jump-table shape.
        # We check by class name to avoid a hard import of Ghidra DataType
        # classes here; the type hierarchy in Ghidra has ``Pointer`` and
        # ``Array`` as stable interface names.
        try:
            from ghidra.program.model.data import Array, Pointer

            if isinstance(dt, Pointer):
                return True
            if isinstance(dt, Array):
                try:
                    inner = dt.getDataType()
                    return isinstance(inner, Pointer)
                except Exception:
                    return False
        except Exception:
            # Fallback if the Ghidra import fails for any reason
            tname = str(dt.getName()).lower()
            if "pointer" in tname or tname.endswith("*"):
                return True
        return False

    def _is_extern_synthetic(self, func: Any, block: Any) -> bool:
        """Detect Ghidra's analogue of CLE's synthetic-extern object.

        Ghidra represents unresolved imports through an "EXTERNAL" memory
        block (``isExternalBlock``) plus the Function's ``isExternal()``
        flag. When the address lives in that block, the v2 classifier
        treats it the same way CLE's externs are treated.
        """
        if block is not None:
            try:
                if block.isExternalBlock():
                    return True
            except Exception:
                pass
            try:
                # Older Ghidra exposes the same concept through the block name.
                if str(block.getName()).upper() == "EXTERNAL":
                    return True
            except Exception:
                pass
        if func is not None:
            try:
                if func.isExternal():
                    return True
            except Exception:
                pass
        return False

    # -- Public API ---------------------------------------------------------
    def lookup(self, addr: int) -> tuple[dict, str]:
        addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
        block = self._block_at(addr_obj)
        block_name = str(block.getName()) if block is not None else ""

        # Enrichments computed once; reused across all classification branches.
        is_plt = _is_plt_block_name(block_name) if block is not None else False
        is_code_ptr_table_slot = _is_code_ptr_table_block_name(block_name) if block is not None else False
        tls = _is_tls_block_name(block_name) if block is not None else False
        is_string, string_encoding, string_bytes = self._classify_string(addr_obj)
        is_vtable = self._is_vtable(addr_obj, block)
        is_jump_table_slot = self._is_jump_table_slot(addr_obj, block)

        def _enrich(meta: dict, *, func: Any = None) -> dict:
            """Stamp the v2 enrichment keys onto an existing meta dict.

            ``func`` is optional; when present it is forwarded to the
            extern-synthetic check (a function-entry address that lives
            in Ghidra's EXTERNAL block).
            """
            meta["is_plt"] = bool(is_plt) or (func is not None and bool(getattr(func, "isThunk", lambda: False)()))
            meta["is_extern_synthetic"] = self._is_extern_synthetic(func, block)
            meta["is_vtable"] = bool(is_vtable)
            meta["is_string"] = bool(is_string)
            meta["string_encoding"] = string_encoding
            meta["string_bytes"] = string_bytes
            meta["is_jump_table_slot"] = bool(is_jump_table_slot)
            meta["is_code_ptr_table_slot"] = bool(is_code_ptr_table_slot)
            meta["tls"] = bool(tls)
            return meta

        # 1. Exact symbol match -- legacy bare ``"symbol"`` is replaced with
        #    a section-derived type. The symbol's name is preserved as-is.
        symbols = self._symbol_table.getSymbols(addr_obj)
        if symbols:
            sym = symbols[0]
            sym_name = str(sym.getName())
            # Derive type from the containing section (if any). For a symbol
            # in ``.text`` whose Ghidra ``SymbolType`` is ``FUNCTION`` we
            # tag it ``local_function``; everything else uses the section's
            # natural type (``rodata`` / ``data`` / ``thread_local_data`` / ...).
            sym_type = "unknown"
            if block is not None:
                base_type = _section_type_from_block(block)
                if base_type == "code":
                    is_function_symbol = False
                    try:
                        st = sym.getSymbolType()
                        # ``SymbolType.FUNCTION`` is the canonical enum value;
                        # compare by name to avoid importing the enum class.
                        is_function_symbol = str(st).upper() == "FUNCTION"
                    except Exception:
                        is_function_symbol = False
                    sym_type = "local_function" if is_function_symbol else base_type
                else:
                    sym_type = base_type
            meta = {
                "name": sym_name,
                "type": sym_type,
                "size": 0,
                "source": "symbol",
                "start_addr": addr,
                "end_addr": addr,
                "library": "unknown",
            }
            # If the symbol is an actual function and we have its body, give
            # the classifier accurate range bounds rather than a zero-width.
            try:
                func = self._fm.getFunctionAt(addr_obj)
            except Exception:
                func = None
            if func is not None:
                try:
                    body = func.getBody()
                    size = int(body.getNumAddresses())
                    entry = int(func.getEntryPoint().getOffset())
                    meta["size"] = size
                    meta["start_addr"] = entry
                    meta["end_addr"] = entry + size
                except Exception:
                    pass
            return _enrich(meta, func=func), "exact"

        # 2. Function match (covers calls/jumps into known functions).
        func = self._fm.getFunctionContaining(addr_obj)
        if func is not None:
            entry = int(func.getEntryPoint().getOffset())
            body = func.getBody()
            size = int(body.getNumAddresses())
            is_external = func.isExternal() or func.isThunk()
            meta = {
                "name": str(func.getName()),
                "type": "library_function" if is_external else "local_function",
                "size": size,
                "start_addr": entry,
                "end_addr": entry + size,
                "source": "function",
                "library": "unknown",
            }
            return _enrich(meta, func=func), "range"

        # 3. Memory-block match -- the address is in a known section but
        #    not inside any function. Use the section-type mapping.
        if block is not None:
            meta = {
                "name": block_name,
                "type": _section_type_from_block(block),
                "size": int(block.getSize()),
                "start_addr": int(block.getStart().getOffset()),
                "end_addr": int(block.getEnd().getOffset()) + 1,
                "source": "section",
                "library": "unknown",
            }
            return _enrich(meta), "range"

        # 4. Fallback -- address not in any known memory region.
        fallback = {
            "start_addr": addr,
            "end_addr": addr,
            "name": f"unknown_{addr:x}",
            "type": "unknown",
            "size": 0,
            "source": "synthetic",
            "library": "unknown",
        }
        return _enrich(fallback), "synthetic"


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

    def as_dict(self) -> dict[int, str]:
        """Return id->name dict for _CapInstruction.reg_name()."""
        return dict(self._id_to_name)


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
    # Conditional jumps — Ghidra form -> Capstone canonical
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


def _compute_fp_width_bytes(ghidra_insn: Any, operand_index: int) -> Optional[int]:
    """Module-level helper backing ``operand_fp_width_bytes``.

    Pulled out so the decode path in ``_ghidra_insn_to_cap`` can call it
    once per operand to stamp the resulting width on ``_CapOperand``,
    keeping the public ``GhidraDisassemblyProvider.operand_fp_width_bytes``
    method as a thin wrapper. Returns one of {2, 4, 8, 10, 16} when the
    operand is FP-typed (Ghidra ``OperandType.FLOAT`` bitmask) or ``None``
    otherwise. See ``GhidraDisassemblyProvider.operand_fp_width_bytes``
    docstring for the full derivation order and the BFloat16/Float16
    ambiguity note.
    """
    from ghidra.program.model.lang import OperandType, Register
    from ghidra.program.model.scalar import Scalar

    try:
        op_type = ghidra_insn.getOperandType(operand_index)
    except Exception:
        return None
    if not bool(op_type & OperandType.FLOAT):
        return None

    max_width_bits = 0
    try:
        objects = ghidra_insn.getOpObjects(operand_index)
    except Exception:
        objects = ()
    for obj in objects or ():
        try:
            if isinstance(obj, Register):
                width = int(obj.getBitLength())
            elif isinstance(obj, Scalar):
                width = int(obj.bitLength())
            else:
                continue
        except Exception:
            continue
        if width > max_width_bits:
            max_width_bits = width

    if max_width_bits == 0:
        # Fall back to the reference-type's reported access size
        # (memory FP loads/stores).
        try:
            ref_type = ghidra_insn.getOperandRefType(operand_index)
            if ref_type is not None:
                size_bytes = int(ref_type.getSize())
                if size_bytes > 0:
                    return size_bytes
        except Exception:
            pass
        return None

    width_bytes = max_width_bits // 8
    return width_bytes if width_bytes > 0 else None


def _ghidra_insn_to_cap(
    ghidra_insn: Any,
    reg_map: _RegisterMap,
    program: Any,
) -> _CapInstruction:
    """Translate a single Ghidra Instruction into a _CapInstruction."""
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import OperandType, Register
    from ghidra.program.model.scalar import Scalar

    # -- Mnemonic / prefix handling -------------------------------------------
    raw_mnemonic = str(ghidra_insn.getMnemonicString())
    base_mnemonic, suffix_prefix_name, suffix_prefix_byte = _split_ghidra_mnemonic(raw_mnemonic)
    base_mnemonic = _GHIDRA_MNEMONIC_ALIASES.get(base_mnemonic, base_mnemonic)

    prefix_set = _extract_x86_prefixes(ghidra_insn)
    if suffix_prefix_byte is not None:
        prefix_set.add(suffix_prefix_byte)
    prefix_bytes = bytes(sorted(prefix_set))

    # Capstone-compatible mnemonic: X86Provider checks mnemonic.startswith("repe") etc.
    if suffix_prefix_name is not None:
        mnemonic = f"{suffix_prefix_name} {base_mnemonic}"
    else:
        mnemonic = base_mnemonic

    # -- Operand handling -----------------------------------------------------
    num_ops = ghidra_insn.getNumOperands()
    op_strs: list[str] = []
    operands: list[_CapOperand] = []

    for i in range(num_ops):
        op_repr = str(ghidra_insn.getDefaultOperandRepresentation(i))
        op_strs.append(op_repr)
        objects = ghidra_insn.getOpObjects(i)

        if not objects:
            continue

        # Detect memory operands via OperandType bitmask (no string parsing).
        # DYNAMIC: register-based memory (e.g. [RSP + 0x10], FS:[0x28])
        # INDIRECT: indirect memory reference (e.g. jmp [0x301f28])
        # SCALAR|ADDRESS w/o REGISTER|CODE: absolute memory in LEA (e.g. [0x10168c])
        op_type = ghidra_insn.getOperandType(i)
        is_memory = (
            bool(op_type & OperandType.DYNAMIC)
            or bool(op_type & OperandType.INDIRECT)
            or (
                bool(op_type & OperandType.ADDRESS)
                and bool(op_type & OperandType.SCALAR)
                and not (op_type & (OperandType.REGISTER | OperandType.CODE))
            )
        )

        # Per-operand FP width (Ghidra OperandType.FLOAT bitmask). Stamped on
        # every _CapOperand we build below so the operand tokenizer can pass
        # it to ``ConstantHandler.process_constant_v2`` without re-deriving
        # the signal at call-site granularity. See ``precedence.md`` step 1
        # for how the classifier consumes it.
        fp_width = _compute_fp_width_bytes(ghidra_insn, i)

        if is_memory:
            # Attach raw Ghidra objects for native tokenization in X86GhidraProvider
            raw_data = _GhidraMemRawData(
                ghidra_insn=ghidra_insn,
                op_objects=list(objects),
                reg_map=reg_map,
            )
            operands.append(_CapOperand(type=_OP_MEM, ghidra_raw_data=raw_data, fp_width_bytes=fp_width))
        else:
            first = objects[0]
            if isinstance(first, Register):
                reg_id = reg_map.get_id(str(first.getName()))
                operands.append(_CapOperand(type=_OP_REG, reg=reg_id, fp_width_bytes=fp_width))
            elif isinstance(first, Scalar):
                operands.append(_CapOperand(type=_OP_IMM, imm=int(first.getValue()), fp_width_bytes=fp_width))
            elif isinstance(first, Address):
                operands.append(_CapOperand(type=_OP_IMM, imm=int(first.getOffset()), fp_width_bytes=fp_width))

    insn_inner = _CapInsnInner(_insn_name=base_mnemonic)

    return _CapInstruction(
        mnemonic=mnemonic,
        op_str=", ".join(op_strs),
        operands=operands,
        prefix=prefix_bytes,
        insn_inner=insn_inner,
        reg_map=reg_map.as_dict(),
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GhidraDisassemblyProvider(DisassemblyProvider):
    """Disassembly provider backed by Ghidra via pyghidra (headless, no IPC).

    Usage::

        provider = GhidraDisassemblyProvider(Path("binary"))
        provider.build_cfg()   # runs Ghidra auto-analysis
        for addr, name, func in provider.iter_functions():
            ...
    """

    def __init__(self, binary_path: Path) -> None:
        import jpype.config
        import pyghidra

        # Prevent JVM from hanging on Python exit (known JPype issue)
        jpype.config.destroy_jvm = False

        if not pyghidra.started():
            pyghidra.start()

        self.binary_path = binary_path

        # pyghidra defaults `project_location` to the binary's parent
        # directory, which is the read-only `--source-already-staged`
        # bind-mount in the SLURM container. Redirect into the
        # writable ephemeral root (`/app/out-tmp` if the container
        # provides it, else `/tmp`). Per-binary subdir keyed off the
        # binary's parent dir name (unique per variant in our corpus
        # layout) so concurrent variants don't collide on the
        # `<binary-name>_ghidra` project name pyghidra derives.
        out_tmp_root = Path("/app/out-tmp")
        project_root = out_tmp_root if out_tmp_root.is_dir() else Path("/tmp")
        project_location = project_root / "ghidra-projects" / binary_path.parent.name
        project_location.mkdir(parents=True, exist_ok=True)
        self._project_location = project_location

        # open_program is the simplest API: import, auto-analyze, return
        # a FlatProgramAPI context manager.  We defer analysis (analyze=False)
        # so build_cfg() controls when it happens.
        self._ctx = pyghidra.open_program(
            binary_path,
            project_location=project_location,
            analyze=False,
        )
        self._flat_api = self._ctx.__enter__()
        self._program = self._flat_api.getCurrentProgram()
        self._fm = self._program.getFunctionManager()
        self._listing = self._program.getListing()
        self._memory = self._program.getMemory()
        self._reg_map = _RegisterMap(self._program)
        self._analyzed = False
        self._closed = False

    def close(self) -> None:
        """Unload the current Program + close the Project so the
        next task in this worker (worker processes are reused
        unless `always_restart_worker=True`) can `open_program`
        again without tripping ``GhidraScriptUtil initialized
        multiple times`` or accumulating analysis threads. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        # `pyghidra.open_program` returns a generator-based context
        # manager. Driving its `__exit__` is the documented way to
        # release the imported Program, the in-memory project, and
        # the analysis-thread pool. Errors on close (e.g. if a
        # prior failure left Ghidra in a half-state) must not
        # mask the original task's exception.
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:
            pass
        # Drop Java-object references so the next provider in this
        # worker can acquire fresh ones without lingering CFG/listing
        # objects pinning the previous program.
        self._flat_api = None
        self._program = None
        self._fm = None
        self._listing = None
        self._memory = None
        self._reg_map = None
        self._ctx = None

    def build_cfg(self) -> None:
        """Run Ghidra's auto-analysis (the equivalent of CFGFast)."""
        self._flat_api.analyzeAll(self._program)
        self._analyzed = True

    def get_text_section_bounds(self) -> tuple[int, int]:
        for block in self._memory.getBlocks():
            name = str(block.getName())
            if name == ".text":
                start = int(block.getStart().getOffset())
                return start, start + int(block.getSize())
        return 0, 0

    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]:
        if sections is None:
            sections = [".rodata"]

        all_entries: list[dict[str, str]] = []
        addr_dict: dict[str, list[str]] = {}

        for block in self._memory.getBlocks():
            name = str(block.getName())
            if name not in sections:
                continue
            if name == ".rodata" and block.isInitialized() and not block.isWrite():
                size = int(block.getSize())
                if size <= 0:
                    continue
                start_addr = block.getStart()
                buf = bytearray(size)
                block.getBytes(start_addr, buf)
                data = bytes(buf)
                base_addr = int(start_addr.getOffset())

                for match in re.finditer(b"[\x20-\x7e]{4,}\x00", data):
                    s = match.group().rstrip(b"\x00").decode("utf-8", errors="ignore")
                    start = base_addr + match.start()
                    entry = {
                        "section": ".rodata",
                        "start": hex(start),
                        "end": hex(start + len(s) + 1),
                        "value": f'"{s}"',
                    }
                    all_entries.append(entry)
                    addr_dict[entry["start"]] = [entry["end"], entry["section"], entry["value"]]

        if output_csv_path:
            csv_path = Path(output_csv_path)
            consts_path = csv_path.parent / f"{csv_path.stem.replace('_output', '')}_consts.txt"
        else:
            consts_path = Path("parsed_constants.txt")

        with open(consts_path, "w") as f:
            for e in all_entries:
                f.write(f"{e['start']} - {e['end']}: {e['section']}: {e['value']}\n")

        print(f"Parsed {len(all_entries)} .rodata constants with exact addresses into {consts_path}")
        return addr_dict

    def create_metadata_lookup(self) -> MetadataLookup:
        return GhidraMetadataLookup(self._program, self._fm)

    def function_count(self) -> int:
        return int(self._fm.getFunctionCount())

    def iter_functions(self) -> Iterable[tuple[int, str, _CapFunction]]:
        """Iterate functions, yielding Capstone-compatible adapter objects.

        Each yielded ``func`` has a ``.blocks`` attribute containing
        ``_CapBlock`` objects whose ``.capstone.insns`` are
        ``_CapInstruction`` adapters — the same shape the tokenizer expects.
        """
        assert self._analyzed, "Analysis not run — call build_cfg() first"

        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import TaskMonitor

        block_model = SimpleBlockModel(self._program)
        monitor = TaskMonitor.DUMMY

        funcs = []
        for func in self._fm.getFunctions(True):
            name = str(func.getName())
            addr = int(func.getEntryPoint().getOffset())
            funcs.append((addr, name, func))

        for addr, name, ghidra_func in sorted(funcs, key=lambda t: t[1]):
            body = ghidra_func.getBody()
            block_iter = block_model.getCodeBlocksContaining(body, monitor)

            cap_blocks: list[_CapBlock] = []
            while block_iter.hasNext():
                gblock = block_iter.next()
                block_start = int(gblock.getMinAddress().getOffset())
                block_size = int(gblock.getMaxAddress().getOffset()) - block_start + 1

                insns: list[_CapInstruction] = []
                insn_iter = self._listing.getInstructions(gblock, True)
                while insn_iter.hasNext():
                    ghidra_insn = insn_iter.next()
                    insn_addr = int(ghidra_insn.getAddress().getOffset())
                    if not body.contains(ghidra_insn.getAddress()):
                        continue
                    cap_insn = _ghidra_insn_to_cap(ghidra_insn, self._reg_map, self._program)
                    insns.append(cap_insn)

                cb = _CapBlock(addr=block_start, size=block_size)
                cb.set_insns(insns)
                cap_blocks.append(cb)

            if not cap_blocks:
                continue

            yield addr, name, _CapFunction(_blocks=cap_blocks, _raw=ghidra_func)

    # ----------------------------------------------------------------------
    # Per-operand FP-immediate detection
    # ----------------------------------------------------------------------
    def operand_fp_width_bytes(self, ghidra_insn: Any, operand_index: int) -> Optional[int]:
        """Return the FP width in bytes for ``operand_index`` of ``ghidra_insn``.

        Returns one of {2, 4, 8, 10, 16} when the operand is FP-typed
        (per Ghidra's ``OperandType.FLOAT`` bitmask) and None otherwise.
        Width derivation order:

        1. Inspect each ``getOpObjects(i)`` element. For ``Register``
           operands, return ``Register.getBitLength() / 8``. For ``Scalar``
           operands, return ``Scalar.bitLength() / 8``. Take the largest
           value seen (x87 ``fld dword ptr [...]`` carries an FP-tagged
           memory operand whose size is the load size).
        2. If no op-object width is available, fall back to
           ``ghidra_insn.getOperandRefType(i).getSize()``.
        3. If neither yields a positive value, return ``None`` (the
           classifier then routes this operand through step 11 of the
           precedence list rather than emitting a malformed ``floatXX``).

        BFloat16-vs-Float16 distinction: width 2 alone is ambiguous;
        SLEIGH does not currently tag bfloat16 distinctly. Callers
        receiving width=2 default to ``Float16`` in the classifier.
        Documented here so the precedence-list implementation can
        cite the limitation.

        Implementation note: the decode path in ``_ghidra_insn_to_cap``
        stamps this width on every ``_CapOperand.fp_width_bytes`` at
        instruction-translation time, so operand tokenizers don't need
        to call this method per-operand. The public method stays on the
        provider for direct callers and for symmetry with the
        Phase 1.B.1 provider-interface contract.
        """
        return _compute_fp_width_bytes(ghidra_insn, operand_index)

    # ----------------------------------------------------------------------
    # Switch-table recovery
    # ----------------------------------------------------------------------
    def iter_switch_tables(self, function: Any) -> Iterable[tuple[int, list[int]]]:
        """Yield ``(jump_table_addr, [target_block_addrs])`` for ``function``.

        Walks every instruction in the function's body; for any computed-jump
        instruction, follows the ``COMPUTED_JUMP`` references to gather the
        list of resolved targets, and locates the backing pointer-array data
        in rodata (the table's address) via the instruction's outbound READ
        references when present. Tables are yielded in dispatch-instruction
        order; duplicate-table elision is the consumer's responsibility.

        Phase-1 scope: pragmatic recovery from Ghidra's reference graph.
        A processor-specific ``SwitchAnalyzer`` cross-check is a Phase 2
        refinement (Phase 2.C.1 jump-table-analysis pass).
        """
        if function is None:
            return

        # ``fill_constant_candidates`` hands us the capstone-compat wrapper
        # produced by ``iter_functions``; unwrap to the underlying Ghidra
        # ``Function`` so we can drive the Ghidra reference graph below.
        if isinstance(function, _CapFunction):
            function = function._raw
        if function is None:
            return

        from ghidra.program.model.symbol import RefType

        body = function.getBody()
        insn_iter = self._listing.getInstructions(body, True)
        while insn_iter.hasNext():
            insn = insn_iter.next()
            try:
                flow_type = insn.getFlowType()
                if not (flow_type.isJump() and flow_type.isComputed()):
                    continue
            except Exception:
                continue

            # Outbound references from the dispatch instruction. Computed
            # jumps surface their resolved targets as COMPUTED_JUMP refs.
            try:
                refs_from = list(insn.getReferencesFrom() or ())
            except Exception:
                refs_from = []

            targets: list[int] = []
            table_addr: Optional[int] = None
            for ref in refs_from:
                try:
                    rtype = ref.getReferenceType()
                except Exception:
                    continue
                # READ references typically point at the table base in rodata.
                try:
                    if rtype.isData() and rtype.isRead():
                        to_addr = int(ref.getToAddress().getOffset())
                        if table_addr is None:
                            table_addr = to_addr
                        continue
                except Exception:
                    pass
                # COMPUTED_JUMP / CONDITIONAL_COMPUTED_JUMP / COMPUTED_CALL
                # references list the resolved target blocks.
                try:
                    if rtype.isJump() and rtype.isComputed():
                        to_addr = int(ref.getToAddress().getOffset())
                        targets.append(to_addr)
                        continue
                except Exception:
                    pass
                # ``RefType.COMPUTED_JUMP`` direct comparison as a fallback
                # for older Ghidra versions where the predicates above
                # are missing.
                try:
                    if rtype == RefType.COMPUTED_JUMP:
                        to_addr = int(ref.getToAddress().getOffset())
                        targets.append(to_addr)
                except Exception:
                    pass

            if not targets:
                continue

            # If we did not find a READ reference, fall back to scanning
            # the instruction's data-typed operand for a memory-base address.
            if table_addr is None:
                try:
                    num_ops = insn.getNumOperands()
                except Exception:
                    num_ops = 0
                for i in range(num_ops):
                    try:
                        for obj in insn.getOpObjects(i) or ():
                            # ghidra.program.model.address.Address has getOffset()
                            if hasattr(obj, "getOffset"):
                                table_addr = int(obj.getOffset())
                                break
                    except Exception:
                        continue
                    if table_addr is not None:
                        break

            if table_addr is None:
                # No locatable table base; skip rather than emit a synthetic.
                continue

            yield table_addr, targets
