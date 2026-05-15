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
from pathlib import Path
from typing import Any, Iterable, Optional

from tokenizer.disasm import DisassemblyProvider, MetadataLookup
from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
    address_kind_from_string,
    encoding_from_string,
    section_kind_from_type_string,
)
from tokenizer.disasm.types import Architecture, FpType, OperandKind

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
# Ghidra metadata lookup + typed view
# ---------------------------------------------------------------------------
class _GhidraAddressMetadataView:
    """Concrete typed view returned by ``GhidraMetadataLookup.lookup()``.

    Pure storage + read-only typed properties. ``GhidraMetadataLookup``
    populates every typed slot at lookup time via ``_populate``;
    consumers read typed properties exclusively. ``slot_target`` is
    resolved lazily on first access by re-entering
    ``GhidraMetadataLookup._classify_address`` with
    ``allow_slot_recursion=False`` (so the inner call writes to a
    fresh view rather than disturbing the cursor and the inner view's
    ``slot_target`` is guaranteed ``None``).

    LIFECYCLE: instance is REUSED across ``lookup()`` calls. Use
    ``copy.deepcopy(view)`` to stash across lookups.
    """

    __slots__ = (
        "_lookup",
        "_kind",
        "_section_kind",
        "_section_name",
        "_string_encoding",
        "_string_bytes",
        "_name",
        "_start_addr",
        "_end_addr",
        "_size",
        "_library",
        "_is_vtable",
        "_tls",
        "_slot_target_addr",
        "_jump_table_base_addr",
        "_jump_table_offset",
    )

    def __init__(self, lookup: "GhidraMetadataLookup") -> None:
        self._lookup = lookup
        self._kind: AddressKind = AddressKind.NONE
        self._section_kind: SectionKind = SectionKind.UNKNOWN
        self._section_name: Optional[str] = None
        self._string_encoding: Encoding = Encoding.UNKNOWN
        self._string_bytes: Optional[bytes] = None
        self._name: Optional[str] = None
        self._start_addr: Optional[int] = None
        self._end_addr: Optional[int] = None
        self._size: Optional[int] = None
        self._library: Optional[str] = None
        self._is_vtable: bool = False
        self._tls: bool = False
        # Slot-target resolution state: populated alongside the typed
        # slots below for slot-bearing addresses; ``slot_target``
        # property reads ``_slot_target_addr`` and re-classifies
        # lazily on first access.
        self._slot_target_addr: Optional[int] = None
        self._jump_table_base_addr: Optional[int] = None
        self._jump_table_offset: Optional[int] = None

    def _populate(
        self,
        *,
        kind: AddressKind,
        section_kind: SectionKind,
        section_name: Optional[str],
        string_encoding: Encoding,
        string_bytes: Optional[bytes],
        name: Optional[str],
        start_addr: Optional[int],
        end_addr: Optional[int],
        size: Optional[int],
        library: Optional[str],
        is_vtable: bool,
        tls: bool,
        slot_target_addr: Optional[int],
        jump_table_base_addr: Optional[int],
        jump_table_offset: Optional[int],
    ) -> None:
        """Replace all typed slot state in one call. Used by the lookup
        at the start of every ``lookup()`` so the consumer sees a
        consistent view bound to the current address.
        """
        self._kind = kind
        self._section_kind = section_kind
        self._section_name = section_name
        self._string_encoding = string_encoding
        self._string_bytes = string_bytes
        self._name = name
        self._start_addr = start_addr
        self._end_addr = end_addr
        self._size = size
        self._library = library
        self._is_vtable = is_vtable
        self._tls = tls
        self._slot_target_addr = slot_target_addr
        self._jump_table_base_addr = jump_table_base_addr
        self._jump_table_offset = jump_table_offset

    # -- Typed property surface (AddressMetadataView Protocol) --------------
    @property
    def kind(self) -> AddressKind:
        return self._kind

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def section_kind(self) -> SectionKind:
        return self._section_kind

    @property
    def section_name(self) -> Optional[str]:
        return self._section_name

    @property
    def start_addr(self) -> Optional[int]:
        return self._start_addr

    @property
    def end_addr(self) -> Optional[int]:
        return self._end_addr

    @property
    def size(self) -> Optional[int]:
        return self._size

    @property
    def library(self) -> Optional[str]:
        return self._library

    @property
    def string_encoding(self) -> Encoding:
        return self._string_encoding

    @property
    def string_bytes(self) -> Optional[bytes]:
        return self._string_bytes

    @property
    def is_vtable(self) -> bool:
        return self._is_vtable

    @property
    def tls(self) -> bool:
        return self._tls

    @property
    def slot_target(self) -> Optional[AddressMetadataView]:
        if self._slot_target_addr is None:
            return None
        # Recursion-bounded re-classification: the lookup builds a
        # FRESH (non-cursor) view for the target so the kind is
        # guaranteed not to be a slot kind itself, and the active
        # cursor view's state is not disturbed.
        return self._lookup._classify_address(
            int(self._slot_target_addr), allow_slot_recursion=False
        )

    @property
    def jump_table_base_addr(self) -> Optional[int]:
        return self._jump_table_base_addr

    @property
    def jump_table_offset(self) -> Optional[int]:
        return self._jump_table_offset

    def __deepcopy__(self, memo) -> "_GhidraAddressMetadataView":
        clone = _GhidraAddressMetadataView(self._lookup)
        clone._kind = self._kind
        clone._section_kind = self._section_kind
        clone._section_name = self._section_name
        clone._string_encoding = self._string_encoding
        clone._string_bytes = self._string_bytes
        clone._name = self._name
        clone._start_addr = self._start_addr
        clone._end_addr = self._end_addr
        clone._size = self._size
        clone._library = self._library
        clone._is_vtable = self._is_vtable
        clone._tls = self._tls
        clone._slot_target_addr = self._slot_target_addr
        clone._jump_table_base_addr = self._jump_table_base_addr
        clone._jump_table_offset = self._jump_table_offset
        return clone


class GhidraMetadataLookup:
    """Address metadata lookup built from Ghidra's analysis results.

    Conforms to the ``MetadataLookup`` protocol (``lookup(addr) ->
    AddressMetadataView``). The returned ``_GhidraAddressMetadataView`` is
    REUSED across ``lookup()`` calls (lifecycle per ``AddressMetadataView``
    docstring); valid only until the next ``lookup()``.

    Per ``lookup()`` the lookup walks Ghidra's symbol/function/memory
    state, derives every typed slot the view exposes, and stamps them in
    one ``_populate`` call. The string -> enum mapping happens here at
    population time, never on the read path.

    Typed slots populated per address:
        kind                -- ``AddressKind`` (precedence-aware: STRING >
                               vtable / code_ptr / jump_table-as-slot >
                               PLT > extern_synthetic > base type)
        section_kind        -- ``SectionKind`` derived from the containing
                               ``MemoryBlock``'s name + permission bits
        section_name        -- raw ELF section name (``.rodata`` etc.) of
                               the containing ``MemoryBlock`` (separate
                               from ``name``, which holds the function /
                               symbol label)
        name                -- function / symbol / section label
        start_addr/end_addr -- range bounds for the matched entity
        size                -- entity size in bytes
        library             -- non-None only for resolved extern targets
        string_encoding     -- ``Encoding`` enum from the Ghidra DataType
        string_bytes        -- raw string bytes when ``kind == STRING``
        is_vtable           -- C++ vtable slot (RTTI-analyzer output)
        tls                 -- TLS section (``.tdata`` / ``.tbss``)
        slot_target         -- lazily resolved on first read; uses
                               ``_slot_target_addr`` populated here
        jump_table_base_addr / jump_table_offset
                            -- populated for ``JUMP_TABLE_SLOT`` addresses

    Cross-provider parity: ``AngrMetadataLookup`` exposes the same typed
    slots; angr-only lookups leave Ghidra-only signals at their
    conservative defaults (``Encoding.UNKNOWN`` for non-ASCII strings,
    ``slot_target=None``, no jump-table info; see
    ``tokenizer/disasm/angr_limitations.md``).
    """

    def __init__(self, program: Any, function_manager: Any) -> None:
        self._program = program
        self._fm = function_manager
        self._memory = program.getMemory()
        self._symbol_table = program.getSymbolTable()
        self._listing = program.getListing()
        # Lazy switch-table cache: built on first JUMP_TABLE_SLOT slot_target
        # access. Maps table-base-addr -> list of resolved target block
        # addresses (slot order). Populated by walking every function once
        # via ``GhidraDisassemblyProvider.iter_switch_tables``.
        self._switch_table_cache: Optional[dict[int, list[int]]] = None
        # REUSED view wrapper - per-`lookup()` call we mutate its state
        # via `_populate` and return the same instance.
        self._view = _GhidraAddressMetadataView(self)

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
    def lookup(self, addr: int) -> AddressMetadataView:
        """Resolve ``addr`` to the typed cursor view.

        Mutates the per-lookup ``_GhidraAddressMetadataView`` and returns
        it. Consumers read typed properties exclusively (``meta.kind``,
        ``meta.name``, ``meta.string_encoding`` etc.).
        """
        return self._classify_address(addr, allow_slot_recursion=True)

    def _classify_address(
        self,
        addr: int,
        *,
        allow_slot_recursion: bool,
    ) -> AddressMetadataView:
        """Classify ``addr`` and return a typed view.

        ``allow_slot_recursion`` controls the active wrapper AND the
        slot-detection branch:

        - ``True`` (public ``lookup()`` path): mutate the cursor view
          (``self._view``) in place and return it. Slot-target resolution
          is permitted - the resulting view's ``slot_target`` property
          will re-enter ``_classify_address`` with
          ``allow_slot_recursion=False`` so the inner call writes to a
          fresh wrapper instead of disturbing the cursor.
        - ``False`` (slot-target resolution path): build a FRESH
          standalone view bound to this lookup. Slot-detection signals
          are suppressed so the inner view's kind is guaranteed NOT to
          be a slot kind (CODE_PTR_TABLE_SLOT / JUMP_TABLE_SLOT) per
          task contract; ``slot_target`` is also guaranteed ``None`` so
          there is no infinite regress.
        """
        addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
        block = self._block_at(addr_obj)
        block_name = str(block.getName()) if block is not None else ""

        # Enrichments computed once; reused across all classification branches.
        is_plt = _is_plt_block_name(block_name) if block is not None else False
        is_code_ptr_table_slot = _is_code_ptr_table_block_name(block_name) if block is not None else False
        tls = _is_tls_block_name(block_name) if block is not None else False
        is_string, string_encoding_str, string_bytes = self._classify_string(addr_obj)
        # Slot-detection signals are suppressed when re-entering for slot
        # resolution: the slot_target's kind must NOT itself be a slot kind.
        if allow_slot_recursion:
            is_vtable = self._is_vtable(addr_obj, block)
            is_jump_table_slot = self._is_jump_table_slot(addr_obj, block)
        else:
            is_vtable = False
            is_jump_table_slot = False
            is_code_ptr_table_slot = False

        # Per-branch raw fields (name / type-string / size / range bounds /
        # func ref). Branches set them; the final populate-typed-state step
        # at the bottom translates string -> enum once.
        type_str: str
        name: Optional[str]
        size: Optional[int]
        start_addr: Optional[int]
        end_addr: Optional[int]
        func: Any = None
        section_name: Optional[str] = block_name if block is not None else None

        # 1. Exact symbol match -- legacy bare ``"symbol"`` is replaced with
        #    a section-derived type. The symbol's name is preserved as-is.
        symbols = self._symbol_table.getSymbols(addr_obj)
        if symbols:
            sym = symbols[0]
            name = str(sym.getName())
            # Derive type from the containing section (if any). For a symbol
            # in ``.text`` whose Ghidra ``SymbolType`` is ``FUNCTION`` we
            # tag it ``local_function``; everything else uses the section's
            # natural type (``rodata`` / ``data`` / ``thread_local_data`` / ...).
            type_str = "unknown"
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
                    type_str = "local_function" if is_function_symbol else base_type
                else:
                    type_str = base_type
            size = 0
            start_addr = addr
            end_addr = addr
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
                    start_addr = entry
                    end_addr = entry + size
                except Exception:
                    pass
        else:
            # 2. Function match (covers calls/jumps into known functions).
            func = self._fm.getFunctionContaining(addr_obj)
            if func is not None:
                entry = int(func.getEntryPoint().getOffset())
                body = func.getBody()
                size = int(body.getNumAddresses())
                is_external = func.isExternal() or func.isThunk()
                name = str(func.getName())
                type_str = "library_function" if is_external else "local_function"
                start_addr = entry
                end_addr = entry + size
            elif block is not None:
                # 3. Memory-block match -- the address is in a known section
                # but not inside any function. Use the section-type mapping.
                name = block_name
                type_str = _section_type_from_block(block)
                size = int(block.getSize())
                start_addr = int(block.getStart().getOffset())
                end_addr = int(block.getEnd().getOffset()) + 1
            else:
                # 4. Fallback -- address not in any known memory region.
                name = f"unknown_{addr:x}"
                type_str = "unknown"
                size = 0
                start_addr = addr
                end_addr = addr

        # Final enrichments that depend on ``func`` / ``block`` and apply
        # uniformly across all four branches above.
        is_thunk = func is not None and bool(getattr(func, "isThunk", lambda: False)())
        is_plt_combined = is_plt or is_thunk
        is_extern_synthetic = self._is_extern_synthetic(func, block)

        # Translate raw signals into typed slots ONCE here (no per-property
        # work on the read path).
        kind = self._derive_address_kind(
            type_str=type_str,
            is_string=is_string,
            is_vtable=is_vtable,
            is_jump_table_slot=is_jump_table_slot,
            is_code_ptr_table_slot=is_code_ptr_table_slot,
            is_plt=is_plt_combined,
            is_extern_synthetic=is_extern_synthetic,
        )
        section_kind = section_kind_from_type_string(type_str)
        string_encoding = encoding_from_string(string_encoding_str)

        # Slot-target / jump-table resolution. Only meaningful on the
        # public lookup path (``allow_slot_recursion=True``); the inner
        # slot_target re-classification path leaves these at None per
        # the task contract.
        slot_target_addr: Optional[int] = None
        jump_table_base_addr: Optional[int] = None
        jump_table_offset: Optional[int] = None
        if allow_slot_recursion:
            if is_vtable or is_code_ptr_table_slot:
                target = self._resolve_pointer_array_slot(addr)
                if target is not None:
                    slot_target_addr = target
            if is_jump_table_slot:
                jt_base = self._jump_table_base_addr(addr)
                if jt_base is not None:
                    jump_table_base_addr = jt_base
                    jump_table_offset = addr - jt_base
                    target = self._jump_table_target(jt_base, addr - jt_base)
                    if target is not None:
                        slot_target_addr = target

        # Library label: index entries currently leave it as the
        # "unknown" sentinel; surface None on the typed view so downstream
        # readers don't accidentally consume the placeholder string.
        library: Optional[str] = None

        view = self._view if allow_slot_recursion else _GhidraAddressMetadataView(self)
        view._populate(
            kind=kind,
            section_kind=section_kind,
            section_name=section_name,
            string_encoding=string_encoding,
            string_bytes=string_bytes,
            name=name,
            start_addr=start_addr,
            end_addr=end_addr,
            size=size,
            library=library,
            is_vtable=bool(is_vtable),
            tls=bool(tls),
            slot_target_addr=slot_target_addr,
            jump_table_base_addr=jump_table_base_addr,
            jump_table_offset=jump_table_offset,
        )
        return view

    @staticmethod
    def _derive_address_kind(
        *,
        type_str: str,
        is_string: bool,
        is_vtable: bool,
        is_jump_table_slot: bool,
        is_code_ptr_table_slot: bool,
        is_plt: bool,
        is_extern_synthetic: bool,
    ) -> AddressKind:
        """Combine raw signals into the precedence-aware ``AddressKind``.

        Order matches ``constant_handler._PRECEDENCE``:
            1. is_string                      -> STRING
            2. is_jump_table_slot             -> JUMP_TABLE_SLOT
            3. is_vtable / is_code_ptr_slot   -> CODE_PTR_TABLE_SLOT
            4. is_plt                         -> PLT_FUNCTION
            5. is_extern_synthetic on a func  -> EXT_FUNCTION_SYNTHETIC
            6. base type-string               -> base AddressKind
        """
        if is_string:
            return AddressKind.STRING
        if is_jump_table_slot:
            return AddressKind.JUMP_TABLE_SLOT
        if is_vtable or is_code_ptr_table_slot:
            return AddressKind.CODE_PTR_TABLE_SLOT
        if is_plt:
            return AddressKind.PLT_FUNCTION
        lowered = (type_str or "").lower()
        if is_extern_synthetic and lowered in {
            "extern_function",
            "library_function",
            "plt_function",
            "unknown_function",
        }:
            return AddressKind.EXT_FUNCTION_SYNTHETIC
        return address_kind_from_string(type_str)

    def _resolve_pointer_array_slot(self, addr: int) -> Optional[int]:
        """Read the pointer value stored at ``addr``.

        For init_array / fini_array / dtors / vtable slots, Ghidra has a
        typed ``Pointer`` ``Data`` whose ``getValue()`` returns the
        ``Address`` object pointing at the resolved target. Returns the
        target offset as a Python int, or ``None`` when no such Data is
        defined (the classifier then leaves ``slot_target`` empty and
        the consumer falls back to the bare ptr token).
        """
        try:
            addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
            data = self._listing.getDataAt(addr_obj)
            if data is None:
                return None
            value = data.getValue()
            if value is None:
                return None
            # ``Address`` exposes ``getOffset()``; ``Scalar`` exposes
            # ``getValue()``. Both are plausible Pointer payloads.
            if hasattr(value, "getOffset"):
                return int(value.getOffset())
            if hasattr(value, "getValue"):
                return int(value.getValue())
            return None
        except Exception:
            return None

    def _jump_table_base_addr(self, addr: int) -> Optional[int]:
        """Return the start address of the ``Data`` array containing ``addr``.

        For a jump-table slot, the containing ``Data`` (an Array of
        Pointer) gives the table's base via ``getMinAddress()``.
        Returns ``None`` if no such containing Data exists.
        """
        try:
            addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
            data = self._listing.getDataContaining(addr_obj)
            if data is None:
                return None
            min_addr = data.getMinAddress()
            if min_addr is None:
                return None
            return int(min_addr.getOffset())
        except Exception:
            return None

    def _jump_table_target(self, base_addr: int, offset: int) -> Optional[int]:
        """Look up the resolved target for a jump-table slot.

        Builds the switch-table cache lazily on first call: walks every
        function once via ``iter_switch_tables`` and indexes
        ``{table_base_addr -> [target_addrs in slot order]}``. Slot
        index is computed from ``offset / sizeof(pointer)``; for the
        common case the pointer width is 8 (LP64) or 4 (ILP32), but we
        derive it from the count of recovered targets vs. the table's
        Data length when available.
        """
        cache = self._ensure_switch_table_cache()
        targets = cache.get(int(base_addr))
        if not targets:
            return None
        # Slot index. The table is an array of pointers; the offset is in
        # bytes. We have to derive pointer size from the program; fall
        # back to dividing by 8 if no specific size is available.
        try:
            ptr_size = int(self._program.getDefaultPointerSize())
        except Exception:
            ptr_size = 8
        if ptr_size <= 0:
            return None
        idx = offset // ptr_size
        if 0 <= idx < len(targets):
            return targets[idx]
        return None

    def _ensure_switch_table_cache(self) -> dict[int, list[int]]:
        """Build (once) the ``{table_base -> [targets]}`` cache.

        Walks every function in the program via ``iter_switch_tables``
        and merges the results. Idempotent; subsequent calls return the
        cached dict directly. Cost is paid on first JUMP_TABLE_SLOT
        slot_target access; pure-data lookups never trigger this.
        """
        if self._switch_table_cache is not None:
            return self._switch_table_cache
        cache: dict[int, list[int]] = {}
        # ``iter_switch_tables`` lives on the provider; we don't have a
        # back-ref here, so re-implement the per-function walk inline.
        # Avoids coupling MetadataLookup to GhidraDisassemblyProvider.
        try:
            from ghidra.program.model.symbol import RefType  # noqa: F401
        except Exception:
            self._switch_table_cache = cache
            return cache
        try:
            funcs = list(self._fm.getFunctions(True))
        except Exception:
            funcs = []
        for func in funcs:
            try:
                body = func.getBody()
                insn_iter = self._listing.getInstructions(body, True)
            except Exception:
                continue
            while True:
                try:
                    if not insn_iter.hasNext():
                        break
                    insn = insn_iter.next()
                except Exception:
                    break
                try:
                    flow_type = insn.getFlowType()
                    if not (flow_type.isJump() and flow_type.isComputed()):
                        continue
                except Exception:
                    continue
                table_addr: Optional[int] = None
                targets: list[int] = []
                try:
                    refs_from = list(insn.getReferencesFrom() or ())
                except Exception:
                    refs_from = []
                for ref in refs_from:
                    try:
                        rtype = ref.getReferenceType()
                    except Exception:
                        continue
                    try:
                        if rtype.isData() and rtype.isRead():
                            if table_addr is None:
                                table_addr = int(ref.getToAddress().getOffset())
                            continue
                    except Exception:
                        pass
                    try:
                        if rtype.isJump() and rtype.isComputed():
                            targets.append(int(ref.getToAddress().getOffset()))
                    except Exception:
                        pass
                if table_addr is not None and targets:
                    cache.setdefault(table_addr, list(targets))
        self._switch_table_cache = cache
        return cache


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


# BFloat16 mnemonic tables (per-ISA). Width=2 alone cannot distinguish IEEE-754
# Float16 from Google's BFloat16 — SLEIGH does not tag the bfloat16 type
# distinctly. The reclassification at width=2 consults these per-ISA mnemonic
# sets; ISAs not represented here keep the default Float16 mapping.
ARM_BF16_MNEMONICS: frozenset[str] = frozenset({
    "BFCVT", "BFCVTN", "BFCVTN2", "BFDOT", "BFMMLA",
    "BFMLAL", "BFMLALB", "BFMLALT", "VFMAB", "VFMAT",
})
X86_BF16_MNEMONICS: frozenset[str] = frozenset({
    "VCVTNE2PS2BF16", "VCVTNEPS2BF16", "VDPBF16PS",
})

# width-in-bytes -> FpType dispatch (default mapping; width=2 may be
# reclassified to BFLOAT16 by ``_compute_fp_type``).
_FP_WIDTH_TO_TYPE: dict[int, FpType] = {
    2: FpType.FLOAT16,
    4: FpType.FLOAT32,
    8: FpType.FLOAT64,
    10: FpType.FLOAT80,
    16: FpType.FLOAT128,
}


def _bfloat16_mnemonic_for_arch(arch: Architecture) -> frozenset[str]:
    """Return the BFloat16 mnemonic set for ``arch`` (empty when unsupported).

    Single dispatcher consulted at width=2 by ``_compute_fp_type`` to decide
    whether to reclassify Float16 -> BFloat16 for this instruction. ISAs
    without a curated table fall through with the default Float16 mapping.
    """
    if arch in (Architecture.ARM32, Architecture.AARCH64):
        return ARM_BF16_MNEMONICS
    if arch == Architecture.X86:
        return X86_BF16_MNEMONICS
    return frozenset()


def _ghidra_processor_to_architecture(program: Any) -> Architecture:
    """Map ``program.getLanguage().getProcessor()`` to the owned ``Architecture``.

    Threads the ISA into the FP-type computation that runs per operand.
    Unknown processors map to ``Architecture.UNKNOWN``; the BFloat16
    reclassification then no-ops.
    """
    try:
        processor = str(program.getLanguage().getProcessor()).lower()
    except Exception:
        return Architecture.UNKNOWN
    if processor.startswith("aarch64"):
        return Architecture.AARCH64
    if processor.startswith("arm"):
        return Architecture.ARM32
    if processor in ("x86", "x64") or processor.startswith("x86"):
        return Architecture.X86
    if processor.startswith("mips"):
        return Architecture.MIPS
    if processor.startswith("powerpc") or processor.startswith("ppc"):
        return Architecture.PPC
    if processor.startswith("riscv"):
        return Architecture.RISCV
    return Architecture.UNKNOWN


def _compute_fp_type(
    ghidra_insn: Any,
    operand_index: int,
    arch: Architecture,
    base_mnemonic: str,
) -> Optional[FpType]:
    """Module-level helper backing ``operand_fp_type``.

    Called per operand from the decode path so the resulting ``FpType``
    can be stamped on the owned operand view, keeping the public
    ``GhidraDisassemblyProvider.operand_fp_type`` method as a thin
    wrapper. Returns the matching ``FpType`` when the
    operand is FP-typed (Ghidra ``OperandType.FLOAT`` bitmask) or
    ``None`` otherwise. The width derivation order is:

    1. Inspect each ``getOpObjects(i)`` element. For ``Register`` operands,
       use ``Register.getBitLength() / 8``. For ``Scalar`` operands, use
       ``Scalar.bitLength() / 8``. Take the largest value seen (x87
       ``fld dword ptr [...]`` carries an FP-tagged memory operand whose
       size is the load size).
    2. If no op-object width is available, fall back to
       ``ghidra_insn.getOperandRefType(i).getSize()``.
    3. Map the resulting width-in-bytes through ``_FP_WIDTH_TO_TYPE``.
    4. At width=2, consult ``_bfloat16_mnemonic_for_arch(arch)`` against
       the instruction's ``base_mnemonic`` (uppercase-compared) and
       reclassify Float16 -> BFloat16 on a hit. SLEIGH does not currently
       tag bfloat16 distinctly, so the mnemonic-based reclassification
       is the only signal available.
    5. Widths outside ``_FP_WIDTH_TO_TYPE`` return ``None`` (the
       classifier then routes through step 11 of the precedence list
       rather than emitting a malformed ``floatXX``).
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

    width_bytes = max_width_bits // 8
    if width_bytes == 0:
        # Fall back to the reference-type's reported access size
        # (memory FP loads/stores).
        try:
            ref_type = ghidra_insn.getOperandRefType(operand_index)
            if ref_type is not None:
                size_bytes = int(ref_type.getSize())
                if size_bytes > 0:
                    width_bytes = size_bytes
        except Exception:
            pass

    fp_type = _FP_WIDTH_TO_TYPE.get(width_bytes)
    if fp_type is None:
        return None

    if fp_type == FpType.FLOAT16:
        bf16_set = _bfloat16_mnemonic_for_arch(arch)
        if bf16_set and base_mnemonic.upper() in bf16_set:
            fp_type = FpType.BFLOAT16

    return fp_type


# ---------------------------------------------------------------------------
# Memory-operand decomposition helpers (per-ISA)
# ---------------------------------------------------------------------------
# Each helper inspects ``ghidra_insn.getOpObjects(op_idx)`` and computes the
# decomposed (base, index, scale, disp, segment) tuple for a single MEM
# operand. Helpers are PURE w.r.t. Ghidra Java state - they read but do not
# mutate. Each returns a tuple-of-strings-and-ints suitable for
# ``_GhidraMemoryOperandView._populate``.
#
# The x86 path is faithfully ported from the legacy Ghidra-side memory
# tokenizer at ``tokenizer/arch/x86/ghidra/operands.py`` (the
# ``_classify_objects + assign_base_index_scale_disp`` block of
# ``tokenize_operand_memory_ghidra``). The ARM and base+disp paths
# expose the same (base, index, scale, disp, segment) wire-shape that
# the typed ``MemoryOperandView`` exposes to consumers.


def _infer_mem_size_from_ghidra_insn(ghidra_insn: Any, default: int = 8) -> int:
    """Infer x86 memory-operand size from sibling register operands.

    Ported verbatim from the legacy
    ``tokenizer/arch/x86/ghidra/operands.py::_infer_size_from_ghidra_insn``
    (deleted in G.3); the post-G.3 consumer (the shared
    ``arch/x86/operands.py::tokenize_operand_memory``) reads ``op.size``
    uniformly across both providers, so the inference now stamps the
    operand spec at decode time.

    Checks ``getResultObjects()`` then ``getInputObjects()`` for
    general-purpose ``Register`` operands and returns the largest
    ``getMinimumByteSize()``. The segment-register filter avoids picking
    up 1-byte flags registers (CF / PF / ZF / SF / OF) that surface in
    result objects for arithmetic instructions.
    """
    from ghidra.program.model.lang import Register

    max_size = 0
    for source in (ghidra_insn.getResultObjects(), ghidra_insn.getInputObjects()):
        if source is None:
            continue
        for obj in source:
            if isinstance(obj, Register):
                name = str(obj.getName()).lower()
                if name not in _SEGMENT_REGISTERS:
                    size = int(obj.getMinimumByteSize())
                    if size > max_size:
                        max_size = size
    return max_size if max_size > 0 else default


def _compute_x86_memory_components(
    ghidra_insn: Any,
    op_idx: int,
    reg_map: "_RegisterMap",
) -> tuple[str, int, str, int, int, int, str, int]:
    """Decompose an x86/x64 MEM operand from raw Ghidra objects.

    Returns ``(base_name, base_id, index_name, index_id, scale, disp,
    segment_name, segment_id)``. Empty name + id=0 means the slot is absent.

    Object-count rules from getOpObjects() (faithful port):
        2 general regs   -> first Scalar = scale, remaining Scalars = disp
        0-1 general regs -> all Scalars = disp
        Address objects  -> disp

    The first GP Register in ``getOpObjects()`` is the base; the second is
    the index. This is the Ghidra SLEIGH spec's documented convention.
    Operands not conforming (3+ regs => reg-list) MUST have been
    classified upstream as ``OperandKind.REG_LIST`` so this function
    never sees them; the assert at the end of the object-walk enforces
    that invariant.
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    objects = ghidra_insn.getOpObjects(op_idx)

    segment_reg_name: str = ""
    segment_reg_id: int = 0
    general_reg_names: list[str] = []
    general_reg_ids: list[int] = []
    scalars: list[int] = []
    signed_scalars: list[int] = []
    disp: int = 0

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            rid = reg_map.get_id(name)
            if name in _SEGMENT_REGISTERS:
                segment_reg_name = name
                segment_reg_id = rid
            else:
                general_reg_names.append(name)
                general_reg_ids.append(rid)
        elif isinstance(obj, Scalar):
            scalars.append(int(obj.getValue()))
            signed_scalars.append(int(obj.getSignedValue()))
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    assert len(general_reg_names) <= 2, (
        f"x86 MEM operand should have at most 2 GP registers, got "
        f"{len(general_reg_names)}: {general_reg_names!r}. If this fires, "
        f"the operand should have classified as REG_LIST upstream."
    )
    assert len(scalars) <= 2, (
        f"x86 MEM operand should have at most 2 Scalar slots (scale + "
        f"disp), got {len(scalars)}: {scalars!r}"
    )

    base_name = general_reg_names[0] if general_reg_names else ""
    base_id = general_reg_ids[0] if general_reg_ids else 0
    index_name = general_reg_names[1] if len(general_reg_names) >= 2 else ""
    index_id = general_reg_ids[1] if len(general_reg_ids) >= 2 else 0
    scale: int = 1

    if len(general_reg_ids) >= 2 and scalars:
        scale = scalars[0]
        if len(scalars) > 1:
            disp = signed_scalars[1]
    elif len(general_reg_ids) <= 1 and scalars:
        disp = signed_scalars[0]

    return base_name, base_id, index_name, index_id, scale, disp, segment_reg_name, segment_reg_id


def _compute_arm_memory_components(
    ghidra_insn: Any,
    op_idx: int,
    reg_map: "_RegisterMap",
) -> tuple[str, int, str, int, int, int, str, int]:
    """Decompose an ARM MEM operand from raw Ghidra objects.

    ARM addressing modes use base + optional index register + optional
    displacement (no scale, no segment). Returns the same 8-tuple shape
    as the x86 helper, with scale=1 fixed and segment slots absent.

    First general-purpose Register -> base; second -> index; first
    Scalar/Address -> disp. This is the Ghidra SLEIGH spec's documented
    convention. Operands not conforming (3+ regs => reg-list)
    MUST have been classified upstream as ``OperandKind.REG_LIST`` so
    this function never sees them (stm/ldm/push/pop/vpush/vpop/vstm/vldm
    family); the assert at the end of the object-walk enforces that
    invariant.
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    objects = ghidra_insn.getOpObjects(op_idx)

    general_reg_names: list[str] = []
    general_reg_ids: list[int] = []
    disp: int = 0

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            general_reg_names.append(name)
            general_reg_ids.append(reg_map.get_id(name))
        elif isinstance(obj, Scalar):
            disp = int(obj.getSignedValue())
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    assert len(general_reg_names) <= 2, (
        f"ARM MEM operand should have at most 2 GP registers, got "
        f"{len(general_reg_names)}: {general_reg_names!r}. If this fires, "
        f"the operand should have classified as REG_LIST upstream "
        f"(stm/ldm/push/pop/vpush/vpop/vstm/vldm family)."
    )

    base_name = general_reg_names[0] if general_reg_names else ""
    base_id = general_reg_ids[0] if general_reg_ids else 0
    index_name = general_reg_names[1] if len(general_reg_names) >= 2 else ""
    index_id = general_reg_ids[1] if len(general_reg_ids) >= 2 else 0

    return base_name, base_id, index_name, index_id, 1, disp, "", 0


def _compute_base_disp_memory_components(
    ghidra_insn: Any,
    op_idx: int,
    reg_map: "_RegisterMap",
) -> tuple[str, int, str, int, int, int, str, int]:
    """Decompose a base+disp MEM operand (MIPS/PPC/RISC-V).

    These ISAs only ever have one base register + one displacement; no
    index, no scale, no segment. Returns the 8-tuple with index slot
    absent, scale=1, segment slots absent.

    The first GP Register in ``getOpObjects()`` is the base. This is the
    Ghidra SLEIGH spec's documented convention. Operands not conforming
    (3+ regs => reg-list) MUST have been classified upstream as
    ``OperandKind.REG_LIST`` so this function never sees them; the
    assert below enforces the invariant.
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    objects = ghidra_insn.getOpObjects(op_idx)

    base_name: str = ""
    base_id: int = 0
    disp: int = 0
    general_regs: list[str] = []

    for obj in objects or ():
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            general_regs.append(name)
            if base_name == "":
                base_name = name
                base_id = reg_map.get_id(base_name)
        elif isinstance(obj, Scalar):
            disp = int(obj.getSignedValue())
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    assert len(general_regs) <= 1, (
        f"base+disp MEM operand should have at most 1 GP register, "
        f"got {len(general_regs)}: {general_regs!r}"
    )

    return base_name, base_id, "", 0, 1, disp, "", 0


# ---------------------------------------------------------------------------
# Per-ISA prefix builders
# ---------------------------------------------------------------------------
# Build typed ``InstructionPrefixView`` instances for a Ghidra Instruction.
# x86 reads the prefix-byte set; ARM / PPC / MIPS / RISC-V return empty
# lists for now (their typed-prefix fields stay at defaults, so
# consumer predicates always fall through until those signals become
# available).

_X86_BYTE_TO_PREFIX_BUILDER: dict[int, Any] = {
    # Filled lazily on first use to avoid importing the prefix subclasses
    # at module load time.
}


def _x86_byte_to_prefix(byte: int) -> Any:
    """Return a typed ``InstructionPrefixView`` for an x86 prefix byte.

    Returns ``None`` for bytes outside the recognized prefix set (caller
    skips). Lazy-initializes the byte->builder map to avoid pulling in
    typed prefix classes at module import time.
    """
    if not _X86_BYTE_TO_PREFIX_BUILDER:
        from tokenizer.disasm.ghidra_views import (
            _AddressSizePrefix,
            _LockPrefix,
            _OperandSizePrefix,
            _RepPrefix,
            _SegmentOverridePrefix,
        )
        from tokenizer.disasm.types import X86Segment

        _X86_BYTE_TO_PREFIX_BUILDER.update({
            0xF0: lambda: _LockPrefix(),
            0xF2: lambda: _RepPrefix(repeat_until_zero=False),  # REPNE
            0xF3: lambda: _RepPrefix(repeat_until_zero=True),   # REPE/REP
            0x26: lambda: _SegmentOverridePrefix(X86Segment.ES),
            0x2E: lambda: _SegmentOverridePrefix(X86Segment.CS),
            0x36: lambda: _SegmentOverridePrefix(X86Segment.SS),
            0x3E: lambda: _SegmentOverridePrefix(X86Segment.DS),
            0x64: lambda: _SegmentOverridePrefix(X86Segment.FS),
            0x65: lambda: _SegmentOverridePrefix(X86Segment.GS),
            0x66: lambda: _OperandSizePrefix(),
            0x67: lambda: _AddressSizePrefix(),
        })
    builder = _X86_BYTE_TO_PREFIX_BUILDER.get(byte)
    if builder is None:
        return None
    return builder()


def _build_prefixes_x86(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for an x86 instruction.

    Reads the same legacy prefix-byte set ``_extract_x86_prefixes``
    populates, then translates each byte into a typed
    ``InstructionPrefixView`` instance via ``_x86_byte_to_prefix``.
    Order: the byte-set is sorted so the produced list is stable across
    calls (the per-byte translation is independent of original encoding
    order).
    """
    prefix_bytes = _extract_x86_prefixes(ghidra_insn)
    out: list[Any] = []
    for byte in sorted(prefix_bytes):
        view = _x86_byte_to_prefix(byte)
        if view is not None:
            out.append(view)
    return out


def _build_prefixes_arm(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for an ARM instruction.

    Stub for forward-compat: ARM-side condition-code / update-flags /
    writeback signals are not extracted by the Ghidra path today, so
    consumers see them as zero/false defaults. Returns an empty list
    until ARM extraction is wired up.
    """
    return []


def _build_prefixes_ppc(ghidra_insn: Any) -> list[Any]:
    """Build typed prefix-view instances for a PPC instruction.

    Stub for forward-compat; same shape as ``_build_prefixes_arm``.
    The ``bc`` and ``update_cr0`` signals are not extracted by the
    Ghidra path today.
    """
    return []


def _build_prefixes_empty(ghidra_insn: Any) -> list[Any]:
    """No-prefix builder for MIPS/RISC-V."""
    return []


def _prefix_builder_for_arch(arch: Architecture) -> Any:
    """Dispatch the per-ISA prefix builder."""
    if arch == Architecture.X86:
        return _build_prefixes_x86
    if arch in (Architecture.ARM32, Architecture.AARCH64):
        return _build_prefixes_arm
    if arch == Architecture.PPC:
        return _build_prefixes_ppc
    return _build_prefixes_empty


# ---------------------------------------------------------------------------
# Decode helper - injected into _GhidraInstructionView wrappers
# ---------------------------------------------------------------------------
class _GhidraDecodeHelper:
    """Provider-owned helper exposing the per-instruction decode surface
    the owned-view wrappers (``ghidra_views.py``) need.

    Construction: one instance per ``GhidraDisassemblyProvider`` (program
    + reg_map are stable for the program's lifetime). The helper is
    passed to each ``_GhidraFunctionView`` -> ``_GhidraBlockView`` ->
    ``_GhidraInstructionView`` constructor so the view chain stays
    self-contained without cross-importing the provider.

    The helper centralizes:
      - mnemonic split + alias canonicalization
      - architecture detection (cached per program)
      - per-operand FP-type computation
      - per-instruction typed-prefix list build
      - per-operand decompose-mem callback construction (lazy: returns a
        zero-arg callable that, when invoked, populates a passed
        ``_GhidraMemoryOperandView``)
      - per-operand spec dict (kwargs ready for
        ``_GhidraOperandView._advance``)
    """

    __slots__ = ("_program", "_reg_map", "_arch")

    def __init__(self, program: Any, reg_map: "_RegisterMap") -> None:
        self._program = program
        self._reg_map = reg_map
        self._arch: Architecture = _ghidra_processor_to_architecture(program)

    @property
    def arch(self) -> Architecture:
        return self._arch

    def split_mnemonic(self, raw: str) -> tuple[str, str | None, int | None]:
        return _split_ghidra_mnemonic(raw)

    def alias_mnemonic(self, base: str) -> str:
        # Ghidra's MIPS SLEIGH spec emits `_sra` / `_li` / ... for the
        # delay-slot variants of `sra` / `li` / .... The underscore is a
        # Ghidra display convention, not a real MIPS-ISA mnemonic
        # distinction (gas/objdump/Capstone all just write `sra`).
        # Delay-slot membership is already encoded positionally by the
        # token sequence, so the duplicate vocab entry is noise.
        if self._arch == Architecture.MIPS and base.startswith("_") and len(base) > 1:
            base = base[1:]
        return _GHIDRA_MNEMONIC_ALIASES.get(base, base)

    def architecture(self, _program: Any) -> Architecture:
        return self._arch

    def compute_fp_type(
        self,
        ghidra_insn: Any,
        operand_index: int,
        arch: Architecture,
        base_mnemonic: str,
    ) -> Optional[FpType]:
        return _compute_fp_type(ghidra_insn, operand_index, arch, base_mnemonic)

    def build_prefixes(self, ghidra_insn: Any, arch: Architecture) -> list[Any]:
        return _prefix_builder_for_arch(arch)(ghidra_insn)

    def _decompose_mem_callback(
        self,
        ghidra_insn: Any,
        op_idx: int,
        arch: Architecture,
    ) -> Any:
        """Return a zero-arg callable that decomposes the MEM operand into
        a passed-in ``_GhidraMemoryOperandView``.

        Selects the per-ISA helper. The closure captures ``ghidra_insn``,
        ``op_idx``, and the provider's ``reg_map`` so the operand wrapper
        only needs to invoke the callback at lazy-decomposition time
        (first ``op.mem`` access).
        """
        reg_map = self._reg_map
        if arch == Architecture.X86:
            compute = _compute_x86_memory_components
        elif arch in (Architecture.ARM32, Architecture.AARCH64):
            compute = _compute_arm_memory_components
        else:
            compute = _compute_base_disp_memory_components

        def _populate(mem_view) -> None:
            (
                base_name,
                base_id,
                index_name,
                index_id,
                scale,
                disp,
                segment_name,
                segment_id,
            ) = compute(ghidra_insn, op_idx, reg_map)
            mem_view._populate(
                base_name=base_name,
                base_id=base_id,
                index_name=index_name,
                index_id=index_id,
                segment_name=segment_name,
                segment_id=segment_id,
                scale=scale,
                disp=disp,
            )

        return _populate

    def _decompose_reg_list_callback(
        self,
        ghidra_insn: Any,
        op_idx: int,
        arch: Architecture,
    ) -> Any:
        """Return a zero-arg callable that decomposes a REG_LIST operand
        into a passed-in ``_GhidraRegisterListView``.

        ARM stm/ldm-family operands surface in ``getOpObjects()`` as a
        flat sequence of Register objects. The Ghidra SLEIGH convention
        for these encodings is: the FIRST Register is the writeback
        target (the base register that lives *outside* the braces in
        the asm); the remaining Registers are the list members (the
        registers *inside* the braces).

        Writeback (`!`) detection is currently unsupported: Ghidra's
        ``OperandType`` bitmask has no documented writeback bit, and the
        existing ``_build_prefixes_arm`` is a stub. We therefore set
        ``writeback=False`` here for every reg-list operand.
        TODO: writeback detection (likely via raw mnemonic parsing or
        PCode self-assignment inspection) is deferred.
        """
        from ghidra.program.model.lang import Register

        reg_map = self._reg_map

        def _populate(reg_list_view) -> None:
            try:
                objects = ghidra_insn.getOpObjects(op_idx)
            except Exception:
                objects = ()

            regs: list[tuple[str, int]] = []
            for obj in objects or ():
                if isinstance(obj, Register):
                    name = str(obj.getName()).lower()
                    regs.append((name, reg_map.get_id(name)))

            if regs:
                base_name, base_id = regs[0]
                member_specs = regs[1:]
            else:
                # Sentinel-absent: name="" + id=0 matches _GhidraRegisterView
                # 's `_set_absent` shape (sentinels are private to
                # ghidra_views.py; using their values directly keeps the
                # cross-module surface clean).
                base_name, base_id = "", 0
                member_specs = []

            reg_list_view._advance(
                base_name=base_name,
                base_id=base_id,
                writeback=False,
                member_specs=member_specs,
            )

        return _populate

    def synthesize_disp_base_mem_spec(
        self,
        disp_spec: dict,
        base_spec: dict,
    ) -> dict:
        """Synthesize a MEM operand spec from a (disp IMM, base REG) pair.

        Used when Ghidra's SLEIGH spec splits a disp(base) memory
        operand into two adjacent flat operands - notably the RISC-V
        compressed-instruction encodings (``c.sdsp ra, 0x8(sp)`` is
        reported as 3 operands: ``ra``, ``0x8`` [DYNAMIC scalar],
        ``sp``). Caller pair-detects adjacent IMM-DYNAMIC + REG operands
        and asks us to fuse them into one synthetic MEM operand whose
        decomposition reads the captured base_name + disp directly
        (rather than going back to ``getOpObjects()`` which only sees
        the disjoint Scalar and Register on separate operand indices).

        The values are pre-captured into the closure on this call so
        subsequent calls (e.g. the next instruction) don't rebind the
        closure-bound values mid-iteration.

        ``type_int`` is the bitwise OR of the two halves so consumers
        peeking the raw OperandType bitmask see both the DYNAMIC bit
        (from the disp half) and the REGISTER bit (from the base half).
        """
        base_name = base_spec["reg_name"]
        base_id = base_spec["reg_id"]
        disp = disp_spec["imm"]
        fp_type = disp_spec["fp_type"]
        type_int = int(disp_spec["type_int"]) | int(base_spec["type_int"])

        def _decompose(view: "_GhidraMemoryOperandView") -> None:
            view._populate(
                base_name=base_name,
                base_id=base_id,
                index_name="",
                index_id=0,
                scale=1,
                disp=disp,
                segment_name="",
                segment_id=0,
            )

        # Mirror the default spec shape from ``operand_spec`` so the
        # consumer's ``_GhidraOperandView._advance(**spec)`` accepts
        # every kwarg without surprise.
        from tokenizer.disasm.types import ShiftKind as _ShiftKind

        spec = dict(
            kind=OperandKind.MEM,
            reg_name="",
            reg_id=0,
            imm=0,
            size=base_spec.get("size", 0),
            fp_type=fp_type,
            type_int=type_int,
            decompose_mem=_decompose,
            shift_kind=_ShiftKind.NONE,
            shift_amount=0,
            crx_reg_name="",
            crx_reg_id=0,
            decompose_reg_list=None,
        )
        return spec

    def operand_spec(
        self,
        ghidra_insn: Any,
        op_idx: int,
        arch: Architecture,
        base_mnemonic: str,
        reg_map: "_RegisterMap",
    ) -> dict:
        """Return a kwargs dict for ``_GhidraOperandView._advance``.

        Classifies the operand kind (REG/IMM/MEM/CRX/OTHER) from
        Ghidra's ``OperandType`` bitmask + ``getOpObjects()`` shape,
        computes per-operand size + FP type, and produces the
        decompose-mem callback when the operand is MEM. Non-MEM
        operands carry ``decompose_mem=None`` so the operand wrapper
        skips lazy MEM decomposition.

        ``reg_map`` is passed explicitly (not read from ``self._reg_map``)
        so the spec composes cleanly with the views' constructor wiring;
        in practice they are the same object.
        """
        from ghidra.program.model.address import Address
        from ghidra.program.model.lang import OperandType, Register
        from ghidra.program.model.scalar import Scalar
        from tokenizer.disasm.types import ShiftKind as _ShiftKind

        try:
            objects = ghidra_insn.getOpObjects(op_idx)
        except Exception:
            objects = ()
        try:
            op_type = ghidra_insn.getOperandType(op_idx)
        except Exception:
            op_type = 0

        # Pre-collect Register objects: used both by the is_memory check
        # below (a memory operand MUST involve at least one base/index
        # register) and the reg-list classifier (>= 3 registers => REG_LIST).
        register_objs = [o for o in objects if isinstance(o, Register)]

        # A memory operand MUST involve at least one base/index register.
        # Without that, Ghidra's DYNAMIC bit on a pure-scalar operand
        # (e.g. RISC-V c.addi's immediate, or c.sdsp's disp scalar that
        # SLEIGH split off from its base register) is misleading and
        # produces a degenerate base-less mem-bracket rendering.
        is_memory = bool(register_objs) and (
            bool(op_type & OperandType.DYNAMIC)
            or bool(op_type & OperandType.INDIRECT)
            or (
                bool(op_type & OperandType.ADDRESS)
                and bool(op_type & OperandType.SCALAR)
                and not (op_type & (OperandType.REGISTER | OperandType.CODE))
            )
        )

        fp_type = _compute_fp_type(ghidra_insn, op_idx, arch, base_mnemonic)

        # Default spec - filled per kind below.
        spec = dict(
            kind=OperandKind.INVALID,
            reg_name="",
            reg_id=0,
            imm=0,
            size=0,
            fp_type=fp_type,
            type_int=int(op_type),
            decompose_mem=None,
            shift_kind=_ShiftKind.NONE,
            shift_amount=0,
            crx_reg_name="",
            crx_reg_id=0,
            decompose_reg_list=None,
        )

        if not objects:
            return spec

        # Reg-list classification (ARM stm/ldm/push/pop/vpush/vpop/vstm/
        # vldm family). Ghidra's SLEIGH spec emits a flat sequence of
        # Register objects for reg-list operands; standard MEM operands
        # on every supported ISA carry at most 2 Registers (base+index
        # on x86/ARM, base-only on MIPS/PPC/RISC-V). Three or more
        # Registers in a single operand can therefore only be a
        # reg-list; classify accordingly so the MEM-decompose helpers
        # never see them (asserts in those helpers enforce the
        # invariant downstream).
        if len(register_objs) >= 3:
            spec["kind"] = OperandKind.REG_LIST
            spec["decompose_reg_list"] = self._decompose_reg_list_callback(
                ghidra_insn, op_idx, arch
            )
            return spec

        if is_memory:
            spec["kind"] = OperandKind.MEM
            # Memory size: inferred from sibling register operands. The
            # legacy ``tokenize_operand_memory_ghidra`` ran this inference
            # at tokenization time via ``_infer_size_from_ghidra_insn``;
            # we hoist it into the operand spec so the post-G.3 consumer
            # (the shared ``arch/x86/operands.py::tokenize_operand_memory``)
            # can read ``op.size`` uniformly across both providers.
            # Default 8 mirrors the legacy ``_infer_size_from_ghidra_insn``
            # signature; ARM / MIPS / PPC / RISC-V consumers do not look
            # at ``op.size`` for MEM operands so the value is harmless on
            # non-x86 ISAs.
            spec["size"] = _infer_mem_size_from_ghidra_insn(ghidra_insn)
            spec["decompose_mem"] = self._decompose_mem_callback(ghidra_insn, op_idx, arch)
            return spec

        first = objects[0]
        if isinstance(first, Register):
            name = str(first.getName()).lower()
            spec["kind"] = OperandKind.REG
            spec["reg_name"] = name
            spec["reg_id"] = reg_map.get_id(name)
            try:
                spec["size"] = int(first.getMinimumByteSize())
            except Exception:
                spec["size"] = 0
            return spec
        if isinstance(first, Scalar):
            spec["kind"] = OperandKind.IMM
            spec["imm"] = int(first.getValue())
            try:
                spec["size"] = int(first.bitLength()) // 8
            except Exception:
                spec["size"] = 0
            return spec
        if isinstance(first, Address):
            spec["kind"] = OperandKind.IMM
            spec["imm"] = int(first.getOffset())
            return spec

        # Unknown op kind - treat as OTHER passthrough so consumers that
        # gate on ``op.kind == OperandKind.OTHER`` can route correctly.
        spec["kind"] = OperandKind.OTHER
        return spec


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
        # Owned-view scaffolding lazily initialized by iter_functions on
        # first call (so close() can null out program references without
        # leaving stale views behind).
        self._function_view: Optional[Any] = None
        # `iter_functions` populates this map with `entry_addr -> Ghidra
        # Function` for every function it visits; `iter_switch_tables`
        # consumes it to look up the underlying Java handle from the
        # FunctionView (no more `function._raw` unwrap dance).
        self._funcs_by_entry: dict[int, Any] = {}

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
        # Drop owned-view scaffolding too so the next task starts
        # from a clean slate.
        self._function_view = None
        self._funcs_by_entry = {}

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

    def iter_functions(self) -> Iterable[tuple[int, str, "_GhidraFunctionView"]]:
        """Iterate functions, yielding (addr, name, function_view) triples.

        Internal model:
            * One reusable ``_GhidraFunctionView`` is mutated each step to
              point at the current Ghidra Function (lazy/reuse contract,
              per ``tokenizer/disasm/types.py``). The reused view is the
              source of truth for consumers.
            * ``self._funcs_by_entry`` is populated as we iterate so
              ``iter_switch_tables(function)`` can look up the Ghidra
              Java handle via ``function.entry`` (no more
              ``isinstance/_raw`` dance).
        """
        assert self._analyzed, "Analysis not run -- call build_cfg() first"

        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import TaskMonitor

        # Lazily initialize the owned-view scaffolding: decode helper +
        # function view (reused across every iter step). The scaffolding
        # is rebuilt per ``iter_functions`` call so each call gets a
        # fresh cursor (defensive: callers that re-enter mid-iteration
        # would otherwise share cursor state).
        decode = _GhidraDecodeHelper(self._program, self._reg_map)
        block_model = SimpleBlockModel(self._program)
        monitor = TaskMonitor.DUMMY
        from tokenizer.disasm.ghidra_views import _GhidraFunctionView

        function_view = _GhidraFunctionView(
            arch=decode.arch,
            program=self._program,
            listing=self._listing,
            reg_map=self._reg_map,
            decode=decode,
            block_model=block_model,
            monitor=monitor,
        )
        self._function_view = function_view

        # Collect + sort by name (legacy contract).
        funcs: list[tuple[int, str, Any]] = []
        for func in self._fm.getFunctions(True):
            name = str(func.getName())
            addr = int(func.getEntryPoint().getOffset())
            funcs.append((addr, name, func))

        # Reset the entry-point map for this iter call (callers that
        # re-iterate get a fresh map, no stale entries).
        self._funcs_by_entry = {}

        for addr, name, ghidra_func in sorted(funcs, key=lambda t: t[1]):
            body = ghidra_func.getBody()
            block_iter = block_model.getCodeBlocksContaining(body, monitor)

            # Count code blocks so the reused FunctionView can expose an
            # O(1) ``len(blocks)`` once we advance it. Functions with zero
            # code blocks are skipped.
            block_count = 0
            while block_iter.hasNext():
                block_iter.next()
                block_count += 1

            if block_count == 0:
                continue

            function_view._advance(ghidra_func, block_count)
            self._funcs_by_entry[addr] = ghidra_func
            yield addr, name, function_view

    # ----------------------------------------------------------------------
    # Per-operand FP-immediate detection
    # ----------------------------------------------------------------------
    def operand_fp_type(self, ghidra_insn: Any, operand_index: int) -> Optional[FpType]:
        """Return the typed ``FpType`` for ``operand_index`` of ``ghidra_insn``.

        Returns one of ``FpType.FLOAT16`` / ``FpType.BFLOAT16`` /
        ``FpType.FLOAT32`` / ``FpType.FLOAT64`` / ``FpType.FLOAT80`` /
        ``FpType.FLOAT128`` when the operand is FP-typed (per Ghidra's
        ``OperandType.FLOAT`` bitmask + the per-ISA BFloat16 mnemonic
        table at width=2) and ``None`` otherwise. Width derivation
        order:

        1. Inspect each ``getOpObjects(i)`` element. For ``Register``
           operands, ``Register.getBitLength() / 8``. For ``Scalar``
           operands, ``Scalar.bitLength() / 8``. Take the largest value
           seen (x87 ``fld dword ptr [...]`` carries an FP-tagged
           memory operand whose size is the load size).
        2. If no op-object width is available, fall back to
           ``ghidra_insn.getOperandRefType(i).getSize()``.
        3. Map width-in-bytes through ``_FP_WIDTH_TO_TYPE``.
        4. At width=2, reclassify Float16 -> BFloat16 when the
           instruction's mnemonic appears in the per-ISA table
           (``_bfloat16_mnemonic_for_arch``). SLEIGH does not currently
           tag bfloat16 distinctly, so the mnemonic-based reclassification
           is the only signal available.
        5. Widths outside ``_FP_WIDTH_TO_TYPE`` return ``None`` (the
           classifier then routes through step 11 of the precedence list
           rather than emitting a malformed ``floatXX``).

        Implementation note: the owned-view decode path stamps this type
        on every operand view's ``fp_type`` at instruction-translation
        time, so operand tokenizers don't need to call this method
        per-operand. The public method stays on the provider for direct
        callers and for symmetry with the Phase 1.B.1
        provider-interface contract.
        """
        arch = _ghidra_processor_to_architecture(self._program)
        try:
            raw_mnemonic = str(ghidra_insn.getMnemonicString())
        except Exception:
            raw_mnemonic = ""
        base_mnemonic, _, _ = _split_ghidra_mnemonic(raw_mnemonic)
        base_mnemonic = _GHIDRA_MNEMONIC_ALIASES.get(base_mnemonic, base_mnemonic)
        return _compute_fp_type(ghidra_insn, operand_index, arch, base_mnemonic)

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

        ``function`` accepts any object with an ``entry`` attribute - the
        typed ``FunctionView`` yielded by ``iter_functions`` or any
        future shape that exposes the function's entry-point address.
        The Ghidra Java handle is looked up via ``self._funcs_by_entry``
        (populated lazily by ``iter_functions`` as it walks each
        function); calling ``iter_switch_tables`` without first
        iterating ``iter_functions`` on the same provider returns an
        empty iterator.
        """
        if function is None:
            return
        try:
            entry = int(function.entry)
        except AttributeError:
            return
        ghidra_function = self._funcs_by_entry.get(entry)
        if ghidra_function is None:
            return

        from ghidra.program.model.symbol import RefType

        body = ghidra_function.getBody()
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
