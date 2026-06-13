"""Section-name + string-DataType + vtable detection helpers.

Pure Ghidra-side classifiers consumed by ``GhidraMetadataLookup``:
- Section-name -> v2 section-type mapping.
- Permission-bit-fallback for unrecognized section names.
- String DataType -> Python encoding name.
- Java byte[] -> Python bytes conversion.
- Vtable symbol/data-type name predicates.
"""

from __future__ import annotations

from typing import Any


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

    JPype 1.6.0 exposes a Java ``byte[]`` through the buffer protocol as a
    signed-``int8`` buffer, so ``bytes(memoryview(raw))`` performs a single
    C-level copy of the underlying storage. Because two's-complement signed
    bytes share the same bit pattern as their unsigned reinterpretation, the
    copy is byte-identical to the old per-element ``int(b) & 0xFF`` masking
    (verified for negative Java bytes, e.g. ``-1 -> 0xFF``) while avoiding the
    52.8M boxed-int allocations the generator path incurred on multi-MB Data
    blobs. ``raw`` may be None when the Data object hasn't materialized its
    byte payload -- in that case return ``b""``.
    """
    if raw is None:
        return b""
    return bytes(memoryview(raw))


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
