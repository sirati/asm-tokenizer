"""Owned address-metadata domain types - lazy view returned by `MetadataLookup`.

Replaces the v1 stringly-keyed dict that `MetadataLookup.lookup(addr)` used
to return. See `tokenizer/disasm/types.py` for the broader lifecycle contract
that also applies to the wrapper here: the returned `AddressMetadataView` is
REUSED across `lookup()` calls and is valid only until the next call. Use
`copy.deepcopy(view)` to stash across lookups.

Phase D.3 (task #40) retired the transitional dict-shim base class and the
v1-tuple adapter. Providers populate the typed slots directly at lookup
time; consumers (`ConstantHandler`, arch operand modules) read typed
properties exclusively. The string -> enum mapping helpers below
(`section_kind_from_type_string`, `encoding_from_string`,
`address_kind_from_string`) are still exported because providers use them
ONCE at population time to translate raw backend strings into the typed
enums; nothing reads them on the consumer side.
"""

from enum import IntEnum
from typing import Hashable, Optional, Protocol, runtime_checkable


class Encoding(IntEnum):
    UNKNOWN = 0
    ASCII = 1
    UTF8 = 2
    UTF16_LE = 3
    UTF16_BE = 4
    LATIN_1 = 5
    PASCAL_ASCII = 6
    PASCAL_UTF16 = 7
    UTF_32_LE = 8
    UTF_32_BE = 9
    MBCS = 10


class SectionKind(IntEnum):
    UNKNOWN = 0
    CODE = 1
    RODATA = 2
    DATA = 3
    BSS = 4
    TLS_DATA = 5         # .tdata
    TLS_BSS = 6          # .tbss
    PLT = 7
    INIT_ARRAY = 8       # .init_array / .ctors
    FINI_ARRAY = 9       # .fini_array / .dtors
    SYNTHETIC_EXTERN = 10
    OTHER = 99


class AddressKind(IntEnum):
    UNKNOWN = 0
    LOCAL_FUNCTION = 1
    BLOCK_IN_FUNCTION = 2
    PLT_FUNCTION = 3
    EXT_FUNCTION_REAL = 4
    EXT_FUNCTION_SYNTHETIC = 5
    STRING = 6
    RODATA = 7
    DATA = 8
    BSS = 9
    THREAD_LOCAL_DATA = 10
    CODE_PTR_TABLE_SLOT = 11
    JUMP_TABLE_SLOT = 12
    SYNTHETIC_SECTION = 13
    NONE = 99


@runtime_checkable
class AddressMetadataView(Protocol):
    """Typed lazy view returned by `MetadataLookup.lookup(addr)`.

    Read-only typed surface. Providers stamp every field at population
    time so reads are slot loads with no per-access string -> enum work.

    LIFECYCLE: the wrapper is REUSED across `lookup()` calls - valid only
    until the next `lookup()`. `copy.deepcopy(view)` produces a fresh
    wrapper bound to a snapshot of the current state. `slot_target` is
    lazily computed on first access within a lookup cycle (Ghidra path
    only; angr returns None).
    """

    @property
    def kind(self) -> AddressKind: ...

    @property
    def name(self) -> Optional[str]: ...        # canonical human-readable label

    @property
    def comment(self) -> Optional[str]: ...
    # Provider-supplied "context" string for this function address (the
    # demangled C++ scoped signature when available; ``None`` for C / asm
    # symbols and for any non-function address). The lookup populates
    # this field alongside ``name`` so consumers / sidecar writers that
    # already key on ``meta.name`` for cross-ISA-stable identity get the
    # same answer as the FunctionDataManager's canonical-name path: the
    # ``name`` is ALREADY the result of
    # ``canonical_function_name(raw_name, comment, identity_key)``.

    @property
    def identity_key(self) -> Optional[Hashable]: ...
    # Provider-supplied "stronger-than-name" identity for this function
    # address. PLT thunks: a typed
    # :class:`tokenizer.function_deduper.ThunkIdentity` keyed on the
    # imported symbol name for external-target thunks (cross-binary
    # stable) or on the hex entry-point offset for local-target thunks
    # (within-binary stable). Non-thunk functions / non-function
    # addresses: ``None``. The lookup populates this alongside ``name``
    # for the same reason ``comment`` is populated -- so consumers see
    # the same canonical-name basis the FunctionDataManager uses.

    @property
    def section_kind(self) -> SectionKind: ...

    @property
    def section_name(self) -> Optional[str]: ...  # raw ELF section name (e.g. ".rodata")

    @property
    def start_addr(self) -> Optional[int]: ...

    @property
    def end_addr(self) -> Optional[int]: ...

    @property
    def size(self) -> Optional[int]: ...

    @property
    def library(self) -> Optional[str]: ...     # for plt_function / extern_function

    @property
    def string_encoding(self) -> Encoding: ...  # typed enum, not str

    @property
    def string_bytes(self) -> Optional[bytes]: ...

    @property
    def is_vtable(self) -> bool: ...

    @property
    def tls(self) -> bool: ...

    @property
    def slot_target(self) -> Optional["AddressMetadataView"]: ...

    @property
    def jump_table_base_addr(self) -> Optional[int]: ...

    @property
    def jump_table_offset(self) -> Optional[int]: ...

    def __deepcopy__(self, memo) -> "AddressMetadataView": ...


# ---------------------------------------------------------------------------
# String -> enum mapping helpers
# ---------------------------------------------------------------------------
# Providers translate their raw backend strings into typed enums ONCE at
# population time using these helpers. Consumers never see the strings.

_SECTION_KIND_BY_STRING: dict[str, "SectionKind"] = {
    "rodata": SectionKind.RODATA,
    "data": SectionKind.DATA,
    "bss": SectionKind.BSS,
    "thread_local_data": SectionKind.TLS_DATA,
    "tls_data": SectionKind.TLS_DATA,
    "tls_bss": SectionKind.TLS_BSS,
    "plt": SectionKind.PLT,
    "code": SectionKind.CODE,
    "init_array": SectionKind.INIT_ARRAY,
    "fini_array": SectionKind.FINI_ARRAY,
    "ctors": SectionKind.INIT_ARRAY,
    "dtors": SectionKind.FINI_ARRAY,
    "code_ptr_table": SectionKind.INIT_ARRAY,
    "synthetic": SectionKind.SYNTHETIC_EXTERN,
    "synthetic_extern": SectionKind.SYNTHETIC_EXTERN,
    "extern": SectionKind.SYNTHETIC_EXTERN,
}


def section_kind_from_type_string(type_str: Optional[str]) -> "SectionKind":
    """Map a provider type-string to a ``SectionKind`` enum.

    Conservative default: unknown / function / non-section types fall to
    ``SectionKind.UNKNOWN`` so the consumer can detect that the address
    is not section-resident.
    """
    if not type_str:
        return SectionKind.UNKNOWN
    return _SECTION_KIND_BY_STRING.get(type_str.lower(), SectionKind.UNKNOWN)


_ENCODING_BY_STRING: dict[str, "Encoding"] = {
    "ascii": Encoding.ASCII,
    "utf-8": Encoding.UTF8,
    "utf8": Encoding.UTF8,
    "utf-16-le": Encoding.UTF16_LE,
    "utf16-le": Encoding.UTF16_LE,
    "utf-16-be": Encoding.UTF16_BE,
    "utf16-be": Encoding.UTF16_BE,
    "latin-1": Encoding.LATIN_1,
    "latin1": Encoding.LATIN_1,
    "iso-8859-1": Encoding.LATIN_1,
    "utf-32-le": Encoding.UTF_32_LE,
    "utf32-le": Encoding.UTF_32_LE,
    "utf-32-be": Encoding.UTF_32_BE,
    "utf32-be": Encoding.UTF_32_BE,
    "mbcs": Encoding.MBCS,
}


def encoding_from_string(encoding_str: Optional[str]) -> "Encoding":
    """Map a provider encoding-string to an ``Encoding`` enum.

    Unknown / missing -> ``Encoding.UNKNOWN``. Pascal variants and any
    encoding not listed here likewise demote to ``UNKNOWN`` and the
    classifier emits a generic ``string_ptr`` without an encoding hint.
    """
    if not encoding_str:
        return Encoding.UNKNOWN
    return _ENCODING_BY_STRING.get(encoding_str.lower(), Encoding.UNKNOWN)


# Provider-string -> ``AddressKind`` base mapping. Provider lookups
# resolve the final ``AddressKind`` by combining this base with their
# own override signals (is_string / is_vtable / is_plt etc.) at
# population time.
_BASE_KIND_BY_STRING: dict[str, "AddressKind"] = {
    "local_function": AddressKind.LOCAL_FUNCTION,
    "library_function": AddressKind.EXT_FUNCTION_REAL,
    "extern_function": AddressKind.EXT_FUNCTION_REAL,
    "plt_function": AddressKind.PLT_FUNCTION,
    "unknown_function": AddressKind.UNKNOWN,
    "rodata": AddressKind.RODATA,
    "data": AddressKind.DATA,
    "bss": AddressKind.BSS,
    "thread_local_data": AddressKind.THREAD_LOCAL_DATA,
    "code": AddressKind.UNKNOWN,
    "code_ptr_table": AddressKind.RODATA,
    "synthetic": AddressKind.SYNTHETIC_SECTION,
    "synthetic_extern": AddressKind.SYNTHETIC_SECTION,
    "extern": AddressKind.SYNTHETIC_SECTION,
    "unknown": AddressKind.NONE,
}


def address_kind_from_string(type_str: Optional[str]) -> "AddressKind":
    """Map a provider base type-string to an ``AddressKind`` enum.

    Returns ``AddressKind.NONE`` for unknown / missing strings. Providers
    layer override signals (string / vtable / plt / extern-synthetic /
    jump-table) on top of this base when populating the typed view.
    """
    if not type_str:
        return AddressKind.NONE
    return _BASE_KIND_BY_STRING.get(type_str.lower(), AddressKind.NONE)
