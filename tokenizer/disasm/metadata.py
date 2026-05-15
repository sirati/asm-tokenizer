"""Owned address-metadata domain types - lazy view returned by `MetadataLookup`.

Replaces the v1 stringly-keyed dict that `MetadataLookup.lookup(addr)` used
to return. See `tokenizer/disasm/types.py` for the broader lifecycle contract
that also applies to the wrapper here: the returned `AddressMetadataView` is
REUSED across `lookup()` calls and is valid only until the next call. Use
`copy.deepcopy(view)` to stash across lookups.
"""

from enum import IntEnum
from typing import Optional, Protocol, runtime_checkable


class Encoding(IntEnum):
    UNKNOWN = 0
    ASCII = 1
    UTF8 = 2
    UTF16_LE = 3
    UTF16_BE = 4
    LATIN_1 = 5
    PASCAL_ASCII = 6
    PASCAL_UTF16 = 7


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
    """Typed lazy view returned by `MetadataLookup.lookup(addr)`. Replaces
    the stringly-keyed dict that the v1 lookup returned.

    LIFECYCLE: the wrapper is REUSED across `lookup()` calls - valid only
    until the next `lookup()`. `copy.deepcopy(view)` produces a fresh
    wrapper bound to the same per-address provider state (the data is
    stable per binary). `slot_target` is lazily computed on first access
    within a lookup cycle.
    """

    @property
    def kind(self) -> AddressKind: ...

    @property
    def name(self) -> Optional[str]: ...        # human-readable label

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
