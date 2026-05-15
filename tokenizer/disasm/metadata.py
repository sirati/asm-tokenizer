"""Owned address-metadata domain types - lazy view returned by `MetadataLookup`.

Replaces the v1 stringly-keyed dict that `MetadataLookup.lookup(addr)` used
to return. See `tokenizer/disasm/types.py` for the broader lifecycle contract
that also applies to the wrapper here: the returned `AddressMetadataView` is
REUSED across `lookup()` calls and is valid only until the next call. Use
`copy.deepcopy(view)` to stash across lookups.

Transitional adapter - until Phase D.3 (task #40) migrates `ConstantHandler`
off of `meta.get(...)` / `meta[...]` access, the concrete view classes ALSO
implement a `Mapping`-shaped surface (``__getitem__`` / ``get`` / ``keys`` /
``__contains__`` / ``copy``) returning the v1 dict shape. Tuple unpacking
``meta, kind = lookup.lookup(addr)`` is also preserved through ``__iter__``
yielding ``(self, kind_string)``. The ABC contract (`MetadataLookup.lookup`)
returns `AddressMetadataView`; the unpacking + dict-like surface is purely
a Phase-D.2-bridge implementation detail. ``legacy_dict()`` exposes the
canonical v1 shape directly so D.3 can grep-and-replace without ambiguity.
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

    def legacy_dict(self) -> tuple[dict, str]:
        """Transitional - returns the v1 ``(meta_dict, kind_string)`` tuple.

        Phase-D.2 bridge: until ``ConstantHandler`` migrates to typed
        property access (Phase D.3 / task #40), call sites that need the
        v1 dict shape invoke this to get back the exact pre-typing
        payload. Removed when every ``meta.get(...)`` / ``meta[...]``
        site is gone.
        """
        ...


# ---------------------------------------------------------------------------
# String -> enum mapping helpers
# ---------------------------------------------------------------------------
# Both providers' v1 dict carries free-form strings for ``type`` and
# ``string_encoding``; the mapping logic is shared so we keep it here next
# to the enum definitions. Provider-specific concerns (which raw signals
# to inspect, which fields to populate) live in the provider files.

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
    """Map a v1 ``meta["type"]`` string to a ``SectionKind`` enum.

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
}


def encoding_from_string(encoding_str: Optional[str]) -> "Encoding":
    """Map a v1 ``meta["string_encoding"]`` string to an ``Encoding`` enum.

    Unknown / missing -> ``Encoding.UNKNOWN``. Pascal variants and any
    encoding not listed here likewise demote to ``UNKNOWN`` and the
    classifier emits a generic ``string_ptr`` without an encoding hint.
    """
    if not encoding_str:
        return Encoding.UNKNOWN
    return _ENCODING_BY_STRING.get(encoding_str.lower(), Encoding.UNKNOWN)


# ``meta["type"]`` synthesis for ``AddressKind``. The function-vs-data
# distinction is driven by the literal type string; the boolean flags
# (``is_plt`` / ``is_extern_synthetic``) override the function classification
# at a higher precedence. The slot / string / vtable kinds are encoded as
# overrides over the section-derived base.
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


def address_kind_from_meta(meta: dict) -> "AddressKind":
    """Synthesize an ``AddressKind`` from the v1 ``meta`` dict.

    Precedence (highest first), mirroring the v2 classifier in
    ``constant_handler.py:_PRECEDENCE`` so ``kind`` is the same
    discriminator the classifier would route on if it consumed enums
    directly:

    1. ``is_string`` -> ``STRING`` (precedence step 7)
    2. ``is_jump_table_slot`` -> ``JUMP_TABLE_SLOT`` (precedence step 8)
    3. ``is_vtable`` / ``is_code_ptr_table_slot`` -> ``CODE_PTR_TABLE_SLOT``
       (precedence step 8)
    4. ``is_plt`` -> ``PLT_FUNCTION`` (precedence step 2)
    5. ``is_extern_synthetic`` (function-typed) -> ``EXT_FUNCTION_SYNTHETIC``
       (precedence step 6)
    6. literal ``meta["type"]`` -> base kind via ``_BASE_KIND_BY_STRING``
       (precedence steps 3-5, 9-10)
    7. unknown / missing -> ``AddressKind.NONE``

    ``BLOCK_IN_FUNCTION`` is NOT discriminated here because the view does
    not see the address being looked up; the consumer compares
    ``view.start_addr`` against the address to discriminate function entry
    vs. body. ``LOCAL_FUNCTION`` covers both cases at the view level.
    """
    if meta.get("is_string"):
        return AddressKind.STRING
    if meta.get("is_jump_table_slot"):
        return AddressKind.JUMP_TABLE_SLOT
    if meta.get("is_vtable") or meta.get("is_code_ptr_table_slot"):
        return AddressKind.CODE_PTR_TABLE_SLOT
    if meta.get("is_plt"):
        return AddressKind.PLT_FUNCTION
    type_str = (meta.get("type") or "").lower()
    if meta.get("is_extern_synthetic") and type_str in {
        "extern_function",
        "library_function",
        "plt_function",
        "unknown_function",
    }:
        return AddressKind.EXT_FUNCTION_SYNTHETIC
    return _BASE_KIND_BY_STRING.get(type_str, AddressKind.NONE)


# ---------------------------------------------------------------------------
# Mapping-shaped transitional surface
# ---------------------------------------------------------------------------
class _DictBackedAddressMetadataView:
    """Base implementation: stores a v1 dict + kind string, exposes typed
    properties + ``Mapping`` shim + tuple unpacking + ``legacy_dict``.

    Subclasses (one per provider) override ``slot_target`` resolution and
    any provider-specific typed fields (e.g. ``string_encoding`` from a
    Ghidra ``DataType`` rather than the dict's string). The base class
    handles the transitional bridge so the per-provider concrete classes
    only express their resolution-of-typed-fields concern.

    LIFECYCLE - the wrapper is REUSED across ``lookup()`` calls. Each
    ``lookup()`` rebuilds the v1 dict + kind string and assigns them to
    ``self._dict`` / ``self._kind_string`` then returns ``self``. A
    ``deepcopy(view)`` produces a fresh wrapper bound to a deep-copied
    snapshot of the current state - independent of the cursor.
    """

    __slots__ = ("_dict", "_kind_string")

    def __init__(self, initial_dict: Optional[dict] = None, initial_kind: str = "synthetic") -> None:
        self._dict: dict = initial_dict if initial_dict is not None else {}
        self._kind_string: str = initial_kind

    # -- Cursor advance: subclasses call this in their lookup-cycle hook ----
    def _set_state(self, new_dict: dict, new_kind: str) -> None:
        self._dict = new_dict
        self._kind_string = new_kind

    # -- Typed property surface (AddressMetadataView Protocol) --------------
    @property
    def kind(self) -> "AddressKind":
        return address_kind_from_meta(self._dict)

    @property
    def name(self) -> Optional[str]:
        v = self._dict.get("name")
        return None if v is None else str(v)

    @property
    def section_kind(self) -> "SectionKind":
        return section_kind_from_type_string(self._dict.get("type"))

    @property
    def section_name(self) -> Optional[str]:
        # v1 conflated section_name with the meta["name"] (audit-2 noted
        # this). Subclasses override when they have the raw section name
        # separately. Default returns None so callers don't accidentally
        # consume the conflated value.
        return self._dict.get("section_name")

    @property
    def start_addr(self) -> Optional[int]:
        v = self._dict.get("start_addr")
        return None if v is None else int(v)

    @property
    def end_addr(self) -> Optional[int]:
        v = self._dict.get("end_addr")
        return None if v is None else int(v)

    @property
    def size(self) -> Optional[int]:
        v = self._dict.get("size")
        return None if v is None else int(v)

    @property
    def library(self) -> Optional[str]:
        v = self._dict.get("library")
        if v is None or v == "unknown":
            return None
        return str(v)

    @property
    def string_encoding(self) -> "Encoding":
        return encoding_from_string(self._dict.get("string_encoding"))

    @property
    def string_bytes(self) -> Optional[bytes]:
        v = self._dict.get("string_bytes")
        return None if v is None else bytes(v)

    @property
    def is_vtable(self) -> bool:
        return bool(self._dict.get("is_vtable"))

    @property
    def tls(self) -> bool:
        return bool(self._dict.get("tls"))

    @property
    def slot_target(self) -> Optional["AddressMetadataView"]:
        # Subclasses with slot-resolution capability override; default is
        # ``None`` (angr cannot resolve slot targets per
        # ``angr_limitations.md`` sections 2 + 3).
        return None

    @property
    def jump_table_base_addr(self) -> Optional[int]:
        # Subclasses override; default ``None``.
        return None

    @property
    def jump_table_offset(self) -> Optional[int]:
        # Subclasses override; default ``None``.
        return None

    # -- Transitional Mapping-shaped bridge ---------------------------------
    # ConstantHandler reads ``meta.get("name")`` / ``meta["start_addr"]``
    # and does ``"key" in meta`` / ``dict(meta)``. Until Phase D.3 migrates
    # those call sites to the typed properties above, this bridge keeps the
    # old shape working.
    def get(self, key, default=None):
        return self._dict.get(key, default)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        # ``constant_handler.process_constant_v2`` stamps an internal
        # ``__value__`` slot on the dict; supporting __setitem__ keeps
        # that working without forcing the caller to round-trip through
        # ``dict(meta)``. Internal protocol only; goes away with D.3.
        self._dict[key] = value

    def __contains__(self, key) -> bool:
        return key in self._dict

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def copy(self) -> dict:
        # Returns a dict copy (NOT a view copy) to match v1 ``meta.copy()``
        # semantics. Use ``copy.deepcopy(view)`` for a view-shaped copy.
        return dict(self._dict)

    # -- Tuple unpacking: ``meta, kind = lookup.lookup(addr)`` --------------
    def __iter__(self):
        # Yields exactly two elements so the legacy unpacking works.
        # ``dict(view)`` does NOT use this path because ``.keys()`` is
        # present; ``dict()`` of a Mapping-shaped object calls ``.keys()``
        # and indexes through ``__getitem__``.
        yield self
        yield self._kind_string

    # -- Legacy adapter -----------------------------------------------------
    def legacy_dict(self) -> tuple[dict, str]:
        """Return the canonical v1 ``(meta_dict, kind_string)`` tuple.

        The dict is the internal storage (NOT copied) because Phase-D.3
        consumers will be doing read-only access. If a caller mutates the
        returned dict it mutates the wrapper's state - intended for the
        ``__value__`` stamp pattern in ``constant_handler``.
        """
        return self._dict, self._kind_string

    # -- Deepcopy: snapshot semantics ---------------------------------------
    def __deepcopy__(self, memo) -> "AddressMetadataView":
        import copy as _copy
        cls = type(self)
        clone = cls.__new__(cls)
        # Subclasses with extra slots must override and chain through here.
        clone._dict = _copy.deepcopy(self._dict, memo)
        clone._kind_string = self._kind_string
        return clone
