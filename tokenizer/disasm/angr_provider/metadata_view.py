"""Typed view for the angr-side MetadataLookup.

Owns ``_AngrAddressMetadataView``: pure storage + read-only typed
properties. ``AngrMetadataLookup`` calls ``_populate(...)`` to populate
every slot at lookup time; consumers read typed properties exclusively.
angr cannot resolve slot targets (``angr_limitations.md`` sections 2-3),
so ``slot_target`` / ``jump_table_base_addr`` / ``jump_table_offset``
always return ``None``; ``string_encoding`` is always ASCII or UNKNOWN
(``angr_limitations.md`` section 4).
"""

from __future__ import annotations

from typing import Hashable, Optional

from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
)


# ---------------------------------------------------------------------------
# Typed view for the angr-side MetadataLookup
# ---------------------------------------------------------------------------
class _AngrAddressMetadataView:
    """Concrete typed view returned by ``AngrMetadataLookup.lookup()``.

    Pure storage + read-only typed properties. ``AngrMetadataLookup`` calls
    ``_populate(...)`` to populate every slot at lookup time; consumers
    read typed properties exclusively. angr cannot resolve slot targets
    (``angr_limitations.md`` sections 2-3), so ``slot_target`` /
    ``jump_table_base_addr`` / ``jump_table_offset`` always return
    ``None``; ``string_encoding`` is always ASCII or UNKNOWN
    (``angr_limitations.md`` section 4).

    LIFECYCLE: instance is REUSED across ``lookup()`` calls. Use
    ``copy.deepcopy(view)`` to stash across lookups.
    """

    __slots__ = (
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
    )

    def __init__(self) -> None:
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
    ) -> None:
        """Replace all slot state in one call. Used by the lookup at
        the start of every ``lookup()`` so the consumer sees a consistent
        view bound to the current address.

        ``name`` is the CANONICAL function name (already passed through
        :func:`tokenizer.function_deduper.canonical_function_name` by the
        lookup); on the angr path the helper is a no-op because angr has
        no demangler hook and no thunk-identity surface (so the two
        axes ``comment`` / ``identity_key`` are always ``None`` here).
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
    def comment(self) -> Optional[str]:
        # angr has no demangler hook; comment is always None on this
        # path (and the canonical-name helper trivially short-circuits to
        # the raw name when both axes are None).
        return None

    @property
    def identity_key(self) -> Optional[Hashable]:
        # angr has no thunk-identity surface (see angr_limitations.md).
        return None

    @property
    def slot_target(self) -> Optional[AddressMetadataView]:
        # angr cannot resolve slot targets (angr_limitations.md sections 2-3).
        return None

    @property
    def jump_table_base_addr(self) -> Optional[int]:
        return None

    @property
    def jump_table_offset(self) -> Optional[int]:
        return None

    def __deepcopy__(self, memo) -> "_AngrAddressMetadataView":
        clone = _AngrAddressMetadataView()
        clone._kind = self._kind
        clone._section_kind = self._section_kind
        clone._section_name = self._section_name
        clone._string_encoding = self._string_encoding
        # bytes is immutable; safe to share
        clone._string_bytes = self._string_bytes
        clone._name = self._name
        clone._start_addr = self._start_addr
        clone._end_addr = self._end_addr
        clone._size = self._size
        clone._library = self._library
        clone._is_vtable = self._is_vtable
        clone._tls = self._tls
        return clone
