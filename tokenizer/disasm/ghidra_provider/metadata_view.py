"""Concrete typed view returned by ``GhidraMetadataLookup.lookup()``.

Owns ``_GhidraAddressMetadataView``: pure storage + read-only typed
properties. ``GhidraMetadataLookup`` populates every typed slot at
lookup time via ``_populate``; consumers read typed properties exclusively.
The instance is REUSED across ``lookup()`` calls. Use
``copy.deepcopy(view)`` to stash across lookups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, Optional

from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
)

if TYPE_CHECKING:
    from tokenizer.disasm.ghidra_provider.metadata_lookup import GhidraMetadataLookup


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
        "_comment",
        "_identity_key",
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
        self._comment: Optional[str] = None
        self._identity_key: Optional[Hashable] = None
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
        comment: Optional[str],
        identity_key: Optional[Hashable],
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

        ``name`` is the CANONICAL function name (already passed through
        :func:`tokenizer.function_deduper.canonical_function_name` by the
        lookup so callers / emitters get a cross-ISA-stable identifier).
        ``comment`` + ``identity_key`` are the two axes the lookup fed to
        that helper and are exposed here for consumers that need to
        re-derive or audit the canonical name.
        """
        self._kind = kind
        self._section_kind = section_kind
        self._section_name = section_name
        self._string_encoding = string_encoding
        self._string_bytes = string_bytes
        self._name = name
        self._comment = comment
        self._identity_key = identity_key
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
    def comment(self) -> Optional[str]:
        return self._comment

    @property
    def identity_key(self) -> Optional[Hashable]:
        return self._identity_key

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
        clone._comment = self._comment
        clone._identity_key = self._identity_key
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
