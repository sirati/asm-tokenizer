"""Tests for canonical-name cascade through the metadata-lookup layer.

Concern: the typed ``AddressMetadataView`` populated by both providers
(Ghidra default, angr best-effort) must surface ``.name`` already
collapsed through
:func:`tokenizer.function_deduper.canonical_function_name`. This is the
invariant the v2 metadata emitters and the FunctionDataManager both
rely on so caller-side ``meta.name`` is byte-identical to the
callee-side ``add_function_data`` final name (a downstream
``function_lookup[(name, vkey)]`` resolver MISSES otherwise — the
reason this cascade is one cohesive change).

This file exercises the angr-side ``_AngrAddressMetadataView`` (its
``comment`` / ``identity_key`` properties are unconditional None
sentinels) AND the Ghidra-side ``_GhidraAddressMetadataView``
(populated via the ``_populate`` keyword API). Full end-to-end
exercises (against a real Ghidra-loaded program) live elsewhere; this
file checks the typed-view boundary in isolation.
"""

from __future__ import annotations

from tokenizer.disasm.angr_provider.metadata_view import (
    _AngrAddressMetadataView,
)
from tokenizer.disasm.ghidra_provider.metadata_view import (
    _GhidraAddressMetadataView,
)
from tokenizer.disasm.metadata import (
    AddressKind,
    Encoding,
    SectionKind,
)
from tokenizer.function_deduper import canonical_function_name


# ---------------------------------------------------------------------------
# Angr-side typed view
# ---------------------------------------------------------------------------


def test_angr_view_comment_property_is_none() -> None:
    """The angr path has no demangler hook (see
    ``angr_limitations.md``); the ``comment`` property is the
    unconditional None sentinel."""
    view = _AngrAddressMetadataView()
    assert view.comment is None


def test_angr_view_identity_key_property_is_none() -> None:
    """The angr path has no thunk-identity surface; the
    ``identity_key`` property is the unconditional None sentinel."""
    view = _AngrAddressMetadataView()
    assert view.identity_key is None


# ---------------------------------------------------------------------------
# Ghidra-side typed view -- populated via _populate(...)
# ---------------------------------------------------------------------------


class _NopLookup:
    """Standalone lookup stub for slot-target tests.

    The ``_GhidraAddressMetadataView`` holds a back-reference to its
    lookup for ``slot_target`` lazy resolution. None of the assertions
    below touch ``slot_target``, but the wrapper still needs the
    reference; this stub satisfies the type without pulling in a real
    Ghidra ``Program``.
    """

    pass


def _populate_view(
    *,
    name: str,
    comment: object = None,
    identity_key: object = None,
) -> _GhidraAddressMetadataView:
    """Build a typed Ghidra view populated through the lookup-side
    ``_populate`` API, matching the canonical cascade the production
    ``GhidraMetadataLookup._classify_address`` runs.
    """
    view = _GhidraAddressMetadataView(_NopLookup())
    canonical = canonical_function_name(name, comment, identity_key)
    view._populate(
        kind=AddressKind.LOCAL_FUNCTION,
        section_kind=SectionKind.CODE,
        section_name=".text",
        string_encoding=Encoding.UNKNOWN,
        string_bytes=None,
        name=canonical,
        comment=comment if comment != "" else None,
        identity_key=identity_key,
        start_addr=0x1000,
        end_addr=0x1010,
        size=0x10,
        library=None,
        is_vtable=False,
        tls=False,
        slot_target_addr=None,
        jump_table_base_addr=None,
        jump_table_offset=None,
    )
    return view


def test_ghidra_view_name_is_canonical_when_comment_set() -> None:
    """When the lookup populates a function-shaped view with a
    demangled comment, ``meta.name`` is the canonical
    ``<raw_name>@<sanitised_comment>`` form. The comment + identity_key
    axes are also surfaced separately so consumers can audit the
    canonicalization."""
    view = _populate_view(
        name="reset",
        comment="ARPHeader::reset(void)",
    )
    assert view.name == "reset@ARPHeader::reset(void)"
    assert view.comment == "ARPHeader::reset(void)"
    assert view.identity_key is None


def test_ghidra_view_name_is_canonical_when_identity_key_set() -> None:
    """PLT-thunk case: comment is None, identity_key carries the
    resolved-external entry-offset; ``meta.name`` is the thunk-keyed
    canonical form."""
    view = _populate_view(
        name="strcmp",
        identity_key=0xDEAD,
    )
    assert view.name == "strcmp@57005:thunk"
    assert view.comment is None
    assert view.identity_key == 0xDEAD


def test_ghidra_view_name_unchanged_when_axes_absent() -> None:
    """When both axes are absent (the legacy C / asm case), the
    canonical name is the raw name verbatim — no decorative suffix
    forced on the common case."""
    view = _populate_view(name="main")
    assert view.name == "main"
    assert view.comment is None
    assert view.identity_key is None


def test_ghidra_view_distinct_overloads_produce_distinct_names() -> None:
    """The cross-ISA-stable invariant: two C++ methods that share an
    unqualified name (the classic ``ARPHeader::reset`` vs
    ``EthernetHeader::reset``) surface DISTINCT canonical names. This
    is what the caller-side meta.name receives for each, so the
    downstream ``function_lookup[(name, vkey)]`` resolver can keep
    them apart."""
    a = _populate_view(
        name="reset", comment="ARPHeader::reset(void)"
    )
    b = _populate_view(
        name="reset", comment="EthernetHeader::reset(void)"
    )
    assert a.name != b.name
    assert a.name == "reset@ARPHeader::reset(void)"
    assert b.name == "reset@EthernetHeader::reset(void)"


def test_ghidra_view_deepcopy_preserves_canonical_axes() -> None:
    """``__deepcopy__`` snapshots the current state -- the new fields
    ``_comment`` + ``_identity_key`` must be carried into the clone or
    a stashed view drifts from the canonical name on the cursor view."""
    import copy

    view = _populate_view(
        name="reset", comment="ARPHeader::reset(void)"
    )
    clone = copy.deepcopy(view)
    assert clone.name == view.name
    assert clone.comment == view.comment
    assert clone.identity_key == view.identity_key
