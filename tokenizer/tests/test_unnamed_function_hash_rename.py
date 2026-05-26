"""Tests for the Ghidra-placeholder-function rename helper.

Concern: ``tokenizer.disasm.ghidra_views.unnamed_rename`` turns every
``SourceType.DEFAULT`` Ghidra function name (the ``FUN_<hex>`` family)
into a deterministic, binary-scoped, collision-free
``unnamed @{base64-of-128bit-hash}`` opaque label. Real symbol names
(``IMPORTED`` / ``USER_DEFINED`` / ``ANALYSIS``) flow through
unchanged. The tests cover:

* Each ``SourceType`` value: DEFAULT renames, others pass through.
* Cross-binary distinctness: same raw placeholder, two binary
  identity hashes → two distinct renamed labels.
* Determinism: same inputs → same label, repeatedly.
* Structural non-collision with any real C / C++ / ELF / PE symbol
  (the LITERAL SPACE inside ``unnamed @`` is disallowed in every
  symbol grammar in our pipeline).
* End-to-end wiring through ``_GhidraFunctionView._advance``: the
  view applies the rename to the field consumers see as
  ``view.name``.

All tests use hand-rolled mocks (no JVM); JPype-style enum rendering
is matched by mocking ``getSource()`` to return an object whose
``str(...)`` is the enum constant name (matches the production
JPype boundary).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tokenizer.disasm.ghidra_views.function import _GhidraFunctionView
from tokenizer.disasm.ghidra_views.unnamed_rename import (
    PLACEHOLDER_PREFIX,
    placeholder_renamed_name,
)


# ---------------------------------------------------------------------------
# Mock SourceType + Symbol + Function (JPype-compatible: str() returns the
# enum constant name, just like JPype renders Java enums).
# ---------------------------------------------------------------------------


class _MockSourceType:
    """Mimic ``str(SourceType.X) == "X"`` per JPype's enum rendering."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name


_DEFAULT = _MockSourceType("DEFAULT")
_IMPORTED = _MockSourceType("IMPORTED")
_USER_DEFINED = _MockSourceType("USER_DEFINED")
_ANALYSIS = _MockSourceType("ANALYSIS")


class _MockSymbol:
    __slots__ = ("_source",)

    def __init__(self, source: Any) -> None:
        self._source = source

    def getSource(self) -> Any:
        return self._source


class _MockAddress:
    __slots__ = ("_offset",)

    def __init__(self, offset: int) -> None:
        self._offset = offset

    def getOffset(self) -> int:
        return self._offset


class _MockFunction:
    """Minimal Ghidra ``Function`` mock for ``_advance`` end-to-end."""

    __slots__ = ("_name", "_symbol", "_entry")

    def __init__(self, name: str, source: Any, entry: int) -> None:
        self._name = name
        self._symbol = _MockSymbol(source)
        self._entry = _MockAddress(entry)

    def getName(self) -> str:
        return self._name

    def getSymbol(self) -> _MockSymbol:
        return self._symbol

    def getEntryPoint(self) -> _MockAddress:
        return self._entry


def _make_view(binary_id_hash: bytes) -> _GhidraFunctionView:
    """Construct a function view skeleton with stubbed provider state.

    The fields the rename code path touches are ``_binary_id_hash``,
    ``_ghidra_function``, ``_entry``, ``_name``; the block-iteration
    machinery is not exercised here. ``program/listing/reg_map/...``
    can be ``None`` because ``_advance`` does not touch them on the
    rename hot path.
    """
    return _GhidraFunctionView(
        arch=None,  # type: ignore[arg-type]
        program=None,
        listing=None,
        reg_map=None,
        decode=None,
        block_model=None,
        monitor=None,
        binary_id_hash=binary_id_hash,
    )


# Deterministic 16-byte hashes for the cross-binary tests. The exact
# byte values do not matter; the property under test is that two
# DIFFERENT identity hashes produce two DIFFERENT renamed labels for
# the same raw placeholder.
_HASH_A = bytes(range(16))
_HASH_B = bytes(range(16, 32))


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_default_source_renames_to_unnamed_prefix() -> None:
    """A ``SourceType.DEFAULT`` placeholder collapses into the
    ``unnamed @{b64hash}`` opaque label."""
    out = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_A)
    assert out.startswith(PLACEHOLDER_PREFIX)
    # 16-byte digest → base64 → 22 ASCII chars after stripping ``=``
    # padding, drawn from the URL-safe alphabet ``[A-Za-z0-9_-]``.
    label = out[len(PLACEHOLDER_PREFIX) :]
    assert re.fullmatch(r"[A-Za-z0-9_-]{22}", label), label


def test_imported_source_passes_through() -> None:
    """``SourceType.IMPORTED`` (PLT-imported externals) is a real
    symbol; the helper does not touch it."""
    assert placeholder_renamed_name("memcpy", _IMPORTED, _HASH_A) == "memcpy"


def test_user_defined_source_passes_through() -> None:
    """``SourceType.USER_DEFINED`` (analyst-set names) is real."""
    assert placeholder_renamed_name("MyHelper", _USER_DEFINED, _HASH_A) == "MyHelper"


def test_analysis_source_passes_through() -> None:
    """``SourceType.ANALYSIS`` (recovered by Ghidra's analyzers, e.g.
    function signature ID, demangler) is real."""
    assert (
        placeholder_renamed_name("_strlen_avx2", _ANALYSIS, _HASH_A) == "_strlen_avx2"
    )


def test_same_name_different_binaries_get_different_hashes() -> None:
    """The per-binary identity hash XORs into the rendered label, so
    the same raw placeholder in two different binaries yields two
    distinct opaque labels (the user-specified property)."""
    a = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_A)
    b = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_B)
    assert a != b


def test_same_name_same_binary_is_deterministic() -> None:
    """Calling the helper twice with the same inputs is a pure
    function — same output."""
    a = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_A)
    b = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_A)
    assert a == b


def test_different_names_same_binary_get_different_hashes() -> None:
    """Each placeholder (Ghidra embeds the entry offset in the name)
    hashes independently; two distinct placeholders in the same
    binary cannot collide post-rename."""
    a = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_A)
    b = placeholder_renamed_name("FUN_00020000", _DEFAULT, _HASH_A)
    assert a != b


def test_collision_with_real_symbol_structurally_impossible() -> None:
    """The literal SPACE between ``unnamed`` and ``@`` is disallowed
    in every symbol grammar in our pipeline (ELF / PE / C / C++
    mangled / demangled, Rust mangled, Go). Asserting the property
    directly: the prefix contains a space, and any real-symbol regex
    we use to validate identifiers rejects strings with spaces.
    """
    out = placeholder_renamed_name("FUN_00010000", _DEFAULT, _HASH_A)
    assert " " in out, out
    # The conservative real-symbol identifier regex used across linker
    # ecosystems we ingest (mangled or not): no whitespace, no
    # control characters. The rename's literal space therefore
    # guarantees structural non-collision.
    real_symbol_re = re.compile(r"^[^\s\x00-\x1f]+$")
    assert real_symbol_re.match("memcpy")
    assert real_symbol_re.match("_ZN9ARPHeader5resetEv")  # C++ mangled
    assert not real_symbol_re.match(out)


# ---------------------------------------------------------------------------
# End-to-end: ``_GhidraFunctionView._advance`` applies the rename.
# ---------------------------------------------------------------------------


def test_view_advance_renames_default_source_function() -> None:
    """The view-layer integration: a DEFAULT-source function backing
    the cursor surfaces the renamed label as ``view.name``."""
    view = _make_view(_HASH_A)
    func = _MockFunction(name="FUN_00010000", source=_DEFAULT, entry=0x10000)
    view._advance(func, block_count=1)
    assert view.name.startswith(PLACEHOLDER_PREFIX)
    assert view.entry == 0x10000


def test_view_advance_preserves_real_symbol_name() -> None:
    """The view-layer integration: a real-symbol function passes
    through with its name intact."""
    view = _make_view(_HASH_A)
    func = _MockFunction(name="memcpy", source=_IMPORTED, entry=0x401000)
    view._advance(func, block_count=1)
    assert view.name == "memcpy"


def test_view_advance_same_view_two_default_functions_distinct() -> None:
    """Re-advance the cursor over two DEFAULT-source functions in
    the same binary: two distinct names."""
    view = _make_view(_HASH_A)
    func_a = _MockFunction(name="FUN_00010000", source=_DEFAULT, entry=0x10000)
    func_b = _MockFunction(name="FUN_00020000", source=_DEFAULT, entry=0x20000)
    view._advance(func_a, block_count=1)
    name_a = view.name
    view._advance(func_b, block_count=1)
    name_b = view.name
    assert name_a != name_b


def test_view_advance_cross_binary_same_default_function_distinct() -> None:
    """Two view instances bound to two different binaries (different
    identity hashes) see two different renamed labels for the same
    raw placeholder."""
    view_a = _make_view(_HASH_A)
    view_b = _make_view(_HASH_B)
    func = _MockFunction(name="FUN_00010000", source=_DEFAULT, entry=0x10000)
    view_a._advance(func, block_count=1)
    view_b._advance(func, block_count=1)
    assert view_a.name != view_b.name


# ---------------------------------------------------------------------------
# Binary-identity-hash sidecar handling.
# ---------------------------------------------------------------------------


def test_binary_identity_hash_uses_path_when_no_sidecar(tmp_path) -> None:
    """Legacy 4-axis dataset entries have no adjacent sidecar JSON;
    the identity hash collapses to the path-only digest. We assert
    the helper does not crash and returns a 16-byte value."""
    from tokenizer.disasm.ghidra_views.unnamed_rename import compute_binary_identity_hash

    binary = tmp_path / "legacy_binary"
    binary.write_bytes(b"not used")  # content is irrelevant; path is what hashes
    h = compute_binary_identity_hash(binary)
    assert isinstance(h, bytes)
    assert len(h) == 16


def test_binary_identity_hash_folds_sidecar_when_present(tmp_path) -> None:
    """When the dataset's sidecar-folder format places a
    ``<stem>.json`` next to the variant directory, the identity hash
    folds the sidecar content in. Two identical binaries with two
    different sidecar contents therefore get two different identity
    hashes — the user-specified property of the sidecar fold."""
    from tokenizer.disasm.ghidra_views.unnamed_rename import compute_binary_identity_hash

    # Sidecar layout: <dir>/<stem>/<pkg> with <dir>/<stem>.json
    parent = tmp_path
    stem = "variantA"
    variant_dir = parent / stem
    variant_dir.mkdir()
    pkg = variant_dir / "binary"
    pkg.write_bytes(b"")

    # No sidecar yet.
    h_no_sidecar = compute_binary_identity_hash(pkg)

    # Sidecar with one content.
    (parent / f"{stem}.json").write_text('{"k": "v1"}')
    h_v1 = compute_binary_identity_hash(pkg)

    # Sidecar with different content.
    (parent / f"{stem}.json").write_text('{"k": "v2"}')
    h_v2 = compute_binary_identity_hash(pkg)

    assert h_no_sidecar != h_v1
    assert h_v1 != h_v2


def test_binary_identity_hash_sidecar_canonicalisation(tmp_path) -> None:
    """Cosmetic JSON formatting differences (whitespace, key order)
    don't shift the identity hash — the helper canonicalises the JSON
    before hashing."""
    from tokenizer.disasm.ghidra_views.unnamed_rename import compute_binary_identity_hash

    parent = tmp_path
    stem = "variantA"
    variant_dir = parent / stem
    variant_dir.mkdir()
    pkg = variant_dir / "binary"
    pkg.write_bytes(b"")

    (parent / f"{stem}.json").write_text('{"a": 1, "b": 2}')
    h1 = compute_binary_identity_hash(pkg)

    # Re-dump with different key order + whitespace; should hash the same.
    (parent / f"{stem}.json").write_text(' {  "b": 2,\n  "a": 1 } ')
    h2 = compute_binary_identity_hash(pkg)

    assert h1 == h2


# ---------------------------------------------------------------------------
# Parametrized SourceType matrix — every Ghidra source label we care
# about flows through correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expect_rename",
    [
        (_DEFAULT, True),
        (_IMPORTED, False),
        (_USER_DEFINED, False),
        (_ANALYSIS, False),
    ],
)
def test_source_type_matrix(source: Any, expect_rename: bool) -> None:
    out = placeholder_renamed_name("FUN_00010000", source, _HASH_A)
    is_renamed = out.startswith(PLACEHOLDER_PREFIX)
    assert is_renamed == expect_rename
