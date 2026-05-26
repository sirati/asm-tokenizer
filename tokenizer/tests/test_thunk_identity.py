"""Tests for the :class:`ThunkIdentity` typed identity contract.

Concern: providers (Ghidra ``isExternal()`` thunks, angr PLT stubs +
SimProcedures) emit a :class:`ThunkIdentity` so the deduper sees a
cross-binary-stable third identity axis AND so
:func:`canonical_function_name` renders the imported symbol name as the
``@thunk:<key>`` suffix. These tests pin:

* The dataclass equality / hashability contract.
* The :func:`canonical_function_name` rendering for both target kinds.
* The cross-binary stability invariant for the EXTERNAL case (the bug
  the dataloader peer surfaced: same source symbol surfacing as N
  distinct canonical names because Ghidra's per-binary EXTERNAL
  placeholder offset shifted with link order).
* Legacy integer identity_keys keep their historical rendering so
  pre-typed callers do not silently break.
"""

from __future__ import annotations

from tokenizer.function_deduper import (
    FunctionDeduper,
    ThunkIdentity,
    ThunkTargetKind,
    canonical_function_name,
)


# ---------------------------------------------------------------------------
# ThunkIdentity primitive
# ---------------------------------------------------------------------------


def test_thunk_identity_is_hashable() -> None:
    """The dataclass is frozen + hashable so the deduper can use it as
    a dict key without copy-on-write surprises."""
    ident = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="gzseek")
    assert hash(ident) == hash(
        ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="gzseek")
    )
    {ident: 1}  # raises TypeError if not hashable


def test_thunk_identity_equality_requires_both_axes() -> None:
    """Equality folds in kind AND key — two thunks with the same key
    but different target kinds are DISTINCT identities."""
    extern = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="abc")
    local = ThunkIdentity(kind=ThunkTargetKind.LOCAL, key="abc")
    assert extern != local


# ---------------------------------------------------------------------------
# canonical_function_name rendering
# ---------------------------------------------------------------------------


def test_canonical_name_external_thunk_suffix_is_symbol_name() -> None:
    """``canonical_function_name`` renders an external-target
    ``ThunkIdentity`` as ``<name>@thunk:<symbol_name>``."""
    ident = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="gzseek")
    assert canonical_function_name("gzseek", None, ident) == "gzseek@thunk:gzseek"


def test_canonical_name_local_thunk_suffix_is_hex_offset() -> None:
    """Local-target thunks render with the hex offset key, preserving
    within-binary disambiguation."""
    ident = ThunkIdentity(kind=ThunkTargetKind.LOCAL, key="41e000")
    assert canonical_function_name("alias", None, ident) == "alias@thunk:41e000"


def test_canonical_name_external_thunk_suffix_sanitises_symbol_chars() -> None:
    """Imported symbols can carry characters outside the canonical
    allow-list (e.g. ``glibc@@GLIBC_2.2.5`` for versioned imports).
    The suffix passes through the same sanitisation comments do, so
    the result is CSV / filesystem safe."""
    ident = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="glibc@@GLIBC_2.2.5")
    out = canonical_function_name("glibc", None, ident)
    # ``@`` is outside the allow-list and collapses to ``_``; the
    # trailing version suffix survives via the ``.`` allow-list entry.
    assert out == "glibc@thunk:glibc_GLIBC_2.2.5"


def test_canonical_name_legacy_integer_key_unchanged() -> None:
    """Backward-compat: callers passing a bare integer identity_key
    still receive the historical ``str(int)`` rendering. The typed
    dataclass form is an addition, not a replacement."""
    assert canonical_function_name("strcmp", None, 0xDEAD) == "strcmp@thunk:57005"


# ---------------------------------------------------------------------------
# Cross-binary stability invariant (the bug under repair)
# ---------------------------------------------------------------------------


def test_canonical_name_cross_binary_external_thunk_dedup() -> None:
    """The dataloader peer's symptom: two distinct binaries' EXTERNAL
    blocks place ``gzseek`` at DIFFERENT placeholder offsets, yet
    they're the same source symbol. With :class:`ThunkIdentity` the
    canonical name is keyed on the imported symbol name (NOT the
    placeholder offset), so both binaries' thunks resolve to the SAME
    canonical name — the deduper folds them and the function list
    no longer spawns one entry per ELF."""
    a = canonical_function_name(
        "gzseek",
        None,
        ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="gzseek"),
    )
    b = canonical_function_name(
        "gzseek",
        None,
        ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="gzseek"),
    )
    assert a == b == "gzseek@thunk:gzseek"


def test_deduper_folds_cross_binary_external_thunks() -> None:
    """End-to-end: the same external thunk emitted with the same body
    in two simulated binaries folds into ONE deduper slot (the bug
    pre-fix surfaced N distinct entries — one per ELF — because each
    binary's placeholder offset gave a different identity_key)."""
    ident = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="gzseek")
    deduper = FunctionDeduper()
    r1 = deduper.resolve("gzseek", None, ident, "AAAA")
    r2 = deduper.resolve("gzseek", None, ident, "AAAA")
    assert r1.is_duplicate is False
    assert r2.is_duplicate is True
    assert r1.slot_id == r2.slot_id


def test_deduper_distinguishes_external_from_local_same_key() -> None:
    """The kind axis matters in the deduper: an EXTERNAL ``"foo"`` and
    a LOCAL ``"foo"`` resolve to distinct slot_ids even when the body
    matches (defensive — though the LOCAL case keys on hex offsets in
    practice, the contract guarantees no accidental collision)."""
    extern = ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key="foo")
    local = ThunkIdentity(kind=ThunkTargetKind.LOCAL, key="foo")
    deduper = FunctionDeduper()
    r_extern = deduper.resolve("foo", None, extern, "AAAA")
    r_local = deduper.resolve("foo", None, local, "AAAA")
    assert r_extern.is_duplicate is False
    assert r_local.is_duplicate is False
    assert r_extern.slot_id != r_local.slot_id
