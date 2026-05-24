"""Tests for ``function_metadata_snapshot.snapshot_function``.

Concern: the Java-handle introspector. Pin (a) depth-5 terminal
repr-string truncation, (b) graceful handling of getter exceptions /
missing getters, (c) curated-getter-list-per-type fan-out,
(d) collection (Java array / Iterable) traversal.

The orchestrator side (collision detection + pickle write) lives in
``test_duplicate_function_dump``.

Shared mock scaffolding (``MockJavaObject``, ``MockJavaClass``,
``build_ns_chain_to_layer5``) lives in ``_duplicate_function_dump_mocks``.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.ghidra_provider.function_metadata_snapshot import (
    snapshot_function,
)
from tokenizer.tests._duplicate_function_dump_mocks import (
    MockJavaClass,
    MockJavaObject,
    build_ns_chain_to_layer5,
)


def test_snapshot_records_function_layer1_getters() -> None:
    """L1: direct getter results on the Function root land at top-level keys."""
    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={
            "getName": "foo",
            "getEntryPoint": 0x401000,
            "isInline": False,
            "hasNoReturn": True,
        },
    )
    snap = snapshot_function(func)
    assert snap["_java_class"] == "ghidra.program.model.listing.FunctionDB"
    assert snap["Name"] == "foo"
    assert snap["EntryPoint"] == 0x401000
    assert snap["isInline"] is False
    assert snap["hasNoReturn"] is True


def test_snapshot_recurses_into_symbol_at_layer2() -> None:
    """L2: ``getSymbol`` returns a Symbol; the Symbol's curated getters
    fan out as L2 sub-keys.
    """
    symbol = MockJavaObject(
        java_class="ghidra.program.model.symbol.SymbolDB",
        getters={"getName": "foo", "getSource": "USER_DEFINED", "isPrimary": True},
    )
    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "foo", "getSymbol": symbol},
    )
    snap = snapshot_function(func)
    assert isinstance(snap["Symbol"], dict)
    assert snap["Symbol"]["Name"] == "foo"
    assert snap["Symbol"]["Source"] == "USER_DEFINED"
    assert snap["Symbol"]["isPrimary"] is True


def test_snapshot_terminal_leaf_at_layer5_is_repr_string() -> None:
    """L5: getter results at depth 5 must be ``str(repr(value))``
    regardless of the value's underlying type.

    Walk: ``func`` (depth=0 dict root) -> ``Symbol`` (L1 dict) ->
    ``ParentNamespace`` (L2 dict) -> ``ParentNamespace`` (L3 dict) ->
    ``ParentNamespace`` (L4 dict) -> ``ParentNamespace`` (L5: repr-string).
    """
    func, _symbol, _l2, _l3, _l4 = build_ns_chain_to_layer5()
    snap = snapshot_function(func)
    l1 = snap["Symbol"]
    l2 = l1["ParentNamespace"]
    l3 = l2["ParentNamespace"]
    l4 = l3["ParentNamespace"]
    l5 = l4["ParentNamespace"]
    # L5 leaf: repr-string of the L5 mock NS (uses MockJavaObject.__repr__).
    assert isinstance(l5, str)
    assert l5 == "<l5-global>"
    # L4 still a fully-introspected curated dict.
    assert isinstance(l4, dict)
    assert l4["_java_class"] == "ghidra.program.model.symbol.NamespaceDB"


def test_snapshot_layer5_primitive_is_also_repr_stringified() -> None:
    """At depth 5 a plain primitive (str/int/None) becomes ``str(repr(v))``.

    The terminal-layer rule fires BEFORE the primitive short-circuit:
    ``getName`` on L4 returns the primitive ``"l4"``; that value is
    evaluated at depth=5 (L4's children) and must surface as
    ``"'l4'"`` (the repr of the string), not ``"l4"`` itself.
    """
    func, _symbol, _l2, _l3, _l4 = build_ns_chain_to_layer5()
    snap = snapshot_function(func)
    l4 = snap["Symbol"]["ParentNamespace"]["ParentNamespace"]["ParentNamespace"]
    assert l4["Name"] == repr("l4")
    assert l4["ParentNamespace"] == "<l5-global>"


def test_snapshot_does_not_truncate_layer4() -> None:
    """L1..L4: full curated-getter introspection; only L5 collapses."""
    func, _symbol, _l2, _l3, _l4 = build_ns_chain_to_layer5()
    snap = snapshot_function(func)
    # Walk down to L4: Symbol(L1) -> PN(L2) -> PN(L3) -> PN(L4).
    l4 = snap["Symbol"]["ParentNamespace"]["ParentNamespace"]["ParentNamespace"]
    assert isinstance(l4, dict)
    assert l4["_java_class"] == "ghidra.program.model.symbol.NamespaceDB"


def test_snapshot_handles_getter_exception() -> None:
    """A getter that raises records the exception name + message, not crashes."""
    def _boom() -> str:
        raise RuntimeError("simulated JVM trip")

    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "foo", "getEntryPoint": _boom},
    )
    snap = snapshot_function(func)
    assert snap["Name"] == "foo"
    assert isinstance(snap["EntryPoint"], dict)
    assert "RuntimeError" in snap["EntryPoint"]["_error"]
    assert "simulated JVM trip" in snap["EntryPoint"]["_error"]


def test_snapshot_records_missing_getter() -> None:
    """A getter the curated list expects but the handle doesn't expose
    surfaces as ``{"_missing": <getter>}`` rather than blowing up.
    """
    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "foo"},  # no getSymbol, no getSignature, etc.
    )
    snap = snapshot_function(func)
    assert snap["Name"] == "foo"
    assert snap["Symbol"] == {"_missing": "getSymbol"}


def test_snapshot_treats_jni_array_class_as_collection() -> None:
    """A Java-array-like handle whose ``getClass().getName()`` is a
    JNI-style array binary name (``[Lghidra....;``) must be walked as
    a collection, NOT introspected through a curated-getter list that
    happens to substring-match its element type.
    """
    class _JavaArrayLike:
        def __init__(self, items, java_class_name):
            self._items = items
            self._jc = MockJavaClass(java_class_name)

        def getClass(self):
            return self._jc

        def __iter__(self):
            return iter(self._items)

    p0 = MockJavaObject(
        java_class="ghidra.program.model.listing.ParameterImpl",
        getters={"getName": "x", "getOrdinal": 0},
    )
    arr = _JavaArrayLike([p0], "[Lghidra.program.model.listing.Parameter;")
    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "f", "getParameters": arr},
    )
    snap = snapshot_function(func)
    # ``Parameters`` is a list of dicts, NOT a dict-of-missing-Parameter-getters.
    assert isinstance(snap["Parameters"], list)
    assert snap["Parameters"][0]["Name"] == "x"


def test_snapshot_invokes_parameterised_getter() -> None:
    """A curated-list entry of form ``(name, args)`` calls the getter
    with those args - mirrors the Ghidra ``getPrototypeString(bool, bool)``
    and ``getThunkedFunction(bool)`` cases.
    """
    received: list[tuple[Any, ...]] = []

    def _proto(*args: Any) -> str:
        received.append(args)
        return "int main(int, char**)"

    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "main", "getPrototypeString": _proto},
    )
    snap = snapshot_function(func)
    assert snap["PrototypeString"] == "int main(int, char**)"
    assert received == [(True, True)]


def test_snapshot_collects_collection_returning_getter() -> None:
    """``getParameters`` -> list of Parameter objects; each surfaces as
    its own snapshot dict.
    """
    param0 = MockJavaObject(
        java_class="ghidra.program.model.listing.ParameterImpl",
        getters={"getName": "argc", "getOrdinal": 0},
    )
    param1 = MockJavaObject(
        java_class="ghidra.program.model.listing.ParameterImpl",
        getters={"getName": "argv", "getOrdinal": 1},
    )
    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "main", "getParameters": [param0, param1]},
    )
    snap = snapshot_function(func)
    assert isinstance(snap["Parameters"], list)
    assert len(snap["Parameters"]) == 2
    assert snap["Parameters"][0]["Name"] == "argc"
    assert snap["Parameters"][0]["Ordinal"] == 0
    assert snap["Parameters"][1]["Name"] == "argv"
