"""Tests for the duplicate-function metadata dump helpers.

Two concerns under test, separated cleanly:

1. ``function_metadata_snapshot.snapshot_function`` - the Java-handle
   introspector. Tested against a hand-rolled mock with a controllable
   ``getClass().getName()`` simple-name and a deterministic getter set,
   so we can pin (a) depth-3 truncation, (b) graceful handling of
   getter exceptions, (c) the curated-getter-list-per-type fan-out.

2. ``duplicate_function_dump.write_duplicate_function_dump`` - the
   collision-detection + JSON-write orchestrator. Tested against the
   same mock fixtures (or even simpler fakes) to pin (a) the "no
   collisions -> empty groups list" path, (b) the dup-group filtering
   threshold (count >= 2), (c) JSON serialisability of the final
   payload.

A real Ghidra Function would need a JVM; the unit tests stay pure-Python.
"""

from __future__ import annotations

import json
from typing import Any

from tokenizer.disasm.ghidra_provider.duplicate_function_dump import (
    write_duplicate_function_dump,
)
from tokenizer.disasm.ghidra_provider.function_metadata_snapshot import (
    snapshot_function,
)


# ---------------------------------------------------------------------------
# Mock Java-handle scaffolding
# ---------------------------------------------------------------------------


class _MockJavaClass:
    """Mimics the ``getClass()`` return value: a thing with ``getName()``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:
        return self._name


class _MockJavaObject:
    """Minimal mock of a pyghidra Java handle.

    ``java_class`` controls ``getClass().getName()`` (so the snapshot
    helper picks the right curated getter list via substring match);
    ``getters`` is a dict mapping ``method_name -> callable | value``.
    A callable getter is invoked, a value getter is returned wrapped
    in a lambda. ``str(obj)`` returns the supplied ``repr_str`` so
    summary fallbacks have a predictable value.

    The mock implements ``__getattr__`` so any unconfigured getter
    surfaces as missing (matches the real Java side, where the
    snapshot driver's ``_invoke`` returns ``{"_missing": ...}``).
    """

    def __init__(
        self,
        java_class: str,
        getters: dict[str, Any] | None = None,
        repr_str: str = "<MockJavaObject>",
    ) -> None:
        self._java_class = _MockJavaClass(java_class)
        self._getters = getters or {}
        self._repr_str = repr_str

    def getClass(self) -> _MockJavaClass:
        return self._java_class

    def __getattr__(self, name: str) -> Any:
        # ``_java_class`` / ``_getters`` / ``_repr_str`` are intercepted
        # by the normal attribute machinery; only unconfigured getter
        # lookups reach here.
        if "_getters" not in self.__dict__:
            raise AttributeError(name)
        getters = self.__dict__["_getters"]
        if name in getters:
            value = getters[name]
            if callable(value):
                return value
            return lambda v=value: v
        raise AttributeError(name)

    def __str__(self) -> str:
        return self._repr_str


# ---------------------------------------------------------------------------
# snapshot_function: layer-3 cap + curated getter fan-out
# ---------------------------------------------------------------------------


def test_snapshot_records_function_layer1_getters() -> None:
    """L1: direct getter results on the Function root land at top-level keys."""
    func = _MockJavaObject(
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
    symbol = _MockJavaObject(
        java_class="ghidra.program.model.symbol.SymbolDB",
        getters={"getName": "foo", "getSource": "USER_DEFINED", "isPrimary": True},
    )
    func = _MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "foo", "getSymbol": symbol},
    )
    snap = snapshot_function(func)
    assert isinstance(snap["Symbol"], dict)
    assert snap["Symbol"]["Name"] == "foo"
    assert snap["Symbol"]["Source"] == "USER_DEFINED"
    assert snap["Symbol"]["isPrimary"] is True


def test_snapshot_truncates_below_layer3() -> None:
    """L4: Function -> Symbol -> ParentNamespace -> ParentNamespace
    must summarise rather than recurse.
    """
    grandparent_ns = _MockJavaObject(
        java_class="ghidra.program.model.symbol.GlobalNamespace",
        getters={"getName": "global", "getParentNamespace": None},
        repr_str="<global>",
    )
    parent_ns = _MockJavaObject(
        java_class="ghidra.program.model.symbol.NamespaceDB",
        getters={"getName": "std", "getParentNamespace": grandparent_ns},
        repr_str="<std>",
    )
    symbol = _MockJavaObject(
        java_class="ghidra.program.model.symbol.SymbolDB",
        getters={"getName": "foo", "getParentNamespace": parent_ns},
    )
    func = _MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "foo", "getSymbol": symbol},
    )
    snap = snapshot_function(func)
    # L1=Function, L2=Symbol, L3=Symbol.ParentNamespace; the Namespace's
    # own ParentNamespace getter result is at L4 -> must be summarised.
    ns_at_l3 = snap["Symbol"]["ParentNamespace"]
    assert isinstance(ns_at_l3, dict)
    assert ns_at_l3["_java_class"] == "ghidra.program.model.symbol.NamespaceDB"
    # Inside L3's dict, ParentNamespace is at L4 -> summary only:
    # {"_java_class": ..., "_repr": ...} with no recursive Name key.
    nested = ns_at_l3["ParentNamespace"]
    assert isinstance(nested, dict)
    assert "_repr" in nested
    assert "Name" not in nested


def test_snapshot_handles_getter_exception() -> None:
    """A getter that raises records the exception name + message, not crashes."""
    def _boom() -> str:
        raise RuntimeError("simulated JVM trip")

    func = _MockJavaObject(
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
    func = _MockJavaObject(
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
            self._jc = _MockJavaClass(java_class_name)

        def getClass(self):
            return self._jc

        def __iter__(self):
            return iter(self._items)

    p0 = _MockJavaObject(
        java_class="ghidra.program.model.listing.ParameterImpl",
        getters={"getName": "x", "getOrdinal": 0},
    )
    arr = _JavaArrayLike([p0], "[Lghidra.program.model.listing.Parameter;")
    func = _MockJavaObject(
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

    func = _MockJavaObject(
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
    param0 = _MockJavaObject(
        java_class="ghidra.program.model.listing.ParameterImpl",
        getters={"getName": "argc", "getOrdinal": 0},
    )
    param1 = _MockJavaObject(
        java_class="ghidra.program.model.listing.ParameterImpl",
        getters={"getName": "argv", "getOrdinal": 1},
    )
    func = _MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "main", "getParameters": [param0, param1]},
    )
    snap = snapshot_function(func)
    assert isinstance(snap["Parameters"], list)
    assert len(snap["Parameters"]) == 2
    assert snap["Parameters"][0]["Name"] == "argc"
    assert snap["Parameters"][0]["Ordinal"] == 0
    assert snap["Parameters"][1]["Name"] == "argv"


# ---------------------------------------------------------------------------
# write_duplicate_function_dump: collision detection + JSON write
# ---------------------------------------------------------------------------


def _make_func(name: str, entry: int) -> _MockJavaObject:
    return _MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": name, "getEntryPoint": entry},
    )


def test_dump_skip_when_no_collisions(tmp_path) -> None:
    """No name collisions -> JSON written, ``duplicate_groups`` empty,
    return value is 0.
    """
    funcs = [
        (0x401000, "foo", _make_func("foo", 0x401000)),
        (0x402000, "bar", _make_func("bar", 0x402000)),
        (0x403000, "baz", _make_func("baz", 0x403000)),
    ]
    dump_path = tmp_path / "dump.json"
    count = write_duplicate_function_dump(funcs, "binA", dump_path)
    assert count == 0
    payload = json.loads(dump_path.read_text())
    assert payload["binary"] == "binA"
    assert payload["duplicate_groups"] == []


def test_dump_collects_collision_groups(tmp_path) -> None:
    """Two functions sharing a name -> one group entry; singletons absent."""
    funcs = [
        (0x401000, "foo", _make_func("foo", 0x401000)),
        (0x402000, "foo", _make_func("foo", 0x402000)),
        (0x403000, "bar", _make_func("bar", 0x403000)),
        (0x404000, "qux", _make_func("qux", 0x404000)),
        (0x405000, "qux", _make_func("qux", 0x405000)),
        (0x406000, "qux", _make_func("qux", 0x406000)),
    ]
    dump_path = tmp_path / "dump.json"
    count = write_duplicate_function_dump(funcs, "binB", dump_path)
    assert count == 2  # foo and qux
    payload = json.loads(dump_path.read_text())
    names = [g["name"] for g in payload["duplicate_groups"]]
    assert names == ["foo", "qux"]
    foo_group = next(g for g in payload["duplicate_groups"] if g["name"] == "foo")
    assert foo_group["count"] == 2
    assert [f["entry"] for f in foo_group["functions"]] == [0x401000, 0x402000]
    qux_group = next(g for g in payload["duplicate_groups"] if g["name"] == "qux")
    assert qux_group["count"] == 3


def test_dump_payload_is_json_serialisable(tmp_path) -> None:
    """The written file is valid JSON and round-trips through json.loads."""
    funcs = [
        (0x401000, "foo", _make_func("foo", 0x401000)),
        (0x402000, "foo", _make_func("foo", 0x402000)),
    ]
    dump_path = tmp_path / "dump.json"
    write_duplicate_function_dump(funcs, "binC", dump_path)
    text = dump_path.read_text()
    # Strict re-parse: any non-JSON-safe leaf would raise here.
    payload = json.loads(text)
    assert payload["binary"] == "binC"
    # Snapshot for the colliding function must include the Function's
    # class + Name getter result.
    snap = payload["duplicate_groups"][0]["functions"][0]["snapshot"]
    assert snap["_java_class"] == "ghidra.program.model.listing.FunctionDB"
    assert snap["Name"] == "foo"


def test_dump_creates_parent_dirs(tmp_path) -> None:
    """``output_path`` under a non-existent parent dir is auto-created."""
    funcs = [
        (0x401000, "foo", _make_func("foo", 0x401000)),
        (0x402000, "foo", _make_func("foo", 0x402000)),
    ]
    nested = tmp_path / "deeply" / "nested" / "dump.json"
    assert not nested.parent.exists()
    write_duplicate_function_dump(funcs, "binD", nested)
    assert nested.exists()
