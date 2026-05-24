"""Shared pure-Python mocks for the duplicate-function-dump test pair.

Module purpose: keep the hand-rolled pyghidra-Java-handle mock + the
5-deep namespace chain helper in ONE place so both
``test_function_metadata_snapshot`` and ``test_duplicate_function_dump``
can import them without duplicating scaffolding.

A real Ghidra Function would need a JVM; these mocks let the unit
tests stay pure-Python while reproducing the introspection surface
the snapshot driver expects.
"""

from __future__ import annotations

from typing import Any


class MockJavaClass:
    """Mimics the ``getClass()`` return value: a thing with ``getName()``."""

    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:
        return self._name


class MockJavaObject:
    """Minimal mock of a pyghidra Java handle.

    ``java_class`` controls ``getClass().getName()`` (so the snapshot
    helper picks the right curated getter list via substring match);
    ``getters`` is a dict mapping ``method_name -> callable | value``.
    A callable getter is invoked, a value getter is returned wrapped
    in a lambda. ``str(obj)`` / ``repr(obj)`` return the supplied
    ``repr_str`` so summary fallbacks have a predictable value.

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
        self._java_class = MockJavaClass(java_class)
        self._getters = getters or {}
        self._repr_str = repr_str

    def getClass(self) -> MockJavaClass:
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

    def __repr__(self) -> str:
        return self._repr_str


def build_ns_chain_to_layer5() -> tuple[
    MockJavaObject,
    MockJavaObject,
    MockJavaObject,
    MockJavaObject,
    MockJavaObject,
]:
    """Build a 5-deep Function -> Symbol -> NS -> NS -> NS -> NS chain.

    Returns ``(func, symbol, parent_ns_l2, parent_ns_l3, parent_ns_l4)``
    in shape-walk order; the fifth NS (``parent_ns_l5``) is captured
    internally as ``l4_ns.getParentNamespace()``. Depth numbering
    matches the ``_snapshot`` recursion: Symbol = L1 (depth=1),
    parent_ns_l2 = L2 (depth=2), ..., parent_ns_l5 = L5 (depth=5, the
    terminal repr-string layer).
    """
    l5_ns = MockJavaObject(
        java_class="ghidra.program.model.symbol.GlobalNamespace",
        getters={"getName": "global", "getParentNamespace": None},
        repr_str="<l5-global>",
    )
    l4_ns = MockJavaObject(
        java_class="ghidra.program.model.symbol.NamespaceDB",
        getters={"getName": "l4", "getParentNamespace": l5_ns},
        repr_str="<l4>",
    )
    l3_ns = MockJavaObject(
        java_class="ghidra.program.model.symbol.NamespaceDB",
        getters={"getName": "l3", "getParentNamespace": l4_ns},
        repr_str="<l3>",
    )
    l2_ns = MockJavaObject(
        java_class="ghidra.program.model.symbol.NamespaceDB",
        getters={"getName": "l2", "getParentNamespace": l3_ns},
        repr_str="<l2>",
    )
    symbol = MockJavaObject(
        java_class="ghidra.program.model.symbol.SymbolDB",
        getters={"getName": "foo", "getParentNamespace": l2_ns},
    )
    func = MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": "foo", "getSymbol": symbol},
    )
    return func, symbol, l2_ns, l3_ns, l4_ns


def make_func_mock(name: str, entry: int) -> MockJavaObject:
    """Convenience: a minimal Function-handle mock with ``getName``/``getEntryPoint``."""
    return MockJavaObject(
        java_class="ghidra.program.model.listing.FunctionDB",
        getters={"getName": name, "getEntryPoint": entry},
    )


