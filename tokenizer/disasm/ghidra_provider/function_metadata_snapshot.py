"""Ghidra ``Function`` Java-handle introspection (5-layer deep snapshot).

Concern: produce a pickle-serialisable ``dict[str, Any]`` capturing
the metadata a Ghidra ``Function`` exposes, down to a fixed depth of
5 nested layers, so a human can decide which fields are both
within-binary-disambiguating AND cross-ISA-stable for the same
logical C++ function.

Depth semantics (read carefully):

    L1 - direct getter results on the ``Function`` handle itself.
    L2 - direct getter results on each L1 sub-object (Symbol,
         Namespace, FunctionSignature, ParameterImpl, ...).
    L3 - direct getter results on each L2 sub-object.
    L4 - direct getter results on each L3 sub-object.
    L5 - terminal layer: EVERY value is stored as ``str(repr(value))``
         regardless of its Python/Java type. This is the safety wall;
         nothing past L5 is recursed into.

The module is BACKEND-SPECIFIC (pyghidra Java handles). It contains
NO collision-detection / NO file I/O / NO CLI knowledge - it is a
pure transform from one ``Function`` handle to one Python dict. The
orchestrator (``duplicate_function_dump``) owns those concerns.

Curated-getter lists per Java type:
    The Java type hierarchy under ``ghidra.program.model.*`` exposes
    hundreds of getters via ``Object`` (``getClass``, ``hashCode``,
    ``wait``, ...) that carry no metadata. Naive enumeration would
    bury the disambiguator candidates in noise AND incur per-getter
    JVM round-trip cost on every snapshot. The per-type lists below
    encode the *disambiguator-relevant* getters - additive when a
    new candidate shows up in offline inspection.

    Entries are either a plain ``str`` (no-arg getter) or a tuple
    ``(method_name, args_tuple)`` for getters Ghidra requires
    arguments on (e.g. ``getPrototypeString(boolean, boolean)``,
    ``getThunkedFunction(boolean)``). The tuple form keeps the
    args-encoding declarative; the snapshot driver does not infer
    them from method signatures.

Threading model: the helper accepts a Ghidra Function Java handle
already advanced to the target function; the caller (provider's
``iter_functions`` orchestration) is responsible for keeping the
handle live for the duration of the call.
"""

from __future__ import annotations

from typing import Any, Union

# Curated-getter entry: either ``"getName"`` (no-arg) or
# ``("getPrototypeString", (True, True))`` (Ghidra-required args).
GetterSpec = Union[str, tuple[str, tuple[Any, ...]]]

# ---------------------------------------------------------------------------
# Curated getter lists (per Java type)
#
# Each entry is the getter name (without parens). The snapshot driver
# invokes each, catches any exception (the Java method may not exist
# in older Ghidra versions, or may raise on a partially-populated
# program), and stores the result under the getter name with the
# leading ``get`` stripped (camelCase preserved).
# ---------------------------------------------------------------------------

# ``ghidra.program.model.listing.Function``
# ``getPrototypeString(boolean formalSignature, boolean includeCallingConvention)``
# and ``getThunkedFunction(boolean recursive)`` are Ghidra getters that
# REQUIRE arguments - the tuple form passes them declaratively.
_FUNCTION_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    "getEntryPoint",
    "getBody",
    "getSignature",
    ("getPrototypeString", (True, True)),
    "getCallingConventionName",
    "getCallingConvention",
    "getReturnType",
    "getParameters",
    "getParameterCount",
    "getLocalVariables",
    "getStackFrame",
    "getSymbol",
    "getParentNamespace",
    "getProgram",
    ("getThunkedFunction", (True,)),
    "isThunk",
    "isInline",
    "hasNoReturn",
    "hasCustomVariableStorage",
    "hasVarArgs",
    "isExternal",
    "isDeleted",
    "getRepeatableComment",
    "getComment",
    "getTags",
    "getID",
    "getSignatureSource",
    "getExternalLocation",
)

# ``ghidra.program.model.symbol.Symbol``
_SYMBOL_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    "getSource",
    "getSymbolType",
    "getParentNamespace",
    "getParentSymbol",
    "getAddress",
    "isPrimary",
    "isGlobal",
    "isExternal",
    "isPinned",
    "getID",
)

# ``ghidra.program.model.symbol.Namespace``
_NAMESPACE_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    "getParentNamespace",
    "getSymbol",
    "getID",
    "isGlobal",
    "isExternal",
)

# ``ghidra.program.model.listing.FunctionSignature``
# ``getPrototypeString(boolean formalSignature, boolean includeCallingConvention)``
# is the same parameterised getter as on Function.
_SIGNATURE_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    ("getPrototypeString", (True, True)),
    "getCallingConventionName",
    "getReturnType",
    "getArguments",
    "hasVarArgs",
    "getGenericCallingConvention",
)

# ``ghidra.program.model.listing.Parameter`` / ``Variable``
_PARAMETER_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    "getDataType",
    "getOrdinal",
    "getLength",
    "isAutoParameter",
    "isForcedIndirect",
    "getSource",
    "getRegister",
    "getStackOffset",
)

# ``ghidra.program.model.data.DataType``
_DATATYPE_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    "getDisplayName",
    "getLength",
    "getPathName",
    "getCategoryPath",
)

# ``ghidra.program.model.address.Address``
_ADDRESS_GETTERS: tuple[GetterSpec, ...] = (
    "getOffset",
    "getAddressSpace",
    "toString",
)

# ``ghidra.program.model.listing.Program``
_PROGRAM_GETTERS: tuple[GetterSpec, ...] = (
    "getName",
    "getExecutablePath",
    "getExecutableFormat",
    "getLanguageID",
    "getCompilerSpec",
)

# ``ghidra.program.model.symbol.ExternalLocation`` (the thunk's
# resolved external; carries the original symbol name + library +
# label - this is THE strong cross-ISA-stable disambiguator for
# PLT-thunk collisions, since every ISA variant binds the same
# thunk to the same external symbol).
_EXTERNAL_LOCATION_GETTERS: tuple[GetterSpec, ...] = (
    "getLibraryName",
    "getOriginalImportedName",
    "getLabel",
    "getAddress",
    "getSource",
    "getExternalSpaceAddress",
    "isFunction",
    "getDataType",
    "getSymbol",
)

# Type -> getter list. Selection uses ``getClass().getName()``
# membership against a simple-name fragment (e.g. "Symbol", "Function").
# Specificity wins: order matters, more specific names checked first.
# ``Function`` lives below ``FunctionSignature`` so the more-specific
# match wins on the signature handle.
_TYPE_GETTER_TABLE: tuple[tuple[str, tuple[GetterSpec, ...]], ...] = (
    ("FunctionSignature", _SIGNATURE_GETTERS),
    ("ExternalLocation", _EXTERNAL_LOCATION_GETTERS),
    ("Parameter", _PARAMETER_GETTERS),
    ("Variable", _PARAMETER_GETTERS),
    ("Namespace", _NAMESPACE_GETTERS),
    ("Symbol", _SYMBOL_GETTERS),
    ("DataType", _DATATYPE_GETTERS),
    ("Address", _ADDRESS_GETTERS),
    ("Program", _PROGRAM_GETTERS),
    ("Function", _FUNCTION_GETTERS),
)

# Max recursion depth (root Function counts as depth 0 - its direct
# getter results are L1). At ``depth == _MAX_DEPTH`` the snapshot
# driver emits ``str(repr(value))`` as the terminal leaf regardless
# of the value's type - this is the universal safety wall that keeps
# the dump bounded and pickle-safe.
_MAX_DEPTH: int = 5


# Sentinel signalling ``_to_python_native_scalar`` saw a non-scalar
# value (Java handle, collection, dict, opaque object). Using a unique
# module-level object lets the caller branch on identity rather than
# on ``None`` (which is itself a legitimate scalar payload).
_NOT_A_SCALAR: Any = object()


# ---------------------------------------------------------------------------
# Snapshot driver
# ---------------------------------------------------------------------------


def snapshot_function(ghidra_func: Any) -> dict[str, Any]:
    """Return a 5-layer-deep dict snapshot of the Ghidra Function handle.

    The returned dict's top-level keys are camelCase getter names with
    a leading ``get`` stripped (e.g. ``getName`` -> ``Name``,
    ``isInline`` -> ``isInline``); the leading-``get`` strip matches
    how Ghidra's JavaDoc tends to refer to properties.

    Values at depths 1..4 may be dicts (sub-objects within depth
    budget), primitive scalars, lists of nested values, or summary
    dicts of the form ``{"_java_class": "<class>", "_repr": "<repr>"}``
    for Java types not covered by the curated-getter table. At depth
    5 (the terminal layer), every value is stored as
    ``str(repr(value))`` regardless of type - this is the universal
    safety wall that keeps the dump bounded and pickle-safe.
    """
    return _snapshot(ghidra_func, depth=0)


def _snapshot(obj: Any, depth: int) -> Any:
    """Recurse into ``obj`` up to ``_MAX_DEPTH``; at depth, repr-string."""
    # Terminal layer: every L5 value becomes ``str(repr(value))``
    # regardless of type. Single-point safety wall, evaluated BEFORE
    # any other shape rule so primitives, dicts, Java handles, and
    # collections all collapse uniformly here.
    if depth == _MAX_DEPTH:
        return str(repr(obj))
    if obj is None:
        return None
    # Primitive scalars (Python-native OR JPype proxy that subclasses a
    # Python primitive) must be cast to a strict-native Python value
    # before being stored - otherwise the pickle carries a
    # ``jpype.types.JLong`` / ``JInt`` / ``JFloat`` / ... reference and
    # loading the dump requires a live JVM.
    native = _to_python_native_scalar(obj)
    if native is not _NOT_A_SCALAR:
        return native
    # Collections (Java arrays / Iterables surface as Python tuples/lists
    # in pyghidra; explicit ``list(obj)`` works for any iterable).
    java_class = _java_class_name(obj)
    if _is_collection(obj, java_class):
        try:
            elements = list(obj)
        except Exception as exc:
            return _summarise(obj, java_class, exc)
        return [_snapshot(elt, depth + 1) for elt in elements]
    getters = _getters_for(java_class)
    if not getters:
        return _summarise(obj, java_class, None)
    out: dict[str, Any] = {"_java_class": java_class}
    for spec in getters:
        name, args = _unpack_spec(spec)
        out[_strip_get(name)] = _invoke(obj, name, args, depth + 1)
    return out


def _invoke(
    obj: Any, getter: str, args: tuple[Any, ...], child_depth: int
) -> Any:
    """Call ``obj.<getter>(*args)``; recurse on the result at ``child_depth``."""
    method = getattr(obj, getter, None)
    if method is None:
        return {"_missing": getter}
    try:
        value = method(*args)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}
    return _snapshot(value, child_depth)


def _unpack_spec(spec: GetterSpec) -> tuple[str, tuple[Any, ...]]:
    """Return ``(name, args)`` for a string-or-tuple curated entry."""
    if isinstance(spec, tuple):
        return spec[0], spec[1]
    return spec, ()


def _getters_for(java_class: str) -> tuple[GetterSpec, ...]:
    """Return the curated getter list whose key appears in ``java_class``.

    Matches by simple substring (``"ghidra.program.model.symbol.Symbol"``
    contains ``"Symbol"``). The table is order-sensitive (specificity
    first); the first match wins.
    """
    for key, getters in _TYPE_GETTER_TABLE:
        if key in java_class:
            return getters
    return ()


def _java_class_name(obj: Any) -> str:
    """Best-effort Java class name for ``obj``; falls back to Python type."""
    get_class = getattr(obj, "getClass", None)
    if get_class is None:
        return f"<python:{type(obj).__name__}>"
    try:
        return str(get_class().getName())
    except Exception:
        return f"<python:{type(obj).__name__}>"


def _is_collection(obj: Any, java_class: str) -> bool:
    """Heuristic: treat Java ``Iterable`` / Python list/tuple as a collection.

    We avoid recursing into ``str`` (Java strings surface as Python str
    via pyghidra, and the primitive check above already handled them).

    Java arrays surface either through JPype's JArray (caught via
    class-name token ``JArray``) or as JNI-style binary names
    ``[L<element-class>;`` / ``[I``, ``[B``, etc. - the leading ``[``
    is the discriminator the JLS uses for any array type.
    """
    if isinstance(obj, (list, tuple, set)):
        return True
    # Java arrays appear as JArray; iterating them yields elements.
    if java_class.startswith("[") or "[]" in java_class or "JArray" in java_class:
        return True
    # Container types whose only useful introspection is iteration.
    for marker in ("Iterator", "List", "Set", "Collection"):
        if marker in java_class:
            return True
    return False


def _summarise(obj: Any, java_class: str, exc: Exception | None) -> dict[str, Any]:
    """Fallback summary for Java types not covered by the curated-getter table.

    Emitted at depths 1..4 when the snapshot driver has no curated
    introspection list for ``java_class`` (or when iterating a
    suspected collection raised). At depth 5 the universal repr-string
    wall fires first and this fallback is never reached.
    """
    try:
        repr_str = str(obj)
    except Exception:
        repr_str = "<repr-failed>"
    summary: dict[str, Any] = {"_java_class": java_class, "_repr": repr_str}
    if exc is not None:
        summary["_iter_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _strip_get(name: str) -> str:
    """``getName`` -> ``Name``; ``isInline`` -> ``isInline``."""
    if name.startswith("get") and len(name) > 3 and name[3].isupper():
        return name[3:]
    return name


# Scalar-cast dispatch table: ``(isinstance-predicate, constructor)``
# pairs evaluated in order. ``bool`` is intentionally first - JBoolean
# subclasses ``int`` in JPype, and so does Python's ``bool``; without
# the bool-before-int order a Java boolean would be cast to ``int``
# and lose its type identity. Constructors always return a strict
# Python-native instance (``type(int(JLong(42))) is int``), which is
# the property that lets the pickle round-trip without a JVM.
_NATIVE_SCALAR_CASTS: tuple[tuple[type, Any], ...] = (
    (bool, bool),
    (int, int),
    (float, float),
    (bytes, bytes),
    (str, str),
)


def _to_python_native_scalar(value: Any) -> Any:
    """Return a strict-native Python copy of ``value`` if it is a scalar.

    Returns the module-level ``_NOT_A_SCALAR`` sentinel when ``value``
    is not one of bool / int / float / bytes / str (including any
    JPype proxy that subclasses one of those types - JLong, JInt,
    JFloat, JDouble, JBoolean, JChar, JByte, JShort all subclass a
    Python primitive). The caller then routes the value through the
    collection / curated-getter / summary path instead.

    Why ``isinstance`` + constructor (not ``type() is``): the latter
    would let Python-native subclasses (``IntEnum``, ``namedtuple``
    indices, etc.) escape; the former + constructor-cast guarantees
    the output's exact type matches one of the five primitive types,
    which is precisely the property pickle relies on for JVM-free
    loading.
    """
    for predicate_type, constructor in _NATIVE_SCALAR_CASTS:
        if isinstance(value, predicate_type):
            return constructor(value)
    return _NOT_A_SCALAR
