"""Lazy module-level access to the Ghidra/JVM classes the decode path uses.

Owns the single "resolve a JVM class for Python-side use" concern. The
per-call ``from ghidra... import X`` statements the decode helpers used
to carry paid Python import-machinery overhead on every invocation —
per operand on the hot path. Hoisting them to plain module-level imports
is NOT possible: these modules are imported (e.g. by unit tests and by
``import tokenizer.run_tokenizer``) before the Ghidra JVM is started,
and ``import ghidra...`` requires a running JVM.

This module is import-safe with no JVM: resolution happens on FIRST
attribute access (the JVM is guaranteed started before any decode call)
via PEP 562 module ``__getattr__``, which caches the resolved class into
the module globals — every later ``jvm_types.X`` access is a plain
module-dict hit.

Usage at a call site::

    from tokenizer.disasm.ghidra_provider import jvm_types

    def hot_function(...):
        Register = jvm_types.Register   # one dict lookup per call
        ...
"""

from __future__ import annotations

from importlib import import_module

# Exposed name -> (Java module path, attribute). Every JVM class the
# ghidra_provider decode modules consult lives here; adding a new class
# is a one-line table entry.
_RESOLVERS: dict[str, tuple[str, str]] = {
    "Address": ("ghidra.program.model.address", "Address"),
    "OperandType": ("ghidra.program.model.lang", "OperandType"),
    "Register": ("ghidra.program.model.lang", "Register"),
    "Scalar": ("ghidra.program.model.scalar", "Scalar"),
    "PcodeOp": ("ghidra.program.model.pcode", "PcodeOp"),
    "JavaCharacter": ("java.lang", "Character"),
}


def __getattr__(name: str):
    try:
        module_path, attr = _RESOLVERS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    resolved = getattr(import_module(module_path), attr)
    globals()[name] = resolved
    return resolved
