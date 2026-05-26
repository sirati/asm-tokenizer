"""Identity-key extraction for angr ``Function`` handles.

Single concern: produce an :class:`Optional[ThunkIdentity]` from an
angr ``Function`` so the angr provider can populate the
``FunctionView.identity_key`` axis with the same cross-binary-stable
semantics the Ghidra provider does (see
``tokenizer/disasm/ghidra_views/function.py``).

The angr signal: ``Function.is_plt`` and ``Function.is_simprocedure``
are exposed on every architecture angr supports (x86 / x86_64 / AArch64
/ ARM32 / MIPS / PPC ELFv2 — per ``angr_limitations.md``). PLT stubs
forwarded to imported symbols and SimProcedures backing imported
library functions both surface with ``func.name`` already set to the
imported symbol name; that name IS the cross-binary-stable identity
key.

Non-thunk angr functions return ``None`` (legacy disambiguation path
— the provider declines to assert identity beyond name).
"""

from __future__ import annotations

from typing import Any, Optional

from tokenizer.function_deduper import ThunkIdentity, ThunkTargetKind


def _angr_identity_key(angr_function: Any) -> Optional[ThunkIdentity]:
    """Return a :class:`ThunkIdentity` for PLT stubs / SimProcedures,
    else ``None``.

    The two angr flags (``is_plt`` and ``is_simprocedure``) both
    indicate a function that resolves to an external/imported symbol —
    angr exposes the resolved name on ``func.name`` (the loader (CLE)
    binds the PLT slot to the import-table symbol at load time). That
    name is cross-binary stable for the same source symbol, mirroring
    the Ghidra provider's ``Function.getThunkedFunction(True).getName()``
    for ``isExternal()``-true thunks.

    Resilient to partially-populated angr Knowledge Bases: any
    ``AttributeError`` / unexpected type from the angr side collapses
    to ``None`` (= "no merge"), matching the defensive pattern in
    :func:`tokenizer.disasm.ghidra_views.function._ghidra_identity_key`.
    """
    if angr_function is None:
        return None
    is_plt = bool(getattr(angr_function, "is_plt", False))
    is_simproc = bool(getattr(angr_function, "is_simprocedure", False))
    if not (is_plt or is_simproc):
        return None
    raw_name = getattr(angr_function, "name", None)
    if raw_name is None:
        return None
    try:
        name = str(raw_name)
    except Exception:
        return None
    if name == "":
        return None
    return ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key=name)
