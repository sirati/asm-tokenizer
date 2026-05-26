"""Tests for the angr-path ``FunctionView`` identity-key + comment
contracts.

The angr/Capstone provider lacks the demangler-driven plate-comment
surface (``comment`` is unconditionally ``None``) but DOES surface
PLT/SimProcedure thunk identity (``identity_key`` is a
:class:`ThunkIdentity` keyed on the imported symbol name when angr
flags ``is_plt`` or ``is_simprocedure``, ``None`` otherwise). These
sentinels keep angr in cross-provider canonical-name parity with the
Ghidra path's ``isExternal()`` thunks.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.angr_provider.views import _AngrFunctionView
from tokenizer.disasm.types import Architecture
from tokenizer.function_deduper import ThunkIdentity, ThunkTargetKind


# ---------------------------------------------------------------------------
# angr Function mock — minimal enough for the identity-key probe
# ---------------------------------------------------------------------------


class _MockAngrFunction:
    """Minimal angr ``Function`` stand-in for identity-key tests."""

    def __init__(
        self,
        *,
        name: str = "",
        is_plt: bool = False,
        is_simprocedure: bool = False,
    ) -> None:
        self.name = name
        self.is_plt = is_plt
        self.is_simprocedure = is_simprocedure


def _make_view_with_func(func: Any) -> _AngrFunctionView:
    view = _AngrFunctionView(Architecture.X86)
    view._set(func)
    return view


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


def test_angr_function_view_comment_is_none() -> None:
    """``comment`` is unconditionally ``None`` on the angr path (no
    demangler hook — see ``angr_limitations.md``)."""
    view = _AngrFunctionView(Architecture.X86)
    assert view.comment is None


def test_angr_function_view_identity_key_is_none_for_unbound_view() -> None:
    """An unbound view (no ``_set(...)`` call yet) has no backing
    Function; ``identity_key`` collapses to ``None``."""
    view = _AngrFunctionView(Architecture.X86)
    assert view.identity_key is None


def test_angr_function_view_identity_key_is_none_for_non_thunk() -> None:
    """A non-PLT, non-SimProcedure angr function returns ``None`` so
    every regular angr-path function lands on the legacy
    disambiguation path."""
    func = _MockAngrFunction(name="main", is_plt=False, is_simprocedure=False)
    view = _make_view_with_func(func)
    assert view.identity_key is None


def test_angr_function_view_identity_key_for_plt_stub() -> None:
    """A PLT stub returns ``ThunkIdentity(EXTERNAL, name)`` — the
    imported symbol name is the cross-binary-stable key, matching the
    Ghidra provider's ``isExternal()`` branch."""
    func = _MockAngrFunction(name="calloc", is_plt=True)
    view = _make_view_with_func(func)
    assert view.identity_key == ThunkIdentity(
        kind=ThunkTargetKind.EXTERNAL, key="calloc"
    )


def test_angr_function_view_identity_key_for_simprocedure() -> None:
    """A SimProcedure (CLE synthetic extern slot for an unresolved /
    stub import) is treated like a PLT stub — the identity_key is the
    imported symbol name."""
    func = _MockAngrFunction(name="malloc", is_simprocedure=True)
    view = _make_view_with_func(func)
    assert view.identity_key == ThunkIdentity(
        kind=ThunkTargetKind.EXTERNAL, key="malloc"
    )


def test_angr_function_view_cross_binary_plt_stub_is_stable() -> None:
    """The cross-binary-stability invariant on the angr path: two
    binaries with the SAME imported symbol but DIFFERENT PLT-stub
    addresses produce the SAME identity_key (and therefore the SAME
    canonical name through ``canonical_function_name``)."""
    binary_a = _make_view_with_func(_MockAngrFunction(name="gzseek", is_plt=True))
    binary_b = _make_view_with_func(_MockAngrFunction(name="gzseek", is_plt=True))
    assert binary_a.identity_key == binary_b.identity_key
