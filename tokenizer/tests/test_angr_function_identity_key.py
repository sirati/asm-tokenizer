"""Tests for the angr-side ``FunctionView.identity_key`` extraction.

Concern: ``_angr_identity_key`` in
:mod:`tokenizer.disasm.angr_provider.function_identity` reads
``is_plt``, ``is_simprocedure`` and ``name`` off an angr ``Function``
to produce a cross-binary-stable :class:`ThunkIdentity` for PLT stubs
and SimProcedures. The unit test exercises the helper against
hand-rolled mocks (no angr/CLE/cfg required), mirroring the structure
of ``test_ghidra_function_identity_key.py``.
"""

from __future__ import annotations

from typing import Any

from tokenizer.disasm.angr_provider.function_identity import _angr_identity_key
from tokenizer.function_deduper import ThunkIdentity, ThunkTargetKind


# ---------------------------------------------------------------------------
# Mock angr Function
# ---------------------------------------------------------------------------


class _MockAngrFunction:
    """Minimal angr ``Function`` mock for identity-key extraction.

    Only the three attributes the helper reads (``name``, ``is_plt``,
    ``is_simprocedure``) are populated; everything else angr/CLE
    surface on real Functions is irrelevant to the identity-key path.
    """

    def __init__(
        self,
        *,
        name: Any = "",
        is_plt: bool = False,
        is_simprocedure: bool = False,
    ) -> None:
        self.name = name
        self.is_plt = is_plt
        self.is_simprocedure = is_simprocedure


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_identity_key_none_for_none_function() -> None:
    """Defensive: an unbound view (no backing Function) collapses to
    None rather than raising."""
    assert _angr_identity_key(None) is None


def test_identity_key_none_for_non_thunk_function() -> None:
    """A regular angr function (neither PLT nor SimProcedure) has no
    stronger-than-name identity; the extractor returns None and the
    caller falls back to the legacy disambiguation path."""
    func = _MockAngrFunction(name="main", is_plt=False, is_simprocedure=False)
    assert _angr_identity_key(func) is None


def test_identity_key_external_for_plt_stub() -> None:
    """A PLT stub resolves to an imported symbol; angr surfaces the
    imported name on ``func.name``. The extractor wraps it as
    ``ThunkIdentity(EXTERNAL, name)`` — cross-binary stable."""
    func = _MockAngrFunction(name="calloc", is_plt=True)
    assert _angr_identity_key(func) == ThunkIdentity(
        kind=ThunkTargetKind.EXTERNAL, key="calloc"
    )


def test_identity_key_external_for_simprocedure() -> None:
    """A SimProcedure (CLE synthetic extern slot for an unresolved /
    stub import) is treated identically — the imported symbol name is
    the cross-binary-stable identity."""
    func = _MockAngrFunction(name="malloc", is_simprocedure=True)
    assert _angr_identity_key(func) == ThunkIdentity(
        kind=ThunkTargetKind.EXTERNAL, key="malloc"
    )


def test_identity_key_external_when_both_flags_true() -> None:
    """A SimProcedure that also happens to be flagged is_plt (the
    CLE-installed import resolver for a PLT slot) lands on the EXTERNAL
    path; the helper does not need to disambiguate which flag fired."""
    func = _MockAngrFunction(
        name="strlen", is_plt=True, is_simprocedure=True
    )
    assert _angr_identity_key(func) == ThunkIdentity(
        kind=ThunkTargetKind.EXTERNAL, key="strlen"
    )


def test_identity_key_none_when_imported_name_is_empty() -> None:
    """Defensive: a PLT stub with an empty-string name (a CLE edge
    case for stripped binaries) declines to assert identity — an
    empty suffix would produce ``<raw_name>@thunk:`` which is not a
    useful disambiguator."""
    func = _MockAngrFunction(name="", is_plt=True)
    assert _angr_identity_key(func) is None


def test_identity_key_cross_binary_plt_stub_is_stable() -> None:
    """The cross-binary-stability invariant on the angr path: two
    distinct binaries' PLT slots for the SAME imported symbol produce
    the SAME identity_key (the imported name is loader-resolved and
    independent of link order)."""
    binary_a = _MockAngrFunction(name="gzseek", is_plt=True)
    binary_b = _MockAngrFunction(name="gzseek", is_plt=True)
    assert _angr_identity_key(binary_a) == _angr_identity_key(binary_b)
