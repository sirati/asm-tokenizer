"""Tests for the angr-path ``FunctionView`` sentinel-None contracts.

The angr/Capstone provider lacks both the thunk-resolution surface
(``identity_key``) and the demangler-driven plate-comment surface
(``comment``). Both properties return ``None`` unconditionally so
every angr-path function lands on the legacy disambiguation path in
the deduper. The tests pin these constants so a future provider
extension doesn't silently drift.
"""

from __future__ import annotations

from tokenizer.disasm.angr_provider.views import _AngrFunctionView
from tokenizer.disasm.types import Architecture


def test_angr_function_view_identity_key_is_none() -> None:
    view = _AngrFunctionView(Architecture.X86)
    assert view.identity_key is None


def test_angr_function_view_comment_is_none() -> None:
    view = _AngrFunctionView(Architecture.X86)
    assert view.comment is None
