"""Shared scaffolding for the :func:`render_block` test package.

The renderer consumes a parsed :class:`BlockTokenList` + the parent
:class:`Section` + pre-built per-variant invariants + three
closures/maps, and emits an ordered list of typed :class:`LineItem` s.
These tests stub the BlockTokenList walk via minimal lookalike classes
matching the slice of the API ``render_block`` touches; :class:`Section`
+ :class:`VariantBlock` inputs are real frozen dataclasses.

Per-concern test files import scaffolding via the explicit relative
form (``from .conftest import _make_section``) because the surrounding
``tests/`` directory is an explicit Python package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionKind,
    SectionPointerSpec,
)
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
    Section,
)
from tokenizer.tokens import TokenType


# ---------------------------------------------------------------------------
# Block/instruction stubs (the renderer touches a thin slice of these APIs)
# ---------------------------------------------------------------------------


@dataclass
class _StubToken:
    """Minimal metatoken stand-in: ``.token_type`` + ``.id`` + per-token
    ``to_asm_like()`` text.

    The renderer reads ``token.token_type`` to discriminate (LOCAL_FUNC
    / PLT_FUNC / EXT_FUNC / BLOCK_V2 lift to inline call / jump
    openables; any other type renders its ``to_asm_like()`` atom
    verbatim). ``int(token.id)`` carries the per-Category counter (for
    calls) or jump-target block index (BLOCK_V2). ``asm_text`` is the
    per-token ``to_asm_like`` atom; defaults match the production
    :meth:`Tokens.to_asm_like` shape (``f"<basename>:<id>"`` for
    identity tokens), and pure-text instructions pass plain text via
    a non-discriminated :class:`TokenType` (e.g. ``PLATFORM``) with
    ``asm_text="nop"``.
    """

    token_type: TokenType
    id: int = 0
    asm_text: str = ""

    def to_asm_like(self) -> str:
        return self.asm_text


def _text_token(text: str) -> _StubToken:
    """Plain-text token: renders as ``text`` with no inline-openable
    discrimination (TokenType.PLATFORM is the non-call/non-jump
    discriminator chosen here; the renderer only inspects
    ``token.token_type`` for the call/jump kinds, so any other type
    works)."""
    return _StubToken(token_type=TokenType.PLATFORM, id=0, asm_text=text)


def _plain_insn(asm: str) -> "_StubInsn":
    """Build a pure-text :class:`_StubInsn` from a space-joined asm
    string: each space-delimited atom becomes one plain-text token, so
    the per-token renderer produces the same joined text back."""
    return _StubInsn(tokens=[_text_token(atom) for atom in asm.split(" ")])


@dataclass
class _StubInsn:
    """Stand-in for :class:`InsnTokenList`.

    Implements :meth:`iter_tokens` for the per-token walk in
    :func:`_render_insn`; the AsmLine text is rebuilt by the renderer
    from each token's :meth:`to_asm_like`, mirroring the production
    :meth:`InsnTokenList.to_asm_like` contract.
    """

    tokens: List[_StubToken] = field(default_factory=list)

    def iter_tokens(self) -> Iterable[_StubToken]:
        return iter(self.tokens)


@dataclass
class _StubBlock:
    """Stand-in for :class:`BlockTokenList`.

    The renderer only calls ``block.iter_insn(transient=True)``;
    ``transient`` is opaque from the renderer's side (the real impl
    uses it to reuse an InsnTokenList shell -- correctness here only
    requires we yield each insn).
    """

    insns: List[_StubInsn] = field(default_factory=list)

    def iter_insn(self, transient: bool = False) -> Iterable[_StubInsn]:
        return iter(self.insns)


# ---------------------------------------------------------------------------
# Section / VariantBlock builders
# ---------------------------------------------------------------------------


def _make_call_target(
    *,
    type_: CallTargetType,
    function_name_ptr: int,
    function_section_ptr: int,
    is_matched: bool = True,
) -> CallTarget:
    return CallTarget(
        function_name_ptr=function_name_ptr,
        function_section_ptr=function_section_ptr,
        type=type_,
        is_matched=is_matched,
    )


def _make_section(call_targets: List[CallTarget]) -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=call_targets,
        variants=[],
    )


def _kind_to_idx(call_targets: List[CallTarget]) -> dict:
    out: dict[CallTargetType, list[int]] = {k: [] for k in CallTargetType}
    for i, ct in enumerate(call_targets):
        out[ct.type].append(i)
    return out


def _make_resolver(spec_for_offset: dict[int, SectionPointerSpec]):
    def resolver(offset: int) -> SectionPointerSpec | None:
        return spec_for_offset.get(int(offset))

    return resolver


# ---------------------------------------------------------------------------
# Common scaffolding for "no calls / jumps" cases
# ---------------------------------------------------------------------------


_EMPTY_SECTION = _make_section([])
_EMPTY_KIND_MAP = _kind_to_idx([])
_NO_PINS: dict[int, int] = {}
_NO_NAMES: dict[int, str] = {}
_NO_PROVIDERS: dict[int, str] = {}


def _resolver_never_called(_offset: int) -> SectionPointerSpec | None:
    raise AssertionError("callee_arm_resolver should not have been invoked")


def _flatten_openables(items):
    """Collect openables from every :class:`AsmLine` in stream order.

    Post-R2g the FTL renderer emits one :class:`AsmLine` per
    instruction with inline call/jump entries on
    :attr:`AsmLine.openables`; this helper concatenates them across
    every line so tests that care about the per-block list of inline
    entries (regardless of which instruction they hang off) can keep
    their old assertion shape.
    """
    out = []
    for it in items:
        out.extend(it.openables)
    return out


# ---------------------------------------------------------------------------
# Higher-level builders for the recurring "one LOCAL + one PLT + one EXTERN
# call_target" pattern used by the kind-dispatch tests. Keeping the shape
# centralised here avoids re-stamping the same three-CT block stub in every
# kind-dispatch test (see CLAUDE.md "never duplicate logic" -- the only
# inter-test variation is which fields each test asserts on, not the input
# shape).
# ---------------------------------------------------------------------------


def _ct(kind: CallTargetType, name_ptr: int, section_ptr: int) -> CallTarget:
    """Positional alias for :func:`_make_call_target` -- the kw-only
    surface is correct for production callers but verbose in tests
    where ``(kind, name_ptr, section_ptr)`` is the recurring shape."""
    return _make_call_target(
        type_=kind, function_name_ptr=name_ptr, function_section_ptr=section_ptr
    )


def _insn_with_call(asm: str, token_type: TokenType, *, id: int = 0) -> "_StubInsn":
    """One asm line carrying a single call-or-jump metatoken.

    ``asm`` is retained as a positional hint for test readability but
    is no longer used by the renderer; the line text is derived from
    the metatoken's resolved label. Tests that pin call/jump text
    assert on the resolved label, not on ``asm``.
    """
    del asm  # noqa: F841 — kept positionally for test readability
    return _StubInsn(tokens=[_StubToken(token_type, id=id)])


def _three_kind_call_targets(
    *, local_section_ptr: int, plt_section_ptr: int, extern_section_ptr: int
) -> List[CallTarget]:
    """Canonical LOCAL/PLT/EXTERN call-target triple. function_name_ptr
    is fixed at 1/2/3 so callers can index ``line_to_name`` directly."""
    return [
        _ct(CallTargetType.LOCAL, 1, local_section_ptr),
        _ct(CallTargetType.PLT, 2, plt_section_ptr),
        _ct(CallTargetType.EXTERN, 3, extern_section_ptr),
    ]


def _three_kind_block() -> "_StubBlock":
    """Block stub matching :func:`_three_kind_call_targets`: one insn per
    kind with counter_id=0 (each kind's first encounter)."""
    return _StubBlock(
        insns=[
            _insn_with_call("call local", TokenType.LOCAL_FUNC),
            _insn_with_call("call plt", TokenType.PLT_FUNC),
            _insn_with_call("call ext", TokenType.EXT_FUNC),
        ]
    )
