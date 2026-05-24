"""Shared rendering core for the inspector.

This package owns the per-block emission body, the typed
``RenderBackend`` Protocol both backends produce, and the dispatch
tables (band classification + per-Category emitters) consumed by the
``_render_block`` body.

Phase A2 ships the typed Protocol + dataclasses
(:mod:`tokenizer.inspector._render._protocol`); phase A2.5 ships the
band + per-Category dispatch registries
(:mod:`tokenizer.inspector._render._band` +
:mod:`tokenizer.inspector._render._category_dispatch`) the two
rendering backends register into at module load; phase A2.5 also
re-homes the FTL per-block walker at
:mod:`tokenizer.inspector._render._render_block` (the canonical body
both backends evolve through). The re-exports below preserve the
pre-Wave-5 import surface (``from tokenizer.inspector._render
import AsmLine, render_block, ...``) so existing call sites + tests
keep working across the phase chain.

``AsmLine`` / ``InlineCallEntry`` / ``InlineJumpEntry`` / ``LineItem``
canonical definitions live in :mod:`._protocol`; ``_render_block``
re-exports them so a single in-process object identity flows through
the package -- callers' ``isinstance`` checks agree across both
import paths.
"""

from tokenizer.inspector._render._render_block import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
    render_block,
)


__all__ = [
    "AsmLine",
    "InlineCallEntry",
    "InlineJumpEntry",
    "LineItem",
    "render_block",
]
