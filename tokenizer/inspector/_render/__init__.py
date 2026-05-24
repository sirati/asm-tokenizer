"""Shared rendering core for the inspector.

This package owns the per-block emission body, the typed
``RenderBackend`` Protocol both backends produce, and the dispatch
tables (band classification + per-Category emitters) consumed by the
``_render_block`` body.

Phase A2 ships the typed Protocol + dataclasses
(:mod:`tokenizer.inspector._render._protocol`); the legacy per-block
walker lives at :mod:`tokenizer.inspector._render._legacy` until
phase A3 surgical-edits its body into ``_render_block.py`` per plan
section 3 / decision 27. The re-exports below preserve the
pre-Wave-5 import surface (``from tokenizer.inspector._render
import AsmLine, render_block, ...``) so existing call sites + tests
keep working across the phase chain.

The legacy ``AsmLine`` / ``InlineCallEntry`` / ``InlineJumpEntry`` /
``LineItem`` types share the shape declared by ``_protocol``;
post-A3, ``_legacy``'s exports collapse into the Protocol module's
definitions and the legacy submodule is deleted.
"""

from tokenizer.inspector._render._legacy import (
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
