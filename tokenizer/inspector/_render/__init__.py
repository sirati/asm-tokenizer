"""Shared rendering core for the inspector.

This package owns the per-block emission body and the typed
``RenderBackend`` Protocol both backends produce.

:mod:`._protocol` hosts the Protocol + the typed
``LineItem``/``RenderedBlock``/``RenderedVariant``/``FunctionHandle``
dataclasses; :mod:`._render_block` hosts the FTL per-block walker
body. The BatchDecode walker lives under :mod:`._batch_decode_backend`
(self-contained — band classification + per-Category dispatch are
private to that subpackage). The re-exports below preserve the
pre-Wave-5 import surface (``from tokenizer.inspector._render
import AsmLine, render_block, ...``) so existing call sites + tests
keep working.

``AsmLine`` / ``InlineCallEntry`` / ``InlineJumpEntry`` / ``LineItem``
canonical definitions live in :mod:`._protocol`; ``_render_block``
re-exports them so a single in-process object identity flows through
the package -- callers' ``isinstance`` checks agree across both
import paths.
"""

from tokenizer.inspector._render._protocol import (
    BackendFactory,
    FunctionHandle,
    RenderBackend,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._render._render_block import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
    render_block,
)


__all__ = [
    "AsmLine",
    "BackendFactory",
    "FunctionHandle",
    "InlineCallEntry",
    "InlineJumpEntry",
    "LineItem",
    "RenderBackend",
    "RenderedBlock",
    "RenderedVariant",
    "render_block",
]
