"""Typed :class:`NodePath` capture + restore for the inspector tree.

Package boundary collecting three concerns each below the
~400-LOC cap:

* :mod:`._keys` -- per-kind typed :class:`NodeKey` dataclasses +
  the :data:`NodeKey` / :data:`NodePath` aliases.
* :mod:`._walk` -- :func:`node_key_for`, :func:`node_path_for`,
  :func:`capture_expand_state`, :func:`logical_path_of`, and the
  :class:`CapturedExpandState` payload. Owns the tree-walk
  capture side of the contract.
* :mod:`._match` -- :func:`should_expand_for_capture` + the
  :class:`VariantGroupNode`-descendancy gating. Owns the
  post-mount matcher.

External callers (the :mod:`._order_hooks` consumer + tests)
import the flat re-export below; the submodules stay private to
the package.
"""

from __future__ import annotations

from ._keys import (
    AsmLeafTerminalKey,
    AsmLeafWrapperCallKey,
    AsmLeafWrapperJumpKey,
    AsmLeafWrapperNumberKey,
    BlockKey,
    FunctionKey,
    InlineCallKey,
    InlineJumpKey,
    NodeKey,
    NodePath,
    NumberPrecisionKey,
    ShowAllVariantsKey,
    VariantGroupKey,
    VariantKey,
)
from ._match import should_expand_for_capture
from ._walk import (
    CapturedExpandState,
    capture_expand_state,
    logical_path_of,
    node_key_for,
    node_path_for,
)


__all__ = [
    "AsmLeafTerminalKey",
    "AsmLeafWrapperCallKey",
    "AsmLeafWrapperJumpKey",
    "AsmLeafWrapperNumberKey",
    "BlockKey",
    "CapturedExpandState",
    "FunctionKey",
    "InlineCallKey",
    "InlineJumpKey",
    "NodeKey",
    "NodePath",
    "NumberPrecisionKey",
    "ShowAllVariantsKey",
    "VariantGroupKey",
    "VariantKey",
    "capture_expand_state",
    "logical_path_of",
    "node_key_for",
    "node_path_for",
    "should_expand_for_capture",
]
