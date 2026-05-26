"""Tree-model node dataclasses for the inspector.

Pure model layer (no Textual imports): every node is a dataclass with
``can_expand: bool`` + ``expand() -> list[Node]``. The UI layer
(``_app.py``) drives expand and wraps it in a single try/except
dispatcher -- node methods raise normally; ``is_failed`` is stamped by
the UI.

Each :class:`FunctionNode` carries a :class:`BackendFactory` ref + a
typed :class:`FunctionHandle`; on first expand it calls
:meth:`BackendFactory.make` and threads the resulting
:class:`RenderBackend` to every descendant. The descendants
(:class:`VariantNode`, :class:`BlockNode`, :class:`InlineJumpNode`,
:class:`InlineCallNode`) hold the backend ref directly and call its
:meth:`variants` / :meth:`blocks` / :meth:`render_block` methods on
demand. :class:`InlineCallNode.expand` constructs a synthetic callee
:class:`FunctionNode` against the same factory.

The package is split by concern (one node family per submodule):

* :mod:`._nodes_leaf` -- terminal :class:`AsmLeaf` +
  :class:`NumberPrecisionLeaf` + :class:`InlineCallMissingVariantLeaf`
  + :class:`ShowAllVariantsNode`.
* :mod:`._nodes_function` -- top-level :class:`FunctionNode`.
* :mod:`._nodes_variant` -- :class:`VariantNode`.
* :mod:`._nodes_block` -- :class:`BlockNode` + :class:`InlineJumpNode`.
* :mod:`._nodes_call` -- :class:`InlineCallNode`.

Mutual imports between submodules are deliberate (an
:class:`InlineCallNode` ``expand`` constructs a synthetic
:class:`FunctionNode` for the callee); the package boundary collects
related dataclasses and the flat re-export below lets external
callers ignore the split.
"""

from __future__ import annotations

from ._nodes_block import BlockNode, InlineJumpNode
from ._nodes_call import InlineCallNode
from ._nodes_function import FunctionNode
from ._nodes_leaf import (
    AsmLeaf,
    InlineCallMissingVariantLeaf,
    NumberPrecisionLeaf,
    ShowAllVariantsNode,
)
from ._nodes_variant import VariantNode


# Union of every concrete node type the model can produce; the UI
# layer pins this single name instead of sprinkling typing.Union.
Node = (
    AsmLeaf
    | BlockNode
    | FunctionNode
    | InlineCallMissingVariantLeaf
    | InlineCallNode
    | InlineJumpNode
    | NumberPrecisionLeaf
    | ShowAllVariantsNode
    | VariantNode
)


__all__ = [
    "AsmLeaf",
    "BlockNode",
    "FunctionNode",
    "InlineCallMissingVariantLeaf",
    "InlineCallNode",
    "InlineJumpNode",
    "Node",
    "NumberPrecisionLeaf",
    "ShowAllVariantsNode",
    "VariantNode",
]
