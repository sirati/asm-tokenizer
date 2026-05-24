"""Tree-model node dataclasses for the inspector.

Pure model layer (no Textual imports): every node is a frozen
dataclass with ``can_expand: bool`` + ``expand(session, *,
vocab_manager) -> list[Node]``. The UI layer (``_app.py``) drives
expand and wraps it in a single try/except dispatcher per plan D8 --
node methods raise normally; ``is_failed`` is stamped by the UI.

Lazy expansion contract (per plan D2): only :class:`FunctionNode` and
:class:`InlineCallNode` invoke :func:`batch_decode` (every call uses
``include_fid_sidecar=True`` + ``keep_intermediate=True``).
:class:`VariantNode` / :class:`BlockNode` / :class:`InlineJumpNode`
consume data already in hand. PLT / EXT call sites are non-expandable
leaves (only ``can_expand`` for local-matched calls).

The :class:`DecodeContext` bundles the per-FunctionNode
``fid_sidecar`` / ``fid_row_offsets`` / ``line_to_name`` / vocab
references and the session-bound :attr:`callee_arm_resolver` closure
so every descendant carries ONE shared reference instead of five.

The package is split BY CONCERN (one node family per submodule):

* :mod:`._context` -- shared :class:`DecodeContext` dataclass.
* :mod:`._nodes_leaf` -- terminal :class:`AsmLeaf` +
  :class:`ShowAllVariantsNode`.
* :mod:`._nodes_function` -- top-level :class:`FunctionNode` + the
  result-to-variants builder.
* :mod:`._nodes_variant` -- :class:`VariantNode` + the one-shot
  FTL-parse + section-invariant prebuild (:class:`VariantContext`).
* :mod:`._nodes_block` -- :class:`BlockNode` + :class:`InlineJumpNode`
  + the shared body-expansion helper.
* :mod:`._nodes_call` -- :class:`InlineCallNode`.

Mutual imports between submodules are deliberate (an
:class:`InlineCallNode` ``expand`` constructs a synthetic
:class:`FunctionNode` for the callee) -- the package boundary
collects related dataclasses; pulling them into one flat namespace
via this ``__init__`` lets external callers ignore the split.
"""

from __future__ import annotations

from ._context import DecodeContext
from ._nodes_block import BlockNode, InlineJumpNode
from ._nodes_call import InlineCallNode
from ._nodes_function import FunctionNode
from ._nodes_leaf import AsmLeaf, ShowAllVariantsNode
from ._nodes_variant import VariantContext, VariantNode


# Union of every concrete node type the model can produce; the UI
# layer pins this single name instead of sprinkling typing.Union.
Node = (
    AsmLeaf
    | BlockNode
    | FunctionNode
    | InlineCallNode
    | InlineJumpNode
    | ShowAllVariantsNode
    | VariantNode
)


__all__ = [
    "AsmLeaf",
    "BlockNode",
    "DecodeContext",
    "FunctionNode",
    "InlineCallNode",
    "InlineJumpNode",
    "Node",
    "ShowAllVariantsNode",
    "VariantContext",
    "VariantNode",
]
