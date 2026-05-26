"""End-to-end: row-walker emit -> ``BlockNode.expand`` -> ``AsmLeaf`` only.

Closes the cross-seam coverage gap audit R-CRIT highlighted: the
BatchDecode row walker's per-instruction emit (one :class:`AsmLine`
per instruction with :class:`InlineCallEntry` / :class:`InlineJumpEntry`
on :attr:`AsmLine.openables`) must round-trip cleanly through
:meth:`BlockNode.expand` -- i.e. the AsmLine-only narrowed
``LineItem`` contract holds end-to-end and inline call/jump sidecars
flow as openables on the surfaced :class:`AsmLeaf`, NOT as sibling
top-level model rows.

Single synthetic row carries both:

* one ``LOCAL_FUNC`` IDENTITY (resolves to ``my_func`` via the FID
  sidecar + ``line_to_name`` map), and
* one in-block ``BLOCK_V2`` IDENTITY (non-header, since
  ``pending_header`` was already consumed by the leading silent-header
  BLOCK_V2) -- becomes an :class:`InlineJumpEntry`.

Both atoms live inside ONE 2-slot BODY instruction (via
``insn_runlength=[2]``) so the per-instruction collector folds both
openables onto the SAME emitted :class:`AsmLine`. The test then
constructs a minimal :class:`RenderBackend`-spec'd stub whose
``render_block`` returns that section's items, builds a
:class:`BlockNode`, calls :meth:`BlockNode.expand`, and asserts every
child is an :class:`AsmLeaf` -- the row walker's openables ride on
the surfaced leaf's ``openables`` tuple, not as sibling model rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BackendFactory,
    BlockKind,
    InlineCallEntry,
    InlineJumpEntry,
    RenderBackend,
)
from tokenizer.inspector._tree_model import AsmLeaf, BlockNode

from ._row_walk_fixtures import (
    BLOCK_V2,
    EMPTY_NUMBERS,
    LOCAL_FUNC,
    NULL_CALLEE_RESOLVER,
    make_fid_table,
    make_result,
    vocab_stub,
)


def _walk_one_body_block_with_call_and_jump():
    """Synthesize the row + walk it; return the BODY section's items.

    Tokens: ``[BLOCK_V2(c=0), LOCAL_FUNC(c=0), BLOCK_V2(c=0), 0]``.

    * Slot 0: leading ``BLOCK_V2`` -- absorbed as the FUNCTION_ID
      -> BODY transition silent-header (block_idx=0).
    * Slots 1+2: ``LOCAL_FUNC`` (FID 42 -> ``my_func``) +
      ``BLOCK_V2(c=0)`` (in-block jump back to block 0; the target
      matches the only BODY section's block_idx so the jump
      openable passes the row walker's post-walk resolvability
      gate -- see :mod:`.._jump_validity`). Bundled into ONE 2-slot
      BODY instruction via ``insn_runlength=[2]`` so both openables
      fold onto the SAME emitted :class:`AsmLine`.
    """
    sig_default, se_default = EMPTY_NUMBERS
    blocks = render_row_blocks(
        result=make_result(
            tokens_row=np.asarray(
                [BLOCK_V2, LOCAL_FUNC, BLOCK_V2, 0], dtype=np.uint16,
            ),
            identities=np.asarray([0, 0, 0], dtype=np.uint16),
            numbers_sig=sig_default,
            numbers_se=se_default,
            insn_runlength=np.asarray([2], dtype=np.uint32),
        ),
        row=0, n_axis=0,
        partial_cut_lengths=[3],
        call_targets_per_ct=[[]],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
            sidecar=np.asarray([42], dtype=np.uint32),
        ),
        line_to_name={42: "my_func"},
        line_to_provider={},
        callee_arm_resolver=NULL_CALLEE_RESOLVER,
    )
    assert len(blocks) == 1
    body = blocks[0]
    assert body.kind is BlockKind.BODY
    assert body.block_idx == 0
    return body.items


# ---------------------------------------------------------------------------
# Row-walker emit shape -- one AsmLine carrying both openables
# ---------------------------------------------------------------------------


def test_row_walker_emits_one_asmline_with_call_and_jump_openables():
    """Cluster #3 cross-backend uniformity: BatchDecode emits exactly
    ONE :class:`AsmLine` per instruction; inline call + jump sidecars
    ride on its :attr:`AsmLine.openables` tuple, NOT as sibling
    top-level :class:`InlineCallEntry` / :class:`InlineJumpEntry`
    items in the section's item stream.
    """
    items = _walk_one_body_block_with_call_and_jump()

    # No sibling InlineCallEntry / InlineJumpEntry at top level --
    # cluster #3 + W3-2 W4-AMENDED contract: ``LineItem == AsmLine``.
    assert all(isinstance(item, AsmLine) for item in items)
    assert len(items) == 1
    line = items[0]
    # Both openables fold onto the single instruction's openables tuple.
    assert len(line.openables) == 2
    call, jump = line.openables
    assert isinstance(call, InlineCallEntry)
    assert call.kind is CallTargetType.LOCAL
    assert call.callee_name == "my_func"
    assert isinstance(jump, InlineJumpEntry)
    assert jump.target_block_idx == 0


# ---------------------------------------------------------------------------
# BlockNode.expand -- AsmLeaf-only children, openables surfaced on each leaf
# ---------------------------------------------------------------------------


def test_block_node_expand_surfaces_asmleaf_only_carrying_openables():
    """End-to-end: row walker -> BatchDecode-shaped backend ->
    :meth:`BlockNode.expand` produces a list of :class:`AsmLeaf` only.

    The narrowed ``LineItem == AsmLine`` contract (R2f) means
    :func:`_translate_line_items` raises :class:`TypeError` on any
    non-AsmLine item; a green test here pins that the BatchDecode
    walker's emit shape stays narrowed. The surfaced :class:`AsmLeaf`'s
    :attr:`AsmLeaf.openables` carries the same two entries the row
    walker buffered, so :meth:`AsmLeaf.expand` (the 2+-arm wrap-each
    path) sees both at the UI layer.
    """
    items = _walk_one_body_block_with_call_and_jump()

    factory = MagicMock(spec=BackendFactory)
    backend = MagicMock(spec=RenderBackend)
    backend.render_block.return_value = tuple(items)

    block_node = BlockNode(
        factory=factory,
        backend=backend,
        variant_idx=0,
        kind=BlockKind.BODY,
        block_idx=0,
        preview="",
    )

    children = block_node.expand()

    # No sibling rows -- the section's lone AsmLine becomes the lone
    # AsmLeaf; the two openables stay attached as sidecar.
    assert len(children) == 1
    leaf = children[0]
    assert isinstance(leaf, AsmLeaf)
    assert len(leaf.openables) == 2
    assert isinstance(leaf.openables[0], InlineCallEntry)
    assert isinstance(leaf.openables[1], InlineJumpEntry)
    # AsmLeaf threads the parent's factory/backend/variant_idx so its
    # own expand path can construct InlineCall/InlineJump model nodes.
    assert leaf.factory is factory
    assert leaf.backend is backend
    assert leaf.variant_idx == 0

    # AsmLeaf.expand (2+-arm wrap-each) surfaces two wrapper AsmLeaf
    # rows, each carrying exactly its own openable -- pins the
    # end-to-end cluster #3 contract through the model expansion.
    wrappers = leaf.expand()
    assert len(wrappers) == 2
    assert all(isinstance(w, AsmLeaf) for w in wrappers)
    assert all(len(w.openables) == 1 for w in wrappers)
