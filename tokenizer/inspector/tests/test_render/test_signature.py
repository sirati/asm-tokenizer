"""Public-surface tests for the renderer module.

Pins the :func:`render_block` keyword-only signature, the
:data:`LineItem` union membership, and the :class:`InlineCallEntry`
dataclass field set -- the contract downstream consumers (tree model
+ label rendering) couple to.
"""

from __future__ import annotations

import inspect

from tokenizer.inspector._render import (
    AsmLine,
    InlineCallEntry,
    InlineJumpEntry,
    LineItem,
    render_block,
)


def test_render_block_signature_no_unused_params():
    """Pin the cleaned-up keyword-only signature.

    The renderer was scoped down during the inspector-plan audit to
    exactly the parameters it needs. Re-introducing any of
    ``function_data`` / ``block_idx`` / ``fid_sidecar`` /
    ``fid_row_offsets`` / ``batch_row_idx`` / ``vocab_manager`` /
    ``arm`` would reintroduce coupling and re-parse antipatterns
    (see CLAUDE.md "no re-parsing in call chains" + the per-call
    invariant build is the tree-model layer's job).
    """
    sig = inspect.signature(render_block)
    params = set(sig.parameters.keys())
    assert params == {
        "block",
        "section",
        "kind_to_called_idx",
        "variant_pins",
        "line_to_name",
        "line_to_provider",
        "callee_arm_resolver",
    }
    # And every parameter is keyword-only -- positional ordering would
    # turn a call-site rename into a silent semantic shift across this
    # 7-arg boundary.
    for name, p in sig.parameters.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name!r} is not keyword-only ({p.kind})"
        )


def test_line_item_union_covers_all_render_output_classes():
    """The :data:`LineItem` union is what downstream consumers
    (the tree model's ``_lift_render_items_to_nodes``) match against.
    Pin it to exactly the three dataclasses the renderer yields so a
    silent expansion of the union surfaces as a test break."""
    # NB: LineItem is a `X | Y | Z` PEP-604 union; the runtime form is
    # ``types.UnionType`` whose ``__args__`` enumerates the members.
    members = set(LineItem.__args__)
    assert members == {AsmLine, InlineCallEntry, InlineJumpEntry}


def test_inline_call_entry_dataclass_field_set():
    """Pin the wire-shape of :class:`InlineCallEntry`.

    Tree-model nodes (``InlineCallNode``) consume these fields by name;
    a silent rename here would break the downstream label / expand
    paths without surfacing in this module's own tests. The set is
    small enough to enumerate explicitly."""
    expected = {
        "kind",
        "counter_id",
        "callee_name",
        "callee_section_pointer",
        "variant_idx",
        "provider",
    }
    assert {f.name for f in InlineCallEntry.__dataclass_fields__.values()} == expected
