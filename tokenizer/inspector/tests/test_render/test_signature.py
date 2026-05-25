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
        "caller_variant_idx",
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


def test_line_item_alias_is_asm_line_after_openables_migration():
    """Post-R2 the :data:`LineItem` alias narrows to :class:`AsmLine`
    only (cluster #3 plan W3-2 W4-amended): inline call sites, jump
    targets, and number-precision expansions ride
    :attr:`AsmLine.openables` rather than sibling top-level items.
    Pin the alias so a future re-broadening surfaces here."""
    assert LineItem is AsmLine


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
        "caller_variant_idx",
    }
    assert {f.name for f in InlineCallEntry.__dataclass_fields__.values()} == expected
