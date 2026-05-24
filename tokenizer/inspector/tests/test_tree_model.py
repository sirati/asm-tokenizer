"""Unit tests for the inspector tree-model ``expand`` contracts.

Covers the plan-D2 ``batch_decode`` kwargs invariant on
:meth:`FunctionNode.expand`, the variant-count -> :class:`VariantNode`
mapping, idempotent re-expansion, :class:`InlineCallNode.can_expand`
gating per :class:`CallTargetType`, :class:`AsmLeaf` terminal contract,
:class:`DecodeContext` field surface, and the central
``_on_node_expanded`` dispatcher's failure-path
``is_failed=True`` stamping (the latter ``importorskip`` 'textual').
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._tree_model import (
    AsmLeaf,
    DecodeContext,
    FunctionNode,
    InlineCallNode,
    VariantNode,
)
from tokenizer.inspector._tree_model import _nodes_function as nodes_function_mod


# ---------------------------------------------------------------------------
# Fixtures -- synthetic 2-variant Section + a fake session + a fake
# batch_decode result that build_variants_from_result will walk.
# ---------------------------------------------------------------------------


def _make_function_data(tokens_len: int = 8) -> SimpleNamespace:
    """Minimal :class:`FunctionData` stand-in for label rendering +
    token-len peeking.

    The tree model touches ``.tokens`` (for ``len(...)``) and
    ``.metadata`` (read by :func:`variant_label`); the auto-size helper
    additionally reads ``.variant_tokens.shape[0]``.
    """
    return SimpleNamespace(
        tokens=np.zeros(tokens_len, dtype=np.uint16),
        variant_tokens=np.zeros(0, dtype=np.uint16),
        metadata={"arch": "x86", "compiler": "clang", "compilerversion": "8", "opt": "O3"},
    )


def _make_section(n_variants: int) -> SimpleNamespace:
    """Synthetic :class:`Section` with ``n_variants`` variant blocks."""
    variants = [
        SimpleNamespace(
            variant_ref_offset=i,
            data_offset_shifted=i,
            per_call_entries=[],
        )
        for i in range(n_variants)
    ]
    return SimpleNamespace(
        function_name_ptr=0,
        section_offset=0,
        call_targets=[],
        variants=variants,
    )


def _make_stage1_variant(batch_idx: int, variant_idx: int, function_data) -> SimpleNamespace:
    """One :class:`Stage1Variant`-shaped entry; the model walks
    ``call_targets[0].function_data`` for the variant's root body."""
    return SimpleNamespace(
        variant_idx=variant_idx,
        variant_ref_offset=variant_idx,
        batch_idx=batch_idx,
        call_targets=[SimpleNamespace(function_data=function_data)],
        variant_tokens=np.zeros(0, dtype=np.uint16),
    )


def _make_batch_decode_result(section: SimpleNamespace, n_variants: int) -> SimpleNamespace:
    """Assemble a fake :class:`BatchDecodeResult` shaped just enough for
    :func:`build_variants_from_result` -- ``.intermediate.stage2.stage1.
    sections[0].{section,variants}``."""
    function_data = _make_function_data()
    stage1_variants = [
        _make_stage1_variant(batch_idx=i, variant_idx=i, function_data=function_data)
        for i in range(n_variants)
    ]
    stage1_section = SimpleNamespace(section=section, variants=stage1_variants)
    stage1_batch = SimpleNamespace(sections=[stage1_section])
    stage2_batch = SimpleNamespace(stage1=stage1_batch)
    stage3_batch = SimpleNamespace(stage2=stage2_batch)
    return SimpleNamespace(
        fid_sidecar=np.zeros(0, dtype=np.uint32),
        fid_row_offsets=np.zeros(1, dtype=np.uint32),
        intermediate=stage3_batch,
    )


@pytest.fixture
def n_variants() -> int:
    return 3


@pytest.fixture
def synthetic_section(n_variants):
    return _make_section(n_variants)


@pytest.fixture
def fake_session(synthetic_section):
    """Session with the bare-minimum API the model touches:
    ``_load_matched_section_and_variants`` (the private helper that
    :func:`compute_auto_sizes` + :func:`resolve_section_pointers`
    delegate to), ``get_metadata``, ``_idx_for_section_offset``."""
    session = MagicMock(name="BinarySession")
    matched = SimpleNamespace(
        func_name="main",
        variants=[_make_function_data() for _ in synthetic_section.variants],
    )
    section_offset = synthetic_section.section_offset
    session._load_matched_section_and_variants.return_value = (
        synthetic_section,
        section_offset,
        matched,
    )
    session.get_metadata.return_value = {}  # empty line_to_name / line_to_provider
    session._idx_for_section_offset.return_value = None
    return session


@pytest.fixture
def vocab_manager():
    """:class:`VocabularyManager` is only stashed onto :class:`DecodeContext`;
    a ``MagicMock`` suffices since the variant-expand pipeline (the only
    code path that consumes it) is never reached in these tests."""
    return MagicMock(name="VocabularyManager")


@pytest.fixture
def batch_decode_spy(monkeypatch, synthetic_section, n_variants):
    """Replace ``batch_decode`` at its import site in
    :mod:`tokenizer.inspector._tree_model._nodes_function` and record
    invocations. Returns a ``MagicMock`` -- inspect ``.call_args_list``
    + ``.call_count``."""
    result = _make_batch_decode_result(synthetic_section, n_variants)
    spy = MagicMock(return_value=result)
    monkeypatch.setattr(nodes_function_mod, "batch_decode", spy)
    return spy


# ---------------------------------------------------------------------------
# FunctionNode.expand -- the plan-D2 invariant + idempotence
# ---------------------------------------------------------------------------


def test_function_node_expand_uses_required_batch_decode_kwargs(
    fake_session, batch_decode_spy, vocab_manager, n_variants
):
    """Plan D2: every inspector-driven ``batch_decode`` call MUST pin
    ``include_fid_sidecar=True``, ``keep_intermediate=True``,
    ``max_depth=0``, and ``num_variants_per_section`` = real variant
    count (no PAD_NULL row inflation, no inline expansion)."""
    node = FunctionNode(arm=SectionKind.MATCHED, idx=0, name="main")
    node.expand(fake_session, vocab_manager=vocab_manager)

    assert batch_decode_spy.call_count == 1
    _, kwargs = batch_decode_spy.call_args
    assert kwargs["include_fid_sidecar"] is True
    assert kwargs["keep_intermediate"] is True
    assert kwargs["max_depth"] == 0
    assert kwargs["num_variants_per_section"] == n_variants


def test_function_node_expand_returns_one_variant_node_per_variant(
    fake_session, batch_decode_spy, vocab_manager, n_variants
):
    """Every surviving (non-padding) :class:`Stage1Variant` maps to
    exactly one :class:`VariantNode` child."""
    node = FunctionNode(arm=SectionKind.MATCHED, idx=0, name="main")
    children = node.expand(fake_session, vocab_manager=vocab_manager)

    assert len(children) == n_variants
    assert all(isinstance(child, VariantNode) for child in children)


def test_function_node_expand_is_idempotent(
    fake_session, batch_decode_spy, vocab_manager, n_variants
):
    """Re-expanding the same :class:`FunctionNode` is observably the
    same: same number of variant children, no cached state surprises.

    The dispatcher in :mod:`_app.py` collapses + re-expands to retry a
    failed open (resetting ``is_failed``), so the model itself must
    tolerate repeated ``expand`` calls.
    """
    node = FunctionNode(arm=SectionKind.MATCHED, idx=0, name="main")
    first = node.expand(fake_session, vocab_manager=vocab_manager)
    second = node.expand(fake_session, vocab_manager=vocab_manager)

    assert len(first) == n_variants
    assert len(second) == n_variants
    assert batch_decode_spy.call_count == 2


# ---------------------------------------------------------------------------
# InlineCallNode.can_expand -- LOCAL-with-pointer is the ONE expandable
# combination (plan: PLT / EXTERN have no inlineable body).
# ---------------------------------------------------------------------------


def _make_decode_context() -> DecodeContext:
    """Throwaway :class:`DecodeContext` with the minimum required
    fields populated."""
    return DecodeContext(
        arm=SectionKind.MATCHED,
        fid_sidecar=np.zeros(0, dtype=np.uint32),
        fid_row_offsets=np.zeros(1, dtype=np.uint32),
        line_to_name={},
        line_to_provider={},
        vocab_manager=MagicMock(name="VocabularyManager"),
        callee_arm_resolver=lambda _offset: None,
    )


def test_inline_call_node_can_expand_for_local_with_pointer():
    """LOCAL + resolved :class:`SectionPointerSpec` is the only
    expandable inline-call shape (the callee's body is inlineable via
    a fresh :func:`batch_decode`)."""
    node = InlineCallNode(
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=1),
        variant_idx=0,
        provider=None,
        decode_context=_make_decode_context(),
    )
    assert node.can_expand is True


def test_inline_call_node_cannot_expand_for_local_without_pointer():
    """LOCAL without a resolved pointer (cross-arm callee / missing
    section) must surface as a non-expandable leaf -- no
    :func:`batch_decode` is callable without an ``(arm, idx)`` spec."""
    node = InlineCallNode(
        kind=CallTargetType.LOCAL,
        counter_id=0,
        callee_name="callee",
        callee_section_pointer=None,
        variant_idx=0,
        provider=None,
        decode_context=_make_decode_context(),
    )
    assert node.can_expand is False


@pytest.mark.parametrize("kind", [CallTargetType.PLT, CallTargetType.EXTERN])
def test_inline_call_node_cannot_expand_for_plt_or_extern(kind):
    """PLT / EXTERN have no inlineable body regardless of pointer
    presence -- a PLT stub is just a thunk, an EXTERN resolves to
    library code outside the binary."""
    node = InlineCallNode(
        kind=kind,
        counter_id=0,
        callee_name="callee",
        callee_section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=1),
        variant_idx=0,
        provider=None,
        decode_context=_make_decode_context(),
    )
    assert node.can_expand is False


# ---------------------------------------------------------------------------
# AsmLeaf -- terminal, gating on ``can_expand``.
# ---------------------------------------------------------------------------


def test_asm_leaf_cannot_expand():
    assert AsmLeaf(text="nop").can_expand is False


def test_asm_leaf_expand_raises():
    """Calling :meth:`AsmLeaf.expand` is a contract violation; the
    docstring tells callers to gate on ``can_expand``. The model raises
    ``NotImplementedError`` rather than silently returning ``[]`` so a
    UI-side bug surfaces immediately."""
    with pytest.raises(NotImplementedError):
        AsmLeaf(text="nop").expand(session=MagicMock(), vocab_manager=MagicMock())


# ---------------------------------------------------------------------------
# DecodeContext -- pin the post-audit field surface so future field
# renames break loudly.
# ---------------------------------------------------------------------------


def test_decode_context_carries_callee_arm_resolver():
    """:attr:`DecodeContext.callee_arm_resolver` is the boundary that
    keeps :class:`InlineCallNode` from reaching into session internals.
    """
    sentinel = SectionPointerSpec(arm=SectionKind.MATCHED, idx=42)
    ctx = DecodeContext(
        arm=SectionKind.MATCHED,
        fid_sidecar=None,
        fid_row_offsets=None,
        line_to_name={},
        line_to_provider={},
        vocab_manager=MagicMock(),
        callee_arm_resolver=lambda _offset: sentinel,
    )
    assert ctx.callee_arm_resolver(0) is sentinel


def test_decode_context_carries_line_to_provider():
    """:attr:`DecodeContext.line_to_provider` is the post-P2/P3-audit
    addition that lets EXTERN inline-call labels render the
    ``@<provider>`` suffix without a second metadata roundtrip."""
    providers = {7: "libc"}
    ctx = DecodeContext(
        arm=SectionKind.MATCHED,
        fid_sidecar=None,
        fid_row_offsets=None,
        line_to_name={},
        line_to_provider=providers,
        vocab_manager=MagicMock(),
        callee_arm_resolver=lambda _offset: None,
    )
    assert ctx.line_to_provider is providers


# ---------------------------------------------------------------------------
# Expand-error path through the central app dispatcher (plan D8).
# Importing ``_app.py`` pulls in ``textual``; skip the test cleanly when
# the dep isn't available (default ``nix develop`` shell).
# ---------------------------------------------------------------------------


def test_expand_error_path_marks_is_failed_via_app_dispatcher():
    """Plan D8: the inspector dispatcher is the ONE try/except wrapping
    model ``expand``. On exception it stamps ``is_failed=True`` on the
    model + attaches a dim-red error leaf + refreshes the prefix glyph.

    We drive the path by constructing a :class:`FunctionNode` whose
    ``expand`` raises, then invoking :meth:`InspectorApp._on_node_expanded`
    directly with a hand-rolled event whose ``.node.data`` is the
    failing node. ``Tree.NodeExpanded`` is wrapped by a ``MagicMock`` so
    we don't depend on the Textual message-pump.
    """
    pytest.importorskip("textual")

    from pathlib import Path
    import tempfile

    from tokenizer.inspector._app import InspectorApp

    # Force the FunctionNode's expand to raise; the dispatcher must
    # catch and stamp ``is_failed`` without bubbling.
    fn = FunctionNode(arm=SectionKind.MATCHED, idx=0, name="main")
    boom = RuntimeError("synthetic expand failure")

    def _raise(_session, *, vocab_manager):  # noqa: ARG001 -- match signature
        raise boom

    fn.expand = _raise  # type: ignore[method-assign]

    # Minimal app instance -- ``compose`` isn't invoked because we call
    # the dispatcher directly, so the dataset only needs the fields
    # ``_on_node_expanded`` actually reads (``vocab_manager``).
    dataset = MagicMock(name="BinaryDataset")
    dataset.vocab_manager = MagicMock(name="VocabularyManager")
    session = MagicMock(name="BinarySession")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "inspector.log"
        app = InspectorApp(dataset=dataset, session=session, log_path=log_path)

        # Hand-roll an event with the surface the dispatcher touches:
        # ``stop()``, ``.node.data``, ``.node.remove_children()``,
        # ``.node.add_leaf()``, ``.node.refresh()``.
        tree_node = MagicMock(name="TreeNode")
        tree_node.data = fn
        event = MagicMock(name="NodeExpanded")
        event.node = tree_node

        app._on_node_expanded(event)

    assert fn.is_failed is True
    # Tree node should have had its error-leaf attached + refresh called.
    tree_node.add_leaf.assert_called_once()
    tree_node.refresh.assert_called()
