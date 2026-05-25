"""Tests for :class:`BatchDecodeBackend` (Protocol + lifecycle).

Pins:

* :class:`RenderBackend` Protocol compliance via ``isinstance`` (the
  Protocol is ``@runtime_checkable``).
* :meth:`close` -> :attr:`closed` True; subsequent public methods
  raise :class:`RuntimeError` (audit A-HIGH-3).
* Lazy constructor (audit A-MED-4): instantiation MUST NOT call
  :func:`batch_decode` / :func:`compute_auto_sizes`; the first
  :meth:`variants` / :meth:`blocks` / :meth:`render_block` access
  triggers :meth:`_ensure_result`.
* :func:`compute_auto_sizes` invoked with the correct pointer spec
  (= the backend's :attr:`handle`'s ``(arm, idx)``).

Plan reference: ``inspector-render-backends.md`` §6 + decisions #19,
#20, #22 + audits A-MED-4 + A-HIGH-3.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._render._batch_decode_backend import (
    BatchDecodeBackend,
)
from tokenizer.inspector._render._protocol import (
    BlockKind,
    FunctionHandle,
    RenderBackend,
)
from tokenizer.token_manager import VocabularyManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(**overrides) -> BatchDecodeBackend:
    """Cheap :class:`BatchDecodeBackend` instance with stub deps.

    The constructor is lazy (audit A-MED-4); none of the stubs need
    real wiring for instantiation. Tests that exercise the lazy-trigger
    inject patched :func:`batch_decode` + :func:`compute_auto_sizes`
    via :func:`unittest.mock.patch`.
    """
    kwargs: dict = dict(
        session=MagicMock(name="session"),
        vocab_manager=MagicMock(spec=VocabularyManager),
        handle=FunctionHandle(
            arm=SectionKind.MATCHED, idx=42, name="dummy_func",
        ),
        line_to_name={},
        line_to_provider={},
        callee_arm_resolver=lambda _offset: None,
    )
    kwargs.update(overrides)
    return BatchDecodeBackend(**kwargs)


# ---------------------------------------------------------------------------
# Protocol compliance (runtime_checkable)
# ---------------------------------------------------------------------------


def test_satisfies_render_backend_protocol() -> None:
    """``BatchDecodeBackend`` MUST satisfy the runtime-checkable
    :class:`RenderBackend` Protocol. Pins decision #22 (the public
    surface IS the Protocol; concrete class is private).
    """
    backend = _make_backend()
    assert isinstance(backend, RenderBackend)


def test_handle_property_returns_constructor_handle() -> None:
    """:attr:`handle` exposes the typed coordinate the constructor
    received -- no copy/transform.
    """
    handle = FunctionHandle(
        arm=SectionKind.UNMATCHED, idx=7, name="another_func",
    )
    backend = _make_backend(handle=handle)
    assert backend.handle is handle


# ---------------------------------------------------------------------------
# Lazy construction (audit A-MED-4)
# ---------------------------------------------------------------------------


def test_constructor_does_not_call_batch_decode() -> None:
    """Constructor stores refs only; :func:`batch_decode` /
    :func:`compute_auto_sizes` MUST NOT run until first
    :meth:`variants` / :meth:`blocks` / :meth:`render_block` access.
    """
    backend_mod = "tokenizer.inspector._render._batch_decode_backend._backend"
    with patch(f"{backend_mod}.batch_decode") as mock_decode, patch(
        f"{backend_mod}.compute_auto_sizes"
    ) as mock_size:
        _make_backend()
        mock_decode.assert_not_called()
        mock_size.assert_not_called()


def test_closed_property_is_false_after_construction() -> None:
    """Construction MUST NOT set ``closed=True`` (the lifecycle starts
    open).
    """
    backend = _make_backend()
    assert backend.closed is False


# ---------------------------------------------------------------------------
# close() lifecycle (audit A-HIGH-3)
# ---------------------------------------------------------------------------


def test_close_sets_closed_flag() -> None:
    """:meth:`close` flips :attr:`closed` to ``True`` -- the
    observable flag the tree-model checks before re-issuing a method.
    """
    backend = _make_backend()
    backend.close()
    assert backend.closed is True


def test_close_is_idempotent() -> None:
    """Calling :meth:`close` twice is a no-op (the contract docstring
    pins idempotency).
    """
    backend = _make_backend()
    backend.close()
    backend.close()  # second call MUST NOT raise
    assert backend.closed is True


def test_variants_after_close_raises_runtime_error() -> None:
    """:meth:`variants` after :meth:`close` raises
    :class:`RuntimeError` with the class-name-bearing message
    (audit A-HIGH-3's exact contract).
    """
    backend = _make_backend()
    backend.close()
    with pytest.raises(RuntimeError, match="BatchDecodeBackend closed"):
        backend.variants()


def test_blocks_after_close_raises_runtime_error() -> None:
    """:meth:`blocks` after :meth:`close` raises :class:`RuntimeError`."""
    backend = _make_backend()
    backend.close()
    with pytest.raises(RuntimeError, match="BatchDecodeBackend closed"):
        backend.blocks(variant_idx=0)


def test_render_block_after_close_raises_runtime_error() -> None:
    """:meth:`render_block` after :meth:`close` raises
    :class:`RuntimeError`.
    """
    backend = _make_backend()
    backend.close()
    with pytest.raises(RuntimeError, match="BatchDecodeBackend closed"):
        backend.render_block(variant_idx=0, kind=BlockKind.BODY, block_idx=0)


# ---------------------------------------------------------------------------
# Lazy trigger: first variants() call invokes compute_auto_sizes
# ---------------------------------------------------------------------------


def test_first_variants_call_invokes_compute_auto_sizes_with_handle() -> None:
    """The first :meth:`variants` call triggers :meth:`_ensure_result`
    which calls :func:`compute_auto_sizes` with a
    :class:`SectionPointerSpec` built from the backend's handle
    (audit A-MED-4 + decision #19).

    Patches both helpers so we can inspect the arg shape without
    needing a real :class:`BinarySession`. The :func:`batch_decode`
    patch returns a stub whose ``intermediate.stage2.stage1.sections``
    is empty -- :meth:`_build_variants` will fail downstream, but the
    test stops at the :func:`compute_auto_sizes` call-shape check.
    """
    backend_mod = "tokenizer.inspector._render._batch_decode_backend._backend"
    handle = FunctionHandle(
        arm=SectionKind.MATCHED, idx=42, name="probe_func",
    )
    backend = _make_backend(handle=handle)

    with patch(f"{backend_mod}.batch_decode") as mock_decode, patch(
        f"{backend_mod}.compute_auto_sizes"
    ) as mock_size, patch(
        f"{backend_mod}.FidBaseTable.from_result"
    ) as mock_fid:
        mock_size.return_value = MagicMock(
            num_variants_per_section=1, context_len=64,
        )
        # batch_decode return: minimal stub with the chain
        # ``.intermediate.stage2.stage1.sections`` -- _build_variants
        # asserts ``len(sections) == 1`` so populate accordingly.
        decoded = MagicMock(name="decoded")
        decoded.intermediate.stage2.stage1.sections = []  # empty: assert fires
        mock_decode.return_value = decoded
        mock_fid.return_value = MagicMock(name="fid_table")

        with pytest.raises(AssertionError):
            # _build_variants's ``len(sections) == 1`` guard fires; we
            # only need to observe that compute_auto_sizes was reached
            # with the right pointer spec before the failure.
            backend.variants()

    # compute_auto_sizes call: positional args (session, [SectionPointerSpec])
    assert mock_size.call_count == 1
    args, kwargs = mock_size.call_args
    session_arg, pointer_specs = args
    assert session_arg is backend._session
    assert isinstance(pointer_specs, list) and len(pointer_specs) == 1
    spec = pointer_specs[0]
    assert isinstance(spec, SectionPointerSpec)
    assert spec.arm is SectionKind.MATCHED
    assert spec.idx == 42


def test_render_block_unknown_block_raises_keyerror() -> None:
    """:meth:`render_block` for a non-existent section raises
    :class:`KeyError` (audit B-LOW-12: typed missing-key signal, not
    a silent empty iterable).
    """
    backend_mod = "tokenizer.inspector._render._batch_decode_backend._backend"
    backend = _make_backend()

    with patch(f"{backend_mod}.batch_decode") as mock_decode, patch(
        f"{backend_mod}.compute_auto_sizes"
    ) as mock_size, patch(
        f"{backend_mod}.FidBaseTable.from_result"
    ) as mock_fid, patch.object(
        BatchDecodeBackend, "_row_sections_for_variant"
    ) as mock_walk:
        mock_size.return_value = MagicMock(
            num_variants_per_section=1, context_len=64,
        )
        # batch_decode stub; _build_variants reads
        # ``intermediate.stage2.stage1.sections`` -- patch _row_sections
        # so we sidestep the walker and feed an empty row-sections list.
        decoded = MagicMock(name="decoded")
        stage1_section = MagicMock(name="section")
        stage1_section.variants = []
        decoded.intermediate.stage2.stage1.sections = [stage1_section]
        mock_decode.return_value = decoded
        mock_fid.return_value = MagicMock(name="fid_table")
        mock_walk.return_value = []  # variant exists, no sections

        # Need a variant in _variant_row_index for render_block to dispatch.
        backend._variant_row_index = {0: 0}
        backend._result = decoded
        with pytest.raises(KeyError, match="no section"):
            backend.render_block(
                variant_idx=0, kind=BlockKind.BODY, block_idx=999,
            )
