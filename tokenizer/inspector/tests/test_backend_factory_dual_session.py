"""Per-arm :class:`BinarySession` ownership in :class:`_BatchDecodeBackendFactory`.

Pins the cross-arm-navigation fix: the factory holds ONE
:class:`BinarySession` per arm so a MATCHED expand followed by an
inlined UNMATCHED callee never asks one session to switch arms
mid-flight (which :meth:`BinarySession._open_data` rejects).

These tests exercise the factory directly with stub sessions / dataset
-- the public ``make_batch_decode_factory`` opener is covered by
inspector smoke runs against a real corpus.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._render._backend_factory import (
    _BatchDecodeBackendFactory,
)
from tokenizer.inspector._render._protocol import FunctionHandle


def _stub_dataset() -> SimpleNamespace:
    """Minimal :class:`BinaryDataset` stand-in for factory wiring tests.

    Only the attributes :class:`_BatchDecodeBackendFactory.make` reaches
    for (``vocab_manager`` + the two name maps) are exposed; everything
    else stays out of the way so the test stays focused on session
    ownership.
    """
    return SimpleNamespace(
        vocab_manager=MagicMock(name="vocab_manager"),
        line_to_name={},
        line_to_provider={},
    )


def test_make_routes_matched_handle_to_matched_session() -> None:
    """A handle with ``arm=MATCHED`` MUST produce a backend bound to the
    matched session; the unmatched session stays untouched.
    """
    matched_session = MagicMock(name="matched_session")
    unmatched_session = MagicMock(name="unmatched_session")
    factory = _BatchDecodeBackendFactory(
        dataset=_stub_dataset(),
        sessions={
            SectionKind.MATCHED: matched_session,
            SectionKind.UNMATCHED: unmatched_session,
        },
        handles=(),
    )
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name="m_fn")
    backend = factory.make(handle)
    # The backend stores its session under ``_session``; the dual-session
    # ownership model means cross-arm inlining picks up the OTHER session
    # without forcing the matched one to switch arms.
    assert backend._session is matched_session


def test_make_routes_unmatched_handle_to_unmatched_session() -> None:
    """A handle with ``arm=UNMATCHED`` (the cross-arm-navigation case)
    MUST produce a backend bound to the unmatched session. This pins
    the original bug: pre-fix, the factory would re-use the single
    matched session and :meth:`BinarySession._open_data` would raise
    ``cannot switch to unmatched mid-session``.
    """
    matched_session = MagicMock(name="matched_session")
    unmatched_session = MagicMock(name="unmatched_session")
    factory = _BatchDecodeBackendFactory(
        dataset=_stub_dataset(),
        sessions={
            SectionKind.MATCHED: matched_session,
            SectionKind.UNMATCHED: unmatched_session,
        },
        handles=(),
    )
    handle = FunctionHandle(arm=SectionKind.UNMATCHED, idx=0, name="u_fn")
    backend = factory.make(handle)
    assert backend._session is unmatched_session


def test_make_raises_when_arm_has_no_session() -> None:
    """When a binary has no unmatched data the factory omits the
    UNMATCHED session; :meth:`make` MUST raise on a stale unmatched
    handle (rather than silently misroute or open a missing file).
    """
    matched_session = MagicMock(name="matched_session")
    factory = _BatchDecodeBackendFactory(
        dataset=_stub_dataset(),
        sessions={SectionKind.MATCHED: matched_session},
        handles=(),
    )
    handle = FunctionHandle(arm=SectionKind.UNMATCHED, idx=0, name="u_fn")
    with pytest.raises(KeyError, match="UNMATCHED"):
        factory.make(handle)


def test_post_init_requires_matched_session() -> None:
    """The MATCHED arm always has data (the inspector seeds matched
    handles); omitting its session is a construction-time bug, not a
    runtime fallback.
    """
    with pytest.raises(ValueError, match="MATCHED session"):
        _BatchDecodeBackendFactory(
            dataset=_stub_dataset(),
            sessions={},
            handles=(),
        )


def test_close_closes_every_owned_session() -> None:
    """:meth:`close` MUST exit every session the factory owns; closing
    only the matched session would leak the unmatched memmap handles.
    """
    matched_session = MagicMock(name="matched_session")
    unmatched_session = MagicMock(name="unmatched_session")
    factory = _BatchDecodeBackendFactory(
        dataset=_stub_dataset(),
        sessions={
            SectionKind.MATCHED: matched_session,
            SectionKind.UNMATCHED: unmatched_session,
        },
        handles=(),
    )
    factory.close()
    matched_session.close.assert_called_once_with()
    unmatched_session.close.assert_called_once_with()


def test_close_is_idempotent() -> None:
    """A second :meth:`close` is a no-op; mirrors the
    :class:`_FtlBackendFactory.close` contract."""
    matched_session = MagicMock(name="matched_session")
    factory = _BatchDecodeBackendFactory(
        dataset=_stub_dataset(),
        sessions={SectionKind.MATCHED: matched_session},
        handles=(),
    )
    factory.close()
    factory.close()
    matched_session.close.assert_called_once_with()
