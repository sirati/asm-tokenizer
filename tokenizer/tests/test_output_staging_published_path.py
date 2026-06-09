"""``published_path`` is the mode-aware inverse of ``staged_publish``'s
scope layout.

``staged_publish(scope=...)`` lands a write at
``<root>/<scope>/<file>`` under container deployment but flat at
``<root>/<file>`` standalone (scope ignored). A reader of a
scoped-publish artifact must resolve it the same way the writer placed
it; ``published_path`` owns that single layout rule so producer and
consumer stay symmetric without either knowing the other's mode.

The deployment mode is forced via ``is_container_deployment`` so the
two branches are exercised deterministically regardless of whether the
test host has the ``/app/out-tmp`` mount.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tokenizer.output_staging as staging
from tokenizer.output_staging import UNIFY_VOCAB_SCOPE, published_path


@pytest.fixture
def container_mode(monkeypatch) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)


@pytest.fixture
def standalone_mode(monkeypatch) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)


def test_container_layout_nests_under_scope(container_mode: None) -> None:
    # Container mode mirrors the staging subdir into the destination, so
    # the file lands at <root>/<scope>/<file>.
    assert published_path(
        Path("/out"), UNIFY_VOCAB_SCOPE, "unified_vocab.csv"
    ) == Path("/out/unify_vocab/unified_vocab.csv")


def test_standalone_layout_is_flat(standalone_mode: None) -> None:
    # Standalone writes go to output_dir directly; scope is ignored.
    assert published_path(
        Path("/out"), UNIFY_VOCAB_SCOPE, "unified_vocab.csv"
    ) == Path("/out/unified_vocab.csv")


@pytest.mark.parametrize("mode", ["container", "standalone"])
def test_absolute_filename_passes_through(mode: str, monkeypatch) -> None:
    # An absolute filename (a standalone caller handing an explicit
    # location) is invariant to the root + scope in BOTH branches —
    # pathlib's `/` keeps an absolute right-hand side.
    monkeypatch.setattr(
        staging, "is_container_deployment", lambda: mode == "container"
    )
    assert published_path(
        Path("/out"), UNIFY_VOCAB_SCOPE, "/abs/elsewhere/unified_vocab.csv"
    ) == Path("/abs/elsewhere/unified_vocab.csv")
