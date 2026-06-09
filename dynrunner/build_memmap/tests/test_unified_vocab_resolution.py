"""The build_memmap worker resolves ``--unified-vocab`` to wherever the
unify_vocab phase actually published the corpus vocab, in either
deployment mode.

The composite chain passes the memmap worker a bare-basename
``--unified-vocab unified_vocab.csv`` (a per-run constant, not a
discovered path). The unify_vocab phase wrote that file via
``staged_publish(scope=UNIFY_VOCAB_SCOPE)``, which lands it under a
``unify_vocab/`` subdir in container deployment but flat at the source
root standalone. ``_on_args`` must mirror that placement through
``published_path`` so the fail-fast existence check at worker startup
passes in both modes — the regression that previously crashed the
chain's memmap phase silently was the worker looking for
``<source>/unified_vocab.csv`` while the container-mode publish had put
it at ``<source>/unify_vocab/unified_vocab.csv``.

Deployment mode is forced via ``is_container_deployment`` so both
branches run deterministically without an ``/app/out-tmp`` mount.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import tokenizer.output_staging as staging
import dynrunner.build_memmap.worker as worker
from tokenizer.output_staging import UNIFY_VOCAB_SCOPE


_VOCAB_FILENAME = "unified_vocab.csv"


def _args(source: Path, unified_vocab: str) -> Namespace:
    """Minimal Namespace `_on_args` reads. No log file / queue so the
    handler takes the stdout-logging branch and skips socket setup.
    """
    return Namespace(
        source=str(source),
        output=str(source / "memmap"),
        vocab_source=None,
        unified_vocab=unified_vocab,
        log_file=None,
        dynamic_queue=None,
        socket_path="/unused.sock",
        skip_existing=False,
    )


def _lay_out_both(source: Path) -> None:
    """Write the vocab at BOTH the container-nested and the standalone-flat
    locations, so the assertion pins which one the resolver actually
    selected rather than merely that *a* file exists.
    """
    nested = source / UNIFY_VOCAB_SCOPE / _VOCAB_FILENAME
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("vocab")
    (source / _VOCAB_FILENAME).write_text("vocab")


def test_chain_basename_resolves_under_scope_in_container(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    source = tmp_path / "out"
    source.mkdir()
    _lay_out_both(source)

    worker._on_args(_args(source, _VOCAB_FILENAME))

    # `_on_args` canonicalises via `.resolve()`; resolve the expected
    # side too so a symlinked tmpdir (e.g. /tmp -> /private/tmp) doesn't
    # spuriously fail the comparison.
    assert worker._UNIFIED_VOCAB_PATH == (
        source / UNIFY_VOCAB_SCOPE / _VOCAB_FILENAME
    ).resolve()


def test_chain_basename_resolves_flat_in_standalone(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)
    source = tmp_path / "out"
    source.mkdir()
    _lay_out_both(source)

    worker._on_args(_args(source, _VOCAB_FILENAME))

    assert worker._UNIFIED_VOCAB_PATH == (source / _VOCAB_FILENAME).resolve()


@pytest.mark.parametrize("container", [True, False])
def test_absolute_unified_vocab_passes_through(
    tmp_path: Path, monkeypatch, container: bool
) -> None:
    # Standalone `--task build-memmap` callers hand an absolute
    # `--unified-vocab`; it must bind to exactly that file regardless of
    # source root or deployment mode.
    monkeypatch.setattr(
        staging, "is_container_deployment", lambda: container
    )
    source = tmp_path / "out"
    source.mkdir()
    explicit = tmp_path / "elsewhere" / _VOCAB_FILENAME
    explicit.parent.mkdir(parents=True, exist_ok=True)
    explicit.write_text("vocab")

    worker._on_args(_args(source, str(explicit)))

    assert worker._UNIFIED_VOCAB_PATH == explicit.resolve()


def test_missing_vocab_fails_fast(tmp_path: Path, monkeypatch) -> None:
    # The startup existence guard must still fire when the resolved path
    # is absent — turning a per-task storm into one readable failure.
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    source = tmp_path / "out"
    source.mkdir()
    # Deliberately do NOT create the nested vocab.

    with pytest.raises(FileNotFoundError):
        worker._on_args(_args(source, _VOCAB_FILENAME))
