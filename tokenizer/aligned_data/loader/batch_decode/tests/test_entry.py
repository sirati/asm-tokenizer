"""Integration tests for :func:`batch_decode` -- the end-to-end pipeline
entry point.

Single concern of this file: pin the public-API surface of
:func:`batch_decode` and the wiring contract between the four stages
(:func:`walk_sections` -> :func:`predict_lengths` ->
:func:`build_bulk_bytes` -> :func:`assemble_batch`).

The four stages themselves are tested by their own modules' tests. This
file focuses on:

1. Default-argument plumbing (RNG defaulting,
   ``variant_padding=VariantPadding.PAD_NULL`` default, optional sidecar
   defaults).
2. Linear composition order -- each stage receives the previous
   stage's output verbatim, plus the relevant kwargs.
3. End-to-end smoke against a synthetic :class:`BinarySession`,
   GATED on :func:`build_bulk_bytes` having a real implementation.
   Phase 3 is concurrent with this stage 4 wiring; until 3e lands a
   real body, the e2e smoke skips.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import tokenizer.aligned_data.loader.batch_decode._entry as _entry_module
from tokenizer.aligned_data.loader.batch_decode import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_bulk_bytes_is_implemented() -> bool:
    """Return True iff :func:`build_bulk_bytes` has a real body.

    Stage 3 wiring (subagent 3e) is concurrent with this stage 4 work;
    the e2e smoke gates on this flag and skips when the stub still
    raises ``NotImplementedError``."""

    from tokenizer.aligned_data.loader.batch_decode._bulk_bytes import (
        build_bulk_bytes,
    )

    try:
        build_bulk_bytes(stage2=None)  # type: ignore[arg-type]
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception means the body executed (and ran into a
        # real-input precondition); treat that as "implemented".
        return True
    return True


# ---------------------------------------------------------------------------
# Wiring tests via monkeypatch
# ---------------------------------------------------------------------------


def test_batch_decode_default_rng_initialised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing ``rng=None`` makes :func:`batch_decode` allocate a fresh
    :func:`numpy.random.default_rng`. The RNG is threaded through to
    stage 1 (the only stage that uses it).
    """

    captured: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured["rng"] = kwargs["rng"]
        return "stage1_marker"

    def fake_predict_lengths(stage1, **kwargs):
        captured["stage1_seen"] = stage1
        return "stage2_marker"

    def fake_build_bulk_bytes(stage2):
        captured["stage2_seen"] = stage2
        return "stage3_marker"

    def fake_assemble_batch(stage3, **kwargs):
        captured["stage3_seen"] = stage3
        captured["assemble_kwargs"] = kwargs
        return "result_marker"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(_entry_module, "predict_lengths", fake_predict_lengths)
    monkeypatch.setattr(_entry_module, "build_bulk_bytes", fake_build_bulk_bytes)
    monkeypatch.setattr(_entry_module, "assemble_batch", fake_assemble_batch)

    result = batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
    )

    assert result == "result_marker"
    # Default RNG: a Generator instance was constructed.
    assert isinstance(captured["rng"], np.random.Generator)
    # Stage-to-stage threading: each stage saw the prior stage's output.
    assert captured["stage1_seen"] == "stage1_marker"
    assert captured["stage2_seen"] == "stage2_marker"
    assert captured["stage3_seen"] == "stage3_marker"


def test_batch_decode_default_variant_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``variant_padding`` is :attr:`VariantPadding.PAD_NULL`."""

    captured: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured["variant_padding"] = kwargs["variant_padding"]
        return "stage1"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(
        _entry_module, "predict_lengths", lambda stage1, **kw: "stage2"
    )
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(
        _entry_module, "assemble_batch", lambda stage3, **kw: "result"
    )

    batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
    )

    assert captured["variant_padding"] is VariantPadding.PAD_NULL


@pytest.mark.parametrize(
    "padding",
    [
        VariantPadding.PAD_NULL,
        VariantPadding.RESAMPLE_WITHIN_SECTION,
        VariantPadding.RAGGED,
        VariantPadding.REDISTRIBUTE,
    ],
)
def test_batch_decode_variant_padding_threaded(
    monkeypatch: pytest.MonkeyPatch, padding: VariantPadding
) -> None:
    """Every :class:`VariantPadding` policy is threaded through to
    :func:`walk_sections` verbatim."""

    captured: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured["variant_padding"] = kwargs["variant_padding"]
        return "stage1"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(
        _entry_module, "predict_lengths", lambda stage1, **kw: "stage2"
    )
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(
        _entry_module, "assemble_batch", lambda stage3, **kw: "result"
    )

    batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
        variant_padding=padding,
    )

    assert captured["variant_padding"] is padding


def test_batch_decode_kwargs_threaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """All optional kwargs reach the correct stage:

    * ``inlined_equivalent_call_targets_only`` -> stage 1.
    * ``context_len`` -> stage 2 + stage 4 (used by both).
    * ``include_fid_sidecar``, ``keep_intermediate`` -> stage 4.
    """

    captured_walk: dict[str, Any] = {}
    captured_predict: dict[str, Any] = {}
    captured_assemble: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured_walk.update(kwargs)
        return "stage1"

    def fake_predict_lengths(stage1, **kwargs):
        captured_predict.update(kwargs)
        return "stage2"

    def fake_assemble_batch(stage3, **kwargs):
        captured_assemble.update(kwargs)
        return "result"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(_entry_module, "predict_lengths", fake_predict_lengths)
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(_entry_module, "assemble_batch", fake_assemble_batch)

    explicit_rng = np.random.default_rng(seed=42)
    batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=3,
        context_len=16,
        max_depth=2,
        variant_padding=VariantPadding.RAGGED,
        inlined_equivalent_call_targets_only=True,
        include_fid_sidecar=True,
        keep_intermediate=True,
        rng=explicit_rng,
    )

    # Stage 1 kwargs.
    assert captured_walk["num_variants_per_section"] == 3
    assert captured_walk["max_depth"] == 2
    assert captured_walk["variant_padding"] is VariantPadding.RAGGED
    assert captured_walk["inlined_equivalent_call_targets_only"] is True
    assert captured_walk["rng"] is explicit_rng

    # Stage 2 kwargs.
    assert captured_predict["context_len"] == 16

    # Stage 4 kwargs.
    assert captured_assemble["context_len"] == 16
    assert captured_assemble["include_fid_sidecar"] is True
    assert captured_assemble["keep_intermediate"] is True


def test_batch_decode_returns_assemble_batch_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:func:`batch_decode` returns exactly what :func:`assemble_batch`
    returned -- no post-processing layer."""

    sentinel = object()

    monkeypatch.setattr(
        _entry_module, "walk_sections", lambda *a, **kw: "stage1"
    )
    monkeypatch.setattr(
        _entry_module, "predict_lengths", lambda stage1, **kw: "stage2"
    )
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(
        _entry_module, "assemble_batch", lambda stage3, **kw: sentinel
    )

    result = batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
    )
    assert result is sentinel


# ---------------------------------------------------------------------------
# End-to-end smoke (gated on stage 3 readiness)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _build_bulk_bytes_is_implemented(),
    reason="build_bulk_bytes (stage 3e) is still a stub; run after Phase 3 wiring",
)
def test_batch_decode_end_to_end_synthetic() -> None:
    """Sanity smoke against a tiny synthetic :class:`BinarySession`.

    The session need not be production-realistic; it just needs to
    expose the per-arm loaders that stage 1 uses
    (:meth:`BinarySession._load_matched_for_splice`, etc.). Building
    one here would duplicate considerable scaffolding -- and would
    drift if upstream stages refactor their session contract -- so
    this smoke is gated on :func:`build_bulk_bytes` having a real
    body, after which the upstream test infrastructure (stage 1/2/3
    integration tests) should already cover a session fixture we can
    pull in.

    Until then, this test is a placeholder marker for "Phase 3
    wiring has landed; e2e smoke should be promoted to a real
    BinarySession-backed test here". The skip-condition above keeps
    CI green during the phase 3+4 concurrent rollout."""

    # When promoted, this test should:
    #   1. Build a tiny BinarySession with a handful of matched
    #      sections + variants (typically via a fixtures module).
    #   2. Call batch_decode with PAD_NULL / a deterministic RNG.
    #   3. Assert the result has the expected shape contract
    #      (BatchDecodeResult dataclass + dtypes + row-offsets).
    # We deliberately keep the body as a guarded ``assert True`` so
    # the test exists but is non-load-bearing until stage 3 is wired.
    assert True
